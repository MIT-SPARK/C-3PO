"""

"""
import argparse
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from datetime import datetime
from pytorch3d import ops
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")

from c3po.datasets.ycb import DepthYCB
from c3po.datasets.ycb import YCB
from c3po.datasets.utils_dataset import toFormat
# from c3po.models.certifiability import confidence, confidence_kp
# from c3po.utils.general import TrackingMeter
from c3po.utils.visualization_utils import display_results
from c3po.utils.loss_functions import certify, self_supervised_training_loss \
    as self_supervised_loss, self_supervised_validation_loss as validation_loss
# evaluation metrics
from c3po.utils.evaluation_metrics import add_s_error
from c3po.expt_ycb.supervised_training import train_with_supervision
from c3po.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel
from c3po.expt_shapenet.evaluation import evaluate


# Visualize
def visual_test(test_loader, model, device=None, hyper_param=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, predicted_model_keypoints \
            = model(input_point_cloud)

        # certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=predicted_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints,
                        epsilon=hyper_param['epsilon'])

        print("Certifiable: ", certi)

        # adds for evaluation
        ground_truth_point_cloud = R_target @ model.cad_models + t_target

        adds_err_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                threshold=hyper_param["adds_threshold"])
        print("ADD-S Score", adds_err_)

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_p = predicted_point_cloud.clone().detach().to('cpu')
        kp = keypoints_target.clone().detach().to('cpu')
        kp_p = predicted_keypoints.clone().detach().to('cpu')
        # display_results(input_point_cloud=pc_p, detected_keypoints=kp_p, target_point_cloud=pc,
        #                 target_keypoints=kp)
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc,
                        target_keypoints=kp)

        del pc, pc_p, kp, kp_p
        del input_point_cloud, keypoints_target, R_target, t_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        if i >= 10:
            break


def visualize_detector(hyper_param, detector_type, model_id,
                       evaluate_models=True, use_corrector=False,
                       visualize_before=True, visualize_after=True, device=None, dataset=None):
    """

    """

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_model_save_file = best_model_save_location + '_best_baseline_regression_model_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_batch_size = hyper_param['eval_batch_size'][model_id]

    if dataset is None or dataset == "ycb":
        eval_dataset = DepthYCB(model_id=model_id,
                                split='test',
                                only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                                num_of_points=hyper_param['num_of_points_to_sample'])
        eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else hyper_param['eval_batch_size'][model_id]

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
        data_type = "ycb"
    else:

        type = dataset.split('.')[1]
        eval_dataset = YCB(type=type, object=model_id, length=50, num_points=1024, split="test")
        eval_dataset = toFormat(eval_dataset)

        eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else \
            hyper_param['eval_batch_size'][model_id]

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                                  pin_memory=True)
        data_type = dataset

    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector,
                          need_predicted_keypoints=True).to(device)

    if not os.path.isfile(best_model_save_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

    state_dict = torch.load(best_model_save_file)
    model.load_state_dict(state_dict)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # Evaluation:
    if evaluate_models:

        print(">>"*40)
        print("PRE-TRAINED MODEL:")
        print(">>" * 40)
        log_dir = "eval/KeyPoReal/" + detector_type
        # evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param, certification=True,
        #              device=device)
        evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param,
                 device=device, log_dir=log_dir, data_type=data_type)

    # # Visual Test
    if visualize_before:
        dataset_batch_size = 1
        dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
        loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False)
        print(">>" * 40)
        print("VISUALIZING PRE-TRAINED MODEL:")
        print(">>" * 40)
        visual_test(test_loader=loader, model=model, hyper_param=hyper_param)

    del model, state_dict

    return None


def evaluate_model(detector_type, model_ids, only_models=None,
                           evaluate_models=True,
                           visualize=True,
                           use_corrector=False,
                           visualize_before=True,
                           visualize_after=True, dataset=None):

    if not visualize:
        visualize_before, visualize_after \
            = False, False

    for model_id in model_ids:
        if model_id in only_models:
            hyper_param_file = "self_supervised_training.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param[detector_type]
            hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

            print(">>"*40)
            print("Analyzing Baseline for Object: ", model_id)
            visualize_detector(detector_type=detector_type,
                               model_id=model_id,
                               hyper_param=hyper_param,
                               evaluate_models=evaluate_models,
                               use_corrector=use_corrector,
                               visualize_before=visualize_before,
                               visualize_after=visualize_after,
                               dataset=dataset)


if __name__ == "__main__":

    """
    usage: 
    >> python evaluate_baseline.py \
    --detector "point_transformer" \
    --object "001_chips_can" \
    --dataset "ycb.real" 
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", help="specify the detector type.", type=str)
    parser.add_argument("--object", help="specify the ycb model id.", type=str)
    parser.add_argument("--dataset",
                        choices=["ycb", "ycb.sim", "ycb.real"], type=str)

    args = parser.parse_args()

    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.model_id)
    detector_type = args.detector
    model_id = args.object
    dataset = args.dataset

    only_models = [model_id]

    stream = open("model_ids.yml", "r")
    model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']

    evaluate_model(detector_type=detector_type, model_ids=model_ids, use_corrector=False, only_models=only_models,
                   visualize=False, dataset=dataset)





