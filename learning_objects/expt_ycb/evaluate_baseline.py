"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
import argparse
import pickle
from pytorch3d import ops

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, DepthPC, CLASS_NAME, \
    FixedDepthPC, CLASS_ID
from learning_objects.datasets.ycb import DepthYCB
from learning_objects.models.certifiability import confidence, confidence_kp

from learning_objects.utils.general import display_results, TrackingMeter

# loss functions
# from learning_objects.expt_self_supervised_correction.loss_functions import chamfer_loss
from learning_objects.expt_self_supervised_correction.loss_functions import certify
from learning_objects.expt_self_supervised_correction.loss_functions import self_supervised_training_loss \
    as self_supervised_loss
from learning_objects.expt_self_supervised_correction.loss_functions import self_supervised_validation_loss \
    as validation_loss
# evaluation metrics
from learning_objects.expt_self_supervised_correction.evaluation_metrics import evaluation_error, add_s_error

from learning_objects.expt_ycb.supervised_training import train_with_supervision

#SYMMETRIC_MODEL_IDS = ["001_chips_can", "002_master_chef_can", "004_sugar_box", \
#                       "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box" \
#                       "009_gelatin_box", "010_potted_meat_can", "036_wood_block", "040_large_marker", \
#                       "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]
#
# Train
def train_detector(hyper_param, detector_type='point_transformer', model_id="019_pitcher_base"):
    """

    """

    print('-' * 20)
    print("Training baseline regression model: ", datetime.now())
    print("Detector: ", detector_type)
    print("Object: ", model_id)
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    torch.cuda.empty_cache()

    # shapenet
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    best_model_save_file = best_model_save_location + '_best_baseline_regression_model_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_baseline_train_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_baseline_val_loss_' + detector_type + '.pkl'
    # cert_save_file = best_model_save_location + '_certi_all_batches_' + regression_model + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['baseline_lr_sgd']
    momentum_sgd = hyper_param['baseline_momentum_sgd']

    # object symmetry
    hyper_param["is_symmetric"] = False

    # real dataset:
    train_batch_size = hyper_param['self_supervised_train_batch_size'][model_id]
    num_of_points_to_sample = hyper_param['num_of_points_to_sample']

    train_dataset = DepthYCB(model_id=model_id,
                             split='train',
                             num_of_points=num_of_points_to_sample)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=False)

    # validation dataset:
    val_batch_size = hyper_param['val_batch_size'][model_id]
    val_dataset = DepthYCB(model_id=model_id,
                           split='val',
                           num_of_points=num_of_points_to_sample)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = train_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # model
    from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel
    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type).to(device)

    if not os.path.isfile(sim_trained_model_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
    state_dict = torch.load(sim_trained_model_file)
    model.load_state_dict(state_dict)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)

    # training
    train_loss, val_loss = train_with_supervision(supervised_training_loader=train_loader,
                                                  validation_loader=val_loader,
                                                  model=model,
                                                  optimizer=optimizer,
                                                  correction_flag=False,
                                                  best_model_save_file=best_model_save_file,
                                                  device=device,
                                                  hyper_param=hyper_param)

    with open(train_loss_save_file, 'wb') as outp:
        pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(val_loss_save_file, 'wb') as outp:
        pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

    # with open(cert_save_file, 'wb') as outp:
    #     pickle.dump(fra_cert_, outp, pickle.HIGHEST_PROTOCOL)

    return None


# Visualize
#ToDo: Did not change.
def visual_test(test_loader, model, correction_flag=False, device=None, hyper_param=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # torch.cuda.empty_cache()

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, predicted_model_keypoints \
            = model(input_point_cloud, correction_flag=correction_flag, need_predicted_keypoints=True)

        # certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=predicted_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints,
                        epsilon=hyper_param['epsilon'], is_symmetric=hyper_param['is_symmetric'])

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
                       evaluate_models=True,
                       visualize_without_corrector=True, visualize_with_corrector=True,
                       visualize_before=True, visualize_after=True, device=None):
    """

    """

    # print('-' * 20)
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('device is ', device)
    # print('-' * 20)
    # torch.cuda.empty_cache()
    # if models_to_analyze=='both':
    #     pre_ = True
    #     post_ = True
    # elif models_to_analyze == 'pre':
    #     pre_ = True
    #     post_ = False
    # elif models_to_analyze == 'post':
    #     pre_ = False
    #     post_ = True
    # else:
    #     return NotImplementedError

    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_model_save_file = best_model_save_location + '_best_baseline_regression_model_' + detector_type + '.pth'
    # best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    # best_post_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_batch_size = hyper_param['eval_batch_size'][model_id]
    eval_dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
    eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else hyper_param['eval_batch_size'][model_id]

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)


    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel

    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                                     keypoint_detector=detector_type).to(device)

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
        evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param, certification=True,
                     device=device, correction_flag=False)

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
        if visualize_without_corrector:
            print("Without corrector")
            visual_test(test_loader=loader, model=model, correction_flag=False, hyper_param=hyper_param)
        if visualize_with_corrector:
            print("With corrector")
            visual_test(test_loader=loader, model=model, correction_flag=True, hyper_param=hyper_param)

    del model, state_dict

    return None


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.
from learning_objects.expt_self_supervised_correction.evaluation import evaluate


## Wrapper
def train_kp_detectors(detector_type, model_ids, only_models=None):

    for model_id in model_ids:
        if model_id in only_models:
            hyper_param_file = "baseline_training.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param[detector_type]
            hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

            print(">>"*40)
            print("Training Model ID: ", str(model_id))
            train_detector(detector_type=detector_type,
                           model_id=model_id,
                           hyper_param=hyper_param)


def visualize_kp_detectors(detector_type, model_ids, only_models=None,
                           evaluate_models=True,
                           visualize=True,
                           visualize_without_corrector=True,
                           visualize_with_corrector=False,
                           visualize_before=True,
                           visualize_after=True):

    if not visualize:
        visualize_with_corrector, visualize_without_corrector, visualize_before, visualize_after \
            = False, False, False, False

    for model_id in model_ids:
        if model_id in only_models:
            hyper_param_file = "self_supervised_training.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param[detector_type]
            hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

            hyper_param["is_symmetric"] = False

            print(">>"*40)
            print("Analyzing Baseline for Object: ", model_id)
            visualize_detector(detector_type=detector_type,
                               model_id=model_id,
                               hyper_param=hyper_param,
                               evaluate_models=evaluate_models,
                               visualize_without_corrector=visualize_without_corrector,
                               visualize_with_corrector=visualize_with_corrector,
                               visualize_before=visualize_before,
                               visualize_after=visualize_after)




if __name__ == "__main__":

    """
    usage: 
    >> python train_baseline.py "point_transformer" "chair"
    >> python train_baseline.py "pointnet" "chair"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("model_id", help="specify the ycb model id.", type=str)

    args = parser.parse_args()

    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.model_id)
    detector_type = args.detector_type
    model_id = args.model_id
    only_models = [model_id]

    stream = open("model_ids.yml", "r")
    model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']

    # train_kp_detectors(detector_type=detector_type, model_ids=model_ids, only_models=only_models)
    visualize_kp_detectors(detector_type=detector_type, model_ids=model_ids, \
                           visualize_without_corrector=True, only_models=only_models, visualize=True)





