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

from learning_objects.datasets.ycb import DepthYCB, DepthYCBAugment
from learning_objects.utils.general import display_results, TrackingMeter

from learning_objects.utils.loss_functions import certify, self_supervised_training_loss \
    as self_supervised_loss, self_supervised_validation_loss as validation_loss
# evaluation metrics
from learning_objects.expt_self_supervised_correction.evaluation_metrics import evaluation_error, add_s_error

from learning_objects.expt_ycb.supervised_training import train_with_supervision
from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel


# Train
def train_detector(hyper_param, detector_type='point_transformer', model_id="019_pitcher_base", use_corrector=False):
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

    sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'
    best_model_save_file = best_model_save_location + '_best_baseline_regression_model_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_baseline_train_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_baseline_val_loss_' + detector_type + '.pkl'
    # cert_save_file = best_model_save_location + '_certi_all_batches_' + regression_model + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['baseline_lr_sgd']
    momentum_sgd = hyper_param['baseline_momentum_sgd']

    # real dataset:
    train_batch_size = hyper_param['self_supervised_train_batch_size'][model_id]
    num_of_points_to_sample = hyper_param['num_of_points_to_sample']

    train_dataset = DepthYCBAugment(model_id=model_id,
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
    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector).to(device)

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
                       evaluate_models=True,
                       use_corrector=False,
                       visualize_before=True, visualize_after=True, device=None):
    """

    """
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_model_save_file = best_model_save_location + '_best_baseline_regression_model_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
    eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else hyper_param['eval_batch_size'][model_id]
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)


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
        evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param, certification=True,
                     device=device)

    # # Visual Test
    if visualize_before:
        dataset_batch_size = 1
        dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= False,
                            num_of_points=hyper_param['num_of_points_to_sample'])
        loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False)
        print(">>" * 40)
        print("VISUALIZING PRE-TRAINED MODEL:")
        print(">>" * 40)
        visual_test(test_loader=loader, model=model, hyper_param=hyper_param)

    del model, state_dict

    return None


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.
from learning_objects.expt_self_supervised_correction.evaluation import evaluate

def visualize_kp_detectors(detector_type, model_ids, only_models=None,
                           evaluate_models=True,
                           visualize=True,
                           use_corrector=False,
                           visualize_before=True,
                           visualize_after=True):

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
    if model_id not in model_ids:
        raise Exception('Invalid model_id')

    # visualize_kp_detectors(detector_type=detector_type, model_ids=model_ids, use_corrector=False, only_models=only_models, visualize=True)





