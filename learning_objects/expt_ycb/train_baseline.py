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
from learning_objects.utils.general import TrackingMeter
from learning_objects.utils.visualization_utils import display_results

from learning_objects.utils.loss_functions import certify, self_supervised_training_loss \
    as self_supervised_loss, self_supervised_validation_loss as validation_loss
# evaluation metrics
from learning_objects.utils.evaluation_metrics import add_s_error

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

    # ycb
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
