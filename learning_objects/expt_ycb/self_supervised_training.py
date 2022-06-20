"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

"""

import argparse
import numpy as np
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

# model and datasets
from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel
from learning_objects.datasets.ycb import DepthYCB, DepthYCBAugment, viz_rgb_pcd, MODEL_TO_KPT_GROUPS as MODEL_TO_KPT_GROUPS_YCB
from learning_objects.models.certifiability import confidence, confidence_kp

from learning_objects.utils.general import display_results, TrackingMeter, temp_expt_1_viz

# loss functions
from learning_objects.utils.loss_functions import self_supervised_training_loss \
    as self_supervised_loss, self_supervised_validation_loss as validation_loss, certify
# evaluation metrics
from learning_objects.utils.evaluation_metrics import add_s_error, \
    is_pcd_nondegenerate
from learning_objects.expt_self_supervised_correction.evaluation import evaluate

def self_supervised_train_one_epoch(training_loader, model, optimizer, device, hyper_param):
    running_loss = 0.
    fra_certi_track = []

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        input_point_cloud, _, _, _ = data
        input_point_cloud = input_point_cloud.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, corrected_keypoints, _, _, correction, predicted_model_keypoints = \
            model(input_point_cloud)

        # Certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=corrected_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints,
                        epsilon=hyper_param['epsilon'])
        certi = certi.squeeze(-1)  # (B,)

        # Compute the loss and its gradients
        loss, pc_loss, kp_loss, fra_cert = \
            self_supervised_loss(input_point_cloud=input_point_cloud,
                                 predicted_point_cloud=predicted_point_cloud,
                                 keypoint_correction=correction, certi=certi, theta=hyper_param['theta'])
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1 == 0:
            print("Batch ", (i+1), " loss: ", loss.item(), " pc loss: ", pc_loss.item(), " kp loss: ", kp_loss.item())
            print("Batch ", (i + 1), " fra cert: ", fra_cert.item())

        fra_certi_track.append(fra_cert)

        del input_point_cloud, predicted_point_cloud, correction

    ave_tloss = running_loss / (i + 1)

    return ave_tloss, fra_certi_track


def validate(validation_loader, model, device, hyper_param):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, corrected_keypoints, \
            R_predicted, t_predicted, correction, \
            predicted_model_keypoints = model(input_point_cloud)

            # certification
            certi = certify(input_point_cloud=input_point_cloud,
                            predicted_point_cloud=predicted_point_cloud,
                            corrected_keypoints=corrected_keypoints,
                            predicted_model_keypoints=predicted_model_keypoints,
                            epsilon=hyper_param['epsilon'])

            vloss = validation_loss(input_point_cloud,
                                    predicted_point_cloud,
                                    certi=certi)

            running_vloss += vloss

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, certi

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


# Train + Val Loop
def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer,
                              best_model_save_file, device, hyper_param, train_loss_save_file,
                              val_loss_save_file, cert_save_file, last_epoch_model_dict_file):

    num_epochs = hyper_param['num_epochs']
    best_vloss = 1_000_000.

    train_loss = TrackingMeter()
    val_loss = TrackingMeter()
    certi_all_train_batches = TrackingMeter()
    epoch_number = 0

    for epoch in range(num_epochs):
        print('EPOCH :', epoch_number + 1, "TIME: ", datetime.now())

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on real data with self-supervision: ")
        ave_loss_self_supervised, _fra_cert = self_supervised_train_one_epoch(self_supervised_train_loader,
                                                                   model,
                                                                   optimizer,
                                                                   device=device,
                                                                   hyper_param=hyper_param)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(validation_loader, model, device=device, hyper_param=hyper_param)

        print('LOSS self-supervised train {}, valid (%cert) {}'.format(ave_loss_self_supervised, -avg_vloss))
        train_loss.add_item(ave_loss_self_supervised)
        val_loss.add_item(-avg_vloss)
        certi_all_train_batches.add_item(_fra_cert)

        # Saving the model with the best vloss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), best_model_save_file)

        # Saving the model of the last epoch
        if epoch == num_epochs-1:
            torch.save(model.state_dict(), last_epoch_model_dict_file)

        epoch_number += 1

        with open(train_loss_save_file, 'wb') as outp:
            pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(val_loss_save_file, 'wb') as outp:
            pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(cert_save_file, 'wb') as outp:
            pickle.dump(_fra_cert, outp, pickle.HIGHEST_PROTOCOL)

        if -avg_vloss > hyper_param['train_stop_cert_threshold']:
            print("ENDING TRAINING. REACHED MAX. CERTIFICATION (AT VALIDATION).")
            break

    return train_loss, val_loss, certi_all_train_batches


# Train
def train_detector(hyper_param, detector_type='point_transformer', model_id="019_pitcher_base", use_corrector=True):
    """

    """

    print('-' * 20)
    print("Running self_supervised_training: ", datetime.now())
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)

    # ycb
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'
    best_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + 'data_augment.pth'
    train_loss_save_file = best_model_save_location + '_sstrain_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_ssval_loss_' + detector_type + '.pkl'
    cert_save_file = best_model_save_location + '_certi_all_batches_' + detector_type + '.pkl'
    last_epoch_model_dict_file = best_model_save_location + '_last_epoch_self_supervised_kp_' + detector_type + '.pth'
    hyperparam_file = best_model_save_location + '_self_supervised_kp_' + detector_type + '_hyperparams.pth'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    print("LR_SGD", lr_sgd)
    momentum_sgd = hyper_param['momentum_sgd']
    epsilon = hyper_param['epsilon']
    print("epsilon", epsilon)

    # real dataset:
    self_supervised_train_batch_size = hyper_param['self_supervised_train_batch_size'][model_id]
    num_of_points_to_sample = hyper_param['num_of_points_to_sample']

    self_supervised_train_dataset = DepthYCBAugment(model_id=model_id,
                                             split='train',
                                             num_of_points=num_of_points_to_sample)
    self_supervised_train_loader = torch.utils.data.DataLoader(self_supervised_train_dataset,
                                                               batch_size=self_supervised_train_batch_size,
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
    cad_models = self_supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = self_supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # model
    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector,
                          need_predicted_keypoints=True).to(device)

    if not os.path.isfile(sim_trained_model_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
    state_dict = torch.load(sim_trained_model_file)
    model.load_state_dict(state_dict)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)

    # training
    train_loss, val_loss, fra_cert_ = train_without_supervision(self_supervised_train_loader=self_supervised_train_loader,
                                                                validation_loader=val_loader,
                                                                model=model,
                                                                optimizer=optimizer,
                                                                best_model_save_file=best_model_save_file,
                                                                device=device,
                                                                hyper_param=hyper_param,
                                                                train_loss_save_file=train_loss_save_file,
                                                                val_loss_save_file=val_loss_save_file,
                                                                cert_save_file=cert_save_file,
                                                                last_epoch_model_dict_file=last_epoch_model_dict_file)

    with open(train_loss_save_file, 'wb') as outp:
        pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(val_loss_save_file, 'wb') as outp:
        pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(cert_save_file, 'wb') as outp:
        pickle.dump(fra_cert_, outp, pickle.HIGHEST_PROTOCOL)

    with open(hyperparam_file, 'wb') as outp:
        pickle.dump(hyper_param, outp, pickle.HIGHEST_PROTOCOL)

    return None


# Visualize
def visual_test(test_loader, model, device=None, hyper_param=None, degeneracy_eval=False):

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
        if degeneracy_eval:
            nondeg_indicator = is_pcd_nondegenerate(model.model_id, input_point_cloud, predicted_model_keypoints,
                                                    MODEL_TO_KPT_GROUPS_YCB)
            print("Is Non-Degenerate", nondeg_indicator)

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_p = predicted_point_cloud.clone().detach().to('cpu')
        kp = keypoints_target.clone().detach().to('cpu')
        kp_p = predicted_keypoints.clone().detach().to('cpu')

        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=None,
                        target_keypoints=kp)
        # if i == 7:
        #    display_results(input_point_cloud=None, detected_keypoints=kp_p, target_point_cloud=pc_p,
        #                 target_keypoints=kp)
        #    # pcd_rgb = viz_rgb_pcd("002_master_chef_can", "NP2", "NP5", "207")
        #    # pcd_rgb = viz_rgb_pcd("052_extra_large_clamp", "NP1", "NP5", "201") # 7
        #    pcd_rgb = viz_rgb_pcd("003_cracker_box", "NP2", "NP5", "63")
        #
        #    pcd_torch = torch.from_numpy(np.asarray(pcd_rgb.points)).transpose(0, 1)  # (3, m)
        #    pcd_torch = pcd_torch.to(torch.float).unsqueeze(0)
        #
        #    temp_expt_1_viz(pcd_torch, model_keypoints=kp_p, gt_keypoints=kp, colors=pcd_rgb.colors)

        del pc, pc_p, kp, kp_p
        del input_point_cloud, keypoints_target, R_target, t_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        if i >= 20:
            break

def evaluate_model(detector_type, model_id,
                   evaluate_models=True,
                   visualize=True,
                   use_corrector=True,
                   models_to_analyze='post',
                   degeneracy_eval=False,
                   average_metrics=False):
    if models_to_analyze == 'pre':
        evaluate_pretrained = True
        evaluate_trained = False
    elif models_to_analyze == 'post':
        evaluate_pretrained = False
        evaluate_trained = True
    else:
        return NotImplementedError

    hyper_param_file = "self_supervised_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]
    hyper_param['epsilon'] = hyper_param['epsilon'][model_id]
    print(">>" * 40)
    print("Analyzing Trained Model for Object: " + str(model_id))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'
    best_post_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + 'data_augment.pth'
    hyper_param_file = best_model_save_location + '_self_supervised_kp_point_transformer_hyperparams.pth'
    if os.path.exists(hyper_param_file):
        with open(hyper_param_file, "rb") as f:
            hyper_param = pickle.load(f)

    metrics_file_base = best_model_save_location + "_averaged_eval_metrics"
    if evaluate_pretrained:
        metrics_file_base = metrics_file_base + "_sim_trained"
    if degeneracy_eval:
        metrics_file_base = metrics_file_base + "_with_degeneracy"
    if not use_corrector:
        metrics_file_base = metrics_file_base + "_no_corrector"
    metrics_file = metrics_file_base + ".pkl"

    if average_metrics:
        if os.path.exists(metrics_file):
            with open(metrics_file, "rb") as f:
                averaged_metrics = pickle.load(f)
        else:
            averaged_metrics = [0] * 6 if not degeneracy_eval else [0] * 14
        print("LOADING AVERAGE OF " + str(averaged_metrics[-1]) + " RUNS!")
        print(averaged_metrics)
        averaged_metrics = np.array(averaged_metrics)

    # Evaluation
    # test dataset:
    eval_dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
    eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else hyper_param['eval_batch_size'][model_id]

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)


    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector,
                          need_predicted_keypoints=True).to(device)

    if evaluate_pretrained:
        if not os.path.isfile(best_pre_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
        state_dict = torch.load(best_pre_model_save_file)
    elif evaluate_trained:
        if not os.path.isfile(best_post_model_save_file):
            print("ERROR: CAN'T LOAD TRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
        state_dict = torch.load(best_post_model_save_file, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # Evaluation:
    if evaluate_models:
        print(">>"*40)
        log_info = "PRE-TRAINED MODEL:" if evaluate_pretrained else "(SELF-SUPERVISED) TRAINED MODEL:"
        print(log_info)
        print(">>" * 40)
        eval_metrics = evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param, certification=True,
                 device=device, normalize_adds=False, degeneracy=degeneracy_eval)

    # # Visual Test
    dataset_batch_size = 1
    dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False, pin_memory=True)

    if visualize:
        print(">>" * 40)
        log_info = "VISUALIZING PRE-TRAINED MODEL:" if evaluate_pretrained else "VISUALIZING (SELF-SUPERVISED) TRAINED MODEL:"
        print(log_info)
        print(">>" * 40)
        visual_test(test_loader=loader, model=model, hyper_param=hyper_param,
                    degeneracy_eval=degeneracy_eval)

    del state_dict, model
    if average_metrics and evaluate_models:
        eval_metrics = eval_metrics + [1]
        averaged_metrics = averaged_metrics + np.array(eval_metrics)
        for elt_idx in range(len(averaged_metrics)):
            if np.isnan(averaged_metrics[elt_idx]):
                averaged_metrics[elt_idx] = 0
        print("NEW AVERAGED METRICS: ", averaged_metrics)
        with open(metrics_file, 'wb+') as f:
            pickle.dump(averaged_metrics, f)
    elif evaluate_models:
        return eval_metrics
    else:
        return None


if __name__ == "__main__":

    """
    usage: 
    >> python self_supervised_training.py "point_transformer" "021_bleach_cleanser"
    """
    #free memory
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("model_id", help="specify the ycb model id.", type=str)

    args = parser.parse_args()

    print("KP detector type: ", args.detector_type)
    print("CAD Model class: ", args.model_id)
    detector_type = args.detector_type
    model_id = args.model_id
    # keeping for code monkey param happiness
    with open("model_ids.yml", 'r') as stream:
        model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']
        if model_id not in model_ids:
            raise Exception('Invalid model_id')

    evaluate_model(detector_type=detector_type, model_id=model_id, evaluate_models=True,
                   visualize=False, use_corrector=True, models_to_analyze = 'post', degeneracy_eval=False, average_metrics=False)



