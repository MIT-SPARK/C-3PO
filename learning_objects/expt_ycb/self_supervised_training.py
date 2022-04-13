"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

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
from learning_objects.datasets.ycb import DepthYCBAugment
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

SYMMETRIC_MODEL_IDS = ["001_chips_can", "002_master_chef_can", "004_sugar_box", \
                       "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box" \
                       "009_gelatin_box", "010_potted_meat_can", "036_wood_block", "040_large_marker", \
                       "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]

# Train
def self_supervised_train_one_epoch(training_loader, model, optimizer, correction_flag, device, hyper_param):
    running_loss = 0.
    fra_certi_track = []

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # print("Running batch ", i+1, "/", len(training_loader))
        input_point_cloud, _, _, _ = data
        input_point_cloud = input_point_cloud.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, corrected_keypoints, _, _, correction, predicted_model_keypoints = \
            model(input_point_cloud, correction_flag=correction_flag, need_predicted_keypoints=True)

        # Certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=corrected_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints,
                        epsilon=hyper_param['epsilon'],
                        is_symmetric=hyper_param["is_symmetric"])
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
        # torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss, fra_certi_track


# Val
def validate(validation_loader, model, correction_flag, device, hyper_param):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints = model(input_point_cloud,
                                                                                            correction_flag=correction_flag, need_predicted_keypoints=True)

            # certification
            certi = certify(input_point_cloud=input_point_cloud,
                            predicted_point_cloud=predicted_point_cloud,
                            corrected_keypoints=corrected_keypoints,
                            predicted_model_keypoints=predicted_model_keypoints,
                            epsilon=hyper_param['epsilon'], is_symmetric=hyper_param["is_symmetric"])

            vloss = validation_loss(input_point_cloud,
                                    predicted_point_cloud,
                                    certi=certi)

            running_vloss += vloss

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, certi

            # torch.cuda.empty_cache()
        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


# Train + Val Loop
def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer, correction_flag,
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
                                                                   correction_flag=correction_flag,
                                                                   device=device,
                                                                   hyper_param=hyper_param)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(validation_loader, model, correction_flag=correction_flag,
                             device=device, hyper_param=hyper_param)

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

        # torch.cuda.empty_cache()
        if -avg_vloss > hyper_param['train_stop_cert_threshold']:
            print("ENDING TRAINING. REACHED MAX. CERTIFICATION (AT VALIDATION).")
            break

    return train_loss, val_loss, certi_all_train_batches


# Train
def train_detector(hyper_param, detector_type='point_transformer', model_id="019_pitcher_base"):
    """

    """

    print('-' * 20)
    print("Running self_supervised_training: ", datetime.now())
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    # torch.cuda.empty_cache()

    # ycb
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    best_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_sstrain_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_ssval_loss_' + detector_type + '.pkl'
    cert_save_file = best_model_save_location + '_certi_all_batches_' + detector_type + '.pkl'
    last_epoch_model_dict_file = best_model_save_location + '_last_epoch_self_supervised_kp_' + detector_type + '.pth'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    print("LR_SGD", lr_sgd)
    momentum_sgd = hyper_param['momentum_sgd']
    epsilon = hyper_param['epsilon']
    print("epsilon", epsilon)

    # object symmetry
    if model_id in SYMMETRIC_MODEL_IDS:
        hyper_param["is_symmetric"] = True
    else:
        hyper_param["is_symmetric"] = False

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
    val_dataset = DepthYCBAugment(model_id=model_id,
                           split='val',
                           num_of_points=num_of_points_to_sample)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = self_supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = self_supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)

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
    train_loss, val_loss, fra_cert_ = train_without_supervision(self_supervised_train_loader=self_supervised_train_loader,
                                                                validation_loader=val_loader,
                                                                model=model,
                                                                optimizer=optimizer,
                                                                correction_flag=True,
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

    return None


# Visualize
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

        if i >= 20:
            break


def visualize_detector(hyper_param, detector_type, model_id,
                       evaluate_models=True, models_to_analyze='both',
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
    if models_to_analyze=='both':
        pre_ = True
        post_ = True
    elif models_to_analyze == 'pre':
        pre_ = True
        post_ = False
    elif models_to_analyze == 'post':
        pre_ = False
        post_ = True
    else:
        return NotImplementedError



    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    best_post_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_batch_size = hyper_param['eval_batch_size'][model_id]
    eval_dataset = DepthYCBAugment(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)


    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel

    if pre_:
        model_before = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                                     keypoint_detector=detector_type).to(device)

        if not os.path.isfile(best_pre_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

        state_dict_pre = torch.load(best_pre_model_save_file)
        model_before.load_state_dict(state_dict_pre)

        num_parameters = sum(param.numel() for param in model_before.parameters() if param.requires_grad)
        print("Number of trainable parameters: ", num_parameters)

    if post_:
        model_after = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                                    keypoint_detector=detector_type).to(device)

        if not os.path.isfile(best_post_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

        state_dict_post = torch.load(best_post_model_save_file)
        model_after.load_state_dict(state_dict_post)

        num_parameters = sum(param.numel() for param in model_after.parameters() if param.requires_grad)
        print("Number of trainable parameters: ", num_parameters)

    # Evaluation:
    if evaluate_models:
        if pre_:
            print(">>"*40)
            print("PRE-TRAINED MODEL:")
            print(">>" * 40)
            evaluate(eval_loader=eval_loader, model=model_before, hyper_param=hyper_param, certification=True,
                     device=device, normalize_adds=False)
        if post_:
            print(">>" * 40)
            print("(SELF-SUPERVISED) TRAINED MODEL:")
            print(">>" * 40)
            evaluate(eval_loader=eval_loader, model=model_after, hyper_param=hyper_param, certification=True,
                     device=device, normalize_adds=False)

    # # Visual Test
    dataset_batch_size = 1
    dataset = DepthYCBAugment(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= False,
                            num_of_points=hyper_param['num_of_points_to_sample'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False, pin_memory=True)

    if visualize_before and pre_:
        print(">>" * 40)
        print("VISUALIZING PRE-TRAINED MODEL:")
        print(">>" * 40)
        if visualize_without_corrector:
            print("Without corrector")
            visual_test(test_loader=loader, model=model_before, correction_flag=False, hyper_param=hyper_param)
        if visualize_with_corrector:
            print("With corrector")
            visual_test(test_loader=loader, model=model_before, correction_flag=True, hyper_param=hyper_param)

    if visualize_after and post_:
        print(">>" * 40)
        print("(SELF-SUPERVISED) TRAINED MODEL:")
        print(">>" * 40)
        if visualize_without_corrector:
            print("Without corrector")
            visual_test(test_loader=loader, model=model_after, correction_flag=False, hyper_param=hyper_param)
        if visualize_with_corrector:
            print("With corrector")
            visual_test(test_loader=loader, model=model_after, correction_flag=True, hyper_param=hyper_param)

    if pre_:
        del model_before, state_dict_pre
    if post_:
        del model_after, state_dict_post
    # torch.cuda.empty_cache()
    return None


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.
from learning_objects.expt_self_supervised_correction.evaluation import evaluate


## Wrapper
def train_kp_detectors(detector_type, model_ids, only_models=None):
    for model_id in model_ids:
        if model_id in only_models:
            hyper_param_file = "self_supervised_training.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param[detector_type]
            hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

            print(">>" * 40)
            print("Training Model ID:", str(model_id))
            train_detector(detector_type=detector_type,
                           model_id=model_id,
                           hyper_param=hyper_param)


def visualize_kp_detectors(detector_type, model_ids, only_models=None,
                           evaluate_models=True,
                           models_to_analyze='both',
                           visualize=True,
                           visualize_without_corrector=False,
                           visualize_with_corrector=True,
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

            if model_id in SYMMETRIC_MODEL_IDS:
                hyper_param["is_symmetric"] = True
            else:
                hyper_param["is_symmetric"] = False

            print(">>" * 40)
            print("Analyzing Trained Model for Object: ", model_id)
            visualize_detector(detector_type=detector_type,
                               model_id=model_id,
                               hyper_param=hyper_param,
                               evaluate_models=evaluate_models,
                               models_to_analyze=models_to_analyze,
                               visualize_without_corrector=visualize_without_corrector,
                               visualize_with_corrector=visualize_with_corrector,
                               visualize_before=visualize_before,
                               visualize_after=visualize_after)



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
    only_models = [model_id]
    # this isn't useful when using point transformer because we can't load multiple ProposedModels with point transformer in a row
    # keeping for code monkey param happiness
    with open("class_model_ids.yml", 'r') as stream:
        model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']

    train_kp_detectors(detector_type=detector_type, model_ids=model_ids, only_models=only_models)
    # visualize_kp_detectors(detector_type=detector_type, model_ids=model_ids, only_models=only_models, visualize=True, models_to_analyze='post')





