"""
This code implements full self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

"""

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
import yaml
import argparse
import pickle
# from pytorch3d import ops

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, DepthPC, CLASS_NAME, \
    FixedDepthPC, CLASS_ID, MixedFixedDepthPC
from learning_objects.datasets.ycb import DepthYCB, DepthYCBAugment, MixedDepthYCBAugment, viz_rgb_pcd

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
from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel
from learning_objects.expt_full_self_supervised_correction.evaluation import evaluate

# Train
def self_supervised_train_one_epoch(training_loader, model, optimizer, device, hyper_param):
    running_loss = 0.
    fra_certi_track = []

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        print("i :", i)
        input_point_cloud, _, _ = data
        input_point_cloud = input_point_cloud.to(device)

        # print(input_point_cloud.shape)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, corrected_keypoints, _, _, correction, predicted_model_keypoints = \
            model(input_point_cloud, need_predicted_keypoints=True)

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
def validate(validation_loader, model, device, hyper_param):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, _, _ = vdata
            input_point_cloud = input_point_cloud.to(device)

            # Make predictions for this batch
            predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints = model(input_point_cloud, need_predicted_keypoints=True)

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

            del input_point_cloud, \
                predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, certi

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


# Train + Val Loop
def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer,
                              best_model_save_file, device, hyper_param, train_loss_save_file,
                              val_loss_save_file, cert_save_file):

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
        avg_vloss = validate(validation_loader, model,
                             device=device, hyper_param=hyper_param)

        print('LOSS self-supervised train {}, valid (%cert) {}'.format(ave_loss_self_supervised, -avg_vloss))
        train_loss.add_item(ave_loss_self_supervised)
        val_loss.add_item(-avg_vloss)
        certi_all_train_batches.add_item(_fra_cert)

        # Saving the model with the best vloss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), best_model_save_file)

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
def train_detector(hyper_param, detector_type='point_transformer', model_id="003_cracker_box", use_corrector=True):
    """

    """

    print('-' * 20)
    print("Running self_supervised_training: ", datetime.now())
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    # torch.cuda.empty_cache()

    # shapenet
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'
    best_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_sstrain_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_ssval_loss_' + detector_type + '.pkl'
    cert_save_file = best_model_save_location + '_certi_all_batches_' + detector_type + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    momentum_sgd = hyper_param['momentum_sgd']

    hyper_param["is_symmetric"] = False

    # real dataset:
    self_supervised_train_batch_size = hyper_param['self_supervised_train_batch_size']
    num_of_points_to_sample = hyper_param['num_of_points_to_sample']


    self_supervised_train_dataset = MixedDepthYCBAugment(model_id=model_id, split='train', num_of_points=num_of_points_to_sample)
    print("Dataset length: ", self_supervised_train_dataset.len)
    self_supervised_train_loader = torch.utils.data.DataLoader(self_supervised_train_dataset,
                                                               batch_size=self_supervised_train_batch_size,
                                                               shuffle=True)

    # validation dataset:
    val_batch_size = hyper_param['val_batch_size']
    val_dataset = MixedDepthYCBAugment(model_id=model_id, split='val', num_of_points=num_of_points_to_sample)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=True)

    # Generate a shape category, CAD model objects, etc.
    cad_models = self_supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = self_supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # model
    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, correction_flag=use_corrector).to(device)

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
                                                                cert_save_file=cert_save_file)

    with open(train_loss_save_file, 'wb') as outp:
        pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(val_loss_save_file, 'wb') as outp:
        pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(cert_save_file, 'wb') as outp:
        pickle.dump(fra_cert_, outp, pickle.HIGHEST_PROTOCOL)

    return None


# Visualize
def visual_test(test_loader, model, hyper_param, device=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # torch.cuda.empty_cache()

    cad_models = test_loader.dataset._get_cad_models()
    model_keypoints = test_loader.dataset._get_model_keypoints()
    cad_models = cad_models.to(device)
    model_keypoints = model_keypoints.to(device)

    for i, vdata in enumerate(test_loader):
        input_point_cloud, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)
        keypoints_target = R_target @ model_keypoints + t_target

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, predicted_model_keypoints \
            = model(input_point_cloud, need_predicted_keypoints=True)

        # certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=predicted_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints,
                        epsilon=hyper_param['epsilon'], is_symmetric=hyper_param["is_symmetric"])

        print("Certifiable: ", certi)

        # add-s
        pc_t = R_target @ cad_models + t_target
        add_s = add_s_error(predicted_point_cloud=predicted_point_cloud,
                            ground_truth_point_cloud=pc_t,
                            threshold=hyper_param['adds_threshold'])
        print("ADD-S: ", add_s)

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_p = predicted_point_cloud.clone().detach().to('cpu')
        pc_t = pc_t.clone().detach().to('cpu')
        kp = keypoints_target.clone().detach().to('cpu')
        kp_p = predicted_keypoints.clone().detach().to('cpu')
        print("DISPLAY: INPUT PC")
        display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc,
                        target_keypoints=None)
        print("DISPLAY: INPUT AND PREDICTED PC")
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc_p,
                        target_keypoints=kp)
        print("DISPLAY: TRUE AND PREDICTED PC")
        display_results(input_point_cloud=pc_p, detected_keypoints=kp_p, target_point_cloud=pc_t,
                        target_keypoints=kp)

        del pc, pc_p, kp, kp_p, pc_t
        del input_point_cloud, keypoints_target, R_target, t_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        if i >= 9:
            break


def visualize_detector(hyper_param, detector_type, model_id,
                       dataset_model_id,
                       evaluate_models=True, models_to_analyze='both',
                       use_corrector=True,
                       visualize=False, device=None):
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
    eval_dataset = MixedDepthYCBAugment(model_id=dataset_model_id, split='test', num_of_points=hyper_param['num_of_points_to_sample'], mixed_data=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True)


    # model
    temp_dataset = MixedDepthYCBAugment(model_id=model_id, split='test', num_of_points=hyper_param['num_of_points_to_sample'], mixed_data=False)

    cad_models = temp_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = temp_dataset._get_model_keypoints().to(torch.float).to(device=device)
    del temp_dataset

    if pre_:
        model_before = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                                     keypoint_detector=detector_type, correction_flag=use_corrector).to(device)

        if not os.path.isfile(best_pre_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

        state_dict_pre = torch.load(best_pre_model_save_file)
        model_before.load_state_dict(state_dict_pre)

        num_parameters = sum(param.numel() for param in model_before.parameters() if param.requires_grad)
        print("Number of trainable parameters: ", num_parameters)

    if post_:
        model_after = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                                    keypoint_detector=detector_type, correction_flag=use_corrector).to(device)

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
                     device=device)
        if post_:
            print(">>" * 40)
            print("(SELF-SUPERVISED) TRAINED MODEL:")
            print(">>" * 40)
            evaluate(eval_loader=eval_loader, model=model_after, hyper_param=hyper_param, certification=True,
                     device=device)

    # # Visual Test
    dataset_len = 20
    dataset_batch_size = 1
    # dataset = DepthPC(class_id=class_id,
    #                   model_id=model_id,
    #                   n=hyper_param['num_of_points_selfsupervised'],
    #                   num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
    #                   dataset_len=dataset_len,
    #                   rotate_about_z=True)
    dataset = MixedDepthYCBAugment(model_id=model_id, split='test', num_of_points=hyper_param['num_of_points_to_sample'], mixed_data=False)

    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=True)

    if visualize and pre_:
        print(">>" * 40)
        print("VISUALIZING PRE-TRAINED MODEL:")
        print(">>" * 40)
        visual_test(test_loader=loader, model=model_before, hyper_param=hyper_param)

    if visualize and post_:
        print(">>" * 40)
        print("(SELF-SUPERVISED) TRAINED MODEL:")
        print(">>" * 40)
        visual_test(test_loader=loader, model=model_after, hyper_param=hyper_param)

    if pre_:
        del model_before, state_dict_pre
    if post_:
        del model_after, state_dict_post

    return None


## Wrapper
def train_kp_detectors(detector_type, model_ids, only_models=None, use_corrector=True):

    for model_id in model_ids:
        if model_id in only_models:
            hyper_param_file = "./full_self_supervised_training_ycb.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param[detector_type]
            hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

            print(">>"*40)
            print("Training: Model ID:", str(model_id))
            train_detector(detector_type=detector_type,
                           model_id=model_id,
                           hyper_param=hyper_param,
                           use_corrector=use_corrector)


def visualize_kp_detectors(detector_type, model_ids, dataset_name, only_models=None,
                           evaluate_models=True,
                           models_to_analyze='both',
                           visualize=True,
                           use_corrector=True):

    dataset_model_id = dataset_name

    for model_ids in model_ids:
        if model_id in only_models:
            hyper_param_file = "./full_self_supervised_training_ycb.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param[detector_type]
            hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

            hyper_param["is_symmetric"] = False

            print(">>"*40)
            print("Analyzing Trained Model for Object: ", str(model_id))
            print("On dataset of: ", dataset_name)
            # print("class id: ", class_id)
            visualize_detector(detector_type=detector_type,
                               model_id=model_id,
                               dataset_model_id=dataset_model_id,
                               hyper_param=hyper_param,
                               evaluate_models=evaluate_models,
                               models_to_analyze=models_to_analyze,
                               use_corrector=use_corrector,
                               visualize=visualize)




if __name__ == "__main__":

    """
    usage: 
    >> python full_self_supervised_training.py "point_transformer" "003_cracker_box"
    """
    #free memory
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("model_id", help="specify the ycb model id.", type=str)

    args = parser.parse_args()

    detector_type = args.detector_type
    model_id = args.model_id
    only_models = [model_id]

    with open("class_model_ids_ycb.yml", 'r') as stream:
        model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']

    train_kp_detectors(detector_type=detector_type, model_ids=model_ids, only_models=only_models, use_corrector=True)
    # visualize_kp_detectors(detector_type=detector_type, model_ids=model_ids, dataset_name='006_mustard_bottle', only_models=only_models, visualize=True,
    #                        evaluate_models=True, use_corrector=True, models_to_analyze='post')





