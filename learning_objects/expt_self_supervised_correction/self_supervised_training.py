"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

"""

import argparse
import os
import pickle
import sys
import torch
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")

from learning_objects.datasets.keypointnet import DepthPC, CLASS_NAME, FixedDepthPC, CLASS_ID
from learning_objects.utils.general import TrackingMeter
from learning_objects.utils.visualization_utils import display_results

from learning_objects.utils.loss_functions import certify, self_supervised_training_loss as self_supervised_loss, \
    self_supervised_validation_loss as validation_loss

from learning_objects.utils.evaluation_metrics import add_s_error
from learning_objects.expt_self_supervised_correction.evaluation import evaluate
from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel

# Train
def self_supervised_train_one_epoch(training_loader, model, optimizer, device, hyper_param):
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


# Val
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
            predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, correction, \
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
        avg_vloss = validate(validation_loader, model, device=device, hyper_param=hyper_param)

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

        if -avg_vloss > hyper_param['train_stop_cert_threshold']:
            print("ENDING TRAINING. REACHED MAX. CERTIFICATION (AT VALIDATION).")
            break

    return train_loss, val_loss, certi_all_train_batches


# Train
def train_detector(hyper_param, detector_type='pointnet', class_id="03001627",
                   model_id="1e3fba4500d20bb49b9f2eb77f5e247e", use_corrector=True):
    """

    """

    print('-' * 20)
    print("Running self_supervised_training: ", datetime.now())
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)

    # shapenet
    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    if class_name == 'car':
        sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
        # sim_trained_model_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '_mid.pth'
    else:
        sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'

    best_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_sstrain_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_ssval_loss_' + detector_type + '.pkl'
    cert_save_file = best_model_save_location + '_certi_all_batches_' + detector_type + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    momentum_sgd = hyper_param['momentum_sgd']

    # real dataset:
    self_supervised_train_dataset_len = hyper_param['self_supervised_train_dataset_len']
    self_supervised_train_batch_size = hyper_param['self_supervised_train_batch_size']
    num_of_points_to_sample = hyper_param['num_of_points_to_sample']
    num_of_points_selfsupervised = hyper_param['num_of_points_selfsupervised']

    # self_supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=num_of_points_selfsupervised,
    #                                         num_of_points_to_sample=num_of_points_to_sample,
    #                                         dataset_len=self_supervised_train_dataset_len,
    #                                         rotate_about_z=True)
    self_supervised_train_dataset = FixedDepthPC(class_id=class_id,
                                                 model_id=model_id,
                                                 n=num_of_points_selfsupervised,
                                                 num_of_points_to_sample=num_of_points_to_sample,
                                                 base_dataset_folder=hyper_param['dataset_folder'])
    print("Dataset length: ", self_supervised_train_dataset.len)
    self_supervised_train_loader = torch.utils.data.DataLoader(self_supervised_train_dataset,
                                                               batch_size=self_supervised_train_batch_size,
                                                               shuffle=False)

    # validation dataset:
    val_dataset_len = hyper_param['val_dataset_len']
    val_batch_size = hyper_param['val_batch_size']
    val_dataset = DepthPC(class_id=class_id,
                          model_id=model_id,
                          n=num_of_points_selfsupervised,
                          num_of_points_to_sample=num_of_points_to_sample,
                          dataset_len=val_dataset_len,
                          rotate_about_z=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=val_batch_size,
                                             shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = self_supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = self_supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # model
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, correction_flag=use_corrector,
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

    cad_models = test_loader.dataset._get_cad_models()
    cad_models = cad_models.to(device)

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
        display_results(input_point_cloud=pc, detected_keypoints=None, target_point_cloud=None,
                        target_keypoints=kp)
        print("DISPLAY: INPUT AND PREDICTED PC")
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc_p,
                        target_keypoints=kp)
        print("DISPLAY: TRUE AND PREDICTED PC")
        display_results(input_point_cloud=pc_t, detected_keypoints=kp_p, target_point_cloud=pc_p,
                        target_keypoints=kp)

        del pc, pc_p, kp, kp_p, pc_t
        del input_point_cloud, keypoints_target, R_target, t_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        if i >= 9:
            break


def visualize_detector(hyper_param, detector_type, class_id, model_id,
                       evaluate_models=True, models_to_analyze='post',
                       use_corrector=True,
                       visualize=False, device=None,
                       cross=False, cross_class_id=None, cross_model_id=None):
    """

    """

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if models_to_analyze == 'pre':
        pre_ = True
        post_ = False
    elif models_to_analyze == 'post':
        pre_ = False
        post_ = True
    else:
        return NotImplementedError



    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    best_post_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_dataset_len = hyper_param['eval_dataset_len']
    eval_batch_size = hyper_param['eval_batch_size']
    eval_dataset = FixedDepthPC(class_id=class_id, model_id=model_id,
                                n=hyper_param['num_of_points_selfsupervised'],
                                num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                                dataset_len=eval_dataset_len,
                                rotate_about_z=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    if cross:
        eval_dataset = FixedDepthPC(class_id=cross_class_id, model_id=cross_model_id,
                                    n=hyper_param['num_of_points_selfsupervised'],
                                    num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                                    dataset_len=eval_dataset_len,
                                    rotate_about_z=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    if pre_:
        model_before = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                                     keypoint_detector=detector_type, correction_flag=use_corrector,
                                     need_predicted_keypoints=True).to(device)

        if not os.path.isfile(best_pre_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

        state_dict_pre = torch.load(best_pre_model_save_file)
        model_before.load_state_dict(state_dict_pre)

        num_parameters = sum(param.numel() for param in model_before.parameters() if param.requires_grad)
        print("Number of trainable parameters: ", num_parameters)

    if post_:
        model_after = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                                    keypoint_detector=detector_type, correction_flag=use_corrector,
                                    need_predicted_keypoints=True).to(device)

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
    dataset = DepthPC(class_id=class_id,
                      model_id=model_id,
                      n=hyper_param['num_of_points_selfsupervised'],
                      num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                      dataset_len=dataset_len,
                      rotate_about_z=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False)

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

def evaluate_model(detector_type, class_name, model_id,
                   evaluate_models=True,
                   models_to_analyze='post',
                   visualize=True,
                   use_corrector=True,
                   cross=False,
                   cross_class_id=None,
                   cross_model_id=None):
    class_id = CLASS_ID[class_name]

    if cross:
        hyper_param_file = "../expt_full_self_supervised_correction/full_self_supervised_training.yml"
    else:
        hyper_param_file = "self_supervised_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]
    hyper_param['epsilon'] = hyper_param['epsilon'][class_name]


    print(">>"*40)
    print("Analyzing Trained Model for Object: ", class_name, "; Model ID:", str(model_id))
    visualize_detector(detector_type=detector_type,
                       class_id=class_id,
                       model_id=model_id,
                       hyper_param=hyper_param,
                       evaluate_models=evaluate_models,
                       models_to_analyze=models_to_analyze,
                       use_corrector=use_corrector,
                       visualize=visualize,
                       cross=cross,
                       cross_class_id=cross_class_id,
                       cross_model_id=cross_model_id)




