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
import pickle
from pytorch3d import ops

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, DepthPC, CLASS_NAME
from learning_objects.models.certifiability import confidence, confidence_kp

from learning_objects.utils.general import display_results, TrackingMeter

# loss functions
from learning_objects.expt_self_supervised_correction.loss_functions import chamfer_loss
from learning_objects.expt_self_supervised_correction.loss_functions import certify
from learning_objects.expt_self_supervised_correction.loss_functions import self_supervised_training_loss \
    as self_supervised_loss
from learning_objects.expt_self_supervised_correction.loss_functions import self_supervised_validation_loss \
    as validation_loss
# evaluation metrics
from learning_objects.expt_self_supervised_correction.evaluation_metrics import evaluation_error


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

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


# Train + Val Loop
def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer, correction_flag,
                              best_model_save_file, device, hyper_param):

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

        epoch_number += 1

        # torch.cuda.empty_cache()

    return train_loss, val_loss, certi_all_train_batches


# Train
def train_detector(hyper_param, detector_type='pointnet', class_id="03001627",
                   model_id="1e3fba4500d20bb49b9f2eb77f5e247e"):
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
    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    sim_trained_model_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    best_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_sstrain_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_ssval_loss_' + detector_type + '.pkl'
    cert_save_file = best_model_save_location + '_certi_all_batches_' + detector_type + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    momentum_sgd = hyper_param['momentum_sgd']

    # object symmetry
    if class_name == "bottle":
        hyper_param["is_symmetric"] = True
    else:
        hyper_param["is_symmetric"] = False

    # real dataset:
    self_supervised_train_dataset_len = hyper_param['self_supervised_train_dataset_len']
    self_supervised_train_batch_size = hyper_param['self_supervised_train_batch_size']
    num_of_points_to_sample = hyper_param['num_of_points_to_sample']
    num_of_points_selfsupervised = hyper_param['num_of_points_selfsupervised']

    self_supervised_train_dataset = DepthPC(class_id=class_id,
                                            model_id=model_id,
                                            n=num_of_points_selfsupervised,
                                            num_of_points_to_sample=num_of_points_to_sample,
                                            dataset_len=self_supervised_train_dataset_len,
                                            rotate_about_z=True)
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
    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, use_pretrained_regression_model=False).to(device)            #ToDo: use_pretrained_regression_model needs to be depreciated.

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
                                                                hyper_param=hyper_param)

    with open(train_loss_save_file, 'wb') as outp:
        pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(val_loss_save_file, 'wb') as outp:
        pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(cert_save_file, 'wb') as outp:
        pickle.dump(fra_cert_, outp, pickle.HIGHEST_PROTOCOL)

    return None


# Visualize
def visual_test(test_loader, model, correction_flag=False, device=None):

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
                        predicted_model_keypoints=predicted_model_keypoints)

        print("Certifiable: ", certi)

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


def visualize_detector(hyper_param, detector_type, class_id, model_id):
    """

    """

    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    # torch.cuda.empty_cache()

    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    best_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_dataset_len = hyper_param['eval_dataset_len']
    eval_batch_size = hyper_param['eval_batch_size']
    eval_dataset = DepthPC(class_id=class_id, model_id=model_id,
                           n=hyper_param['num_of_points_selfsupervised'],
                           num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                           dataset_len=eval_dataset_len,
                           rotate_about_z=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)


    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, use_pretrained_regression_model=False).to(device)            # ToDo: use_pretrained_regression_model needs to be depreciated.

    if not os.path.isfile(best_model_save_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
    state_dict = torch.load(best_model_save_file)
    model.load_state_dict(state_dict)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # Evaluatiion:
    print("Evaluating model: ")
    evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param, certification=True, device=device)

    # # Visual Test
    print("Visualizing the trained model.")
    dataset_len = 20
    dataset_batch_size = 1
    dataset = DepthPC(class_id=class_id,
                      model_id=model_id,
                      n=hyper_param['num_of_points_selfsupervised'],
                      num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                      dataset_len=dataset_len,
                      rotate_about_z=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False)

    visual_test(test_loader=loader, model=model, correction_flag=False)
    visual_test(test_loader=loader, model=model, correction_flag=True)

    return None


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.
def evaluate(eval_loader, model, hyper_param, certification=True, device=None):

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        pc_err = 0.0
        kp_err = 0.0
        R_err = 0.0
        t_err = 0.0

        pc_err_cert = 0.0
        kp_err_cert = 0.0
        R_err_cert = 0.0
        t_err_cert = 0.0

        num_cert = 0.0
        num_batches = len(eval_loader)

        for i, vdata in enumerate(eval_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)
            batch_size = input_point_cloud.shape[0]

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints\
                = model(input_point_cloud, correction_flag=True, need_predicted_keypoints=True)

            if certification:
                certi = certify(input_point_cloud=input_point_cloud,
                                predicted_point_cloud=predicted_point_cloud,
                                corrected_keypoints=predicted_keypoints,
                                predicted_model_keypoints=predicted_model_keypoints,
                                epsilon=hyper_param['epsilon'])

            # fraction certifiable
            # error of all objects
            # error of certified objects

            pc_err_, kp_err_, R_err_, t_err_ = \
                evaluation_error(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                   output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))

            # error for all objects
            pc_err += pc_err_.sum()
            kp_err += kp_err_.sum()
            R_err += R_err_.sum()
            t_err += t_err_.sum()

            if certification:
                # fraction certifiable
                num_cert += certi.sum()

                # error for certifiable objects
                pc_err_cert += (pc_err_ * certi).sum()
                kp_err_cert += (kp_err_ * certi).sum()
                R_err_cert += (R_err_ * certi).sum()
                t_err_cert += (t_err_ * certi).sum()

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        # avg_vloss = running_vloss / (i + 1)
        ave_pc_err = pc_err / ((i + 1)*batch_size)
        ave_kp_err = kp_err / ((i + 1)*batch_size)
        ave_R_err = R_err / ((i + 1)*batch_size)
        ave_t_err = t_err / ((i + 1)*batch_size)

        if certification:
            ave_pc_err_cert = pc_err_cert / num_cert
            ave_kp_err_cert = kp_err_cert / num_cert
            ave_R_err_cert = R_err_cert / num_cert
            ave_t_err_cert = t_err_cert / num_cert

            fra_cert = num_cert / ((i + 1)*batch_size)

        print(">>>>>>>>>>>>>>>> EVALUATING MODEL >>>>>>>>>>>>>>>>>>>>")
        print("Evaluating performance across all objects:")
        print("pc error: ", ave_pc_err.item())
        print("kp error: ", ave_kp_err.item())
        print("R error: ", ave_R_err.item())
        print("t error: ", ave_t_err.item())
        print("Evaulating certification: ")
        print("fraction certifiable: ", fra_cert.item())
        print("Evaluating performance for certifiable objects: ")
        print("pc error: ", ave_pc_err_cert.item())
        print("kp error: ", ave_kp_err_cert.item())
        print("R error: ", ave_R_err_cert.item())
        print("t error: ", ave_t_err_cert.item())

    return None




if __name__ == "__main__":

    class_id = "03001627"  # chair
    class_name = CLASS_NAME[class_id]
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    stream = open("self_supervised_training_pointnet.yml", 'r')
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    train_detector(detector_type='pointnet', class_id=class_id, model_id=model_id, hyper_param=hyper_param)
    visualize_detector(detector_type='pointnet', class_id=class_id, model_id=model_id, hyper_param=hyper_param)

    # stream = open("self_supervised_training_point_transformer.yml", 'r')
    # hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    # train_detector(detector_type='point_transformer', class_id=class_id, model_id=model_id, hyper_param=hyper_param)
    # visualize_detector(detector_type='point_transformer', class_id=class_id, model_id=model_id, hyper_param=hyper_param)






