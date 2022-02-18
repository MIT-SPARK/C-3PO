"""
This code implements supervised training of keypoint detector in simulation.

It can use registration during supervised training.

"""

import torch
import pickle
import yaml
from pytorch3d import ops
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME
from learning_objects.utils.general import display_results

# SAVE_LOCATION = '../../data/learning_objects/expt_registration/runs/'

# loss functions
from learning_objects.expt_self_supervised_correction.loss_functions import \
    keypoints_loss, rotation_loss, translation_loss, chamfer_loss

from learning_objects.expt_self_supervised_correction.loss_functions import supervised_training_loss as supervised_loss
from learning_objects.expt_self_supervised_correction.loss_functions import supervised_validation_loss as validation_loss

from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel
from learning_objects.utils.general import TrackingMeter


# Training code
def supervised_train_one_epoch(training_loader, model, optimizer, correction_flag, device):

    running_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        pc, kp, R, t = data
        pc = pc.to(device)
        kp = kp.to(device)
        R = R.to(device)
        t = t.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # print("Test:: pc_shape: ", pc.shape)
        # print(pc[0, ...])
        out = model(pc, correction_flag=correction_flag)
        kp_pred = out[1]

        loss = ((kp - kp_pred)**2).sum(dim=1).mean(dim=1).mean()
        # loss = keypoints_loss(kp, kp_pred)
        # loss = supervised_loss(kp, kp_pred)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()         # Note: the output of supervised_loss is already averaged over batch_size
        if i % 10 == 0:
            print("Batch ", (i+1), " loss: ", loss.item())

        del pc, kp, R, t, kp_pred
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Validation code
def validate(validation_loader, model, correction_flag, device):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud,
                                                                                            correction_flag=correction_flag)

            # vloss = validation_loss(input=(input_point_cloud, keypoints_target, R_target, t_target),
            #                        output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))
            # vloss = keypoints_loss(keypoints_target, predicted_keypoints)
            vloss = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).mean(dim=1).mean()
            running_vloss += vloss

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def train_with_supervision(supervised_training_loader, validation_loader, model, optimizer, correction_flag,
                           best_model_save_file, device, hyper_param):

    num_epochs = hyper_param['num_epochs']
    best_vloss = 1_000_000.

    train_loss = TrackingMeter()
    val_loss = TrackingMeter()
    epoch_number = 0
    
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on simulated data with supervision:")
        avg_loss_supervised = supervised_train_one_epoch(supervised_training_loader, model,
                                                         optimizer, correction_flag=correction_flag, device=device)
        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(validation_loader, model, correction_flag=correction_flag, device=device)

        print('LOSS supervised-train {}, valid {}'.format(avg_loss_supervised, avg_vloss))
        train_loss.add_item(avg_loss_supervised)
        val_loss.add_item(avg_vloss)

        # Saving the model with the best vloss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), best_model_save_file)

        epoch_number += 1

        torch.cuda.empty_cache()

    return train_loss, val_loss


def train_detector(hyper_param, detector_type='pointnet', class_id="03001627",
                   model_id="1e3fba4500d20bb49b9f2eb77f5e247e"):
    """

    """

    print('-' * 20)
    print("Running supervised_training: ", datetime.now())
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    torch.cuda.empty_cache()

    # shapenet
    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    best_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_train_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_val_loss_' + detector_type + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    momentum_sgd = hyper_param['momentum_sgd']

    # simulated training data
    train_dataset_len = hyper_param['train_dataset_len']
    train_batch_size = hyper_param['train_batch_size']
    train_num_of_points = hyper_param['train_num_of_points']

    supervised_train_dataset = SE3PointCloud(class_id=class_id,
                                             model_id=model_id,
                                             num_of_points=train_num_of_points,
                                             dataset_len=train_dataset_len)
    supervised_train_loader = torch.utils.data.DataLoader(supervised_train_dataset,
                                                          batch_size=train_batch_size,
                                                          shuffle=False)

    # simulated validation dataset:
    val_dataset_len = hyper_param['val_dataset_len']
    val_batch_size = hyper_param['val_batch_size']
    val_num_of_points = hyper_param['val_num_of_points']

    val_dataset = SE3PointCloud(class_id=class_id,
                                model_id=model_id,
                                num_of_points=val_num_of_points,
                                dataset_len=val_dataset_len)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=val_batch_size,
                                                          shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, use_pretrained_regression_model=False).to(device)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)

    # training
    train_loss, val_loss = train_with_supervision(supervised_training_loader=supervised_train_loader,
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

    del supervised_train_dataset, supervised_train_loader, val_dataset, val_loader, cad_models, model_keypoints, \
        optimizer, model

    return None


def visual_test(test_loader, model, correction_flag=False, device=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud,
                                                                                        correction_flag=correction_flag)
        model.train()
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


def visualize_detector(hyper_param,
                       detector_type='pointnet',
                       class_id="03001627",
                       model_id="1e3fba4500d20bb49b9f2eb77f5e247e"):
    """

    """

    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    torch.cuda.empty_cache()

    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    best_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'

    # Test 1:
    dataset = SE3PointCloud(class_id=class_id,
                            model_id=model_id,
                            num_of_points=hyper_param['train_num_of_points'],
                            dataset_len=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = dataset._get_model_keypoints().to(torch.float).to(device=device)



    # model
    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel

    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, use_pretrained_regression_model=False).to(device)

    if not os.path.isfile(best_model_save_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
    state_dict = torch.load(best_model_save_file)
    model.load_state_dict(state_dict)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    #
    print("Visualizing the trained model.")
    visual_test(test_loader=loader, model=model, correction_flag=False, device=device)

    # # Test 2:
    # dataset = SE3PointCloud(class_id=class_id,
    #                         model_id=model_id,
    #                         num_of_points=200,
    #                         dataset_len=10)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # visual_test(test_loader=loader, model=model, correction_flag=False, device=device)
    #
    # dataset = SE3PointCloud(class_id=class_id,
    #                         model_id=model_id,
    #                         num_of_points=100,
    #                         dataset_len=10)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # visual_test(test_loader=loader, model=model, correction_flag=False, device=device)

    # Test 3: testing on real dataset
    real_dataset = DepthPC(class_id=class_id, model_id=model_id, n=2000, num_of_points_to_sample=1000,
                           dataset_len=10)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=1, shuffle=False)
    visual_test(test_loader=real_loader, model=model, correction_flag=False, device=device)
    visual_test(test_loader=real_loader, model=model, correction_flag=True, device=device)


if __name__ == "__main__":

    class_id = "03001627"  # chair
    class_name = CLASS_NAME[class_id]
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    stream = open("supervised_training.yml", "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)

    train_detector(detector_type='pointnet', class_id=class_id, model_id=model_id, hyper_param=hyper_param)
    visualize_detector(detector_type='pointnet', class_id=class_id, model_id=model_id, hyper_param=hyper_param)

    train_detector(detector_type='point_transformer', class_id=class_id, model_id=model_id, hyper_param=hyper_param)
    visualize_detector(detector_type='point_transformer', class_id=class_id, model_id=model_id, hyper_param=hyper_param)

