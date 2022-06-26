"""
This code implements supervised training of keypoint detector in simulation.

It can use registration during supervised training.

"""

import argparse
import os
import pickle
import sys
import torch
import yaml
from datetime import datetime
from pytorch3d import ops
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../")

from learning_objects.datasets.ycb import SE3PointCloudYCB, SE3PointCloudYCBAugment, DepthYCB
from learning_objects.utils.visualization_utils import display_results

# loss functions
from learning_objects.utils.loss_functions import supervised_training_loss as supervised_loss, \
    avg_kpt_distance_regularizer, supervised_validation_loss as validation_loss

from learning_objects.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel
from learning_objects.utils.general import TrackingMeter


# Training code
def supervised_train_one_epoch(training_loader, model, optimizer, device):

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
        out = model(pc)
        kp_pred = out[1]

        kp_reg = avg_kpt_distance_regularizer(kp_pred)
        loss = 2 * (((kp - kp_pred)**2)).sum(dim=1).mean(dim=1).mean() - .1*kp_reg
        # old_loss functions
        # loss = (((kp - kp_pred) ** 2)).sum(dim=1).mean(dim=1).mean()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()         # Note: the output of supervised_loss is already averaged over batch_size
        if i % 10 == 0:
            print("Batch ", (i+1), " loss: ", loss.item())

        del pc, kp, R, t, kp_pred, out
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)
    
    return ave_tloss


# Validation code
def validate(validation_loader, model, device):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud)

            vloss = ((keypoints_target - predicted_keypoints) ** 2).sum(dim=1).mean(dim=1).mean()
            running_vloss += vloss

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def train_with_supervision(supervised_training_loader, validation_loader, model, optimizer,
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
                                                         optimizer, device=device)
        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(validation_loader, model, device=device)

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


def train_detector(hyper_param, detector_type='pointnet', model_id="019_pitcher_base", use_corrector=False):
    """

    """

    print('-' * 20)
    print("Running supervised_training: ", datetime.now())
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

    best_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'
    train_loss_save_file = best_model_save_location + '_train_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_val_loss_' + detector_type + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    momentum_sgd = hyper_param['momentum_sgd']

    # simulated training data
    train_dataset_len = hyper_param['train_dataset_len']
    train_batch_size = hyper_param['train_batch_size']
    train_num_of_points = hyper_param['train_num_of_points']

    supervised_train_dataset = SE3PointCloudYCBAugment(model_id=model_id,
                                             num_of_points=train_num_of_points,
                                             dataset_len=train_dataset_len)
    supervised_train_loader = torch.utils.data.DataLoader(supervised_train_dataset,
                                                          batch_size=train_batch_size,
                                                          shuffle=False)

    # simulated validation dataset:
    val_dataset_len = hyper_param['val_dataset_len']
    val_batch_size = hyper_param['val_batch_size']
    val_num_of_points = hyper_param['val_num_of_points']

    val_dataset = SE3PointCloudYCB(model_id=model_id,
                                num_of_points=val_num_of_points,
                                dataset_len=val_dataset_len)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=val_batch_size,
                                                          shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector).to(device)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd, weight_decay=0.0001)

    # training
    train_loss, val_loss = train_with_supervision(supervised_training_loader=supervised_train_loader,
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

    del supervised_train_dataset, supervised_train_loader, val_dataset, val_loader, cad_models, model_keypoints, \
        optimizer, num_parameters, model.keypoint_detector, model

    return None


def visual_test(test_loader, model, device=None):

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
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud)
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

        if i >= 5:
            break


def visualize_detector(hyper_param,
                       detector_type='pointnet',
                       model_id="019_pitcher_base",
                       use_corrector=False):
    """

    """

    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    torch.cuda.empty_cache()

    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'

    # Test 1:
    dataset = SE3PointCloudYCB(model_id=model_id,
                            num_of_points=hyper_param['train_num_of_points'],
                            dataset_len=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = dataset._get_model_keypoints().to(torch.float).to(device=device)



    # model
    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector).to(device)

    if not os.path.isfile(best_model_save_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
    state_dict = torch.load(best_model_save_file)
    model.load_state_dict(state_dict)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    #
    print("Visualizing the trained model.")
    visual_test(test_loader=loader, model=model, device=device)

    # Test 2: testing on real dataset
    real_dataset = DepthYCB(model_id=model_id, split='test', num_of_points=1000)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=1, shuffle=False)
    visual_test(test_loader=real_loader, model=model, device=device)
    del num_parameters, model.keypoint_detector, model


if __name__ == "__main__":
    """
    Change dataset used in visualize_detector function code to visualize trained model results on simulated or real data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("model_id", help="specify the ycb model id.", type=str)

    args = parser.parse_args()

    print("KP detector type: ", args.detector_type)
    print("CAD Model class: ", args.model_id)
    detector_type = args.detector_type
    model_id = args.model_id

    stream = open("supervised_training.yml", "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)

    visualize_detector(detector_type=detector_type, model_id=model_id, hyper_param=hyper_param, use_corrector=False)

