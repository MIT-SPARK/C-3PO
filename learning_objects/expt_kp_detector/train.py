
import torch
import torch.nn as nn
import torch
import yaml
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

import sys
sys.path.append("../../")

from learning_objects.models.keypoint_detector import RegressionKeypoints
from learning_objects.datasets.catkeypointnet import NUM_KEYPOINTS, CategoryKeypointNetDataset
from learning_objects.utils.general import display_results, TrackingMeter


# Proposed Kp Detector Model
class keyPointDetector(nn.Module):

    def __init__(self, class_name, device, regression_model=None):
        super().__init__()
        """ 
        model_keypoints     : torch.tensor of shape (K, 3, N)
        cad_models          : torch.tensor of shape (K, 3, n) 
        regression_model    : 'pointnet' or 'point_transformer'

        """

        # Parameters
        self.class_name = class_name
        self.N = int(NUM_KEYPOINTS[self.class_name])
        self.device_ = device

        # Keypoint Detector
        if regression_model == None:
            self.regression_model = RegressionKeypoints(N=self.N, method='pointnet').to(device=self.device_)

        elif regression_model == 'pointnet':
            self.regression_model = RegressionKeypoints(N=self.N, method='pointnet').to(device=self.device_)

        elif regression_model == 'point_transformer':
            self.regression_model = RegressionKeypoints(N=self.N, method='point_transformer').to(device=self.device_)

        else:
            raise NotImplementedError

    def forward(self, input_point_cloud):
        """
        point_cloud : torch.tensor of shape (B, 3, m)

        output:
        rotation        : torch.tensor of shape (B, 3, 3)
        translation     : torch.tensor of shape (B, 3, 1)
        predicted_pc    :   torch.tensor of shape (B, 3, n)

        """

        batch_size, _, m = input_point_cloud.shape
        # device_ = input_point_cloud.device

        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        kp = self.regression_model(pc_centered)

        return kp + center


############
# Train
def train_one_epoch(training_loader, model, optimizer, device, hyper_param):

    running_loss = 0.
    lossMSE = torch.nn.MSELoss(reduction='none')

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # print("Running batch ", i+1, "/", len(training_loader))
        input_point_cloud, keypoints_gt, _, _ = data
        input_point_cloud = input_point_cloud.to(device)
        keypoints_gt = keypoints_gt.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print("pc shape: ", input_point_cloud.shape)
        keypoints_pred = model(input_point_cloud)

        # Compute the loss and its gradients
        loss = lossMSE(keypoints_gt, keypoints_pred)
        loss = loss.sum(dim=1).mean(dim=1)  # (B,)
        loss = loss.mean()

        # Backprop
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1 == 0:
            print("Batch ", (i + 1), " loss: ", loss.item())

        del input_point_cloud, keypoints_gt, keypoints_pred
        # torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Val
def validate(validation_loader, model, device, hyper_param):

    lossMSE = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_gt, _, _ = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_gt = keypoints_gt.to(device)

            # Make predictions for this batch
            keypoints_pred = model(input_point_cloud)

            # Validation loss
            vloss = lossMSE(keypoints_gt, keypoints_pred)
            vloss = vloss.sum(dim=1).mean(dim=1)  # (B,)
            vloss = vloss.mean()

            running_vloss += vloss

            del input_point_cloud, keypoints_gt, keypoints_pred

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


# Train + Val Loop
def train(train_loader, validation_loader, model, optimizer, best_model_save_file, device, hyper_param,
          train_loss_save_file, val_loss_save_file):

    num_epochs = hyper_param['num_epochs']
    best_vloss = 1_000_000.

    train_loss = TrackingMeter()
    val_loss = TrackingMeter()
    epoch_number = 0

    for epoch in range(num_epochs):
        print('EPOCH :', epoch_number + 1, "TIME: ", datetime.now())

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training: ")
        ave_loss_self_supervised = train_one_epoch(train_loader, model, optimizer, device=device,
                                                   hyper_param=hyper_param)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation: ")
        avg_vloss = validate(validation_loader, model, device=device, hyper_param=hyper_param)

        print('LOSS train {}, val {}'.format(ave_loss_self_supervised, avg_vloss))
        train_loss.add_item(ave_loss_self_supervised)
        val_loss.add_item(avg_vloss)

        # Saving the model with the best vloss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), best_model_save_file)

        epoch_number += 1

        with open(train_loss_save_file, 'wb') as outp:
            pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

        with open(val_loss_save_file, 'wb') as outp:
            pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

        # torch.cuda.empty_cache()

    return train_loss, val_loss


# Train
def train_detector(hyper_param, detector_type='pointnet', class_name='airplane'):
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
    best_model_save_location = save_folder + class_name + '/'
    if not os.path.exists(best_model_save_location):
        os.makedirs(best_model_save_location)

    best_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    train_loss_save_file = best_model_save_location + '_sstrain_loss_' + detector_type + '.pkl'
    val_loss_save_file = best_model_save_location + '_ssval_loss_' + detector_type + '.pkl'

    # optimization parameters
    lr_sgd = hyper_param['lr_sgd']
    momentum_sgd = hyper_param['momentum_sgd']

    # train dataset
    train_dataset = CategoryKeypointNetDataset(class_name=class_name, dataset_len=hyper_param['train_dataset_len'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyper_param['train_batch_size'], shuffle=True)

    # validation dataset:
    val_dataset = CategoryKeypointNetDataset(class_name=class_name, dataset_len=hyper_param['val_dataset_len'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyper_param['val_batch_size'], shuffle=True)

    # model
    model = keyPointDetector(class_name=class_name, regression_model=detector_type, device=device)
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)

    # training
    train_loss, val_loss = train(train_loader=train_loader,
                                 validation_loader=val_loader,
                                 model=model,
                                 optimizer=optimizer,
                                 best_model_save_file=best_model_save_file,
                                 device=device,
                                 hyper_param=hyper_param,
                                 train_loss_save_file=train_loss_save_file,
                                 val_loss_save_file=val_loss_save_file)

    with open(train_loss_save_file, 'wb') as outp:
        pickle.dump(train_loss, outp, pickle.HIGHEST_PROTOCOL)

    with open(val_loss_save_file, 'wb') as outp:
        pickle.dump(val_loss, outp, pickle.HIGHEST_PROTOCOL)

    return None


# Wrapper
def train_kp_detectors(detector_type, only_categories=None):

    for class_name in only_categories:

        hyper_param_file = "./hyperparam.yml"
        stream = open(hyper_param_file, "r")
        hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
        hyper_param = hyper_param[detector_type]

        print(">>"*40)
        print("Training: ", class_name)
        train_detector(class_name=class_name,
                       detector_type=detector_type,
                       hyper_param=hyper_param)


if __name__ == "__main__":

    """
    usage: 
    >> python train.py "point_transformer" "chair"
    >> python train.py "pointnet" "chair"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)

    args = parser.parse_args()

    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.class_name)
    detector_type = args.detector_type
    class_name = args.class_name
    only_categories = [class_name]

    train_kp_detectors(detector_type=detector_type, only_categories=only_categories)