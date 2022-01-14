"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with pace.
It uses pace during supervised training. It uses pace plus corrector during self-supervised training.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import sys
sys.path.append("../../")


from learning_objects.utils.ddn.node import ParamDeclarativeFunction
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import chamfer_half_distance, keypoint_error, soft_chamfer_half_distance
from learning_objects.utils.general import rotation_error, shape_error, translation_error
from learning_objects.utils.general import display_results

from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.pace_ddn import PACEbp, PACEddn
from learning_objects.models.pace import PACE, PACEmodule
from learning_objects.models.modelgen import ModelFromShape, ModelFromShapeModule

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud


SAVE_LOCATION = '../../data/learning_objects/expt_pace/runs/'


# # Given ShapeNet class_id, model_id, this generates a dataset and a dataset loader with
# # various transformations of the object point cloud.
# #
# # Variations: point density, SE3 transformations, and isotropic scaling
# #
# class_id = "03001627"  # chair
# model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
# dataset_dir = '../../data/learning_objects/'
# dataset_len = 10000
# batch_size = 4
#
#
# se3_dataset100 = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=100, dataset_len=dataset_len)
# # se3_dataset500 = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=dataset_len)
# # se3_dataset1k = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=1000, dataset_len=dataset_len)
# # se3_dataset10k = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=10000, dataset_len=dataset_len)
#
#
# se3_dataset100_loader = torch.utils.data.DataLoader(se3_dataset100, batch_size=batch_size, shuffle=False)
# # se3_dataset500_loader = torch.utils.data.DataLoader(se3_dataset500, batch_size=batch_size, shuffle=False)
# # se3_dataset1k_loader = torch.utils.data.DataLoader(se3_dataset1k, batch_size=batch_size, shuffle=False)
# # se3_dataset10k_loader = torch.utils.data.DataLoader(se3_dataset10k, batch_size=batch_size, shuffle=False)
#
# # Generate a shape category, CAD model objects, etc.



# Proposed Model
class ProposedModel(nn.Module):
    def __init__(self, model_keypoints, cad_models, weights=None, lambda_constant=torch.tensor([1.0])):
        super().__init__()
        """
        keypoint_type   : 'heatmap' or 'regression'
        keypoint_method : 'pointnet' or 'point_transformer'
        model_keypoints : torch.tensor of shape (K, 3, N)
        cad_models      : torch.tensor of shape (K, 3, n)
        weights         : torch.tensor of shape (N, 1) or None
        lambda_constant : torch.tensor of shape (1, 1) or None
        keypoint_correction     : True or False
        """

        # Parameters
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # self.keypoint_method = 'pointnet'
        self.keypoint_method = 'point_transformer'

        self.weights = weights
        if weights == None:
            self.weights = torch.ones((self.N, 1), device=self.device_)

        self.lambda_constant = lambda_constant.to(device=self.device_)
        if lambda_constant == None:
            self.lambda_constant = torch.sqrt(torch.tensor([self.N/self.K])).to(device=self.device_)


        # Keypoint Detector
        self.keypoint_detector = RegressionKeypoints(N=self.N, method=self.keypoint_method, dim=[3, 16, 32, 64, 128])


        # PACE
        self.pace = PACEmodule(weights=self.weights, model_keypoints=self.model_keypoints,
                               lambda_constant=self.lambda_constant).to(device=self.device_)


        # Model Generator
        self.generate_model = ModelFromShapeModule(cad_models=self.cad_models,
                                                   model_keypoints=self.model_keypoints).to(device=self.device_)


    def forward(self, input_point_cloud, pre_train=False):
        """
        input:
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        where
        B = batch size
        m = number of points in each point cloud

        output:
        keypoints           : torch.tensor of shape (B, 3, self.N)
        target_point_cloud  : torch.tensor of shape (B, 3, n)
        # rotation          : torch.tensor of shape (B, 3, 3)
        # translation       : torch.tensor of shape (B, 3, 1)
        # shape             : torch.tensor of shape (B, self.K, 1)
        """

        # detect keypoints
        detected_keypoints = self.keypoint_detector(input_point_cloud)
        if not pre_train:
            R, t, c = self.pace(detected_keypoints)
            target_keypoints, target_point_cloud = self.generate_model(shape=c)
            target_point_cloud = R @ target_point_cloud + t

            return detected_keypoints, target_keypoints, target_point_cloud, R, t, c
        else:
            return detected_keypoints


# loss function
def loss(input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints):
    """
    inputs:
    input_point_cloud   : torch.tensor of shape (B, 3, m)
    target_point_cloud  : torch.tensor of shape (B, 3, n)
    detected_keypoints  : torch.tensor of shape (B, 3, N)
    target_keypoints    : torch.tensor of shape (B, 3, N)

    output:
    loss                : torch.tensor of shape (B, 1)
    """
    theta = 25.0

    # pc_loss = soft_chamfer_half_distance(input_point_cloud, target_point_cloud, radius=1000.0)
    pc_loss = chamfer_half_distance(input_point_cloud, target_point_cloud)
    pc_loss = pc_loss.mean()

    # kp_loss = keypoint_error(detected_keypoints, target_keypoints)
    lossMSE = torch.nn.MSELoss()
    kp_loss = lossMSE(detected_keypoints, target_keypoints)
    # print(detected_keypoints)
    # print(target_keypoints)
    # print("PC loss: ", pc_loss.mean())
    # print("KP loss: ", kp_loss.mean())

    return pc_loss + theta*kp_loss


# Train the keypoint detector with PACE.
def train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        input_point_cloud, R_target, t_target = data
        input_point_cloud = input_point_cloud.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        detected_keypoints = model(input_point_cloud, pre_train=True)
        batch_size = detected_keypoints.shape[0]
        target_keypoints = model.model_keypoints.repeat(batch_size, 1, 1)
        target_point_cloud = model.cad_models.repeat(batch_size, 1, 1)

        target_point_cloud = R_target @ target_point_cloud + t_target
        target_keypoints = R_target @ target_keypoints + t_target

        # Compute the loss and its gradients
        loss = loss_fn(input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 500 == 0:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

            # # Disply results after each 1000 training iterations
            # pc = input_point_cloud.clone().detach().to('cpu')
            # pc_t = target_point_cloud.clone().detach().to('cpu')
            # kp = detected_keypoints.clone().detach().to('cpu')
            # kp_t = target_keypoints.clone().detach().to('cpu')
            # display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc_t, target_keypoints=kp_t)
            #
            # del pc, pc_t, kp, kp_t

        del input_point_cloud, R_target, t_target, detected_keypoints, target_point_cloud, target_keypoints

        torch.cuda.empty_cache()

    return last_loss


# Validation code
def validate(writer, validation_loader, model, loss_fn):
    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            # detected_keypoints, target_keypoints, target_point_cloud, _, _, _ = model(input_point_cloud)
            detected_keypoints = model(input_point_cloud, pre_train=True)
            batch_size = detected_keypoints.shape[0]
            target_keypoints = model.model_keypoints.repeat(batch_size, 1, 1)
            target_point_cloud = model.cad_models.repeat(batch_size, 1, 1)

            target_point_cloud = R_target @ target_point_cloud + t_target
            target_keypoints = R_target @ target_keypoints + t_target

            vloss = loss_fn(input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints)
            running_vloss += vloss

            # if i==0:
            #     # Display results for validation
            #     pc = input_point_cloud.clone().detach().to('cpu')
            #     pc_t = target_point_cloud.clone().detach().to('cpu')
            #     kp = detected_keypoints.clone().detach().to('cpu')
            #     kp_t = target_keypoints.clone().detach().to('cpu')
            #     display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc_t,
            #                     target_keypoints=kp_t)
            #     del pc, pc_t, kp, kp_t

            del input_point_cloud, R_target, t_target, detected_keypoints, target_point_cloud, target_keypoints

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss




def train(training_loader, validation_loader, model, optimizer, loss_fn):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_loader, model, optimizer, loss_fn)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        avg_vloss = validate(writer, validation_loader, model, loss_fn)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = SAVE_LOCATION + 'expt_keypoint_detect_' + 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

        torch.cuda.empty_cache()


    return None


# Test the keypoint detector with PACE. See if you can learn the keypoints.
def visual_test(test_loader, model, loss_fn):

    for i, vdata in enumerate(test_loader):
        input_point_cloud, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        # detected_keypoints, target_keypoints, target_point_cloud, _, _, _ = model(input_point_cloud)
        detected_keypoints = model(input_point_cloud, pre_train=True)
        batch_size = detected_keypoints.shape[0]
        target_keypoints = model.model_keypoints.repeat(batch_size, 1, 1)
        target_point_cloud = model.cad_models.repeat(batch_size, 1, 1)

        target_point_cloud = R_target @ target_point_cloud + t_target
        target_keypoints = R_target @ target_keypoints + t_target

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_t = target_point_cloud.clone().detach().to('cpu')
        kp = detected_keypoints.clone().detach().to('cpu')
        kp_t = target_keypoints.clone().detach().to('cpu')
        display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc_t,
                        target_keypoints=kp_t)
        del pc, pc_t, kp, kp_t
        del input_point_cloud, R_target, t_target, detected_keypoints, target_point_cloud, target_keypoints

        if i >= 4:
            break


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.



if __name__ == "__main__":

    print('-' * 20)
    print("Running expt_keypoint_detect.py")
    print('-' * 20)
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)
    torch.cuda.empty_cache()


    # dataset

    # Given ShapeNet class_id, model_id, this generates a dataset and a dataset loader with
    # various transformations of the object point cloud.
    #
    # Variations: point density, SE3 transformations, and isotropic scaling
    #
    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    dataset_dir = '../../data/learning_objects/'
    dataset_len = 12000
    batch_size = 120
    lr_sgd = 0.02
    momentum_sgd = 0.9
    lr_adam = 0.001
    num_of_points = 500

    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=num_of_points,
                                dataset_len=dataset_len)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=batch_size, shuffle=False)


    # Generate a shape category, CAD model objects, etc.
    cad_models = se3_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = se3_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    model = ProposedModel(model_keypoints=model_keypoints, cad_models=cad_models).to(device)
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # loss function
    loss_fn = loss

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)


    # training
    train(training_loader=se3_dataset_loader, validation_loader=se3_dataset_loader,
          model=model, optimizer=optimizer, loss_fn=loss_fn)


    # test
    print("Visualizing the trained model.")
    visual_test(test_loader=se3_dataset_loader, model=model, loss_fn=loss_fn)



