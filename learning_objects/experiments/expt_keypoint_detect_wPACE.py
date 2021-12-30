"""
This code is to use PACE for training a keypoint detector in a self-supervised manner.

We are given a CAD model with keypoints.
4.
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
from learning_objects.utils.general import chamfer_half_distance, keypoint_error
from learning_objects.utils.general import rotation_error, shape_error, translation_error
from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.pace_ddn import PACEbp, PACEddn
from learning_objects.models.pace import PACE, PACEmodule
from learning_objects.models.modelgen import ModelFromShape, ModelFromShapeModule
from learning_objects.models.keypoint_corrector import PACEwKeypointCorrection, from_y


dataset_dir = '../data/'

# Given a CAD model with keypoints, write a dataset and a dataset loader with various transformations of the point cloud
# Variations: rotations, translations, shape (isotropic scaling), point density
def get_dataset():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST(dataset_dir, train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST(dataset_dir, train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))


    return training_loader, validation_loader, classes


# Generate a shape category, CAD model objects, etc.



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

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        self.keypoint_method = 'pointnet'

        self.weights = weights
        if weights == None:
            self.weights = torch.ones(self.N, 1)

        self.lambda_constant = lambda_constant
        if lambda_constant == None:
            self.lambda_constant = torch.sqrt(torch.tensor([self.N/self.K]))


        # Keypoint Detector
        self.keypoint_detector = RegressionKeypoints(k=self.N, method=self.keypoint_method)


        # PACE
        self.pace = PACE(weights=self.weights, model_keypoints=self.model_keypoints,
                         lambda_constant=self.lambda_constant)


        # Model Generator
        self.generate_model = ModelFromShapeModule(cad_models=self.cad_models, model_keypoints=self.model_keypoints)


    def forward(self, input_point_cloud):
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
        R, t, c = self.pace(detected_keypoints)
        target_keypoints, target_point_cloud = ModelFromShapeModule(c)
        target_point_cloud = R @ target_point_cloud + t


        return detected_keypoints, target_keypoints, target_point_cloud





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
    theta = 5.0

    pc_loss = chamfer_half_distance(input_point_cloud, target_point_cloud)
    kp_loss = keypoint_error(detected_keypoints, target_keypoints)

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
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



def train(training_loader, model, optimizer, loss_fn):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_loader, model, optimizer, loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
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
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    return None




# Test the keypoint detector with PACE. See if you can learn the keypoints.



# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.



if __name__ == "__main__":

    # dataset
    training_loader, validation_loader, classes = get_dataset()     #ToDo: to write

    # model
    model = ProposedModel()

    # loss function
    loss_fn = loss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # training
    train(training_loader, model, optimizer, loss_fn)




# Note: If this works, do it with occlusions at input.