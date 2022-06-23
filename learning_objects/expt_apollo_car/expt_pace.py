"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with pace.
It uses pace during supervised training. It uses pace plus corrector during self-supervised training.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch3d import ops

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import sys
sys.path.append("../../")


from learning_objects.utils.visualization_utils import display_results

from learning_objects.utils.ddn.node import AbstractDeclarativeNode, ParamDeclarativeFunction
from learning_objects.models.keypoint_detector import RegressionKeypoints
from learning_objects.models.pace import PACEmodule
from learning_objects.models.keypoint_corrector import kp_corrector_pace
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.utils.loss_functions import translation_loss, rotation_loss, shape_loss, chamfer_loss

from learning_objects.datasets.keypointnet import SE3nIsotropicShapePointCloud, DepthIsoPC


SAVE_LOCATION = '../../data/learning_objects/expt_pace/runs/'


# Proposed Model
class ProposedModel(nn.Module):
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """
    def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None, use_pretrained_model=False):
        super().__init__()
        """ 
        model_keypoints     : torch.tensor of shape (K, 3, N)
        cad_models          : torch.tensor of shape (K, 3, n)  
        keypoint_detector   : torch.nn.Module   : detects N keypoints for any sized point cloud input       
                                                  should take input : torch.tensor of shape (B, 3, m)
                                                  should output     : torch.tensor of shape (B, 3, N)
                                                  
        """

        # Parameters
        self.class_name = class_name
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.use_pretrained_model = use_pretrained_model

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # Keypoint Detector
        if keypoint_detector==None:
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='point_transformer',
                                                         dim=[3, 16, 32, 64, 128])
        else:
            self.keypoint_detector = keypoint_detector(class_name=class_name, N=self.N)

        # PACE
        self.pace = PACEmodule(model_keypoints=self.model_keypoints)

        # Registration + Correction
        corrector_node = kp_corrector_pace(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        self.corrector = ParamDeclarativeFunction(problem=corrector_node)

        # Model generator
        self.modelgen = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)



    def forward(self, input_point_cloud, correction_flag=False):
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
        device_ = input_point_cloud.device
        detected_keypoints = self.keypoint_detector(input_point_cloud)

        if not correction_flag:
            # NOTE: This uses pace as a function. The backprop will happen through the iterations of the algorithm.
            R, t, c = self.pace(detected_keypoints)

            _, predicted_point_cloud = self.modelgen.forward(c)
            predicted_point_cloud = R @ predicted_point_cloud + t

            return predicted_point_cloud, detected_keypoints, R, t, c, None

        else:
            #NOTE: This uses keypoint corrector as a function. The backprop will happen through the iterations of the algorithm.
            correction = self.corrector.forward(detected_keypoints=detected_keypoints,
                                                input_point_cloud=input_point_cloud)

            corrected_keypoints = detected_keypoints + correction

            R, t, c = self.pace(corrected_keypoints)
            _, predicted_point_cloud = self.modelgen.forward(c)
            predicted_point_cloud = R @ predicted_point_cloud + t

            return predicted_point_cloud, corrected_keypoints, R, t, c, correction



# loss functions

def self_supervised_loss(input_point_cloud, predicted_point_cloud, keypoint_correction):
    """
    inputs:
    input_point_cloud       : torch.tensor of shape (B, 3, m)
    predicted_point_cloud   : torch.tensor of shape (B, 3, n)
    keypoint_correction     : torch.tensor of shape (B, 3, N)

    outputs:
    loss    : torch.tensor of shape (1, )

    """
    theta = 25.0

    pc_loss = chamfer_loss(input_point_cloud, predicted_point_cloud, max_loss=False)
    pc_loss = pc_loss.mean()

    lossMSE = torch.nn.MSELoss()
    kp_loss = lossMSE(keypoint_correction, torch.zeros_like(keypoint_correction)).mean()

    return pc_loss + theta*kp_loss


def supervised_loss(input, output):
    """
    inputs:
        input   : tuple of length 4 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
                                      input[4]  : torch.tensor of shape (B, K, 1) : shape_true
        output  : tuple of length 4 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation
                                      output[4]  : torch.tensor of shape (B, K, 1) : shape

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    pc_loss = chamfer_loss(input[0], output[0], max_loss=False)
    pc_loss = pc_loss.mean()

    lossMSE = torch.nn.MSELoss()
    kp_loss = lossMSE(input[1], output[1]).mean()

    R_loss = rotation_loss(input[2], output[2]).mean()
    t_loss = translation_loss(input[3], output[3]).mean()
    c_loss = shape_loss(input[4], output[4]).mean()

    return 10*pc_loss + 50*kp_loss + R_loss + t_loss + c_loss


def validation_loss(input, output):
    """
    inputs:
        input   : tuple of length 5 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
                                      input[4]  : torch.tensor of shape (B, K, 1) : shape_true
        output  : tuple of length 5 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation
                                      output[4]  : torch.tensor of shape (B, K, 1) : shape

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    pc_loss = chamfer_loss(input[0], output[0], max_loss=False)
    pc_loss = pc_loss.mean()

    lossMSE = torch.nn.MSELoss()
    kp_loss = lossMSE(input[1], output[1]).mean()

    R_loss = rotation_loss(input[2], output[2]).mean()  #ToDo: use utils.general.rotation_err. It computes angles.
    t_loss = translation_loss(input[3], output[3]).mean()
    c_loss = shape_loss(input[4], output[4]).mean()

    return pc_loss + kp_loss + R_loss + t_loss + c_loss




# Training code
def self_supervised_train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        print("Running batch: ", i, "/", len(training_loader))

        input_point_cloud, _, _, _, _ = data
        input_point_cloud = input_point_cloud.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, _, _, _, _, correction = model(input_point_cloud, correction_flag=True)

        # Compute the loss and its gradients
        loss = self_supervised_loss(input_point_cloud=input_point_cloud,
                                    predicted_point_cloud=predicted_point_cloud,
                                    keypoint_correction=correction)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (i+1) % 500 == 0:
            last_loss = running_loss / 500  # loss per batch
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

        del input_point_cloud, predicted_point_cloud, correction
        torch.cuda.empty_cache()

    return last_loss


def supervised_train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        print("Running batch: ", i)

        input_point_cloud, keypoints_target, R_target, t_target, c_target = data
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)
        c_target = c_target.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted, _ = model(input_point_cloud,
                                                                                     correction_flag=False)

        # Compute the loss and its gradients
        loss = supervised_loss(input=(input_point_cloud, keypoints_target, R_target, t_target, c_target),
                               output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (i+1) % 500 == 0:
            last_loss = running_loss / 500 # loss per batch
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

        del input_point_cloud, keypoints_target, R_target, t_target, c_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted
        torch.cuda.empty_cache()

    return last_loss


# Validation code
def validate(writer, validation_loader, model):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target, c_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)
            c_target = c_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted, _ \
                = model(input_point_cloud, correction_flag=True)

            vloss = validation_loss(input=(input_point_cloud, keypoints_target,
                                           R_target, t_target, c_target),
                                   output=(predicted_point_cloud, predicted_keypoints,
                                           R_predicted, t_predicted, c_predicted))
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

            del input_point_cloud, keypoints_target, R_target, t_target, c_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def train_with_supervision(supervised_training_loader, self_supervised_train_loader, validation_loader, model,
                           optimizer):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on simulated data with supervision:")
        avg_loss_supervised = supervised_train_one_epoch(epoch_number, writer, supervised_training_loader, model,
                                                         optimizer)
        print("Training on real data with self-supervision:")
        ave_loss_self_supervised = self_supervised_train_one_epoch(epoch_number, writer, self_supervised_train_loader,
                                                                model, optimizer)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(writer, validation_loader, model)

        print('LOSS supervised-train {}, self-supervised train {}, valid {}'.format(avg_loss_supervised,
                                                                                    ave_loss_self_supervised,
                                                                                    avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training (supervised)': avg_loss_supervised,
                            'Training (self-supervised)': ave_loss_self_supervised,
                            'Validation': avg_vloss},
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


def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on real data with self-supervision: ")
        ave_loss_self_supervised = self_supervised_train_one_epoch(epoch_number, writer, self_supervised_train_loader,
                                                                model, optimizer)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(writer, validation_loader, model)

        print('LOSS self-supervised train {}, valid {}'.format(ave_loss_self_supervised, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training (self-supervised)': ave_loss_self_supervised,
                            'Validation': avg_vloss},
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
def visual_test(test_loader, model):

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target, c_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted, _ = \
            model(input_point_cloud, correction_flag=False)

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_p = predicted_point_cloud.clone().detach().to('cpu')
        kp = keypoints_target.clone().detach().to('cpu')
        kp_p = predicted_keypoints.clone().detach().to('cpu')
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc_p,
                        target_keypoints=kp)

        del pc, pc_p, kp, kp_p
        del input_point_cloud, keypoints_target, R_target, t_target, c_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, c_predicted

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
    #
    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    dataset_dir = '../../data/learning_objects/'
    supervised_train_dataset_len = 1200
    supervised_train_batch_size = 1
    self_supervised_train_dataset_len = 100
    self_supervised_train_batch_size = 1
    lr_sgd = 0.02
    momentum_sgd = 0.9
    lr_adam = 0.001
    n = 500
    num_of_points = 500

    # supervised and self-supervised training data
    supervised_train_dataset = SE3nIsotropicShapePointCloud(class_id=class_id,
                                                            model_id=model_id,
                                                            num_of_points=num_of_points,
                                                            dataset_len=supervised_train_dataset_len)
    supervised_train_loader = torch.utils.data.DataLoader(supervised_train_dataset,
                                                          batch_size=supervised_train_batch_size,
                                                          shuffle=False)
    self_supervised_train_dataset = DepthIsoPC(class_id=class_id,model_id=model_id,
                                               n=n,
                                               num_of_points_to_sample=num_of_points,
                                               dataset_len=self_supervised_train_dataset_len)
    self_supervised_train_loader = torch.utils.data.DataLoader(self_supervised_train_dataset,
                                                               batch_size=self_supervised_train_batch_size,
                                                               shuffle=False)


    # Generate a shape category, CAD model objects, etc.
    cad_models = supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    model = ProposedModel(model_keypoints=model_keypoints, cad_models=cad_models).to(device)
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)


    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)


    # training
    train_with_supervision(self_supervised_train_loader=self_supervised_train_loader,
                           supervised_training_loader=supervised_train_loader,
                           validation_loader=self_supervised_train_loader,
                           model=model,
                           optimizer=optimizer)
    # train_without_supervision(self_supervised_train_loader=self_supervised_train_loader,
    #                           validation_loader=self_supervised_train_loader,
    #                           model=model,
    #                           optimizer=optimizer)

    # test
    print("Visualizing the trained model.")
    visual_test(test_loader=self_supervised_train_loader, model=model)



