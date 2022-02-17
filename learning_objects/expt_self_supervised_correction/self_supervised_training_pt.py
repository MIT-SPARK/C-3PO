"""
This code implements supervised and self-supervised training, and validation, for keypoint detector with registration.
It uses registration during supervised training. It uses registration plus corrector during self-supervised training.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch3d import ops

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, DepthPC, CLASS_NAME
from learning_objects.models.certifiability import confidence, confidence_kp

from learning_objects.utils.general import display_results

SAVE_LOCATION = '../../data/learning_objects/expt_registration/runs/'



# loss functions
from learning_objects.expt_self_supervised_correction.loss_functions import chamfer_loss


def certify(input_point_cloud, predicted_point_cloud, corrected_keypoints, predicted_model_keypoints, epsilon=0.99):
    """
    inputs:
    input_point_cloud           : torch.tensor of shape (B, 3, m)
    predicted_point_cloud       : torch.tensor of shape (B, 3, n)
    corrected_keypoints         : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    certificate     : torch.tensor of shape (B, 1)  : dtype = torch.bool

    """

    confidence_ = confidence(input_point_cloud, predicted_point_cloud)
    confidence_kp_ = confidence_kp(corrected_keypoints, predicted_model_keypoints)

    return (confidence_ >= epsilon) & (confidence_kp_ >= epsilon)


def self_supervised_loss(input_point_cloud, predicted_point_cloud, keypoint_correction, certi):
    """
    inputs:
    input_point_cloud       : torch.tensor of shape (B, 3, m)
    predicted_point_cloud   : torch.tensor of shape (B, 3, n)
    keypoint_correction     : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    loss    : torch.tensor of shape (1,)

    """
    theta = 25.0
    device_ = input_point_cloud.device

    if certi.sum() == 0:
        print("NO DATA POINT CERTIFIABLE IN THIS BATCH")
        pc_loss, kp_loss, fra_certi = 0.0, 0.0, 0.0

    else:
        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi / certi.shape[0]  # not to be used for training

        # pc loss
        pc_loss = chamfer_loss(pc=input_point_cloud,
                               pc_=predicted_point_cloud)  # Using normal chamfer loss here, as the max chamfer is used in certification
        pc_loss = pc_loss * certi
        pc_loss = pc_loss.sum() / num_certi

        lossMSE = torch.nn.MSELoss(reduction='none')
        if keypoint_correction is None:
            kp_loss = torch.zeros(pc_loss.shape)
        else:
            kp_loss = lossMSE(keypoint_correction, torch.zeros_like(keypoint_correction))
            kp_loss = kp_loss.sum(dim=1).mean(dim=1)    # (B,)
            kp_loss = kp_loss * certi
            kp_loss = kp_loss.mean()

    # return pc_loss + theta*kp_loss, pc_loss, kp_loss, fra_certi   # for the first 20 epochs
    return 25*pc_loss + kp_loss, pc_loss, kp_loss, fra_certi        # for the next 20 epochs

# validation loss:
def validation_loss(input, output, certi=None):
    """
    inputs:
        input   : tuple of length 4 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
        output  : tuple of length 4 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    if certi == None:
        pc_loss = chamfer_loss(pc=input[0], pc_=output[0])
        vloss = pc_loss.mean()
    else:
        vloss = certi

    return vloss


# evaluation metrics:
def translation_error(t, t_):
    """
    inputs:
    t: torch.tensor of shape (3, 1) or (B, 3, 1)
    t_: torch.tensor of shape (3, 1) or (B, 3, 1)

    output:
    t_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if t.dim() == 2:
        return torch.norm(t - t_, p=2)/3.0
    elif t.dim() == 3:
        return torch.norm(t-t_, p=2, dim=1)/3.0
    else:
        return ValueError


def rotation_error(R, R_):
    """
    inputs:
    R: torch.tensor of shape (3, 3) or (B, 3, 3)
    R_: torch.tensor of shape (3, 3) or (B, 3, 3)

    output:
    R_err: torch.tensor of shape (1, 1) or (B, 1)
    """

    if R.dim() == 2:
        return torch.arccos(0.5*(torch.trace(R.T @ R)-1))
        # return transforms.matrix_to_euler_angles(torch.matmul(R.T, R_), "XYZ").abs().sum()/3.0
        # return torch.abs(0.5*(torch.trace(R.T @ R_) - 1).unsqueeze(-1))
        # return 1 - 0.5*(torch.trace(R.T @ R_) - 1).unsqueeze(-1)
        # return torch.norm(R.T @ R_ - torch.eye(3, device=R.device), p='fro')
    elif R.dim() == 3:
        # return transforms.matrix_to_euler_angles(torch.transpose(R, 1, 2) @ R_, "XYZ").abs().mean(1).unsqueeze(1)
        return torch.acos(0.5*(torch.einsum('bii->b', torch.transpose(R, -1, -2) @ R_) - 1).unsqueeze(-1))
        # return 1 - 0.5 * (torch.einsum('bii->b', torch.transpose(R, 1, 2) @ R_) - 1).unsqueeze(-1)
        # return torch.norm(R.transpose(-1, -2) @ R_ - torch.eye(3, device=R.device), p='fro', dim=[1, 2])
    else:
        return ValueError

def keypoints_error(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='none')

    return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)


def evaluation_error(input, output):
    """
    inputs:
        input   : tuple of length 4 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
        output  : tuple of length 4 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    pc_loss = chamfer_loss(pc=input[0], pc_=output[0])
    pc_err = pc_loss

    kp_err = keypoints_error(input[1], output[1])

    R_err = rotation_error(input[2], output[2])
    t_err = translation_error(input[3], output[3])

    print("pc_err shape: ", pc_err.shape)
    print("kp_err shape: ", kp_err.shape)
    print("R_err shape: ", R_err.shape)
    print("t_err shape: ", t_err.shape)

    return pc_err, kp_err, R_err, t_err
    # return pc_loss


# Training code
def self_supervised_train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, correction_flag):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # print("Running batch ", i+1, "/", len(training_loader))
        input_point_cloud, _, _, _ = data
        input_point_cloud = input_point_cloud.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, corrected_keypoints, _, _, correction, predicted_model_keypoints = \
            model(input_point_cloud, correction_flag=True, need_predicted_keypoints=True)

        # Certification
        # certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=corrected_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints)
        certi = certi.squeeze(-1)  # (B,)

        # Compute the loss and its gradients
        loss, pc_loss, kp_loss, fra_cert = \
            self_supervised_loss(input_point_cloud=input_point_cloud,
                                 predicted_point_cloud=predicted_point_cloud,
                                 keypoint_correction=correction, certi=certi)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1 == 0:
            print("Batch ", (i+1), " loss: ", loss.item(), " pc loss: ", pc_loss.item(), " kp loss: ", kp_loss.item())
            print("Batch ", (i + 1), " fra cert: ", fra_cert.item())

        del input_point_cloud, predicted_point_cloud, correction
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Validation code
def validate(writer, validation_loader, model, correction_flag):

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
                                                                                            correction_flag=True, need_predicted_keypoints=True)

            # certification
            certi = certify(input_point_cloud=input_point_cloud,
                            predicted_point_cloud=predicted_point_cloud,
                            corrected_keypoints=corrected_keypoints,
                            predicted_model_keypoints=predicted_model_keypoints)

            vloss = validation_loss(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                   output=(predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted), certi=certi)

            running_vloss += vloss

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, corrected_keypoints, R_predicted, t_predicted, certi

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss

# Training + Validation Loop
def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer, correction_flag):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 10

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH :', epoch_number + 1, "TIME: ", datetime.now())

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on real data with self-supervision: ")
        ave_loss_self_supervised = self_supervised_train_one_epoch(epoch_number, writer, self_supervised_train_loader,
                                                                model, optimizer, correction_flag=True)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(writer, validation_loader, model, correction_flag=True)

        print('LOSS self-supervised train {}, valid {}'.format(ave_loss_self_supervised, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training (self-supervised)': ave_loss_self_supervised,
                            'Validation': avg_vloss},
                           epoch_number + 1)
        #ToDo: Have the certified fraction as well
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = SAVE_LOCATION + 'expt_keypoint_detect_' + 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            best_model_path = SAVE_LOCATION + 'best_self_supervised_keypoint_detect_pt.pth'
            torch.save(model.state_dict(), best_model_path)

        epoch_number += 1

        torch.cuda.empty_cache()


    return None


# Test the keypoint detector with PACE. See if you can learn the keypoints.
def visual_test(test_loader, model, correction_flag=False):

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


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.


def evaluate(eval_loader, model, certification=True):

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
                                predicted_model_keypoints=predicted_model_keypoints)

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

    print('-' * 20)
    print("Running expt_keypoint_detect.py")
    print('-' * 20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-' * 20)
    torch.cuda.empty_cache()


    # dataset

    # Given ShapeNet class_id, model_id, this generates a dataset and a dataset loader with
    # various transformations of the object point cloud.
    #
    #
    class_id = "03001627"  # chair
    class_name = CLASS_NAME[class_id]
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    dataset_dir = '../../data/learning_objects/'
    # correction_flag = True

    # optimization parameters
    lr_sgd = 0.02
    momentum_sgd = 0.9
    lr_adam = 0.001
    num_of_points = 500

    # sim dataset:
    # supervised_train_dataset_len = 10000
    # supervised_train_batch_size = 5#0
    # num_of_points_supervised = 500

    # real dataset:
    self_supervised_train_dataset_len = 500
    self_supervised_train_batch_size = 50 #can increase to make it faster
    num_of_points_to_sample = 1000#0
    num_of_points_selfsupervised = 2048

    # # supervised and self-supervised training data
    # supervised_train_dataset = SE3PointCloud(class_id=class_id,
    #                                          model_id=model_id,
    #                                          num_of_points=num_of_points_supervised,
    #                                          dataset_len=supervised_train_dataset_len)
    # supervised_train_loader = torch.utils.data.DataLoader(supervised_train_dataset,
    #                                                       batch_size=supervised_train_batch_size,
    #                                                       shuffle=False)

    self_supervised_train_dataset = DepthPC(class_id=class_id,
                                            model_id=model_id,
                                            n=num_of_points_selfsupervised,
                                            num_of_points_to_sample=num_of_points_to_sample,
                                            dataset_len=self_supervised_train_dataset_len)
    # self_supervised_train_dataset = DepthPointCloud2(class_id=class_id,
    #                                         model_id=model_id,
    #                                         num_of_points=num_of_points_selfsupervised,
    #                                         dataset_len=self_supervised_train_dataset_len)
    self_supervised_train_loader = torch.utils.data.DataLoader(self_supervised_train_dataset,
                                                               batch_size=self_supervised_train_batch_size,
                                                               shuffle=False)

    # validation dataset:
    val_dataset_len = 50
    val_batch_size = 50
    val_dataset = DepthPC(class_id=class_id, model_id=model_id,
                                            n=num_of_points_selfsupervised,
                                            num_of_points_to_sample=num_of_points_to_sample,
                                            dataset_len=val_dataset_len)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)



    # Generate a shape category, CAD model objects, etc.
    cad_models = self_supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = self_supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel
    # Note: set use_pretrained_regression_model to True if we're using models pretrained on se3 dataset with supervision
    # if keypoint_detector=RSNetKeypoints, it will automatically load the pretrained model inside the detector,
    # so set use_pretrained_regression_model=False
    # This difference is because RSNetKeypoints was trained with supervision in KeypointNet,
    # whereas RegressionKeypoints was trained with supervision in this script and saved ProposedModel weights
    # model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for supervised training of point_net or point-transformer
    #                       keypoint_detector=None, use_pretrained_regression_model=False).to(device)
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for self-supervised training of a pre-trained point_net or point-transformer
                          keypoint_detector='point_transformer', use_pretrained_regression_model=True).to(device)
    # model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for self-supervised training of a pre-trained RSNetKeypoints model
    #                       keypoint_detector='', use_pretrained_regression_model=True).to(device)

    if model.use_pretrained_regression_model:
        print("USING PRETRAINED REGRESSION MODEL, ONLY USE THIS WITH SELF-SUPERVISION")
        best_model_checkpoint = os.path.join(SAVE_LOCATION, '_best_supervised_keypoint_detect_pt_se3.pth')
        # best_model_checkpoint = os.path.join(SAVE_LOCATION, 'best_self_supervised_keypoint_detect_pt.pth')
        if not os.path.isfile(best_model_checkpoint):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
        state_dict = torch.load(best_model_checkpoint)
        model.load_state_dict(state_dict)
        model.train()
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)


    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)

    # training
    train_without_supervision(self_supervised_train_loader=self_supervised_train_loader,                                # Use this for self-supervised training
                              validation_loader=val_loader,
                              model=model,
                              optimizer=optimizer,
                              correction_flag=True)

    del optimizer, self_supervised_train_dataset, self_supervised_train_loader, val_dataset, val_loader



    # Evaluation
    # validation dataset:
    eval_dataset_len = 50
    eval_batch_size = 50
    eval_dataset = DepthPC(class_id=class_id, model_id=model_id,
                          n=num_of_points_selfsupervised,
                          num_of_points_to_sample=num_of_points_to_sample,
                          dataset_len=eval_dataset_len)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    evaluate(eval_loader=eval_loader, model=model, certification=True)



    # # Visual Test
    print("Visualizing the trained model.")
    dataset_len = 20
    dataset_batch_size = 1
    dataset = DepthPC(class_id=class_id,
                                            model_id=model_id,
                                            n=num_of_points_selfsupervised,
                                            num_of_points_to_sample=num_of_points_to_sample,
                                            dataset_len=dataset_len)
    # self_supervised_train_dataset = DepthPointCloud2(class_id=class_id,
    #                                         model_id=model_id,
    #                                         num_of_points=num_of_points_selfsupervised,
    #                                         dataset_len=self_supervised_train_dataset_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False)

    # visual_test(test_loader=supervised_train_loader, model=model, correction_flag=False)
    visual_test(test_loader=loader, model=model, correction_flag=False)
    visual_test(test_loader=loader, model=model, correction_flag=True)




