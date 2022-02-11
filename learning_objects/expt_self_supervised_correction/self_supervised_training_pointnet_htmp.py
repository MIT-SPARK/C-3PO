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
from learning_objects.models.keypoint_detector import RSNetKeypoints
from learning_objects.models.certifiability import confidence, confidence_kp
from learning_objects.utils.general import display_results

SAVE_LOCATION = '../../data/learning_objects/expt_registration/runs/'


# loss functions
def keypoints_loss(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(kp, kp_)


def rotation_loss(R, R_):

    device_ = R.device

    err_mat = R @ R_.transpose(-1, -2) - torch.eye(3, device=device_)
    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(err_mat, torch.zeros_like(err_mat))


def translation_loss(t, t_):
    """
    t   : torch.tensor of shape (B, 3, N)
    t_  : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(t, t_)


def chamfer_loss(pc, pc_, pc_padding=None, max_loss=False):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
    max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    output:
    loss    : (B, 1)
        returns max_loss if max_loss is true
    """

    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)
        # pc_padding = torch.zeros(batch_size, n).to(device=device_)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)

    if max_loss:
        loss = sq_dist.max(dim=1)[0]
    else:
        loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)


def contrastive_loss(heatmap, labels, certi=None, epsilon=0.4):
    """
    inputs:
    heatmap : torch.tensor of shape (B, m)  : values between 0:1
    labels  : torch.tensor of shape (B)     : index from 0:m
    certi   : torch.tensor of shape (B,)    : dtype.bool: indicating if a datapoint is certifiable (True) or not (False)
    epsilon : gap between true and false

    """
    device_ = heatmap.device
    if certi==None:
        certi=torch.ones_like(labels).to(dtype=torch.bool).to(device=device_)

    if certi.sum() == 0:
        print("NO DATAPOINT CERTIFIABLE")
        certi = torch.ones_like(labels).to(dtype=torch.bool).to(device=device_)

    b = torch.tensor([heatmap[i, labels[i]] for i in range(heatmap.shape[0])])       #ToDo: find an efficient way of doing this.
    b = b.unsqueeze(-1).to(device=device_)
    contrast = b - epsilon - heatmap

    # another way:
    # contrast = torch.zeros_like(heatmap)
    # contrast[i, j] = heatmap[i, labels[i]] - epsilon - heatmap[i, j]

    contrast = contrast.mean(dim=1)
    contrast = contrast * certi
    loss = contrast.sum()/certi.sum()

    return -loss


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



def self_supervised_loss(input_point_cloud, predicted_point_cloud,
                         corrected_keypoints, predicted_model_keypoints, heatmap):
    """
    inputs:
    input_point_cloud       : torch.tensor of shape (B, 3, m)
    predicted_point_cloud   : torch.tensor of shape (B, 3, n)
    corrected_keypoints     : torch.tensor of shape (B, 3, N)
    heatmap                 : torch.tensor of shape (B, N, m)

    outputs:
    loss    : torch.tensor of shape (1,)

    """
    theta = 25.0
    device_ = heatmap.device

    # certification
    certi = certify(input_point_cloud=input_point_cloud,
                    predicted_point_cloud=predicted_point_cloud,
                    corrected_keypoints=corrected_keypoints,
                    predicted_model_keypoints=predicted_model_keypoints)
    certi = certi.squeeze(-1)   # (B,)

    if certi.sum() == 0:
        print("NO DATA POINT CERTIFIABLE IN THIS BATCH")
        pc_loss, ce_loss, contra_loss, fra_certi = 0.0, 0.0, 0.0, 0.0
    else:

        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi/certi.shape[0]    # not to be used for training

        # pc loss
        pc_loss = chamfer_loss(pc=input_point_cloud, pc_=predicted_point_cloud)         # Using normal chamfer loss here, as the max chamfer is used in certification
        pc_loss = pc_loss * certi
        pc_loss = pc_loss.sum()/num_certi

        # ce and contra loss
        _, kp_idx, _ = ops.knn_points(p1=corrected_keypoints.transpose(-1, -2),
                                      p2=input_point_cloud.transpose(-1, -2),
                                      K=1, return_nn=True, return_sorted=False)

        kp_idx = kp_idx  # (B, N, 1)    : index in range 0:m
        kp_idx = kp_idx.squeeze(-1)     # (B, N)

        B, N, m = heatmap.shape
        certi_long = torch.kron(certi, torch.ones(N).to(device=device_)).to(device=device_).to(dtype=torch.bool)
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        ce_loss = cross_entropy_loss(heatmap.reshape(B * N, m), kp_idx.reshape(B * N).long())
        ce_loss = ce_loss * certi_long
        ce_loss = ce_loss.sum()/certi_long.sum()
        contra_loss = contrastive_loss(heatmap.reshape(B * N, m), kp_idx.reshape(B * N).long(), certi=certi_long)

    return pc_loss, ce_loss, contra_loss, fra_certi



def validation_loss(input, output):
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

    pc_loss = chamfer_loss(pc=input[0], pc_=output[0], max_loss=True)
    pc_loss = pc_loss.mean()

    lossMSE = torch.nn.MSELoss()
    kp_loss = lossMSE(input[1], output[1]).mean()

    R_loss = rotation_loss(input[2], output[2]).mean()      #ToDo: use utils.general.rotation_err. It computes angles.
    t_loss = translation_loss(input[3], output[3]).mean()

    # return pc_loss + kp_loss + R_loss + t_loss  #ToDo: change the validation loss. We can't have the ground truth.
    return pc_loss


# Training code
def self_supervised_train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, correction_flag):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    fra_cert_ = torch.zeros(len(training_loader))
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        # print("Running batch ", i+1, "/", len(training_loader))
        input_point_cloud, _, _, _ = data
        input_point_cloud = input_point_cloud.to(device)
        fra_cert_ = fra_cert_.to(device=device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        predicted_point_cloud, corrected_keypoints, _, _, correction, heatmap, predicted_model_keypoints \
            = model(input_point_cloud, correction_flag=True, need_predicted_keypoints=True)

        # Compute the loss and its gradients
        pc_loss, ce_loss, contra_loss, fra_cert = \
            self_supervised_loss(input_point_cloud=input_point_cloud,
                                 predicted_point_cloud=predicted_point_cloud,
                                 corrected_keypoints=corrected_keypoints,
                                 predicted_model_keypoints=predicted_model_keypoints,
                                 heatmap=heatmap)

        loss = pc_loss + 25*(ce_loss + 0.001*contra_loss)         # first 20 epochs: place high weight on kp loss
        # loss = 25*pc_loss + ce_loss + 0.001 * contra_loss         # next 20 epochs: place high weight on pc loss

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1 == 0:
            print("Batch ", (i+1), " loss: ", loss.item(),
                  " pc loss: ", pc_loss.item(),
                  " ce loss: ", ce_loss.item(),
                  " contra loss(-): ", -contra_loss.item())
            print("Batch ", (i+1), " fra cert: ", fra_cert.item())

        fra_cert_[i] = fra_cert
        del input_point_cloud, predicted_point_cloud, correction, heatmap, predicted_model_keypoints
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss, fra_cert_


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
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, _ = model(input_point_cloud,
                                                                                            correction_flag=True)

            vloss = validation_loss(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                   output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))
            # vloss = self_supervised_loss(input_point_cloud=input_point_cloud,
            #                         predicted_point_cloud=predicted_point_cloud,
            #                         keypoint_correction=correction)
            running_vloss += vloss

            # if i<3:
            #     # Display results for validation
            #     pc = predicted_point_cloud.clone().detach().to('cpu')
            #     pc_t = input_point_cloud.clone().detach().to('cpu')
            #     kp = predicted_keypoints.clone().detach().to('cpu')
            #     kp_t = keypoints_target.clone().detach().to('cpu')
            #     display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc_t,
            #                     target_keypoints=kp_t)
            #     del pc, pc_t, kp, kp_t

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss

# Training + Validation Loop
def train_without_supervision(self_supervised_train_loader, validation_loader, model, optimizer, correction_flag):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 40

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH :', epoch_number + 1, "TIME: ", datetime.now())

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on real data with self-supervision: ")
        ave_loss_self_supervised, fra_cert = self_supervised_train_one_epoch(epoch_number,
                                                                             writer, self_supervised_train_loader,
                                                                             model, optimizer, correction_flag=True)

        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(writer, validation_loader, model, correction_flag=True)

        print('LOSS self-supervised train {}, valid {}'.format(ave_loss_self_supervised, avg_vloss))
        print("fra certi: ", fra_cert)
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
            best_model_path = SAVE_LOCATION + 'best_self_supervised_keypoint_detect_pointnet_htmp2.pth'
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
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, _ = model(input_point_cloud,
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


# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.



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
    momentum_sgd = 0.99
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
    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedHeatmapModel as ProposedModel
    # Note: set use_pretrained_regression_model to True if we're using models pretrained on se3 dataset with supervision
    # if keypoint_detector=RSNetKeypoints, it will automatically load the pretrained model inside the detector,
    # so set use_pretrained_regression_model=False
    # This difference is because RSNetKeypoints was trained with supervision in KeypointNet,
    # whereas RegressionKeypoints was trained with supervision in this script and saved ProposedModel weights
    # model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for supervised training of point_net or point-transformer
    #                       keypoint_detector=None, use_pretrained_regression_model=False).to(device)
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for self-supervised training of a pre-trained point_net or point-transformer
                          keypoint_detector='pointnet', use_pretrained_regression_model=True).to(device)
    # model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for self-supervised training of a pre-trained RSNetKeypoints model
    #                       keypoint_detector='', use_pretrained_regression_model=True).to(device)

    if model.use_pretrained_regression_model:
        print("USING PRETRAINED REGRESSION MODEL, ONLY USE THIS WITH SELF-SUPERVISION")
        # best_model_checkpoint = os.path.join(SAVE_LOCATION,
        #                                      '_best_supervised_keypoint_detect_pointnet_htmp_se3_Wcontraloss.pth')
        best_model_checkpoint = os.path.join(SAVE_LOCATION, 'best_self_supervised_keypoint_detect_pointnet_htmp.pth')
        # best_model_checkpoint = os.path.join(SAVE_LOCATION, 'best_self_supervised_keypoint_detect_pointnet_htmp2.pth')
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
    # train_with_supervision(supervised_training_loader=supervised_train_loader,                                          # Use this for supervised training
    #                        validation_loader=supervised_train_loader,
    #                        model=model,
    #                        optimizer=optimizer,
    #                        correction_flag=False)
    train_without_supervision(self_supervised_train_loader=self_supervised_train_loader,                                # Use this for self-supervised training
                              validation_loader=val_loader,
                              model=model,
                              optimizer=optimizer,
                              correction_flag=True)

    # test
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
    loader = torch.utils.data.DataLoader(self_supervised_train_dataset,
                                                               batch_size=dataset_batch_size,
                                                               shuffle=False)

    # visual_test(test_loader=supervised_train_loader, model=model, correction_flag=False)
    visual_test(test_loader=loader, model=model, correction_flag=False)
    visual_test(test_loader=loader, model=model, correction_flag=True)




