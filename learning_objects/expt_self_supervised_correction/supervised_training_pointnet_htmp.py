"""
This code implements supervised training and validation for keypoint detector with registration.

It can use registration during supervised training.

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

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, \
    DepthPC, CLASS_NAME, SE3PointCloudHtmpKP


from learning_objects.models.keypoint_detector import RSNetKeypoints
# from learning_objects.expt_self_supervised_correction.proposed_model import ProposedModel

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


def chamfer_loss(pc, pc_, pc_padding=None):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n) #input pointcloud
    pc_ : torch.tensor of shape (B, 3, m) #predicted pointcloud
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in

    output:
    loss    : torch.tensor of shape (B, 1)

    """

    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)
        # pc_padding = torch.zeros(batch_size, n).to(device=device_)
    #pc may have a ton of points at 0,0,0
    #for every point in the input point cloud pc, we want the closest point in the predicted pc_
    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False)

    # dist (B, n, 1): distance from point in X to the nearest point in Y
    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)
    loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)


def supervised_loss(input, output):
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
    pc_loss = pc_loss.mean()

    # lossMSE = torch.nn.MSELoss()
    # kp_loss = lossMSE(input[1], output[1]).mean()
    kp_loss = (input[1] - output[1])**2
    kp_loss = kp_loss.sum(dim=1).sum(dim=1).mean()

    R_loss = rotation_loss(input[2], output[2]).mean()
    t_loss = translation_loss(input[3], output[3]).mean()

    return pc_loss + kp_loss + R_loss + t_loss
    # return kp_loss


def contrastive_loss(heatmap, labels, epsilon=0.4):
    """
    inputs:
    heatmap : torch.tensor of shape (B, m)  : values between 0:1
    labels  : torch.tensor of shape (B)     : index from 0:m
    epsilon : gap between true and false

    """
    device_ = heatmap.device
    # epsilon = epsilon.to(device=device_)

    b = torch.tensor([heatmap[i, labels[i]] for i in range(heatmap.shape[0])])       #ToDo: find an efficient way of doing this.
    b = b.unsqueeze(-1).to(device=device_)
    contrast = b - epsilon - heatmap

    # another way:
    # contrast = torch.zeros_like(heatmap)
    # contrast[i, j] = heatmap[i, labels[i]] - epsilon - heatmap[i, j]


    contrast = contrast
    loss = contrast.mean()

    return -loss


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

    pc_loss = chamfer_loss(pc=input[0], pc_=output[0])
    pc_loss = pc_loss.mean()

    # lossMSE = torch.nn.MSELoss()
    # kp_loss = lossMSE(input[1], output[1]).mean()           #ToDo: These losses may not be correct.
    kp_loss = (input[1] - output[1]) ** 2
    kp_loss = kp_loss.sum(dim=1).mean(dim=1).mean()

    R_loss = rotation_loss(input[2], output[2]).mean()      #ToDo: use utils.general.rotation_err. It computes angles.
    t_loss = translation_loss(input[3], output[3]).mean()

    # return pc_loss + kp_loss + R_loss + t_loss
    return kp_loss


# Training code
def supervised_train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, correction_flag):
    running_loss = 0.
    last_loss = 0.

    cross_entropy_loss = nn.CrossEntropyLoss()

    for i, data in enumerate(training_loader):
        # # Every data instance is an input + label pair
        # input_point_cloud, keypoints_target, R_target, t_target = data
        # input_point_cloud = input_point_cloud.to(device)
        # keypoints_target = keypoints_target.to(device)
        # R_target = R_target.to(device)
        # t_target = t_target.to(device)
        #
        # # Zero your gradients for every batch!
        # optimizer.zero_grad()
        #
        # # Make predictions for this batch
        # input_point_cloud.requires_grad = True
        # predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud,
        #                                                                              correction_flag=correction_flag)   #ToDo: Set correction flag to False! We don't use it in supervised training.
        #
        # # Compute the loss and its gradients                                                                            #ToDo: Add another
        # loss = supervised_loss(input=(input_point_cloud, keypoints_target, R_target, t_target),
        #                        output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))
        # loss.backward()

        # Every data instance is an input + label pair
        pc, kp, R, t, kp_idx = data     # kp_idx (B, N, 1)
        pc = pc.to(device)
        kp = kp.to(device)
        R = R.to(device)
        t = t.to(device)
        kp_idx = kp_idx.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        out = model(pc, correction_flag=False)
        # kp_pred = out[1]
        heatmap = out[5]    # (B, N, m)

        B, N, m = heatmap.shape
        loss1 = cross_entropy_loss(heatmap.reshape(B * N, m), kp_idx.reshape(B * N).long())
        loss2 = contrastive_loss(heatmap.reshape(B * N, m), kp_idx.reshape(B * N).long())
        # loss = loss1
        loss = loss1 + 0.001*loss2
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()         # Note: the output of supervised_loss is already averaged over batch_size
        if i % 10 == 0:
            print("Batch ", (i+1), " loss: ", loss.item(), " CE loss: ", loss1.item(), " contra. loss(-): ", -loss2.item())

        del pc, kp, R, t, kp_idx, heatmap
        torch.cuda.empty_cache()

    ave_tloss = running_loss / (i + 1)

    return ave_tloss


# Validation code
def validate(writer, validation_loader, model, correction_flag):

    with torch.no_grad():

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            input_point_cloud, keypoints_target, R_target, t_target, _ = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, _ = model(input_point_cloud,
                                                                                            correction_flag=correction_flag)

            vloss = validation_loss(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                   output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))
            # vloss = loss_fn(input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints)
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


def train_with_supervision(supervised_training_loader, validation_loader, model, optimizer, correction_flag):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 10

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print("Training on simulated data with supervision:")
        avg_loss_supervised = supervised_train_one_epoch(epoch_number, writer, supervised_training_loader, model,
                                                         optimizer, correction_flag=correction_flag)
        # Validation. We don't need gradients on to do reporting.
        model.train(False)
        print("Validation on real data: ")
        avg_vloss = validate(writer, validation_loader, model, correction_flag=correction_flag)

        print('LOSS supervised-train {}, valid {}'.format(avg_loss_supervised, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training (supervised)': avg_loss_supervised,
                            'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = SAVE_LOCATION + 'expt_keypoint_detect_' + 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            best_model_path = SAVE_LOCATION + '_best_supervised_keypoint_detect_pointnet_htmp_se3_Wcontraloss.pth'
            torch.save(model.state_dict(), best_model_path)

        epoch_number += 1

        torch.cuda.empty_cache()

    return None



# Test the keypoint detector with PACE. See if you can learn the keypoints.
def visual_test(test_loader, model, correction_flag=False):

    for i, vdata in enumerate(test_loader):

        input_point_cloud = vdata[0]
        keypoints_target = vdata[1]
        R_target = vdata[2]
        t_target = vdata[3]
        # input_point_cloud, keypoints_target, R_target, t_target = vdata
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
    print("Running supervised_training for expt_keypoint_detect")
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

    # optimization parameters
    lr_sgd = 0.02
    momentum_sgd = 0.9
    lr_adam = 0.001
    num_of_points = 500

    # sim dataset:
    supervised_train_dataset_len = 50000
    supervised_train_batch_size = 100
    num_of_points_supervised = 1000

    # supervised and self-supervised training data
    supervised_train_dataset = SE3PointCloudHtmpKP(class_id=class_id,
                                                   model_id=model_id,
                                                   num_of_points=num_of_points_supervised,
                                                   dataset_len=supervised_train_dataset_len)
    # supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=supervised_train_dataset_len)
    supervised_train_loader = torch.utils.data.DataLoader(supervised_train_dataset,
                                                          batch_size=supervised_train_batch_size,
                                                          shuffle=False)

    # sim validation dataset:
    val_dataset_len = 1000
    val_batch_size = 50

    # supervised and self-supervised training data
    val_dataset = SE3PointCloudHtmpKP(class_id=class_id,
                                      model_id=model_id,
                                      num_of_points=num_of_points_supervised,
                                      dataset_len=val_dataset_len)
    # val_dataset = DepthPC(class_id=class_id, model_id=model_id, n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=val_dataset_len)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=val_batch_size,
                                                          shuffle=False)


    # Generate a shape category, CAD model objects, etc.
    cad_models = supervised_train_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = supervised_train_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedHeatmapModel as ProposedModel
    # Note: set use_pretrained_regression_model to True if we're using models pretrained on se3 dataset with supervision
    # if keypoint_detector=RSNetKeypoints, it will automatically load the pretrained model inside the detector,
    # so set use_pretrained_regression_model=False
    # This difference is because RSNetKeypoints was trained with supervision in KeypointNet,
    # whereas RegressionKeypoints was trained with supervision in this script and saved ProposedModel weights
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for supervised training of pointnet
                          keypoint_detector='pointnet', use_pretrained_regression_model=False).to(device=device)
    # model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                  # Use this for visualizing a pre-trained pointnet
    #                       keypoint_detector=None, use_pretrained_regression_model=True).to(device)

    if model.use_pretrained_regression_model:
        print("USING PRETRAINED REGRESSION MODEL, ONLY USE THIS WITH SELF-SUPERVISION")
        # best_model_checkpoint = os.path.join(SAVE_LOCATION, '_best_supervised_keypoint_detect_pointnet_htmp_se3.pth')
        best_model_checkpoint = os.path.join(SAVE_LOCATION, '_best_supervised_keypoint_detect_pointnet_htmp_se3_Wcontraloss.pth')
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
    train_with_supervision(supervised_training_loader=supervised_train_loader,                                          # Use this for supervised training
                           validation_loader=val_loader,
                           model=model,
                           optimizer=optimizer,
                           correction_flag=False)

    del optimizer, supervised_train_dataset, supervised_train_loader, val_dataset, val_loader

    # test
    print("Visualizing the trained model.")
    # visual_test(supervised_train_loader, model, correction_flag=False)

    dataset_len = 20
    print("Visual Test: SE3 Point Cloud (#pts: same as training)")
    dataset = SE3PointCloudHtmpKP(class_id=class_id,
                                             model_id=model_id,
                                             num_of_points=num_of_points_supervised,
                                             dataset_len=dataset_len)
    # supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=supervised_train_dataset_len)
    loader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=1,
                                                          shuffle=False)
    visual_test(test_loader=loader, model=model, correction_flag=False)

    print("Visual Test: SE3 Point Cloud (200)")
    dataset = SE3PointCloudHtmpKP(class_id=class_id,
                            model_id=model_id,
                            num_of_points=200,
                            dataset_len=supervised_train_dataset_len)
    # supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=supervised_train_dataset_len)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False)
    visual_test(test_loader=loader, model=model, correction_flag=False)

    print("Visual Test: SE3 Point Cloud (100)")
    dataset = SE3PointCloudHtmpKP(class_id=class_id,
                            model_id=model_id,
                            num_of_points=100,
                            dataset_len=supervised_train_dataset_len)
    # supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=supervised_train_dataset_len)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False)
    visual_test(test_loader=loader, model=model, correction_flag=False)



    # testing on real dataset
    print("Visual Test: Depth Point Cloud (1000)")
    real_batch_size = 1
    real_dataset = DepthPC(class_id=class_id, model_id=model_id, n=2000, num_of_points_to_sample=1000,
                           dataset_len=val_dataset_len)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=real_batch_size, shuffle=False)
    print(".... without correction")
    visual_test(test_loader=real_loader, model=model, correction_flag=False)
    print(".... with correction")
    visual_test(test_loader=real_loader, model=model, correction_flag=True)

