"""
This code implements supervised training and validation for keypoint detector with registration.

It can use registration during supervised training.

"""

import torch
from pytorch3d import ops
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME
from learning_objects.utils.general import display_results

SAVE_LOCATION = '../../data/learning_objects/expt_registration/runs/'




# loss functions
from learning_objects.expt_self_supervised_correction.loss_functions import \
    keypoints_loss, rotation_loss, translation_loss, chamfer_loss

from learning_objects.expt_self_supervised_correction.loss_functions import supervised_training_loss as supervised_loss
from learning_objects.expt_self_supervised_correction.loss_functions import supervised_validation_loss as validation_loss


# Training code
def supervised_train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, correction_flag):
    running_loss = 0.
    last_loss = 0.

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
        pc, kp, R, t = data
        pc = pc.to(device)
        kp = kp.to(device)
        R = R.to(device)
        t = t.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        out = model(pc, correction_flag=False)
        kp_pred = out[1]

        # loss1 = ((kp - kp_pred)**2).sum(dim=1).mean(dim=1).mean()
        # loss = loss1
        loss = supervised_loss(kp, kp_pred)
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
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _ = model(input_point_cloud,
                                                                                            correction_flag=correction_flag)

            vloss = validation_loss(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                   output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))
            running_vloss += vloss

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def train_with_supervision(supervised_training_loader, validation_loader, model, optimizer, correction_flag):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(SAVE_LOCATION + 'expt_keypoint_detect_{}'.format(timestamp)) #ToDo: HP
    epoch_number = 0

    EPOCHS = 10 #ToDo: HP

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
            model_path = SAVE_LOCATION + 'expt_keypoint_detect_' + 'model_{}_{}'.format(timestamp, epoch_number) #ToDo: HP
            torch.save(model.state_dict(), model_path)
            best_model_path = SAVE_LOCATION + '_best_supervised_keypoint_detect_pt_se3.pth' #ToDo: HP
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
    dataset_dir = '../../../data/learning_objects/'

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
    supervised_train_dataset = SE3PointCloud(class_id=class_id,
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
    val_dataset = SE3PointCloud(class_id=class_id,
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
    from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel
    # Note: set use_pretrained_regression_model to True if we're using models pretrained on se3 dataset with supervision
    # if keypoint_detector=RSNetKeypoints, it will automatically load the pretrained model inside the detector,
    # so set use_pretrained_regression_model=False
    # This difference is because RSNetKeypoints was trained with supervision in KeypointNet,
    # whereas RegressionKeypoints was trained with supervision in this script and saved ProposedModel weights
    model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                # Use this for supervised training of pointnet
                          keypoint_detector='point_transformer', use_pretrained_regression_model=False).to(device)
    # model = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,                  # Use this for visualizing a pre-trained pointnet
    #                       keypoint_detector=None, use_pretrained_regression_model=True).to(device)

    if model.use_pretrained_regression_model:
        print("USING PRETRAINED REGRESSION MODEL, ONLY USE THIS WITH SELF-SUPERVISION")
        best_model_checkpoint = os.path.join(SAVE_LOCATION, '_best_supervised_keypoint_detect_pt_se3.pth')
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

    # test
    print("Visualizing the trained model.")
    visual_test(supervised_train_loader, model, correction_flag=False)

    dataset = SE3PointCloud(class_id=class_id,
                                             model_id=model_id,
                                             num_of_points=500,
                                             dataset_len=supervised_train_dataset_len)
    # supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=supervised_train_dataset_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=supervised_train_batch_size, shuffle=False)

    visual_test(test_loader=loader, model=model, correction_flag=False)

    dataset = SE3PointCloud(class_id=class_id,
                            model_id=model_id,
                            num_of_points=200,
                            dataset_len=supervised_train_dataset_len)
    # supervised_train_dataset = DepthPC(class_id=class_id,
    #                                         model_id=model_id,
    #                                         n=2000,
    #                                         num_of_points_to_sample=1000,
    #                                         dataset_len=supervised_train_dataset_len)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=supervised_train_batch_size,
                                         shuffle=False)
    visual_test(test_loader=loader, model=model, correction_flag=False)

    # dataset = SE3PointCloud(class_id=class_id,
    #                         model_id=model_id,
    #                         num_of_points=100,
    #                         dataset_len=supervised_train_dataset_len)
    # # supervised_train_dataset = DepthPC(class_id=class_id,
    # #                                         model_id=model_id,
    # #                                         n=2000,
    # #                                         num_of_points_to_sample=1000,
    # #                                         dataset_len=supervised_train_dataset_len)
    # loader = torch.utils.data.DataLoader(dataset,
    #                                      batch_size=supervised_train_batch_size,
    #                                      shuffle=False)
    # visual_test(test_loader=loader, model=model, correction_flag=False)



    # testing on real dataset
    real_batch_size = 1
    real_dataset = DepthPC(class_id=class_id, model_id=model_id, n=2000, num_of_points_to_sample=1000,
                           dataset_len=val_dataset_len)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=real_batch_size, shuffle=False)
    visual_test(test_loader=real_loader, model=model, correction_flag=False)
    visual_test(test_loader=real_loader, model=model, correction_flag=True)

