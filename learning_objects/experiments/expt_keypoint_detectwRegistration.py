"""
This code is to use PACE for training a keypoint detector in a self-supervised manner.

We are given a CAD model with keypoints.
4.
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import open3d as o3d
import sys
sys.path.append("../../")


from learning_objects.utils.ddn.node import ParamDeclarativeFunction
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import max_chamfer_half_distance, chamfer_half_distance, max_chamfer_distance, \
    chamfer_distance, keypoint_error, soft_chamfer_half_distance
from learning_objects.utils.general import rotation_error, shape_error, translation_error
from learning_objects.utils.general import display_results
from learning_objects.utils.general import pos_tensor_to_o3d

from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.pace_ddn import PACEbp, PACEddn
from learning_objects.models.pace import PACE, PACEmodule
from learning_objects.models.point_set_registration import point_set_registration
from learning_objects.models.modelgen import ModelFromShape, ModelFromShapeModule

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud



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
    def __init__(self, model_keypoints, cad_models, weights=None, batch_size=32):
        super().__init__()
        """
        model_keypoints : torch.tensor of shape (1, 3, N)
        cad_models      : torch.tensor of shape (1, 3, n)
        weights         : torch.tensor of shape (N, 1) or None  
        """

        # Parameters
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.batch_size = batch_size

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        # self.K = self.model_keypoints.shape[0]  # (1, 1)

        # self.keypoint_method = 'pointnet'
        self.keypoint_method = 'point_transformer'

        self.weights = weights
        if weights == None:
            self.weights = torch.ones((1, self.N), device=self.device_)


        # Keypoint Detector
        self.keypoint_detector = RegressionKeypoints(N=self.N, method=self.keypoint_method, dim=[3, 16, 32, 64, 128])


        # point_set_registration


        # Model Generator
        self.generate_model = ModelFromShapeModule(cad_models=self.cad_models,
                                                   model_keypoints=self.model_keypoints).to(device=self.device_)


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
        batch_size = input_point_cloud.shape[0]



        # detect keypoints
        detected_keypoints = self.keypoint_detector(input_point_cloud)      # (B, 3, N)


        # registration
        rotation, translation = point_set_registration(source_points=self.model_keypoints.repeat(batch_size, 1, 1),
                                                       target_points=detected_keypoints, weights=self.weights)


        # model generation
        target_keypoints = rotation @ self.model_keypoints.repeat(batch_size, 1, 1) + translation
        target_point_cloud = rotation @ self.cad_models.repeat(batch_size, 1, 1) + translation


        return detected_keypoints, target_keypoints, target_point_cloud, rotation, translation



# loss function

def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (3, n)
    pc2 : torch.tensor of shape (3, m)
    """
    pc1 = pc1.to('cpu')
    pc2 = pc2.to('cpu')

    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None

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



def chamfer_loss(pc, pc_):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)

    output:
    loss    :
    """
    return chamfer_distance(pc, pc_).mean()




def loss(input_point_cloud, target_point_cloud,
         detected_keypoints, target_keypoints, keypoints_true,
         rotation_true, rotation_est,
         translation_true, translation_est):
    """
    inputs:
    input_point_cloud   : torch.tensor of shape (B, 3, m)
    target_point_cloud  : torch.tensor of shape (B, 3, n)

    detected_keypoints  : torch.tensor of shape (B, 3, N)
    target_keypoints    :
    keypoints_true      :

    rotation_true       : torch.tensor of shape (B, 3, 3)
    rotation_est        :

    translation_true    : torch.tensor of shape (B, 3, 1)
    translation_est     :

    output:
    loss                : torch.tensor of shape (1, 1)
    """
    const_rotation = 5.0
    const_translation = 5.0
    const_kp1 = 10.0
    const_kp2 = 0.0
    const_pc = 0.0

    loss_rotation = rotation_loss(rotation_true, rotation_est)
    loss_translation = translation_loss(translation_true, translation_est)
    loss_kp1 = keypoints_loss(keypoints_true, detected_keypoints)
    loss_kp2 = keypoints_loss(keypoints_true, target_keypoints)
    # loss_pc = pytorch3d.loss.chamfer_distance(input_point_cloud.transpose(-1, -2),
    #                                 target_point_cloud.transpose(-1, -2),
    #                                 batch_reduction="mean")[0].mean()
    loss_pc = chamfer_loss(input_point_cloud, target_point_cloud)

    return const_rotation*loss_rotation + const_translation*loss_translation + const_kp1*loss_kp1 + const_kp2*loss_kp2 \
           + const_pc*loss_pc



# Train the keypoint detector with point set registration
def train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, loss_fn, device):
    running_loss = 0.
    last_loss = 0.
    batch_size = training_loader.batch_size
    num_batches = len(training_loader)
    view_every_n_batches = int(0.2*num_batches)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # print("Training batch: ", i)
        # Every data instance is an input + label pair
        input_point_cloud, rotation_true, translation_true = data
        input_point_cloud = input_point_cloud.to(device)
        rotation_true = rotation_true.to(device)
        translation_true = translation_true.to(device)

        # Get ground truth
        keypoints_true = rotation_true @ model.model_keypoints + translation_true
        # point_cloud_true = rotation_true @ model.cad_models + translation_true    # same as input point cloud

        # display input_point_cloud, point_cloud_true
        # display_two_pcs(pc1=input_point_cloud[0, ...], pc2=point_cloud_true[0, ...])


        for iter in range(1):

            # print("iter: ", iter)
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            detected_keypoints, target_keypoints, target_point_cloud, rotation_est, translation_est = model(input_point_cloud)


            # Compute the loss and its gradients
            loss = loss_fn(input_point_cloud, target_point_cloud, detected_keypoints, target_keypoints,
                                keypoints_true, rotation_true, rotation_est, translation_true, translation_est)
            loss.backward()
            # print("Loss: ", loss)

            # Adjust learning weights
            optimizer.step()


        # Gather data and report
        running_loss += loss.item()

        if i % view_every_n_batches == view_every_n_batches-1:
            last_loss = running_loss / view_every_n_batches # loss per batch
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

        del input_point_cloud, rotation_true, translation_true, keypoints_true
        del detected_keypoints, target_point_cloud, target_keypoints, rotation_est, translation_est

        torch.cuda.empty_cache()

    return last_loss


save_location = '../../data/learning_objects/runs/'
def train(training_loader, validation_loader, model, optimizer, loss_fn, device):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(save_location + 'expt_keypoint_detect_wRegistration{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 100

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_loader, model, optimizer, loss_fn, device)


        # Display results


        # Validation. We don't need gradients on to do reporting.
        print('EPOCH {}:'.format(epoch_number + 1) + ' validation')
        model.train(False)
        with torch.no_grad():

            running_vloss = 0.0

            for i, vdata in enumerate(validation_loader):

                input_point_cloud, rotation_true, translation_true = vdata
                input_point_cloud = input_point_cloud.to(device)
                rotation_true = rotation_true.to(device)
                translation_true = translation_true.to(device)

                # Get ground truth
                keypoints_true = rotation_true @ model.model_keypoints + translation_true
                # point_cloud_true = rotation_true @ model.cad_models + translation_true    # same as input point cloud

                # Make predictions for this batch
                detected_keypoints, target_keypoints, target_point_cloud, rotation_est, translation_est = model(
                    input_point_cloud)


                vloss = loss_fn(input_point_cloud, target_point_cloud, detected_keypoints, target_keypoints,
                                keypoints_true, rotation_true, rotation_est, translation_true, translation_est)
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

                del input_point_cloud, rotation_true, translation_true, keypoints_true
                del detected_keypoints, target_point_cloud, target_keypoints, rotation_est, translation_est


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
                model_path = save_location + 'expt_keypoint_detect_wRegistration_' + 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            epoch_number += 1


        torch.cuda.empty_cache()


    return None




# Evaluation. Use the fact that you know rotation, translation, and shape of the generated data.
def evaluation(rotation, rotation_target, translation, translation_target, shape=None, shape_target=None):
    """
    rotation, rotation_target       :   torch.tensor of shape (B, 3, 3)
    translation, translation_target :   torch.tensor of shape (B, 3, 1)
    shape, shape_target             :   torch.tensor of shpae (B, K, 1)
    """

    if shape == None:
        print("Rotation error: ", rotation_error(rotation, rotation_target).mean())
        print("Translation error: ", translation_error(translation, translation_target).mean())
    else:
        print("Rotation error: ", rotation_error(rotation, rotation_target).mean())
        print("Translation error: ", translation_error(translation, translation_target).mean())
        print("Shape error: ", shape_error(shape, shape_target).mean())


# Test the keypoint detector with PACE. See if you can learn the keypoints.
def visual_test(test_loader, model, loss_fn):

    for i, vdata in enumerate(test_loader):
        input_point_cloud, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        detected_keypoints, target_keypoints, target_point_cloud, R, t = model(input_point_cloud)

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_t = target_point_cloud.clone().detach().to('cpu')
        kp = detected_keypoints.clone().detach().to('cpu')
        kp_t = target_keypoints.clone().detach().to('cpu')
        display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc_t,
                        target_keypoints=kp_t)

        evaluation(rotation=R, rotation_target=R_target,
                   translation=t, translation_target=t_target)

        del pc, pc_t, kp, kp_t
        del input_point_cloud, R_target, t_target, detected_keypoints, target_point_cloud, target_keypoints

        if i >= 4:
            break






if __name__ == "__main__":

    print('-' * 20)
    print("Running expt_keypoint_detectwRegistration.py")
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
    dataset_len = 32000
    # batch_size = 200
    batch_size = 32
    lr_sgd = 0.002
    momentum_sgd = 0.9
    lr_adam = 0.02
    num_of_points = 500

    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=num_of_points, dataset_len=dataset_len)
    # print("Object size: ", se3_dataset.diameter)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=batch_size, shuffle=False)

    # Generate a shape category, CAD model objects, etc.
    cad_models = se3_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = se3_dataset._get_model_keypoints().to(torch.float).to(device=device)


    # model
    model = ProposedModel(model_keypoints=model_keypoints, cad_models=cad_models, batch_size=batch_size).to(device)
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # loss function
    loss_fn = loss

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=momentum_sgd, weight_decay=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)


    # training
    train(training_loader=se3_dataset_loader, validation_loader=se3_dataset_loader,
          model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)


    # test
    print("Visualizing the trained model.")
    visual_test(test_loader=se3_dataset_loader, model=model, loss_fn=loss_fn)


