


import torch
import open3d as o3d
import numpy as np

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3nIsotorpicShapePointCloud, SE3PointCloud

from learning_objects.utils.general import pos_tensor_to_o3d



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





if __name__ == "__main__":

    print('-' * 20)
    print("Running expt_keypoint_detectPACE_noShape.py")
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
    shape_scaling = torch.tensor([1.0, 5.0])
    dataset_dir = '../../data/learning_objects/'
    dataset_len = 12000
    batch_size = 120
    lr_sgd = 0.02
    momentum_sgd = 0.9
    lr_adam = 0.02
    num_of_points = 500


    # se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=num_of_points,
    #                             dataset_len=dataset_len)
    se3_dataset = SE3nIsotorpicShapePointCloud(class_id=class_id, model_id=model_id, num_of_points=num_of_points,
                                               dataset_len=dataset_len)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=batch_size, shuffle=False)


    # Generate a shape category, CAD model objects, etc.
    # cad_models = se3_dataset._get_cad_models().to(torch.float).to(device=device)
    # model_keypoints = se3_dataset._get_model_keypoints().to(torch.float).to(device=device)
    # print("cad_models shape: ", cad_models.shape)
    # print("model_keypoints shape: ", model_keypoints.shape)

    # keypoints_xyz = se3_dataset.keypoints_xyz
    # print(keypoints_xyz.shape)




    for i, data in enumerate(se3_dataset_loader):

        mesh_o3d, pcd_o3d, pcd, keypoints, R, t, c = data

        o3d.visualization.draw_geometries([mesh_o3d, pcd_o3d])

        print(pcd.shape)
        print(keypoints.shape)
        # o3d.visualization.draw_geometries([pcd])
        display_two_pcs(pc1=pcd[0, ...], pc2=keypoints[0, ...])

        if i >= 2:
            break





