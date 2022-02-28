import copy
from enum import Enum

DATASET_PATH: str = '../../data/ycb/models/ycb/'


import csv
import torch
import pandas as pd
import open3d as o3d
import json
import numpy as np
import pytorch3d
from pytorch3d import transforms, ops

import os
import sys
sys.path.append("../../")

from learning_objects.models.modelgen import ModelFromShape
from learning_objects.utils.general import pos_tensor_to_o3d
import learning_objects.utils.general as gu


class ScaleAxis(Enum):
    X = 0
    Y = 1
    Z = 2
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


def get_model_and_keypoints(model_id):
    """
    Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints from the KeypointNet dataset.

    inputs:
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    pcd         : o3d.geometry.PointCloud
    keypoints   : o3d.utils.Vector3dVector(nx3)
    """

    object_mesh_file = DATASET_PATH + model_id + '/google_16k/nontextured.ply'
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals() #how long does this take
    pcd = None
    kpt_filename = os.path.join(DATASET_PATH + model_id, "kpts_xyz.npy")
    keypoints_xyz = np.load(kpt_filename)

    return mesh, pcd, keypoints_xyz


def visualize_model_n_keypoints(model_list, keypoints_xyz, camera_locations=o3d.geometry.PointCloud()):

    d = 0
    for model in model_list:
        max_bound = model.get_max_bound()
        min_bound = model.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

    keypoint_radius = 0.01 * d

    keypoint_markers = []
    for xyz in keypoints_xyz:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(new_mesh)

    camera_locations.paint_uniform_color([0.1, 0.5, 0.1])
    o3d.visualization.draw_geometries(keypoint_markers + model_list + [camera_locations])

    return keypoint_markers


def visualize_model(model_id):
    """ Given class_id and model_id this function outputs the colored mesh and keypoints
    from the ycb dataset and plots them using open3d.visualization.draw_geometries"""

    mesh, _, keypoints_xyz = get_model_and_keypoints(model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)

    return mesh, None, keypoints_xyz, keypoint_markers


def visualize_torch_model_n_keypoints(cad_models, model_keypoints):
    """
    inputs:
    cad_models      : torch.tensor of shape (B, 3, m)
    model_keypoints : torch.tensor of shape (B, 3, N)

    """
    batch_size = model_keypoints.shape[0]

    for b in range(batch_size):

        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...]

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()

        visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

    return 0


class SE3PointCloudYCB(torch.utils.data.Dataset):
    """
    Given model_id, and number of points generates various point clouds and SE3 transformations
    of the ycb object.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, model_id, num_of_points=1000, dataset_len=10000,
                 dir_location='../../data/learning-objects/ycb_datasets/'):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        #center the cad model
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        # return R @ model_pcd_torch + t, R, t
        return R @ model_pcd_torch + t, R @ self.keypoints_xyz.squeeze(0) + t, R, t

    def _get_cad_models_as_mesh(self):
        """
        Returns the open3d Mesh object of the ShapeNetCore model

        """

        return self.model_mesh

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ShapeNetCore model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0

class DepthYCB(torch.utils.data.Dataset):
    """
    Given model_id and split, get real depth images from YCB dataset.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, model_id, split='train', num_of_points=500,
                 dir_location='../../data/learning-objects/ycb_datasets/'):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain

        """

        self.model_id = model_id
        self.split = split
        self.num_of_points = num_of_points

        self.pcd_data_root = os.path.join(DATASET_PATH + model_id, "clouds/largest_cluster/")

        self.split_filenames = np.load(self.pcd_data_root + split + '_split.npy')
        print("dataset len",  self.split_filenames.shape[0])
        self.len = self.split_filenames.shape[0]

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        # #center the cad model WE DON'T DO THIS FOR REAL DEPTH DATA BECAUSE WE DON'T HAVE
        # TRANSFORMATIONS TO A CENTERED VERSION OF THE PCL
        # center = self.model_mesh.get_center()
        # self.model_mesh.translate(-center)
        #
        # self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """
        pcd = o3d.io.read_point_cloud(self.pcd_data_root + self.split_filenames[idx])
        pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        pcd_torch = pcd_torch.to(torch.float)

        #downsample to number of points expected
        m = pcd_torch.shape[-1]
        if m > self.num_of_points:
            shuffle_idxs = torch.randperm(m)
            point_cloud = pcd_torch[:, shuffle_idxs[:self.num_of_points]]

        #load ground truth R, ground truth t
        print(self.split_filenames[idx])#.split('_'))
        _, viewpoint_camera, reference_camera, viewpoint_angle, _ = tuple(self.split_filenames[idx].split('_'))
        # return R @ model_pcd_torch + t, R, t
        rgbFromObj_filename = os.path.join(DATASET_PATH + self.model_id, "poses/gt_wrt_rgb/",
                                           '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
        rgbFromObj = np.load(rgbFromObj_filename)
        R_true = torch.from_numpy(rgbFromObj[:3, :3]).to(torch.float)
        t_true = torch.from_numpy(rgbFromObj[:3,3]).unsqueeze(-1).to(torch.float)

        print("inside DepthYCB dataloader getitem")
        print("point_cloud.shape", point_cloud.shape)
        print("keypoints_xyz.shape", self.keypoints_xyz.shape)
        print("R_true.shape", R_true.shape)
        print("t_true.shape", t_true.shape)
        return point_cloud, R_true @ self.keypoints_xyz.squeeze(0) + t_true, R_true, t_true

    def _get_cad_models_as_mesh(self):
        """
        Returns the open3d Mesh object of the ShapeNetCore model

        """

        return self.model_mesh

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ShapeNetCore model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0

if __name__ == "__main__":

    # Testing the workings of DepthPointCloud(torch.utils.data.Dataset) and SE3PointCloud(torch.utils.data.Dataset)
    dir_location = '../../data/learning_objects/ycb_datasets/'
    model_id = "019_pitcher_base"  # a particular chair model
    batch_size = 5
    #
    #
    print("Test: DepthPC()")
    dataset = DepthYCB(model_id=model_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 50:
            break

    #

    #
    print("Test: get_model_and_keypoints()")
    mesh, _, keypoints_xyz = get_model_and_keypoints(model_id=model_id)
    # print(keypoints_xyz)
    # print(type(keypoints_xyz))
    # print(type(keypoints_xyz[0]))

    #
    print("Test: visualize_model_n_keypoints()")
    visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)

    #
    print("Test: visualize_model()")
    visualize_model(model_id=model_id)

    #
    # print("Test: SE3PoiontCloud(torch.utils.data.Dataset)")
    # dataset = SE3PointCloud(class_id=class_id, model_id=model_id)
    #
    # model = dataset.model_mesh
    # length = dataset.len
    # class_id = dataset.class_id
    # model_id = dataset.model_id
    # num_of_points = dataset.num_of_points
    #
    # print("Shape of keypoints_xyz: ", keypoints_xyz.shape)
    #
    # diameter = dataset._get_diameter()
    # model_keypoints = dataset._get_model_keypoints()
    # cad_models = dataset._get_cad_models_as_point_clouds()
    #
    # print("diameter: ", diameter)
    # print("shape of model keypoints: ", model_keypoints.shape)
    # print("shape of cad models: ", cad_models.shape)
    #
    # #
    # print("Test: visualize_torch_model_n_keypoints()")
    # visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    # dataset._visualize()
    #
    #
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t = data
    #     print(pc.shape)
    #     print(kp.shape)
    #     visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #     if i >= 2:
    #         break
    #
    #
    # #
    # print("Test: DepthPoiontCloud(torch.utils.data.Dataset)")
    # dataset = DepthPointCloud(class_id=class_id, model_id=model_id)
    #
    # model = dataset.model_pcd
    # length = dataset.len
    # class_id = dataset.class_id
    # model_id = dataset.model_id
    #
    # diameter = dataset._get_diameter()
    # model_keypoints = dataset._get_model_keypoints()
    # cad_models = dataset._get_cad_models()
    #
    # print("diameter: ", diameter)
    # print("shape of model keypoints: ", model_keypoints.shape)
    # print("shape of cad models: ", cad_models.shape)
    #
    # #
    # print("Test: visualize_torch_model_n_keypoints()")
    # visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    # dataset._visualize()
    #
    #
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #
    # for i, data in enumerate(loader):
    #     pc = data
    #     kp = model_keypoints
    #     print(pc.shape)
    #     print(kp.shape)
    #     visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #     if i >= 2:
    #         break
