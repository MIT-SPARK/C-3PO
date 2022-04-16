import random

ANNOTATIONS_FOLDER: str = '../../data/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = '../../data/KeypointNet/KeypointNet/pcds/'
MESH_FOLDER_NAME: str = '../../data/KeypointNet/ShapeNetCore.v2.ply/'
OBJECT_CATEGORIES: list = ['airplane', 'bathtub', 'bed', 'bottle',
                           'cap', 'car', 'chair', 'guitar',
                           'helmet', 'knife', 'laptop', 'motorcycle',
                           'mug', 'skateboard', 'table', 'vessel']
CLASS_ID: dict = {'airplane': "02691156",
                  'bathtub': "02808440",
                  'bed': "02818832",
                  'bottle': "02876657",
                  'cap': "02954340",
                  'car': "02958343",
                  'chair': "03001627",
                  'guitar': "03467517",
                  'helmet': "03513137",
                  'knife': "03624134",
                  'laptop': "03642806",
                  'motorcycle': "03790512",
                  'mug': "03797390",
                  'skateboard': "04225987",
                  'table': "04379243",
                  'vessel': "04530566"}

CLASS_NAME: dict = {"02691156": 'airplane',
                    "02808440": 'bathtub',
                    "02818832": 'bed',
                    "02876657": 'bottle',
                    "02954340": 'cap',
                    "02958343": 'car',
                    "03001627": 'chair',
                    "03467517": 'guitar',
                    "03513137": 'helmet',
                    "03624134": 'knife',
                    "03642806": 'laptop',
                    "03790512": 'motorcycle',
                    "03797390": 'mug',
                    "04225987": 'skateboard',
                    "04379243": 'table',
                    "04530566": 'vessel'}

NUM_KEYPOINTS: dict = {'airplane': "14",
                  'bathtub': "16",
                  'bed': "10",
                  'bottle': "17",
                  'cap': "6",
                  'car': "22",
                  'chair': "10",
                  'guitar': "9",
                  'helmet': "9",
                  'knife': "6",
                  'laptop': "6",
                  'motorcycle': "14",
                  'mug': "11",
                  'skateboard': "10",
                  'table': '8',
                  'vessel': "17"}

# import csv
import torch
import math
# import pandas as pd
import open3d as o3d
import json
import numpy as np
# import pytorch3d
from pytorch3d import transforms, ops
# import random

# import os
import sys
sys.path.append("../../")

# from learning_objects.models.modelgen import ModelFromShape
from learning_objects.utils.general import pos_tensor_to_o3d
import learning_objects.utils.general as gu


def get_model_ids(class_name):
    """
    Given class_id this function outputs all the model_ids.

    """

    class_id = CLASS_ID[class_name]
    annotation_file = ANNOTATIONS_FOLDER + CLASS_NAME[str(class_id)] + '.json'
    file_temp = open(str(annotation_file))
    anotation_data = json.load(file_temp)

    model_id_list = []
    keypoints_list = []
    for idx, entry in enumerate(anotation_data):

        model_id = entry['model_id']
        keypoints = entry['keypoints']
        keypoints_xyz = []
        for aPoint in keypoints:
            keypoints_xyz.append(aPoint['xyz'])

        keypoints_xyz = np.array(keypoints_xyz)

        if keypoints_xyz.shape[0] == int(NUM_KEYPOINTS[class_name]):
            model_id_list.append(model_id)
            keypoints_list.append(keypoints_xyz)
            # print("Model ID: ", model_id, " : num. keypoints: ", keypoints_xyz.shape[0], " : Expected: ",
            #       NUM_KEYPOINTS[class_name])

    return model_id_list, keypoints_list


def get_model_and_keypoints(class_id, model_id):
    """
    Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints from the KeypointNet dataset.

    inputs:
    class_id    : string
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    pcd         : o3d.geometry.PointCloud
    keypoints   : o3d.utils.Vector3dVector(nx3)
    """

    object_pcd_file = PCD_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.pcd'
    object_mesh_file = MESH_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.ply'

    pcd = o3d.io.read_point_cloud(filename=object_pcd_file)
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals()

    annotation_file = ANNOTATIONS_FOLDER + CLASS_NAME[str(class_id)] + '.json'
    file_temp = open(str(annotation_file))
    anotation_data = json.load(file_temp)

    for idx, entry in enumerate(anotation_data):
        if entry['model_id'] == str(model_id):
            keypoints = entry['keypoints']
            break

    keypoints_xyz = []
    for aPoint in keypoints:
        keypoints_xyz.append(aPoint['xyz'])

    keypoints_xyz = np.array(keypoints_xyz)

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


def visualize_torch_model_n_keypoints(cad_models, model_keypoints):
    """
    inputs:
    cad_models      : torch.tensor of shape (B, 3, m)
    model_keypoints : torch.tensor of shape (B, 3, N)

    """
    batch_size = model_keypoints.shape[0]
    pc_list = []
    for b in range(batch_size):

        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()
        pc_list.append(point_cloud)

        # visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)
    o3d.visualization.draw_geometries(pc_list)
    return 0


class CategoryKeypointNetDataset(torch.utils.data.Dataset):
    """
    Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
    of the ShapeNetCore object.

    Note:
        The output depth point clouds will not contain the same number of points. Therefore, when using with a
        dataloader, fix the batch_size=1.

    Returns
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, class_name,
                 radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points=1000,
                 dataset_len=10000,
                 rotate_about_z=False,
                 depth=False,
                 add_noise=True, noise_var=0.01,
                 no_se3_transformation=False):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset  

        """

        self.class_id = CLASS_ID[class_name]
        self.radius_multiple = radius_multiple
        self.num_of_points = num_of_points
        self.dataset_len = dataset_len
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])
        self.diameter = 1.0
        self.depth = depth
        self.add_noise = add_noise
        self.noise_var = noise_var
        self.no_se3_transformation = no_se3_transformation
        # set a camera location, with respect to the origin
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1)
        self.folder_name = PCD_FOLDER_NAME + self.class_id + '/'

        self.model_id_list, _ = get_model_ids(class_name)

    def __len__(self):

        # return len(self.model_id_list)
        return self.dataset_len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)                  : the depth point cloud
        keypoints           : torch.tensor of shape (3, N)                  : transformed keypoints
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        model_pcd_torch     : torch.tensor of shape (3, self.num_of_points) : transformed full point cloud
        """

        # randomly pick a model in the object category
        model_id = random.choice(self.model_id_list)
        # model_id = self.model_id_list[idx]

        # get model
        model_mesh, _, keypoints_xyz = get_model_and_keypoints(self.class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)

        keypoints_xyz = keypoints_xyz - center
        keypoints_xyz = torch.from_numpy(keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # Randomly rotate the self.model_mesh
        # model_mesh = copy.deepcopy(model_mesh)
        if self.rotate_about_z:
            R = torch.eye(3)
            angle = 2 * self.pi * torch.rand(1)
            c = torch.cos(angle)
            s = torch.sin(angle)

            # # z
            # R[0, 0] = c
            # R[0, 1] = -s
            # R[1, 0] = s
            # R[1, 1] = c

            # # x
            # R[1, 1] = c
            # R[1, 2] = -s
            # R[2, 1] = s
            # R[2, 2] = c

            # y
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c

        else:
            R = transforms.random_rotation()
        # Rnumpy = R.detach()
        # Rnumpy = Rnumpy.numpy()
        if not self.no_se3_transformation:
            model_mesh = model_mesh.rotate(R=R.numpy())
            keypoints_xyz = R @ keypoints_xyz

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        if self.depth:
            beta = torch.rand(1, 1)
            camera_location_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
            camera_location_factor = camera_location_factor * self.diameter
            radius = gu.get_radius(cam_location=camera_location_factor*self.camera_location.numpy(),
                                   object_diameter=self.diameter)
            depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

            depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
            depth_pcd_torch = depth_pcd_torch.to(torch.float)
            pcd = depth_pcd_torch
        else:
            model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
            model_pcd_torch = model_pcd_torch.to(torch.float)
            pcd = model_pcd_torch

        # Translate by a random t
        if not self.no_se3_transformation:
            t = torch.rand(3, 1)
            pcd = pcd + t
            keypoints_xyz = keypoints_xyz + t
        else:
            t = torch.zeros(3, 1)

        if self.add_noise:
            pcd += torch.normal(mean=torch.zeros_like(pcd), std=self.noise_var * torch.ones_like(pcd))

        return pcd, keypoints_xyz.squeeze(0), R, t,


if __name__ == "__main__":

    # model_ids, _ = get_model_ids('chair')
    # print("num of models: ", len(model_ids))
    dataset = CategoryKeypointNetDataset(class_name='chair', no_se3_transformation=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, data in enumerate(loader):

        depth_pcd, kp, R, t = data

        visualize_torch_model_n_keypoints(cad_models=depth_pcd, model_keypoints=kp)
        if idx >= 10:
            break

