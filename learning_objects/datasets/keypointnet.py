import copy

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


import csv
import torch
import pandas as pd
import open3d as o3d
import json
import numpy as np
import pytorch3d
from pytorch3d import transforms

import os
import sys
sys.path.append("../../")

import learning_objects.utils.general as gu


def get_model_and_keypoints(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
        from the KeypointNet dataset.
        output: mesh (o3d.geometry.TriangleMesh)
                pcd (o3d.geometry.PointCloud)
                keypoints (o3d.utils.Vector3dVector(nx3))
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


def visualize_model(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
    from the KeypointNet dataset and plots them using open3d.visualization.draw_geometries"""

    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    _ = visualize_model_n_keypoints([pcd], keypoints_xyz=keypoints_xyz)

    return mesh, pcd, keypoints_xyz, keypoint_markers



def generate_depth_data(class_id, model_id, radius_multiple = [1.2, 3.0],
                        num_of_points=100000, num_of_depth_images_per_radius=200,
                        dir_location='../../data/learning-objects/keypointnet_datasets/'):
    """ Generates depth point clouds of the CAD model """

    radius_multiple = np.asarray(radius_multiple)

    location = dir_location + str(class_id) + '/' + str(model_id) + '/'
    # get model
    model_mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id, model_id)
    model_pcd = model_mesh.sample_points_uniformly(number_of_points=num_of_points)

    center = model_pcd.get_center()
    model_pcd.translate(-center)
    model_mesh.translate(-center)
    keypoints_xyz = keypoints_xyz - center

    model_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # determining size of the 3D object
    diameter = np.linalg.norm(np.asarray(model_pcd.get_max_bound()) - np.asarray(model_pcd.get_min_bound()))

    # determining camera locations and radius
    camera_distance_vector = diameter*radius_multiple
    camera_locations = gu.get_camera_locations(camera_distance_vector, number_of_locations=num_of_depth_images_per_radius)
    radius = gu.get_radius(object_diameter=diameter, cam_location=np.max(camera_distance_vector))

    # visualizing 3D object and all the camera locations
    _ = visualize_model_n_keypoints([model_pcd], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)
    _ = visualize_model_n_keypoints([model_mesh], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)

    # generating radius for view sampling
    gu.sample_depth_pcd(centered_pcd=model_pcd, camera_locations=camera_locations, radius=radius, folder_name=location)

    # save keypoints_xyz at location
    np.save(file=location+'keypoints_xyz.npy', arr=keypoints_xyz)



class DepthPointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for a given CAD model (class_id, model_id).

    It outputs various depth point clouds of the given CAD model.
    """
    def __init__(self, class_id, model_id, dir_name='../../data/learning_objects/keypointnet_datasets/',
                 radius_multiple=[1.2, 3.0], num_of_depth_images_per_radius=500, num_of_points=1000, torch_out=False):

        # Generating Data
        if not os.path.isdir(dir_name + class_id + '/'):
            os.mkdir(path=dir_name + class_id + '/')
            os.mkdir(path=dir_name + class_id + '/' + model_id + '/')

            location = dir_name + class_id + '/' + model_id + '/'
            generate_depth_data(class_id=class_id, model_id=model_id,
                                       radius_multiple=radius_multiple,
                                       num_of_points=num_of_points,
                                       dir_location=dir_name,
                                       num_of_depth_images_per_radius=num_of_depth_images_per_radius)
        else:
            print("It appears that the data was either already generated or the path specified is not empty.")

        object_file = 'object.pcd'
        camera_locations_file = 'camera_locations.pcd'
        metadata_file = 'metadata.csv'
        keypoint_numpy_file = 'keypoints_xyz.npy'

        # instead of object_file, have to work with class_id and model_id
        self.metadata_file = metadata_file
        self.class_id = class_id
        self.model_id = model_id
        self.dir_name = dir_name + str(self.class_id) + '/' +str(self.model_id) + '/'
        self.object_file = object_file
        self.camera_locations_file = camera_locations_file
        self.torch_out = torch_out

        self.metadata = pd.read_csv(self.dir_name + self.metadata_file)
        self.object = o3d.io.read_point_cloud(self.dir_name + self.object_file)
        self.camera_locations = o3d.io.read_point_cloud(self.dir_name + self.camera_locations_file)

        self.keypoints_xyz = np.load(file=self.dir_name + str(keypoint_numpy_file))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        depth_pcd_file_name = self.metadata.iloc[idx, 0]
        depth_pcd_file = os.path.join(self.dir_name, depth_pcd_file_name)

        depth_pcd = o3d.io.read_point_cloud(depth_pcd_file)
        depth_pcd.estimate_normals()
        depth_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        if self.torch_out:
            return torch.from_numpy(np.asarray(depth_pcd.points)).transpose(1, 2).to(torch.float)
        else:
            return depth_pcd

    def _get_cad_models(self):

        return torch.from_numpy(np.asarray(self.object.points)).transpose(0, 1).unsqueeze(0).to(torch.float)

    def _get_model_keypoints(self):

        return torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)


class SE3PointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for a given CAD model (class_id, model_id).

    It outputs various SE(3) transformations of the given input point cloud.
    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):

        self.class_id = class_id
        self.model_id = model_id

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        self.model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=num_of_points)
        center = self.model_pcd.get_center()
        self.model_pcd.translate(-center)
        self.model_mesh.translate(-center)
        self.keypoints_xyz = self.keypoints_xyz - center
        self.model_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        self.model_pcd_torch = torch.from_numpy(np.asarray(self.model_pcd.points)).transpose(0, 1)  # (3, m)
        self.model_pcd_torch = self.model_pcd_torch.to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_pcd.get_max_bound()) - np.asarray(self.model_pcd.get_min_bound()))

        # length of the dataset
        self.len = dataset_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        return R @ self.model_pcd_torch + t, R, t

    def _get_cad_models(self):

        return self.model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):

        return torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)


def get_dataset(class_id, model_id, dir_location, batch_size=4):
    """
    This function shows how to obtain depth and SE3 datasets.
    """
    # Parameters for depth dataset:
    radius_multiple = [1.2, 3.0]
    num_of_depth_images_per_radius = 500
    num_of_points = 1000

    # Depth Dataset
    depth_dataset = DepthPointCloud(class_id=class_id, model_id=model_id,
                                           radius_multiple=radius_multiple,
                                           num_of_depth_images_per_radius=num_of_depth_images_per_radius,
                                           num_of_points=num_of_points,
                                           dir_name=dir_location,
                                           torch_out=True)

    # Depth Dataset Loader
    depth_dataset_loader = torch.utils.data.DataLoader(depth_dataset, batch_size=batch_size, shuffle=True)




    # Parameters for SE3 dataset:
    num_of_points = 1000
    dataset_len = len(radius_multiple) * num_of_depth_images_per_radius

    # SE3 Dataset
    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=num_of_points,
                                       dataset_len=dataset_len)

    # SE3 Dataset Loader
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=batch_size, shuffle=False)


    return depth_dataset, depth_dataset_loader, se3_dataset, se3_dataset_loader



if __name__ == "__main__":

    # Testing the workings of DepthPointCloud(torch.utils.data.Dataset) and SE3PointCloud(torch.utils.data.Dataset)
    dir_location = '../../data/learning_objects/keypointnet_datasets/'
    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    batch_size = 5

    depth_dataset, depth_dataset_loader, se3_dataset, se3_dataset_loader = get_dataset(class_id=class_id,
                                                                                       model_id=model_id,
                                                                                       dir_location=dir_location,
                                                                                       batch_size=batch_size)

    cad_models = depth_dataset._get_cad_models()
    model_keypoints = depth_dataset._get_model_keypoints()

    print("CAD models shape: ", cad_models.shape)
    print("Model keypoints shape: ", model_keypoints.shape)


