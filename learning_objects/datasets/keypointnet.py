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

from learning_objects.models.modelgen import ModelFromShape
from learning_objects.utils.general import pos_tensor_to_o3d
import learning_objects.utils.general as gu


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


def visualize_model(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
    from the KeypointNet dataset and plots them using open3d.visualization.draw_geometries"""

    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    _ = visualize_model_n_keypoints([pcd], keypoints_xyz=keypoints_xyz)

    return mesh, pcd, keypoints_xyz, keypoint_markers


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


class DepthAndIsotorpicShapePointCloud(torch.utils.data.Dataset):
    def __init__(self, class_id, model_id, shape_scaling=torch.tensor([0.5, 2.0]), radius_multiple=[1.2, 3.0],
                 num_of_points=1000, dataset_len=10000):
        super().__init__()

        self.class_id = class_id
        self.model_id = model_id
        self.shape_scaling = shape_scaling
        self.radius_multiple = radius_multiple
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        #
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud : torch.tensor of shape (3, m)
        R           : torch.tensor of shape (3, 3)
        t           : torch.tensor of shape (3, 1)
        """

        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
        R = transforms.random_rotation()
        # Rnumpy = R.detach()
        # Rnumpy = Rnumpy.numpy()
        model_mesh = model_mesh.rotate(R=R.numpy())

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_lcation_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
        radius = gu.get_radius(cam_location=camera_lcation_factor*self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)


        # Scale the depth point cloud, by a random quantity, bounded within the self.radius_multiple
        alpha = torch.rand(1, 1)
        scaling_factor = alpha * (self.shape_scaling[1] - self.shape_scaling[0]) + self.shape_scaling[0]

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)
        depth_pcd_torch = scaling_factor * depth_pcd_torch

        keypoints_xyz = R @ self.keypoints_xyz
        keypoints_xyz = scaling_factor * keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        # shape parameter
        shape = torch.zeros(2, 1)
        shape[0] = 1 - alpha
        shape[1] = alpha

        # model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = scaling_factor * model_pcd_torch + t

        return depth_pcd_torch, keypoints_xyz.squeeze(0), R, t, shape, model_pcd_torch


    def _get_cad_models(self):

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        cad_models = torch.zeros_like(model_pcd_torch).unsqueeze(0)
        cad_models = cad_models.repeat(2, 1, 1)

        cad_models[0, ...] = self.shape_scaling[0]*model_pcd_torch
        cad_models[1, ...] = self.shape_scaling[1]*model_pcd_torch

        return cad_models

    def _get_model_keypoints(self):

        keypoints = self.keypoints_xyz  # (3, N)

        model_keypoints = torch.zeros_like(keypoints)
        model_keypoints = model_keypoints.repeat(2, 1, 1)

        model_keypoints[0, ...] = self.shape_scaling[0]*keypoints
        model_keypoints[1, ...] = self.shape_scaling[1]*keypoints

        return model_keypoints

    def _get_diameter(self):
        """
        returns the diameter of the mid-sized object.
        """
        return self.diameter*(self.shape_scaling[0] + self.shape_scaling[1])*0.5

    def _visualize(self):

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


class DepthPointCloud2(torch.utils.data.Dataset):
    def __init__(self, class_id, model_id, radius_multiple=[1.2, 3.0], num_of_points=1000, dataset_len=10000):
        super().__init__()

        self.class_id = class_id
        self.model_id = model_id
        self.radius_multiple = radius_multiple
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        #
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud : torch.tensor of shape (3, m)
        R           : torch.tensor of shape (3, 3)
        t           : torch.tensor of shape (3, 1)
        """

        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
        R = transforms.random_rotation()
        # Rnumpy = R.detach()
        # Rnumpy = Rnumpy.numpy()
        model_mesh = model_mesh.rotate(R=R.numpy())

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_lcation_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
        radius = gu.get_radius(cam_location=camera_lcation_factor*self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)

        keypoints_xyz = R @ self.keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t


        # model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = model_pcd_torch + t

        return depth_pcd_torch, keypoints_xyz.squeeze(0), R, t, model_pcd_torch


    def _get_cad_models_as_point_clouds(self):

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_cad_models_as_mesh(self):

        return self.model_mesh

    def _get_cad_models(self):
        #Depreciated. Use _get_cad_models_as_point_clouds().

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):

        return self.keypoints_xyz

    def _get_diameter(self):

        return self.diameter

    def _visualize(self):

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


class DepthPointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for a given CAD model (class_id, model_id).

    It outputs various depth point clouds of the given CAD model.
    """
    def __init__(self, class_id, model_id, dir_name='../../data/learning_objects/keypointnet_datasets/',
                 radius_multiple=[1.2, 3.0], num_of_depth_images_per_radius=500, num_of_points=1000, torch_out=True):

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
        self.model_pcd = self.object
        self.camera_locations = o3d.io.read_point_cloud(self.dir_name + self.camera_locations_file)

        self.keypoints_xyz = np.load(file=self.dir_name + str(keypoint_numpy_file))

        self.len = len(self.metadata)
        self.diameter = np.linalg.norm(
            np.asarray(self.model_pcd.get_max_bound()) - np.asarray(self.model_pcd.get_min_bound()))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        depth_pcd_file_name = self.metadata.iloc[idx, 0]
        depth_pcd_file = os.path.join(self.dir_name, depth_pcd_file_name)

        depth_pcd = o3d.io.read_point_cloud(depth_pcd_file)
        depth_pcd.estimate_normals()
        depth_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        if self.torch_out:
            return torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1).to(torch.float)
        else:
            return depth_pcd

    def _get_cad_models(self):

        return torch.from_numpy(np.asarray(self.object.points)).transpose(0, 1).unsqueeze(0).to(torch.float)

    def _get_model_keypoints(self):

        return torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

    def _get_diameter(self):

        return self.diameter

    def _visualize(self):

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


class SE3PointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for a given CAD model (class_id, model_id).

    It outputs various SE(3) transformations of the given input point cloud.
    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):

        self.class_id = class_id
        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
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
        point_cloud : torch.tensor of shape (3, m)
        R           : torch.tensor of shape (3, 3)
        t           : torch.tensor of shape (3, 1)
        """

        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)


        return R @ model_pcd_torch + t, R, t


    def _get_cad_models_as_point_clouds(self):

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_cad_models_as_mesh(self):

        return self.model_mesh

    def _get_cad_models(self):
        #Depreciated. Use _get_cad_models_as_point_clouds().

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):

        return self.keypoints_xyz

    def _get_diameter(self):

        return self.diameter

    def _visualize(self):

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


class SE3nIsotorpicShapePointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for a given CAD model (class_id, model_id).

    It outputs various SE(3) transformations and Shape by isotropic scaling of the given input point cloud.

    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 shape_scaling=torch.tensor([0.5, 2.0]),
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):

        self.class_id = class_id
        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len
        self.shape_scaling = shape_scaling

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)
        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # # scaling to minimum scale
        # scaling_factor_min = self.shape_scaling[0].clone().detach().numpy()
        # scaling = np.eye(4)
        # scaling[0, 0] = scaling_factor_min
        # scaling[1, 1] = scaling_factor_min
        # scaling[2, 2] = scaling_factor_min
        #
        # self.model_mesh = self.model_mesh.transform(scaling)
        # self.keypoints_xyz = scaling_factor_min*self.keypoints_xyz


        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # random scaling
        alpha = torch.rand(1, 1)
        scaling_factor = alpha*(self.shape_scaling[1]-self.shape_scaling[0]) + self.shape_scaling[0]

        model_mesh = self.model_mesh
        model_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = scaling_factor*model_pcd_torch

        keypoints_xyz = scaling_factor*self.keypoints_xyz


        # random rotation and translation
        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd_torch = R @ model_pcd_torch + t
        keypoints_xyz = R @ keypoints_xyz + t

        # shape parameter
        shape = torch.zeros(2, 1)
        shape[0] = 1 - alpha
        shape[1] = alpha

        return model_pcd_torch, keypoints_xyz.squeeze(0), R, t, shape


    def _get_cad_models(self):

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        cad_models = torch.zeros_like(model_pcd_torch).unsqueeze(0)
        cad_models = cad_models.repeat(2, 1, 1)

        cad_models[0, ...] = self.shape_scaling[0]*model_pcd_torch
        cad_models[1, ...] = self.shape_scaling[1]*model_pcd_torch

        return cad_models

    def _get_model_keypoints(self):

        keypoints = self.keypoints_xyz  # (3, N)

        model_keypoints = torch.zeros_like(keypoints)
        model_keypoints = model_keypoints.repeat(2, 1, 1)

        model_keypoints[0, ...] = self.shape_scaling[0]*keypoints
        model_keypoints[1, ...] = self.shape_scaling[1]*keypoints

        return model_keypoints

    def _get_diameter(self):
        """
        returns the diameter of the mid-sized object.
        """
        return self.diameter*(self.shape_scaling[0] + self.shape_scaling[1])*0.5

    def _visualize(self):

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


if __name__ == "__main__":

    # Testing the workings of DepthPointCloud(torch.utils.data.Dataset) and SE3PointCloud(torch.utils.data.Dataset)
    dir_location = '../../data/learning_objects/keypointnet_datasets/'
    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    batch_size = 5


    #
    print("Test: get_model_and_keypoints()")
    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)
    # print(keypoints_xyz)
    # print(type(keypoints_xyz))
    # print(type(keypoints_xyz[0]))

    #
    print("Test: visualize_model_n_keypoints()")
    visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)

    #
    print("Test: visualize_model()")
    visualize_model(class_id=class_id, model_id=model_id)

    #
    print("Test: SE3PoiontCloud(torch.utils.data.Dataset)")
    dataset = SE3PointCloud(class_id=class_id, model_id=model_id)

    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id
    num_of_points = dataset.num_of_points

    print("Shape of keypoints_xyz: ", keypoints_xyz.shape)

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models_as_point_clouds()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()


    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, R, t = data
        kp = R @ model_keypoints + t
        print(pc.shape)
        print(kp.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break


    #
    print("Test: DepthPoiontCloud(torch.utils.data.Dataset)")
    dataset = DepthPointCloud(class_id=class_id, model_id=model_id)

    model = dataset.model_pcd
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()


    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc = data
        kp = model_keypoints
        print(pc.shape)
        print(kp.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break



    #
    print("Test: SE3nIsotorpicShapePointCloud(torch.utils.data.Dataset)")
    dataset = SE3nIsotorpicShapePointCloud(class_id=class_id, model_id=model_id,
                                           shape_scaling=torch.tensor([5.0, 20.0]))

    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()


    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)

    for i, data in enumerate(loader):
        pc, kp, R, t, c = data
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)


        # getting the model from R, t, c
        kp_gen, pc_gen = modelgen.forward(shape=c)
        kp_gen = R @ kp_gen + t
        pc_gen = R @ pc_gen + t
        display_two_pcs(pc1=pc[0, ...], pc2=pc_gen[0, ...])
        display_two_pcs(pc1=kp[0, ...], pc2=kp_gen[0, ...])

        if i >= 5:
            break




    #
    print("Test: DepthPoiontCloud2(torch.utils.data.Dataset)")
    dataset = DepthAndIsotorpicShapePointCloud(class_id=class_id, model_id=model_id)

    model = dataset.model_mesh
    length = dataset.len
    class_id = dataset.class_id
    model_id = dataset.model_id

    diameter = dataset._get_diameter()
    model_keypoints = dataset._get_model_keypoints()
    cad_models = dataset._get_cad_models()

    print("diameter: ", diameter)
    print("shape of model keypoints: ", model_keypoints.shape)
    print("shape of cad models: ", cad_models.shape)

    #
    print("Test: visualize_torch_model_n_keypoints()")
    visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)
    dataset._visualize()

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  #Note: the batch size has to be one!

    for i, data in enumerate(loader):
        depth_pcd_torch, keypoints_xyz, R, t, shape, model_pcd_torch = data
        pc = depth_pcd_torch
        kp = keypoints_xyz
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        print(shape.shape)
        print(model_pcd_torch.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 2:
            break