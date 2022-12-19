import copy
import json
import math
import numpy as np
import open3d as o3d
import os
import pickle
# import pytorch3d
import sys
import torch
from enum import Enum
from tqdm import tqdm
# from pytorch3d import transforms, ops
from scipy.spatial.transform import Rotation as Rot
import random
from pathlib import Path

sys.path.append("../../")

# from c3po.models.modelgen import ModelFromShape
# from c3po.utils.general import pos_tensor_to_o3d
from c3po.utils.visualization_utils import display_two_pcs, visualize_model_n_keypoints, \
    visualize_torch_model_n_keypoints, display_results
import c3po.utils.general as gu
from c3po.datasets.utils_dataset import PointRegistrationMedium, PointRegistrationEasy, fromFormat

BASE_DIR = Path(__file__).parent.parent

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

CLASS_MODEL_ID: dict = {'airplane': '3db61220251b3c9de719b5362fe06bbb',
                        'bathtub': '90b6e958b359c1592ad490d4d7fae486',
                        'bed': '7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f',
                        'bottle': '41a2005b595ae783be1868124d5ddbcb',
                        'cap': '3dec0d851cba045fbf444790f25ea3db',
                        'car': 'ad45b2d40c7801ef2074a73831d8a3a2',
                        'chair': '1cc6f2ed3d684fa245f213b8994b4a04',
                        'guitar': '5df08ba7af60e7bfe72db292d4e13056',
                        'helmet': '3621cf047be0d1ae52fafb0cab311e6a',
                        'knife': '819e16fd120732f4609e2d916fa0da27',
                        'laptop': '519e98268bee56dddbb1de10c9529bf7',
                        'motorcycle': '481f7a57a12517e0fe1b9fad6c90c7bf',
                        'mug': 'f3a7f8198cc50c225f5e789acd4d1122',
                        'skateboard': '98222a1e5f59f2098745e78dbc45802e',
                        'table': '3f5daa8fe93b68fa87e2d08958d6900c',
                        'vessel': '5c54100c798dd681bfeb646a8eadb57'}


class ScaleAxis(Enum):
    X = 0
    Y = 1
    Z = 2


MODEL_TO_KPT_GROUPS = {
    "mug": [set([9])],
    "cap": [set([1])]
    }


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


def visualize_model(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
    from the KeypointNet dataset and displays them using open3d.visualization.draw_geometries"""

    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    _ = visualize_model_n_keypoints([pcd], keypoints_xyz=keypoints_xyz)

    return mesh, pcd, keypoints_xyz, keypoint_markers


def generate_depth_data(class_id, model_id, radius_multiple = [1.2, 3.0],
                        num_of_points=100000, num_of_depth_images_per_radius=200,
                        dir_location='../../data/learning-objects/keypointnet_datasets/'):
    """ Generates a dataset of depth point clouds of a CAD model from randomly generated views. """

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


class DepthPC(torch.utils.data.Dataset):
    """
    Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
    of the ShapeNetCore object.

    Note:
        Outputs depth point clouds of the same shape by appending with
        zero points. It also outputs a flag to tell the user which outputs have been artificially added.

        This dataset can be used with a dataloader for any batch_size.

    Returns
        input_point_cloud, keypoints, rotation, translation
    """

    def __init__(self, class_id, model_id, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=10000, rotate_about_z=False):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """

        self.class_id = class_id
        self.model_id = model_id
        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = num_of_points_to_sample
        self.len = dataset_len
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        # Randomly rotate the self.model_mesh
        model_mesh = copy.deepcopy(self.model_mesh)
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
            # R = transforms.random_rotation()
            R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)

        model_mesh = model_mesh.rotate(R=R.numpy())

        # Sample a point cloud from the self.model_mesh
        pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_to_sample)

        # Take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta*(self.radius_multiple[1]-self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * self.diameter
        radius = gu.get_radius(cam_location=camera_location_factor*self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)

        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float)

        keypoints_xyz = R @ self.keypoints_xyz

        # Translate by a random t
        t = torch.rand(3, 1)

        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t

        model_pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)
        model_pcd_torch = model_pcd_torch + t

        point_cloud, padding = self._convert_to_fixed_sized_pc(depth_pcd_torch, n=self.n)

        return point_cloud, keypoints_xyz.squeeze(0), R, t

    def _convert_to_fixed_sized_pc(self, pc, n):
        """
        Adds (0,0,0) points to the point cloud if the number of points is less than
        n such that the resulting point cloud has n points.

        inputs:
        pc  : torch.tensor of shape (3, m)  : input point cloud of size m (m could be anything)
        n   : int                           : number of points the output point cloud should have

        outputs:
        point_cloud     : torch.tensor of shape (3, n)
        padding         : torch.tensor of shape (n)

        """

        m = pc.shape[-1]

        if m > n:
            idx = torch.randperm(m)
            point_cloud = pc[:, idx[:n]]
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        elif m < n:

            pc_pad = torch.zeros(3, n - m)
            point_cloud = torch.cat([pc, pc_pad], dim=1)
            padding1 = torch.zeros(size=(m,), dtype=torch.bool)
            padding2 = torch.ones(size=(n - m,), dtype=torch.bool)
            padding = torch.cat([padding1, padding2], dim=0)
            # Write code to pad pc with (n-m) zeros

        else:
            point_cloud = pc
            padding = torch.zeros(size=(n,), dtype=torch.bool)

        return point_cloud, padding

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points_to_sample
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)
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


class FixedDepthPC(torch.utils.data.Dataset):
    """
        Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
        of the ShapeNetCore object.

        Note:
            Outputs depth point clouds of the same shape by appending with
            zero points. It also outputs a flag to tell the user which outputs have been artificially added.

            This dataset can be used with a dataloader for any batch_size.

        Returns
            input_point_cloud, keypoints, rotation, translation
        """

    def __init__(self, class_id, model_id, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=1, rotate_about_z=False,
                 base_dataset_folder='../../data/learning_objects/shapenet_depthpc_eval_data/',
                 mixed_data=False):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """
        self.base_dataset_folder = base_dataset_folder
        self.class_id = class_id
        self.class_name = CLASS_NAME[self.class_id]
        self.model_id = model_id
        if mixed_data:
            self.dataset_folder = self.base_dataset_folder + 'mixed/'
        else:
            self.dataset_folder = self.base_dataset_folder + self.class_name + '/' + self.model_id + '/'

        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = 1000
        self.len = len(os.listdir(self.dataset_folder))
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin


    def __len__(self):

        return self.len

    def __getitem__(self, idx):

        filename = self.dataset_folder + 'item_' + str(idx) + '.pkl'
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)

        return data[0].squeeze(0), data[1].squeeze(0), data[2].squeeze(0), data[3].squeeze(0)

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points_to_sample
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)
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


class MixedFixedDepthPC(torch.utils.data.Dataset):
    """
        Given class id, model id, and number of points, it generates various depth point clouds and SE3 transformations
        of the ShapeNetCore object.

        Note:
            Outputs depth point clouds of the same shape by appending with
            zero points. It also outputs a flag to tell the user which outputs have been artificially added.

            This dataset can be used with a dataloader for any batch_size.

        Returns
            input_point_cloud, keypoints, rotation, translation
        """

    def __init__(self, class_id, model_id, base_dataset_folder, n=1000, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points_to_sample=10000, dataset_len=1, rotate_about_z=False, mixed_data=True):
        super().__init__()
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object 
        n               : int   : number of points in the output point cloud
        radius_multiple : torch.tensor of shape (2) : lower and upper limit of the distance from which depth point 
                                                        cloud is constructed
        num_of_points_to_sample   : int   : number of points sampled on the surface of the CAD model object
        dataset_len     : int   : size of the dataset  

        """
        self.base_dataset_folder = base_dataset_folder
        self.class_id = class_id
        self.class_name = CLASS_NAME[self.class_id]
        self.model_id = model_id

        if mixed_data:
            self.dataset_folder = self.base_dataset_folder + 'mixed/'
        else:
            self.dataset_folder = self.base_dataset_folder + self.class_name + '/' + self.model_id + '/'

        self.n = n
        self.radius_multiple = radius_multiple
        self.num_of_points_to_sample = 1000
        self.len = len(os.listdir(self.dataset_folder))
        self.rotate_about_z = rotate_about_z
        self.pi = torch.tensor([math.pi])

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(class_id, model_id)
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(
            np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin


    def __len__(self):

        return self.len

    def __getitem__(self, idx):

        filename = self.dataset_folder + 'item_' + str(idx) + '.pkl'
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)

        return data[0].squeeze(0), data[2].squeeze(0), data[3].squeeze(0)


    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ShapeNetcore model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """
        if self.n is None:
            self.n = self.num_of_points
        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.n)
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


class SE3PointCloud(torch.utils.data.Dataset):
    """
    Given class_id, model_id, and number of points generates various point clouds and SE3 transformations
    of the ShapeNetCore object.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, class_id, model_id, num_of_points=1000, dataset_len=10000,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

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
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        # R = transforms.random_rotation()
        R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)
        t = torch.rand(3, 1)

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return R @ model_pcd_torch + t, R @ self.keypoints_xyz.squeeze(0) + t, R, t

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


class SE3PointCloudAll(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformations.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, num_of_points=1024, dataset_len=2048,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.objects = OBJECT_CATEGORIES
        self.num_of_points = num_of_points
        self.len = dataset_len

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        # randomly choose an object category name
        class_name = random.choice(self.objects)
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, kp = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)
        kp = kp - center
        kp1 = torch.from_numpy(kp).transpose(0, 1).unsqueeze(0).to(torch.float)

        # diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))

        # R = transforms.random_rotation()
        R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)
        t = torch.rand(3, 1)

        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)
        pc1 = pc1.to(torch.float)

        pc2 = R @ pc1 + t
        kp2 = R @ kp1 + t

        return (pc1, pc2, kp1, kp2, R, t)


class DepthPCAll(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformed depth rendering.
    pc2 is depth point cloud.
    This doesn't do zero padding for depth point clouds.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, num_of_points1=1024, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points2=2048, dataset_len=10000, rotate_about_z=False):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.objects = OBJECT_CATEGORIES
        self.num_of_points_pc1 = num_of_points1
        self.len = dataset_len
        self.num_of_points_pc2 = num_of_points2
        self.radius_multiple = radius_multiple
        self.rotate_about_z = rotate_about_z
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1)
        # set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        # randomly choose an object category name
        class_name = random.choice(self.objects)
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, kp = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)
        kp = kp - center

        # computing diameter
        diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))

        # extracting the first data
        kp1 = torch.from_numpy(kp).transpose(0, 1).unsqueeze(0).to(torch.float)
        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_pc1)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)  # (3, m)
        pc1 = pc1.to(torch.float)

        # apply random rotation
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
            # R = transforms.random_rotation()
            R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)

        model_mesh = model_mesh.rotate(R=R.numpy())

        # sample a point cloud from the self.model_mesh
        pc2_pcd_ = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_pc2)

        # take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * diameter
        radius = gu.get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                               object_diameter=diameter)
        pc2_pcd = gu.get_depth_pcd(centered_pcd=pc2_pcd_, camera=self.camera_location.numpy(), radius=radius)

        pc2 = torch.from_numpy(np.asarray(pc2_pcd.points)).transpose(0, 1)  # (3, m)
        pc2 = pc2.to(torch.float)

        # Translate by a random t
        t = torch.rand(3, 1)
        pc2 = pc2 + t
        kp2 = R @ kp1 + t

        return (pc1, pc2, kp1, kp2, R, t)


class ShapeNet(torch.utils.data.Dataset):
    def __init__(self, type, object, length, num_points, adv_option='hard', from_file=False, filename=None):

        assert adv_option in ['hard', 'medium', 'easy']
        # hard: c3po rotation errors
        # easy: lk rotation errors
        # medium: deepgmr rotation errors

        assert type in ['sim', 'real']
        # sim: full point clouds
        # real: depth point clouds

        assert object in OBJECT_CATEGORIES + ['all']
        # object: category name in ShapeNet

        self.type = type
        self.class_name = object
        self.length = length
        self.num_points = num_points

        self.adv_option = adv_option
        self.from_file = from_file
        self.filename = filename

        # new
        self.from_file = False

        if self.from_file:
            with open(self.filename, 'rb') as f:
                self.data_ = pickle.load(f)

        # else:
        if self.type == 'real':

            # new
            # self.ds_ = DepthPC(class_name=self.class_name,
            #                    dataset_len=self.length,
            #                    num_of_points=self.num_points)
            self.ds_ = FixedDepthPC(class_id=CLASS_ID[self.class_name],
                                    model_id=CLASS_MODEL_ID[self.class_name])
            self.ds_ = fromFormat(self.ds_)

        elif self.type == 'sim':

            # new
            # self.ds_ = SE3PointCloud(class_name=self.class_name,
            #                          dataset_len=self.length,
            #                          num_of_points=self.num_points)
            self.ds_ = SE3PointCloud(class_id=CLASS_ID[self.class_name],
                                     model_id=CLASS_MODEL_ID[self.class_name],
                                     num_of_points=self.num_points,
                                     dataset_len=self.length)
            self.ds_ = fromFormat(self.ds_)

        else:
            raise ValueError

        if self.adv_option == 'hard':
            self.ds = self.ds_
        elif self.adv_option == 'easy':
            self.ds = PointRegistrationEasy(self.ds_)
        elif self.adv_option == 'medium':
            self.ds = PointRegistrationMedium(self.ds_)
        else:
            raise ValueError

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, item):

        if self.from_file:
            pc1, pc2, kp1, kp2, R, t = self.data_[item]
        else:
            pc1, pc2, kp1, kp2, R, t = self.ds[item]

        return (pc1, pc2, kp1, kp2, R, t)

    def save_dataset(self, filename):

        data_ = []
        for i in tqdm(range(self.ds.__len__())):
            data = self.ds[i]
            data_.append(data)

        with open(filename, 'wb') as f:
            pickle.dump(data_, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_cad_models(self):

        return self.ds_.cad_models

    def _get_model_keypoints(self):

        return self.ds_.model_keypoints

if __name__ == "__main__":

    dir_location = '../../data/learning_objects/keypointnet_datasets/'
    class_id = "03001627"  # chair
    model_id = "1cc6f2ed3d684fa245f213b8994b4a04"  # a particular chair model
    batch_size = 5



    # ds = SE3PointCloudAll()
    ds = DepthPCAll()
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    for i, data in enumerate(dl):
        pc1, pc2, kp1, kp2, R, t = data
        display_two_pcs(pc1, pc2)
        display_results(pc1, kp1, pc2, kp2)
        if i >=5:
            break


    # #
    # print("Test: DepthIsoPC()")
    # dataset = DepthIsoPC(class_id=class_id, model_id=model_id)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t, c = data
    #     print(pc.shape)
    #     print(kp.shape)
    #     print(R.shape)
    #     print(t.shape)
    #     # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #     if i >= 2:
    #         break
    # print("Test: DepthAnsoPC()")
    # dataset = DepthAnisoPC(class_id=class_id, model_id=model_id)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t, c = data
    #     print(pc.shape)
    #     print(kp.shape)
    #     print(R.shape)
    #     print(t.shape)
    #     # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #     if i >= 2:
    #         break
    #
    #
    # #
    # print("Test: DepthPC()")
    # dataset = DepthPC(class_id=class_id, model_id=model_id)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t = data
    #     print(pc.shape)
    #     print(kp.shape)
    #     print(R.shape)
    #     print(t.shape)
    #     # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #     if i >= 2:
    #         break
    #
    # #
    # print("Test: get_model_and_keypoints()")
    # mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)
    # # print(keypoints_xyz)
    # # print(type(keypoints_xyz))
    # # print(type(keypoints_xyz[0]))
    #
    # #
    # print("Test: visualize_model_n_keypoints()")
    # visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    #
    # #
    # print("Test: visualize_model()")
    # visualize_model(class_id=class_id, model_id=model_id)
    #
    # #
    # print("Test: SE3PointCloud(torch.utils.data.Dataset)")
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
    # cad_models = dataset._get_cad_models()
    #
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
    # dataset = FixedDepthPC(class_id=class_id, model_id=model_id)
    # model = dataset.model_mesh
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
    # loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t = data
    #     print(pc.shape)
    #     print(kp.shape)
    #     print(R.shape)
    #     print(t.shape)
    #     # visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #     if i >= 2:
    #         break
    #
    # #
    # print("Test: SE3nIsotropicShapePointCloud(torch.utils.data.Dataset)")
    # dataset = SE3nIsotropicShapePointCloud(class_id=class_id, model_id=model_id,
    #                                        shape_scaling=torch.tensor([5.0, 20.0]))
    # model = dataset.model_mesh
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
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #
    # modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t, c = data
    #     visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #
    #     # getting the model from R, t, c
    #     kp_gen, pc_gen = modelgen.forward(shape=c)
    #     kp_gen = R @ kp_gen + t
    #     pc_gen = R @ pc_gen + t
    #     display_two_pcs(pc1=pc, pc2=pc_gen)
    #     display_two_pcs(pc1=kp, pc2=kp_gen)
    #
    #     if i >= 5:
    #         break
    #
    # #
    # print("Test: SE3nAnisotropicScalingPointCloud(torch.utils.data.Dataset)")
    # dataset = SE3nAnisotropicScalingPointCloud(class_id=class_id, model_id=model_id,
    #                                        shape_scaling=torch.tensor([1.0, 2.0]),
    #                                        scale_direction=ScaleAxis.X)
    #
    # model = dataset.model_mesh
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
    # modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)
    #
    # for i, data in enumerate(loader):
    #     pc, kp, R, t, c = data
    #     visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #
    #
    #     # getting the model from R, t, c
    #     kp_gen, pc_gen = modelgen.forward(shape=c)
    #     kp_gen = R @ kp_gen + t
    #     pc_gen = R @ pc_gen + t
    #     display_two_pcs(pc1=pc, pc2=pc_gen)
    #     display_two_pcs(pc1=kp, pc2=kp_gen)
    #
    #     if i >= 5:
    #         break

