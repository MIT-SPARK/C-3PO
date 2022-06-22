import csv
import libmcubes
import numpy as np
import open3d as o3d
import os
import pytorch3d
import random
import sys
import torch
import torch.nn as nn
from pytorch3d import transforms

sys.path.append("../../")
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api','utils'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','car_averaging'))
import apollo_utils as uts
from car_instance.car_models import *


from learning_objects.models.modelgen_apolloscape import ModelFromShapeApollo, ModelFromShapeApolloModule
from learning_objects.utils.general import pos_tensor_to_o3d
from learning_objects.utils.visualization_utils import visualize_model_n_keypoints, visualize_torch_model_n_keypoints
import learning_objects.utils.general as gu

MODEL_NAMES_TO_IGNORE = ['036-CAR01', '037-CAR02', 'aodi-a6', 'baoma-X5', 'baoshijie-kayan',
                         'benchi-GLK-300', 'benchi-ML500', 'benchi-SUR', 'biyadi-F3',
                         'dongfeng-fengguang-S560', 'jili-boyue',
                         'feiyate', 'fengtian-liangxiang', 'fengtian-SUV-gai',
                         'kaidilake-CTS',
                         'mazida-6-2015', 'sikeda-jingrui', 'Skoda_Fabia-2011', 'yingfeinidi-SUV']
APOLLOSCAPE_DATASET_SIZE = 79
PATH_TO_O3D_CAR_MODELS_WATERTIGHT = "../../dataset/apollo_car_3d/3d_car_instance_sample/car_models_watertight_scaled/"
PATH_TO_SCALED_KEYPOINTS = "../../dataset/apollo_car_3d/keypoints_3d/1_scaled_kpts/"
PATH_TO_TSDFS = "../../third_party/apolloscape/car_averaging/car_models/fused_tsdf/"


PATH_TO_PCD: str = '../../dataset/apollo_car_3d/pointclouds/apolloscape_pointclouds_09072021/largest_cluster/'
DEPTH_PCD_FOLDER_NAME: str = '../../dataset/apollo_car_3d/pointclouds/apolloscape_pointclouds_depth/'


def get_model_and_keypoints(model_id):
    """
    Given model_id this function outputs the watertight clustered and scaled mesh, and keypoints from the Apolloscape Dataset.

    inputs:
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    keypoints   : numpy.array(nx3)
    """
    model_name = car_id2name[model_id].name
    object_mesh_file = PATH_TO_O3D_CAR_MODELS_WATERTIGHT + str(model_name) + '.ply'

    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals()
    # add keypoints
    keypoints_xyz = []
    with open(os.path.join(PATH_TO_SCALED_KEYPOINTS, str(model_name) + '.csv'), newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            keypoints_xyz.append([float(x) for x in row])
    # rotation fix
    # (note: this is a rotation + reflection)
    # 90 deg around y, then a reflection across yz (x->-x)
    keypoints_xyz = np.array(keypoints_xyz)
    R = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    for i in range(keypoints_xyz.shape[0]):
        keypoints_xyz[i, :] = np.transpose(R @ keypoints_xyz[i, :].T)

    return mesh, keypoints_xyz


def visualize_model(model_id):
    """ Given model_id this function outputs the mesh and keypoints
    from the Apolloscape dataset and plots them using open3d.visualization.draw_geometries"""

    mesh, keypoints_xyz = get_model_and_keypoints(model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    _ = visualize_model_n_keypoints([pcd], keypoints_xyz=keypoints_xyz)

    return mesh, keypoints_xyz, keypoint_markers

def visualize_torch_model(cad_models):
    """
    inputs:
    cad_models      : torch.tensor of shape (B, 3, m)

    """
    cad_models.cpu()
    batch_size = cad_models.shape[0]

    for b in range(batch_size):

        point_cloud = cad_models[b, ...]

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        point_cloud.estimate_normals()
        o3d.visualization.draw_geometries([point_cloud])

    return 0


def generate_depth_data(model_id, radius_multiple = [1.2, 3.0],
                        num_of_points=100000, num_of_depth_images_per_radius=200,
                        dir_location=PATH_TO_O3D_CAR_MODELS_WATERTIGHT):
    """ Generates depth point clouds of the CAD model """

    radius_multiple = np.asarray(radius_multiple)
    model_name = car_id2name[model_id].name

    model_location = "{}/{}.ply".format(dir_location, model_name)
    # get model
    model_mesh, keypoints_xyz = get_model_and_keypoints(model_id)
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

    location = DEPTH_PCD_FOLDER_NAME + str(model_name) + '/'
    print("location", location)
    if not os.path.exists(location):
        os.makedirs(location)
        print("made new folder")
    # generating radius for view sampling
    gu.sample_depth_pcd(centered_pcd=model_pcd, camera_locations=camera_locations, radius=radius, folder_name=location)

    # save keypoints_xyz at location
    np.save(file=location+'keypoints_xyz.npy', arr=keypoints_xyz)

class SE3ApolloDataset(torch.utils.data.Dataset):
    """
    SE(3) transformation of a given input point cloud
    restricted to random rotations around the vertical axis
    Input: number of points, weight_mask mask of length n that defines the weights of the weighted avg output model
    Output: sampled point clouds from averaged CAD mesh models and
            random SE3 transformations
    Returns:
        input_point_cloud, keypoints, rotation, translation, weight_mask


    """
    def __init__(self, num_of_points=1000, dataset_len=10000, weight_mask_random = True,
                 weight_mask = torch.tensor([]), dir_location='../../dataset/apollo_car_3d/'):
        """

        :param num_of_points:
        :param dataset_len:
        :param weight_mask_random:
        :param weight_mask:
        :param dir_location:
        """

        # self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len
        self.weight_mask_random = weight_mask_random
        self.weight_mask = weight_mask

        self.modelgen = ModelFromShapeApolloModule(num_of_points=num_of_points)

        # get model
        self.model_pcd = None #model_pcd.squeeze()
        self.keypoints_xyz = None #keypoints_xyz.squeeze()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cad_models = None
        self.model_keypoints = None
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)                  : the depth point cloud
        keypoints           : torch.tensor of shape (3, N)                  : transformed keypoints
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        c                   : torch.tensor of shape (79,)                  : shape parameter

        """

        if self.weight_mask_random:
            keypoints_xyz, model_pcd, mask = self.modelgen()
        else:
            keypoints_xyz, model_pcd, mask = self.modelgen(self.weight_mask)
        self.model_pcd = model_pcd
        self.keypoints_xyz = keypoints_xyz
        self.weight_mask = mask

        # random rotation and translation
        random_angle = random.uniform(-np.pi, np.pi)
        print("random_angle,", random_angle)
        R = transforms.axis_angle_to_matrix(torch.from_numpy(np.array([0, random_angle, 0]).transpose())).to(self.device)
        # R = transforms.random_rotation().to(self.device)
        t = torch.rand(3, 1).to(self.device)
        model_pcd_torch = R.float() @ self.model_pcd.float() + t
        keypoints_xyz = R.float() @ self.keypoints_xyz.float() + t

        return model_pcd_torch.squeeze(0), keypoints_xyz.squeeze(0), R.squeeze(0), t.squeeze(0), self.weight_mask

    def _get_cad_models(self):
        """
        Returns APOLLOSCAPE_DATASET_SIZE point clouds as shape models.

        output:
        cad_models  : torch.tensor of shape (APOLLOSCAPE_DATASET_SIZE, 3, self.num_of_points)
        """
        if self.cad_models is not None:
            return self.cad_models
        cad_models = None
        model_keypoints = None
        for id in range(APOLLOSCAPE_DATASET_SIZE):
            if car_id2name[id].name not in MODEL_NAMES_TO_IGNORE:
                mask = np.zeros(79)
                mask[id] = 1.
                keypoints_xyz, model_pcd_torch, _ = self.modelgen(torch.from_numpy(mask.transpose()))
                model_pcd_torch = model_pcd_torch.to(torch.float)
                keypoints_xyz = keypoints_xyz.to(torch.float)

                if cad_models is None:
                    cad_models = model_pcd_torch
                    model_keypoints = keypoints_xyz
                else:
                    cad_models = torch.vstack((cad_models, model_pcd_torch))
                    model_keypoints = torch.vstack((model_keypoints, keypoints_xyz))
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        return cad_models
    def _get_model_keypoints(self):
        """
        Returns APOLLOSCAPE_DATASET_SIZE sets of keypoints.

        output:
        model_keypoints : torch.tensor of shape (APOLLOSCAPE_DATASET_SIZE, 3, N)

        where
        N = number of keypoints
        """

        if self.model_keypoints is not None:
            return self.model_keypoints
        cad_models = None
        model_keypoints = None
        for id in range(APOLLOSCAPE_DATASET_SIZE):
            if car_id2name[id].name not in MODEL_NAMES_TO_IGNORE:
                mask = np.zeros(79)
                mask[id] = 1.
                keypoints_xyz, model_pcd_torch, _ = self.modelgen(torch.from_numpy(mask.transpose()))
                model_pcd_torch = model_pcd_torch.to(torch.float)
                keypoints_xyz = keypoints_xyz.to(torch.float)

                if cad_models is None:
                    cad_models = model_pcd_torch
                    model_keypoints = keypoints_xyz
                else:
                    cad_models = torch.vstack((cad_models, model_pcd_torch))
                    model_keypoints = torch.vstack((model_keypoints, keypoints_xyz))
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        return model_keypoints

class ApolloDepthPointCloudDataset(torch.utils.data.Dataset):
    """
    Given and number of points, it generates various depth point clouds and SE3 transformations
    of Apolloscape car models parameterized by shape..

    Note:
        The output depth point clouds will not contain the same number of points. Therefore, when using with a
        dataloader, fix the batch_size=1.

    Returns
        input_point_cloud, keypoints, rotation, translation, weight_mask
    """
    def __init__(self, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points=1000, dataset_len=10000, weight_mask_random = True,
                 weight_mask = torch.tensor([])):
        """

        :param radius_multiple: torch.tensor of shape (2) : lower and upper limit of the distance from which depth point
                                cloud is constructed
        :param num_of_points: max. number of points the depth point cloud will contain
        :param dataset_len: size of the dataset
        :param weight_mask_random:
        :param weight_mask:
        """
        super().__init__()

        self.radius_multiple = radius_multiple
        self.num_of_points = num_of_points
        self.len = dataset_len


        self.weight_mask_random = weight_mask_random
        self.weight_mask = weight_mask

        self.modelgen = ModelFromShapeApolloModule(num_of_points=num_of_points)

        # get model
        self.model_pcd = None
        self.keypoints_xyz = None

        # size of the model
        self.diameter = None

        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1) #set a camera location, with respect to the origin

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #for saving cad models and model_keypoints
        self.cad_models = None
        self.model_keypoints = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)                  : the depth point cloud
        keypoints           : torch.tensor of shape (3, N)                  : transformed keypoints
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        c                   : torch.tensor of shape (79,)                  : shape parameter
        """
#
        if self.weight_mask_random:
            keypoints_xyz, model_pcd, mask = self.modelgen()
        else:
            keypoints_xyz, model_pcd, mask = self.modelgen(self.weight_mask)
        self.model_pcd = model_pcd
        self.keypoints_xyz = keypoints_xyz
        self.weight_mask = mask

        # random rotation
        random_angle = random.uniform(-np.pi, np.pi)
        print("random_angle,", random_angle)
        R = transforms.axis_angle_to_matrix(torch.from_numpy(np.array([0, random_angle, 0]).transpose())).to(self.device)
        # R = transforms.random_rotation().to(self.device)
        model_pcd_torch = R.float() @ self.model_pcd.float()# + t
        keypoints_xyz = R.float() @ self.keypoints_xyz.float()# + t

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(model_pcd_torch.cpu()).transpose())
        self.diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

        beta = torch.rand(1, 1)
        camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
        radius = gu.get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                               object_diameter=self.diameter)
        depth_pcd = gu.get_depth_pcd(centered_pcd=pcd, camera=self.camera_location.numpy(), radius=radius)
        depth_pcd_torch = torch.from_numpy(np.asarray(depth_pcd.points)).transpose(0, 1)  # (3, m)
        depth_pcd_torch = depth_pcd_torch.to(torch.float).to(self.device)

        # random translation
        t = torch.rand(3, 1).to(self.device)
        depth_pcd_torch = depth_pcd_torch + t
        keypoints_xyz = keypoints_xyz + t


        return depth_pcd_torch.squeeze(), keypoints_xyz.squeeze(), R.squeeze(), t.squeeze(0), self.weight_mask

    def _get_cad_models(self):
        """
        Returns APOLLOSCAPE_DATASET_SIZE point clouds as shape models.

        output:
        cad_models  : torch.tensor of shape (APOLLOSCAPE_DATASET_SIZE, 3, self.num_of_points)
        """
        if self.cad_models is not None:
            return self.cad_models
        cad_models = None
        model_keypoints = None
        for id in range(APOLLOSCAPE_DATASET_SIZE):
            if car_id2name[id].name not in MODEL_NAMES_TO_IGNORE:
                mask = np.zeros(79)
                mask[id] = 1.
                keypoints_xyz, model_pcd_torch, _ = self.modelgen(torch.from_numpy(mask.transpose()))
                model_pcd_torch = model_pcd_torch.to(torch.float)
                keypoints_xyz = keypoints_xyz.to(torch.float)

                if cad_models is None:
                    cad_models = model_pcd_torch
                    model_keypoints = keypoints_xyz
                else:
                    cad_models = torch.vstack((cad_models, model_pcd_torch))
                    model_keypoints = torch.vstack((model_keypoints, keypoints_xyz))
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        return cad_models
    def _get_model_keypoints(self):
        """
        Returns APOLLOSCAPE_DATASET_SIZE sets of keypoints.

        output:
        model_keypoints : torch.tensor of shape (APOLLOSCAPE_DATASET_SIZE, 3, N)

        where
        N = number of keypoints
        """

        if self.model_keypoints is not None:
            return self.model_keypoints
        cad_models = None
        model_keypoints = None
        for id in range(APOLLOSCAPE_DATASET_SIZE):
            if car_id2name[id].name not in MODEL_NAMES_TO_IGNORE:
                mask = np.zeros(79)
                mask[id] = 1.
                keypoints_xyz, model_pcd_torch, _ = self.modelgen(torch.from_numpy(mask.transpose()))
                model_pcd_torch = model_pcd_torch.to(torch.float)
                keypoints_xyz = keypoints_xyz.to(torch.float)

                if cad_models is None:
                    cad_models = model_pcd_torch
                    model_keypoints = keypoints_xyz
                else:
                    cad_models = torch.vstack((cad_models, model_pcd_torch))
                    model_keypoints = torch.vstack((model_keypoints, keypoints_xyz))
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        return model_keypoints


class ApolloSegPointCloudDataset(torch.utils.data.Dataset):
    """
    Generates a segmented point cloud from real images in Apolloscape

    Returns
        input_point_cloud
    """
    def __init__(self, num_of_points=1000, dataset_len=10000):
        """

        :param num_of_points: max. number of points the depth point cloud will contain
        :param dataset_len: size of the dataset
        """
        super().__init__()
        self.num_of_points = num_of_points
        self.len = dataset_len
        self.model_pcd = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        depth_pcd_torch     : torch.tensor of shape (3, m)  : the depth point cloud
        file                : str                           : the filename of the pcl
        """
        #sample a random pcd from pointclouds
        files = []
        for filename in os.listdir(PATH_TO_PCD):
            files.append(os.path.normpath(os.path.join(PATH_TO_PCD, filename)))
        random_file = random.choice(files)
        pcd = o3d.io.read_point_cloud(random_file)
        pcd.estimate_normals()
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        if len(pcd.points) > self.num_of_points:
            sampling_ratio = self.num_of_points/len(pcd.points)
            pcd = pcd.random_down_sample(sampling_ratio= sampling_ratio)
        # pcd.paint_uniform_color([0, 0.5, 0.5])
        # o3d.visualization.draw_geometries([pcd])
        # print("files", random_file)

        return torch.from_numpy(np.asarray(pcd.points).transpose()).to(self.device), random_file

    def _get_cad_models(self):
        """
        Returns the segmented depth point cloud
        output:
        model_pcd_torch : torch.tensor (1, 3, num_of_pts_in_pcd)
        :return:
        """
        if self.model_pcd is not None:
            return self.model_pcd
        else:
            print("WARNING: CAN'T GET POINT CLOUD BECAUSE IT IS NOT GENERATED YET")
        return None

if __name__ == "__main__":
    # #---------------------------------------------------#
    # print("Test: SE3ApolloDataset")
    #  # average of models 3 and 4
    # # mask = np.zeros(79)
    # # mask[3] = 0.5
    # # mask[4] = 0.5
    # # mask = mask.transpose()
    # # #random shapes
    # # dataset = SE3ApolloDataset(weight_mask_random=False, weight_mask=torch.from_numpy(mask))
    # dataset = SE3ApolloDataset(weight_mask_random=True)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # for i, data in enumerate(loader):
    #     print("Shapes of returned data")
    #     pc, kp, R, t, c = data
    #     print(pc.shape)
    #     print(kp.shape)
    #     print(R.shape)
    #     print(t.shape)
    #     print(c.shape)
    #     visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
    #
    #     if i >= 1:
    #         break
    # # dataset._get_cad_models()
    # dataset._get_model_keypoints()
    # #---------------------------------------------------#
    # print("Test: ApolloDepthPointCloudDataset")
    # #average of models 3 and 4
    # mask = np.zeros(79)
    # mask[3] = 0.5
    # mask[4] = 0.5
    # mask = mask.transpose()
    # #random shapes
    # dataset = ApolloDepthPointCloudDataset(weight_mask_random=False, weight_mask=torch.from_numpy(mask))
    # # dataset = ApolloDepthPointCloudDataset(weight_mask_random=True)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # for i, data in enumerate(loader):
    #     print("hi")
    #     depth_pc, kp, R, t, c = data
    #     print(depth_pc.shape)
    #     print(kp.shape)
    #     print(R.shape)
    #     print(t.shape)
    #     print(c.shape)
    #     visualize_torch_model_n_keypoints(cad_models=depth_pc, model_keypoints=kp)
    #
    #     if i >= 0:
    #         break
    # # dataset._get_cad_models()
    # dataset._get_model_keypoints()

    # ---------------------------------------------------#
    print("Test: ApolloSegPointCloudDataset")
    dataset = ApolloSegPointCloudDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(loader):
        depth_pc, _ = data
        print(depth_pc.shape)
        # visualize_torch_model_n_keypoints(cad_models=depth_pc, model_keypoints=torch.tensor([0]).repeat(1, 3, 1000))
        visualize_torch_model(cad_models=depth_pc)

        if i >= 1:
            break
    cad_models = dataset._get_cad_models()
    print("cad_models.shape", cad_models.shape)