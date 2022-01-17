"""
This implements model generation modules.

Note:
    Given shape parameters, model generation modules output a model corresponding to that shape.
"""

import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
import os
import sys
import csv
sys.path.append("../../")
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api','utils'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','car_averaging'))

from car_instance.car_models import *

import libmcubes


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

class ModelFromShapeApollo():
    def __init__(self, num_of_points):
        """
        :param num_of_points: number of points in the point cloud

        cad_models      : torch.tensor of shape (K, 3, n)
        model_keypoints : torch.tensor of shape (K, 3, N)

        where
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is a point cloud of n points: (3, n)
        """
        super().__init__()
        self.K = APOLLOSCAPE_DATASET_SIZE
        self.n = num_of_points

    def generate_shape_mask(self, max_shapes = 3):
        """
        Generates a mask over all models that defines weights of weighted average of model shapes.
        Assumes the weights add up to 1
        :param max_shapes:
        :return:
        """
        num_shapes = np.random.randint(1, max_shapes+1)
        mask_length = len(car_id2name.keys())
        mask = np.zeros(mask_length)
        selected_ids = []
        #randomly sample up to max_sample shapes, rejecting if in ignore
        while len(selected_ids) != num_shapes:
            sample = np.random.randint(0, APOLLOSCAPE_DATASET_SIZE)
            if car_id2name[sample].name not in MODEL_NAMES_TO_IGNORE:
                selected_ids.append(sample)

        samples = np.random.randint(1, 4, num_shapes)
        normalized_weights = iter(samples/np.sum(samples))
        for id in selected_ids:
            mask[id] = next(normalized_weights)
        print("mask", mask)
        assert np.abs(np.sum(np.asarray(mask))-1) <= 1e-3
        return mask

    def cluster_mesh(self, o3d_mesh):
        """
        Use clustering to remove artifacts and keep the largest component of the mesh

        :param o3d_mesh:
        :return:
        """
        triangle_clusters, cluster_n_triangles, cluster_area = (o3d_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        o3d_mesh.remove_triangles_by_mask(triangles_to_remove)

        return o3d_mesh


    def tsdf_to_mesh(self, avg_tsdf, visualization=False):
        """
        Use marching cubes to generate a mesh from a TSDF
        """
        vertices, triangles = libmcubes.marching_cubes(-avg_tsdf, 0)
        vertices /= 256 #self.options.resolution
        vertices -= 0.5
        avg_mesh = {"vertices": vertices, "triangles": triangles}

        if visualization:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(avg_mesh['vertices'])
            o3d_mesh.triangles = o3d.utility.Vector3iVector(avg_mesh['triangles'])
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.compute_triangle_normals()
            o3d.visualization.draw_geometries([o3d_mesh])
        return avg_mesh

    def average_kpts(self, kpt_list, weights):
        assert (len(kpt_list) == len(weights))
        avg_kpts = np.zeros(kpt_list[0].shape)
        weights_sum = 0
        for kpts, weight in zip(kpt_list, weights):
            avg_kpts += kpts * weight
            weights_sum += weight
        avg_kpts /= weights_sum
        return avg_kpts

    def average_tsdf(self, tsdf_list, weights):
        """
        :param tsdf_list: list of loaded tsdfs
        :param weights: weights to average the tsdfs
        :return:
        """

        assert (len(tsdf_list) == len(weights))
        avg_tsdf = np.zeros(tsdf_list[0].shape)
        weights_sum = 0
        for tsdf, weight in zip(tsdf_list, weights):
            avg_tsdf += tsdf * weight
            weights_sum += weight
        avg_tsdf /= weights_sum
        return avg_tsdf

    def load_scaled_kpts(self, model_paths):
        """

        :param model_paths: list of model paths
        :return: kpt_list: np.array of xyz keypoints of shape (66, 3)
        """
        kpt_list = []
        for path in model_paths:
            # add keypoints
            keypoints_xyz = []
            with open(os.path.join(PATH_TO_SCALED_KEYPOINTS, os.path.basename(path).split('.')[0] + '.csv'), newline='') as f:
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

            kpt_list.append(keypoints_xyz)

        return kpt_list


    def load_tsdf(self, model_paths):
        """

        :param model_paths: list of model paths e.g. [<path_to>/aodi-Q7-SUV.npy, <path_to>/aodi-a6.off.npy]
        :return: tsdf_list: list of tsdf models
        """
        tsdf_list = []
        for filepath in model_paths:
            tsdf = np.load(filepath)
            tsdf_list.append(tsdf)

        return tsdf_list

    def get_model_paths(self, ids):
        model_paths = []
        for id in ids:
            model_path = os.path.join(PATH_TO_TSDFS, car_id2name[id].name + '.npy')
            model_paths.append(model_path)
        return model_paths

    def forward(self, shape=None):
        """
        shape: torch.tensor of shape (B, K, 1)

        where
        B = batch size
        K = number of cad models

        intermediate:
        self.cad_models: torch.tensor of shape (1, K, 3, n)

        output:
        keypoints: torch.tensor of shape (B=1, 3, N)
        model: torch.tensor of shape (B=1, 3, n)
        weight_mask: torch.tensor of shape (1, K, 1)
        """
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if shape is not None:
            mask = np.asarray(shape.cpu()).flatten()
            print(mask)
            print(mask.shape)
        else:
            mask = self.generate_shape_mask()
        ids = np.nonzero(mask)[0]
        print("ids:", ids)
        model_paths = self.get_model_paths(ids)
        print(model_paths)
        tsdf_list = self.load_tsdf(model_paths)

        #average tsdfs
        weights = mask[mask != 0]
        print("weights", weights)
        avg_tsdf = self.average_tsdf(tsdf_list, weights)

        avg_mesh = self.tsdf_to_mesh(avg_tsdf)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(avg_mesh['vertices'])
        o3d_mesh.triangles = o3d.utility.Vector3iVector(avg_mesh['triangles'])
        o3d_mesh.compute_vertex_normals()
        final_mesh = self.cluster_mesh(o3d_mesh)
        final_pcd = final_mesh.sample_points_uniformly(self.n)

        kpt_list = self.load_scaled_kpts(model_paths)
        avg_kpts = self.average_kpts(kpt_list, weights)
        keypoint_markers = []
        for xyz in avg_kpts:
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.005)
            kpt_mesh.translate(xyz)
            kpt_mesh.paint_uniform_color([0.8, 0.0, 0.0])
            keypoint_markers.append(kpt_mesh)
        o3d.visualization.draw_geometries([o3d_mesh] + keypoint_markers)
        avg_kpts_tensor = torch.from_numpy(np.asarray([avg_kpts.transpose()])).to(device=device)
        avg_cad_model = torch.from_numpy(np.asarray([np.asarray(final_pcd.points).transpose()])).to(device=device)

        return avg_kpts_tensor, avg_cad_model, torch.from_numpy(mask.transpose()).to(device=device)


class ModelFromShapeApolloModule(nn.Module):
    def __init__(self, num_of_points):
        """

        :param num_of_points: number of points in the point cloud

        cad_models      : torch.tensor of shape (K, 3, n)
        model_keypoints : torch.tensor of shape (K, 3, N)

        where
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is a point cloud of n points: (3, n)

        """
        super().__init__()



        self.model_from_shape = ModelFromShapeApollo(num_of_points=num_of_points)


    def forward(self, shape=None):
        """
        shape: torch.tensor of shape (B, K, 1)

        where
        B = batch size
        K = number of cad models

        intermediate:
        self.cad_models: torch.tensor of shape (1, K, 3, n)

        output:
        keypoints: torch.tensor of shape (B, 3, N)
        model: torch.tensor of shape (B, 3, n)
        """

        return self.model_from_shape.forward(shape=shape)




if __name__ == "__main__":
    """Important Note: Only generates one averaged model at a time"""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    K = 79 #79 models
    n = 100

    shape = torch.rand(1, K).to(device=device)

    shape_to_model_fn = ModelFromShapeApolloModule(num_of_points=n).to(device=device)
    # keypoints, model, mask = shape_to_model_fn(shape=shape)
    keypoints, model, mask = shape_to_model_fn()


    print("output model has shape: ", model.shape)
    print("output keypoints has shape: ", keypoints.shape)
    print("output mask has shape: ", mask.shape)