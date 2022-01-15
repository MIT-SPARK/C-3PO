"""
This implements model generation modules.

Note:
    Given shape parameters, model generation modules output a model corresponding to that shape.
"""

import torch
import torch.nn as nn
import open3d as o3d

import os
import sys
sys.path.append("../../")


class ModelFromShape():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (K, 3, n)
        model_keypoints : torch.tensor of shape (K, 3, N)

        where 
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is a point cloud of n points: (3, n)
        I am assuming that the intermediate shape can be obtained from 
            the shape parameter c and the cad models 
            by just weighted average 
        """
        self.cad_models = cad_models.unsqueeze(0)  # (1, K, 3, n)
        self.model_keypoints = model_keypoints.unsqueeze(0)  # (1, K, 3, N)


    def tsdf_to_mesh(self, avg_tsdf, visualization=False):
        """
        Use marching cubes to generate meshes from a list of TSDF volumes
        """
        vertices, triangles = libmcubes.marching_cubes(-avg_tsdf, 0)
        vertices /= self.options.resolution
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


    def average_tsdf(self, weights):
        """

        :param weights: list of numbers that sum up to 1 that represent the weighting of cad model average
        :return:
        """
        #returns an averaged tsdf
        model_paths = []
        for i in range(len(self.cad_models.shape[0])):
            model_name = ""  # cad_models.name
            model_paths.append(os.path.join(tsdf_dir, model_name + ".npy"))
        tsdf_list = self.load_tsdf(model_paths)

        assert (len(tsdf_list) == len(weights))
        avg_tsdf = np.zeros(tsdf_list[0].shape)
        weights_sum = 0
        for tsdf, weight in zip(tsdf_list, weights):
            avg_tsdf += tsdf * weight
            weights_sum += weight
        avg_tsdf /= weights_sum
        return avg_tsdf


    def load_tsdf(self, model_paths):
        """

        :param model_paths: list of model names e.g. [<path_to>/aodi-Q7-SUV.npy, <path_to>/aodi-a6.off.npy]
        :return: tsdf_list: list of tsdf models
        """
        tsdf_list = []
        for filepath in model_paths:
            tsdf = np.load(filepath)
            tsdf_list.append(tsdf)

        return tsdf_list

    def forward(self, shape):
        """
        shape: torch.tensor of shape (B, K, 1)

        where
        B = batch size
        K = number of cad models

        intermediate:
        self.cad_models: torch.tensor of shape (1, K, 3, n)

        output:
        predicted_keypoints: torch.tensor of shape (B, 3, N)
        predicted_model: torch.tensor of shape (B, 3, n)
        """
        shape = shape.unsqueeze(-1) # (B, K, 1, 1)
        # averaging model keypoints
        return torch.einsum('bkmn,ukij->bij', shape, self.model_keypoints), torch.einsum('bkmn,ukij->bij', shape, self.cad_models)


class ModelFromShapeModule(nn.Module):
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (K, 3, n)
        model_keypoints : torch.tensor of shape (K, 3, N)

        where 
        K = number of cad models
        n = number of points in each cad model

        Assumption:
        I am assuming that each CAD model is a point cloud of n points: (3, n)
        I am assuming that the intermediate shape can be obtained from 
            the shape parameter c and the cad models 
            by just weighted average 
        """

        self.model_from_shape = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)


    def forward(self, shape):
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

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    B = 10
    K = 5
    N = 8
    n = 100
    cad_models = torch.rand(K, 3, n).to(device=device)
    model_keypoints = cad_models[:, :, 0:N]

    shape = torch.rand(B, K, 1).to(device=device)
    shape = shape/shape.sum(1).unsqueeze(1)

    shape_to_model_fn = ModelFromShapeModule(cad_models=cad_models, model_keypoints=model_keypoints).to(device=device)
    keypoints, model = shape_to_model_fn(shape=shape)

    print("cad models have shape: ", cad_models[0, :, :].shape)
    print("output model has shape: ", model.shape)
    print("output keypoints has shape: ", keypoints.shape)