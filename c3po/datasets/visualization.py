
import copy
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import sys

sys.path.append("../../")
from c3po.utils.visualization_utils import display_two_pcs, visualize_model_n_keypoints, \
    visualize_torch_model_n_keypoints, display_results

from c3po.datasets.shapenet import get_model_and_keypoints
import c3po.utils.general as gu

VISPOSE = {
    'cap':  (torch.tensor([[[-0.7497,  0.0000, -0.6618],
                            [ 0.0000,  1.0000,  0.0000],
                            [ 0.6618,  0.0000, -0.7497]]]),
             torch.tensor([[[0.4728],
                            [0.0693],
                            [0.6590]]]))
}


class DepthPCVis(torch.utils.data.Dataset):
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