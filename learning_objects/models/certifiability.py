"""
This code defines a metric of (epsilon, delta)-certifiability. Given two registered, shape aligned point clouds
pc and pc_ it determines if the registration + shape alignment is certifiable or not.

"""

import torch
from pytorch3d import ops

import os
import sys
sys.path.append("../../")

from learning_objects.utils.general import pos_tensor_to_o3d
from learning_objects.utils.general import chamfer_distance, chamfer_half_distance



def confidence(pc, pc_):
    """
    inputs:
    pc  : input point cloud : torch.tensor of shape (B, 3, n)
    pc_ : model point cloud : torch.tensor of shape (B, 3, m)

    output:
    confidence  : torch.tensor of shape (B, 1)
    """

    return torch.exp(-chamfer_half_distance(pc, pc_))
    # return chamfer_distance(pc, pc_)


def completeness(pc, pc_, radius=0.3):
    """
    inputs:
    pc  : input point cloud : torch.tensor of shape (B, 3, n)
    pc_ : model point cloud : torch.tensor of shape (B, 3, m)

    output:
    fraction    : torch.tensor of shape (B, 1)
    """

    # fraction of points in pc_ that have a radius-distance neighbor in pc
    sq_dist, _, _ = ops.knn_points(torch.transpose(pc_, -1, -2), torch.transpose(pc, -1, -2), K=1)

    sq_dist = sq_dist.squeeze(-1)
    dist = torch.sqrt(sq_dist)
    fraction = ((dist <= radius).int().float().sum(-1)/dist.shape[-1]).unsqueeze(-1)

    return fraction


class certifiability():
    def __init__(self, epsilon, delta):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.radius = 0.3


    def forward(self, X, Z):
        """
        inputs:
        X   : input :   torch.tensor of shape (B, 3, n)
        Z   : model :   torch.tensor of shape (B, 3, m)

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input X with the model Z
        """

        confidence_ = confidence(X, Z)
        completeness_ = completeness(X, Z)

        return (confidence_ >= self.epsilon) & (completeness_ >= self.delta), completeness_

    def forward_with_distances(self, sq_dist_XZ, sq_dist_ZX):
        """
        inputs:
        sq_dist_XZ  : torch.tensor of shape (B, n, 1)   : sq. distance from every point in X to the closest point in Z
        sq_dist_ZX  : torch.tensor of shape (B, m, 1)   : sq. distance from every point in Z to the closest point in X

        where:
            X   : input point cloud
            Z   : model point cloud
            n   : number of points in X
            m   : number of points in Z
            B   : batch size

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input X with the model Z
        """

        confidence_ = torch.exp(-sq_dist_XZ.mean(dim=1))

        sq_dist = sq_dist_ZX.squeeze(-1)
        dist = torch.sqrt(sq_dist)
        completeness_ = ((dist <= self.radius).int().float().sum(-1)/dist.shape[-1]).unsqueeze(-1)

        return (confidence_ >= self.epsilon) & (completeness_ >= self.delta), completeness_





if __name__ == "__main__":

    print("test")

    pc = torch.rand(10, 3, 5)
    pc_ = pc + 0.1*torch.rand(size=pc.shape)

    epsilon = 0.2
    delta = 0.5
    certify = certifiability(epsilon=epsilon, delta=delta)
    cert, comp = certify.forward(pc, pc_)

    print(cert)
    print(comp)