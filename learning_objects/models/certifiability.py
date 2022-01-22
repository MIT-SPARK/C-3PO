"""
This code defines a metric of (epsilon, delta)-certifiability. Given two registered, shape aligned point clouds
pc and pc_ it determines if the registration + shape alignment is certifiable or not.

"""

import torch
from pytorch3d import ops

import os
import sys
sys.path.append("../../")


def chamfer_loss(pc, pc_, pc_padding=None):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in

    output:
    loss    : (B, 1)
    """

    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)
        # pc_padding = torch.zeros(batch_size, n).to(device=device_)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)
    loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)

def confidence(pc, pc_):
    """
    inputs:
    pc  : input point cloud : torch.tensor of shape (B, 3, n)
    pc_ : model point cloud : torch.tensor of shape (B, 3, m)

    output:
    confidence  : torch.tensor of shape (B, 1)
    """

    return torch.exp(-chamfer_loss(pc, pc_))
    # return chamfer_distance(pc, pc_)

def confidence_kp(kp, kp_):
    """
    inputs:
    kp  : input point cloud : torch.tensor of shape (B, 3, n)
    kp_ : model point cloud : torch.tensor of shape (B, 3, m)

    output:
    confidence  : torch.tensor of shape (B, 1)

    """

    return torch.exp(-((kp-kp_)**2).sum(dim=1).max(dim=1)[0].unsqueeze(-1))


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
    def __init__(self, epsilon, delta, radius=0.3):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.radius = radius


    def forward(self, X, Z, kp=None, kp_=None):
        """
        inputs:
        X   : input :   torch.tensor of shape (B, 3, n)
        Z   : model :   torch.tensor of shape (B, 3, m)
        kp  : detected/correct_keypoints    : torch.tensor of shape (B, 3, N)
        kp_ : model keypoints               : torch.tensor of shape (B, 3, N)

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input X with the model Z
        """

        confidence_ = confidence(X, Z)
        completeness_ = completeness(X, Z)

        if kp==None or kp_==None:
            confidence_kp_ = 100000*torch.ones_like(confidence_)
            print("Certifiability is not taking keypoint errors into account.")
        else:
            confidence_kp_ = confidence_kp(kp, kp_)

        return (confidence_ >= self.epsilon) & (confidence_kp_ >= self.epsilon) & (completeness_ >= self.delta), completeness_

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