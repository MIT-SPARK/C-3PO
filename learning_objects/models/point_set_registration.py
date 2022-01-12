"""
This code implements outlier-free point set registration as torch function

"""
import torch
import cvxpy as cp
import pymanopt as pym
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
from pytorch3d import transforms

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode, EqConstDeclarativeNode, DeclarativeLayer, ParamDeclarativeFunction
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import shape_error, translation_error, rotation_error




def wahba(source_points, target_points, device_=None):
    """
    source_points: torch.tensor of shape (B, 3, N)
    target_points: torch.tensor of shape (B, 3, N)

    where
        B = batch size
        N = number of points in each point set
    """
    batch_size = source_points.shape[0]

    if device_==None:
        device_ = source_points.device

    mat = target_points @ source_points.transpose(-1, -2)   # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(mat)

    D = torch.eye(3).to(device=device_)     # (3, 3)
    D = D.unsqueeze(0)                      # (1, 3, 3)
    D = D.repeat(batch_size, 1, 1)          # (B, 3, 3)

    D[:, 2, 2] = torch.linalg.det(U)*torch.linalg.det(Vh)

    return U @ D @ Vh   # (B, 3, 3)





def point_set_registration(source_points, target_points, weights=None, device_=None):
    """
    source_points: torch.tensor of shape (B, 3, N)
    target_points: torch.tensor of shape (B, 3, N)
        
    where
        B = batch size
        N = number of points in each point set
    """

    batch_size, d, N = source_points.shape
    if device_==None:
        device_ = source_points.device

    if weights==None:
        weights = torch.ones((1, N), device=device_)

    source_points_ave = torch.einsum('bdn,ln->bd', source_points, weights)/weights.sum()    # (B, 3)
    target_points_ave = torch.einsum('bdn,ln->bd', target_points, weights) / weights.sum()  # (B, 3)

    # getting the rotation
    source_points_centered = source_points - source_points_ave.unsqueeze(-1)    # (B, 3, N)
    target_points_centered = target_points - target_points_ave.unsqueeze(-1)    # (B, 3, N)

    source_points_centered = torch.einsum('bdn,ln->bdn', source_points_centered, weights)   # (B, 3, N)
    target_points_centered = torch.einsum('bdn,ln->bdn', target_points_centered, weights)   # (B, 3, N)

    rotation = wahba(source_points=source_points_centered, target_points=target_points_centered, device_=device_)

    # getting the translation
    translation = target_points_ave.unsqueeze(-1) - rotation @ source_points_ave.unsqueeze(-1)

    return rotation, translation




if __name__ == '__main__':

    B = 10
    N = 20
    d = 3

    source_points = torch.rand(B, d, N)
    rotation = transforms.random_rotations(B)

    target_points = rotation @ source_points
    target_points += 0.01*torch.rand(size=target_points.shape)


    print('-' * 40)
    print("Testing wahba()")
    print('-' * 40)
    rotation_est = wahba(source_points=source_points - source_points.mean(-1).unsqueeze(-1),
                                      target_points=target_points-target_points.mean(-1).unsqueeze(-1))

    print("Output shape: ", rotation_est.shape)

    err = rotation_error(rotation, rotation_est)
    print("Rotation error: ", err.mean())



    print('-'*40)
    print("Testing point_set_registration()")
    print('-' * 40)

    B = 10
    N = 20
    d = 3

    source_points = torch.rand(B, d, N)
    rotation = transforms.random_rotations(B)
    translation = torch.rand(B, d, 1)

    target_points = rotation @ source_points + translation
    target_points += 0.01*torch.rand(size=target_points.shape)


    rotation_est, translation_est = point_set_registration(source_points=source_points, target_points=target_points)

    print("Output rotation shape: ", rotation.shape)
    print("Output translation shape: ", translation.shape)

    err_rot = rotation_error(rotation, rotation_est)
    err_trans = translation_error(translation, translation_est)
    print("Rotation error: ", err_rot.mean())
    print("Translation error: ", err_trans.mean())


