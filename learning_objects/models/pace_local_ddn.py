import torch
import cvxpy as cp
import pymanopt as pym
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import pymanopt

import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode, EqConstDeclarativeNode, DeclarativeLayer, ParamDeclarativeFunction
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import shape_error, translation_error, rotation_error


class AbstractSO3DeclarativeNode(AbstractDeclarativeNode):
    """
    This is a node, who's output y is a rotation matrix.

    We need to modify the gradient computation of the AbstractDeclarativeNode, and project it back to SO(3)
    """
    #ToDo: Write this!






class PACERotationLocal(AbstractSO3DeclarativeNode):
    def __init__(self, N, K, M, h):
        super().__init__()
        """
        N   : int       : number of keypoints
        K   : int       : number of cad models in the shape category
        M   : torch.tensor of shape (3N+K, 3N)  : [see eq (16)]
        h   : torch.tensor of shape (3N + K) : [see eq (16)] 
        """

        self.N = N
        self.K = K
        self.M = M
        self.h = h


    def objective(self, normalized_keypoints, y):
        """
        inputs:
        normalized_keypoints    : torch.tensor of shape (B, 3*N) [see eq (9)]
        y                       : torch.tensor of shape (B, 3*3)

        output:
        cost                    : torch.tensor of shape (B, 1)
        """
        batch_size = y.shape[0]
        rotation = y.reshape(batch_size, 3, 3)

        e = torch.eye(self.N).unsqueeze(0)
        vec = self.M.unsqueeze(0) @ torch.kron(e, rotation.transpose(-1, -2)) @ normalized_keypoints.unsqueeze(-1) + self.h     # (B, 3N+K)

        return (vec**2).sum(1).unsqueeze(-1)


    def solve(self, normalized_keypoints):
        """
        inputs:
        normalized_keypoints    : torch.tensor of shape (B, 3*N) [see eq (9)]

        outputs:
        y                       : torch.tensor of shape (B, 3*3)
        """

        batch_size = normalized_keypoints.shape[0]

        #ToDo: write the optimization

        y = torch.rand(batch_size, 3*3)
        return y