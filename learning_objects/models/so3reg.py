import torch
import cvxpy as cp
import pymanopt as pym
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode


class so3reg(AbstractDeclarativeNode):
    def __init__(self, y_target):
        super().__init__(eps=, gamma=, chunk_size=)
        """
        y_target: torch.tensor of shape (1, 3, N)
        """

        self.y_target = y_target


    def objective(self, y, R):
        """
        R: torch.tensor of shape (B, 3, 3)
        y: torch.tensor of shape (B, 3, N)
        """
        return torch.norm(self.y_target - torch.matmul(R, y), dim=1, p=2).mean(dim=1).unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)

    def gradient(self, y, R):
        """
        R: torch.tensor of shape (B, 3, 3)
        y: torch.tensor of shape (B, 3, N)
        """



    def solve(self, y):
        """
        y: torch.tensor of shape (B, 3, N)

        Note:
        This solves the optimization problem
        minimize ||self.y_target - R y ||^2
        over R in SO(3)
        """

