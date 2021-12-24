"""
This implements PACE with keypoint correction as a function and a torch.nn.Module.

"""
import torch
import torch.nn as nn
import cvxpy as cp

import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode
from learning_objects.utils.general import chamfer_half_distance, soft_chamfer_half_distance
from learning_objects.models.pace import PACEmodule, PACE
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.utils.general import generate_random_keypoints



class PACEwKeypointCorrection():
    """
    This implements PACE + Keypoint Correction.
    """
    def __init__(self, model_keypoints, cad_models, weights=None, lambda_constant=None):
        super().__init__()
        """ 
        Inputs:
        model_keypoints : torch.tensor of shape (K, 3, N) 
        cad_models      : torch.tensor of shape (K, 3, n)
        weights         : torch.tensor of shape (N, 1) or None
        lambda_constant : torch.tensor of shape (1, 1) or None

        where 
        N = number of semantic keypoints
        K = number of cad models
        n = number of points in each cad model 

        Assumption:
        We are assuming that each CAD model is a point cloud of n points: (3, n)
        We are assuming that given K point clouds (K, 3, n) and a shape parameter c (K, 1) 
            there is a differentiable function "ModelFromShape(nn.Module)" which 
            can be used to generate intermediate shape from the cad models and the shape 
            parameter

        Note:
        The module implements keypoint correction optimization.
        """
        self.N = model_keypoints.shape[-1]
        self.K = model_keypoints.shape[0]
        self.model_keypoints = model_keypoints  # (K, 3, N)
        self.cad_models = cad_models  # (K, 3, n)
        self.weights = weights  # (N, 1)
        self.lambda_constant = lambda_constant  # (1, 1)
        self.theta = torch.tensor([10.0])  # weight for keypoint component of the cost function

        # We use this model for computing objective function, and the backward pass.
        self.pace_model = PACEmodule(weights=self.weights, model_keypoints=self.model_keypoints,
                                     lambda_constant=self.lambda_constant)
        # self.pace_model = PACE(weights=self.weights, model_keypoints=self.model_keypoints,
        #                              lambda_constant=self.lambda_constant)
        self.model_from_shape = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)

    def forward(self, input_point_cloud, detected_keypoints):
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        detected_keypoints  : torch.tensor of shape (B, 3, N)

        where
        m = number of points in the input point cloud
        N = number of keypoints
        B = batch size
        K = number of cad models in the shape category

        output:
        rotation            : torch.tensor of shape (B, 3, 3)
        translation         : torch.tensor of shape (B, 3, 1)
        shape               : torch.tensor of shape (B, K, 1)
        correction          : torch.tensor of shape (B, 3, N)
        corrected_keypoints : torch.tensor of shape (B, 3, N)
        """

        learning_rate = 0.5
        num_steps = 20

        correction = torch.rand(detected_keypoints.shape, requires_grad=True)
        history = []
        for iter in range(num_steps):
            R, t, c = self.pace_model(y=detected_keypoints + correction)
            _, model = self.model_from_shape(shape=c)
            model = R @ model + t
            loss = self._loss(input_point_cloud=input_point_cloud, model_point_cloud=model)
            loss.backward()
            correction_new = correction - learning_rate * correction.grad
            correction = correction_new.detach().requires_grad_(True)
            history.append(loss.mean(0))

        corrected_keypoints = detected_keypoints + correction
        rotation, translation, shape = self.pace_model(y=corrected_keypoints)

        return rotation, translation, shape, correction, corrected_keypoints

        # ToDo:
        # A problem: I don't think we can use backprop inside a forward prop. This may cause confusion during training.

    def _loss(self, input_point_cloud, model_point_cloud):
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        model_point_cloud   : torch.tensor of shape (B, 3, n)

        where
        m = number of points in the input point cloud
        n = number of points in the model point cloud
        B = batch size

        output:
        loss = half chamfer : torch.tensor of shape (B, 1)
        """

        return chamfer_half_distance(X=input_point_cloud, Y=model_point_cloud)


class PACEwKeypointCorrectionModule(nn.Module):
    """
    This implements PACE + Keypoint Correction Module.
    """
    def __init__(self, model_keypoints, cad_models, weights=None, lambda_constant=None):
        super().__init__()
        """ 
        Inputs:
        model_keypoints : torch.tensor of shape (K, 3, N) 
        cad_models      : torch.tensor of shape (K, 3, n)
        weights         : torch.tensor of shape (N, 1) or None
        lambda_constant : torch.tensor of shape (1, 1) or None

        where 
        N = number of semantic keypoints
        K = number of cad models
        n = number of points in each cad model 
        
        Assumption:
        We are assuming that each CAD model is a point cloud of n points: (3, n)
        We are assuming that given K point clouds (K, 3, n) and a shape parameter c (K, 1) 
            there is a differentiable function "ModelFromShape(nn.Module)" which 
            can be used to generate intermediate shape from the cad models and the shape 
            parameter

        Note:
        The module implements keypoint correction optimization.
        """
        self.N = model_keypoints.shape[-1]
        self.K = model_keypoints.shape[0]
        self.model_keypoints = model_keypoints      # (K, 3, N)
        self.cad_models = cad_models                # (K, 3, n)
        self.weights = weights                      # (N, 1)
        self.lambda_constant = lambda_constant      # (1, 1)
        self.theta = torch.tensor([10.0])  # weight for keypoint component of the cost function

        # We use this model for computing objective function, and the backward pass.
        self.pace_model = PACEmodule(weights=self.weights, model_keypoints=self.model_keypoints,
                                     lambda_constant=self.lambda_constant)
        # self.pace_model = PACE(weights=self.weights, model_keypoints=self.model_keypoints,
        #                              lambda_constant=self.lambda_constant)
        self.model_from_shape = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)



    def forward(self, input_point_cloud, detected_keypoints):
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        detected_keypoints  : torch.tensor of shape (B, 3, N)

        where
        m = number of points in the input point cloud
        N = number of keypoints
        B = batch size
        K = number of cad models in the shape category

        output:
        rotation            : torch.tensor of shape (B, 3, 3)
        translation         : torch.tensor of shape (B, 3, 1)
        shape               : torch.tensor of shape (B, K, 1)
        correction          : torch.tensor of shape (B, 3, N)
        corrected_keypoints : torch.tensor of shape (B, 3, N)
        """

        learning_rate = 0.5
        num_steps = 20

        correction = torch.rand(detected_keypoints.shape, requires_grad=True)
        history = []
        for iter in range(num_steps):
            R, t, c = self.pace_model(y=detected_keypoints+correction)
            _, model = self.model_from_shape(shape=c)
            model = R @ model + t
            loss = self._loss(input_point_cloud=input_point_cloud, model_point_cloud=model)
            loss.backward()
            correction_new = correction - learning_rate * correction.grad
            correction = correction_new.detach().requires_grad_(True)
            history.append(loss.mean(0))

        corrected_keypoints = detected_keypoints + correction
        rotation, translation, shape = self.pace_model(y=corrected_keypoints)

        return rotation, translation, shape, correction, corrected_keypoints

        #ToDo:
        # A problem: I don't think we can use backprop inside a forward prop. This may cause confusion during training.


    def _loss(self, input_point_cloud, model_point_cloud):
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        model_point_cloud   : torch.tensor of shape (B, 3, n)

        where
        m = number of points in the input point cloud
        n = number of points in the model point cloud
        B = batch size

        output:
        loss = half chamfer : torch.tensor of shape (B, 1)
        """

        return chamfer_half_distance(X=input_point_cloud, Y=model_point_cloud)



if __name__ == "__main__":
    print("test")