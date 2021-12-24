import torch
import torch.nn as nn
import cvxpy as cp

import os
import sys
sys.path.append("../../")


from learning_objects.utils.ddn.node import AbstractDeclarativeNode

from learning_objects.utils.general import chamfer_half_distance, soft_chamfer_half_distance
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import shape_error, translation_error, rotation_error


from learning_objects.models.background_segmentor import SegmentBackground
from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.pace import PACE, PACEmodule, PACEddn
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.models.keypoint_corrector import PACEwKeypointCorrectionModule



class ProposedModel(nn.Module):
    def __init__(self, model_keypoints, cad_models, num_keypoints=44, keypoint_type='regression',
                 keypoint_method='point_transformer', weights=None, lambda_constant=None, keypoint_correction=False):
        super().__init__()
        """
        num_keypoints   : int
        keypoint_type   : 'heatmap' or 'regression'
        keypoint_method : 'pointnet' or 'point_transformer'
        model_keypoints : torch.tensor of shape (K, 3, N)
        cad_models      : torch.tensor of shape (K, 3, n)
        weights         : torch.tensor of shape (N, 1) or None
        lambda_constant : torch.tensor of shape (1, 1) or None
        keypoint_correction     : True or False
        """
        self.num_keypoints = num_keypoints
        self.keypoint_type = keypoint_type
        self.keypoint_correction = keypoint_correction
        self.keypoint_method = keypoint_method
        self.model_keypoints = model_keypoints
        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)
        self.cad_models = cad_models

        self.weights = weights
        if weights == None:
            self.weights = torch.ones(self.N, 1)

        self.lambda_constant = lambda_constant
        if lambda_constant == None:
            self.lambda_constant = torch.sqrt(torch.tensor([self.N/self.K]))

        # Keypoint detector
        self.keypoint_detector = RegressionKeypoints(k=num_keypoints, method=keypoint_type)
        if keypoint_type=='heatmap':
            self.keypoint_detector = HeatmapKeypoints(k=num_keypoints, method=keypoint_type)

        # PACE
        self.PACE = PACEmodule(weights=self.weights, model_keypoints=self.model_keypoints,
                               lambda_constant=self.lambda_constant)
        if self.keypoint_correction:
            self.PACEwKeypointCorrection = PACEwKeypointCorrectionModule(model_keypoints=self.model_keypoints,
                                                                         cad_models=self.cad_models,
                                                                         weights=self.weights,
                                                                         lambda_constant=self.lambda_constant)

        # Model Generator
        self.generate_model = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)


    def forward(self, input_point_cloud, train=True):
        """
        input:
        input_point_cloud: torch.tensor of shape (B, 3, m)

        where
        B = batch size
        m = number of points in each point cloud

        output:
        keypoints: torch.tensor of shape (B, 3, N)
        predicted_model: torch.tensor of shape (B, 3, n)
        """

        detected_keypoints = torch.transpose(
            self.keypoint_detector(point_cloud=torch.transpose(input_point_cloud, -1, -2)), -1, -2)

        if train or (not self.keypoint_correction):
            # During training or when not using keypoint_correction
            #           keypoints = detected_keypoints
            #
            R, t, c = self.PACE(y=detected_keypoints)
            keypoints = detected_keypoints
        else:
            # During testing and when using keypoint_correction
            #           keypoints = corrected_keypoints from the bi-level optimization.
            #
            R, t, c, correction, keypoints = self.PACEwKeypointCorrection.forward(
                input_point_cloud=input_point_cloud, detected_keypoints=detected_keypoints)

        target_keypoints, target_point_cloud = self.generate_model(shape=c)
        target_keypoints = R @ target_keypoints + t
        target_point_cloud = R @ target_point_cloud + t

        return keypoints, target_keypoints, target_point_cloud












if __name__ == '__main__':

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)


    # Test: SegmentBackground()
    print('Test: SegmentBackground()')
    pc = torch.rand(4, 1000, 9)
    pc = pc.to(device)
    print(pc.is_cuda)

    segback = SegmentBackground().to(device=device)

    pc_out, num_in, num_out = segback(pointcloud=pc)

    print(pc_out.size())
    print(num_in.size())
    print(num_out.size())
    print('-'*20)

    # Test: HeatmapKeypoints() with method='pointnet'
    print('Test: HeatmapKeypoints() with method=\'pointnet\'')

    pc = torch.rand(4, 1000, 11)
    pc = pc.to(device=device)
    kp = HeatmapKeypoints(method='pointnet', k=63).to(device=device)
    # kp = HeatmapKeypoints(method='point_transformer', dim=[8, 32, 64], k=63).to(device=device)
    y = kp(pointcloud=pc)
    print(y.size())
    print('-' * 20)

    # Test: HeatmapKeypoints() with method='point_transformer'
    print('Test: HeatmapKeypoints() with method=\'point_transformer\'')

    pc = torch.rand(4, 1000, 6)
    pc = pc.to(device=device)
    # kp = HeatmapKeypoints(method='pointnet', k=63).to(device=device)
    kp = HeatmapKeypoints(method='point_transformer', dim=[3, 16, 24], k=63).to(device=device)
    y = kp(pointcloud=pc)
    print(y.size())
    print('-' * 20)


    # Test: PACE Module
    print('Test: PACE()')
    N = 10
    K = 4
    n = 40
    weights = torch.rand(N, 1)
    model_keypoints = torch.rand(K, 3, N)
    lambda_constant = torch.tensor([1.0])
    cad_models = torch.rand(K, 3, n)

    pace_model = PACEmodule(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant).to(device=device)

    B = 20
    keypoints, rotations, translations, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints)
    keypoints.requires_grad = True
    rot_est, trans_est, shape_est = pace_model(keypoints)

    er_shape = shape_error(shape, shape_est)
    er_trans = translation_error(translations, trans_est)
    er_rot = rotation_error(rotations, rot_est)

    print("rotation error: ", er_rot.mean())
    print("translation error: ", er_trans.mean())
    print("shape error: ", er_shape.mean())

    loss = er_shape.mean() + er_trans.mean() + er_rot.mean()
    loss.backward()

    print(keypoints.grad)




    

