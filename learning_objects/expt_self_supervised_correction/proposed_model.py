"""
This writes a proposed model for expt_registration
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch3d import ops

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import sys
sys.path.append("../../")


from learning_objects.utils.ddn.node import ParamDeclarativeFunction
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import chamfer_half_distance, keypoint_error, soft_chamfer_half_distance
from learning_objects.utils.general import rotation_error, shape_error, translation_error
from learning_objects.utils.general import display_results

from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.point_set_registration import PointSetRegistration
from learning_objects.models.keypoint_corrector import kp_corrector_reg, correctorNode

# from learning_objects.models.modelgen import ModelFromShape, ModelFromShapeModule

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, DepthPC

from learning_objects.utils.ddn.node import DeclarativeLayer, ParamDeclarativeFunction

class ProposedModel2(nn.Module):
    #ToDo: RT working on this. Do not use this.
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(self, model_keypoints, cad_models, keypoint_detector=None):
        super().__init__()
        """ 
        model_keypoints     : torch.tensor of shape (K, 3, N)
        cad_models          : torch.tensor of shape (K, 3, n)  
        keypoint_detector   : torch.nn.Module   : detects N keypoints for any sized point cloud input       
                                                  should take input : torch.tensor of shape (B, 3, m)
                                                  should output     : torch.tensor of shape (B, 3, N)

        """

        # Parameters
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # Keypoint Detector
        if keypoint_detector == None:
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='point_transformer',
                                                         dim=[3, 16, 32, 64, 128])
        else:
            self.keypoint_detector = keypoint_detector

        # Registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)

        # Corrector
        #ToDo: Change to a corrector that backprops using gradient computation.
        self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)

    def forward(self, input_point_cloud, correction_flag=False):
        """
        input:
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        where
        B = batch size
        m = number of points in each point cloud

        output:
        keypoints           : torch.tensor of shape (B, 3, self.N)
        target_point_cloud  : torch.tensor of shape (B, 3, n)
        # rotation          : torch.tensor of shape (B, 3, 3)
        # translation       : torch.tensor of shape (B, 3, 1)
        # shape             : torch.tensor of shape (B, self.K, 1)

        """
        batch_size = input_point_cloud.shape[0]
        device_ = input_point_cloud.device
        detected_keypoints = self.keypoint_detector(input_point_cloud)

        if not correction_flag:
            R, t = self.point_set_registration.forward(detected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t

            return predicted_point_cloud, detected_keypoints, R, t, None

        else:
            correction = self.corrector.forward(detected_keypoints, input_point_cloud)
            corrected_keypoints = detected_keypoints + correction
            R, t = self.point_set_registration.forward(corrected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t

            return predicted_point_cloud, corrected_keypoints, R, t, correction