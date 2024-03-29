"""
This writes a proposed model for expt_registration

"""


import copy
import numpy as np
import open3d as o3d
import sys
import torch
import torch.nn as nn

sys.path.append("../../")

from c3po.models.keypoint_detector import RegressionKeypoints
from c3po.models.point_set_registration import PointSetRegistration
from c3po.models.keypoint_corrector import kp_corrector_reg

from c3po.utils.ddn.node import ParamDeclarativeFunction
from c3po.utils.visualization_utils import display_results


class ProposedRegressionModel(nn.Module):
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None, local_max_pooling=False,
                 correction_flag=False, need_predicted_keypoints=False, viz_keypoint_correction=False,
                 mesh_model=None, hyper_param=None):
        super().__init__()
        """ 
        model_keypoints     : torch.tensor of shape (K, 3, N)
        cad_models          : torch.tensor of shape (K, 3, n)  
        keypoint_detector   : torch.nn.Module   : detects N keypoints for any sized point cloud input       
                                                  should take input : torch.tensor of shape (B, 3, m)
                                                  should output     : torch.tensor of shape (B, 3, N)
        """

        # Parameters
        self.class_name = class_name
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.viz_keypoint_correction = viz_keypoint_correction

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)
        self.local_max_pooling = local_max_pooling
        self.correction_flag = correction_flag
        self.need_predicted_keypoints = need_predicted_keypoints
        self.mesh_model = mesh_model
        self.hyper_param = hyper_param

        # Keypoint Detector
        if keypoint_detector == None:
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')

        elif keypoint_detector == 'pointnet':
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')

        elif keypoint_detector == 'point_transformer':
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='point_transformer')

        else:
            self.keypoint_detector = keypoint_detector(class_name=class_name, N=self.N)

        # Registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)

        # Corrector
        # self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        if self.viz_keypoint_correction:
            corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints,
                                              animation_update=True, vis=self.viz_keypoint_correction,
                                              model_mesh=self.mesh_model, class_name=self.class_name,
                                              hyper_param=self.hyper_param)
        else:
            corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)

        self.corrector = ParamDeclarativeFunction(problem=corrector_node)

    def forward(self, input_point_cloud):
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
        batch_size, _, m = input_point_cloud.shape
        device_ = input_point_cloud.device

        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        detected_keypoints = self.keypoint_detector(pc_centered)
        detected_keypoints += center

        if self.viz_keypoint_correction:
            inp = input_point_cloud.clone().detach().to('cpu')
            det_kp = detected_keypoints.clone().detach().to('cpu')
            # print("FINISHED DETECTOR")

        if not self.correction_flag:
            R, t = self.point_set_registration.forward(detected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t

            # return predicted_point_cloud, detected_keypoints, R, t, None
            if not self.need_predicted_keypoints:
                return predicted_point_cloud, detected_keypoints, R, t, None
            else:
                predicted_model_keypoints = R @ self.model_keypoints + t
                return predicted_point_cloud, detected_keypoints, R, t, None, predicted_model_keypoints

        else:
            correction = self.corrector.forward(detected_keypoints, input_point_cloud)
            corrected_keypoints = detected_keypoints + correction
            R, t = self.point_set_registration.forward(corrected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t
            # if self.viz_keypoint_correction:
                # print("FINISHED CORRECTOR")
                # print("visualizing corrected keypoints")
                # Rt_inp = predicted_point_cloud.clone().detach().to('cpu')
                # corrected_kp = corrected_keypoints.clone().detach().to('cpu')
                # display_results(inp, det_kp, inp, corrected_kp)

            # return predicted_point_cloud, corrected_keypoints, R, t, correction
            if not self.need_predicted_keypoints:
                return predicted_point_cloud, corrected_keypoints, R, t, correction
            else:
                predicted_model_keypoints = R @ self.model_keypoints + t
                return predicted_point_cloud, corrected_keypoints, R, t, correction, predicted_model_keypoints

