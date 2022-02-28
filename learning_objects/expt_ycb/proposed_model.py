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

from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.point_set_registration import PointSetRegistration
from learning_objects.models.keypoint_corrector import kp_corrector_reg

from learning_objects.models.point_transformer import kNN_torch, index_points

from learning_objects.utils.ddn.node import ParamDeclarativeFunction
from learning_objects.utils.general import display_results


# # Proposed Model
# class ProposedModel(nn.Module):
#     """
#     This is depreciated. Use either ProposedRegressionModel() or ProposedHeatmapModel()
#     """
#     """
#     Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation
#
#     Returns:
#         predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
#         predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
#     """
#
#     def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None,
#                  use_pretrained_regression_model=False, keypoint_detector_type='regression'):
#         super().__init__()
#         """
#         model_keypoints     : torch.tensor of shape (K, 3, N)
#         cad_models          : torch.tensor of shape (K, 3, n)
#         keypoint_detector   : torch.nn.Module   : detects N keypoints for any sized point cloud input
#                                                   should take input : torch.tensor of shape (B, 3, m)
#                                                   should output     : torch.tensor of shape (B, 3, N)
#
#         keypoint_detector_type  : 'regression' or 'heatmap'
#
#         """
#
#         # Parameters
#         self.class_name = class_name
#         self.model_keypoints = model_keypoints
#         self.cad_models = cad_models
#         self.device_ = self.cad_models.device
#         self.viz_keypoint_correction = False
#         self.use_pretrained_regression_model = use_pretrained_regression_model
#         self.keypoint_detector_type = keypoint_detector_type
#
#         self.N = self.model_keypoints.shape[-1]  # (1, 1)
#         self.K = self.model_keypoints.shape[0]  # (1, 1)
#
#         # Keypoint Detector
#         if keypoint_detector_type == 'regression':
#             if keypoint_detector == None:
#                 self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')
#
#             elif keypoint_detector == 'pointnet':
#                 self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')
#
#             elif keypoint_detector == 'point_transformer':
#                 self.keypoint_detector = RegressionKeypoints(N=self.N, method='point_transformer')
#
#             else:
#                 self.keypoint_detector = keypoint_detector(class_name=class_name, N=self.N)
#
#         elif keypoint_detector_type == 'heatmap':
#             if keypoint_detector == None:
#                 self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')
#
#             elif keypoint_detector == 'pointnet':
#                 self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')
#
#             elif keypoint_detector == 'point_transformer':
#                 raise NotImplementedError
#
#             else:
#                 raise NotImplementedError
#
#         else:
#             raise NotImplementedError
#
#
#
#         # Registration
#         self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)
#
#         # Corrector
#         # self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
#         corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
#         self.corrector = ParamDeclarativeFunction(problem=corrector_node)
#
#     def forward(self, input_point_cloud, correction_flag=False):
#         """
#         input:
#         input_point_cloud   : torch.tensor of shape (B, 3, m)
#
#         where
#         B = batch size
#         m = number of points in each point cloud
#
#         output:
#         keypoints           : torch.tensor of shape (B, 3, self.N)
#         target_point_cloud  : torch.tensor of shape (B, 3, n)
#         # rotation          : torch.tensor of shape (B, 3, 3)
#         # translation       : torch.tensor of shape (B, 3, 1)
#         # shape             : torch.tensor of shape (B, self.K, 1)
#
#         """
#         batch_size, _, m = input_point_cloud.shape
#         device_ = input_point_cloud.device
#
#         num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
#         num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
#         num_nonzero_pts = m - num_zero_pts
#         num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)
#
#         center = torch.sum(input_point_cloud, dim=-1)/num_nonzero_pts
#         center = center.unsqueeze(-1)
#         pc_centered = input_point_cloud - center
#         detected_keypoints = self.keypoint_detector(pc_centered)
#         detected_keypoints += center
#
#         if self.viz_keypoint_correction:
#             inp = input_point_cloud.clone().detach().to('cpu')
#             det_kp = detected_keypoints.clone().detach().to('cpu')
#             # print("FINISHED DETECTOR")
#
#         if not correction_flag:
#             R, t = self.point_set_registration.forward(detected_keypoints)
#             predicted_point_cloud = R @ self.cad_models + t
#
#             return predicted_point_cloud, detected_keypoints, R, t, None
#
#         else:
#             correction = self.corrector.forward(detected_keypoints, input_point_cloud)
#             corrected_keypoints = detected_keypoints + correction
#             R, t = self.point_set_registration.forward(corrected_keypoints)
#             predicted_point_cloud = R @ self.cad_models + t
#             if self.viz_keypoint_correction:
#                 # print("FINISHED CORRECTOR")
#                 print("visualizing corrected keypoints")
#                 Rt_inp = predicted_point_cloud.clone().detach().to('cpu')
#                 corrected_kp = corrected_keypoints.clone().detach().to('cpu')
#                 display_results(inp, det_kp, inp, corrected_kp)
#
#             return predicted_point_cloud, corrected_keypoints, R, t, correction
#

class ProposedRegressionModel(nn.Module):
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(self, model_id, model_keypoints, cad_models, keypoint_detector=None):
        super().__init__()
        """ 
        model_keypoints     : torch.tensor of shape (K, 3, N)
        cad_models          : torch.tensor of shape (K, 3, n)  
        keypoint_detector   : torch.nn.Module   : detects N keypoints for any sized point cloud input       
                                                  should take input : torch.tensor of shape (B, 3, m)
                                                  should output     : torch.tensor of shape (B, 3, N)

        keypoint_detector_type  : 'regression' or 'heatmap'

        """

        # Parameters
        self.model_id = model_id
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.viz_keypoint_correction = False

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # Keypoint Detector
        if keypoint_detector == None:
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')

        elif keypoint_detector == 'pointnet':
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')

        elif keypoint_detector == 'point_transformer':
            self.keypoint_detector = RegressionKeypoints(N=self.N, method='point_transformer')

        else:
            self.keypoint_detector = keypoint_detector(model_name=model_name, N=self.N)

        # Registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)

        # Corrector
        # self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        self.corrector = ParamDeclarativeFunction(problem=corrector_node)

    def forward(self, input_point_cloud, correction_flag=False, need_predicted_keypoints=False):
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
        detected_keypoints += center #keypoints are put back to original positions

        if self.viz_keypoint_correction:
            inp = input_point_cloud.clone().detach().to('cpu')
            det_kp = detected_keypoints.clone().detach().to('cpu')
            # print("FINISHED DETECTOR")

        if not correction_flag:
            R, t = self.point_set_registration.forward(detected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t

            # return predicted_point_cloud, detected_keypoints, R, t, None
            if not need_predicted_keypoints:
                return predicted_point_cloud, detected_keypoints, R, t, None
            else:
                predicted_model_keypoints = R @ self.model_keypoints + t
                return predicted_point_cloud, detected_keypoints, R, t, None, predicted_model_keypoints

        else:
            correction = self.corrector.forward(detected_keypoints, input_point_cloud)
            corrected_keypoints = detected_keypoints + correction
            R, t = self.point_set_registration.forward(corrected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t
            if self.viz_keypoint_correction:
                # print("FINISHED CORRECTOR")
                print("visualizing corrected keypoints")
                Rt_inp = predicted_point_cloud.clone().detach().to('cpu')
                corrected_kp = corrected_keypoints.clone().detach().to('cpu')
                display_results(inp, det_kp, inp, corrected_kp)

            # return predicted_point_cloud, corrected_keypoints, R, t, correction
            if not need_predicted_keypoints:
                return predicted_point_cloud, corrected_keypoints, R, t, correction
            else:
                predicted_model_keypoints = R @ self.model_keypoints + t
                return predicted_point_cloud, corrected_keypoints, R, t, correction, predicted_model_keypoints


# class ProposedHeatmapModel(nn.Module):
#     """
#     Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation
#
#     Returns:
#         predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
#         predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
#     """
#
#     def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None,
#                  use_pretrained_regression_model=False):
#         super().__init__()
#         """
#         model_keypoints     : torch.tensor of shape (K, 3, N)
#         cad_models          : torch.tensor of shape (K, 3, n)
#         keypoint_detector   : torch.nn.Module   : detects N keypoints for any sized point cloud input
#                                                   should take input : torch.tensor of shape (B, 3, m)
#                                                   should output     : torch.tensor of shape (B, 3, N)
#
#         keypoint_detector_type  : 'regression' or 'heatmap'
#
#         """
#
#         # Parameters
#         self.class_name = class_name
#         self.model_keypoints = model_keypoints
#         self.cad_models = cad_models
#         self.device_ = self.cad_models.device
#         self.viz_keypoint_correction = False
#         self.use_pretrained_regression_model = use_pretrained_regression_model
#
#         self.N = self.model_keypoints.shape[-1]  # (1, 1)
#         self.K = self.model_keypoints.shape[0]  # (1, 1)
#
#         # Keypoint Detector
#         if keypoint_detector == None:
#             self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')
#
#         elif keypoint_detector == 'pointnet':
#             self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')
#
#         elif keypoint_detector == 'point_transformer':
#             raise NotImplementedError
#
#         else:
#             raise NotImplementedError
#
#
#         # Registration
#         self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)
#
#         # Corrector
#         # self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
#         corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
#         self.corrector = ParamDeclarativeFunction(problem=corrector_node)
#
#     def forward(self, input_point_cloud, correction_flag=False, need_predicted_keypoints=False):
#         """
#         input:
#         input_point_cloud   : torch.tensor of shape (B, 3, m)
#
#         where
#         B = batch size
#         m = number of points in each point cloud
#
#         output:
#         keypoints           : torch.tensor of shape (B, 3, self.N)
#         target_point_cloud  : torch.tensor of shape (B, 3, n)
#         # rotation          : torch.tensor of shape (B, 3, 3)
#         # translation       : torch.tensor of shape (B, 3, 1)
#         # shape             : torch.tensor of shape (B, self.K, 1)
#
#         """
#         batch_size, _, m = input_point_cloud.shape
#         device_ = input_point_cloud.device
#
#         num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
#         num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
#         num_nonzero_pts = m - num_zero_pts
#         num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)
#
#         center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
#         center = center.unsqueeze(-1)
#         pc_centered = input_point_cloud - center
#         detected_keypoints, heatmap = self.keypoint_detector(pc_centered)
#         # print("detected_keypoints on: ", detected_keypoints.device)
#         # print("heatmap on: ", heatmap.device)
#         # print("center on: ", center.device)
#         detected_keypoints += center
#
#         if self.viz_keypoint_correction:
#             inp = input_point_cloud.clone().detach().to('cpu')
#             det_kp = detected_keypoints.clone().detach().to('cpu')
#             # print("FINISHED DETECTOR")
#
#         if not correction_flag:
#             R, t = self.point_set_registration.forward(detected_keypoints)
#             predicted_point_cloud = R @ self.cad_models + t
#
#             if not need_predicted_keypoints:
#                 return predicted_point_cloud, detected_keypoints, R, t, None, heatmap
#             else:
#                 predicted_model_keypoints = R @ self.model_keypoints + t
#                 return predicted_point_cloud, detected_keypoints, R, t, None, heatmap, predicted_model_keypoints
#
#         else:
#             correction = self.corrector.forward(detected_keypoints, input_point_cloud)
#             corrected_keypoints = detected_keypoints + correction
#             R, t = self.point_set_registration.forward(corrected_keypoints)
#             predicted_point_cloud = R @ self.cad_models + t
#             if self.viz_keypoint_correction:
#                 # print("FINISHED CORRECTOR")
#                 print("visualizing corrected keypoints")
#                 Rt_inp = predicted_point_cloud.clone().detach().to('cpu')
#                 corrected_kp = corrected_keypoints.clone().detach().to('cpu')
#                 display_results(inp, det_kp, inp, corrected_kp)
#
#             if not need_predicted_keypoints:
#                 return predicted_point_cloud, corrected_keypoints, R, t, correction, heatmap
#             else:
#                 predicted_model_keypoints = R @ self.model_keypoints + t
#                 return predicted_point_cloud, corrected_keypoints, R, t, correction, heatmap, predicted_model_keypoints