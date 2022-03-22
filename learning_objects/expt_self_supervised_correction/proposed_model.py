"""
This writes a proposed model for expt_registration

"""


import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch3d import ops
import copy

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import sys
sys.path.append("../../")

from learning_objects.models.keypoint_detector import HeatmapKeypoints, RegressionKeypoints
from learning_objects.models.point_set_registration import PointSetRegistration
from learning_objects.models.keypoint_corrector import kp_corrector_reg

from learning_objects.models.point_transformer import kNN_torch, index_points

from learning_objects.utils.ddn.node import ParamDeclarativeFunction
from learning_objects.utils.general import display_results, pos_tensor_to_o3d


# Proposed Model
class ProposedModel(nn.Module):
    """
    This is depreciated. Use either ProposedRegressionModel() or ProposedHeatmapModel()
    """
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None,
                 use_pretrained_regression_model=False, keypoint_detector_type='regression'):
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
        self.class_name = class_name
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.viz_keypoint_correction = False
        self.use_pretrained_regression_model = use_pretrained_regression_model
        self.keypoint_detector_type = keypoint_detector_type

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # Keypoint Detector
        if keypoint_detector_type == 'regression':
            if keypoint_detector == None:
                self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')

            elif keypoint_detector == 'pointnet':
                self.keypoint_detector = RegressionKeypoints(N=self.N, method='pointnet')

            elif keypoint_detector == 'point_transformer':
                self.keypoint_detector = RegressionKeypoints(N=self.N, method='point_transformer')

            else:
                self.keypoint_detector = keypoint_detector(class_name=class_name, N=self.N)

        elif keypoint_detector_type == 'heatmap':
            if keypoint_detector == None:
                self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')

            elif keypoint_detector == 'pointnet':
                self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')

            elif keypoint_detector == 'point_transformer':
                raise NotImplementedError

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError



        # Registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)

        # Corrector
        # self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        self.corrector = ParamDeclarativeFunction(problem=corrector_node)

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
        batch_size, _, m = input_point_cloud.shape
        device_ = input_point_cloud.device

        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1)/num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        detected_keypoints = self.keypoint_detector(pc_centered)
        detected_keypoints += center

        if self.viz_keypoint_correction:
            inp = input_point_cloud.clone().detach().to('cpu')
            det_kp = detected_keypoints.clone().detach().to('cpu')
            # print("FINISHED DETECTOR")

        if not correction_flag:
            R, t = self.point_set_registration.forward(detected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t

            return predicted_point_cloud, detected_keypoints, R, t, None

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

            return predicted_point_cloud, corrected_keypoints, R, t, correction


class BaselineRegressionModel(nn.Module):

    def __init__(self, class_name, model_keypoints, cad_models, regression_model=None,
                 use_pretrained_regression_model=False):
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
        self.class_name = class_name
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # Keypoint Detector
        if regression_model == None:
            self.regression_model = RegressionKeypoints(N=9 + self.K, method='pointnet')

        elif regression_model == 'pointnet':
            self.regression_model = RegressionKeypoints(N=9 + self.K, method='pointnet')

        elif regression_model == 'point_transformer':
            self.regression_model = RegressionKeypoints(N=9 + self.K, method='point_transformer')

        else:
            raise NotImplementedError

    def forward(self, input_point_cloud, need_predicted_model=False):
        """
        point_cloud : torch.tensor of shape (B, 3, m)

        output:
        rotation        : torch.tensor of shape (B, 3, 3)
        translation     : torch.tensor of shape (B, 3, 1)
        predicted_pc    :   torch.tensor of shape (B, 3, n)

        """

        batch_size, _, m = input_point_cloud.shape
        # device_ = input_point_cloud.device

        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        x = self.regression_model(pc_centered)

        R = x[:, 9].reshape(-1, 3, 3)
        t = x[:, 9:]
        t += center

        if need_predicted_model:
            return R, t, R @ self.cad_models + t
        else:
            return R, t, None


class ProposedRegressionModel(nn.Module):
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None,
                 use_pretrained_regression_model=False):
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
        self.class_name = class_name
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.viz_keypoint_correction = False
        self.use_pretrained_regression_model = use_pretrained_regression_model      #ToDo: This has to be always False. Was written for RSNet.

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
            self.keypoint_detector = keypoint_detector(class_name=class_name, N=self.N)

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
        detected_keypoints += center

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


class ProposedHeatmapModel(nn.Module):
    """
    Given input point cloud, returns keypoints, predicted point cloud, rotation, and translation

    Returns:
        predicted_pc, detected_keypoints, rotation, translation     if correction_flag=False
        predicted_pc, corrected_keypoints, rotation, translation    if correction_flag=True
    """

    def __init__(self, class_name, model_keypoints, cad_models, keypoint_detector=None,
                 use_pretrained_regression_model=False):
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
        self.class_name = class_name
        self.model_keypoints = model_keypoints
        self.cad_models = cad_models
        self.device_ = self.cad_models.device
        self.viz_keypoint_correction = False
        self.use_pretrained_regression_model = use_pretrained_regression_model

        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)

        # Keypoint Detector
        if keypoint_detector == None:
            self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')

        elif keypoint_detector == 'pointnet':
            self.keypoint_detector = HeatmapKeypoints(N=self.N, method='pointnet')

        elif keypoint_detector == 'point_transformer':
            raise NotImplementedError

        else:
            raise NotImplementedError


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
        detected_keypoints, heatmap = self.keypoint_detector(pc_centered)
        # print("detected_keypoints on: ", detected_keypoints.device)
        # print("heatmap on: ", heatmap.device)
        # print("center on: ", center.device)
        detected_keypoints += center

        if self.viz_keypoint_correction:
            inp = input_point_cloud.clone().detach().to('cpu')
            det_kp = detected_keypoints.clone().detach().to('cpu')
            # print("FINISHED DETECTOR")

        if not correction_flag:
            R, t = self.point_set_registration.forward(detected_keypoints)
            predicted_point_cloud = R @ self.cad_models + t

            if not need_predicted_keypoints:
                return predicted_point_cloud, detected_keypoints, R, t, None, heatmap
            else:
                predicted_model_keypoints = R @ self.model_keypoints + t
                return predicted_point_cloud, detected_keypoints, R, t, None, heatmap, predicted_model_keypoints

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

            if not need_predicted_keypoints:
                return predicted_point_cloud, corrected_keypoints, R, t, correction, heatmap
            else:
                predicted_model_keypoints = R @ self.model_keypoints + t
                return predicted_point_cloud, corrected_keypoints, R, t, correction, heatmap, predicted_model_keypoints


# Baseline Implementations:
class ICP():
    def __init__(self, cad_models):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        
        """
        self.cad_models = cad_models
        self.source_points = pos_tensor_to_o3d(pos=cad_models.squeeze(0).to('cpu'), estimate_normals=False)
        self.voxel_size = 0.1
        self.threshold = 0.01

        self.source_down, self.source_fpfh = self.preprocess_point_cloud(self.source_points)
        # self.trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
        #                               [0.0, 1.0, 0.0, 0.0],
        #                               [0.0, 0.0, 1.0, 0.0],
        #                               [0.0, 0.0, 0.0, 0.0]])

    def forward(self, input_point_cloud):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """
        device_ = input_point_cloud.device
        batch_size, _, m = input_point_cloud.shape

        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center


        pc = pc_centered.to('cpu')
        R = torch.eye(3)
        R = R.reshape((1, 3, 3))
        R = R.repeat(batch_size, 1, 1)
        R = R.to('cpu')
        # R = torch.zeros(batch_size, 3, 3).to('cpu')
        t = torch.zeros(batch_size, 3, 1).to('cpu')

        for b in range(batch_size):
            target_points = pc[b, ...]
            target_points = pos_tensor_to_o3d(pos=target_points, estimate_normals=False)

            target_down, target_fpfh = self.preprocess_point_cloud(target_points)
            result_ransac = self.execute_global_registration(target_down, target_fpfh)
            # print("/////////////////////////////////////// GLOBAL REGISTRATION")
            # print(result_ransac)
            # print(result_ransac.transformation)

            reg_p2p = o3d.pipelines.registration.registration_icp(self.source_points,
                                                                  target_points,
                                                                  self.threshold,
                                                                  result_ransac.transformation,
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                  o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))

            # self.draw_registration_result(self.source_points, target_points, reg_p2p.transformation)
            # print("//////////////////////////////////// ICP")
            # print(reg_p2p)
            T = reg_p2p.transformation
            # print(T)
            R_ = np.array(T[:3, :3])
            t_ = np.array(T[3, :3])
            # print(R_)
            # print(t_)
            # print(R.shape)
            # x = torch.from_numpy(R_)
            # print(x.shape)
            R[b, ...] = torch.from_numpy(R_)
            t[b, :, 0] = torch.from_numpy(t_)

        R = R.to(device=device_)
        t = t.to(device=device_)
        t = t + center

        return R @ self.cad_models + t, R, t

    def execute_global_registration(self, target_down, target_fpfh):

        voxel_size = self.voxel_size
        source_down = self.source_down
        source_fpfh = self.source_fpfh
        distance_threshold = voxel_size * 1.5
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def preprocess_point_cloud(self, pcd):
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        voxel_size = self.voxel_size
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def draw_registration_result(self, source, target, transformation):

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


