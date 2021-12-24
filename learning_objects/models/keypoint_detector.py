"""
This implements keypoint detector modules.

Note:
    There are two kinds of keypoint detectors: heatmap based and regression based.

    Current implementations use Point Transformer and PointNet++ architectures for the keypoint detectors,
    and can be chosen by specifying the "method" variable in the input.

"""

import torch
import torch.nn as nn

import os
import sys
sys.path.append("../../")

from learning_objects.models.point_transformer import PointTransformerSegment, PointTransformerCls
from learning_objects.models.pointnet import PointNetDenseCls, PointNetCls



class HeatmapKeypoints(nn.Module):
    """
    This module generates keypoints of an input point cloud using heatmaps.

    Note:
    The output keypoints, by definition, are a subset of the input point cloud.
    This method uses point cloud segmentation architecture.
    """
    def __init__(self, k=20, method='point_transformer', dim=[3, 32, 64, 128, 256, 512]):
        super().__init__()
        """
        Inputs: 
        ------
            k       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
            dim     : list(int)   : dimensions of various layers, in the case of point transformer architecture. 
        """
        self.k = k
        self.method = method
        self.dim = dim

        if self.method == 'point_transformer':
            self.keypoint_heatmap = PointTransformerSegment(output_dim=self.k, dim=self.dim)
        elif self.method == 'pointnet':
            self.keypoint_heatmap = PointNetDenseCls(k=self.k)
        else:
            raise NotImplementedError

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
            pointcloud: (B, N, 3+d) torch.Tensor
                        B = batch size
                        N = number of points per point cloud
                        d = input feature dimension
                        pointcloud[..., 0:3] = locations
                        pointcloud[..., 3:] = input features

        Outputs:
        -------
            y: (B, self.k, 3) torch.Tensor
                        B = batch size
                        self.k = number of keypoints
                        y[..., 0:3] = location of keypoints
        """

        if self.method == 'point_transformer':

            heatmap = self.keypoint_heatmap(pointcloud)
            _, idx = torch.max(heatmap, dim=1)

        elif self.method == 'pointnet':

            pos, _ = self._break_up_pc(pointcloud)
            heatmap, _, _ = self.keypoint_heatmap(pos.transpose(1, 2))
            _, idx = torch.max(heatmap, dim=1)

        else:
            raise ValueError


        y = torch.zeros(size=(pointcloud.size(0), self.k, pointcloud.size(2)))
        for i in range(idx.size(0)):
            y[i, :, :] = pointcloud[i, idx[i], :]

        return y


class RegressionKeypoints(nn.Module):
    """
    This module generates keypoints as a regression of the input point cloud.

    Note:
    The output keypoints may not be a subset of the input point cloud.
    This method uses point cloud classification architecture.
    """
    def __init__(self, k=20, method='point_transformer'):
        super().__init__()
        """
        Inputs: 
        ------
            k       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
        """
        self.k = k
        self.method = method

        if self.method == 'point_transformer':
            self.keypoint = PointTransformerCls(output_dim=3*self.k)
        elif self.method == 'pointnet':
            self.keypoint = PointNetCls(k=3*self.k)
        else:
            raise NotImplementedError

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
            pointcloud: (B, N, 3+d) torch.Tensor
                    B = batch size
                    N = number of points per point cloud
                    d = input feature dimension
                    pointcloud[..., 0:3] = locations
                    pointcloud[..., 3:] = input features

        Outputs:
        -------
            y: (B, self.k, 3) torch.Tensor
                    B = batch size
                    self.k = number of keypoints
                    y[..., 0:3] = location of keypoints
        """

        if self.method == 'point_transformer':

            keypoints = self.keypoint(pointcloud)

        elif self.method == 'pointnet':

            pos, _ = self._break_up_pc(pointcloud)
            keypoints = self.keypoint(pos.transpose(1, 2))

        else:
            raise ValueError

        return torch.reshape(keypoints, (keypoints.shape[0], self.k, 3))



if __name__ == "__main__":

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    print('-'*20)
    print('Testing Heatmap Models')
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

    print('-' * 20)
    print('Testing Regression Models')
    print('-' * 20)

