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
    def __init__(self, N=20, method='point_transformer', dim=[3, 32, 64, 128, 256, 512]):
        super().__init__()
        """
        Inputs: 
        ------
            N       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
            dim     : list(int)   : dimensions of various layers, in the case of point transformer architecture. 
        """
        self.N = N
        self.method = method
        self.dim = dim

        if self.method == 'point_transformer':
            self.keypoint_heatmap = PointTransformerSegment(output_dim=self.N, dim=self.dim)
        elif self.method == 'pointnet':
            self.keypoint_heatmap = PointNetDenseCls(k=self.N)
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
        pointcloud: (B, 3+d, self.N) torch.Tensor
                        B = batch size
                        self.N = number of points per point cloud
                        d = input feature dimension
                        pointcloud[B, 0:3, self.N] = locations
                        pointcloud[B, 3:, self.N] = input features

        Outputs:
        -------
            y: (B, 3, self.N) torch.Tensor
                        B = batch size
                        self.N = number of keypoints
                        y[B, 0:3, self.N] = location of keypoints
        """

        pointcloud = torch.transpose(pointcloud, -1, -2)    # (B, N, 3+d)

        if self.method == 'point_transformer':

            heatmap = self.keypoint_heatmap(pointcloud)
            _, idx = torch.max(heatmap, dim=1)

        elif self.method == 'pointnet':

            pos, _ = self._break_up_pc(pointcloud)
            heatmap, _, _ = self.keypoint_heatmap(pos.transpose(1, 2))
            _, idx = torch.max(heatmap, dim=1)

        else:
            raise ValueError


        y = torch.zeros(size=(pointcloud.size(0), self.N, 3))
        for i in range(idx.size(0)):
            y[i, :, :] = pointcloud[i, idx[i], :3]

        return y.transpose(-1, -2)      # (B, 3, self.N)


class RegressionKeypoints(nn.Module):
    """
    This module generates keypoints as a regression of the input point cloud.

    Note:
    The output keypoints may not be a subset of the input point cloud.
    This method uses point cloud classification architecture.
    """
    def __init__(self, N=20, method='point_transformer', dim=[3, 16, 32, 64, 128, 256]):
        super().__init__()
        """
        Inputs: 
        ------
            N       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
        """
        self.N = N
        self.method = method

        if self.method == 'point_transformer':
            self.keypoint = PointTransformerCls(output_dim=3*self.N, channels=dim, sampling_ratio=0.25)
        elif self.method == 'pointnet':
            self.keypoint = PointNetCls(k=3*self.N)
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
            pointcloud: (B, 3+d, m) torch.Tensor
                    B = batch size
                    m = number of points per point cloud
                    d = input feature dimension
                    pointcloud[..., 0:3] = locations
                    pointcloud[..., 3:] = input features

        Outputs:
        -------
            y: (B, 3, self.N) torch.Tensor
                    B = batch size
                    self.N = number of keypoints
                    y[..., 0:3] = location of keypoints
        """
        batch_size = pointcloud.shape[0]
        pointcloud = torch.transpose(pointcloud, -1, -2)  # (B, N, 3+d)


        if self.method == 'point_transformer':
            keypoints = self.keypoint(pointcloud)

        elif self.method == 'pointnet':
            pos, _ = self._break_up_pc(pointcloud)
            keypoints, _, _ = self.keypoint(pos.transpose(1, 2))

        else:
            raise ValueError

        return torch.reshape(keypoints, (batch_size, 3, self.N))



if __name__ == "__main__":

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    # Parameters:
    N = 44      # number of keypoints
    m = 100     # number of points in the input point cloud
    b = 10      # batch size

    print('-'*20)
    print('Testing Heatmap Models')
    print('-'*20)
    # Test: HeatmapKeypoints() with method='pointnet'
    print('Test: HeatmapKeypoints() with method=\'pointnet\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)
    # pc = torch.rand(4, 1000, 11)
    pc = pc.to(device=device)
    kp = HeatmapKeypoints(method='pointnet', N=N).to(device=device)
    # kp = HeatmapKeypoints(method='point_transformer', dim=[8, 32, 64], k=63).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)



    # Test: HeatmapKeypoints() with method='point_transformer'
    print('Test: HeatmapKeypoints() with method=\'point_transformer\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)
    # pc = torch.rand(4, 1000, 6)
    pc = pc.to(device=device)
    # kp = HeatmapKeypoints(method='pointnet', k=63).to(device=device)
    kp = HeatmapKeypoints(method='point_transformer', dim=[8, 16, 24], N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)



    print('-'*20)
    print('Testing Regression Models')
    print('-'*20)
    # Test: RegressionKeypoints() with method='pointnet'
    print('Test: RegressionKeypoints() with method=\'pointnet\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)

    pc = pc.to(device=device)
    kp = RegressionKeypoints(method='pointnet', N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)



    # Test: RegressionKeypoints() with method='point_transformer'
    print('Test: RegressionKeypoints() with method=\'point_transformer\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)
    # pc = torch.rand(4, 1000, 6)
    pc = pc.to(device=device)
    kp = RegressionKeypoints(method='point_transformer', dim=[8, 16, 24], N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)

