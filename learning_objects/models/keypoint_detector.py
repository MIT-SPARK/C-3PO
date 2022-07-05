"""
This implements keypoint detector modules.

Note:
    Current implementations use Point Transformer and PointNet++ architectures for the keypoint detectors,
    and can be chosen by specifying the "method" variable in the input.

"""

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../../")


from learning_objects.models.point_transformer import PointTransformerSegment, PointTransformerCls
from learning_objects.models.pointnet import PointNetDenseCls, PointNetCls


class ModelWrapper(torch.nn.Module):
    def __init__(self, model_impl) -> None:
        super().__init__()
        self.model_impl = model_impl

    def forward(self, data):
        pc = data[0]
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()
        res = self.model_impl(pc.transpose(1, 2).cuda())
        return res

class CorrespondenceCriterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, outputs):
        print("---------------------------------------------------")
        print("inside CorrespondenceCriterion")
        print("inputs.shape", inputs.shape)
        print("outputs.shape", outputs.shape)
        kp_indexs = inputs[1]

        loss = []
        for b, kp_index in enumerate(kp_indexs):
            loss_rot = []
            for rot_kp_index in kp_index:
                loss_rot.append(F.cross_entropy(outputs[b][None], rot_kp_index[None].long().cuda(), ignore_index=-1))
            loss.append(torch.min(torch.stack(loss_rot)))
        loss = torch.mean(torch.stack(loss))
        return loss

class RegressionKeypoints(nn.Module):
    """
    This module generates keypoints as a regression of the input point cloud.

    Note:
    The output keypoints may not be a subset of the input point cloud.
    This method uses point cloud classification architecture.
    """
    def __init__(self, N=20, method='point_transformer', local_max_pooling=True):
        super().__init__()
        """
        Inputs: 
        ------
            N       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
        """
        self.N = N
        self.method = method
        self.local_max_pooling = local_max_pooling

        if self.method == 'point_transformer':
            self.keypoint = PointTransformerCls(output_dim=3*self.N, local_max_pooling=self.local_max_pooling)
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
    kp = RegressionKeypoints(method='point_transformer',  dim=[8, 16, 24], N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)

