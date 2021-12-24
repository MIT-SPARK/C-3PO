"""
This implements a background segmentation module.

Note:
    Given an input point cloud, a backgroud segmentor segments out the background.

"""

import torch
import torch.nn as nn

import os
import sys
sys.path.append("../../")

from learning_objects.models.point_transformer import PointTransformerSegment


class SegmentBackground(nn.Module):
    def __init__(self, method='point_transformer'):
        super().__init__()

        self.method = method
        self.softmax = torch.nn.Softmax(dim=2)
        self.threshold = 0.5

        if self.method == 'point_transformer':
            self.inlier_heatmap = PointTransformerSegment(dim=[3, 32, 64], output_dim=1)
        else:
            raise NotImplementedError

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
            pointscloud: (B, N, 3+d)
                B = batch size
                N = number of points in a batch
                d = input feature dimension
                pointcloud[..., 0:3] = positions
                pointcloud[..., 3:]  = feature

        Outputs:
        -------
            pointcloud_inlier: (B, N, 3+d)
                location and features will be set to 0 for outlier points
            num_inliers: (B, 1) torch.Tensor.int32
                is the number of inliers
            num_outliers: (B, 1) torch.Tensor.int32
                is the number of outliers
        """

        pos, _ = self._break_up_pc(pc=pointcloud)
        pointcloud_pos = torch.cat((pos, pos), dim=2)
        heatmap = self.inlier_heatmap(pointcloud_pos)

        pointcloud_inlier = torch.where((heatmap >= self.threshold), pointcloud, torch.zeros_like(pointcloud))

        count = (heatmap >= self.threshold)
        count_out = (heatmap < self.threshold)
        num_inliers = torch.sum(count, dim=1)
        num_outliers = torch.sum(count_out, dim=1)

        return pointcloud_inlier, num_inliers, num_outliers



if __name__ == "__main__":

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


