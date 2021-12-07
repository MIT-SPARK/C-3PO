import torch
import torch.nn as nn
# import numpy as np
# import torch.nn.functional as f
from torch.autograd import Function
from point_transformer import PointTransformerSegment
from pointnet import PointNetDenseCls

import autograd.numpy as np



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

        print(heatmap.is_cuda)
        print(pointcloud.is_cuda)


        pointcloud_inlier = torch.where((heatmap >= self.threshold), pointcloud, torch.zeros(pointcloud.size()).cuda())

        count = (heatmap >= self.threshold)
        count_out = (heatmap < self.threshold)
        num_inliers = torch.sum(count, dim=1)
        num_outliers = torch.sum(count_out, dim=1)

        return pointcloud_inlier, num_inliers, num_outliers




class Keypoints(nn.Module):
    def __init__(self, k=64, method='point_transformer', dim=[6, 32, 64]):
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
            self.keypoint_heatmap = PointTransformerSegment(dim=dim, output_dim=self.k)
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
            raise IOError


        y = torch.zeros(size=(pointcloud.size(0), self.k, pointcloud.size(2))).cuda()
        for i in range(idx.size(0)):
            y[i, :, :] = pointcloud[i, idx[i], :]

        return y




class Rotation(nn.Module):
    def __init__(self, Q):
        super(Rotation, self).__init__()
        """
        Q: 10x10 torch.tensor
        """
        self.n = Q.shape[0]
        A = []
        for i in range(16):
            A.append(np.zeros((n, n)))

        A[0][0, 0] = 1

        A[1][0, 0] = 1
        A[1][1, 1] = -1
        A[1][2, 2] = -1
        A[1][3, 3] = -1

        A[2][0, 0] = 1
        A[2][4, 4] = -1
        A[2][5, 5] = -1
        A[2][6, 6] = -1

        A[3][0, 0] = 1
        A[3][7, 7] = -1
        A[3][8, 8] = -1
        A[3][9, 9] = -1

        A[4][1, 4] = 1
        A[4][2, 5] = 1
        A[4][3, 6] = 1
        A[4][4, 1] = 1
        A[4][5, 2] = 1
        A[4][6, 3] = 1

        A[5][3, 4] = 1
        A[5][1, 6] = 1
        A[5][3, 9] = 1
        A[5][4, 3] = 1
        A[5][6, 1] = 1
        A[5][9, 3] = 1

        A[6][4, 7] = 1
        A[6][5, 8] = 1
        A[6][6, 9] = 1
        A[6][7, 4] = 1
        A[6][8, 5] = 1
        A[6][9, 6] = 1

        A[7][2, 6] = 1
        A[7][3, 5] = -1
        A[7][0, 7] = -1
        A[7][6, 2] = 1
        A[7][5, 3] = -1
        A[7][7, 0] = -1

        A[8][3, 4] = 1
        A[8][1, 6] = -1
        A[8][0, 8] = -1
        A[8][4, 3] = 1
        A[8][6, 1] = -1
        A[8][8, 0] = -1

        A[9][1, 5] = 1
        A[9][0, 9] = -1
        A[9][2, 4] = -1
        A[9][5, 1] = 1
        A[9][9, 0] = -1
        A[9][4, 2] = -1

        A[10][5, 9] = 1
        A[10][0, 1] = -1
        A[10][6, 8] = -1
        A[10][9, 5] = 1
        A[10][1, 0] = -1
        A[10][8, 6] = -1

        A[11][6, 7] = 1
        A[11][4, 9] = -1
        A[11][0, 2] = -1
        A[11][7, 6] = 1
        A[11][9, 4] = -1
        A[11][2, 0] = -1

        A[12][4, 8] = 1
        A[12][0, 3] = -1
        A[12][5, 7] = -1
        A[12][8, 4] = 1
        A[12][3, 0] = -1
        A[12][7, 5] = -1

        A[13][3, 8] = 1
        A[13][2, 9] = -1
        A[13][0, 4] = -1
        A[13][8, 3] = 1
        A[13][9, 2] = -1
        A[13][4, 0] = -1

        A[14][1, 9] = 1
        A[14][0, 5] = -1
        A[14][3, 7] = -1
        A[14][9, 1] = 1
        A[14][5, 0] = -1
        A[14][7, 3] = -1

        A[15][2, 7] = 1
        A[15][1, 8] = -1
        A[15][0, 6] = -1
        A[15][7, 2] = 1
        A[15][8, 1] = -1
        A[15][6, 0] = -1

        b = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(A[i] @ X) == b[i] for i in range(16)
        ]

        prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints=constraints)
        prob.solve()

        # Step(2): projects X.vale onto the rank 1 PSD manifold and extracts the rotation matrix R
        u, s, vh = np.linalg.svd(X.value)
        u0 = np.sqrt(s[0]) * u[:, 0]
        u0 = u0 / u0[0]
        rvec = u0[1:]
        R = rvec.reshape((3, 3))







if __name__ == '__main__':

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # Test: Keypoints() with method='pointnet'
    print('Test: Keypoints() with method=\'pointnet\'')

    pc = torch.rand(4, 1000, 11)
    pc = pc.to(device=device)
    kp = Keypoints(method='pointnet', k=63).to(device=device)
    # kp = Keypoints(method='point_transformer', dim=[8, 32, 64], k=63).to(device=device)
    y = kp(pointcloud=pc)
    print(y.size())
    print('-' * 20)

    # Test: Keypoints() with method='point_transformer'
    print('Test: Keypoints() with method=\'point_transformer\'')

    pc = torch.rand(4, 1000, 11)
    pc = pc.to(device=device)
    # kp = Keypoints(method='pointnet', k=63).to(device=device)
    kp = Keypoints(method='point_transformer', dim=[8, 32, 64], k=63).to(device=device)
    y = kp(pointcloud=pc)
    print(y.size())
    print('-' * 20)


    # Test: PACE
    

