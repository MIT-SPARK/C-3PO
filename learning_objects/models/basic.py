import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import os
import sys
sys.path.append("../../")

from learning_objects.models.point_transformer import PointTransformerSegment
from learning_objects.models.pointnet import PointNetDenseCls




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




class PACE(nn.Module):
    def __init__(self, weights, model_keypoints, lambda_constant=torch.tensor(1.0)):
        super().__init__()
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: torch.tensor of shape (1, 1)
        """

        self.w = weights.unsqueeze(0)               # (1, N, 1)
        self.b = model_keypoints                    # (K, 3, N)
        self.lambda_constant = lambda_constant      # (1, 1)

        self.N = self.b.shape[-1]                   # (1, 1)
        self.K = self.b.shape[0]                    # (1, 1)

        self.b_w = self._get_b_w()                  # (1, K, 3)
        self.bar_B = self._get_bar_B()              # (1, 3N, K)
        self.G, self.g, self.M, self.h = self._get_GgMh()
                                                    # G: (1, K, K)
                                                    # g: (1, K, 1)
                                                    # M: (1, 3N+K, 3N)
                                                    # h: (1, 3N+K, 1)

        self.A = self._getA()                       # A: numpy.array
        self.d = self._getd()                       # d: list of size N
        self.P = self._getP()                       # P: numpy.array


        # Defining the SDP Layer
        X = cp.Variable((10, 10), symmetric=True)
        Q = cp.Parameter((10, 10), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(self.A[i, :, :] @ X) == self.d[i] for i in range(16)
        ]
        problem = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints=constraints)
        assert problem.is_dpp()

        self.sdp_for_rotation = CvxpyLayer(problem, parameters=[Q], variables=[X])


    def forward(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size
        """

        y_w = self._get_y_w(y=y)
        bar_y = self._get_bar_y(y=y, y_w=y_w)

        # R, Qbatch = self._rotation(bar_y=bar_y) # this was for verification
        R, _ = self._rotation(bar_y=bar_y)
        c = self._shape(bar_y=bar_y, R=R)
        t = self._translation(y_w=y_w, R=R, c=c)

        # return R, t, c, y_w, bar_y, Qbatch # This was for verification
        return R, t, c


    def _get_y_w(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        intermediate:
        self.w: torch.tensor of shape (1, N, 1)

        output:
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size
        """

        return torch.matmul(y, self.w)/self.w.sum()


    def _rotation(self, bar_y):
        """
        input:
        bar_y: torch.tensor of shape (B, 3, N), where B = batch size

        intermediate:
        self.M: torch.tensor of shape (1, 3N+K, 3N)
        self.h: torch.tensor of shape (1, 3N+K, 1)
        self.P: torch.tensor of shape (9, 9)

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        """

        R = torch.zeros(bar_y.shape[0], 3, 3)
        Qbatch = torch.zeros(bar_y.shape[0], 10, 10)

        M = self.M.squeeze(0)
        h = self.h.squeeze(0)

        for batch in range(bar_y.shape[0]):

            Y = bar_y[batch, :, :]

            Q = torch.zeros(10, 10)
            Q[0, 0] = torch.matmul(h.T, h)
            tempA = torch.matmul(h.T, M)
            tempB = Y.T
            tempB = tempB.contiguous()
            tempB = torch.kron(tempB, torch.eye(3))
            tempC = torch.matmul(tempB, self.P)
            Q[0, 1:] = torch.matmul(tempA, tempC)
            Q[1:, 0] = Q[0, 1:].T

            tempD = torch.matmul(M, tempB)
            tempE = torch.matmul(tempD, self.P)
            Q[1:, 1:] =  torch.matmul(tempE.T, tempE)

            Qbatch[batch, :, :] = Q[:, :] # Qbatch for verification
            tempR = self._get_rotation(Q=Q)

            R[batch, :, :] = tempR


        return R, Qbatch # Returning Qbatch


    def _shape(self, bar_y, R):
        """
        input:
        bar_y: torch.tensor of shape (B, 3, N), where B = batch size
        R: torch.tensor of shape (B, 3, 3), where B = batch size

        intermediate:
        self.G: torch.tensor of shape (K, K)
        self.g: torch.tensor of shape (K, 1)

        output:
        c: torch.tensor of shape (B, K), where B = batch size
        """

        temp_bar_y = torch.transpose(bar_y, -1, -2).reshape(bar_y.shape[0], bar_y.shape[-1]*bar_y.shape[-2], 1) # (B, 3N, 1)
        A = torch.matmul(self.G, torch.transpose(self.bar_B, -1, -2))
        tempK = torch.transpose(R, -1, -2)
        tempK = tempK.contiguous()
        tempL = torch.kron(torch.eye(self.N), tempK)
        B = torch.matmul(tempL, temp_bar_y)

        return 2*torch.matmul(A, B) + self.g


    def _translation(self, y_w, R, c):
        """
        input:
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size

        intermediate:
        self.b_w: torch.tensor of shape (1, K, 3)
        self.g: torch.tensor of shape (K, 1)

        output:
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        """

        return y_w - torch.matmul(R, torch.matmul(torch.transpose(self.b_w, -1, -2), c))


    def _get_rotation(self, Q):
        """
        input:
        Q: torch.tensor of shape (10, 10)

        output:
        R: torch.tensor of shape (3, 3)
        """
        #
        # The function computes the rotation matrix R. It does so in two steps:
        # (1) solves the optimization problem specified in (18) [1] to get a PSD matrix X
        # (2) projects the solution X onto rank 1 matrix manifold to get R

        # Step (1)
        X, = self.sdp_for_rotation(Q)

        # Step (2): computes rotation matrix from X
        ev, evec = torch.linalg.eigh(X)
        idx = torch.argsort(ev)
        evmax = ev[idx[-1]]
        evsmax = ev[idx[-2]]
        vec = evec[:, idx[-1]]
        vec = vec / vec[0]
        r = vec[1:]
        A = torch.reshape(r, (3,3)).T

        # # Projecting A to SO(3) to get R
        # # Note: should ideally pass R, but R and A tend to be same!!
        # # Note: omitted this projection for better back-propagation!!
        # U, S, Vh = torch.linalg.svd(A)
        # R = torch.matmul(U, Vh)
        # if torch.linalg.det(R) < 0:
        #     R = torch.matmul(torch.matmul(U, np.diag([1, 1, -1])), Vh)

        return A


    def _get_b_w(self):
        """
        intermediate:
        self.b: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)

        output:
        b_w: torch.tensor of shape (1, K, 3)
        """

        b_w = torch.matmul(self.b, self.w)/self.w.sum() # (K, 3, 1)

        return b_w.squeeze(-1).unsqueeze(0) # (1, K, 3)


    def _get_bar_y(self, y, y_w):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size

        intermediate:
        self.w: torch.tensor of shape (1, N, 1)

        output:
        bar_y: torch.tensor of shape (B, 3, N), where B = batch size
        """

        return torch.sqrt(torch.transpose(self.w, -1, -2)) * (y-y_w)


    def _get_bar_B(self):
        """
        intermediate:
        self.b: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)
        b_w: torch.tensor of shape (1, K, 3)

        output:
        bar_B: torch.tensor of shape (1, 3N, K), where B = batch size
        """

        bar_b = torch.sqrt(torch.transpose(self.w, -1, -2))*(self.b - self.b_w.squeeze(0).unsqueeze(-1)) # (K, 3, N)
        bar_B = torch.transpose(bar_b, -1, -2).reshape(bar_b.shape[0], bar_b.shape[-1]*bar_b.shape[-2], 1) # (K, 3N, 1)
        bar_B = bar_B.squeeze(-1) # (K, 3N)
        bar_B = torch.transpose(bar_B, -1, -2) # (3N, K)

        return bar_B.unsqueeze(0) #(1, 3N, K)


    def _get_GgMh(self):
        """
        intermediate:
        self.bar_B: torch.tensor of shpae (1, 3N, K), where B = batch size

        output:
        self.G: torch.tensor of shape (1, K, K)
        self.g: torch.tensor of shape (1, K, 1)
        self.M: torch.tensor of shape (1, 3N+K, 3N)
        self.h: torch.tensor of shape (1, 3N+K, 1)
        """

        bar_B = self.bar_B.squeeze(0)

        bar_H = 2 * (torch.matmul(bar_B.T, bar_B) + self.lambda_constant*torch.eye(self.K))
        bar_Hinv = torch.inverse(bar_H)
        Htemp = torch.matmul(bar_Hinv, torch.ones(bar_Hinv.shape[-1], 1))

        G = bar_Hinv - (torch.matmul(Htemp, Htemp.T))/(torch.matmul(torch.ones(1, Htemp.shape[0]), Htemp)) # (K, K)
        g = Htemp/(torch.matmul(torch.ones(1, Htemp.shape[0]), Htemp))  # (K, 1)

        M = torch.zeros(3*self.N + self.K, 3*self.N)   # (3N+K, 3N)
        h = torch.zeros(3*self.N + self.K, 1)   # (3N+K, 1)

        M[0:3*self.N, :] = 2*torch.matmul( bar_B, torch.matmul(G, bar_B.T) ) - torch.eye(3*self.N)
        M[3*self.N:, :] = 2*torch.sqrt(self.lambda_constant)*torch.matmul(G, bar_B.T)

        h[0:3*self.N, :] = torch.matmul(self.bar_B, g)
        h[3*self.N:, :] = torch.sqrt(self.lambda_constant)*g # This is something that was added later by Jingnan in his code.
                                                             # Jingnan: 2021 Oct-11: added missing sqrt(lam) before g

        return G.unsqueeze(0), g.unsqueeze(0), M.unsqueeze(0), h.unsqueeze(0)


    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10)

        A[0, 0, 0] = 1

        A[1, 0, 0] = 1
        A[1, 1, 1] = -1
        A[1, 2, 2] = -1
        A[1, 3, 3] = -1

        A[2, 0, 0] = 1
        A[2, 4, 4] = -1
        A[2, 5, 5] = -1
        A[2, 6, 6] = -1

        A[3, 0, 0] = 1
        A[3, 7, 7] = -1
        A[3, 8, 8] = -1
        A[3, 9, 9] = -1

        A[4, 1, 4] = 1
        A[4, 2, 5] = 1
        A[4, 3, 6] = 1
        A[4, 4, 1] = 1
        A[4, 5, 2] = 1
        A[4, 6, 3] = 1

        A[5, 1, 7] = 1 #
        A[5, 2, 8] = 1
        A[5, 3, 9] = 1
        A[5, 7, 1] = 1
        A[5, 8, 2] = 1
        A[5, 9, 3] = 1

        A[6, 4, 7] = 1
        A[6, 5, 8] = 1
        A[6, 6, 9] = 1
        A[6, 7, 4] = 1
        A[6, 8, 5] = 1
        A[6, 9, 6] = 1

        A[7, 2, 6] = 1
        A[7, 3, 5] = -1
        A[7, 0, 7] = -1
        A[7, 6, 2] = 1
        A[7, 5, 3] = -1
        A[7, 7, 0] = -1

        A[8, 3, 4] = 1
        A[8, 1, 6] = -1
        A[8, 0, 8] = -1
        A[8, 4, 3] = 1
        A[8, 6, 1] = -1
        A[8, 8, 0] = -1

        A[9, 1, 5] = 1
        A[9, 0, 9] = -1
        A[9, 2, 4] = -1
        A[9, 5, 1] = 1
        A[9, 9, 0] = -1
        A[9, 4, 2] = -1

        A[10, 5, 9] = 1
        A[10, 0, 1] = -1
        A[10, 6, 8] = -1
        A[10, 9, 5] = 1
        A[10, 1, 0] = -1
        A[10, 8, 6] = -1

        A[11, 6, 7] = 1
        A[11, 4, 9] = -1
        A[11, 0, 2] = -1
        A[11, 7, 6] = 1
        A[11, 9, 4] = -1
        A[11, 2, 0] = -1

        A[12, 4, 8] = 1
        A[12, 0, 3] = -1
        A[12, 5, 7] = -1
        A[12, 8, 4] = 1
        A[12, 3, 0] = -1
        A[12, 7, 5] = -1

        A[13, 3, 8] = 1
        A[13, 2, 9] = -1
        A[13, 0, 4] = -1
        A[13, 8, 3] = 1
        A[13, 9, 2] = -1
        A[13, 4, 0] = -1

        A[14, 1, 9] = 1
        A[14, 0, 5] = -1
        A[14, 3, 7] = -1
        A[14, 9, 1] = 1
        A[14, 5, 0] = -1
        A[14, 7, 3] = -1

        A[15, 2, 7] = 1
        A[15, 1, 8] = -1
        A[15, 0, 6] = -1
        A[15, 7, 2] = 1
        A[15, 8, 1] = -1
        A[15, 6, 0] = -1

        return A


    def _getd(self):

        d = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return d


    def _getP(self):
        """
        output:
        P: torch.tensor of shape (9, 9)
        """

        P = torch.zeros((9, 9))

        P[0, 0] = 1
        P[1, 3] = 1
        P[2, 6] = 1
        P[4, 4] = 1
        P[5, 7] = 1
        P[8, 8] = 1
        P[3, 1] = 1
        P[6, 2] = 1
        P[7, 5] = 1

        return P









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
    

