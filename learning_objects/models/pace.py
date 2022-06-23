"""
This implements the registration and shape estimation solution from [1], called PACE.

This code provides torch implementations of PACE as a torch.nn.Module.
It uses Cvxpylayers to solve and backprop the rotation SDP.

Note:
[1] Jingnan Shi, Heng Yang, Luca Carlone "Optimal Pose and Shape Estimation for Category-level
    3D Object Perception" RSS 2021.

[2] cvxpylayers: https://github.com/cvxgrp/cvxpylayers

"""

import os
import pickle
import sys
import torch
import torch.nn as nn

sys.path.append("../../")

from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.evaluation_metrics import shape_error, translation_error, rotation_matrix_error
from learning_objects.datasets.shapenet import CLASS_NAME, CLASS_ID

from learning_objects.models.sdp import RotationSDP

PATH_TO_OPTIMIZED_LAMBDA_CONSTANTS = '../../data/KeypointNet/KeypointNet/lambda_constants/'


def is_psd(mat):

    symmetric = mat == mat.transpose(-1, -2)
    symmetric = torch.all(symmetric, dim=-1)
    symmetric = torch.all(symmetric, dim=-1)

    psd = torch.linalg.eigh(mat)[0][:,0]>=0

    return torch.logical_and(symmetric, psd)

class PACEmodule(nn.Module):
    """
    PACE implementation as a nn.Module

    Note:
        The rotation computation is implemented as a cvxpylayer, which is also a nn.Module
    """
    def __init__(self, model_keypoints, weights=None, lambda_constant=torch.tensor(1.0), use_optimized_lambda_constant=False, class_id='03001627'):
        super(PACEmodule, self).__init__()
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: torch.tensor of shape (1, 1)
        """

        self.b = model_keypoints                    # (K, 3, N)
        self.device_ = model_keypoints.device
        if use_optimized_lambda_constant and class_id is not None:
            fp = open(PATH_TO_OPTIMIZED_LAMBDA_CONSTANTS + class_id + '.pkl', 'rb')
            lambda_constant = pickle.load(fp)
            fp.close()
            self.lambda_constant = lambda_constant.to(device=self.device_)
            print("lambda_constant loaded from pkl file:", self.lambda_constant)
        else:
            self.lambda_constant = lambda_constant.to(device=self.device_)  # (1, 1)
        self.N = self.b.shape[-1]                   # (1, 1)
        self.K = self.b.shape[0]                    # (1, 1)

        if weights==None:
            self.w = torch.ones(1, self.N, 1).to(device=self.device_)
        else:
            self.w = weights.unsqueeze(0)  # (1, N, 1)

        self.b_w = self._get_b_w()                  # (1, K, 3)
        self.bar_B = self._get_bar_B()              # (1, 3N, K)
        self.G, self.g, self.M, self.h = self._get_GgMh()
                                                    # G: (1, K, K)
                                                    # g: (1, K, 1)
                                                    # M: (1, 3N+K, 3N)
                                                    # h: (1, 3N+K, 1)

        self.P = self._getP()                       # P: numpy.array

        self.rotation_sdp = RotationSDP()

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
        R = self._rotation(bar_y=bar_y)
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
        batch_size = bar_y.shape[0]

        Qbatch = torch.zeros(batch_size, 10, 10, device=self.device_)

        #ToDo: Compute this in one go, without using the loop.
        M = self.M.squeeze(0)
        h = self.h.squeeze(0)

        for batch in range(bar_y.shape[0]):

            Y = bar_y[batch, :, :]

            Q = torch.zeros(10, 10, device=self.device_)
            Q[0, 0] = torch.matmul(h.T, h)
            tempA = torch.matmul(h.T, M)
            tempB = Y.T
            tempB = tempB.contiguous()
            tempB = torch.kron(tempB, torch.eye(3, device=self.device_))
            tempC = torch.matmul(tempB, self.P)
            Q[0, 1:] = torch.matmul(tempA, tempC)
            Q[1:, 0] = Q[0, 1:].T

            tempD = torch.matmul(M, tempB)
            tempE = torch.matmul(tempD, self.P)
            Q[1:, 1:] =  torch.matmul(tempE.T, tempE)

            Qbatch[batch, :, :] = Q[:, :]

        # New code to get R from Qbatch:
        X = self.rotation_sdp.forward(Qbatch)
        ev, evec = torch.linalg.eigh(X)
        idx = torch.argsort(ev, -1)
        b = range(batch_size)
        evmax = ev[b, idx[b, -1]]
        evsmax = ev[b, idx[b, -2]]
        vec = evec[b, :, idx[b, -1]]
        vec = vec / vec[:, 0].unsqueeze(-1)
        r = vec[:, 1:]

        # Atemp = torch.reshape(r, (batch_size, 3, 3)).transpose(-1, -2)
        # Projecting A to SO(3) to get R
        # Note: should ideally pass R, but R and A tend to be same!!
        # Note: this helps compute the correct gradient of R (on SO(3)) with respect to input parameters
        # U, S, Vh = torch.linalg.svd(Atemp)
        # R = U @ Vh
        # flag = (torch.linalg.det(R) < 0).unsqueeze(-1).unsqueeze(-1)
        # D = torch.diag(torch.tensor([1.0, 1.0, -1.0])).repeat(batch_size, 1, 1).to(device=self.device_)
        # R = (U @ D @ Vh)*flag \
        #     + torch.logical_not(flag)*R
        R = torch.reshape(r, (batch_size, 3, 3)).transpose(-1, -2)

        # Qpsd_flag = is_psd(Qbatch)
        # Qrank = torch.linalg.matrix_rank(Qbatch)
        # Rcheck = torch.norm(R @ R.transpose(-1, -2) - torch.eye(3).to(device=self.device_), dim=(-1, -2), p='fro')
        # print("Q is PSD: ", Qpsd_flag)
        # print("Q rank: ", Qrank)
        # print("Rotation check (0 = correct): ", Rcheck)

        return R

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
        tempL = torch.kron(torch.eye(self.N, device=self.device_), tempK)
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

        bar_H = 2 * (torch.matmul(bar_B.T, bar_B) + self.lambda_constant*torch.eye(self.K, device=self.device_))
        bar_Hinv = torch.inverse(bar_H)
        Htemp = torch.matmul(bar_Hinv, torch.ones(bar_Hinv.shape[-1], 1, device=self.device_))

        G = bar_Hinv - (torch.matmul(Htemp, Htemp.T))/(torch.matmul(torch.ones(1, Htemp.shape[0], device=self.device_), Htemp)) # (K, K)
        g = Htemp/(torch.matmul(torch.ones(1, Htemp.shape[0], device=self.device_), Htemp))  # (K, 1)

        M = torch.zeros(3*self.N + self.K, 3*self.N, device=self.device_)   # (3N+K, 3N)
        h = torch.zeros(3*self.N + self.K, 1, device=self.device_)   # (3N+K, 1)

        M[0:3*self.N, :] = 2*torch.matmul( bar_B, torch.matmul(G, bar_B.T) ) - torch.eye(3*self.N, device=self.device_)
        M[3*self.N:, :] = 2*torch.sqrt(self.lambda_constant)*torch.matmul(G, bar_B.T)

        h[0:3*self.N, :] = torch.matmul(self.bar_B, g)
        h[3*self.N:, :] = torch.sqrt(self.lambda_constant)*g # This is something that was added later by Jingnan in his code.
                                                             # Jingnan: 2021 Oct-11: added missing sqrt(lam) before g

        return G.unsqueeze(0), g.unsqueeze(0), M.unsqueeze(0), h.unsqueeze(0)

    def _getP(self):
        """
        output:
        P: torch.tensor of shape (9, 9)
        """

        P = torch.zeros((9, 9), device=self.device_)

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


if __name__ == "__main__":

    # Test: PACEmodule
    print('Testing PACEmodule(torch.nn.Module)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    N = 10
    K = 4
    n = 40
    weights = torch.rand(N, 1).to(device=device)
    model_keypoints = torch.rand(K, 3, N).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)
    cad_models = torch.rand(K, 3, n).to(device=device)

    pace_model = PACEmodule(weights=weights, model_keypoints=model_keypoints,
                            lambda_constant=lambda_constant).to(device=device)

    B = 200
    keypoints, rotations, translations, shape = generate_random_keypoints(batch_size=B,
                                                                          model_keypoints=model_keypoints.to('cpu'))
    keypoints = keypoints.to(device=device)
    rotations = rotations.to(device=device)
    translations = translations.to(device=device)
    shape = shape.to(device=device)

    keypoints.requires_grad = True
    rot_est, trans_est, shape_est = pace_model(keypoints)

    er_shape = shape_error(shape, shape_est)
    er_trans = translation_error(translations, trans_est)
    er_rot = rotation_matrix_error(rotations, rot_est)

    print("rotation error: ", er_rot.mean())
    print("translation error: ", er_trans.mean())
    print("shape error: ", er_shape.mean())

    loss = er_shape.mean() + er_trans.mean() + er_rot.mean()
    loss.backward()

    print(keypoints.grad.shape)

    grad_ = keypoints.grad
    nan_flag = torch.isnan(grad_)
    inf_flag = torch.isinf(grad_)

    nan_flag = torch.any(nan_flag, dim=-1)
    nan_flag = torch.any(nan_flag, dim=-1)
    print("Nan flag: ", nan_flag)

    inf_flag = torch.any(inf_flag, dim=-1)
    inf_flag = torch.any(inf_flag, dim=-1)
    print("Inf flag: ", inf_flag)

    print('-'*20)
