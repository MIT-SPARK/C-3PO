"""
This code is an attempt to implement ddn as a torch.autograd.Function

"""

import torch
import cvxpy as cp
import pymanopt as pym
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import numpy as np

import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode, EqConstDeclarativeNode, DeclarativeLayer, ParamDeclarativeFunction
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import shape_error, translation_error, rotation_error


#ToDo: This is what we use.
class PACErotation(EqConstDeclarativeNode):
    """
    This implements the rotation computation in PACE as a declarative node (ddn.node.EqConstDeclarativeNode).

    Note:
        The forward prop is implemented using the solution to the SDP
        The backprop is implemented using the equality constraint optimization problem

        Make sure that this runs on cpu, and not on gpu.
    """
    def __init__(self, weights, model_keypoints, lambda_constant=torch.tensor(1.0), batch_size=32, device='cpu'):
        super().__init__(eps=0.05)
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: torch.tensor of shape (1, 1)

        for the ddn module:
        n: input dimension: 3*N = 3*model_keypoints.shape[-1]
        m: output_dimension: 10
        b: batch size:
        """
        self.w = weights.unsqueeze(0)               # (1, N, 1)
        self.model_keypoints = model_keypoints                    # (K, 3, N)
        self.lambda_constant = lambda_constant      # (1, 1)
        self.device_ = device

        self.N = self.model_keypoints.shape[-1]                   # (1, 1)
        self.K = self.model_keypoints.shape[0]                    # (1, 1)

        self.n = 3*self.N       # input dimension
        self.m = 10             # output dimension
        self.b = batch_size     # batch_size

        self.b_w = self._get_b_w()                  # (1, K, 3)
        self.bar_B = self._get_bar_B()              # (1, 3N, K)
        self.G, self.g, self.M, self.h = self._get_GgMh()
        # G: (1, K, K)
        # g: (1, K, 1)
        # M: (1, 3N+K, 3N)
        # h: (1, 3N+K, 1)

        self.A = self._getA()                       # A: (16, 10, 10)
        self.d = self._getd()                       # d: (16)
        self.P = self._getP()                       # P: (9, 9)


    def solve(self, keypoints):
        """
        input:
        keypoints: torch.tensor of shape (B, 3*N), where B = batch size

        output:
        r: torch.tensor of shape (B, 10), where B = batch size
        """
        # print(keypoints.shape)
        batch_size = keypoints.shape[0]
        # print("kp shape: ", keypoints.shape)
        # print(self.N)
        keypoints = torch.reshape(keypoints, (batch_size, 3, self.N))
        y_w = self._get_y_w(keypoints=keypoints)
        bar_y = self._get_bar_y(keypoints=keypoints, y_w=y_w)

        # R, Qbatch = self._rotation(bar_y=bar_y) # this was for verification
        R, _ = self._rotation(bar_y=bar_y)
        r = self._vecR(R=R)

        # print(r.shape)

        return r, None
        # return r.squeeze(0)


    def equality_constraints(self, keypoints, y):
        #ToDo: for some reason, it is not able to backpropagate
        """
        input:
        keypoints: torch.tensor of shape (B, 3*N), where B = batch size
        y=r: torch.tensor of shape (B, 10), where B = batch size

        intermediate:
        self.A = torch.tensor of shape (16, 10, 10)
        self.d = torch.tensor of shape (16, 1)

        output:
        eq_constraints: torch.tensor of shape (B, 16), where B = batch size and 16 = number of constraints
        """
        return self.batch_equality_constraints(keypoints=keypoints, r=y)


    def single_equality_constraints(self, keypoints, r):
        """
        input:
        keypoints: torch.tensor of shape (3*N)
        r: torch.tensor of shape (10)

        output:
        eq_constraints: torch.tensor of shape (16)
        """
        keypoints = keypoints.unsqueeze(0)
        r = r.unsqueeze(0)
        eq_constraints = self.batch_equality_constraints(keypoints=keypoints, r=r)

        return eq_constraints.squeeze(0)

    def batch_equality_constraints(self, keypoints, r):
        """
        input:
        keypoints: torch.tensor of shape (B, 3*N), where B = batch size
        r: torch.tensor of shape (B, 10), where B = batch size

        intermediate:
        self.A = torch.tensor of shape (16, 10, 10)
        self.d = torch.tensor of shape (16, 1)

        output:
        eq_constraints: torch.tensor of shape (B, 16), where B = batch size and 16 = number of constraints
        """
        batch_size = r.shape[0]
        keypoints = torch.reshape(keypoints, (batch_size, 3, self.N))
        eq_constraints = torch.zeros((batch_size, 16), device=self.device_)

        for constraint_idx in range(16):
            temp_const = r.unsqueeze(1) @ self.A[constraint_idx, :, :].unsqueeze(0) @ r.unsqueeze(-1) - self.d[constraint_idx]
            temp_const = temp_const.squeeze(-1).squeeze(-1)
            eq_constraints[:, constraint_idx] = temp_const

        return eq_constraints


    def objective(self, keypoints, y):
        """
        input:
        keypoints: torch.tensor of shape (B, 3*N), where B = batch size
        r: torch.tensor of shape (B, 10), where B = batch size

        output:
        cost: torch.tensor of shape (B, 1), where B = batch size
        """
        return self.batch_objective(keypoints=keypoints, r=y)


    def single_objective(self, keypoints, r):
        """
        input:
        keypoints: torch.tensor of shape (3*N)
        r: torch.tensor of shape (10)

        output:
        cost: torch.tensor of shape (1)
        """
        keypoints = keypoints.unsqueeze(0)
        r = r.unsqueeze(0)

        cost = self.batch_objective(keypoints=keypoints, r=r)
        return cost.squeeze(0).squeeze(0)



    def batch_objective(self, keypoints, r):
        """
        input:
        keypoints: torch.tensor of shape (B, 3*N), where B = batch size
        r: torch.tensor of shape (B, 10), where B = batch size

        output:
        cost: torch.tensor of shape (B, 1), where B = batch size
        """
        batch_size = keypoints.shape[0]
        keypoints = torch.reshape(keypoints, (batch_size, 3, self.N))
        y_w = self._get_y_w(keypoints=keypoints)
        bar_y = self._get_bar_y(keypoints=keypoints, y_w=y_w)

        Q = self._getQ(bar_y=bar_y)  # shape (B, 10, 10)
        # print('Print Q shape: ', Q.shape)
        # print("Q device: ", Q.device.type)
        # print("r device: ", r.device.type)
        cost = r.unsqueeze(1) @ Q @ r.unsqueeze(-1)

        # print('Cost shape: ', cost.shape)

        return cost.squeeze(-1)


    def _get_y_w(self, keypoints):
        """
        input:
        keypoints: torch.tensor of shape (B, 3, N), where B = batch size

        intermediate:
        self.w: torch.tensor of shape (1, N, 1)

        output:
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size
        """

        return torch.matmul(keypoints, self.w ) /self.w.sum()


    def _getQ(self, bar_y):
        """
        input:
        bar_y: torch.tensor of shape (B, 3, N), where B = batch size

        intermediate:
        self.M: torch.tensor of shape (1, 3N+K, 3N)
        self.h: torch.tensor of shape (1, 3N+K, 1)
        self.P: torch.tensor of shape (9, 9)

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        Qbatch: torch.tensor of shape (B, 10, 10), where B = batch size
        """

        Qbatch = torch.zeros(bar_y.shape[0], 10, 10, device=self.device_)

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
            Q[1:, 1:] = torch.matmul(tempE.T, tempE)

            Qbatch[batch, :, :] = 0.5*(Q + Q.T)

        return Qbatch


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
        Qbatch: torch.tensor of shape (B, 10, 10), where B = batch size
        """

        R = torch.zeros(bar_y.shape[0], 3, 3, device=self.device_)
        Qbatch = torch.zeros(bar_y.shape[0], 10, 10, device=self.device_)

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
            Q[1:, 1:] = torch.matmul(tempE.T, tempE)

            Qbatch[batch, :, :] = Q[:, :] # Qbatch for verification
            tempR = self._get_rotation(Q=0.5*(Q +Q.T))

            R[batch, :, :] = tempR


        return R, Qbatch # Returning Qbatch


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

        # Defining the SDP Layer
        Xvar = cp.Variable((10, 10), symmetric=True)
        Qparam = Q.detach().cpu().numpy()
        # Qparam = cp.Parameter((10, 10), symmetric=True)
        constraints = [Xvar >> 0]
        constraints += [
            cp.trace(self.A[i, :, :].detach().cpu().numpy() @ Xvar) == self.d[i].detach().cpu().numpy() for i in range(16)
        ]
        sdp_for_rotation = cp.Problem(cp.Minimize(cp.trace(Qparam @ Xvar)), constraints=constraints)
        assert sdp_for_rotation.is_dpp()

        # self.sdp_for_rotation = CvxpyLayer(self.sdp_for_rotation, parameters=[Q], variables=[X])

        # Step (1)
        # Qparam.value = Q.detach().cpu().numpy()
        sol = sdp_for_rotation.solve()
        # print("-"*40)
        # print("Problem status: ", sdp_for_rotation.status)
        # print("Optimal value: ", sdp_for_rotation.value)
        # print("Optimal variable: ", Xvar.value)
        # print("Qparam: ", Qparam)
        # print("-"*40)
        X = torch.from_numpy(Xvar.value)
        X = X.to(device=self.device_)

        # Step (2): computes rotation matrix from X
        ev, evec = torch.linalg.eigh(X)
        idx = torch.argsort(ev)
        evmax = ev[idx[-1]]
        evsmax = ev[idx[-2]]
        vec = evec[:, idx[-1]]
        vec = vec / vec[0]
        r = vec[1:]
        Atemp = torch.reshape(r, (3, 3)).T

        # Projecting A to SO(3) to get R
        # Note: should ideally pass R, but R and A tend to be same!!
        # Note: this helps compute the correct gradient of R (on SO(3)) with respect to input parameters
        U, S, Vh = torch.linalg.svd(Atemp)
        R = torch.matmul(U, Vh)
        if torch.linalg.det(R) < 0:
            R = torch.matmul(torch.matmul(U, torch.diag(torch.tensor([1, 1, -1]))), Vh)

        return R


    def _vecR(self, R):
        """
        input:
        R: torch.tensor of shape (B, 3, 3)

        where
        B = batch size

        output:
        r: torch.tensor of shape (B, 10)
        """
        batch_size = R.shape[0]
        r = torch.ones((batch_size, 10), device=self.device_)
        r[:, 1:] = torch.reshape(torch.transpose(R, 1, 2), (batch_size, 9))
        return r


    def _get_b_w(self):
        """
        intermediate:
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)

        output:
        b_w: torch.tensor of shape (1, K, 3)
        """

        b_w = torch.matmul(self.model_keypoints, self.w ) /self.w.sum() # (K, 3, 1)

        return b_w.squeeze(-1).unsqueeze(0) # (1, K, 3)


    def _get_bar_y(self, keypoints, y_w):
        """
        input:
        keypoints: torch.tensor of shape (B, 3, N), where B = batch size
        y_w: torch.tensor of shape (B, 3, 1), where B = batch size

        intermediate:
        self.w: torch.tensor of shape (1, N, 1)

        output:
        bar_y: torch.tensor of shape (B, 3, N), where B = batch size
        """

        return torch.sqrt(torch.transpose(self.w, -1, -2)) * (keypoints -y_w)


    def _get_bar_B(self):
        """
        intermediate:
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)
        b_w: torch.tensor of shape (1, K, 3)

        output:
        bar_B: torch.tensor of shape (1, 3N, K), where B = batch size
        """

        bar_b = torch.sqrt(torch.transpose(self.w, -1, -2) ) *(self.model_keypoints - self.b_w.squeeze(0).unsqueeze(-1)) # (K, 3, N)
        bar_B = torch.transpose(bar_b, -1, -2).reshape(bar_b.shape[0], bar_b.shape[-1 ] *bar_b.shape[-2], 1) # (K, 3N, 1)
        bar_B = bar_B.squeeze(-1) # (K, 3N)
        bar_B = torch.transpose(bar_B, -1, -2) # (3N, K)

        return bar_B.unsqueeze(0)  # (1, 3N, K)


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

        G = bar_Hinv - (torch.matmul(Htemp, Htemp.T) ) / \
            (torch.matmul(torch.ones(1, Htemp.shape[0], device=self.device_), Htemp)) # (K, K)
        g = Htemp/(torch.matmul(torch.ones(1, Htemp.shape[0], device=self.device_), Htemp))  # (K, 1)

        M = torch.zeros( 3 *self.N + self.K, 3* self.N, device=self.device_)  # (3N+K, 3N)
        h = torch.zeros(3 * self.N + self.K, 1, device=self.device_)  # (3N+K, 1)

        M[0:3 * self.N, :] = 2 * torch.matmul(bar_B, torch.matmul(G, bar_B.T)) - torch.eye(3 * self.N,
                                                                                           device=self.device_)
        M[3 * self.N:, :] = 2 * torch.sqrt(self.lambda_constant) * torch.matmul(G, bar_B.T)

        h[0:3 * self.N, :] = torch.matmul(self.bar_B, g)
        h[3 * self.N:, :] = torch.sqrt(
            self.lambda_constant) * g  # This is something that was added later by Jingnan in his code.
        # Jingnan: 2021 Oct-11: added missing sqrt(lam) before g

        return G.unsqueeze(0), g.unsqueeze(0), M.unsqueeze(0), h.unsqueeze(0)

    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10, device=self.device_)

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

        A[5, 1, 7] = 1  #
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

        d = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device_)
        return d

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


class PACEddn(nn.Module):
    """
    PACE implementation (as torch.nn.Module) using the declarative node PACErotation.

    Note:
        The rotation computation is implemented as a declarative node (ddn.node.EqConstDeclarativeNode)
        Requires PACErotation(EqConstDeclarativeNode)

        Make sure that this runs on cpu, and not on gpu.
    """
    def __init__(self, weights, model_keypoints, batch_size=32):
        super().__init__()
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: torch.tensor of shape (1, 1)
        """

        self.w = weights.unsqueeze(0)               # (1, N, 1)
        self.model_keypoints = model_keypoints                    # (K, 3, N)
        self.device_ = self.model_keypoints.device

        self.N = self.model_keypoints.shape[-1]                   # (1, 1)
        self.K = self.model_keypoints.shape[0]                    # (1, 1)
        self.batch_size = batch_size
        # self.lambda_constant = torch.tensor(np.sqrt(self.K/self.N)).float()
        self.lambda_constant = torch.tensor([1.0])

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


        # Rotation ddn layer
        pace_rotation_ddn = PACErotation(weights=weights, model_keypoints=model_keypoints,
                                         lambda_constant=self.lambda_constant, batch_size=self.batch_size,
                                         device=self.device_)
        self.pace_rotation_ddn_layer = DeclarativeLayer(pace_rotation_ddn)
        # self.pace_rotation_ddn_fn = ParamDeclarativeFunction(pace_rotation_ddn)

    def forward(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size
        """
        batch_size = y.shape[0]
        y_w = self._get_y_w(y=y)
        bar_y = self._get_bar_y(y=y, y_w=y_w)

        # R, Qbatch = self._rotation(bar_y=bar_y) # this was for verification
        # R, _ = self._rotation(bar_y=bar_y)
        z = bar_y.view(batch_size, -1)
        # print(z.shpae)
        r = self.pace_rotation_ddn_layer(z)
        # r = self.pace_rotation_ddn_fn.forward(z)  # out: (B, 10), in: B, 3N

        R = torch.transpose(torch.reshape(r[:, 1:], (batch_size, 3, 3)), -1, -2)
        # print(R.shape)
        # print(bar_y.shape)
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

        R = torch.zeros(bar_y.shape[0], 3, 3, device=self.device_)
        Qbatch = torch.zeros(bar_y.shape[0], 10, 10, device=self.device_)

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

            Qbatch[batch, :, :] = Q[:, :] # Qbatch for verification
            tempR = self._get_rotation(Q=0.5*(Q+Q.T))

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
        X, = self.sdp_for_rotation(Q.cpu())
        X = X.to(device=self.device_)

        # Step (2): computes rotation matrix from X
        ev, evec = torch.linalg.eigh(X)
        idx = torch.argsort(ev)
        evmax = ev[idx[-1]]
        evsmax = ev[idx[-2]]
        vec = evec[:, idx[-1]]
        vec = vec / vec[0]
        r = vec[1:]
        Atemp = torch.reshape(r, (3,3)).T

        # Projecting A to SO(3) to get R
        # Note: should ideally pass R, but R and A tend to be same!!
        # Note: this helps compute the correct gradient of R (on SO(3)) with respect to input parameters
        U, S, Vh = torch.linalg.svd(Atemp)
        R = torch.matmul(U, Vh)
        if torch.linalg.det(R) < 0:
            R = torch.matmul(torch.matmul(U, torch.diag(torch.tensor([1, 1, -1]))), Vh)

        return R


    def _get_b_w(self):
        """
        intermediate:
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)

        output:
        b_w: torch.tensor of shape (1, K, 3)
        """

        b_w = torch.matmul(self.model_keypoints, self.w)/self.w.sum() # (K, 3, 1)

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
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)
        b_w: torch.tensor of shape (1, K, 3)

        output:
        bar_B: torch.tensor of shape (1, 3N, K), where B = batch size
        """

        bar_b = torch.sqrt(torch.transpose(self.w, -1, -2))*(self.model_keypoints - self.b_w.squeeze(0).unsqueeze(-1)) # (K, 3, N)
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


    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10, device=self.device_)

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

        d = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device_)
        return d


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


#ToDo: This is what we use.
class PACEbp():
    """
    PACE implementation as a differentiable function. The class parameterizes the PACE function.

    Note:
        The rotation computation is implemented as a declarative node (ddn.node.EqConstDeclarativeNode)
        Requires PACErotation(EqConstDeclarativeNode)
        *The PACErotation declarative node gets implemented here as a parametric torch.autograd.function
        *This(^) makes PACEbp a function that can be used just as any other differentiable function

        Make sure that this runs on cpu, and not on gpu.
    """
    def __init__(self, weights, model_keypoints, batch_size=32):
        super().__init__()
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: torch.tensor of shape (1, 1)
        """

        self.w = weights.unsqueeze(0)               # (1, N, 1)
        self.model_keypoints = model_keypoints                    # (K, 3, N)
        self.device_ = self.model_keypoints.device

        self.N = self.model_keypoints.shape[-1]                   # (1, 1)
        self.K = self.model_keypoints.shape[0]                    # (1, 1)
        self.batch_size = batch_size
        # self.lambda_constant = torch.tensor([np.sqrt(self.K/self.N)]).float()
        self.lambda_constant = torch.tensor([1.0])

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


        # Rotation ddn layer
        pace_rotation_ddn = PACErotation(weights=weights, model_keypoints=self.model_keypoints,
                                         lambda_constant=self.lambda_constant, batch_size=self.batch_size)
        # self.pace_rotation_ddn_layer = DeclarativeLayer(pace_rotation_ddn)
        self.pace_rotation_ddn_fn = ParamDeclarativeFunction(pace_rotation_ddn)

    def forward(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size
        """
        batch_size = y.shape[0]
        y_w = self._get_y_w(y=y)
        bar_y = self._get_bar_y(y=y, y_w=y_w)

        # R, Qbatch = self._rotation(bar_y=bar_y) # this was for verification
        # R, _ = self._rotation(bar_y=bar_y)
        z = bar_y.view(batch_size, -1)
        # print(z.shpae)
        # r = self.pace_rotation_ddn_layer(z)
        r = self.pace_rotation_ddn_fn.forward(z)  # out: (B, 10), in: B, 3N

        R = torch.transpose(torch.reshape(r[:, 1:], (batch_size, 3, 3)), -1, -2)
        # print(R.shape)
        # print(bar_y.shape)
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

        R = torch.zeros(bar_y.shape[0], 3, 3, device=self.device_)
        Qbatch = torch.zeros(bar_y.shape[0], 10, 10, device=self.device_)

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

            Qbatch[batch, :, :] = Q[:, :] # Qbatch for verification
            tempR = self._get_rotation(Q=0.5*(Q+Q.T))

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
        X, = self.sdp_for_rotation(Q.cpu())
        X = X.to(device=self.device_)

        # Step (2): computes rotation matrix from X
        ev, evec = torch.linalg.eigh(X)
        idx = torch.argsort(ev)
        evmax = ev[idx[-1]]
        evsmax = ev[idx[-2]]
        vec = evec[:, idx[-1]]
        vec = vec / vec[0]
        r = vec[1:]
        Atemp = torch.reshape(r, (3,3)).T

        # Projecting A to SO(3) to get R
        # Note: should ideally pass R, but R and A tend to be same!!
        # Note: this helps compute the correct gradient of R (on SO(3)) with respect to input parameters
        U, S, Vh = torch.linalg.svd(Atemp)
        R = torch.matmul(U, Vh)
        if torch.linalg.det(R) < 0:
            R = torch.matmul(torch.matmul(U, torch.diag(torch.tensor([1, 1, -1]))), Vh)

        return R


    def _get_b_w(self):
        """
        intermediate:
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)

        output:
        b_w: torch.tensor of shape (1, K, 3)
        """

        b_w = torch.matmul(self.model_keypoints, self.w)/self.w.sum() # (K, 3, 1)

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
        self.model_keypoints: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)
        b_w: torch.tensor of shape (1, K, 3)

        output:
        bar_B: torch.tensor of shape (1, 3N, K), where B = batch size
        """

        bar_b = torch.sqrt(torch.transpose(self.w, -1, -2))*(self.model_keypoints - self.b_w.squeeze(0).unsqueeze(-1)) # (K, 3, N)
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


    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10, device=self.device_)

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

        d = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device_)
        return d


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

    # Test: PACErotation(EqConstDeclarativeNode)
    print('Testing PACErotation(EqConstDeclarativeNode)')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    N = 10
    K = 4
    n = 40
    B = 20
    weights = torch.rand(N, 1).to(device=device)
    model_keypoints = torch.rand(K, 3, N).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)
    cad_models = torch.rand(K, 3, n).to(device=device)

    pace_rotation_ddn = PACErotation(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant,
                                     batch_size=B, device=device)
    pace_rotation = DeclarativeLayer(pace_rotation_ddn).to(device=device)

    keypoints, rotations, translations, shape = generate_random_keypoints(batch_size=B,
                                                                          model_keypoints=model_keypoints.to('cpu'))
    keypoints = keypoints.to(device=device)
    rotations = rotations.to(device=device)
    translations = translations.to(device=device)
    shape = shape.to(device=device)

    keypoints.requires_grad = True
    z = keypoints.view(B, -1).to(device=device)
    print(z.shape)
    rot_estX = pace_rotation(z)
    batch_size = rot_estX.shape[0]
    rot_est = torch.transpose(torch.reshape(rot_estX[:, 1:], (batch_size, 3, 3)), -1, -2).to(device=device) # (B, 3, 3)
    print(rot_est.shape)

    er_rot = rotation_error(rotations, rot_est)
    #
    print("rotation error: ", er_rot.mean())
    #

    # loss = pace_rotation_ddn.objective(keypoints=torch.rand(5, 3*N).to(device=device), y=torch.rand(5, 10).to(device=device))
    # print(loss.shape)
    # print(loss)
    # # loss = loss.squeeze(0)
    # # print(loss.shape)
    # # loss = loss.squeeze(0)
    # # print(loss.shape)
    #
    # eq_const = pace_rotation_ddn.equality_constraints(keypoints=z.to(device=device), y=rot_estX.to(device=device))
    # print(eq_const.shape)
    # print(z.mean())
    #
    #
    # eq_const = pace_rotation_ddn.equality_constraints(keypoints=torch.rand(5, 3 * N).to(device=device), y=torch.rand(5, 10).to(device=device))
    # print(eq_const.shape)

    print(er_rot.device.type)
    loss = er_rot.mean().to(device=device)
    print("Shpae of loss: ", loss.shape)
    print(loss.device.type)
    loss.backward()

    print(keypoints.grad)
    print('-' * 20)



    # Test: PACEddn(torch.nn.Module)
    print('Testing PACEddn(torch.nn.Module)')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    B = 20
    N = 10
    K = 4
    n = 40
    weights = torch.rand(N, 1).to(device=device)
    model_keypoints = torch.rand(K, 3, N).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)
    cad_models = torch.rand(K, 3, n).to(device=device)

    pace_model = PACEddn(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant, batch_size=B).to(device=device)

    keypoints, rotations, translations, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints.to('cpu'))
    keypoints = keypoints.to(device=device)
    rotations = rotations.to(device=device)
    translations = translations.to(device=device)
    shape = shape.to(device=device)

    keypoints.requires_grad = True
    rot_est, trans_est, shape_est = pace_model(keypoints)

    er_shape = shape_error(shape, shape_est)
    er_trans = translation_error(translations, trans_est)
    er_rot = rotation_error(rotations, rot_est)

    print("rotation error: ", er_rot.mean())
    print("translation error: ", er_trans.mean())
    print("shape error: ", er_shape.mean())

    loss = er_shape.mean() + er_trans.mean() + er_rot.mean()
    loss.backward()

    print(keypoints.grad)
    print('-'*20)


    # Test: PACEbp()
    print('Testing PACEbp()')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    B = 20
    N = 10
    K = 4
    n = 40
    weights = torch.rand(N, 1).to(device=device)
    model_keypoints = torch.rand(K, 3, N).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)
    cad_models = torch.rand(K, 3, n).to(device=device)

    pace_model = PACEbp(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant, batch_size=B)

    keypoints, rotations, translations, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints.to('cpu'))
    keypoints = keypoints.to(device=device)
    rotations = rotations.to(device=device)
    translations = translations.to(device=device)
    shape = shape.to(device=device)

    keypoints.requires_grad = True
    rot_est, trans_est, shape_est = pace_model.forward(y=keypoints)

    er_shape = shape_error(shape, shape_est)
    er_trans = translation_error(translations, trans_est)
    er_rot = rotation_error(rotations, rot_est)

    print("rotation error: ", er_rot.mean())
    print("translation error: ", er_trans.mean())
    print("shape error: ", er_shape.mean())

    loss = er_shape.mean() + er_trans.mean() + er_rot.mean()
    loss.backward()

    print(keypoints.grad)
    print('-'*20)
