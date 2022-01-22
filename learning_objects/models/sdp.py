"""
This implements rotation SDP (single and batch), that is then used in PACE.

"""
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


import os
import sys
sys.path.append("../../")


class RotationSDP():
    def __init__(self):
        super().__init__()

        # Cvxpylaers work on cpu. We transfer the input to cpu, and then transfer it back to the original device.
        self.device_ = 'cpu'

        self.A = self._getA()
        self.d = self._getd()

        # Defining the SDP Layer
        X = cp.Variable((10, 10), symmetric=True)
        Q = cp.Parameter((10, 10), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(self.A[i, :, :] @ X) == self.d[i] for i in range(16)
        ]
        problem = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints=constraints)
        assert problem.is_dpp()

        self.layer = CvxpyLayer(problem, parameters=[Q], variables=[X])

    def forward(self, Q):
        """
        input:
        Q   : torch.tensor of shape (10, 10) or (B, 10, 10)

        output:
        X   : torch.tensor of shape (10, 10) or (B, 10, 10)

        """
        in_device = Q.device

        X, = self.layer(Q.to(device=self.device_))

        return X.to(device=in_device)

    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10).to(device=self.device_)

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


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    rotation_sdp = RotationSDP()

    A = torch.rand(2, 10, 10)
    Q = A @ A.transpose(-1, -2)
    Q = Q.to(device=device)
    Q.requires_grad = True

    X = rotation_sdp.forward(Q)

    loss = (X**2).sum()

    loss.backward()

    print("Loss: ", loss)
    print("Shape of Q.grad: ", Q.grad.shape)
    print("Q.grad: ", Q.grad)

