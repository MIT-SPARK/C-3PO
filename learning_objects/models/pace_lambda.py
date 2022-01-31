"""
This code implements a function that tunes the lambda_constant for PACE

"""

import torch
from scipy import optimize
import numpy as np

import sys
sys.path.append("../../")

from learning_objects.models.pace import PACEmodule


class GetLambda():
    def __init__(self, model_keypoints):
        super().__init__()

        self.model_keypoints = model_keypoints

    def loss(self, c, c_):
        """
        c   : torch.tensor of shape (B, K, 1)
        c_  : torch.tensor of shape (B, K, 1)

        """

        return ((c - c_)**2).mean(dim=1).mean(dim=1)

    def get_shapes(self):
        """
        model_keypoints : torch.tensor of shape (K, 3, N)

        output:
        keypoints   : torch.tensor of shape (B, 3, N)
            where B = len(eta) * K
        shape       : torch.tensor of shape (B, K, 1)
        """
        model_keypoints = self.model_keypoints

        K, _, N = model_keypoints.shape
        device_ = model_keypoints.device
        eta_range = [0.8, 0.9, 0.99]

        model_keypoints = model_keypoints.unsqueeze(0)  # (1, K, 3, N)
        keypoints = torch.zeros(K * len(eta_range), 3, N)
        shape = torch.zeros(K * len(eta_range), K, 1)
        idx = 0
        for eta in eta_range:

            # c of shape (K, K)
            c = ((1 - eta) / (K - 1)) * torch.ones(K, K).to(device=device_)
            c += (eta - (1 - eta) / (K - 1)) * torch.eye(K).to(device=device_)

            c = c.unsqueeze(-1).unsqueeze(-1)   # (K, K, 1, 1)

            kp = torch.einsum('akii,jkdn->adn', c, model_keypoints)
            keypoints[idx * K:(idx + 1) * K, ...] = kp
            shape[idx * K:(idx + 1) * K, ...] = c.squeeze(-1)

            idx += 1

        return keypoints, shape


    def get(self):

        keypoints, shape = self.get_shapes()  # keypoitns: (B, 3, N), shape: (B, K, 1)

        def fun(lam):
            lam = torch.from_numpy(lam).to(torch.float)

            pace = PACEmodule(model_keypoints=self.model_keypoints, lambda_constant=lam)
            _, _, c = pace(keypoints)

            return self.loss(c, shape).mean()

        # Optimize fun as a function of lambda
        lam0 = np.array([1.0])
        loss_now = 100000
        result = optimize.minimize(fun, lam0, method='trust-constr', bounds=((0, 10.0),))

        print("loss before optimization: ", fun(lam0))
        print("loss after optimization: ", result.fun)
        print("opt status: ", result.status)
        print("num of steps: ", result.nit)
        print("corrector optimization successful: ", result.success)
        lambda_constant = torch.from_numpy(result.x).to(torch.float)

        # print("The lambda constant is: ", lambda_constant)

        return lambda_constant


if __name__ == "__main__":

    B = torch.zeros(3, 8).to(torch.float)
    B[:, 0] = torch.tensor([1.0, 1.0, 1.0]).to(torch.float)
    B[:, 1] = torch.tensor([1.0, 1.0, 0.0]).to(torch.float)
    B[:, 2] = torch.tensor([1.0, 0.0, 1.0]).to(torch.float)
    B[:, 3] = torch.tensor([1.0, 0.0, 0.0]).to(torch.float)
    B[:, 4] = torch.tensor([0.0, 1.0, 1.0]).to(torch.float)
    B[:, 5] = torch.tensor([0.0, 1.0, 0.0]).to(torch.float)
    B[:, 6] = torch.tensor([0.0, 0.0, 1.0]).to(torch.float)
    B[:, 7] = torch.tensor([0.0, 0.0, 0.0]).to(torch.float)

    B = B - torch.tensor([0.5, 0.5, 0.5]).unsqueeze(-1).to(torch.float)

    # print(B)

    B0 = B
    B1 = 2*B

    model_keypoints = torch.cat([B0.unsqueeze(0), B1.unsqueeze(0)], dim=0)  # (K, 3, N)
    Lambda = GetLambda(model_keypoints=model_keypoints)
    lambda_constant = Lambda.get()

    print("The lambda constant is: ", lambda_constant)





