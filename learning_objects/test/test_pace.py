"""
This code checks the correctness of all PACE implementations. This is done by comparing the input-output pairs with
Jingnan's implementation.

"""
import torch
import os
import sys
sys.path.append("../../")

from learning_objects.models.pace import PACE, PACEmodule
from learning_objects.models.pace_ddn import PACEbp, PACEddn
from learning_objects.utils.category_gnc import solve_3dcat_with_sdp
from learning_objects.utils.general import shape_error, rotation_error, translation_error, check_rot_mat, generate_random_keypoints



class PACEimplementation():
    def __init__(self, model_keypoints, weights=None, lambda_constant=None):
        """
        model_keypoints: torch.tensor of shape (K, 3, N)

        """
        super().__init__()

        self.K = model_keypoints.shape[0]
        self.N = model_keypoints.shape[-1]
        self.model_keypoints = model_keypoints
        self.batch_size = 10

        if weights==None:
            self.weights = torch.ones(self.N, 1)
        else:
            self.weights = weights

        if lambda_constant==None:
            self.lambda_constant = torch.tensor([1.0])
        else:
            self.lambda_constant = lambda_constant


        # Generate data: true data and input keypoints
        y, R, t, c = generate_random_keypoints(batch_size=self.batch_size, model_keypoints=model_keypoints)
        self.input_keypoints = y
        self.R = R          # (B, 3, 3)
        self.t = t          # (B, 3, 1)
        self.c = c          # (B, K, 1)

        # PACE: output from Jingnan's implementation
        self.R_ji = torch.zeros_like(self.R)      # (B, 3, 3)
        self.t_ji = torch.zeros_like(self.t)      # (B, 3, 1)
        self.c_ji = torch.zeros_like(self.c)      # (B, K, 1)

        for batch in range(self.batch_size):
            bR_ji, bt_ji, bc_ji, _, _ = solve_3dcat_with_sdp(tgt=self.input_keypoints[batch, ...].numpy(), cad_kpts=model_keypoints.numpy(),
                                                             weights=self.weights.squeeze(-1).numpy(),
                                                             lam=self.lambda_constant.numpy(),
                                                             print_info=False)
            bR_ji = torch.from_numpy(bR_ji).float()
            bt_ji = torch.from_numpy(bt_ji).unsqueeze(-1).float()
            bc_ji = torch.from_numpy(bc_ji).unsqueeze(-1).float()

            self.R_ji[batch, ...] = bR_ji
            self.t_ji[batch, ...] = bt_ji
            self.c_ji[batch, ...] = bc_ji

        R_ji_err = rotation_error(self.R_ji, self.R)
        t_ji_err = translation_error(self.t_ji, self.t)
        c_ji_err = shape_error(self.c_ji, self.c)

        print("Jingnan's implementation:")
        print("Rotation error: ", R_ji_err.mean())
        print("Translation error: ", t_ji_err.mean())
        print("Shape error: ", c_ji_err.mean())
        print("-" * 20)

    def test(self, fn):
        """
        fn: Callable function.
            Takes in    : keypoints : torch.tensor of shape (B, 3, N)
            Takes out   : R         : torch.tensor of shape (B, 3, 3)
            Takes out   : t         : torch.tensor of shape (B, 3, 1)
            Takes out   : c         : torch.tensor of shape (B, self.K, 1)
        """

        R_fn, t_fn, c_fn = fn(self.input_keypoints)

        # write and print error
        R_fn_err = rotation_error(R_fn, self.R)
        t_fn_err = translation_error(t_fn, self.t)
        c_fn_err = shape_error(c_fn, self.c)

        print("Error in implementation:")
        print("Rotation error: ", R_fn_err.mean())
        print("Translation error: ", t_fn_err.mean())
        print("Shape error: ", c_fn_err.mean())
        print("-" * 20)


        print("Error wrt Jingnan's output: ")
        R_err = rotation_error(R_fn, self.R_ji)
        t_err = translation_error(t_fn, self.t_ji)
        c_err = shape_error(c_fn, self.c_ji)
        print("Rotation error: ", R_err.mean())
        print("Translation error: ", t_err.mean())
        print("Shape error: ", c_err.mean())
        print("-" * 20)








if __name__ == '__main__':

    N = 10
    K = 5
    weights = torch.ones(N, 1)
    lambda_constant = torch.tensor([1.0])
    model_keypoints = torch.rand(K, 3, N)

    print('-' * 22)
    pace_implementation = PACEimplementation(model_keypoints=model_keypoints,
                                             weights=weights, lambda_constant=lambda_constant)

    #
    print('-'*22)
    print("Testing learning_objects.models.pace.PACEmodule")
    print('-' * 22)
    fn = PACEmodule(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)
    pace_implementation.test(fn)

    #
    print('-' * 22)
    print("Testing learning_objects.models.pace.PACE")
    print('-' * 22)
    fn = PACE(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)
    pace_implementation.test(fn.forward)

    #
    print('-' * 22)
    print("Testing learning_objects.models.pace_ddn.PACEddn")
    print('-' * 22)
    fn = PACEddn(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)
    pace_implementation.test(fn)


    #
    print('-' * 22)
    print("Testing learning_objects.models.pace_ddn.PACEbp")
    print('-' * 22)
    fn = PACEbp(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)
    pace_implementation.test(fn.forward)



