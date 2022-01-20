"""
This implements the keypoint correction with registration, pace, as (1) a class and (2) as an AbstractDeclarativeNode

"""
import time

import torch
import open3d as o3d
import numpy as np
from scipy import optimize
from pytorch3d import ops
from pytorch3d.loss import chamfer_distance as pyt_chamfer_distance
import torch.nn as nn
import cvxpy as cp
import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode

from learning_objects.models.point_set_registration import point_set_registration, PointSetRegistration
from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, SE3nIsotorpicShapePointCloud

from learning_objects.utils.general import pos_tensor_to_o3d, display_two_pcs
from learning_objects.utils.general import chamfer_distance, chamfer_half_distance, rotation_error, \
    translation_error, shape_error

# from learning_objects.models.pace_ddn import PACEbp
from learning_objects.models.pace_altern_ddn import PACEbp
from learning_objects.models.modelgen import ModelFromShape


#ToDo: I decided to drop padding. We instead remove zero points from the input point cloud in loss computation. See
# chamfer_loss. Therefore, we need to remove padding variables from the code at some point in time. This doesn't hurt
# us as the code works with padding=None.


# def display_two_pcs(pc1, pc2):
#     """
#     pc1 : torch.tensor of shape (3, n)
#     pc2 : torch.tensor of shape (3, m)
#     """
#     pc1 = pc1.to('cpu')
#     pc2 = pc2.to('cpu')
#
#     object1 = pos_tensor_to_o3d(pos=pc1)
#     object2 = pos_tensor_to_o3d(pos=pc2)
#
#     object1.paint_uniform_color([0.8, 0.0, 0.0])
#     object2.paint_uniform_color([0.0, 0.0, 0.8])
#
#     o3d.visualization.draw_geometries([object1, object2])
#
#     return None


def chamfer_loss_with_surface_normals(pc, pc_):
    #ToDo: Not used. If used, need to make sure to get rid of all zero points in pc, when computing loss. See
    # chamfer_loss.
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)

    output:
    loss    :
    """

    normals = ops.estimate_pointcloud_normals(pc.transpose(-1, -2))
    normals_ = ops.estimate_pointcloud_normals(pc_.transpose(-1, -2))

    loss, loss_normals = pyt_chamfer_distance(x=pc.transpose(-1, -2),
                                              y=pc_.transpose(-1, -2),
                                              x_normals=normals, y_normals=normals_)

    # Using surface normals in the loss function, takes a long time, but
    # converges correctly. Not much orientation errors; provided you weigh the loss_normals correctly.
    return loss + 0.01*loss_normals


def chamfer_loss(pc, pc_, pc_padding=None):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in

    output:
    loss    :
    """

    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)
        # pc_padding = torch.zeros(batch_size, n).to(device=device_)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)
    loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)


def keypoints_loss(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='none')

    return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)


def registration_eval(R, R_, t, t_):
    """
    inputs:
    R, R_   : torch.tensor of shape (B, 3, 3)
    t, t_   : torch.tensor of shape (B, 3, 1)

    output:
    loss    : torch.tensor of shape (B, 1)
    """

    return rotation_error(R, R_) + translation_error(t, t_)

def pace_eval(R, R_, t, t_, c, c_):
    """
    inputs:
    R, R_   : torch.tensor of shape (B, 3, 3)
    t, t_   : torch.tensor of shape (B, 3, 1)

    output:
    loss    : torch.tensor of shape (B, 1)
    """

    return rotation_error(R, R_) + translation_error(t, t_) + shape_error(c, c_)


def keypoint_perturbation(keypoints_true, var=0.8, type='uniform', fra=0.2):
    """
    inputs:
    keypoints_true  :  torch.tensor of shape (B, 3, N)
    var             :  float
    type            : 'uniform' or 'sporadic'
    fra             :  float    : used if type == 'sporadic'

    output:
    detected_keypoints  : torch.tensor of shape (B, 3, N)
    """
    device_ = keypoints_true.device

    if type=='uniform':
        # detected_keypoints = keypoints_true + var*torch.randn_like(keypoints_true)
        detected_keypoints = keypoints_true + var * (torch.rand(size=keypoints_true.shape).to(device=device_)-0.5)

        return detected_keypoints

    elif type=='sporadic':
        mask = (torch.rand(size=keypoints_true.shape).to(device=device_) < fra).int().float()
        # detected_keypoints = keypoints_true + var*torch.randn_like(keypoints_true)*mask
        detected_keypoints = keypoints_true + var * (torch.rand(size=keypoints_true.shape).to(device=device_)-0.5) * mask

        return detected_keypoints

    else:
        return None


class kp_corrector_reg():
    def __init__(self, cad_models, model_keypoints, theta=50.0, kappa=10.0):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, m)
        model_keypoints : torch.tensor of shape (1, 3, N)
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.theta = theta
        self.kappa = kappa

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

    def forward(self, detected_keypoints, input_point_cloud, padding=None):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        padding             : None or torch.tensor of shape (B, m) dtype=torch.bool
                                        where True means it is a padded point to be ignored in chamfer loss computation.

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        # correction = self.solve_algo1(detected_keypoints, input_point_cloud)
        correction = self.solve_algo2(detected_keypoints, input_point_cloud, padding)

        return correction

    def objective(self, detected_keypoints, input_point_cloud, correction, padding=None):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)
        padding             : None or torch.tensor of shape (B, m) dtype=torch.bool
                                        where True means it is a padded point to be ignored in chamfer loss computation.

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """

        #Note: kappa = 100.0 and theta = 1.0 showed promising results. It resuled in no orientation errors in registration.

        R, t = self.point_set_registration_fn.forward(detected_keypoints+correction)
        model_estimate = R @ self.cad_models + t
        keypoint_estimate = R @ self.model_keypoints + t

        loss_pc = chamfer_loss(pc=input_point_cloud, pc_=model_estimate, pc_padding=padding)
        # loss_pc = chamfer_loss_with_surface_normals(pc=input_point_cloud, pc_=model_estimate)

        loss_kp = keypoints_loss(kp=detected_keypoints+correction, kp_=keypoint_estimate)
        # loss_kp = 0.0

        return self.kappa*loss_pc + self.theta*loss_kp


    def _get_objective_jacobian(self, fun, correction):
        #ToDo: Move this inside solve_algo2(). This is not used anywhere else.
        batch_size = correction.shape[0]
        dfdcorrection = torch.zeros_like(correction)

        # Do not set create_graph=True in jacobian. It will slow down computation substantially.
        dfdcorrectionX = torch.autograd.functional.jacobian(fun, correction)
        b = range(batch_size)
        dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

        return dfdcorrection

    def solve_algo2(self, detected_keypoints, input_point_cloud, padding=None, lr=0.1, max_iterations=1000, tol=1e-12):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """
        N = detected_keypoints.shape[-1]
        batch_size = detected_keypoints.shape[0]

        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x, padding)

        correction = torch.zeros_like(detected_keypoints)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr

        iter = 0
        obj_ = f(correction)
        flag = torch.ones_like(obj_).to(dtype=torch.bool)
        # flag_idx = flag.nonzero()
        flag = flag.unsqueeze(-1).repeat(1, 3, N)
        while iter < max_iterations:

            iter += 1
            obj = obj_

            dfdcorrection = self._get_objective_jacobian(f, correction)
            correction -= lr*dfdcorrection*flag
            # correction -= lr * dfdcorrection

            obj_ = f(correction)

            if (obj-obj_).abs().max() < tol:
                break
            else:
                flag = (obj-obj_).abs() > tol
                # flag_idx = flag.nonzero()
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

        return correction

    def gradient(self, detected_keypoints, input_point_cloud, correction, output_grad=None):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)
        output_grad         : torch.tensor of shape (B, 3, N)

        outputs:
        input_gradient  : (torch.tensor(B, 3, N), None)

        """
        batch_size, _, N = detected_keypoints.shape
        if output_grad==None:
            output_grad = torch.ones_like(correction)

        f = lambda x, u: self.objective(x, input_point_cloud, u)


        def DfDx(x, u):

            Jf = torch.autograd.functional.jacobian(f, (x, u))

            b = range(batch_size)
            fx = Jf[0][b, 0, b, ...]
            fu = Jf[1][b, 0, b, ...]

            return fx

        def DfDu(x, u):

            Jf = torch.autograd.functional.jacobian(f, (x, u))

            b = range(batch_size)
            fu = Jf[1][b, 0, b, ...]

            return fu

        fx = DfDx(detected_keypoints, correction)
        fu = DfDu(detected_keypoints, correction)

        print("Shape of fx: ", fx.shape)
        print("Shape of fu: ", fu.shape)

        print("fx: ")
        print(fx)               #ToDo: These gradients are also very small!!
        print("fu: ")
        print(fu)
        print("fx-fu:")
        print(fx-fu)


        Jfx = torch.autograd.functional.jacobian(DfDx, (detected_keypoints, correction))
        Jfu = torch.autograd.functional.jacobian(DfDu, (detected_keypoints, correction))

        print("Shape of Jfx[0]: ", Jfx[0].shape)
        print("Shape of Jfx[1]: ", Jfx[1].shape)
        print("Shape of Jfu[0]: ", Jfu[0].shape)
        print("Shape of Jfu[1]: ", Jfu[1].shape)

        b = range(batch_size)
        fxx = Jfx[0][b, :, :, b, :, :]
        fxu = Jfx[1][b, :, :, b, :, :]
        fux = Jfu[0][b, :, :, b, :, :]
        fuu = Jfu[1][b, :, :, b, :, :]

        print("Shape of fxx: ", fxx.shape)
        print("Shape of fxu: ", fxu.shape)
        print("Shape of fux: ", fux.shape)
        print("Shape of fuu: ", fuu.shape)

        fxu = fxu.reshape(batch_size, 3*N, 3*N)
        fuu = fuu.reshape(batch_size, 3*N, 3*N)

        print("After reshaping: ")
        print("Shape of fxu: ", fxu.shape)
        print("Shape of fuu: ", fuu.shape)

        # now solve fuu DuDx = -fxu             #ToDo: This does not solve correctly. The fuu matrix is singular!
        for i in range(batch_size):
            rank = torch.matrix_rank(fuu[i, ...])
            print("Rank of matrix ", i, ": ", rank)     #ToDo: It outputs that the rank of each matrix is zero! Why is this zero rank?
            # print("The matrix fuu[i, ...]")
            # print(fuu[i, ...])

        X = torch.linalg.solve(fuu, -fxu)
        print(X.shape)





        return None



    def objective_numpy(self, detected_keypoints, input_point_cloud, padding, correction):
        # ToDo: Have to change for padding.
        """
        inputs:
        detected_keypoints  : numpy.ndarray of shape (3, N)
        input_point_cloud   : numpy.ndarray of shape (3, m)
        correction          : numpy.ndarray of shape (3*N,)

        output:
        loss    : numpy.ndarray of shape (1,)
        """
        N = detected_keypoints.shape[-1]
        correction = correction.reshape(3, N)

        detected_keypoints = torch.from_numpy(detected_keypoints).unsqueeze(0).to(torch.float)
        input_point_cloud = torch.from_numpy(input_point_cloud).unsqueeze(0).to(torch.float)
        correction = torch.from_numpy(correction).unsqueeze(0).to(torch.float)

        loss = self.objective(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud,
                              padding=padding, correction=correction)

        return loss.squeeze(0).numpy()

    def solve_algo1(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
        #ToDo: Have to change for padding.
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        #ToDo: This corrects and registers the point clout very well. Two problems:
        # 1. It changes the orientation of the object sometimes
        # 2. It takes a few seconds to converge, which may be hard for a real-time operation

        N = detected_keypoints.shape[-1]
        batch_size = detected_keypoints.shape[0]
        correction = torch.zeros_like(detected_keypoints)

        with torch.enable_grad():

            for batch in range(batch_size):

                kp = detected_keypoints[batch, ...]
                pc = input_point_cloud[batch, ...]
                kp = kp.clone().detach().to('cpu').numpy()
                pc = pc.clone().detach().to('cpu').numpy()


                batch_correction_init = 0.001*np.random.rand(3*N)
                fun = lambda x: self.objective_numpy(detected_keypoints=kp, input_point_cloud=pc, correction=x)


                # loss_before = fun(x=batch_correction_init)
                # print("loss before optimization: ", loss_before)

                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='Nelder-Mead')         #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='Powell')              #Note: tried, best so far. Promising visually. Misses orientation a few times. Takes a few seconds.
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='CG')                  #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='BFGS')                #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='Newton-CG')           #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='L-BFGS-B')            #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='TNC')                 #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='COBYLA')              #Note: tried. the two point clouds get closer. returns a False flag for success. Fast.
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='SLSQP')               #Note: tried, does not work
                result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-constr')        #Note: tried, best so far. Promising visually. Misses orientation a few times. Faster than 'Powell'.
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='dogleg')              #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-ncg')           #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-exact')         #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-krylov')        #Note: requires jacobian. can be computed. haven't done. #ToDo


                # print("loss after optimization: ", result.fun)
                # print("opt status: ", result.status)
                # print("num of steps: ", result.nit)
                # print("corrector optimization successful: ", result.success)
                batch_correction = torch.from_numpy(result.x).to(torch.float)
                batch_correction = batch_correction.reshape(3, N)

                correction[batch, ...] = batch_correction


        return correction.clone().detach()


class kp_corrector_pace():
    # ToDo: Have to change for padding.
    # ToDo: This has solve_algo2() written, but uses solve_algo1() because PACEddn packprop is not correctly implemented.
    def __init__(self, cad_models, model_keypoints, batch_size=32, theta=10.0, kappa=50.0):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (K, 3, m)
        model_keypoints : torch.tensor of shape (K, 3, N)
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.theta = theta
        self.kappa = kappa
        self.pace = PACEbp(model_keypoints=self.model_keypoints)
        self.modelgen = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)


    def forward(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """


        #ToDo: Will have to change to solve_algo2() once pace backprop is fixed.
        correction = self.solve_algo1(detected_keypoints, input_point_cloud)
        # correction = self.solve_algo2(detected_keypoints, input_point_cloud)

        return correction

    def objective(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """

        # start = time.perf_counter()
        R, t, c = self.pace.forward(y=detected_keypoints + correction)
        # print("pace run time: ", time.perf_counter()-start)
        # mid = time.perf_counter()
        keypoint_estimate, model_estimate = self.modelgen.forward(shape=c)
        # print("model gen runtime: ", time.perf_counter()-mid)
        model_estimate = R @ model_estimate + t
        keypoint_estimate = R @ keypoint_estimate + t

        loss_pc = chamfer_loss(pc=input_point_cloud, pc_=model_estimate)
        # loss_pc = chamfer_loss_with_surface_normals(pc=input_point_cloud, pc_=model_estimate)

        loss_kp = keypoints_loss(kp=detected_keypoints+correction, kp_=keypoint_estimate)
        # loss_kp = 0.0

        return self.kappa*loss_pc + self.theta*loss_kp



    def objective_numpy(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : numpy.ndarray of shape (3, N)
        input_point_cloud   : numpy.ndarray of shape (3, m)
        correction          : numpy.ndarray of shape (3*N,)

        output:
        loss    : numpy.ndarray of shape (1,)
        """
        N = detected_keypoints.shape[-1]
        correction = correction.reshape(3, N)

        detected_keypoints = torch.from_numpy(detected_keypoints).unsqueeze(0).to(torch.float)
        input_point_cloud = torch.from_numpy(input_point_cloud).unsqueeze(0).to(torch.float)
        correction = torch.from_numpy(correction).unsqueeze(0).to(torch.float)

        loss = self.objective(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud,
                              correction=correction)

        return loss.squeeze(0).numpy()



    def _get_objective_jacobian(self, fun, correction):

        batch_size = correction.shape[0]
        dfdcorrection = torch.zeros_like(correction)

        # Do not set create_graph=True in jacobian. It will slow down computation substantially.
        dfdcorrectionX = torch.autograd.functional.jacobian(fun, correction)
        b = range(batch_size)
        dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

        return dfdcorrection

    def solve_algo2(self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """
        N = detected_keypoints.shape[-1]
        batch_size = detected_keypoints.shape[0]

        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x)

        correction = torch.zeros_like(detected_keypoints)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr

        iter = 0
        obj_ = self.objective(detected_keypoints, input_point_cloud, correction)
        flag = torch.ones_like(obj_).to(dtype=torch.bool)
        # flag_idx = flag.nonzero()
        flag = flag.unsqueeze(-1).repeat(1, 3, N)
        while iter < max_iterations:

            iter += 1
            obj = obj_

            dfdcorrection = self._get_objective_jacobian(f, correction)
            correction -= lr*dfdcorrection*flag
            # correction -= lr * dfdcorrection

            obj_ = self.objective(detected_keypoints, input_point_cloud, correction)

            if (obj-obj_).abs().max() < tol:
                break
            else:
                flag = (obj-obj_).abs() > tol
                # flag_idx = flag.nonzero()
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

        return correction

    def solve_algo1(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

        #This corrects and registers the point clout very well. Two problems:
        # 1. The shape parameter induces a lot of freedome, and sometimes the model point-cloud reduces in size to fix a part of the input point cloud
        # 2. It takes a few seconds to converge, which may be hard for a real-time operation

        N = detected_keypoints.shape[-1]
        batch_size = detected_keypoints.shape[0]
        correction = torch.zeros_like(detected_keypoints)

        with torch.enable_grad():

            for batch in range(batch_size):

                kp = detected_keypoints[batch, ...]
                pc = input_point_cloud[batch, ...]
                kp = kp.clone().detach().to('cpu').numpy()
                pc = pc.clone().detach().to('cpu').numpy()


                batch_correction_init = 0.001*np.random.rand(3*N)
                fun = lambda x: self.objective_numpy(detected_keypoints=kp, input_point_cloud=pc, correction=x)


                loss_before = fun(x=batch_correction_init)
                print("loss before optimization: ", loss_before)

                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='Nelder-Mead')         #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='Powell')              #Note: tried, best so far. Promising visually. Misses orientation a few times. Takes a few seconds.
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='CG')                  #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='BFGS')                #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='Newton-CG')           #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='L-BFGS-B')            #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='TNC')                 #Note: tried, does not work
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='COBYLA')              #Note: tried. the two point clouds get closer. returns a False flag for success. Fast.
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='SLSQP')               #Note: tried, does not work
                result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-constr')        #Note: tried, best so far. Promising visually. Misses orientation a few times. Faster than 'Powell'.
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='dogleg')              #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-ncg')           #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-exact')         #Note: requires jacobian. can be computed. haven't done. #ToDo
                # result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-krylov')        #Note: requires jacobian. can be computed. haven't done. #ToDo


                print("loss after optimization: ", result.fun)
                print("opt status: ", result.status)
                print("num of steps: ", result.nit)
                print("corrector optimization successful: ", result.success)
                batch_correction = torch.from_numpy(result.x).to(torch.float)
                batch_correction = batch_correction.reshape(3, N)

                correction[batch, ...] = batch_correction


        return correction.clone().detach()


class correctorNode(AbstractDeclarativeNode):
    def __init__(self, keypoint_corrector):
        super().__init__()

        self.keypoint_corrector = keypoint_corrector

    def objective(self, detected_keypoints, input_point_cloud, y):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3*N)
        input_point_cloud   : torch.tensor of shape (B, 3*m)
        correction          : torch.tensor of shape (B, 3*N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """
        correction = y

        batch_size = detected_keypoints.shape[0]
        N = int(detected_keypoints.shape[1] / 3.0)
        m = int(input_point_cloud.shape[1] / 3.0)

        detected_keypoints = detected_keypoints.reshape(batch_size, 3, N)
        input_point_cloud = input_point_cloud.reshape(batch_size, 3, m)
        correction = correction.reshape(batch_size, 3, N)

        return self.keypoint_corrector.objective(detected_keypoints=detected_keypoints,
                                                 input_point_cloud=input_point_cloud,
                                                 correction=correction)


    def solve(self, detected_keypoints, input_point_cloud):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3*N)
        input_point_cloud   : torch.tensor of shape (B, 3*m)

        output:
        correction          : torch.tensor of shape (B, 3*N)

        """

        batch_size = detected_keypoints.shape[0]
        N = int(detected_keypoints.shape[1]/3.0)
        m = int(input_point_cloud.shape[1] / 3.0)

        detected_keypoints = detected_keypoints.reshape(batch_size, 3, N)
        input_point_cloud = input_point_cloud.reshape(batch_size, 3, m)

        correction = self.keypoint_corrector.forward(detected_keypoints=detected_keypoints,
                                                     input_point_cloud=input_point_cloud)

        correction = correction.reshape(batch_size, 3*N)

        return correction, None






if __name__ == "__main__":

    ################################################################################################################

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    print("-"*40)
    print("Verifying keypoint corrector with SE3PointCloud dataset and keypoint_perturbation(): ")

    B = 10
    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=1000)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=B, shuffle=False)

    model_keypoints = se3_dataset._get_model_keypoints()    # (1, 3, N)
    cad_models = se3_dataset._get_cad_models()              # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)


    # define the keypoint corrector
    corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    point_set_reg = PointSetRegistration(source_points=model_keypoints)

    for i, data in enumerate(se3_dataset_loader):

        input_point_cloud, keypoints_true, rotation_true, translation_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)

        # generating perturbed keypoints
        # keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type='sporadic', var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using the keypoint corrector
        start = time.perf_counter()
        correction = corrector.forward(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud)
        end = time.perf_counter()
        print("Corrector time: ", 1000*(end-start)/B, ' ms')
        #
        print("Testing backward: ")
        grad_out = corrector.gradient(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud,
                           correction=correction)


        #
        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints+correction)
        # model_estimate = R @ cad_models + t
        # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())

        # evaluate the two metrics
        print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
        # the claim is that with the correction we can

        break

    print("-" * 40)






    ################################################################################################################

    # # device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # print('device is ', device)
    # print('-' * 20)
    #
    # class_id = "03001627"  # chair
    # model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    #
    # print("-"*40)
    # print("Verifying keypoint corrector with SE3PointCloud dataset and keypoint_perturbation(): ")
    #
    # B = 10
    # se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=1000)
    # se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=B, shuffle=False)
    #
    # model_keypoints = se3_dataset._get_model_keypoints()    # (1, 3, N)
    # cad_models = se3_dataset._get_cad_models()              # (1, 3, m)
    # model_keypoints = model_keypoints.to(device=device)
    # cad_models = cad_models.to(device=device)
    #
    #
    # # define the keypoint corrector
    # corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    # point_set_reg = PointSetRegistration(source_points=model_keypoints)
    #
    # for i, data in enumerate(se3_dataset_loader):
    #
    #     input_point_cloud, keypoints_true, rotation_true, translation_true = data
    #
    #     input_point_cloud = input_point_cloud.to(device=device)
    #     keypoints_true = keypoints_true.to(device=device)
    #     rotation_true = rotation_true.to(device=device)
    #     translation_true = translation_true.to(device=device)
    #
    #     # generating perturbed keypoints
    #     # keypoints_true = rotation_true @ model_keypoints + translation_true
    #     # detected_keypoints = keypoints_true
    #     detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type='sporadic', var=0.8, fra=1.0)
    #     detected_keypoints = detected_keypoints.to(device=device)
    #
    #     # estimate model: using point set registration on perturbed keypoints
    #     start = time.perf_counter()
    #     R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
    #     end = time.perf_counter()
    #     print("Naive registration time: ", 1000*(end-start)/B, " ms")
    #     # model_estimate = R_naive @ cad_models + t_naive
    #     # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())
    #
    #     # estimate model: using the keypoint corrector
    #     start = time.perf_counter()
    #     correction = corrector.forward(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud)
    #     end = time.perf_counter()
    #     print("Corrector time: ", 1000*(end-start)/B, ' ms')
    #     #
    #     print("Testing backward: ")
    #     corrector.gradient(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud, correction=correction)
    #     #
    #     # correction = torch.zeros_like(correction)
    #     R, t = point_set_reg.forward(target_points=detected_keypoints+correction)
    #     # model_estimate = R @ cad_models + t
    #     # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())
    #
    #     # evaluate the two metrics
    #     print("Evaluation error (wo correction): ", registration_eval(R_naive, rotation_true, t_naive, translation_true).mean())
    #     print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
    #     # the claim is that with the correction we can
    #
    #     if i >= 5:
    #         break
    #
    # print("-" * 40)


    # print("-"*40)
    # print("Verifying keypoint corrector with DepthPointCloud2 dataset and keypoint_perturbation(): ")
    #
    # B=1
    # depth_dataset = DepthPointCloud2(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=100)
    # depth_dataset_loader = torch.utils.data.DataLoader(depth_dataset, batch_size=B, shuffle=False)
    #
    # model_keypoints = depth_dataset._get_model_keypoints()    # (1, 3, N)
    # cad_models = depth_dataset._get_cad_models()              # (1, 3, m)
    # model_keypoints = model_keypoints.to(device=device)
    # cad_models = cad_models.to(device=device)
    #
    # # define the keypoint corrector
    # corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    # point_set_reg = PointSetRegistration(source_points=model_keypoints)
    #
    # for i, data in enumerate(depth_dataset_loader):
    #
    #     input_point_cloud, keypoints_true, rotation_true, translation_true = data
    #
    #     input_point_cloud = input_point_cloud.to(device=device)
    #     keypoints_true = keypoints_true.to(device=device)
    #     rotation_true = rotation_true.to(device=device)
    #     translation_true = translation_true.to(device=device)
    #
    #     # generating perturbed keypoints
    #     # detected_keypoints = keypoints_true
    #     detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type='sporadic', var=0.8, fra=1.0)
    #     detected_keypoints = detected_keypoints.to(device=device)
    #
    #     # estimate model: using point set registration on perturbed keypoints
    #     start = time.process_time()
    #     R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
    #     end = time.process_time()
    #     print("Naive registration time: ", 1000 * (end - start) / B, " ms")
    #     model_estimate = R_naive @ cad_models + t_naive
    #     display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())
    #
    #     # estimate model: using the keypoint corrector
    #     start = time.process_time()
    #     correction = corrector.forward(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud)
    #     end = time.process_time()
    #     print("Corrector time: ", 1000 * (end - start) / B, ' ms')
    #     # correction = torch.zeros_like(correction)
    #     R, t = point_set_reg.forward(target_points=detected_keypoints+correction)
    #     model_estimate = R @ cad_models + t
    #     display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())
    #
    #     # evaluate the two metrics
    #     print("Evaluation error (wo correction): ", registration_eval(R_naive, rotation_true, t_naive, translation_true))
    #     print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true))
    #     # the claim is that with the correction we can
    #
    #     if i >= 5:
    #         break
    #
    # print("-" * 40)










