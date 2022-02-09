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

from learning_objects.utils.ddn.node import AbstractDeclarativeNode, ParamDeclarativeFunction

from learning_objects.models.point_set_registration import point_set_registration, PointSetRegistration
from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, SE3nIsotorpicShapePointCloud, DepthPC

from learning_objects.utils.general import pos_tensor_to_o3d, display_two_pcs
from learning_objects.utils.general import chamfer_distance, chamfer_half_distance, rotation_error, \
    translation_error, shape_error

# from learning_objects.models.pace_ddn import PACEbp
# from learning_objects.models.pace_altern_ddn import PACEbp
from learning_objects.models.pace import PACEmodule
from learning_objects.models.modelgen import ModelFromShape


def chamfer_loss(pc, pc_, pc_padding=None, max_loss=False):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
    max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    output:
    loss    : (B, 1)
        returns max_loss if max_loss is true
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

    if max_loss:
        loss = sq_dist.max(dim=1)[0]
    else:
        loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)


def keypoints_loss(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='none')

    return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)


def rotation_loss(R, R_):

    device_ = R.device

    err_mat = R @ R_.transpose(-1, -2) - torch.eye(3, device=device_)
    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(err_mat, torch.zeros_like(err_mat))


def translation_loss(t, t_):
    """
    t   : torch.tensor of shape (B, 3, N)
    t_  : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(t, t_)


def shape_loss(c, c_):
    """
    c   : torch.tensor of shape (B, K, 1)
    c_  : torch.tensor of shape (B, K, 1)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(c, c_)


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
    def __init__(self, cad_models, model_keypoints, theta=50.0, kappa=10.0, algo='torch'):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, m)
        model_keypoints : torch.tensor of shape (1, 3, N)
        algo            : 'scipy' or 'torch'
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.theta = theta
        self.kappa = kappa
        self.algo = algo

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

    def objective(self, detected_keypoints, input_point_cloud, correction):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        correction          : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """

        R, t = self.point_set_registration_fn.forward(detected_keypoints+correction)
        model_estimate = R @ self.cad_models + t
        keypoint_estimate = R @ self.model_keypoints + t

        loss_pc = chamfer_loss(pc=input_point_cloud, pc_=model_estimate, max_loss=True)
        # loss_pc = max_chamfer_loss(pc=input_point_cloud, pc_=model_estimate)
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

    def solve(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        if self.algo == 'scipy':
            correction = self.solve_algo1(detected_keypoints, input_point_cloud)
        elif self.algo == 'torch':
            correction = self.solve_algo2(detected_keypoints, input_point_cloud)
        else:
            raise NotImplementedError

        return correction, None

    def solve_algo1(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """

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

    def solve_algo2(self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """
        def _get_objective_jacobian(fun, x):

            torch.set_grad_enabled(True)
            batch_size = x.shape[0]
            dfdcorrection = torch.zeros_like(x)

            # Do not set create_graph=True in jacobian. It will slow down computation substantially.
            dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
            b = range(batch_size)
            dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

            return dfdcorrection

        N = detected_keypoints.shape[-1]
        correction = torch.zeros_like(detected_keypoints)

        f = lambda x: self.objective(detected_keypoints, input_point_cloud, x)

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

            dfdcorrection = _get_objective_jacobian(f, correction)
            correction -= lr*dfdcorrection*flag

            obj_ = f(correction)

            if (obj-obj_).abs().max() < tol:
                break
            else:
                flag = (obj-obj_).abs() > tol
                # flag_idx = flag.nonzero()
                flag = flag.unsqueeze(-1).repeat(1, 3, N)

        return correction

    def solve_algo3(self, *xs, correction):
        #ToDo: Not tested. See if this works. This is from the ddn library pnp_node.py
        with torch.enable_grad():
            opt = torch.optim.LBFGS([correction],
                                    lr=1.0,
                                    max_iter=1000,
                                    max_eval=None,
                                    tolerance_grad=1e-40,
                                    tolerance_change=1e-40,
                                    history_size=100,
                                    line_search_fn="strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                f = self.objective(*xs, correction=correction).sum() # sum over batch elements
                f.backward()
                return f
            opt.step(reevaluate)
        return correction

    def gradient(self, detected_keypoints, input_point_cloud, y=None, v=None, ctx=None):

        if v==None:
            v = torch.ones_like(detected_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_keypoints.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)



class kp_corrector_pace():
    def __init__(self, cad_models, model_keypoints, theta=10.0, kappa=50.0, algo='torch'):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (K, 3, m)
        model_keypoints : torch.tensor of shape (K, 3, N)
        algo            : 'scipy' or 'torch'
        
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints
        self.theta = theta
        self.kappa = kappa
        self.pace = PACEmodule(model_keypoints=self.model_keypoints)
        self.modelgen = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)
        self.algo = algo

        self.device_ = model_keypoints.device

    def objective(self, detected_keypoints, input_point_cloud, y):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        y                   : torch.tensor of shape (B, 3, N)

        outputs:
        loss    : torch.tensor of shape (B, 1)

        """
        correction = y

        # start = time.perf_counter()
        R, t, c = self.pace(detected_keypoints + correction)
        # print("pace run time: ", time.perf_counter()-start)
        # mid = time.perf_counter()
        keypoint_estimate, model_estimate = self.modelgen.forward(shape=c)
        # print("model gen runtime: ", time.perf_counter()-mid)
        model_estimate = R @ model_estimate + t
        keypoint_estimate = R @ keypoint_estimate + t

        loss_pc = chamfer_loss(pc=input_point_cloud, pc_=model_estimate)
        # loss_pc = chamfer_loss_with_surface_normals(pc=input_point_cloud, pc_=model_estimate)

        loss_kp = keypoints_loss(kp=detected_keypoints + correction, kp_=keypoint_estimate)
        # loss_kp = 0.0

        return self.kappa * loss_pc + self.theta * loss_kp

    def objective_numpy(self, detected_keypoints, input_point_cloud, y):
        """
        inputs:
        detected_keypoints  : numpy.ndarray of shape (3, N)
        input_point_cloud   : numpy.ndarray of shape (3, m)
        y                   : numpy.ndarray of shape (3*N,)

        output:
        loss    : numpy.ndarray of shape (1,)
        """
        correction = y
        N = detected_keypoints.shape[-1]
        correction = correction.reshape(3, N)

        detected_keypoints = torch.from_numpy(detected_keypoints).unsqueeze(0).to(torch.float)
        input_point_cloud = torch.from_numpy(input_point_cloud).unsqueeze(0).to(torch.float)
        correction = torch.from_numpy(correction).unsqueeze(0).to(torch.float)

        loss = self.objective(detected_keypoints=detected_keypoints.to(device=self.device_),
                              input_point_cloud=input_point_cloud.to(device=self.device_),
                              y=correction.to(device=self.device_))

        return loss.squeeze(0).to('cpu').numpy()

    def solve(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        if self.algo == 'scipy':
            correction = self.solve_algo1(detected_keypoints, input_point_cloud)
        elif self.algo == 'torch':
            correction = self.solve_algo2(detected_keypoints, input_point_cloud)
        else:
            raise NotImplementedError

        return correction, None

    def solve_algo2(self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12):
        """
        inputs:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        outputs:
        correction          : torch.tensor of shape (B, 3, N)
        """
        def _get_objective_jacobian(fun, x):

            torch.set_grad_enabled(True)
            batch_size = x.shape[0]
            dfdcorrection = torch.zeros_like(x)

            # Do not set create_graph=True in jacobian. It will slow down computation substantially.
            dfdcorrectionX = torch.autograd.functional.jacobian(fun, x)
            b = range(batch_size)
            dfdcorrection[b, ...] = dfdcorrectionX[b, 0, b, ...]

            return dfdcorrection

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

            # print(iter)
            # Note: each iteration computes does a forward pass on the objective function to compute the jacobian.
            dfdcorrection = _get_objective_jacobian(f, correction)
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

    def solve_algo1(self, detected_keypoints, input_point_cloud):
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

                print("batch: ", batch)

                kp = detected_keypoints[batch, ...]
                pc = input_point_cloud[batch, ...]
                kp = kp.to('cpu').numpy()
                pc = pc.to('cpu').numpy()


                batch_correction_init = 0.001*np.random.rand(3*N)
                fun = lambda x: self.objective_numpy(detected_keypoints=kp, input_point_cloud=pc, y=x)


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


        return correction

    def gradient(self, detected_keypoints, input_point_cloud, y=None, v=None, ctx=None):

        if v==None:
            v = torch.ones_like(detected_keypoints)

        # v = gradient of ML loss with respect to correction.
        # Therefore, the gradient to backpropagate is -v for detected_keypoints.
        # We don't backpropagate gradient with respect to the input_point_cloud
        return (-v, None)


if __name__ == "__main__":

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    ####################################################################################################################
    print("-"*40)
    print("Verifying kp_corrector_reg() with SE3PointCloud(dataset) and keypoint_perturbation(): ")

    B = 10
    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=1000)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=B, shuffle=False)

    model_keypoints = se3_dataset._get_model_keypoints()    # (1, 3, N)
    cad_models = se3_dataset._get_cad_models()              # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)


    # define the keypoint corrector
    # option 1: use the backprop through all the iterations of optimization
    # corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)  #Note: DO NOT USE THIS.
    # option 2: use autograd computed gradient for backprop.
    corrector_node = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)


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

        # estimate model: using point set registration on perturbed keypoints
        start = time.perf_counter()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end = time.perf_counter()
        print("Naive registration time: ", 1000*(end-start)/B, " ms")
        # model_estimate = R_naive @ cad_models + t_naive
        # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())

        # # estimate model: using the keypoint corrector
        detected_keypoints.requires_grad = True
        start = time.perf_counter()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end = time.perf_counter()
        print("Corrector time: ", 1000*(end-start)/B, ' ms')
        #

        loss = torch.norm(correction, p=2)**2
        loss = loss.sum()
        print("Testing backward: ")
        loss.backward()
        print("Shape of detected_keypoints.grad: ", detected_keypoints.grad.shape)
        print("Sum of abs() of all elements in the detected_keypoints.grad: ", detected_keypoints.grad.abs().sum())
        #

        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints+correction)
        # model_estimate = R @ cad_models + t
        # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())

        # evaluate the two metrics
        print("Evaluation error (wo correction): ", registration_eval(R_naive, rotation_true, t_naive, translation_true).mean())
        print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)

    ####################################################################################################################
    print("-"*40)
    print("Verifying kp_corrector_reg() with DepthPC(dataset) and keypoint_perturbation(): ")

    B=10
    depth_dataset = DepthPC(class_id=class_id, model_id=model_id, n=500, num_of_points_to_sample=1000, dataset_len=100)
    depth_dataset_loader = torch.utils.data.DataLoader(depth_dataset, batch_size=B, shuffle=False)

    model_keypoints = depth_dataset._get_model_keypoints()    # (1, 3, N)
    cad_models = depth_dataset._get_cad_models()              # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)

    # define the keypoint corrector
    # option 1: use the backprop through all the iterations of optimization
    # corrector = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)  #Note: DO NOT USE THIS.
    # option 2: use autograd computed gradient for backprop.
    corrector_node = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)


    point_set_reg = PointSetRegistration(source_points=model_keypoints)

    for i, data in enumerate(depth_dataset_loader):

        input_point_cloud, keypoints_true, rotation_true, translation_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)

        # generating perturbed keypoints
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type='sporadic', var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.process_time()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end = time.process_time()
        print("Naive registration time: ", 1000 * (end - start) / B, " ms")
        # model_estimate = R_naive @ cad_models + t_naive
        # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())

        # estimate model: using the keypoint corrector
        detected_keypoints.requires_grad = True
        start = time.process_time()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end = time.process_time()
        print("Corrector time: ", 1000 * (end - start) / B, ' ms')

        #
        loss = torch.norm(correction, p=2) ** 2
        loss = loss.sum()
        print("Testing backward: ")
        loss.backward()
        print("Shape of detected_keypoints.grad: ", detected_keypoints.grad.shape)
        print("Sum of abs() of all elements in the detected_keypoints.grad: ", detected_keypoints.grad.abs().sum())
        #

        # correction = torch.zeros_like(correction)
        R, t = point_set_reg.forward(target_points=detected_keypoints+correction)
        # model_estimate = R @ cad_models + t
        # display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_estimate[0, ...].detach())

        # evaluate the two metrics
        print("Evaluation error (wo correction): ", registration_eval(R_naive, rotation_true, t_naive, translation_true).mean())
        print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)

    ###################################################################################################################
    print("-"*40)
    print("Verifying kp_corrector_pace() with SE3nIsotropicShapePointCloud(dataset) and keypoint_perturbation(): ")

    B = 1
    se3_dataset = SE3nIsotorpicShapePointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=1000)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=B, shuffle=False)

    model_keypoints = se3_dataset._get_model_keypoints()    # (1, 3, N)
    cad_models = se3_dataset._get_cad_models()              # (1, 3, m)
    model_keypoints = model_keypoints.to(device=device)
    cad_models = cad_models.to(device=device)


    # define the keypoint corrector
    # option 1: use the backprop through all the iterations of optimization
    # corrector = kp_corrector_pace(cad_models=cad_models, model_keypoints=model_keypoints) #Note: DO NOT USE THIS.
    # option 2: use autograd computed gradient for backprop.
    corrector_node = kp_corrector_pace(cad_models=cad_models, model_keypoints=model_keypoints)
    corrector = ParamDeclarativeFunction(problem=corrector_node)

    pace = PACEmodule(model_keypoints=model_keypoints)
    modelgen = ModelFromShape(cad_models=cad_models, model_keypoints=model_keypoints)



    for i, data in enumerate(se3_dataset_loader):

        input_point_cloud, keypoints_true, rotation_true, translation_true, shape_true = data

        input_point_cloud = input_point_cloud.to(device=device)
        keypoints_true = keypoints_true.to(device=device)
        rotation_true = rotation_true.to(device=device)
        translation_true = translation_true.to(device=device)
        shape_true = shape_true.to(device=device)

        # generating perturbed keypoints
        # keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type='sporadic', var=0.8, fra=0.2)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.perf_counter()
        R_naive, t_naive, c_naive = pace(detected_keypoints)
        end = time.perf_counter()
        print("Naive pace time: ", 1000*(end-start)/B, " ms")
        keypoint_naive, model_naive = modelgen.forward(shape=c_naive)
        model_naive = R_naive @ model_naive + t_naive
        keypoint_naive = R_naive @ keypoint_naive + t_naive
        display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model_naive[0, ...].detach())


        # # estimate model: using the keypoint corrector
        detected_keypoints.requires_grad = True
        start = time.perf_counter()
        correction = corrector.forward(detected_keypoints, input_point_cloud)
        end = time.perf_counter()
        print("Corrector with pace time: ", 1000*(end-start)/B, ' ms')
        #

        loss = torch.norm(correction, p=2)**2
        loss = loss.sum()
        print("Testing backward: ")
        loss.backward()
        print("Shape of detected_keypoints.grad: ", detected_keypoints.grad.shape)
        print("Sum of abs() of all elements in the detected_keypoints.grad: ", detected_keypoints.grad.abs().sum())
        #

        # correction = torch.zeros_like(correction)
        R, t, c = pace.forward(detected_keypoints+correction)
        end = time.perf_counter()
        print("Naive registration time: ", 1000 * (end - start) / B, " ms")
        keypoints, model = modelgen.forward(shape=c)
        model = R_naive @ model + t_naive
        keypoints = R_naive @ keypoints + t_naive
        # model_estimate = R @ cad_models + t
        display_two_pcs(pc1=input_point_cloud[0, ...].detach(), pc2=model[0, ...].detach())

        # evaluate the two metrics
        print("Evaluation error (wo correction): ", pace_eval(R_naive, rotation_true, t_naive, translation_true, c_naive, shape_true).mean())
        print("Evaluation error (w correction): ", pace_eval(R, rotation_true, t, translation_true, c, shape_true).mean())
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)

