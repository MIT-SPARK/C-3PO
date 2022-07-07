"""
This implements the keypoint correction with registration

"""
import numpy as np
import open3d as o3d
import os
import sys
import time
import torch
import torch.nn as nn
from pytorch3d import ops
from scipy import optimize

sys.path.append("../../")

from c3po.utils.ddn.node import ParamDeclarativeFunction

from c3po.models.point_set_registration import PointSetRegistration
from c3po.datasets.shapenet import SE3PointCloud, SE3nIsotropicShapePointCloud, DepthPC

from c3po.utils.visualization_utils import display_two_pcs, update_pos_tensor_to_keypoint_markers
from c3po.utils.loss_functions import chamfer_loss, keypoints_loss
from c3po.utils.evaluation_metrics import shape_error, translation_error, rotation_euler_error

from c3po.models.modelgen import ModelFromShape


def registration_eval(R, R_, t, t_):
    """
    inputs:
    R, R_   : torch.tensor of shape (B, 3, 3)
    t, t_   : torch.tensor of shape (B, 3, 1)

    output:
    loss    : torch.tensor of shape (B, 1)
    """

    return rotation_euler_error(R, R_) + translation_error(t, t_)


def keypoint_perturbation(keypoints_true, var=0.8, fra=0.2):
    """
    inputs:
    keypoints_true  :  torch.tensor of shape (B, 3, N)
    var             :  float
    fra             :  float    : used if type == 'sporadic'

    output:
    detected_keypoints  : torch.tensor of shape (B, 3, N)
    """
    device_ = keypoints_true.device

    mask = (torch.rand(size=keypoints_true.shape).to(device=device_) < fra).int().float()
    detected_keypoints = keypoints_true + var * (torch.rand(size=keypoints_true.shape).to(device=device_)-0.5) * mask

    return detected_keypoints


class kp_corrector_reg:
    def __init__(self, cad_models, model_keypoints, theta=50.0, kappa=10.0, algo='torch', animation_update=False, vis=None):
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
        self.device_ = model_keypoints.device
        self.animation_update = animation_update
        self.vis = vis

        self.markers = None

        self.point_set_registration_fn = PointSetRegistration(source_points=self.model_keypoints)

    def set_markers(self, markers):
        self.markers = markers

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

        loss_kp = keypoints_loss(kp=detected_keypoints+correction, kp_=keypoint_estimate)

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

        loss = self.objective(detected_keypoints=detected_keypoints.to(device=self.device_),
                              input_point_cloud=input_point_cloud.to(device=self.device_),
                              correction=correction.to(device=self.device_))

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
            correction = self.scipy_trust_region(detected_keypoints, input_point_cloud)
        elif self.algo == 'torch':
            correction = self.batch_gradient_descent(detected_keypoints, input_point_cloud)
        else:
            raise NotImplementedError
        return correction, None

    def scipy_trust_region(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
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
        device_ = input_point_cloud.device

        with torch.enable_grad():

            for batch in range(batch_size):

                kp = detected_keypoints[batch, ...]
                pc = input_point_cloud[batch, ...]
                kp = kp.clone().detach().to('cpu').numpy()
                pc = pc.clone().detach().to('cpu').numpy()


                batch_correction_init = 0.001*np.random.rand(3*N)
                fun = lambda x: self.objective_numpy(detected_keypoints=kp, input_point_cloud=pc, correction=x)


                # Note: tried other methods and trust-constr works the best
                result = optimize.minimize(fun=fun, x0=batch_correction_init, method='trust-constr')        #Note: tried, best so far. Promising visually. Misses orientation a few times. Faster than 'Powell'.

                batch_correction = torch.from_numpy(result.x).to(torch.float)
                batch_correction = batch_correction.reshape(3, N)

                correction[batch, ...] = batch_correction

        return correction.to(device=device_)

    def batch_gradient_descent(self, detected_keypoints, input_point_cloud, lr=0.1, max_iterations=1000, tol=1e-12):
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
            if self.vis is not None and self.animation_update and self.markers is not None:
                self.markers = update_pos_tensor_to_keypoint_markers(self.vis, detected_keypoints + correction, self.markers)
                print("ATTEMPTED TO UPDATE VIS")
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

    print("model_keypoints shape", model_keypoints.shape)
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
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.perf_counter()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end = time.perf_counter()
        print("Naive registration time: ", 1000*(end-start)/B, " ms")
        # model_estimate = R_naive @ cad_models + t_naive
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

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
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

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
        detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, var=0.8, fra=1.0)
        detected_keypoints = detected_keypoints.to(device=device)

        # estimate model: using point set registration on perturbed keypoints
        start = time.process_time()
        R_naive, t_naive = point_set_reg.forward(target_points=detected_keypoints)
        end = time.process_time()
        print("Naive registration time: ", 1000 * (end - start) / B, " ms")
        # model_estimate = R_naive @ cad_models + t_naive
        # display_two_pcs(pc1=input_point_cloud, pc2=model_estimate)

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

        # evaluate the two metrics
        print("Evaluation error (wo correction): ", registration_eval(R_naive, rotation_true, t_naive, translation_true).mean())
        print("Evaluation error (w correction): ", registration_eval(R, rotation_true, t, translation_true).mean())
        # the claim is that with the correction we can

        if i >= 3:
            break

    print("-" * 40)


