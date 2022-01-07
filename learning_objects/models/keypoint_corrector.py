"""
This implements PACE with keypoint correction as a function and a torch.nn.Module.

"""

import torch
import open3d as o3d
import numpy as np
from scipy import optimize
import torch.nn as nn
import cvxpy as cp
import os
import sys
sys.path.append("../../")

from learning_objects.models.point_set_registration import point_set_registration
from learning_objects.datasets.keypointnet import SE3PointCloud

from learning_objects.utils.general import pos_tensor_to_o3d
from learning_objects.utils.general import chamfer_distance, chamfer_half_distance


#ToDo: test with depth-point clouds and half-chamfer loss
#ToDo: test with chamfer loss with the component of surface normals

#ToDo: replace registration with PACE


def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (3, n)
    pc2 : torch.tensor of shape (3, m)
    """
    pc1 = pc1.to('cpu')
    pc2 = pc2.to('cpu')

    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None


def chamfer_loss_wsurface_normals(pc, pc_):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)

    output:
    loss    :
    """


def chamfer_loss(pc, pc_):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)

    output:
    loss    :
    """

    #Note: we got more errors when using chamfer_half_distance, than when using chamfer distance

    return chamfer_half_distance(pc, pc_).mean()
    # return chamfer_distance(pc, pc_).mean()

def keypoints_loss(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(kp, kp_)

class kp_corrector():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models      : torch.tensor of shape (1, 3, m)
        model_keypoints : torch.tensor of shape (1, 3, N)
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints


    def forward(self, detected_keypoints, input_point_cloud):
        """
        input:
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        input_point_cloud   : torch.tensor of shape (B, 3, m)

        output:
        correction          : torch.tensor of shape (B, 3, N)
        """

        correction = self.solve_alog1(detected_keypoints, input_point_cloud)

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
        theta = 50.0
        kappa = 10.0

        #Note: kappa = 100.0 and theta = 1.0 showed promising results. It resuled in no orientation errors in registration.

        R, t = point_set_registration(source_points=self.model_keypoints, target_points=detected_keypoints+correction)
        model_estimate = R @ self.cad_models + t
        keypoint_estimate = R @ self.model_keypoints + t

        loss_pc = chamfer_loss(pc=input_point_cloud, pc_=model_estimate)
        loss_kp = keypoints_loss(kp=detected_keypoints+correction, kp_=keypoint_estimate)
        # loss_kp = 0.0

        return kappa*loss_pc + theta*loss_kp

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



    def solve_alog1(self, detected_keypoints, input_point_cloud, lr=0.1, num_steps=20):
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




if __name__ == "__main__":


    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model
    se3_dataset = SE3PointCloud(class_id=class_id, model_id=model_id, num_of_points=500, dataset_len=100)
    se3_dataset_loader = torch.utils.data.DataLoader(se3_dataset, batch_size=1, shuffle=False)

    model_keypoints = se3_dataset._get_model_keypoints()    # (1, 3, N)
    cad_models = se3_dataset._get_cad_models()              # (1, 3, m)

    # define the keypoint corrector
    corrector = kp_corrector(cad_models=cad_models, model_keypoints=model_keypoints)

    for i, data in enumerate(se3_dataset_loader):

        input_point_cloud, rotation_true, translation_true = data

        # generating perturbed keypoints
        keypoints_true = rotation_true @ model_keypoints + translation_true
        # detected_keypoints = keypoints_true
        detected_keypoints = keypoints_true + torch.rand(size=keypoints_true.shape)

        # estimate model: using point set registration on perturbed keypoints
        R, t = point_set_registration(source_points=model_keypoints, target_points=detected_keypoints)
        model_estimate = R @ cad_models + t
        display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate.squeeze(0))

        # estimate model: using the keypoint corrector
        correction = corrector.forward(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud)
        # correction = torch.zeros_like(correction)
        R, t = point_set_registration(source_points=model_keypoints, target_points=detected_keypoints+correction)
        model_estimate = R @ cad_models + t
        display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate.squeeze(0))

        # the claim is that with the correction we can

        if i >= 5:
            break












