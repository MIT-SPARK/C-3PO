"""
This implements PACE with keypoint correction as a function and a torch.nn.Module.

"""
import torch
import torch.nn as nn
import cvxpy as cp

import os
import sys
sys.path.append("../../")

from learning_objects.utils.ddn.node import AbstractDeclarativeNode, EqConstDeclarativeNode, DeclarativeLayer, ParamDeclarativeFunction
from learning_objects.utils.general import chamfer_half_distance, soft_chamfer_half_distance
from learning_objects.models.pace import PACEmodule, PACE
from learning_objects.models.pace_ddn import PACEbp
from learning_objects.models.modelgen import ModelFromShapeModule, ModelFromShape
from learning_objects.utils.general import generate_random_keypoints
from learning_objects.utils.general import rotation_error, translation_error, shape_error, keypoint_error



 
class PACEwKeypointCorrection(AbstractDeclarativeNode):
    """
    This implements PACE + Keypoint Correction.

    Note:
        This uses PACEbp() implemented in learning_objects.models.pace_ddn
        This uses ModelFromShape() implemented in learning_objects.models.modelgen
    """
    def __init__(self, model_keypoints, cad_models, weights=None, lambda_constant=None):
        super().__init__()
        """ 
        Inputs:
        model_keypoints : torch.tensor of shape (K, 3, N) 
        cad_models      : torch.tensor of shape (K, 3, n)
        weights         : torch.tensor of shape (N, 1) or None
        lambda_constant : torch.tensor of shape (1, 1) or None

        where 
        N = number of semantic keypoints
        K = number of cad models
        n = number of points in each cad model 

        Assumption:
        We are assuming that each CAD model is a point cloud of n points: (3, n)
        We are assuming that given K point clouds (K, 3, n) and a shape parameter c (K, 1) 
            there is a differentiable function "ModelFromShape(nn.Module)" which 
            can be used to generate intermediate shape from the cad models and the shape 
            parameter

        Note:
        The module implements keypoint correction optimization.
        """
        self.N = model_keypoints.shape[-1]
        self.K = model_keypoints.shape[0]
        self.ydim = 3 * 3 + 3 + self.K + 3 * self.N
        self.model_keypoints = model_keypoints  # (K, 3, N)
        self.cad_models = cad_models  # (K, 3, n)
        self.weights = weights  # (N, 1)
        self.lambda_constant = lambda_constant  # (1, 1)
        self.theta = torch.tensor([0.0])  # weight for keypoint component of the cost function

        # PACE and Model Generation
        self.pace_model = PACEbp(weights=self.weights, model_keypoints=self.model_keypoints,
                                     lambda_constant=self.lambda_constant)
        self.model_from_shape = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)

        # PACE and Model Generation for Optimization
        self.pace_model_opt = PACEbp(weights=self.weights, model_keypoints=self.model_keypoints,
                                     lambda_constant=self.lambda_constant)
        self.model_from_shape_opt = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)

    def solve(self, input_point_cloud, detected_keypoints):
        # *xs = input_point_cloud, detected_keypoints
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        detected_keypoints  : torch.tensor of shape (B, 3, N)

        where
        m = number of points in the input point cloud
        N = number of keypoints
        B = batch size
        K = number of cad models in the shape category

        output:
        rotation            : torch.tensor of shape (B, 3, 3)
        translation         : torch.tensor of shape (B, 3, 1)
        shape               : torch.tensor of shape (B, K, 1)
        correction          : torch.tensor of shape (B, 3, N)
        """

        # Optimization algorithm is implemented as a simple gradient descent
        rotation, translation, shape, correction = self.solve_algo_basic(input_point_cloud=input_point_cloud,
                                                                   detected_keypoints=detected_keypoints)

        # Optimization algorithm is implemented with torch.optim
        # rotation, translation, shape, correction = self.solve_algo_adv(input_point_cloud=input_point_cloud,
        #                                                                detected_keypoints=detected_keypoints)

        return self._to_y(rotation=rotation, translation=translation, shape=shape, correction=correction), None

    def _equality_constraints(self, input_point_cloud, detected_keypoints, y):
        # *xs = input_point_cloud, detected_keypoints
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        y                   : torch.tensor of shape (B, self.ydim)

        eq_constraints: torch.tensor of shape (B, 3)
        """
        #ToDo: We probably don't need an eqality constraint in implementing this.
        # We can pose this as an unconstrained problem, that only outputs correction term.
        # The R, t, c and model generation can be done after obtaining the keypoint correction.

        rotation, translation, shape, correction = self._from_y(y=y)

        rot, trans, sh = self.pace_model.forward(y=model_keypoints+correction)

        err_rotation = rotation_error(rot, rotation)                # (B, 1)
        err_translation = translation_error(trans, translation)     # (B, 1)
        err_shape = shape_error(sh, shape)                          # (B, 1)

        return torch.cat((err_rotation, err_translation, err_shape), dim=1)

    def objective(self, input_point_cloud, detected_keypoints, y):
        # *xs = input_point_cloud, detected_keypoints
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        detected_keypoints  : torch.tensor of shape (B, 3, N)
        y                   : torch.tensor of shape (B, self.ydim)

        eq_constraints: torch.tensor of shape (B, 1)
        """

        # unpacking y
        rotation, translation, shape, correction = self._from_y(y=y)

        # generating model keypoints and point cloud
        model_keypoints, model_point_cloud = self.model_from_shape.forward(shape=shape)
        model_keypoints = rotation @ model_keypoints + translation
        model_point_cloud = rotation @ model_keypoints + translation

        # computing loss
        val = self._pc_loss(input_point_cloud, model_point_cloud) + self.theta * keypoint_error(model_keypoints,
                                                                                             detected_keypoints + correction)

        return val

    def solve_algo_basic(self, input_point_cloud, detected_keypoints, lr=0.1, num_steps=20):
        """
        input:
        correction_init : correction initialization : torch.tensor of shape (B, 3, self.N)

        output:
        rotation    :   torch.tensor of shape (B, 3, 3)
        translation :   torch.tensor of shape (B, 3, 1)
        shape       :   torch.tensor of shape (B, self.K, 1)
        correction  :   torch.tensor of shape (B, 3, self.N)
        """
        batch_size = detected_keypoints.shape[0]

        rotation = torch.zeros(batch_size, 3, 3)
        translation = torch.zeros(batch_size, 3, 1)
        shape = torch.zeros(batch_size, self.K, 1)
        correction = torch.zeros_like(detected_keypoints)

        with torch.enable_grad():

            # history = []
            for batch in range(batch_size):

                batch_correction = torch.rand(3, self.N)
                batch_correction.requires_grad = True

                #ToDo: This is slow. Substitute with a faster implementation.
                for iter in range(num_steps):

                    kp = detected_keypoints[batch, :, :].unsqueeze(0) + batch_correction.unsqueeze(0)

                    R, t, c = self.pace_model_opt.forward(y=kp)
                    model_keypoints, model = self.model_from_shape_opt.forward(shape=c)
                    model = R @ model + t
                    model_keypoints = R @ model_keypoints + t

                    loss = self._pc_loss(input_point_cloud=input_point_cloud[batch, ...].unsqueeze(0),
                                         model_point_cloud=model) + self.theta * keypoint_error(model_keypoints, kp)
                    loss.squeeze(0).squeeze(0)
                    # print(loss.shape)
                    loss.backward()

                    print("batch: ", batch, ": iter: ", iter, ": loss: ", loss)

                    batch_correction_new = batch_correction - lr * batch_correction.grad

                    batch_correction = batch_correction_new.detach().requires_grad_(True)


                rotation[batch, :, :] = R
                translation[batch, :, :] = t
                shape[batch, :, :] = c
                correction[batch, :, :] = batch_correction.detach()

        return rotation, translation, shape, correction

    def _to_y(self, rotation, translation, shape, correction):

        batch_size = rotation.shape[0]

        y = torch.zeros(batch_size, self.ydim)

        y[:, 0:9] = torch.reshape(rotation, (batch_size, 9))
        y[:, 9:12] = translation[:, :, 0]
        y[:, 12:12 + self.K] = shape[:, :, 0]
        y[:, 12 + self.K:] = torch.reshape(correction, (batch_size, 3 * self.N))

        return y

    def _from_y(self, y):

        batch_size = y.shape[0]

        rotation = torch.reshape(y[:, 0:9], (batch_size, 3, 3))
        translation = torch.reshape(y[:, 9:12], (batch_size, 3, 1))
        shape = torch.reshape(y[:, 12:12 + self.K], (batch_size, self.K, 1))
        correction = torch.reshape(y[:, 12 + self.K:], (batch_size, 3, self.N))

        return rotation, translation, shape, correction

    def _pc_loss(self, input_point_cloud, model_point_cloud):
        """
        inputs:
        input_point_cloud   : torch.tensor of shape (B, 3, m)
        model_point_cloud   : torch.tensor of shape (B, 3, n)

        where
        m = number of points in the input point cloud
        n = number of points in the model point cloud
        B = batch size

        output:
        loss = half chamfer : torch.tensor of shape (B, 1)
        """

        return chamfer_half_distance(X=input_point_cloud, Y=model_point_cloud)

    def _correction_opt_loss(self, correction, input_point_cloud, detected_keypoints):
        """
        input:
        correction: torch.tensor of shape (B, 3, self.N)

        output:
        loss: torch.tensor of shape (B, 1)
        """

        # pace with corrected keypoints
        kp = detected_keypoints + correction
        R, t, c = self.pace_model_opt.forward(y=kp)

        # model generation
        model_keypoints, model = self.model_from_shape_opt.forward(shape=c)
        model = R @ model + t
        model_keypoints = R @ model_keypoints + t

        # computing loss
        loss = self._pc_loss(input_point_cloud=input_point_cloud,
                             model_point_cloud=model) + self.theta * keypoint_error(model_keypoints,
                                                                                    detected_keypoints + correction)

        return loss


    def solve_algo_adv(self, input_point_cloud, detected_keypoints, lr=0.1, num_steps=40):
        """
        input:
        input_point_cloud : torch.tensor of shape (B, 3, n)
        detected_keypoints: torch.tensor of shape (B, 3, self.N)

        output:
        final_correction  :   torch.tensor of shape (B, 3, self.N)
        """

        batch_size = detected_keypoints.shape[0]
        final_correction = torch.zeros_like(detected_keypoints)


        with torch.enable_grad():

            for batch in range(batch_size):

                correction = torch.rand(1, 3, self.N)
                correction.requires_grad = True

                optimizer = torch.optim.Adam([correction], lr=0.5)
                # optimizer = torch.optim.SGD([correction], lr=0.1)
                # optimizer = torch.optim.SGD([correction], lr=0.1, momentum=0.9)
                # optimizer = torch.optim.SGD([correction], lr=0.01, momentum=0.9)

                for iter in range(num_steps):
                    loss = self._correction_opt_loss(correction=correction,
                                                     input_point_cloud=input_point_cloud[batch, ...].unsqueeze(0),
                                                     detected_keypoints=detected_keypoints[batch, ...].unsqueeze(
                                                         0))
                    optimizer.zero_grad()
                    loss.backward()

                    print("batch: ", batch, ": iter: ", iter, ": loss: ", loss)

                    optimizer.step()

                final_correction[batch, ...] = correction[0, ...]

        # final_correction
        kp = detected_keypoints + final_correction
        R, t, c = self.pace_model_opt.forward(y=kp)

        return R, t, c, final_correction



# This is for just trial. Delete before code release.
class OptimizeTry():
    def __init__(self, A):
        super().__init__()
        """
        A: torch.tensor of shape (3, 3)
        """
        self.A = A.unsqueeze(0)     # (1, 3, 3)


    def cost(self, x, B):
        """
        input:
        x: torch.tensor of shape (B, 3, 1)

        output:
        cost: torch.tensor of shape (B, 1)
        """
        batch_size = x.shape[0]

        z = x @ torch.transpose(x, -1, -2) - B
        z = z.view(batch_size, -1)
        return torch.norm(z, p=2, dim=-1).unsqueeze(-1)

    def optimize(self, x_init=None, lr=0.1, num_steps=40):
        """
        input:
        x_init: torch.tensor of shape (B, 3, 1)

        output:
        x_opt: torch.tensor of shape (B, 3, 1)
        """
        if x_init == None:
            x_init = torch.rand(1, 3, 1)

        x = x_init
        x.requires_grad = True

        optimizer = torch.optim.Adam([x], lr=0.1)
        for iter in range(num_steps):
            loss = self.cost(x=x, B=self.A)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return x, self.cost(x=x, B=self.A)


def from_y(self, y, K, N):
    """
    This is a function in PACEwKeypointCorrection

    Note:
        is useful to reconstruct rotation, translation, shape, and correction from the output y of the module
    """
    batch_size = y.shape[0]

    rotation = torch.reshape(y[:, 0:9], (batch_size, 3, 3))
    translation = torch.reshape(y[:, 9:12], (batch_size, 3, 1))
    shape = torch.reshape(y[:, 12:12 + K], (batch_size, K, 1))
    correction = torch.reshape(y[:, 12 + K:], (batch_size, 3, N))

    return rotation, translation, shape, correction



if __name__ == "__main__":
    print("test")

    # A = torch.rand(3, 3)
    # my_class = OptimizeTry(A=A)
    #
    # batch_size = 10
    # x = torch.rand(batch_size, 3, 1)
    # val = my_class.cost(x=x, B=A)
    # print(val.shape)
    # print(val)
    #
    # x_opt, val_opt = my_class.optimize()
    # print(x_opt)
    # print(val_opt)
    #
    #
    # print('-'*20)




    # device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    #
    B = 2   # batch size
    N = 8   # number of keypoints
    K = 5   # number of cad models in the shape category
    n = 20  # number of points in the model_point_cloud
    m = 15  # number of points in the input_point_cloud

    cad_models = torch.rand(K, 3, n).to(device=device)
    model_keypoints = cad_models[:, :, 0:N]
    weights = torch.rand(N, 1).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)

    # initializing model
    pace_and_correction_node = PACEwKeypointCorrection(model_keypoints=model_keypoints, cad_models=cad_models,
                                                        weights=weights, lambda_constant=lambda_constant)

    detected_keypoints, rotation, translation, shape = generate_random_keypoints(batch_size=B,
                                                                                 model_keypoints=model_keypoints.cpu())
    correction = torch.rand(B, 3, N)
    y = pace_and_correction_node._to_y(rotation, translation, shape, correction)
    rot, trans, sh, cor = pace_and_correction_node._from_y(y)

    err_rotation = rotation_error(rot, rotation)
    err_translation = translation_error(trans, translation)
    err_shape = shape_error(sh, shape)
    err_correction = keypoint_error(cor, correction)

    print("Rotation error: ", err_rotation.mean())
    print("Translation error: ", err_translation.mean())
    print("Shape error: ", err_shape.mean())
    print("Correction error: ", err_correction.mean())

    pace_and_correction = ParamDeclarativeFunction(pace_and_correction_node)

    # generating data
    detected_keypoints, rotation, translation, shape = generate_random_keypoints(batch_size=B,
                                                                                 model_keypoints=model_keypoints.cpu())
    model_gen_for_data = ModelFromShape(cad_models=cad_models.cpu(), model_keypoints=model_keypoints.cpu())
    _, input_point_cloud = model_gen_for_data.forward(shape=shape)
    input_point_cloud = rotation @ input_point_cloud + translation

    # transferring generated data to device
    input_point_cloud = input_point_cloud.to(device=device)
    detected_keypoints = detected_keypoints.to(device=device)
    rotation = rotation.to(device=device)
    translation = translation.to(device=device)
    shape = shape.to(device=device)


    # applying model
    y = pace_and_correction.forward(input_point_cloud, detected_keypoints)
    est_rotation, est_translation, est_shape, correction = pace_and_correction_node._from_y(y)

    corrected_keypoints = detected_keypoints + correction

    _, est_model = model_gen_for_data.forward(shape=est_shape)
    est_model = est_rotation @ est_model + est_translation


    # analyzing shape of the output
    print("shape of rotation: ", est_rotation.shape)
    print("shape of translation: ", est_translation.shape)
    print("shape of shape parameter: ", est_shape.shape)
    print("shape of corrected_keypoints: ", corrected_keypoints.shape)
    print("shape of estimated model: ", est_model.shape)

    # computing loss
    err_rotation = rotation_error(est_rotation, rotation)
    err_translation = translation_error(est_translation, translation)
    err_shape = shape_error(est_shape, shape)
    err_point_cloud = chamfer_half_distance(X=input_point_cloud, Y=est_model)

    print("Rotation error: ", err_rotation.mean())
    print("Translation error: ", err_translation.mean())
    print("Shape error: ", err_shape.mean())
    print("Point cloud error: ", err_point_cloud.mean())

    loss = err_rotation.mean(0) + err_translation.mean(0) + err_shape.mean(0) + err_point_cloud.mean(0)
    print("shape of loss: ", loss.shape)
    print("loss: ", loss)



