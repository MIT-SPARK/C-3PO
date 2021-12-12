"""
This code is written to test the workings of PACE module in learning_objects.models.basic
"""

import numpy as np
# import numpy.linalg as la
import cvxpy as cp
import torch
import torch.nn as nn
# import pymanopt
# from pymanopt.manifolds import PositiveDefinite
from cvxpylayers.torch import CvxpyLayer
from scipy.spatial.transform import Rotation
from pytorch3d import transforms
import os
import sys

sys.path.append("../../")

from learning_objects.models.basic import PACE
from learning_objects.utils.sdp_data import get_rotation_relaxation_constraints, get_vectorization_permutation
from learning_objects.utils.category_gnc import solve_3dcat_with_sdp



def shape_error(c, c_):
    """
    inputs:
    c: torch.tensor of shape (K, 1)
    c_: torch.tensor of shape (K, 1)

    output:
    c_err: torch.tensor of shape (1, 1)
    """

    return torch.norm(c - c_, p=2)/c.shape[0]

def translation_error(t, t_):
    """
    inputs:
    t: torch.tensor of shape (3, 1)
    t_: torch.tensor of shape (3, 1)

    output:
    t_err: torch.tensor of shape (1, 1)
    """

    return torch.norm(t - t_, p=2)/3.0


def rotation_error(R, R_):
    """
    inputs:
    R: torch.tensor of shape (3, 3)
    R_: torch.tensor of shape (3, 3)

    output:
    R_err: torch.tensor of shape (1, 1)
    """

    return transforms.matrix_to_euler_angles(torch.matmul(R.T, R_), "XYZ").abs().sum()/3.0



def check_rot_mat(R, tol=0.001):
    """
    This checks if the matrix R is a 3x3 rotation matrix
    R: rotation matrix: numpy.ndarray of size (3, 3)

    Output:
    True/False: determining if R is a rotation matrix/ or not
    """
    tol = tol
    if torch.norm( torch.matmul(R.T, R) - np.eye(3), ord='fro') < tol:
        return True
    else:
        return False


def test_cvxpylayers():
    """
    This is to test if cvxpylayers is working correctly.
    """

    # Setting the convex optimization layer
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=2))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

    # Generating data
    A_tch = torch.randn(m, n, requires_grad=True)
    b_tch = torch.randn(m, requires_grad=True)

    M = torch.matmul(A_tch.T, A_tch)
    h = torch.matmul(A_tch.T, b_tch)
    x = torch.matmul(torch.inverse(M), h)

    # solve the problem
    est_x, = cvxpylayer(A_tch, b_tch)
    # print(est_x)

    # compute the gradient of the sum of the solution with respect to A, b
    est_x.sum().backward()
    # print(est_x)

    print('x: ', x)
    print("est_x: ", est_x)

    est_error = torch.norm(x-est_x)
    print("Prediction error: ", est_error)


def generate_data(model_keypoints):
    """
    input:
    model_keypoints: torch.tensor of shape (K, 3, N)

    output:
    y: object keypoints: torch.tensor of shape (3, N)
    c: shape: torch.tensor of shape (K, 1)
    R: rotation: torch.tensor of shape (3, 3)
    t: translation: torch.tensor of shape (3, 1)
    """
    sigma_sq = 0.0001

    K = model_keypoints.shape[0]
    N = model_keypoints.shape[-1]

    # Generate random shape, translation, and rotation
    c = torch.rand(K) # shape (K)
    c = c/c.sum()
    t = torch.rand(3, 1)
    # t = torch.zeros(3, 1)
    R = torch.from_numpy(Rotation.random().as_matrix()).float()
    # R = torch.eye(3)


    y = torch.einsum('k,kdn->dn', c, model_keypoints) # (3, N)
    y = torch.matmul(R, y) + t # (3, N)
    y = y + torch.normal(mean=torch.zeros(y.shape), std=sigma_sq*torch.ones(y.shape))

    return y, c.unsqueeze(-1), R, t


def get_shape(model_keypoints, R, t, c):
    """
    input:
    model_keypoints: torch.tensor of shape (K, 3, N)
    R: torch.tensor of shape (3, 3)
    t: torch.tensor of shape (3, 1)
    c: torch.tensor of shape (K, 1)

    output:
    y: torch.tensor of shape (3, N)
    """

    y = torch.einsum('k,kdn->dn', c.squeeze(-1), model_keypoints)  # (3, N)
    y = torch.matmul(R, y) + t  # (3, N)

    return y


def simple_testPACE():
    """
    This is a simple test to see if PACE is working correctly.
    """

    # Testing A and P matrices
    K = 5
    N = 10
    weights = torch.ones(N, 1)
    model_keypoints = torch.rand(K, 3, N)
    pace_model = PACE(weights, model_keypoints)
    A = pace_model.A
    P = pace_model.P

    A_test_np, _ = get_rotation_relaxation_constraints()
    P_test_np = get_vectorization_permutation()
    A_test = torch.zeros(16, 10, 10)
    P_test = torch.from_numpy(P_test_np)
    for i in range(16):
        A_test[i, :, :] = torch.from_numpy(A_test_np[i])

    Aerr = torch.norm(A-A_test, p='fro')
    Perr = torch.norm(P-P_test, p='fro')

    print("Error in matrix A: ", Aerr)
    print("Error in matrix P: ", Perr)


    # Hault
    print("--------------------------")
    option = input("Do you want to proceed? (y/n)")
    if option != 'y':
        exit()

    # Testing on a problem
    # set problem parameters
    N = 44  # number of keypoints
    K = 200  # number of shapes
    weights = torch.ones(N, 1) # fixed weights
    lambda_constant = torch.tensor(N/K)
    lambda_constant = torch.sqrt(lambda_constant)

    # generate data
    model_keypoints = torch.rand(K, 3, N)
    y, c, R, t = generate_data(model_keypoints)
    y = y.unsqueeze(0)

    # initialize model
    pace_module = PACE(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)

    # predict with the model
    est_R, est_t, est_c = pace_module(y=y)
    est_R = est_R.squeeze(0)
    est_t = est_t.squeeze(0)
    est_c = est_c.squeeze(0)

    est_y = get_shape(model_keypoints=model_keypoints, R=est_R, t=est_t, c=est_c)

    shape_err =  shape_error(c, est_c)
    translation_err =  translation_error(t, est_t)
    rotation_err =  rotation_error(R, est_R)
    rot_matrix_error = torch.norm(torch.matmul(R.T, R) - torch.eye(3), p='fro')
    y = y.squeeze(0)
    keypoint_error = torch.norm(y - est_y, p=2)

    print("Rotation error: ", rotation_err)
    print("Translation error: ", translation_err)
    print("Shape error: ", shape_err)
    print("Rotation matrix error: ", rot_matrix_error)
    print("Keypoint error: ", keypoint_error)


def testPACE(K=10, N=100, S=1000):
    """
    This is to replicate PACE experiment in the paper [1], and therefore, see if the implementation is working correctly.
    """

    # parameters
    # N = N
    # K = K
    S = S
    weights = torch.ones(N, 1) # fixed weights
    lambda_constant = torch.tensor(N/K)
    lambda_constant = torch.sqrt(lambda_constant)

    # generate model data
    # model_keypoints = torch.rand(K, 3, N)
    model_keypoints = torch.normal(mean=torch.zeros(K, 3, N), std=torch.ones(K, 3, N))

    # initialize model
    pace_module = PACE(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)

    # experiment
    shape_err = torch.zeros(S)
    translation_err = torch.zeros(S)
    rotation_err = torch.zeros(S)
    keypoint_err = torch.zeros(S)

    for sample in range(S):
        y, c, R, t = generate_data(model_keypoints)
        y = y.unsqueeze(0)

        # predict with the model
        est_R, est_t, est_c = pace_module(y=y)
        est_R = est_R.squeeze(0)
        est_t = est_t.squeeze(0)
        est_c = est_c.squeeze(0)

        est_y = get_shape(model_keypoints=model_keypoints, R=est_R, t=est_t, c=est_c)

        shape_err[sample] = torch.norm(c - est_c, p=2)
        translation_err[sample] = torch.norm(t - est_t, p=2)
        rotation_err[sample] = transforms.matrix_to_euler_angles(torch.matmul(R.T, est_R), "XYZ").abs().sum()
        y = y.squeeze(0)
        keypoint_err[sample] = torch.norm(y - est_y, p=2)


    sta_shape_err = torch.std_mean(shape_err)
    sta_translation_err = torch.std_mean(translation_err)
    sta_rotation_err = torch.std_mean(rotation_err)
    sta_keypoint_err = torch.std_mean(keypoint_err)

    return sta_shape_err, sta_translation_err, sta_rotation_err, sta_keypoint_err


def exptPACE():
    """
    This is to replicate PACE experiment in the paper [1], and therefore, see if the implementation is working correctly.
    """

    Krange = [1, 10, 50, 100, 200]
    shape_err = []
    translation_err = []
    rotation_err = []
    keypoint_err = []

    for K in Krange:

        _shape_err, _translation_err, _rotation_err, _keypoint_err = testPACE(K=K, N=44, S=100)

        shape_err.append(_shape_err)
        translation_err.append(_translation_err)
        rotation_err.append(_rotation_err)
        keypoint_err.append(_keypoint_err)


    print("shape error (std & mean): ", shape_err)
    print("translation error (std & mean): ", translation_err)
    print("rotation error (std & mean): ", rotation_err)
    print("keypoint error (std & mean): ", keypoint_err)








def compare_testPACE():
    """
    This test is to compare our implementation with that of the original implementation by Jingnan and Henk.
    """

    # Generate inputs
    K = 5
    N = 10
    weights = torch.ones(N, 1) # fixed weights
    lambda_constant = torch.tensor(float(N)/float(K))
    model_keypoints = torch.rand(K, 3, N)

    # Generate data
    y, c, R, t = generate_data(model_keypoints=model_keypoints)


    # initialize our PACE model
    pace_module = PACE(weights=weights, model_keypoints=model_keypoints, lambda_constant=lambda_constant)
    R_us, t_us, c_us, y_w_us, bar_y_us, Q_us = pace_module(y=y)

    A_us = pace_module.A
    P_us = pace_module.P
    b_w_us = pace_module.b_w.squeeze(0)
    bar_B_us = pace_module.bar_B.squeeze(0)
    G_us = pace_module.G.squeeze(0)
    g_us = pace_module.g.squeeze(0)
    M_us = pace_module.M.squeeze(0)
    h_us = pace_module.h.squeeze(0)

    R_us = R_us.squeeze(0)
    t_us = t_us.squeeze(0)
    c_us = c_us.squeeze(0)
    y_w_us = y_w_us.squeeze(0)
    bar_y_us = bar_y_us.squeeze(0)
    Q_us = Q_us.squeeze(0)

    R_us_err = rotation_error(R_us, R)
    t_us_err = translation_error(t_us, t)
    c_us_err = shape_error(c_us, c)

    print("Our implementation:")
    print("Rotation error: ", R_us_err)
    print("Translation error: ", t_us_err)
    print("Shape error: ", c_us_err)
    print("-"*20)

    # PACE model implementation by Jingnan and Henk
    R_ji, t_ji, c_ji, _, _, b_w_ji, Bbar_ji, G_ji, g_ji, M_ji, h_ji, y_w_ji, Ybar_ji, Q_ji = solve_3dcat_with_sdp(tgt=y.numpy(), cad_kpts=model_keypoints.numpy(), weights=weights.squeeze(-1).numpy(), lam=lambda_constant.numpy(), print_info=False)
    R_ji = torch.from_numpy(R_ji).float()
    t_ji = torch.from_numpy(t_ji).unsqueeze(-1).float()
    c_ji = torch.from_numpy(c_ji).unsqueeze(-1).float()
    b_w_ji = torch.from_numpy(b_w_ji).float()
    Bbar_ji = torch.from_numpy(Bbar_ji).float()
    G_ji = torch.from_numpy(G_ji).float()
    g_ji = torch.from_numpy(g_ji).float()
    M_ji = torch.from_numpy(M_ji).float()
    h_ji = torch.from_numpy(h_ji).float()

    y_w_ji = torch.from_numpy(y_w_ji).unsqueeze(-1).float()
    Ybar_ji = torch.from_numpy(Ybar_ji).float()
    Q_ji = torch.from_numpy(Q_ji).float()

    A_temp, b_ji = get_rotation_relaxation_constraints()
    P_ji = get_vectorization_permutation()
    A_ji = np.zeros((16, 10, 10))
    for i in range(16):
        A_ji[i, :, :] = A_temp[i][:, :]

    A_ji = torch.from_numpy(A_ji).float()
    P_ji = torch.from_numpy(P_ji).float()


    R_ji_err = rotation_error(R_ji, R)
    t_ji_err = translation_error(t_ji, t)
    c_ji_err = shape_error(c_ji, c)

    print("Jingnan's implementation:")
    print("Rotation error: ", R_ji_err)
    print("Translation error: ", t_ji_err)
    print("Shape error: ", c_ji_err)
    print("-"*20)


    print("Internal parameter computation errors:")
    print("Error in matrix A: ", torch.norm(A_ji-A_us, p=2))
    print("Error in matrix P: ", torch.norm(P_ji-P_us, p=2))
    print("Error in b_w: ", torch.norm(b_w_us-b_w_ji, p=2))
    print("Error in bar_B: ", torch.norm(bar_B_us-Bbar_ji, p=2))
    print("Error in G: ", torch.norm(G_us-G_ji, p=2))
    print("Error in g: ", torch.norm(g_us-g_ji, p=2))
    print("Error in M: ", torch.norm(M_us-M_ji, p=2))
    print("Error in h: ", torch.norm(h_us-h_ji, p=2))

    print("Error in y_w: ", torch.norm(y_w_us-y_w_ji, p=2))
    print("Error in bar_y: ", torch.norm(Ybar_ji-bar_y_us, p=2))
    print("Error in Q: ", torch.norm(Q_us - Q_ji, p=2))

if __name__ == '__main__':

    # test_cvxpylayers()

    # simple_testPACE()

    exptPACE()

    # compare_testPACE()