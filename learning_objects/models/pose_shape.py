import numpy as np
import numpy.linalg as la
import cvxpy as cp
import torch
import torch.nn as nn
import pymanopt
from pymanopt.manifolds import PositiveDefinite
from cvxpylayers.torch import CvxpyLayer
from scipy.spatial.transform import Rotation
from pytorch3d import transforms




class PACE(nn.Module):
    def __init__(self, weights, model_keypoints, lambda_constant=torch.tensor(1.0)):
        super().__init__()
        """
        weights: torch.tensor of shape (N, 1)
        model_keypoints: torch.tensor of shape (K, 3, N) 
        lambda_constant: int or float32 or double  
        """

        self.w = weights.unsqueeze(0)               # (1, N, 1)
        self.b = model_keypoints                    # (K, 3, N)
        self.lambda_constant = lambda_constant      # (1, 1)

        # self.N = np.shape(b[0])[1]  np to tensor
        # self.K = len(b)  np to tensor
        self.N = self.b.shape[-1]                        # (1, 1)
        self.K = self.b.shape[0]                         # (1, 1)

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


        # Defining the SDP Layer
        X = cp.Variable((10, 10), symmetric=True)
        Q = cp.Parameter((10, 10), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(self.A[i, :, :] @ X) == self.d[i] for i in range(16)
        ]
        problem = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints=constraints)
        assert problem.is_dpp()

        self.sdp_for_rotation = CvxpyLayer(problem, parameters=[Q], variables=[X])


    def forward(self, y):
        """
        input:
        y: torch.tensor of shape (B, 3, N), where B = batch size

        output:
        R: torch.tensor of shape (B, 3, 3), where B = batch size
        t: torch.tensor of shape (B, 3, 1), where B = batch size
        c: torch.tensor of shape (B, K, 1), where B = batch size
        """

        y_w = self._get_y_w(y=y)
        bar_y = self._get_bar_y(y=y, y_w=y_w)

        R = self._rotation(bar_y=bar_y)
        c = self._shape(bar_y=bar_y, R=R)
        t = self._translation(y_w=y_w, R=R, c=c)

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

        R = torch.zeros(bar_y.shape[0], 3, 3)

        M = self.M.squeeze(0)
        h = self.h.squeeze(0)

        for batch in range(bar_y.shape[0]):

            Y = bar_y[batch, :, :]

            Q = torch.zeros(10, 10)
            Q[0, 0] = torch.matmul(h.T, h)
            tempA = torch.matmul(h.T, M)
            tempB = Y.T
            tempB = tempB.contiguous()
            tempB = torch.kron(tempB, torch.eye(3))
            tempC = torch.matmul(tempB, self.P)
            Q[0, 1:] = torch.matmul(tempA, tempC)
            Q[1:, 0] = Q[0, 1:].T

            tempD = torch.matmul(M, tempB)
            tempE = torch.matmul(tempD, self.P)
            Q[1:, 1:] =  torch.matmul(tempE.T, tempE)

            tempR = self._get_rotation(Q=Q)

            R[batch, :, :] = tempR


        return R


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
        tempL = torch.kron(torch.eye(self.N), tempK)
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
        X, = self.sdp_for_rotation(Q)


        # Step(2): projects X.value onto the rank 1 PSD manifold and extracts the rotation matrix R
        u, s, vh = torch.linalg.svd(X)
        u0 = torch.sqrt(s[0]) * u[:, 0]
        u0 = u0 / u0[0]
        rvec = u0[1:]
        R = rvec.reshape(3, 3)

        return R


    def _get_b_w(self):
        """
        intermediate:
        self.b: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)

        output:
        b_w: torch.tensor of shape (1, K, 3)
        """

        b_w = torch.matmul(self.b, self.w)/self.w.sum() # (K, 3, 1)

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
        self.b: torch.tensor of shape (K, 3, N)
        self.w: torch.tensor of shape (1, N, 1)
        b_w: torch.tensor of shape (1, K, 3)

        output:
        bar_B: torch.tensor of shape (1, 3N, K), where B = batch size
        """

        bar_b = torch.sqrt(torch.transpose(self.w, -1, -2))*(self.b - self.b_w.squeeze(0).unsqueeze(-1)) # (K, 3, N)
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

        bar_H = 2 * (torch.matmul(bar_B.T, bar_B) + self.lambda_constant*torch.eye(self.K))
        bar_Hinv = torch.inverse(bar_H)
        Htemp = torch.matmul(bar_Hinv, torch.ones(bar_Hinv.shape[-1], 1))

        G = bar_Hinv - (torch.matmul(Htemp, Htemp.T))/(torch.matmul(torch.ones(1, Htemp.shape[0]), Htemp)) # (K, K)
        g = Htemp/(torch.matmul(torch.ones(1, Htemp.shape[0]), Htemp))  # (K, 1)

        M = torch.zeros(3*self.N + self.K, 3*self.N)   # (3N+K, 3N)
        h = torch.zeros(3*self.N + self.K, 1)   # (3N+K, 1)

        M[0:3*self.N, :] = 2*torch.matmul( bar_B, torch.matmul(G, bar_B.T) ) - torch.eye(3*self.N)
        M[3*self.N:, :] = 2*torch.sqrt(self.lambda_constant)*torch.matmul(G, bar_B.T)

        h[0:3*self.N, :] = torch.matmul(self.bar_B, g)
        h[3*self.N:, :] = g

        return G.unsqueeze(0), g.unsqueeze(0), M.unsqueeze(0), h.unsqueeze(0)


    def _getA(self):
        """
        output:
        A: torch.tensor of shape (16, 10, 10)
        """

        A = torch.zeros(16, 10, 10)

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

        A[5, 3, 4] = 1
        A[5, 1, 6] = 1
        A[5, 3, 9] = 1
        A[5, 4, 3] = 1
        A[5, 6, 1] = 1
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

        d = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return d


    def _getP(self):
        """
        output:
        P: torch.tensor of shape (9, 9)
        """

        P = torch.zeros((9, 9))

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





##############################


def test():
    u = np.random.rand(3, 1)
    A = np.matmul(u, np.transpose(u))
    print(u)
    print(A)

    u, s, vh = np.linalg.svd(A)
    u[:, 1:] = np.zeros((3, 2))
    vh[1:, :] = np.zeros((2, 3))
    print(s)
    print(u)
    print(vh)
    Astar = np.matmul(np.matmul(u, np.diag(s)), vh)
    print(Astar)


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

    shape_err = torch.norm(c - est_c, p=2)
    translation_err = torch.norm(t - est_t, p=2)
    rotation_err = torch.norm(R - est_R, p='fro')
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

    Krange = [10, 50, 100, 200]
    shape_err = []
    translation_err = []
    rotation_err = []
    keypoint_err = []

    for K in Krange:

        _shape_err, _translation_err, _rotation_err, _keypoint_err = testPACE(K=K, N=44, S=10)

        shape_err.append(_shape_err)
        translation_err.append(_translation_err)
        rotation_err.append(_rotation_err)
        keypoint_err.append(_keypoint_err)


    print("shape error (std & mean): ", shape_err)
    print("translation error (std & mean): ", translation_err)
    print("rotation error (std & mean): ", rotation_err)
    print("keypoint error (std & mean): ", keypoint_err)




if __name__ == '__main__':
    # test()

    # test_cvxpylayers()

    # simple_testPACE()

    exptPACE()