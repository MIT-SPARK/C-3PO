import numpy as np
import open3d as o3d
import sys
import torch

sys.path.append("../../")

from learning_objects.utils.general import pos_tensor_to_o3d


# Baseline Implementations:
def ransac(source_points, target_points):
    """
    inputs:
    source_points   : torch.tensor of shape (3, m)
    target_points   : torch.tensor of shape (3, m)

    outputs:
    R   : torch.tensor of shape (3, 3)
    t   : torch.tensor of shape (3, 1)

    Note:
        Input and output will be on the same device, while compute will happen on cpu.

    """
    _, m = source_points.shape
    device_ = source_points.device

    # converting to open3d
    src = pos_tensor_to_o3d(pos=source_points.to('cpu'), estimate_normals=False)
    tar = pos_tensor_to_o3d(pos=target_points.to('cpu'), estimate_normals=False)

    # Initializing the correspondences
    a = torch.arange(0, m, 1).unsqueeze(0)
    c = torch.cat([a, a], dim=0).T
    d = c.numpy().astype('int32')
    corres_init = o3d.utility.Vector2iVector(d)

    # ransac from open3d
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src,
        target=tar,
        corres=corres_init,
        max_correspondence_distance=0.001)
    # The following is from open3d, just for reference: #ToDo: remove in the final version.
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh, True,
    #     distance_threshold,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     3, [
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
    #             0.9),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
    #             distance_threshold)
    #     ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    # extracting result
    T = result_ransac.transformation
    R_ = np.array(T[:3, :3])
    t_ = np.array(T[:3, 3])
    R = torch.from_numpy(R_)
    t = torch.from_numpy(t_)
    t = t.unsqueeze(-1)
    # print("R shape: ", R.shape)
    # print("t shape: ", t.shape)

    return R.to(device=device_), t.to(device=device_)


def teaser(source_points, target_points):
    """
    inputs:
    source_points   : torch.tensor of shape (3, m)
    target_points   : torch.tensor of shape (3, n)

    outputs:
    R   : torch.tensor of shape (3, 3)
    t   : torch.tensor of shape (3, 1)

    Note:
        Input and output will be on the same device, while compute will happen on cpu.

    """
    device_ = source_points.device
    # print("Here!")

    # convert source_points, target_points to numpy src, tar
    src = source_points.to('cpu').numpy()
    tar = target_points.to('cpu').numpy()

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.05
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, tar)

    solution = solver.getSolution()

    R = solution.rotation
    t = solution.translation
    R = torch.from_numpy(R)
    t = torch.from_numpy(t)
    t = t.unsqueeze(-1)

    return R.to(device=device_), t.to(device=device_)


def icp(source_points, target_points, R0, t0):
    """
    inputs:
    source_points   : torch.tensor of shape (3, m)
    target_points   : torch.tensor of shape (3, n)
    R0              : torch.tensor of shape (3, 3)
    t0              : torch.tensor of shape (3, 1)

    outputs:
    R   : torch.tensor of shape (3, 3)
    t   : torch.tensor of shape (3, 1)

    Note:
        Input and output will be on the same device, while compute will happen on cpu.

    """

    # converting to open3d
    src = pos_tensor_to_o3d(pos=source_points.to('cpu'), estimate_normals=False)
    tar = pos_tensor_to_o3d(pos=target_points.to('cpu'), estimate_normals=False)

    # transformation
    T = torch.zeros(4, 4).to('cpu')
    T[:3, :3] = R0.to('cpu')
    T[:3, 3:] = t0.to('cpu')
    T = T.numpy()

    # icp from open3d
    reg_p2p = o3d.pipelines.registration.registration_icp(src, tar, 0.01, T,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(
                                                              max_iteration=200))

    # extracting results
    T = reg_p2p.transformation
    R_ = np.array(T[:3, :3])
    t_ = np.array(T[:3, 3])
    R = torch.from_numpy(R_)
    t = torch.from_numpy(t_)
    t = t.unsqueeze(-1)

    return R, t


class RANSAC():
    """
    This code implements batch RANSAC for input, output given as torch.tensors.

    """
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, m)
        
        """

        self.source_points = source_points.squeeze(0)
        self.device_ = source_points.device

    def forward(self, target_points):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]

        R = torch.zeros(batch_size, 3, 3).to(device=self.device_)
        t = torch.zeros(batch_size, 3, 1).to(device=self.device_)

        for b in range(batch_size):
            tar = target_points[b, ...]
            R_batch, t_batch = ransac(source_points=self.source_points, target_points=tar)
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t


class TEASER():
    """
    This code implements batch TEASER++ for input, output given as torch.tensors.
    """
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, m)

        """

        self.source_points = source_points
        self.device_ = source_points.device

    def forward(self, target_points):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]

        R = torch.zeros(batch_size, 3, 3).to(device=self.device_)
        t = torch.zeros(batch_size, 3, 1).to(device=self.device_)

        for b in range(batch_size):
            tar = target_points[b, ...]
            R_batch, t_batch = teaser(source_points=self.source_points, target_points=tar)
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t


class ICP():
    """
    This code implements batch ICP for input, output given as torch.tensors.
    """
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, m)

        """

        self.source_points = source_points.squeeze(0)

    def forward(self, target_points, R0, t0):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)
        R0              : torch.tensor of shape (B, 3, 3)
        t0              : torch.tensor of shape (B, 3, 1)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]

        R = torch.zeros_like(R0)
        t = torch.zeros_like(t0)

        for b in range(batch_size):

            # removes the padded zero points
            tarX = target_points[b, ...]
            idx = torch.sum(tarX == 0, dim=0) == 3
            tar = tarX[:, torch.logical_not(idx)]  # (3, n')

            # icp
            R_batch, t_batch = icp(source_points=self.source_points,
                                   target_points=tar,
                                   R0=R0[b, ...],
                                   t0=t0[b, ...])
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t


class RANSACwICP():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        model_keypoints     : torch.tensor of shape (1, 3, K)
        
        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints

        self.RANSAC = RANSAC(source_points=self.model_keypoints)
        self.ICP = ICP(source_points=self.cad_models)

    def forward(self, input_point_cloud, detected_keypoints):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)
        detected_keypoints  : torch.tensor of shape (B, 3, K)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """

        _, _, m = input_point_cloud.shape

        # centering. This considers that we may have padded zero points.
        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        kp_centered = detected_keypoints - center

        # global registration
        R0, t0 = self.RANSAC.forward(target_points=kp_centered)

        # icp
        R, t = self.ICP.forward(target_points=pc_centered, R0=R0, t0=t0)

        # re-centering
        t = t + center

        return R @ self.cad_models + t, R, t


class TEASERwICP():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        model_keypoints     : torch.tensor of shape (1, 3, K)

        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints

        self.TEASER = TEASER(source_points=self.model_keypoints)
        self.ICP = ICP(source_points=self.cad_models)

    def forward(self, input_point_cloud, detected_keypoints):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)
        detected_keypoints  : torch.tensor of shape (B, 3, K)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """

        _, _, m = input_point_cloud.shape

        # centering. This considers that we may have padded zero points.
        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        kp_centered = detected_keypoints - center

        # global registration
        R0, t0 = self.TEASER.forward(target_points=kp_centered)

        # icp
        R, t = self.ICP.forward(target_points=pc_centered, R0=R0, t0=t0)

        # re-centering
        t = t + center

        return R @ self.cad_models + t, R, t


class wICP():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        model_keypoints     : torch.tensor of shape (1, 3, K)

        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints

        self.TEASER = TEASER(source_points=self.model_keypoints)
        self.ICP = ICP(source_points=self.cad_models)

    def forward(self, input_point_cloud, R0, t0):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)
        R0                  : torch.tensor of shape (B, 3, 3)
        t0                  : torch.tensor of shape (B, 3, 1)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """
        _, _, m = input_point_cloud.shape

        # centering. This considers that we may have padded zero points.
        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        t0 = t0 - center

        # icp
        R, t = self.ICP.forward(target_points=pc_centered, R0=R0, t0=t0)

        # re-centering
        t = t + center

        #
        # R = R0
        # t = t0 + center

        return R @ self.cad_models + t, R, t