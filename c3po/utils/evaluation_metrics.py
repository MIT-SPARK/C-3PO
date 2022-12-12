# import copy
# import csv
import numpy as np
# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
# import random
# import string
import pickle
import sys
# import time
import torch
# import torch.nn.functional as F
from pytorch3d import ops
from pytorch3d import transforms

from c3po.datasets.ycb import MODEL_TO_KPT_GROUPS as ycb_kp_groups
from c3po.datasets.shapenet import MODEL_TO_KPT_GROUPS as shapenet_kp_groups


sys.path.append('../..')


def is_close(keypoints, pcd, threshold=0.015):
    pcd = pcd.unsqueeze(0)
    keypoints = keypoints.unsqueeze(0)
    closest_dist, _, _ = ops.knn_points(torch.transpose(keypoints, -1, -2), torch.transpose(pcd, -1, -2), K=1,
                                        return_sorted=False)
    if torch.max(torch.sqrt(closest_dist)) < threshold:
        return True
    return False


def is_pcd_nondegenerate(model_id, input_point_clouds, predicted_model_keypoints, model_to_kpt_groups):

    if model_id not in [*shapenet_kp_groups.keys()] + [*ycb_kp_groups.keys()]:
        b = input_point_clouds.shape[0]
        nd = torch.ones(b, 1).to(dtype=torch.bool)

    else:
        nondeg_indicator = torch.zeros(predicted_model_keypoints.shape[0]).to(input_point_clouds.device)
        for batch_idx in range(predicted_model_keypoints.shape[0]):
            predicted_kpts = predicted_model_keypoints[batch_idx]
            for group in model_to_kpt_groups[model_id]:
                kpt_group = predicted_kpts[:,list(group)]
                if is_close(kpt_group, input_point_clouds[batch_idx]):
                    nondeg_indicator[batch_idx] = 1
        nd = nondeg_indicator.to(torch.bool).unsqueeze(-1)
    return nd


def chamfer_dist(pc, pc_, pc_padding=None, max_loss=False):
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
        # print(pc.shape)
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)
        # pc_padding = torch.zeros(batch_size, n).to(device=device_)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    dist = torch.sqrt(sq_dist)

    a = torch.logical_not(pc_padding)

    if max_loss:
        loss = dist.max(dim=1)[0]
    else:
        loss = dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)


def translation_error(t, t_):
    """
    inputs:
    t: torch.tensor of shape (3, 1) or (B, 3, 1)
    t_: torch.tensor of shape (3, 1) or (B, 3, 1)

    output:
    t_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if t.dim() == 2:
        return torch.norm(t - t_, p=2)
    elif t.dim() == 3:
        return torch.norm(t-t_, p=2, dim=1)
    else:
        return ValueError


def rotation_error(R, R_):
    """
    inputs:
    R: torch.tensor of shape (3, 3) or (B, 3, 3)
    R_: torch.tensor of shape (3, 3) or (B, 3, 3)

    output:
    R_err: torch.tensor of shape (1, 1) or (B, 1)
    """

    epsilon = 1e-8
    if R.dim() == 2:
        error = 0.5*(torch.trace(R.T @ R_)-1)
        err = torch.clamp(error, -1 + epsilon, 1 - epsilon)
        return torch.arccos(err)

    elif R.dim() == 3:
        error = 0.5 * (torch.einsum('bii->b', torch.transpose(R, -1, -2) @ R_) - 1).unsqueeze(-1)
        err = torch.clamp(error, -1 + epsilon, 1 - epsilon)
        return torch.acos(err)

    else:
        return ValueError


def rotation_euler_error(R, R_): #no clamp
    """
    inputs:
    R: torch.tensor of shape (3, 3) or (B, 3, 3)
    R_: torch.tensor of shape (3, 3) or (B, 3, 3)

    output:
    R_err: torch.tensor of shape (1, 1) or (B, 1)
    """

    if R.dim() == 2:
        return transforms.matrix_to_euler_angles(torch.matmul(R.T, R_), "XYZ").abs().sum()/3.0
    elif R.dim() == 3:
        return transforms.matrix_to_euler_angles(torch.transpose(R, 1, 2) @ R_, "XYZ").abs().mean(1).unsqueeze(1)
    else:
        return ValueError


def rotation_matrix_error(R, R_):
    """
    R   : torch.tensor of shape (3, 3) or (B, 3, 3)
    R_  : torch.tensor of shape (3, 3) or (B, 3, 3)

    out: torch.tensor of shape (1,) or (B, 1)

    """
    device_ = R.device

    ErrorMat = R @ R_.transpose(-1, -2) - torch.eye(3).to(device=device_)

    return (ErrorMat**2).sum(dim=(-1, -2)).unsqueeze(-1)


def shape_error(c, c_):
    """
    inputs:
    c: torch.tensor of shape (K, 1) or (B, K, 1)
    c_: torch.tensor of shape (K, 1) or (B, K, 1)

    output:
    c_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if c.dim() == 2:
        return torch.norm(c - c_, p=2)/c.shape[0]
    elif c.dim() == 3:
        return torch.norm(c - c_, p=2, dim=1)/c.shape[1]
    else:
        return ValueError


def keypoints_error(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='none')
    kp_err = lossMSE(kp, kp_).sum(1)
    kp_err = torch.sqrt(kp_err)

    return kp_err.mean(1).unsqueeze(-1)


def evaluation_error(input, output):
    """
    inputs:
        input   : tuple of length 4 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
        output  : tuple of length 4 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    pc_loss = chamfer_dist(pc=input[0], pc_=output[0])
    pc_err = pc_loss

    kp_err = keypoints_error(input[1], output[1])

    R_err = rotation_error(input[2], output[2])
    t_err = translation_error(input[3], output[3])

    # print("pc_err shape: ", pc_err.shape)
    # print("kp_err shape: ", kp_err.shape)
    # print("R_err shape: ", R_err.shape)
    # print("t_err shape: ", t_err.shape)

    return pc_err, kp_err, R_err, t_err
    # return pc_loss


def adds_error(templet_pc, T1, T2):
    """
    Args:
        templet_pc: torch.tensor(B, 3, m) or (1, 3, m)  or (3, m)
        T1: torch.tensor(B, 4, 4)   or (4, 4)
        T2: torch.tensor(B, 4, 4)   or (4, 4)

    Returns:

    """
    # breakpoint()
    if len(templet_pc.shape) == 2:
        templet_pc = templet_pc.unsqueeze(0)
        T1 = T1.unsqueeze(0)
        T2 = T2.unsqueeze(0)

    pc1 = T1[:, :3, :3] @ templet_pc + T1[:, :3, 3:]
    pc2 = T2[:, :3, :3] @ templet_pc + T2[:, :3, 3:]

    err1 = chamfer_dist(pc1, pc2)
    err2 = chamfer_dist(pc2, pc1)
    err = err1 + err2
    if len(templet_pc.shape) == 2:
        err = err.squeeze(0)

    return err


# ADD-S and ADD-S (AUC)
def VOCap(rec, threshold):
    device_ = rec.device

    rec = torch.sort(rec)[0]
    rec = torch.where(rec <= threshold, rec, torch.tensor([float("inf")]).to(device=device_))

    n = rec.shape[0]
    prec = torch.cumsum(torch.ones(n)/n, dim=0)

    index = torch.isfinite(rec)
    rec = rec[index]
    prec = prec[index]
    # print(prec)
    # print(prec.shape)
    if rec.nelement() == 0:
        ap = torch.zeros(1)
    else:
        mrec = torch.zeros(rec.shape[0] + 2)
        mrec[0] = 0
        mrec[-1] = threshold
        mrec[1:-1] = rec

        mpre = torch.zeros(prec.shape[0]+2)
        mpre[1:-1] = prec
        mpre[-1] = prec[-1]

        for i in range(1, mpre.shape[0]):
            mpre[i] = max(mpre[i], mpre[i-1])

        ap = 0
        ap = torch.zeros(1)
        for i in range(mrec.shape[0]-1):
            # print("mrec[i+1] ", mrec[i+1])
            # print("mpre[i+1] ", mpre[i+1])
            ap += (mrec[i+1] - mrec[i]) * mpre[i+1] * (1/threshold)

    return ap


def add_s_error(predicted_point_cloud, ground_truth_point_cloud, threshold, certi=None, degeneracy_i=None, degenerate=False):
    """
    predicted_point_cloud       : torch.tensor of shape (B, 3, m)
    ground_truth_point_cloud    : torch.tensor of shape (B, 3, m)

    output:
    torch.tensor(dtype=torch.bool) of shape (B, 1)
    """

    # compute the chamfer distance between the two
    d = chamfer_dist(predicted_point_cloud, ground_truth_point_cloud, max_loss=False)
    if degeneracy_i is not None:
        if not degenerate:
            degeneracy_i = degeneracy_i > 0
        else:
            degeneracy_i = degeneracy_i < 1

    if certi is None:
        if degeneracy_i is not None:
            d = d[degeneracy_i]
        auc = VOCap(d.squeeze(-1), threshold=threshold)
    else:
        if degeneracy_i is not None:
            d = d[degeneracy_i.squeeze()*certi.squeeze()]
        else:
            d = d[certi]
        auc = VOCap(d.squeeze(-1), threshold=threshold)

    return d <= threshold, auc


def get_auc(rec, threshold):

    rec = np.sort(rec)
    rec = np.where(rec <= threshold, rec, np.array([float("inf")]))
    # print(rec)
    # print(rec.shape)
    # breakpoint()

    n = rec.shape[0]
    prec = np.cumsum(np.ones(n) / n, axis=0)

    index = np.isfinite(rec)
    rec = rec[index]
    prec = prec[index]

    if len(rec) == 0:
        # print("returns zero: ", 0.0)
        return np.asarray([0.0])[0]
    else:
        # print(prec)
        # print(prec.shape)
        mrec = np.zeros(rec.shape[0] + 2)
        mrec[0] = 0
        mrec[-1] = threshold
        mrec[1:-1] = rec

        mpre = np.zeros(prec.shape[0] + 2)
        mpre[1:-1] = prec
        mpre[-1] = prec[-1]

        for i in range(1, mpre.shape[0]):
            mpre[i] = max(mpre[i], mpre[i - 1])

        ap = 0
        ap = np.zeros(1)
        for i in range(mrec.shape[0] - 1):
            # print("mrec[i+1] ", mrec[i+1])
            # print("mpre[i+1] ", mpre[i+1])
            # ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1] * (1 / threshold)

        # print(ap)
        # print(type(ap))
        # print("returns ap: ", ap[0])
        # breakpoint()
        return ap[0]


class EvalData:
    def __init__(self, adds_th=0.02, adds_auc_th=0.05):
        self.eval_store_metrics = ["adds", "oc", "nd", \
                                   "rerr", "terr", \
                                   "adds_oc", "adds_oc_nd", \
                                   "adds_auc", "adds_oc_auc", "adds_oc_nd_auc", \
                                   "adds_th_score", "adds_oc_th_score", "adds_oc_nd_th_score", \
                                   "adds_th", "adds_auc_th"]

        self.data = dict()
        for metric in self.eval_store_metrics:
            self.data[metric] = None

        self.n = None
        self.data['adds_th'] = adds_th
        self.data['adds_auc_th'] = adds_auc_th

    def set_adds(self, adds_):
        self.data["adds"] = adds_
        self.n = len(adds_)

    def set_oc(self, oc_):
        self.data["oc"] = oc_

    def set_nd(self, nd_):
        self.data["nd"] = nd_

    def set_rerr(self, rerr_):
        self.data["rerr"] = rerr_
        self.n = len(rerr_)

    def set_terr(self, terr_):
        self.data["terr"] = terr_
        self.n = len(terr_)

    def set_adds_th(self, th_):
        self.data["adds_th"] = th_

    def set_adds_auc_th(self, th_):
        self.data["adds_auc_th"] = th_

    def complete_eval_data(self):

        # breakpoint()
        if self.n is None:
            self.n = len(self.data["adds"])

        # if oc or nd is None, we fill it with all ones
        if self.data["oc"] is None:
            self.data["oc"] = np.ones(self.n)

        if self.data["nd"] is None:
            self.data["nd"] = np.ones(self.n)

        self._check_to_numpy()

        # fill adds_oc, adds_oc_nd
        idx = np.where(self.data["oc"] == 1)[0]
        self.data["adds_oc"] = self.data["adds"][idx]

        idx = np.where(self.data["oc"] * self.data["nd"] == 1)
        self.data["adds_oc_nd"] = self.data["adds"][idx]

        # fill adds_th_score, adds_oc_th_score, adds_oc_nd_th_score
        self.data["adds_th_score"] = (self.data["adds"] <= self.data["adds_th"]).mean()
        # fill adds_auc, adds_oc_auc, adds_oc_nd_auc
        self.data["adds_auc"] = get_auc(self.data["adds"], self.data["adds_auc_th"])

        # oc
        if len(self.data["adds_oc"]) == 0:
            self.data["adds_oc_th_score"] = np.asarray([0.0])[0]
            self.data["adds_oc_auc"] = np.asarray([0.0])[0]
        else:
            self.data["adds_oc_th_score"] = (self.data["adds_oc"] <= self.data["adds_th"]).mean()
            self.data["adds_oc_auc"] = get_auc(self.data["adds_oc"], self.data["adds_auc_th"])

        # nd
        if len(self.data["adds_oc_nd"]) == 0:
            self.data["adds_oc_nd_th_score"] = np.asarray([0.0])[0]
            self.data["adds_oc_nd_auc"] = np.asarray([0.0])[0]
        else:
            self.data["adds_oc_nd_th_score"] = (self.data["adds_oc_nd"] <= self.data["adds_th"]).mean()
            self.data["adds_oc_nd_auc"] = get_auc(self.data["adds_oc_nd"], self.data["adds_auc_th"])

    def compute_oc(self):

        if self.data["oc"] is None or self.data["adds_oc"] is None:
            self.complete_eval_data()

        idx = np.where(self.data["oc"] == 1)[0]
        adds_oc = self.data["adds"][idx]
        rerr_oc = self.data["rerr"][idx]
        terr_oc = self.data["terr"][idx]

        OC = EvalData()
        OC.set_adds(adds_oc)
        OC.set_rerr(rerr_oc)
        OC.set_terr(terr_oc)
        OC.set_adds_th(self.data["adds_th"])
        OC.set_adds_auc_th(self.data["adds_auc_th"])
        OC.complete_eval_data()

        return OC

    def compute_ocnd(self):

        if self.data["oc"] is None or self.data["adds_oc"] is None:
            self.complete_eval_data()

        if self.data["nd"] is None or self.data["adds_oc_nd"] is None:
            self.complete_eval_data()

        idx = np.where(self.data["oc"] * self.data["nd"] == 1)
        adds_ocnd = self.data["adds"][idx]
        rerr_ocnd = self.data["rerr"][idx]
        terr_ocnd = self.data["terr"][idx]

        OCND = EvalData()
        OCND.set_adds(adds_ocnd)
        OCND.set_rerr(rerr_ocnd)
        OCND.set_terr(terr_ocnd)
        OCND.set_adds_th(self.data["adds_th"])
        OCND.set_adds_auc_th(self.data["adds_auc_th"])
        OCND.complete_eval_data()

        return OCND

    def _check_to_numpy(self):

        if isinstance(self.data["adds"], list):
            self.data["adds"] = np.asarray(self.data["adds"])

        if isinstance(self.data["rerr"], list):
            self.data["rerr"] = np.asarray(self.data["rerr"])

        if isinstance(self.data["terr"], list):
            self.data["terr"] = np.asarray(self.data["terr"])

        if isinstance(self.data["oc"], list):
            self.data["oc"] = np.asarray(self.data["oc"])

        if isinstance(self.data["nd"], list):
            self.data["nd"] = np.asarray(self.data["nd"])

    def print(self):
        """prints out the results"""

        raise NotImplementedError

    def save(self, filename):
        """saves object as a pickle file"""
        # breakpoint()
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """load object from pickle file"""

        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict
        # breakpoint()

