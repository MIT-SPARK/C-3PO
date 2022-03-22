
import torch
import sys
from pytorch3d import ops
sys.path.append('../..')


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
        return torch.norm(t - t_, p=2)/3.0
    elif t.dim() == 3:
        return torch.norm(t-t_, p=2, dim=1)/3.0
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

    if R.dim() == 2:
        return torch.arccos(0.5*(torch.trace(R.T @ R)-1))
        # return transforms.matrix_to_euler_angles(torch.matmul(R.T, R_), "XYZ").abs().sum()/3.0
        # return torch.abs(0.5*(torch.trace(R.T @ R_) - 1).unsqueeze(-1))
        # return 1 - 0.5*(torch.trace(R.T @ R_) - 1).unsqueeze(-1)
        # return torch.norm(R.T @ R_ - torch.eye(3, device=R.device), p='fro')
    elif R.dim() == 3:
        # return transforms.matrix_to_euler_angles(torch.transpose(R, 1, 2) @ R_, "XYZ").abs().mean(1).unsqueeze(1)
        error = 0.5 * (torch.einsum('bii->b', torch.transpose(R, -1, -2) @ R_) - 1).unsqueeze(-1)
        epsilon = 1e-8
        return torch.acos(torch.clamp(error, -1 + epsilon, 1 - epsilon))
        # return 1 - 0.5 * (torch.einsum('bii->b', torch.transpose(R, 1, 2) @ R_) - 1).unsqueeze(-1)
        # return torch.norm(R.transpose(-1, -2) @ R_ - torch.eye(3, device=R.device), p='fro', dim=[1, 2])
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
        ap = torch.zeros(1)[0]
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


def add_s_error(predicted_point_cloud, ground_truth_point_cloud, threshold, certi=None):
    """
    predicted_point_cloud       : torch.tensor of shape (B, 3, m)
    ground_truth_point_cloud    : torch.tensor of shape (B, 3, m)

    output:
    torch.tensor(dtype=torch.bool) of shape (B, 1)
    """

    # compute the chamfer distance between the two
    d = chamfer_dist(predicted_point_cloud, ground_truth_point_cloud, max_loss=False)

    if certi==None:
        auc = VOCap(d.squeeze(-1), threshold=threshold)
    else:
        d = d[certi]
        auc = VOCap(d.squeeze(-1), threshold=threshold)

    return d <= threshold, auc

