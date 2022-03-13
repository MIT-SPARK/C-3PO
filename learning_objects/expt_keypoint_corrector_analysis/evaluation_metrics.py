import torch
from pytorch3d import ops
import sys
sys.path.append('../..')


def chamfer_metric(pc, pc_, pc_padding=None, max_dist=False):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
    max_dist : boolean : indicates if output metric should be maximum of the distances between pc and pc_ instead of the mean

    output:
    metric    : (B, 1)
        returns maximum dist between points if max_dist is true, else an average distance
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

    if max_dist:
        metric = dist.max(dim=1)[0]
    else:
        metric = dist.sum(dim=1)/a.sum(dim=1)

    return metric.unsqueeze(-1)