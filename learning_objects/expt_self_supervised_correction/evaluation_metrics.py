
import torch
import sys
sys.path.append('../..')

from learning_objects.expt_self_supervised_correction.loss_functions import chamfer_loss


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
        return torch.acos(0.5*(torch.einsum('bii->b', torch.transpose(R, -1, -2) @ R_) - 1).unsqueeze(-1))
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

    return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)


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

    pc_loss = chamfer_loss(pc=input[0], pc_=output[0])
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