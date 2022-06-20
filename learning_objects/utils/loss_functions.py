import sys
import torch
from pytorch3d import ops

sys.path.append('../..')

from learning_objects.models.certifiability import confidence, confidence_kp

def keypoints_loss(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='none')
    kp_loss = lossMSE(kp, kp_)
    kp_loss = kp_loss.sum(dim=1).sum(dim=1).mean()  # Note: this is not normalized by number of keypoints

    return kp_loss

def avg_kpt_distance_regularizer(kp):
    kp_contiguous = torch.transpose(kp, -1, -2).contiguous()
    euclidian_dists = torch.cdist(kp_contiguous, kp_contiguous, p=2)
    euclidian_dists_squared = torch.square(euclidian_dists)
    #this takes euclidian norms of all pairs, so I want the maximum
    #we want to maximize distance from one point to all others, so sum up
    #all rows (distance to itself will be 0)
    #take avg of that sum, subtract it from the loss
    return torch.mean(euclidian_dists_squared)

def rotation_loss(R, R_):

    device_ = R.device

    err_mat = R @ R_.transpose(-1, -2) - torch.eye(3, device=device_)
    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(err_mat, torch.zeros_like(err_mat))


def translation_loss(t, t_):
    """
    t   : torch.tensor of shape (B, 3, N)
    t_  : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(t, t_)

def shape_loss(c, c_):
    """
    c   : torch.tensor of shape (B, K, 1)
    c_  : torch.tensor of shape (B, K, 1)

    """

    lossMSE = torch.nn.MSELoss(reduction='mean')

    return lossMSE(c, c_)


def chamfer_loss(pc, pc_, pc_padding=None, max_loss=False):
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
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)

    if max_loss:
        loss = sq_dist.max(dim=1)[0]
    else:
        loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)


# supervised training and validation losses
def supervised_training_loss(input, output):
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
    pc_loss = pc_loss.mean()

    kp_loss = keypoints_loss(input[1], output[1])

    R_loss = rotation_loss(input[2], output[2]).mean()
    t_loss = translation_loss(input[3], output[3]).mean()

    return pc_loss + kp_loss + R_loss + t_loss

def supervised_validation_loss(input, output):
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
    pc_loss = pc_loss.mean()

    kp_loss = keypoints_loss(input[1], output[1])

    R_loss = rotation_loss(input[2], output[2]).mean()
    t_loss = translation_loss(input[3], output[3]).mean()

    return pc_loss + kp_loss + R_loss + t_loss
    # return kp_loss


# self-supervised training and validation losses
def certify(input_point_cloud, predicted_point_cloud, corrected_keypoints,
            predicted_model_keypoints, epsilon=0.99):
    """
    inputs:
    input_point_cloud           : torch.tensor of shape (B, 3, m)
    predicted_point_cloud       : torch.tensor of shape (B, 3, n)
    corrected_keypoints         : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    certificate     : torch.tensor of shape (B, 1)  : dtype = torch.bool

    """

    confidence_ = confidence(input_point_cloud, predicted_point_cloud)
    confidence_kp_ = confidence_kp(corrected_keypoints, predicted_model_keypoints)

    out = (confidence_ >= epsilon) & (confidence_kp_ >= epsilon)

    return out


def self_supervised_training_loss(input_point_cloud, predicted_point_cloud, keypoint_correction, certi, theta=25.0):
    """
    inputs:
    input_point_cloud       : torch.tensor of shape (B, 3, m)
    predicted_point_cloud   : torch.tensor of shape (B, 3, n)
    keypoint_correction     : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    loss    : torch.tensor of shape (1,)

    """
    device_ = input_point_cloud.device
    theta = torch.tensor([theta]).to(device=device_)

    if certi.sum() == 0:
        print("NO DATA POINT CERTIFIABLE IN THIS BATCH")
        pc_loss = torch.tensor([0.0]).to(device=device_)
        kp_loss = torch.tensor([0.0]).to(device=device_)
        fra_certi = torch.tensor([0.0]).to(device=device_)
        pc_loss.requires_grad = True
        kp_loss.requires_grad = True
        fra_certi.requires_grad = True

    else:
        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi / certi.shape[0]  # not to be used for training

        # pc loss
        pc_loss = chamfer_loss(pc=input_point_cloud,
                               pc_=predicted_point_cloud)  # Using normal chamfer loss here, as the max chamfer is used in certification
        pc_loss = pc_loss * certi
        pc_loss = pc_loss.sum() / num_certi

        lossMSE = torch.nn.MSELoss(reduction='none')
        if keypoint_correction is None:
            kp_loss = torch.zeros(pc_loss.shape)
        else:
            kp_loss = lossMSE(keypoint_correction, torch.zeros_like(keypoint_correction))
            kp_loss = kp_loss.sum(dim=1).mean(dim=1)    # (B,)
            kp_loss = kp_loss * certi
            kp_loss = kp_loss.mean()

    # return pc_loss + theta*kp_loss, pc_loss, kp_loss, fra_certi   # pointnet
    return theta * pc_loss + kp_loss, pc_loss, kp_loss, fra_certi        # point_transformer: we will try this, as the first gave worse performance for pointnet.


def self_supervised_validation_loss(input_pc, predicted_pc, certi=None):
    """
    inputs:
        input_pc        : torch.tensor of shape (B, 3, m) : input_point_cloud
        predicted_pc    : torch.tensor of shape (B, 3, n) : predicted_point_cloud
        certi           : None or torch.tensor(dtype=torch.bool) of shape (B,)  : certification

    outputs:
        loss    : torch.tensor of shape (1,)

    """

    if certi == None:
        pc_loss = chamfer_loss(pc=input_pc, pc_=predicted_pc)
        vloss = pc_loss.mean()
    else:
        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi / certi.shape[0]
        vloss = -fra_certi

    return vloss
