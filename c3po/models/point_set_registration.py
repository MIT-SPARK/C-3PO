"""
This code implements outlier-free point set registration as torch function

"""
import sys
import time
import torch
from pytorch3d import transforms
from c3po.utils.evaluation_metrics import translation_error, rotation_matrix_error

sys.path.append("../../")

def wahba(source_points, target_points, device_=None):
    """
    inputs:
    source_points: torch.tensor of shape (B, 3, N)
    target_points: torch.tensor of shape (B, 3, N)

    where
        B = batch size
        N = number of points in each point set

    output:
    R   : torch.tensor of shape (B, 3, 3)
    """
    batch_size = source_points.shape[0]

    if device_==None:
        device_ = source_points.device

    mat = target_points @ source_points.transpose(-1, -2)   # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(mat)

    D = torch.eye(3).to(device=device_)     # (3, 3)
    D = D.unsqueeze(0)                      # (1, 3, 3)
    D = D.repeat(batch_size, 1, 1)          # (B, 3, 3)

    D[:, 2, 2] = torch.linalg.det(U)*torch.linalg.det(Vh)

    return U @ D @ Vh   # (B, 3, 3)

class PointSetRegistration:
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, N)
        
        """

        self.source_points = source_points

    def forward(self, target_points, weights=None, device_=None):
        """
        inputs:
        target_points   : torch.tensor of shape (B, 3, N)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)
        """
        batch_size = target_points.shape[0]

        source_points = self.source_points.repeat(batch_size, 1, 1)

        batch_size, d, N = source_points.shape

        if device_==None:
            device_ = source_points.device

        if weights==None:
            weights = torch.ones((1, N), device=device_)

        source_points_ave = torch.einsum('bdn,ln->bd', source_points, weights)/weights.sum()    # (B, 3)
        target_points_ave = torch.einsum('bdn,ln->bd', target_points, weights) / weights.sum()  # (B, 3)

        # getting the rotation
        source_points_centered = source_points - source_points_ave.unsqueeze(-1)    # (B, 3, N)
        target_points_centered = target_points - target_points_ave.unsqueeze(-1)    # (B, 3, N)

        source_points_centered = torch.einsum('bdn,ln->bdn', source_points_centered, weights)   # (B, 3, N)
        target_points_centered = torch.einsum('bdn,ln->bdn', target_points_centered, weights)   # (B, 3, N)

        rotation = wahba(source_points=source_points_centered, target_points=target_points_centered)

        # getting the translation
        translation = target_points_ave.unsqueeze(-1) - rotation @ source_points_ave.unsqueeze(-1)

        return rotation, translation



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)


    B = 2
    N = 10
    d = 3

    source_points = torch.rand(1, d, N).to(device=device)
    print(source_points.shape)
    registration_fxn = PointSetRegistration(source_points=source_points)

    rotation = transforms.random_rotations(B).to(device=device)

    target_points = rotation @ source_points
    target_points += 0.01*torch.rand(size=target_points.shape).to(device=device)
    target_points.requires_grad = True

    print("target_points.shape", target_points.shape)
    print('-' * 40)
    print("Testing wahba()")
    print('-' * 40)
    source_points_repeat = source_points.repeat(B, 1, 1)
    start = time.process_time()
    rotation_est = wahba(source_points=source_points_repeat - source_points_repeat.mean(-1).unsqueeze(-1),
                         target_points=target_points-target_points.mean(-1).unsqueeze(-1))
    end = time.process_time()
    print("Output shape: ", rotation_est.shape)

    err = rotation_matrix_error(rotation, rotation_est)
    print("Rotation error: ", err.mean())
    print("Time for wahba: ", 1000*(end-start)/B, ' ms')

    loss = err.sum()
    loss.backward()
    print("Target point gradient: ", target_points.grad)


    print('-'*40)
    print("Testing point_set_registration()")
    print('-' * 40)



    B = 20
    N = 10
    d = 3

    source_points = torch.rand(1, d, N).to(device=device)

    rotation = transforms.random_rotations(B).to(device=device)
    translation = torch.rand(B, d, 1).to(device=device)

    target_points = rotation @ source_points + translation
    target_points += 0.1*torch.rand(size=target_points.shape).to(device=device)
    target_points.requires_grad = True

    start = time.process_time()
    rotation_est, translation_est = registration_fxn.forward(target_points=target_points)
    end = time.process_time()

    print("Output rotation shape: ", rotation.shape)
    print("Output translation shape: ", translation.shape)

    err_rot = rotation_error(rotation, rotation_est)
    err_trans = translation_error(translation, translation_est)
    print("Rotation error: ", err_rot.mean())
    print("Translation error: ", err_trans.mean())
    print("Time for point_set_registration: ", 1000*(end-start)/B, ' ms')


    loss = 100*err_rot.sum() + 100*err_trans.sum()
    loss.backward()
    print("Target point gradient: ", target_points.grad)

