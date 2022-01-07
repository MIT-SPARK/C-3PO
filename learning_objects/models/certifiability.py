"""
This code defines a metric of (epsilon, delta)-certifiability. Given two registered, shape aligned point clouds
pc and pc_ it determines if the registration + shape alignment is certifiable or not.

"""

import torch

from learning_objects.utils.general import pos_tensor_to_o3d
from learning_objects.utils.general import chamfer_distance, chamfer_half_distance



def chamfer_loss(pc, pc_):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)

    output:
    loss    :
    """

    return chamfer_half_distance(pc, pc_).mean()
    # return chamfer_distance(pc, pc_).mean()

class certifiability():
    def __init__(self, epsilon, delta):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta


    def forward(self, pc, pc_):
        """
        inputs:
        pc  : input :   torch.tensor of shape (B, 3, n)
        pc_ : model :   torch.tensor of shape (B, 3, m)

        outputs:
        cert    : list of len B of boolean variables
        overlap : torch.tensor of shape (B, 1) = overlap of input pc with the model pc_
        """

        #ToDo: Write the code.



if __name__ == "__main__":

    print("test")