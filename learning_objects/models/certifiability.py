"""
This code defines a metric of (epsilon, delta)-certifiability. Given two registered, shape aligned point clouds
pc and pc_ it determines if the registration + shape alignment is certifiable or not.

"""

import torch




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