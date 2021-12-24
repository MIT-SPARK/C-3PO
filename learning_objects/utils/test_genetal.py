import torch
import torch.nn as nn
import os
import sys
sys.path.append("../../")

from learning_objects.utils.general import chamfer_half_distance, soft_chamfer_half_distance, generate_random_keypoints, rotation_error


#ToDo: This is a temporary file. This all the tests here should be moved to learninig_objects.utils.general.py

if __name__ == "__main__":

    B = 10
    K = 5
    N = 7
    model_keypoints = torch.rand(K, 3, N)
    y, rot, trans, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints)
    print(y.shape)
    print(rot.shape)
    print(trans.shape)
    print(shape.shape)

