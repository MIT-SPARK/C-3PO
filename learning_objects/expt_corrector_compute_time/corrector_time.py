"""
This code will compare the compute time of corrector with algo='torch' and algo='scipy'.

"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("../../")

from learning_objects.models.keypoint_corrector import kp_corrector_reg


if __name__ == "__main__":

    print("test")