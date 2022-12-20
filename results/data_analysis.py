
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from c3po.datasets.shapenet import ShapeNet
from c3po.analyze_data.shapenet import analyze_registration_dataset


class dataWrapper:
    def __init__(self):
        self.none = 1.0

    def __call__(self, x):
        pc1, pc2, kp1, kp2, R, t = x

        T = torch.eye(4).to(device=R.device)
        T[:3, :3] = R
        T[:3, 3:] = t

        return pc1, pc2, T


def analyze():

    dset_easy = ShapeNet(type='sim',
                         object='airplane',
                         length=1000,
                         num_points=1000,
                         adv_option='easy')

    dset_hard = ShapeNet(type='sim',
                         object='airplane',
                         length=1000,
                         num_points=1000,
                         adv_option='hard')

    rerr_easy, terr_easy = analyze_registration_dataset(dset_easy, "easy",
                                                        transform=dataWrapper())

    rerr_hard, terr_hard = analyze_registration_dataset(dset_hard, "hard",
                                                        transform=dataWrapper())

    rerr = dict()
    rerr['easy'] = rerr_easy
    rerr['hard'] = rerr_hard

    terr = dict()
    terr['easy'] = terr_easy
    terr['hard'] = terr_hard

    return rerr, terr



