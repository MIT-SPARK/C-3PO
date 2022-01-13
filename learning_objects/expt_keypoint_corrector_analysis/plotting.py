

import torch
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

plt.style.use('seaborn-whitegrid')

import sys
sys.path.append("../../")


def scatter_bar_plot(plt, x, y, label, color='orangered'):
    """
    x   : torch.tensor of shape (n)
    y   : torch.tensor of shape (n, k)

    """
    n, k = y.shape
    width = 0.2*torch.abs(x[1]-x[0])

    x_points = x.unsqueeze(-1).repeat(1, k)
    x_points += width*(torch.rand(size=x_points.shape)-1)
    y_points = y

    plt.scatter(x_points, y_points, s=20.0, c=color, alpha=0.5, label=label)

    return plt



if __name__ == "__main__":

    x = torch.arange(0, 10, 0.1)
    len = x.shape[0]

    f1 = x.unsqueeze(-1) + torch.randn(10, 100)
    f1x = 0.8*x.unsqueeze(-1) + 0.3*torch.randn(10, 100)

    f2 = (x ** 2).unsqueeze(-1) + torch.randn(10, 100)
    f2x = (x ** 2).unsqueeze(-1) + 0.8*torch.randn(10, 100) - 0.5









