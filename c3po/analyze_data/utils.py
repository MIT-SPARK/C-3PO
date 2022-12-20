
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset as Dataset


def analyze_registration_dataset(ds: Dataset, ds_name: str, transform=None) -> tuple:
    """
    ds      : dataset to be analyzed
    ds_name : name of the dataset, used for printing results

    note: the ds should output _, _, T, where T is the pose
    """

    print(f"Analyzing {ds_name}")
    # breakpoint()

    angles = []
    dist = []
    len_ = len(ds)

    for i in tqdm(range(len_), total=len_):
        if transform is None:
            _, _, T = ds[i]
        else:
            # breakpoint()
            _, _, T = transform(ds[i])
        R = T[:3, :3]
        t = T[:3, 3:]

        temp_ = 0.5 * (torch.trace(R) - 1)
        temp1 = torch.min(torch.tensor([temp_, 0.999]))
        temp2 = torch.max(torch.tensor([temp1, -0.999]))
        angles.append(torch.acos(temp2).item())
        dist.append(torch.norm(t).item())

    return torch.tensor(angles), torch.tensor(dist)


def plot_cdf(data, label, filename):
    """
    datapoints: torch.tensor of shape (N,)
    max_val : float

    Returns:
        plots cdf up-to maximum value of max_val
    """

    plot_data = dict()
    plot_data[f"{label}"] = data
    plot_data_ = pd.DataFrame.from_dict(plot_data)

    # sns.set(stype="darkgrid")
    sns_plot = sns.kdeplot(plot_data_, bw_adjust=0.04, cumulative=True, common_norm=False)
    # plt.show()

    fig = sns_plot.get_figure()
    fig.savefig(f"{filename}.png")
    plt.close(fig)

    return None