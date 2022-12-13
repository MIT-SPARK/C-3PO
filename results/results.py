import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets

import sys
sys.path.append("../")
from c3po.datasets.shapenet import OBJECT_CATEGORIES as shapenet_objects
from c3po.datasets.ycb import MODEL_IDS as ycb_objects
from c3po.utils.evaluation_metrics import EvalData

# datasets = ["shapenet.real.hard", "ycb.real"]
shapenet_datasets =["shapenet",
                    "shapenet.sim.easy", "shapenet.sim.hard",
                    "shapenet.real.easy", "shapenet.real.hard"]
ycb_datasets = ["ycb", "ycb.sim", "ycb.real"]

# datasets = ["shapenet", "ycb"]
datasets = shapenet_datasets + ycb_datasets

# baselines = ["deepgmr",
#              "equipose",
#              "fpfh",
#              "pointnetlk"]

baselines = ["KeyPoSim",
             "KeyPoSimICP",
             "KeyPoSimRANSACICP",
             "KeyPoSimCor",
             "KeyPoSimCorICP",
             "KeyPoSimCorRANSACICP",
             "c3po",
             "KeyPoReal"]

detector_types = ["point_transformer", "pointnet"]

# baseline_folders = ["deepgmr/eval/",
#                     "EquiPose/equi-pose/runs",
#                     "fpfh_teaser/runs",
#                     "PointNetLK_Revisited/runs"]


dd_dataset = widgets.Dropdown(
    # options=["shapenet", "ycb"],
    options=datasets,
    # value="shapenet",
    value=datasets[0],
    description="Dataset"
)


dd_object = widgets.Dropdown(
    options=shapenet_objects + ycb_objects,
    value=shapenet_objects[0],
    description="Object"
)


dd_detector = widgets.Dropdown(
    options=detector_types,
    value=detector_types[0],
    description="Detector"
)

dd_metric = widgets.Dropdown(
    options=["adds", "rerr", "terr"],
    value="adds",
    description="Metric"
)

slide_adds_th = widgets.FloatSlider(
    min=0.01,
    max=0.20,
    step=0.005,
    value=0.05,
    description="ADD-S Threshold"
)

slide_adds_auc_th = widgets.FloatSlider(
    min=0.01,
    max=0.20,
    step=0.005,
    value=0.10,
    description="ADD-S AUC Threshold"
)


def extract_data(my_files, my_labels, my_adds_th=0.02, my_adds_auc_th=0.05):

    labels = []
    data = dict()

    for i, label in enumerate(my_labels):
        eval_data = EvalData()

        # print("label: ", label)
        # print("loading file: ", my_files[i])
        eval_data.load(my_files[i])
        eval_data.set_adds_th(my_adds_th)
        eval_data.set_adds_auc_th(my_adds_auc_th)

        #     print(eval_data.data["adds"])
        eval_data.complete_eval_data()
        data[label] = eval_data.data
        labels.append(label)

        if label == "c3po":

            eval_data_oc = eval_data.compute_oc()
            eval_data_oc_nd = eval_data.compute_ocnd()
            label_oc = label + " (oc)"
            label_oc_nd = label + " (oc+nd)"

            data[label_oc] = eval_data_oc.data
            data[label_oc_nd] = eval_data_oc_nd.data

            labels.append(label_oc)
            labels.append(label_oc_nd)

    return data


def table(my_dataset, my_object, my_detector, my_adds_th, my_adds_auc_th):

    #
    if "shapenet" in my_dataset:
    # if my_dataset == "shapenet":
        base_folder = "../c3po/expt_shapenet"

        if my_object not in shapenet_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    elif "ycb" in my_dataset:
    # elif my_dataset == "ycb":
        base_folder = "../c3po/expt_ycb"

        if my_object not in ycb_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

        if my_detector != "point_transformer":
            print("Error: We only trained Point Transformer on YCB, as PointNet showed "
                  "suboptimal performance on ShapeNet.")
            return None

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    my_baselines = []
    my_files = []

    for baseline in baselines:
        _filename = base_folder + '/eval/' + baseline + '/' + my_detector + '/' + my_dataset + '/' \
                    + my_object + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

        else:
            print(_filename)

    #
    data = extract_data(my_files, my_baselines, my_adds_th, my_adds_auc_th)

    #
    df = pd.DataFrame(data, index=["adds_th_score", "adds_auc"])
    df = df.transpose()
    display(100 * df)

    return None


def plot(my_dataset, my_object, my_detector, my_metric):

    #
    if "shapenet" in my_dataset:
    # if my_dataset == "shapenet":
        base_folder = "../c3po/expt_shapenet"

        if my_object not in shapenet_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    elif "ycb" in my_dataset:
    # elif my_dataset == "ycb":
        base_folder = "../c3po/expt_ycb"

        if my_object not in ycb_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

        if my_detector != "point_transformer":
            print("Error: We only trained Point Transformer on YCB, as PointNet showed "
                  "suboptimal performance on ShapeNet.")
            return None

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    my_baselines = []
    my_files = []

    for baseline in baselines:
        _filename = base_folder + '/eval/' + baseline + '/' + my_detector + '/' + my_dataset + '/' \
                    + my_object + '/eval_data.pkl'
        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

    #
    data = extract_data(my_files, my_baselines)

    if my_metric == "adds":
        plot_adds(data)
    elif my_metric == "rerr":
        plot_rerr(data)
    elif my_metric == "terr":
        plot_terr(data)
    else:
        raise ValueError("my_metric not correctly specified.")

    return None


def plot_adds(data):

    sns.set(style="darkgrid")
    adds_data = dict()
    for key in data.keys():
        df_ = pd.DataFrame(dict({key: data[key]["adds"]}))
        adds_data[key] = df_

    conca = pd.concat([adds_data[key].assign(dataset=key) for key in adds_data.keys()])

    sns.kdeplot(conca, bw_adjust=0.1, cumulative=True, common_norm=False)
    plt.xlabel('ADD-S')
    return None


def plot_rerr(data):

    sns.set(style="darkgrid")
    rerr_data = dict()
    for key in data.keys():
        df_ = pd.DataFrame(dict({key: data[key]["rerr"]}))
        rerr_data[key] = df_

    conca = pd.concat([rerr_data[key].assign(dataset=key) for key in rerr_data.keys()])

    sns.kdeplot(conca, bw_adjust=0.1, cumulative=True, common_norm=False)
    plt.xlabel('Rotation Error (axis-angle, in rad)')

    return None


def plot_terr(data):

    sns.set(style="darkgrid")
    terr_data = dict()
    for key in data.keys():
        df_ = pd.DataFrame(dict({key: data[key]["terr"]}))
        terr_data[key] = df_

    conca = pd.concat([terr_data[key].assign(dataset=key) for key in terr_data.keys()])

    sns.kdeplot(conca, bw_adjust=0.1, cumulative=True, common_norm=False)
    plt.xlabel('Translation Error')

    return None


if __name__ == "__main__":

    my_dataset = "shapenet"
    my_object = "chair"
    my_detector = "point_transformer"

    table(my_dataset, my_object, my_detector)






























