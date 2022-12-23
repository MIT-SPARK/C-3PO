import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
# import open3d as o3d
# from open3d.web_visualizer import draw

import sys
sys.path.append("../../")
from c3po.utils.evaluation_metrics import EvalData
# from c3po.datasets.shapenet import SE3PointCloud, CLASS_ID, CLASS_MODEL_ID
# from c3po.datasets.ycb import SE3PointCloudYCB


datasets = ["shapenet.real.hard", "ycb.real"]

shapenet_objects = ["bottle", "chair", "laptop", "skateboard", "table"]
ycb_objects = ["002_master_chef_can", "006_mustard_bottle",
               "011_banana", "037_scissors", "052_extra_large_clamp"]

objects = shapenet_objects + ycb_objects

dd_model_object = widgets.Dropdown(
    options=objects,
    value=objects[0],
    description="Dataset Object"
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


def extract_data(my_files, my_labels, my_adds_th=0.05, my_adds_auc_th=0.10, output_type='table'):

    labels = []
    data = dict()

    for i, label in enumerate(my_labels):
        eval_data = EvalData()

        # print("label: ", label)
        # print("loading file: ", my_files[i])
        eval_data.load(my_files[i])
        eval_data.set_adds_th(my_adds_th)
        eval_data.set_adds_auc_th(my_adds_auc_th)

        eval_data.complete_eval_data()

        if output_type == 'table':
            # eval_data_oc = eval_data.compute_oc()
            eval_data_oc_nd = eval_data.compute_ocnd()
            labels.append(label)
            data[label] = eval_data_oc_nd.data
        elif output_type == 'plot':
            labels.append(label)
            data[label] = eval_data.data
        elif output_type == 'cert_table':
            labels.append(label)
            data[label] = eval_data.data

        # label_oc = label + " (oc)"
        # label_oc_nd = label + " (oc+nd)"

        # data[label_oc] = eval_data_oc.data

        # labels.append(label_oc)
        # labels.append(label_oc_nd)

    return data


def table(my_object, my_adds_th, my_adds_auc_th):

    #
    if my_object in shapenet_objects:
        base_folder = "../../c3po/expt_categoryless/eval/shapenet/point_transformer/post/"
        base_folder = base_folder + my_object + '/'
        data_object_list = shapenet_objects

    elif my_object in ycb_objects:
        base_folder = "../../c3po/expt_categoryless/eval/ycb/point_transformer/post/"
        base_folder = base_folder + my_object + '/'
        data_object_list = ycb_objects

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    my_baselines = []
    my_files = []

    for baseline in data_object_list:
        _filename = base_folder + baseline + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

        else:
            print(_filename)

    #
    data = extract_data(my_files, my_baselines, my_adds_th, my_adds_auc_th, output_type='table')

    #
    df = pd.DataFrame(data, index=["adds_th_score", "adds_auc"])
    df = df.transpose()
    display(100 * df)

    return None


def plot(my_object, my_metric):

    #
    if my_object in shapenet_objects:
        base_folder = "../../c3po/expt_categoryless/eval/shapenet/point_transformer/post/"
        base_folder = base_folder + my_object + '/'
        data_object_list = shapenet_objects

    elif my_object in ycb_objects:
        base_folder = "../../c3po/expt_categoryless/eval/ycb/point_transformer/post/"
        base_folder = base_folder + my_object + '/'
        data_object_list = ycb_objects

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    my_baselines = []
    my_files = []

    for baseline in data_object_list:
        _filename = base_folder + baseline + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

        else:
            print(_filename)

    #
    data = extract_data(my_files, my_baselines, output_type='plot')

    if my_metric == "adds":
        plot_adds(data)
    elif my_metric == "rerr":
        plot_rerr(data)
    elif my_metric == "terr":
        plot_terr(data)
    else:
        raise ValueError("my_metric not correctly specified.")

    return None


def table_certifiable(my_object):
    #
    if my_object in shapenet_objects:
        base_folder = "../../c3po/expt_categoryless/eval/shapenet/point_transformer/post/"
        base_folder = base_folder + my_object + '/'
        data_object_list = shapenet_objects

    elif my_object in ycb_objects:
        base_folder = "../../c3po/expt_categoryless/eval/ycb/point_transformer/post/"
        base_folder = base_folder + my_object + '/'
        data_object_list = ycb_objects

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    my_baselines = []
    my_files = []

    for baseline in data_object_list:
        _filename = base_folder + baseline + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

        else:
            print(_filename)

    #
    data = extract_data(my_files, my_baselines, output_type='cert_table')

    # oc = dict()
    # nd = dict()
    # oc_nd = dict()
    cc = dict()
    for baseline in my_baselines:
        oc = data[baseline]['oc']
        nd = data[baseline]['nd']
        oc_nd = oc * nd

        # percent_all = 100
        # percent_oc = 100 * oc.sum() / len(oc)
        if len(oc_nd) == 0:
            percent_oc_nd = 0.0
        else:
            percent_oc_nd = 100 * oc_nd.sum() / len(oc_nd)
        cc[baseline] = percent_oc_nd

    df = pd.DataFrame(cc, index=my_baselines)
    df = df.transpose()
    display(df)

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






























