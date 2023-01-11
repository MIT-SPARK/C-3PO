
import copy
import pickle
import argparse
import open3d as o3d

import sys

import torch

sys.path.append("../..")

from c3po.datasets.shapenet import CLASS_ID, CLASS_MODEL_ID, get_model_and_keypoints
from c3po.datasets.visualization import VISPOSE
from c3po.utils.general import pos_tensor_to_o3d

iterations = {
    'cap': [1, 5, 10, 100],
    'table': [1, 2, 7, 10],
    'chair': [1, 4, 10, 100, 300]
}


def get_o3dpoc(pc, color="grey"):

    if color == "grey":
        c_ = [0.5, 0.5, 0.5]    # grey
    elif color == "blue":
        c_ = [0.0, 0.0, 1.0]    # blue
    elif color == "red":
        c_ = [1.0, 0.0, 0.0]    # red
    elif color == "green":
        c_ = [0.0, 0.8, 0.0]    # green
    else:
        c_ = [0.0, 0.0, 0.0]    # black (default)

    pc = pos_tensor_to_o3d(pos=pc)
    pc = pc.paint_uniform_color(c_)
    pc.estimate_normals()

    return pc


def get_o3dkp(kp, color="red"):

    if color == "grey":
        c_ = [0.5, 0.5, 0.5]    # grey
    elif color == "blue":
        c_ = [0.0, 0.0, 1.0]    # blue
    elif color == "red":
        c_ = [1.0, 0.0, 0.0]    # red
    elif color == "green":
        c_ = [0.0, 0.8, 0.0]    # green
    else:
        c_ = [0.0, 0.0, 0.0]    # black (default)

    keypoint_markers = []
    keypoints = kp.transpose(0, 1).numpy()
    for xyz in keypoints:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color(c_)
        keypoint_markers.append(new_mesh)

    return keypoint_markers


def degeneracy_plot():

    file_ = "../../c3po/expt_shapenet/eval/vis_" + "cap" + '.pkl'

    with open(file_, 'rb') as fp:
        data = pickle.load(fp)

    # input pc
    input_pc = torch.from_numpy(data['input_pc']).squeeze(0)
    class_name = data['class_name']
    model_keypoints = torch.from_numpy(data['model_keypoints']).squeeze(0)
    cad_model = torch.from_numpy(data['cad_model']).squeeze(0)

    # extracting ground-truth
    R_target, t_target = VISPOSE["cap"]
    R_target = R_target[0, ...]
    t_target = t_target[0, ...]

    # gt
    gt_pc = R_target @ cad_model + t_target
    gt_kp = R_target @ model_keypoints + t_target

    # est
    iter = 900
    R = torch.from_numpy(data[iter]['R']).squeeze(0)
    t = torch.from_numpy(data[iter]['t']).squeeze(0)
    oc = torch.from_numpy(data[iter]['oc']).squeeze(0)
    nd = torch.from_numpy(data[iter]['nd']).squeeze(0)
    corrected_kp = torch.from_numpy(data[iter]['corrected_kp']).squeeze(0)

    pred_pc = R @ cad_model + t
    pred_kp = R @ model_keypoints + t

    # convert to o3d
    input_pc = get_o3dpoc(input_pc, color="grey")
    pred_pc = get_o3dpoc(pred_pc, color="blue")
    gt_pc = get_o3dpoc(gt_pc, color="black")
    pred_kp = get_o3dkp(pred_kp, color="green")
    gt_kp = get_o3dkp(gt_kp, color="red")

    # visualize: input
    o3d.visualization.draw_geometries([input_pc] + gt_kp)
    o3d.visualization.draw_geometries([input_pc] + [pred_pc] + pred_kp + gt_kp)
    o3d.visualization.draw_geometries([pred_pc] + [gt_pc] + pred_kp + gt_kp)

    # visualize: corrected keypoints

    print("iteration: ", iter)
    print("obs. correct: ", oc.item())
    print("non-degenerate: ", nd.item())
    # breakpoint()


if __name__ == "__main__":
    """
    usage:
    >> python plot_degenerate.py
    """

    degeneracy_plot()