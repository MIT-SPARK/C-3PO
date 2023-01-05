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
    'cap': [1, 2, 5, 100],
    'table': [1, 2, 7, 10],
    'chair': [1, 2, 10, 100, 300]
}


def plot(args):

    file_ = "../../c3po/expt_shapenet/eval/vis_" + str(args.object) + '.pkl'

    with open(file_, 'rb') as fp:
        data = pickle.load(fp)

    # extracting input pc
    input_pc = torch.from_numpy(data['input_pc']).squeeze(0)
    class_name = data['class_name']
    model_keypoints = torch.from_numpy(data['model_keypoints']).squeeze(0)
    cad_model = torch.from_numpy(data['cad_model']).squeeze(0)

    # visualize: input point cloud
    point_cloud = pos_tensor_to_o3d(pos=input_pc)
    point_cloud = point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    point_cloud.estimate_normals()

    # extracting CAD model:
    mesh_model, _, _ = get_model_and_keypoints(CLASS_ID[class_name],
                                               CLASS_MODEL_ID[class_name])

    # extracting ground-truth
    R_target, t_target = VISPOSE[args.object]
    R_target = R_target[0, ...]
    t_target = t_target[0, ...]

    for iter in iterations[args.object]:
        R = torch.from_numpy(data[iter]['R']).squeeze(0)
        t = torch.from_numpy(data[iter]['t']).squeeze(0)
        oc = torch.from_numpy(data[iter]['oc']).squeeze(0)
        nd = torch.from_numpy(data[iter]['nd']).squeeze(0)
        corrected_kp = torch.from_numpy(data[iter]['corrected_kp']).squeeze(0)

        # breakpoint()
        # visualize: mesh model (posed)
        mesh_ = copy.deepcopy(mesh_model)
        mesh_ = mesh_.rotate(R.numpy())
        mesh_ = mesh_.translate(t.numpy(), relative=False)

        # visualize: corrected keypoints
        keypoint_markers = []
        keypoints = corrected_kp.transpose(0, 1).numpy()
        for xyz in keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
            keypoint_markers.append(new_mesh)

        o3d.visualization.draw_geometries([point_cloud] + [mesh_] + keypoint_markers)

        print("iteration: ", iter)
        print("obs. correct: ", oc.item())
        print("non-degenerate: ", nd.item())
        # breakpoint()
        # visualize input_pc, corrected_kp, R, t posed CAD model

    # posed CAD model: gt and final iteration
    mesh_gt = copy.deepcopy(mesh_model)
    mesh_gt = mesh_gt.rotate(R_target.numpy())
    mesh_gt = mesh_gt.translate(t_target.numpy())

    o3d.visualization.draw_geometries([mesh_, mesh_gt])


if __name__ == "__main__":
    """
    usage:
    >> python plot.py \
    --object cap
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=["cap", "table", "chair"], type=str)

    args = parser.parse_args()

    plot(args)