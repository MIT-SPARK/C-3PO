
import open3d as o3d
import ipywidgets as widgets

import sys
sys.path.append("../../")
from c3po.datasets.shapenet import OBJECT_CATEGORIES as shapenet_objects
from c3po.datasets.ycb import MODEL_IDS as ycb_objects
from c3po.datasets.shapenet import SE3PointCloud, CLASS_ID, CLASS_MODEL_ID
from c3po.datasets.ycb import SE3PointCloudYCB
from c3po.utils.general import pos_tensor_to_o3d
from c3po.datasets.ycb import get_model_and_keypoints as get_ycb
from c3po.datasets.shapenet import get_model_and_keypoints as get_shapenet

dd_object = widgets.Dropdown(
    options=shapenet_objects + ycb_objects,
    value=shapenet_objects[0],
    description="Object"
)


def display(my_object):

    if my_object in shapenet_objects:
        # dataset_name = "shapenet.real.hard"

        dset = SE3PointCloud(class_id=CLASS_ID[my_object],
                             model_id=CLASS_MODEL_ID[my_object])
        model_mesh = dset.model_mesh
        kp = dset._get_model_keypoints()
        kp = kp[0, ...]
        pcd = dset._get_cad_models()
        pcd = pcd[0, ...]
        pcd = pos_tensor_to_o3d(pcd)

    elif my_object in ycb_objects:
        # dataset_name = "ycb.real"

        dset = SE3PointCloudYCB(model_id=my_object)
        model_mesh = dset.model_mesh
        kp = dset._get_model_keypoints()
        kp = kp[0, ...]
        pcd = dset._get_cad_models()
        pcd = pcd[0, ...]
        pcd = pos_tensor_to_o3d(pcd)

    else:
        raise ValueError("my_object not correctly specified.")

    keypoint_radius = 0.01
    kp_ = kp.transpose(0, 1).numpy()
    keypoints_xyz = kp_
    keypoint_markers = []
    for xyz in keypoints_xyz:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(new_mesh)

    pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries(keypoint_markers + [model_mesh])

    return keypoint_markers + [model_mesh]