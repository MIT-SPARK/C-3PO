"""
This implements various visualization functions that are used in our code.

"""
import copy
import csv
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import random
import string
import time
import torch
import torch.nn.functional as F
from pytorch3d import ops
from pytorch3d import transforms

from c3po.utils.general import pos_tensor_to_o3d

def visualize_model_n_keypoints(model_list, keypoints_xyz, camera_locations=o3d.geometry.PointCloud()):
    """
    Displays one or more models and keypoints.
    :param model_list: list of o3d Geometry objects to display
    :param keypoints_xyz: list of 3d coordinates of keypoints to visualize
    :param camera_locations: optional camera location to display
    :return: list of o3d.geometry.TriangleMesh mesh objects as keypoint markers
    """
    d = 0
    for model in model_list:
        max_bound = model.get_max_bound()
        min_bound = model.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

    keypoint_radius = 0.01 * d

    keypoint_markers = []
    for xyz in keypoints_xyz:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(new_mesh)

    camera_locations.paint_uniform_color([0.1, 0.5, 0.1])
    o3d.visualization.draw_geometries(keypoint_markers + model_list + [camera_locations])

    return keypoint_markers

def visualize_torch_model_n_keypoints(cad_models, model_keypoints):
    """
    cad_models      : torch.tensor of shape (B, 3, m)
    model_keypoints : torch.tensor of shape (B, 3, N)

    """
    batch_size = model_keypoints.shape[0]

    for b in range(batch_size):

        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()

        visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

    return 0



def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (B, 3, n)
    pc2 : torch.tensor of shape (B, 3, m)
    """
    pc1 = pc1.detach()[0, ...].to('cpu')
    pc2 = pc2.detach()[0, ...].to('cpu')
    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None


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

def update_pos_tensor_to_keypoint_markers(vis, keypoints, keypoint_markers):

    keypoints = keypoints[0, ...].to('cpu')
    keypoints = keypoints.numpy().transpose()

    for i in range(len(keypoint_markers)):
        keypoint_markers[i].translate(keypoints[i], relative=False)
        vis.update_geometry(keypoint_markers[i])
        vis.poll_events()
        vis.update_renderer()
    print("FINISHED UPDATING KEYPOINT MARKERS IN CORRECTOR")
    return keypoint_markers

def display_results(input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints=None, render_text=False):
    """
    inputs:
    input_point_cloud   :   torch.tensor of shape (B, 3, m)
    detected_keypoints  :   torch.tensor of shape (B, 3, N)
    target_point_cloud  :   torch.tensor of shape (B, 3, n)
    target_keypoints    :   torch.tensor of shape (B, 3, N)

    where
    B = batch size
    N = number of keypoints
    m = number of points in the input point cloud
    n = number of points in the target point cloud
    """

    if render_text:
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)
        # displaying only the first item in the batch
    if input_point_cloud is not None:
        input_point_cloud = input_point_cloud[0, ...].to('cpu')
    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints[0, ...].to('cpu')
    if target_point_cloud is not None:
        target_point_cloud = target_point_cloud[0, ...].to('cpu')

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz in detected_keypoints:
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.01)
            kpt_mesh.translate(xyz)
            kpt_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(kpt_mesh)
        detected_keypoints = keypoint_markers

    if target_keypoints is not None:
        target_keypoints = target_keypoints[0, ...].to('cpu')
        target_keypoints = target_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz_idx in range(len(target_keypoints)):
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.01)
            kpt_mesh.translate(target_keypoints[xyz_idx])
            kpt_mesh.paint_uniform_color([1, 0.0, 0.0])
            xyz_label = target_keypoints[xyz_idx] + np.array([0.0,0.0,0.0])
            if render_text:
                scene_label = scene.add_3d_label(xyz_label, str(xyz_idx))
                # scene_label.scale = 2.0
            keypoint_markers.append(kpt_mesh)
        target_keypoints = keypoint_markers

    if target_point_cloud is not None:
        target_point_cloud = pos_tensor_to_o3d(target_point_cloud)
        target_point_cloud.paint_uniform_color([0.0, 0.0, 0.7])

    if input_point_cloud is not None:
        input_point_cloud = pos_tensor_to_o3d(input_point_cloud)
        input_point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    elements_to_viz = []
    if target_point_cloud is not None:
        elements_to_viz = elements_to_viz + [target_point_cloud]
        if render_text:
            bounds = target_point_cloud.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

    if input_point_cloud is not None:
        elements_to_viz = elements_to_viz + [input_point_cloud]
    if detected_keypoints is not None:
        elements_to_viz = elements_to_viz + detected_keypoints
    if target_keypoints is not None:
        elements_to_viz = elements_to_viz + target_keypoints

    if render_text:
        for idx, element_to_viz in enumerate(elements_to_viz):
            scene.scene.add_geometry(str(idx), element_to_viz, rendering.MaterialRecord())
        gui.Application.instance.run()  # Run until user closes window
    else:
        # draw_geometries_with_rotation(elements_to_viz)
        o3d.visualization.draw_geometries(elements_to_viz)

    return None

def temp_expt_1_viz(cad_models, model_keypoints, gt_keypoints=None, colors = None):
    batch_size = model_keypoints.shape[0]
    if gt_keypoints is None:
        gt_keypoints = model_keypoints
    print("model_keypoints.shape", model_keypoints.shape)
    print("gt_keypoints.shape", gt_keypoints.shape)
    print("cad_models.shape", cad_models.shape)


    for b in range(batch_size):
        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()
        gt_keypoints = gt_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        if colors is not None:
            point_cloud.colors = colors
        else:
            point_cloud = point_cloud.paint_uniform_color([1.0, 1.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()
        gt_keypoints = gt_keypoints.transpose(0, 1).numpy()

        # visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

        d = 0
        max_bound = point_cloud.get_max_bound()
        min_bound = point_cloud.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

        keypoint_radius = 0.01 * d

        keypoint_markers = []
        for xyz in keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in gt_keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0, 0.0])
            keypoint_markers.append(new_mesh)

        custom_draw_geometry_with_key_callback(keypoint_markers + [point_cloud])
        # o3d.visualization.draw_geometries(keypoint_markers + [point_cloud])

        return keypoint_markers

    return 0

def viz_rgb_pcd(target_object, viewpoint_camera, referenceCamera, viewpoint_angle, viz=False, dataset_path='../../data/ycb/models/ycb/'):
    pcd = o3d.io.read_point_cloud(dataset_path + target_object + \
                                  "/clouds/rgb/pc_" + viewpoint_camera + "_" \
                                  + referenceCamera + "_" + viewpoint_angle \
                                  + "_masked_rgb.ply")
    xyzrgb = np.load(dataset_path + target_object + \
                                  "/clouds/rgb/pc_" + viewpoint_camera + "_" \
                                  + referenceCamera + "_" + viewpoint_angle \
                                  + "_masked_rgb.npy")
    print(xyzrgb.shape)
    rgb = xyzrgb[0,:,3:]
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(float) / 255.0)
    print(np.asarray(pcd.points).shape)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def draw_geometries_with_rotation(elements, toggle=True):

    def rotate_view(vis, toggle=toggle):
        ctr = vis.get_view_control()
        if toggle:
            ctr.rotate(.05, 0)
        else:
            ctr.rotate(-.05, 0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(elements, rotate_view)


def custom_draw_geometry_with_key_callback(elements):
    def rotate_view_cw(vis):
        ctr = vis.get_view_control()
        ctr.rotate(5, 0)
        return False

    def rotate_view_ccw(vis):
        ctr = vis.get_view_control()
        ctr.rotate(-5, 0)
        return False


    key_to_callback = {}
    key_to_callback[ord("A")] = rotate_view_cw
    key_to_callback[ord("D")] = rotate_view_ccw

    o3d.visualization.draw_geometries_with_key_callbacks(elements, key_to_callback)


if __name__ == '__main__':

    print("Testing generate_random_keypoints:")
    B = 10
    K = 5
    N = 7
    model_keypoints = torch.rand(K, 3, N)
    y, rot, trans, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints)
    print(y.shape)
    print(rot.shape)
    print(trans.shape)
    print(shape.shape)

