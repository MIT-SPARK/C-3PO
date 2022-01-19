"""
This implements various generic classes and functions that are used in our code.

"""

import copy
import csv
import os
import random
import string
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from pytorch3d import ops
from pytorch3d import transforms
from torch_geometric.data import Data
from time import time



def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (3, n)
    pc2 : torch.tensor of shape (3, m)
    """
    pc1 = pc1.detach().to('cpu')
    pc2 = pc2.detach().to('cpu')

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



class Timer:
    def __init__(self, tag, print=False):
        self.tag = tag
        self.ts = None
        self.print = print

    def tic(self):
        self.ts = time()

    def toc(self):
        if self.print:
            print("{}: {}s".format(self.tag, time() - self.ts))
        return time()


def chamfer_distance(X, Y):
    """
    inputs:
    X: torch.tensor of shape (B, 3, n)
    Y: torch.tensor of shape (B, 3, m)

    where
    B = batch size
    n, m = number of points in the point cloud

    outputs:
    loss: torch.tensor of shape (B, 1)
    """

    sq_dist_xy, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist_yx, _, _ = ops.knn_points(torch.transpose(Y, -1, -2), torch.transpose(X, -1, -2), K=1)
    # dist (B, n, 1): distance from point in Y to the nearest point in X

    return sq_dist_xy.mean(dim=1) + sq_dist_yx.mean(dim=1)


def chamfer_half_distance(X, Y):
    """
    inputs:
    X: torch.tensor of shape (B, 3, n)
    Y: torch.tensor of shape (B, 3, m)

    where
    B = batch size
    n, m = number of points in the point cloud

    outputs:
    loss: torch.tensor of shape (B, 1)

    Note:
    Output is the mean distance from every point in X to its closest point in Y
    """

    sq_dist, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    return sq_dist.mean(dim=1)


def max_chamfer_distance(X, Y):
    """
    inputs:
    X: torch.tensor of shape (B, 3, n)
    Y: torch.tensor of shape (B, 3, m)

    where
    B = batch size
    n, m = number of points in the point cloud

    outputs:
    loss: torch.tensor of shape (B, 1)
    """

    sq_dist_xy, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist_yx, _, _ = ops.knn_points(torch.transpose(Y, -1, -2), torch.transpose(X, -1, -2), K=1)
    # dist (B, n, 1): distance from point in Y to the nearest point in X

    return sq_dist_xy.max(dim=1)[0] + sq_dist_yx.max(dim=1)[0]


def max_chamfer_half_distance(X, Y):
    """
    inputs:
    X: torch.tensor of shape (B, 3, n)
    Y: torch.tensor of shape (B, 3, m)

    where
    B = batch size
    n, m = number of points in the point cloud

    outputs:
    loss: torch.tensor of shape (B, 1)

    Note:
    Output is the mean distance from every point in X to its closest point in Y
    """

    sq_dist, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    return sq_dist.max(dim=1)[0]


def soft_chamfer_half_distance(X, Y, radius, K=10, theta=10.0):
    """
    inputs:
    X: torch.tensor of shape (B, 3, n)
    Y: torch.tensor of shape (B, 3, m)
    radius: float
    theta: float

    where
    B = batch size
    n, m = number of points in the point cloud

    outputs:
    loss: torch.tensor of shape (B, 1)
    """

    dist, idx, _ = ops.ball_query(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), radius=radius, K=K)
    # dist (B, n, K): distance from point in X to the nearest point in Y
    # idx (B, n, K): indices of the K closest points in Y, for every point in X
    prob = F.softmax(-theta*dist, dim=-1)
    # prob is of shape (B, n, K)


    return ((dist**2)*prob).sum(-1).unsqueeze(-1).mean(dim=1)


def generate_random_keypoints(batch_size, model_keypoints):
    """
    input:
    batch_size      : int
    model_keypoints : torch.tensor of shape (K, 3, N)

    where
    K = number of cad models in a shape category
    N = number of keypoints

    output:
    keypoints   : torch.tensor of shape (batch_size, 3, N)
    rotation    : torch.tensor of shape (batch_size, 3, 3)
    translation : torch.tensor of shape (batch_size, 3, 1)
    shape       : torch.tensor of shape (batch_size, K, 1)
    """

    K = model_keypoints.shape[0]
    N = model_keypoints.shape[-1]

    rotation = transforms.random_rotations(n=batch_size)
    translation = torch.rand(batch_size, 3, 1)

    shape = torch.rand(batch_size, K, 1)
    shape = shape/shape.sum(1).unsqueeze(1) # (batch_size, K, 1)
    shape = shape.unsqueeze(-1) # (batch_size, K, 1, 1)
    keypoints = torch.einsum('bkij,ukdn->bdn', shape, model_keypoints.unsqueeze(0))
    keypoints = rotation @ keypoints + translation

    return keypoints, rotation, translation, shape.squeeze(-1)


def keypoint_error(kp, kp_):
    """
    inputs:
    kp: torch.tensor of shape (3, N) or (B, 3, N)
    kp_: torch.tensor of shape (3, N) or (B, 3, N)

    output:
    kp_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if kp.dim() == 2:
        return torch.norm(kp - kp_, p=2, dim=0).mean()/3.0
    elif kp.dim() == 3:
        return torch.norm(kp - kp_, p=2, dim=1).mean(1).unsqueeze(-1)/3.0
    else:
        return ValueError


def shape_error(c, c_):
    """
    inputs:
    c: torch.tensor of shape (K, 1) or (B, K, 1)
    c_: torch.tensor of shape (K, 1) or (B, K, 1)

    output:
    c_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if c.dim() == 2:
        return torch.norm(c - c_, p=2)/c.shape[0]
    elif c.dim() == 3:
        return torch.norm(c - c_, p=2, dim=1)/c.shape[1]
    else:
        return ValueError


def translation_error(t, t_):
    """
    inputs:
    t: torch.tensor of shape (3, 1) or (B, 3, 1)
    t_: torch.tensor of shape (3, 1) or (B, 3, 1)

    output:
    t_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if t.dim() == 2:
        return torch.norm(t - t_, p=2)/3.0
    elif t.dim() == 3:
        return torch.norm(t-t_, p=2, dim=1)/3.0
    else:
        return ValueError


def rotation_error(R, R_):
    """
    inputs:
    R: torch.tensor of shape (3, 3) or (B, 3, 3)
    R_: torch.tensor of shape (3, 3) or (B, 3, 3)

    output:
    R_err: torch.tensor of shape (1, 1) or (B, 1)
    """

    #ToDo: Get the Rotation angle along the axis?

    if R.dim() == 2:
        return transforms.matrix_to_euler_angles(torch.matmul(R.T, R_), "XYZ").abs().sum()/3.0
    elif R.dim() == 3:
        return transforms.matrix_to_euler_angles(torch.transpose(R, 1, 2) @ R_, "XYZ").abs().mean(1).unsqueeze(1)
    else:
        return ValueError


def check_rot_mat(R, tol=0.001):
    """
    This checks if the matrix R is a 3x3 rotation matrix
    R: rotation matrix: numpy.ndarray of size (3, 3)

    Output:
    True/False: determining if R is a rotation matrix/ or not
    """
    tol = tol
    if torch.norm(torch.matmul(R.T, R) - np.eye(3), ord='fro') < tol:
        return True
    else:
        return False



## These were written for shapenet and keypointnet datasets.

def tensor_to_o3d(normals, pos):
    """
    :param pos: position of points torch.Tensor Nx3
    :param normals: surface normals torch.Tensor Nx3
    :return: open3d PointCloud
    """
    pos_o3d = o3d.utility.Vector3dVector(pos.numpy())
    normals_o3d = o3d.utility.Vector3dVector(normals.numpy())

    object = o3d.geometry.PointCloud()
    object.points = pos_o3d
    object.normals = normals_o3d

    return object


def pos_tensor_to_o3d(pos):
    """
    inputs:
    pos: torch.tensor of shape (3, N)

    output:
    open3d PointCloud
    """
    pos_o3d = o3d.utility.Vector3dVector(pos.transpose(0, 1).to('cpu').numpy())

    object = o3d.geometry.PointCloud()
    object.points = pos_o3d
    object.estimate_normals()

    return object


def display_results(input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints):
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

    # displaying only the first item in the batch
    input_point_cloud = input_point_cloud[0, ...].to('cpu')
    detected_keypoints = detected_keypoints[0, ...].to('cpu')
    target_point_cloud = target_point_cloud[0, ...].to('cpu')
    target_keypoints = target_keypoints[0, ...].to('cpu')

    input_point_cloud = pos_tensor_to_o3d(input_point_cloud)
    detected_keypoints = pos_tensor_to_o3d(detected_keypoints)
    target_point_cloud = pos_tensor_to_o3d(target_point_cloud)
    target_keypoints = pos_tensor_to_o3d(target_keypoints)

    input_point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    detected_keypoints.paint_uniform_color([0.8, 0.0, 0.0])
    target_keypoints.paint_uniform_color([0.0, 0.8, 0.0])
    target_point_cloud.paint_uniform_color([0.0, 0.0, 0.7])

    o3d.visualization.draw_geometries([input_point_cloud, detected_keypoints, target_keypoints, target_point_cloud])

    return None


def generate_filename(chars=string.ascii_uppercase + string.digits, N=10):
    """function generates random strings of length N"""
    return ''.join(random.choice(chars) for _ in range(N))


def get_pose(R, t):
    """outputs 4x4 pose matrix given 3x3 rotation R and translation t"""
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3:] = t.reshape(3, 1)
    return P


def get_extrinsic(view_dir, up, location):
    """
    Generates the extrinsic matrix (for renderer in Open3D
    [open3d.visualization.rendering]) given:
    view_dir (3x1 numpy array)  = direction of the view of the camera
    up (3x1 numpy array)        = up direction
    location (3x1 numpy array)  = location of the camera

    All these must be specified in the global coordinate frame
    """
    x = np.cross(-up, view_dir)
    y = np.cross(view_dir, x)
    z = view_dir

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    R = np.array([x, y, z])
    t = np.zeros((3, 1))
    t[:, 0] = -np.dot(R, location)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:] = t

    return extrinsic


def get_camera_locations(camera_distance_vector, color=np.array([1.0, 0.0, 0.0]), number_of_locations=200):
    """ generating sphere of points around the object """
    camera_locations = o3d.geometry.PointCloud()
    for i in range(len(camera_distance_vector)):
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=camera_distance_vector[i], resolution=20)
        sphere_points_new = sphere_mesh.sample_points_uniformly(number_of_points=number_of_locations)

        tempPoints = np.concatenate((sphere_points_new.points, camera_locations.points), axis=0)
        camera_locations.points = o3d.utility.Vector3dVector(tempPoints)

    camera_locations.paint_uniform_color(color)

    return camera_locations


def get_radius(object_diameter, cam_location):
    """ returns radius, which is the maximum distance from cam_location within which all points in the object lie"""
    return 100*np.sqrt(object_diameter**2 + np.linalg.norm(cam_location)**2)


def get_depth_pcd(centered_pcd, camera, radius, method='1'):
    """ This produces a depth point cloud. Input:
    centered_pcd (o3d.geometry.PointCloud object) = pcd that is centered at (0, 0, 0)
    camera (numpy.ndarray[float64[3, 1]])         = location of camera in the 3d space
    radius (float)                                = radius from camera location, beyond which points are not taken
    """
    pcd = copy.deepcopy(centered_pcd)

    """Method 1"""
    if method == '1':
        _, pt_map = pcd.hidden_point_removal(camera_location=camera, radius=radius)
        pcd = pcd.select_by_index(pt_map)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd

    """Method 2"""
    # Do not use Method 2. It constructs an artificial mesh from sampled/visible points.
    # This leads it to connect points that belong to distinct objects, thereby changing things.
    if method == '2':
        visible_mesh, _ = pcd.hidden_point_removal(camera_location=camera, radius=radius)
        pcd_visible = visible_mesh.sample_points_uniformly(number_of_points=10000)
        pcd_visible.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd_visible


def create_test_object(visualize=False):
    """ This function outputs a test 3D object that is placed at origin """

    cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    mesh = cyl + sphere + box + solid
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    if visualize == True:
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    return mesh, pcd, coordinate_frame


def save_metadata(file_name_with_location, dict_data, dict_data_columns):
    """ This functions stores dict_data as a csv file.
    Columns are given by dict_data_columns.
    """
    csv_file = file_name_with_location
    try:
        with open(csv_file, 'w') as csv_fileTemp:
            writer = csv.DictWriter(csv_fileTemp, fieldnames=dict_data_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def sample_depth_pcd(centered_pcd, camera_locations, radius, folder_name):
    """ This function computes depth pcd for each given camera location.
    The output pcd are stored as obj files in the folder specified by folder_name.
    The function also stores the centered_pcd and camera locations in the same folder.
    The function also creates a csv file listing file name and camera locations.
     """

    # saving object point cloud data
    centered_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud(folder_name + 'object.pcd', centered_pcd)

    # saving camera locations point cloud
    camera_locations.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.io.write_point_cloud(folder_name + 'camera_locations.pcd', camera_locations)

    # generating and saving depth point clouds
    dict_data_columns = ['file_name', 'camera_x', 'camera_y', 'camera_z']
    dict_data = []

    for camera in camera_locations.points:
        # generate depth point cloud
        depth_pcd = get_depth_pcd(centered_pcd, camera, radius)
        depth_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # save depth_pcd as a .pcd file in folder folder_name
        file_name = generate_filename() + '.pcd'
        o3d.io.write_point_cloud(folder_name + file_name, depth_pcd)

        # add file_name and camera coordinates to dict_data
        dict_row = dict()
        dict_row['file_name'] = file_name
        dict_row['camera_x'] = camera[0]
        dict_row['camera_y'] = camera[1]
        dict_row['camera_z'] = camera[2]
        dict_data.append(dict_row)

    save_metadata(folder_name + 'metadata.csv', dict_data, dict_data_columns)



""" The following are test codes. Not to be used in the final project files. """


def test_get_depth_pcd():
    """ THis code tests depth point cloud from 3D objects
    input: 3d object, 3d object location, and camera extrinsic
    output: depth point cloud of the object (open3d.geometry.PointCloud)
    """

    cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    mesh = cyl + sphere + box + solid
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    centered_pcd = mesh.sample_points_uniformly(number_of_points=10000)
    diameter = np.linalg.norm(np.asarray(centered_pcd.get_max_bound()) - np.asarray(centered_pcd.get_min_bound()))
    o3d.visualization.draw_geometries([centered_pcd, coordinate_frame])

    camera = [0, 50, 1.5]
    radius = 20000.0
    pcd = get_depth_pcd(centered_pcd, camera, radius)
    o3d.visualization.draw_geometries([pcd, coordinate_frame])


def test_rendering_depth_images(save_to_folder='../data/tmp/'):
    """ This code tests rendering depth images from 3D Objects """
    render = rendering.OffscreenRenderer(640, 480)

    yellow = rendering.Material()
    yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    yellow.shader = "defaultLit"

    green = rendering.Material()
    green.base_color = [0.0, 0.5, 0.0, 1.0]
    green.shader = "defaultLit"

    grey = rendering.Material()
    grey.base_color = [0.7, 0.7, 0.7, 1.0]
    grey.shader = "defaultLit"

    white = rendering.Material()
    white.base_color = [1.0, 1.0, 1.0, 1.0]
    white.shader = "defaultLit"

    cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    # Adding to renderer
    render.scene.add_geometry("cyl", cyl, green)
    render.scene.add_geometry("sphere", sphere, yellow)
    render.scene.add_geometry("box", box, grey)
    render.scene.add_geometry("solid", solid, white)
    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                     75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    img = render.render_to_image()
    o3d.io.write_image(save_to_folder+"test.png", img, 9)

    # camera setup using (intrinsic, extrinsic)
    # intrinsic = ()
    # extrinsic = 4x4 matrix pose
    view_dir = np.array([1, 0, 0])
    cam_location = np.array([-5, 0, 0])
    up = np.array([0, 0, 1])
    extrinsic = get_extrinsic(view_dir=view_dir, up=up, location=cam_location)
    # print(extrinsic)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    # intrinsic = o3d.camera.PinholeCameraIntrinsic()
    # intrinsic.set_intrinsics(width=2500, height=2000, fx=1000, fy=1000, cx=500, cy=500)
    render.setup_camera(intrinsic, extrinsic)

    img = render.render_to_image()
    depth = render.render_to_depth_image()
    o3d.io.write_image(save_to_folder+"test2.png", img)
    # plt.imshow(depth)
    # plt.show()
    # o3d.io.write_image("tmp/test2_depth.png", depth, 9)
    plt.imsave(save_to_folder+"test2_depth.png", depth)

    # generate and save point cloud from the depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic=intrinsic, depth_scale=0.01, depth_trunc=0.9,
                                                          stride=10)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud(save_to_folder+"test2_depth.pcd", pcd)
    o3d.visualization.draw_geometries([pcd])

    # removing hidden points
    # render.scene.hidden_point_removal()



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

