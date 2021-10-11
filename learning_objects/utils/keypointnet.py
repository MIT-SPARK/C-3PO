import copy

ANNOTATIONS_FOLDER: str = '../datasets/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = '../datasets/KeypointNet/KeypointNet/pcds/'
MESH_FOLDER_NAME: str = '../datasets/KeypointNet/ShapeNetCore.v2.ply/'
OBJECT_CATEGORIES: list = ['airplane', 'bathtub', 'bed', 'bottle',
                           'cap', 'car', 'chair', 'guitar',
                           'helmet', 'knife', 'laptop', 'motorcycle',
                           'mug', 'skateboard', 'table', 'vessel']
CLASS_ID: dict = {'airplane': "02691156",
                  'bathtub': "02808440",
                  'bed': "02818832",
                  'bottle': "02876657",
                  'cap': "02954340",
                  'car': "02958343",
                  'chair': "03001627",
                  'guitar': "03467517",
                  'helmet': "03513137",
                  'knife': "03624134",
                  'laptop': "03642806",
                  'motorcycle': "03790512",
                  'mug': "03797390",
                  'skateboard': "04225987",
                  'table': "04379243",
                  'vessel': "04530566"}

CLASS_NAME: dict = {"02691156": 'airplane',
                    "02808440": 'bathtub',
                    "02818832": 'bed',
                    "02876657": 'bottle',
                    "02954340": 'cap',
                    "02958343": 'car',
                    "03001627": 'chair',
                    "03467517": 'guitar',
                    "03513137": 'helmet',
                    "03624134": 'knife',
                    "03642806": 'laptop',
                    "03790512": 'motorcycle',
                    "03797390": 'mug',
                    "04225987": 'skateboard',
                    "04379243": 'table',
                    "04530566": 'vessel'}


import csv
import torch
import os
import pandas as pd
import open3d as o3d
import json
import numpy as np
import learning_objects.utils.general as gu


def get_model_and_keypoints(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
        from the KeypointNet dataset.
        output: mesh (o3d.geometry.TriangleMesh)
                pcd (o3d.geometry.PointCloud)
                keypoints (o3d.utils.Vector3dVector(nx3))
        """
    object_pcd_file = PCD_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.pcd'
    object_mesh_file = MESH_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.ply'

    pcd = o3d.io.read_point_cloud(filename=object_pcd_file)
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals()

    annotation_file = ANNOTATIONS_FOLDER + CLASS_NAME[str(class_id)] + '.json'
    file_temp = open(str(annotation_file))
    anotation_data = json.load(file_temp)

    for idx, entry in enumerate(anotation_data):
        if entry['model_id'] == str(model_id):
            keypoints = entry['keypoints']
            break

    keypoints_xyz = []
    for aPoint in keypoints:
        keypoints_xyz.append(aPoint['xyz'])

    keypoints_xyz = np.array(keypoints_xyz)

    return mesh, pcd, keypoints_xyz


def visualize_model_n_keypoints(model_list, keypoints_xyz, camera_locations=o3d.geometry.PointCloud()):

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


def visualize_model(class_id, model_id):
    """ Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints
    from the KeypointNet dataset and plots them using open3d.visualization.draw_geometries"""

    mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id=class_id, model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)
    _ = visualize_model_n_keypoints([pcd], keypoints_xyz=keypoints_xyz)

    return mesh, pcd, keypoints_xyz, keypoint_markers



def generate_depth_data(class_id, model_id, radius_multiple = np.array([1.2, 3.0]), num_of_points=100000, location='../data/depth_images/tmp/tmp/'):
    """ Generates depth point clouds of the CAD model """

    # get model
    model_mesh, pcd, keypoints_xyz = get_model_and_keypoints(class_id, model_id)
    model_pcd = model_mesh.sample_points_uniformly(number_of_points=num_of_points)

    center = model_pcd.get_center()
    model_pcd.translate(-center)
    model_mesh.translate(-center)
    keypoints_xyz = keypoints_xyz - center

    model_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # determining size of the 3D object
    diameter = np.linalg.norm(np.asarray(model_pcd.get_max_bound()) - np.asarray(model_pcd.get_min_bound()))

    # determining camera locations and radius
    camera_distance_vector = diameter*radius_multiple
    camera_locations = gu.get_camera_locations(camera_distance_vector)
    radius = gu.get_radius(object_diameter=diameter, cam_location=np.max(camera_distance_vector))

    # visualizing 3D object and all the camera locations
    _ = visualize_model_n_keypoints([model_pcd], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)
    _ = visualize_model_n_keypoints([model_mesh], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)

    # generating radius for view sampling
    gu.sample_depth_pcd(centered_pcd=model_pcd, camera_locations=camera_locations, radius=radius, folder_name=location)

    # save keypoints_xyz at location
    np.save(file=location+'keypoints_xyz.npy', arr=keypoints_xyz)




class Dataset(torch.utils.data.Dataset):
    """ Defines the pytorch dataset of the Keypointnet"""


class DepthPointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for depth point clouds of CAD models.
    It outputs the depth point clouds stored in a given file location.
    It is to be used with dataset loader in Pytorch.
    """
    def __init__(self, dir_name, class_id, model_id, object_file='object.pcd', camera_locations_file='camera_locations.pcd', metadata_file='metadata.csv', keypoint_numpy_file='keypoints_xyz.npy'):
        # instead of object_file, have to work with class_id and model_id
        self.metadata_file = metadata_file
        self.class_id = class_id
        self.model_id = model_id
        self.dir_name = dir_name + str(self.class_id) + '/' +str(self.model_id) + '/'
        self.object_file = object_file
        self.camera_locations_file = camera_locations_file

        self.metadata = pd.read_csv(self.dir_name + self.metadata_file)
        self.object = o3d.io.read_point_cloud(self.dir_name + self.object_file)
        self.camera_locations = o3d.io.read_point_cloud(self.dir_name + self.camera_locations_file)

        self.keypoints_xyz = np.load(file=self.dir_name + str(keypoint_numpy_file))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        depth_pcd_file_name = self.metadata.iloc[idx, 0]
        depth_pcd_file = os.path.join(self.dir_name, depth_pcd_file_name)

        depth_pcd = o3d.io.read_point_cloud(depth_pcd_file)
        depth_pcd.estimate_normals()
        depth_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        return depth_pcd