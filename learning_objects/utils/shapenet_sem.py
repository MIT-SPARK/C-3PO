
# Global Variables:
METADATA_FILE: str = "../../../datasets/ShapeNetSem.v0/metadata.csv"
SYNSET_FILE: str = "../../../datasets/ShapeNetSem.v0/categories.synset.csv"
DIR_NAME: str = "../../../datasets/ShapeNetSem.v0/models-OBJ/models/"
NUM_POINTS_PER_CAD_MODEL: int = 100000
#


"""CHANGES TO BE MADE: RENAMING: 
functions:
process --> process
get_model ----------> get_model
visualize_model ---------> visualize_model
get_sample ---> get_sample
generate_depth_data ----> generate_depth_data

classes:
Dataset3dModels ----------> Dataset3dModels
DepthPointCloud ------> DepthPointCloud


folder name:
dataset_utils --------> shapenet_sem (short would be sn_sem when importing)
"""


import csv
import torch
import os
import pandas as pd
import open3d as o3d
import numpy as np
import learning_objects.utils.general_utils as gu


def process(metadata_file=METADATA_FILE, synset_file=SYNSET_FILE):

    """
    This function takes in the csv files of the ShapeNetSem.v0 and outputs
    category_list - list of object category with 3D models in ShapeNetSem.v0
    category2synset - dictionary mapping object category to synset
    category2objFilename - dictionary mapping object category to a list of all CAD model obj file names
    """

    # Constructing objFilename2category dictionary
    metadata = open(metadata_file, 'r')
    fileTemp = csv.DictReader(metadata)

    objFilename = []
    synset = []
    category = []

    for col in fileTemp:

        # extracting the object filename
        tempName = col['fullId'][4:] + '.obj'
        objFilename.append(tempName)

        # extracting synset
        synset.append(col['wnsynset'])

        # extracting object categories
        categories = col['category']
        categories = categories.split(',')

        if len(categories) == 1 and len(categories[0]) == 0:
            # relabels empty categories from [''] to []
            categories = []

        category.append(categories)

    objFilename2category = {objFilename[i]: category[i] for i in range(len(objFilename))}


    # Get the list of all categories
    categories_synset = open(synset_file, 'r')
    fileTemp = csv.DictReader(categories_synset)

    category_list = []
    synset = []
    for col in fileTemp:
        category_list.append(col['category'])
        synset.append(col['synset'])

    # Form a category to synset dictionary
    category2synset = {category_list[i]: synset[i] for i in range(len(category_list))}


    # Form a category2objFilename dictionary
    category2objFilename = dict()
    for catTemp in category_list:
        category2objFilename[catTemp] = []

    for k, v in objFilename2category.items():
        for catTemp in v:
            if catTemp in category_list:
                category2objFilename[catTemp].append(k)
            # else:
            # not all categories listed in the metadata file are in the category list in the synset file
            # print(catTemp)

    # Removing these categories from the category2synset, category2objFilename, category_list
    category_listTemp = category_list

    for catTemp in category_list:
        if len(category2objFilename[catTemp]) == 0:
            del category2objFilename[catTemp]
            del category2synset[catTemp]
            category_listTemp.remove(catTemp)

    category_list = category_listTemp



    return category_list, category2synset, category2objFilename


def get_model(file_name):
    """
    This code outputs PCD from a ShapeNet object file_name (.obj)
    """
    location = DIR_NAME + file_name

    object_mesh = o3d.io.read_triangle_mesh(location)
    object_mesh.compute_vertex_normals()
    object_mesh.compute_triangle_normals()

    object_pcd = object_mesh.sample_points_uniformly(number_of_points=NUM_POINTS_PER_CAD_MODEL,
                                                     use_triangle_normal=True)
    object_pcd.estimate_normals()

    # points = np.asarray(object_pcd.points)
    # points = torch.from_numpy(points).float()

    # return {'pcd': object_pcd, 'points': points}
    return object_pcd



def visualize_model(pcd_example):
    """
    The function outputs a 3D visualization of the CAD model
    pcd_example - is an o3d.geometry.PointCloud data type
    """
    """ centering the pcd """
    center = pcd_example.get_center()
    max_bound = pcd_example.get_max_bound()
    min_bound = pcd_example.get_min_bound()

    pcd_example.translate(-center)
    center = pcd_example.get_center()

    pcd_example.paint_uniform_color([0.5, 0.5, 0.5])

    """ getting coordinate frame to plot """
    origin_pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
    origin_pose.scale(10, center=center)

    """ generating sphere of points around the object """
    rad = np.linalg.norm(max_bound - min_bound, ord=2)
    rad_vec = [rad, 1.2 * rad]

    sphere_points = o3d.geometry.PointCloud()
    for i in range(len(rad_vec)):
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=rad_vec[i], resolution=20)
        sphere_points_new = sphere_mesh.sample_points_uniformly(number_of_points=100)

        tempPoints = np.concatenate((sphere_points_new.points, sphere_points.points), axis=0)
        sphere_points.points = o3d.utility.Vector3dVector(tempPoints)

        # add sphere points_new to sphere_points

    sphere_points.paint_uniform_color([1.0, 0.0, 0.0])

    """ plotting the axis, object pcd, and random points on the sphere"""
    o3d.visualization.draw_geometries([origin_pose, pcd_example, sphere_points])


class Dataset3dModels(torch.utils.data.Dataset):
    """
    This creates the dataset for ShapeNetSem.
    It outputs ShapeNet models
    It is to be used with dataset loader in Pytorch
    """
    def __init__(self, metadata_file=METADATA_FILE, dir_name=DIR_NAME, number_of_points=NUM_POINTS_PER_CAD_MODEL, transform=None):
        self.metadata_file = pd.read_csv(metadata_file)
        self.dir_name = dir_name
        self.transform = transform
        self.number_of_points = number_of_points

    def __len__(self):
        return len(self.metadata_file)

    def __getitem__(self, idx):
        object_file_name = self.metadata_file.iloc[idx, 0]
        object_file_name = object_file_name[4:] + '.obj'
        object_file = os.path.join(self.dir_name, object_file_name)

        object_mesh = o3d.io.read_triangle_mesh(object_file)
        object_mesh.compute_vertex_normals()
        object_mesh.compute_triangle_normals()

        object_pcd = object_mesh.sample_points_uniformly(number_of_points=self.number_of_points,
                                                         use_triangle_normal=True)
        object_pcd.estimate_normals()

        points = np.asarray(object_pcd.points)
        points = torch.from_numpy(points).float()

        object_data = {'pcd': object_pcd, 'points': points, 'mesh': object_mesh}

        return object_data


def get_sample(idx=9, num_of_points=100000, need_mesh=False):

    dataset = Dataset3dModels(number_of_points=num_of_points)
    object_data = dataset[idx]
    pcd = object_data['pcd']
    points = object_data['points']

    if need_mesh:
        return object_data['mesh']
    else:
        return pcd



def generate_depth_data(idx=9, num_of_points=100000, location='depth_images/'):
    """ Generates depth images for a ShapeNet CAD model """

    # generating the model
    model_pcd = get_sample(idx=idx, num_of_points=num_of_points)
    center = model_pcd.get_center()
    model_pcd.translate(-center)
    model_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    location = location + str(idx) + '/'
    os.mkdir(path=location)

    # determining size of the 3D object
    diameter = np.linalg.norm(np.asarray(model_pcd.get_max_bound()) - np.asarray(model_pcd.get_min_bound()))

    # determining camera locations and radius
    camera_distance_vector = [2 * diameter, 5 * diameter]
    camera_locations = gu.get_camera_locations(camera_distance_vector)
    radius = gu.get_radius(object_diameter=diameter, cam_location=np.max(camera_distance_vector))

    # visualizing 3D object and all the camera locations
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([model_pcd, coordinate_frame, camera_locations])

    # generating radius for view sampling
    gu.sample_depth_pcd(centered_pcd=model_pcd, camera_locations=camera_locations, radius=radius, folder_name=location)



class DepthPointCloud(torch.utils.data.Dataset):
    """
    This creates the dataset for depth point clouds of CAD models.
    It outputs the depth point clouds stored in a given file location.
    It is to be used with dataset loader in Pytorch.
    """
    def __init__(self, dir_name, object_file, camera_locations_file, metadata_file):
        self.metadata_file = metadata_file
        self.dir_name = dir_name
        self.object_file = object_file
        self.camera_locations_file = camera_locations_file

        self.metadata = pd.read_csv(self.dir_name + self.metadata_file)
        self.object = o3d.io.read_point_cloud(self.dir_name + self.object_file)
        self.camera_locations = o3d.io.read_point_cloud(self.dir_name + self.camera_locations_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        depth_pcd_file_name = self.metadata.iloc[idx, 0]
        depth_pcd_file = os.path.join(self.dir_name, depth_pcd_file_name)

        depth_pcd = o3d.io.read_point_cloud(depth_pcd_file)
        depth_pcd.estimate_normals()
        depth_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        return depth_pcd




