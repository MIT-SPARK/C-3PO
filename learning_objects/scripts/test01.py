
"""
This file contains various code snippets that shows you how to meaningfully use the functions in this library.
"""
# MESH_FOLDER_NAME: str = '../../../datasets/KeypointNet/ShapeNetCore.v2.ply/'


# """ Test """
# import csv
# import torch
# import os
# import pandas as pd
# import open3d as o3d
# import json
# import numpy as np
# import learning_objects.utils.general_utils as gu
#
# class_id = "02958343"
# model_id = "1b94aad142e6c2b8af9f38a1ee687286"
# object_mesh_file = MESH_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.ply'
# mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
# mesh.paint_uniform_color([0.5, 0.5, 0.5])
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
# pcd = mesh.sample_points_uniformly(number_of_points=100000)
# o3d.visualization.draw_geometries([pcd])


# """ CODE TO GENERATE AND SAVE DEPTH IMAGES OF A KEYPOINTNET-SHAPENETCORE OBJECT """
# # import learning_objects.utils.shapenet_sem as sn_sem
# import learning_objects.utils.keypointnet as kp_net
# import os
#
# # idx=9 is the 9th object in the dataset of the ShapeNet object
# # Following code will store files in the location 'depth_images/9/' folder
# location = '../data/depth_images/'
# # os.mkdir(path=location)
# class_id = "02958343"                           # car
# model_id = "1b94aad142e6c2b8af9f38a1ee687286"   # a particular car model
# os.mkdir(path=location + class_id + '/')
# os.mkdir(path=location + class_id + '/' + model_id + '/')
# location = location + class_id + '/' + model_id + '/'
# kp_net.generate_depth_data(class_id=class_id, model_id=model_id, num_of_points=100000, location=location)








# """ CODE THAT CREATES A DATASET AND VISUALIZES THE DEPTH IMAGES GENERATED """
# import learning_objects.utils.shapenet_sem as sn_sem
# import open3d as o3d
# import numpy as np
# import random
#
# # comment this out if already generated
# # sn_sem.generate_depth_data(idx=9, num_of_points=100000, location='../data/depth_images/')
# dataset = sn_sem.DepthPointCloud(dir_name='../data/depth_images/9/', object_file='object.pcd', camera_locations_file='camera_locations.pcd', metadata_file='metadata.csv')
# cad_model = dataset.object
#
# # r = list(range(len(dataset)))
# # random.shuffle(r)
#
# for idx in range(len(dataset)):
#    print(idx)
#    depth_pcd = dataset[idx]
#    depth_pcd.remove_non_finite_points()
#
#    depth_pcd_points = depth_pcd.points
#    print(np.shape(depth_pcd_points))
#    if idx % 10 == 0:
#       depth_pcd.paint_uniform_color([0.8, 0.0, 0.0])
#       o3d.visualization.draw_geometries([depth_pcd])








# """ CODE TO GENERATE AND SAVE DEPTH IMAGES OF A SHAPENET OBJECT """
# import learning_objects.utils.shapenet_sem as sn_sem
# import os
#
# # idx=9 is the 9th object in the dataset of the ShapeNet object
# # Following code will store files in the location 'depth_images/9/' folder
# location = '../data/depth_images/'
# os.mkdir(path=location)
# sn_sem.generate_depth_data(idx=9, num_of_points=100000, location=location)








# """ TEST: RENDERING DEPTH IMAGES FROM 3D OBJECTS """
# import learning_objects.utils.general_utils as gu
#
# gu.test_rendering_depth_images()
# # The output image and depth images should be saved in the ../data/tmp/ folder








# """ VISUALIZE A MODEL IN SHAPENET """
# import learning_objects.utils.shapenet_sem as sn_sem
#
# """ getting a point cloud data (pcd) from ShapeNet dataset """
# pcd_example = sn_sem.get_sample(idx=300, num_of_points=100000)
#
# """ visualizing the data """
# sn_sem.visualize_model(pcd_example)








# """ VISUALIZE MODELS IN A GIVEN CATEGORY """
# import learning_objects.utils.shapenet_sem as sn_sem
#
# categoryList, category2synset, category2objFilename = sn_sem.process()
# # print(categoryList)
# # print(category2objFilename)
#
# idx = 20
# num = 10
#
# aCategory = categoryList[idx]
# print(f"Displaying {num} objects of Category: {aCategory}")
#
# for i in range(min(num, len(category2objFilename[aCategory]))):
#     model_pcd = sn_sem.get_model(category2objFilename[aCategory][i])
#     sn_sem.visualize_model(model_pcd)