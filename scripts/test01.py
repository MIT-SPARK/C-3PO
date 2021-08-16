
"""
This file contains various code snippets that shows you how to meaningfully use the functions in this library.
"""





# """ CODE THAT CREATES A DATASET AND VISUALIZES THE DEPTH IMAGES GENERATED """
# import utils.dataset_utils as du
# import open3d as o3d
# import numpy as np
# import random
#
# # comment this out if already generated
# # du.generateDepthData(idx=9, num_of_points=100000, location='../data/depth_images/')
# dataset = du.DepthPointCloud(dir_name='../data/depth_images/9/', object_file='object.pcd', camera_locations_file='camera_locations.pcd', metadata_file='metadata.csv')
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
# import utils.dataset_utils as du
#
# # idx=9 is the 9th object in the dataset of the ShapeNet object
# # Following code will store files in the location 'depth_images/9/' folder
# du.generateDepthData(idx=9, num_of_points=100000, location='../data/depth_images/')








# """ TEST: RENDERING DEPTH IMAGES FROM 3D OBJECTS """
# import utils.general_utils as gu
#
# gu.test_rendering_depth_images()
# # The output image and depth images should be saved in the temp/ folder








# """ VISUALIZE A MODEL IN SHAPENET """
# import utils.dataset_utils as du
#
# """ getting a point cloud data (pcd) from ShapeNet dataset """
# pcd_example = du.getSample_ShapeNet(idx=3572, num_of_points=100000)
#
# """ visualizing the data """
# du.visualizePCD(pcd_example)








# """ VISUALIZE MODELS IN A GIVEN CATEGORY """
# import utils.dataset_utils as du
#
# categoryList, category2synset, category2objFilename = du.process_shapeNetSem()
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
#     model_pcd = du.getCADmodel(category2objFilename[aCategory][i])
#     du.visualizePCD(model_pcd)