import os
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import h5py as h5
from PIL import Image
import IPython
import math
import open3d as o3d
import matplotlib.pyplot as plt
from ycb_processing import get_largest_cluster, \
    load_mesh, get_closest_cluster, \
    save_rgbFromObj, load_image_and_model_ycb_video

import scipy.io



sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','ycb-tools'))
# Define folders
default_ycb_folder = os.path.join("models", "ycb")
print("default_ycb_folder", default_ycb_folder)

from reg_utils import (render_cad,
                       generate_weighted_model,
                       project2d,
                       load_car_poses,
                       dists_from_ray,
                       load_car_model,
                       load_disparity_to_depth,
                       fix_depth_map,
                       set_axes_equal,
                       load_kpt_lib,
                       load_gsnet_keypoints,
                       select_cad_dist_bounds,
                       load_apollo_data,
                       SE3_transform,
                       load_2d_keypoints)
import apollo_stereo_utils

from get_3d_keypoints import get_corresponding_car_for_kpts
from generate_pcl_utils import depth_img_to_pcl, save_pcl, img_to_pcl
import c3po.utils.general as gu

ycb_video_data_folder = "../../../../../../../media/lisa/Figgy/detectron2_datasets/YCB_Video_Dataset/"

#CONSTANTS
seq_num = 92;
nums = [762, 1112, 1719, 2299, 2172, 1506, 1626, 2018, 2991, 1591, 1898,
            1107, 1104, 1800, 1619, 2305, 1335, 1716, 1424, 2218, 1873, 731, 1153, 1618,
            1401, 1444, 1607, 1448, 1558, 1164, 1124, 1952, 1254, 1567, 1554, 1668,
            2512, 2157, 3467, 3160, 2393, 2450, 2122, 2591, 2542, 2509, 2033, 2089,
            2244, 2402, 1917, 2009, 900, 837, 1929, 1830, 1226, 1872, 1720, 1864,
            754, 533, 680, 667, 668, 653, 801, 849, 884, 784, 1016, 951, 890, 719, 908,
            694, 864, 779, 689, 789, 788, 985, 743, 953, 986, 890, 897, 948, 453, 868, 842, 890]

width = 640
height = 480

intrinsic_matrix_color = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
intrinsic_matrix_depth = np.array([[567.6188, 0, 310.0724] ,[0, 568.1618, 242.7912], [0, 0, 1]])

depth2color = np.array([[0.9997563, 0.02131301, -0.005761033, 0.02627148],
                   [0.02132165, 0.9997716, -0.001442874, -0.0001685539],
                   [0.005728965, 0.001565357, 0.9999824, 0.0002760285]])

from ycb_generate_point_could import registerDepthMap, writePLY

# opt.intrinsic_matrix_color_cmu = [1077.836, 0, 323.7872;
# 0, 1078.189
# 279.6921;
# 0, 0, 1];
# opt.intrinsic_matrix_depth_cmu = [576.3624, 0, 319.2682;
# 0, 576.7067
# 243.8256;
# 0, 0, 1];

# opt.depth2color_cmu = [0.9999962, -0.002468782, 0.001219765, 0.02640966;
# ...
# 0.002466791, 0.9999956, 0.001631345, -9.9086e-05;
# ...
# - 0.001223787, -0.00162833, 0.9999979, 0.0002972445];

def depthMapToPointCloud(masked_depth, depth_scale, rgbImage, intrinsic_matrix_depth, organized=False):
    if masked_depth.shape[-1] == 3:
        masked_depth = masked_depth[:,:,0]

    xyzDepth = np.empty((4,1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyzDepth[3] = 1

    invDepthFx = 1.0 / intrinsic_matrix_depth[0,0]
    invDepthFy = 1.0 / intrinsic_matrix_depth[1,1]
    depthCx = intrinsic_matrix_depth[0,2]
    depthCy = intrinsic_matrix_depth[1,2]

    height = masked_depth.shape[0]
    width = masked_depth.shape[1]

    if organized:
        cloud = np.zeros((height, width, 6), dtype=np.float64)
    else:
        cloud = np.zeros((1, height * width, 6), dtype=np.float64)

    goodPointsCount = 0
    for v in range(height):
        for u in range(width):

            depth = masked_depth[v, u] / depth_scale
            print("depth", depth)

            if organized:
                row = v
                col = u
            else:
                row = 0
                col = goodPointsCount

            if depth <= 0:
                if organized:
                    if depth <= 0:
                        cloud[row, col, 0] = float('nan')
                        cloud[row, col, 1] = float('nan')
                        cloud[row, col, 2] = float('nan')
                        cloud[row, col, 3] = 0
                        cloud[row, col, 4] = 0
                        cloud[row, col, 5] = 0
                continue
            # ORIGINAL
            cloud[row, col, 0] = ((u - depthCx) * depth) * invDepthFx
            cloud[row, col, 1] = ((v - depthCy) * depth) * invDepthFy
            cloud[row, col, 2] = depth
            cloud[row, col, 3] = rgbImage[v, u, 0]
            cloud[row, col, 4] = rgbImage[v, u, 1]
            cloud[row, col, 5] = rgbImage[v, u, 2]
            if not organized:
                goodPointsCount += 1

    if not organized:
        cloud = cloud[:, :goodPointsCount, :]
    return cloud

if __name__ == "__main__":
    # smoke test done
    # for target_object in ["002_master_chef_can", "003_cracker_box", \
    #                       "004_sugar_box", "005_tomato_soup_can", \
    #                       "006_mustard_bottle", "007_tuna_fish_can", \
    #                       "008_pudding_box", "009_gelatin_box", \
    #                       "010_potted_meat_can", "011_banana", \
    #                       "019_pitcher_base", "021_bleach_cleanser", \
    #                       "024_bowl", "025_mug", \
    #                       "035_power_drill", "036_wood_block", "037_scissors", \
    #                       "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]:
    #     model_mesh = load_mesh(target_object, viz=False, data_folder= ycb_video_data_folder + "/models/", mesh_filename="textured.obj")

    for target_object in ["021_bleach_cleanser"]:
        print("target_object", target_object)
        model_mesh = load_mesh(target_object, viz=True, data_folder= ycb_video_data_folder + "/models/", mesh_filename="textured.obj")
        # depth_filename = 001100-depth.png
        # rgb_filename = 001100-color.png
        # mask_filename = 001100-label.png
        # metadata = 001100-meta.mat
        referenceCamera = "NP5"
        for scene in range(92):
            data_folder = os.path.join(ycb_video_data_folder, "data", str(scene).zfill(4))
            num_frames = sorted(os.listdir(data_folder))[-1][:6].lstrip('0')
            for frame in range(1,int(num_frames)+1):
                frame = str(frame).zfill(6)
                print("scene", scene)
                print("processing frame", frame)
                if not os.path.exists(os.path.join(ycb_video_data_folder, "clouds", str(scene).zfill(4), target_object)):
                    os.makedirs(os.path.join(ycb_video_data_folder, "clouds", str(scene).zfill(4), target_object))

                basename = "{0}-".format(frame)
                depthFilename = os.path.join(data_folder, basename + "depth.png")
                rgbFilename = os.path.join(data_folder, basename + "color.png")
                maskFilename = os.path.join(data_folder, basename +"label.png")

                metadataFilename = os.path.join(data_folder, basename +"meta.mat")
                metadata = scipy.io.loadmat(metadataFilename)

                print("metadata poses", metadata['poses'])
                print("metadata center", metadata['center'])

                print("metadata intrinsic_matrix", metadata['intrinsic_matrix'])
                print("metadata cls_indexes", metadata['cls_indexes'])
                print("metadata factor_depth", metadata['factor_depth'])
                mask = cv2.imread(maskFilename)

                for obj in range(len(metadata['cls_indexes'])):
                    model_id = metadata['cls_indexes'][obj][0] - 1
                    print("model_id", model_id)

                    objFromRef = metadata['poses'][:,:,obj]
                    print(objFromRef)
                    kernel = np.ones((10, 10), np.uint8)
                    erosion = cv2.erode(mask, kernel, iterations=1)
                    temp_mask = np.asarray(erosion)
                    obj_mask = np.zeros((temp_mask.shape[0], temp_mask.shape[1])).astype(np.uint8)
                    for row in range(temp_mask.shape[0]):
                        for col in range(temp_mask.shape[1]):
                            if temp_mask[row, col, 0] == model_id + 1:  # we want to keep only obj points as white points
                                obj_mask[row, col] = 1
                            else:
                                obj_mask[row, col] = 0

                    # plt.subplot()
                    # obj_mask_img = o3d.geometry.Image(obj_mask.astype(np.float32))
                    # plt.imshow(obj_mask_img)
                    # plt.show()
                    if not os.path.isfile(rgbFilename):
                        print(
                            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
                        exit(1)
                    # load_image_and_model_ycb_video(scene, frame)
                    with Image.open(rgbFilename) as image:
                        rgbImage = np.array(image) #[:,:,::-1]

                        # closing all open windows
                        depthScale = metadata['factor_depth'][0][0]

                        registeredDepthMap = cv2.imread(depthFilename).astype(np.float32)
                        masked_depth = cv2.bitwise_and(registeredDepthMap, registeredDepthMap, mask=obj_mask)

                        #
                        # plt.subplot()
                        # masked_depth_o3d = o3d.geometry.Image(masked_depth)
                        # print("VIZZZZ")
                        # #can't see anything all dark because depth values are small
                        # obj_mask_img = o3d.geometry.Image(masked_depth.astype(np.float32))
                        # plt.imshow(obj_mask_img)
                        # # print("showing depth_o3d")
                        # plt.show()
                        pointCloud = depthMapToPointCloud(masked_depth, depthScale, rgbImage, intrinsic_matrix_depth, organized=False)
                        writePLY(data_folder+target_object+"temp_masked.ply", pointCloud)
                        pcd = o3d.io.read_point_cloud(data_folder+target_object+"temp_masked.ply")
                        print("TRYING TO VIS PCD")
                        o3d.visualization.draw_geometries([pcd])
                        print("TRYING TO VIS PCD")

                        #

                # condense = True
                # if condense:
                #     objFromRef = h5.File(objFromRefFilename)['H_table_from_reference_camera'][:]
                #
                #     mask_filename = os.path.join(ycb_data_folder + target_object, "masks", basename + "_mask.pbm")
                #     mask = cv2.imread(mask_filename)
                #     kernel = np.ones((10,10), np.uint8)
                #     erosion = cv2.erode(mask, kernel, iterations=1)
                #     temp_mask = np.asarray(erosion)
                #     obj_mask = np.zeros((temp_mask.shape[0], temp_mask.shape[1])).astype(np.uint8)
                #     for row in range(temp_mask.shape[0]):
                #         for col in range(temp_mask.shape[1]):
                #             if temp_mask[row,col,0] == 0: #we want to keep black points
                #                 obj_mask[row,col] = 1
                #             else:
                #                 obj_mask(image)
                #     objFromRefFilename = os.path.join(ycb_data_folder+target_object, 'poses', '{0}_{1}_pose.h5'.format(referenceCamera, viewpoint_angle))
                #     objFromRef = h5.File(objFromRefFilename)['H_table_from_reference_camera'][:]
                #
                #     mask_filename = os.path.join(ycb_data_folder + target_object, "masks", basename + "_mask.pbm")
                #     mask = cv2.imread(mask_filename)
                #     kernel = np.ones((10,10), np.uint8)
                #     erosion = cv2.erode(mask, kernel, iterations=1)
                #     temp_mask = np.asarray(erosion)
                #     obj_mask = np.zeros((temp_mask.shape[0], temp_mask.shape[1])).astype(np.uint8)
                #     for row in range(temp_mask.shape[0]):
                #         for col in range(temp_mask.shape[1]):
                #             if temp_mask[row,col,0] == 0: #we want to keep black points
                #                 obj_mask[row,col] = 1
                #             else:
                #                 obj_mask[row,col] = 0
                #
                # if not os.path.isfile(rgbFilename):
                #     print("The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
                #     exit(1)
                # with Image.open(rgbFilename) as image:
                #         rgbImage = np.array(image)
                #         depthK = calibration["{0}_depth_K".format(viewpoint_camera)][:]
                #
                #         rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
                #
                #         depthScale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001 # 100um to meters
                #         H_RGBFromDepth, _ = getRGBFromDepthTransform(calibration, viewpoint_camera, referenceCamera)
                #
                #         unregisteredDepthMap = h5.File(depthFilename)["depth"][:]
                #
                #         unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale
                #
                #         registeredDepthMap = registerDepthMap(unregisteredDepthMap,
                #                                               rgbImage,
                #                                               depthK,
                #                                               rgbK,
                #                                               H_RGBFromDepth)
                #         #mask both rgbImage and depthMap with mask
                #
                #         #
                #         # print("registeredDepthMap.shape", registeredDepthMap.shape)
                #         # print("unregisteredDepthMap.shape", unregisteredDepthMap.shape)
                #         # #
                #         # # plt.subplot()
                #         # # registered_depth_o3d = o3d.geometry.Image(registeredDepthMap.astype(np.float32))
                #         # # plt.imshow(registered_depth_o3d)
                #         # # plt.show()
                #         masked_depth = cv2.bitwise_and(registeredDepthMap, registeredDepthMap, mask=obj_mask)
                #         # depth_o3d = o3d.geometry.Image(masked_depth.astype(np.float32))
                #         #
                #         # segmented_pcd = depth_img_to_pcl(depth_o3d, rgbK)
                #         # pointCloud = registeredDepthMapToPointCloud(registeredDepthMap, rgbImage, rgbK)
                #         # saving point cloud wrt the RGB camera pose
                #         pointCloud = registeredDepthMapToPointCloud(masked_depth, rgbImage, rgbK, None, objFromRef, organized=False)
                #         writePLY(ycb_data_folder+target_object+"/clouds/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+"_masked.ply", pointCloud)
                #         print("pointCloud.shape", pointCloud.shape)
                #         print("Testing IO for point cloud ...")
                # pcd = o3d.io.read_point_cloud(ycb_data_folder+target_object+"/clouds/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+"_masked.ply")
                # # if clustering with dbscan, uncomment the below
                # # diameter = np.linalg.norm(
                # #     np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))
                # # eps = np.min(diameter)
                # # print("eps", eps)
                # # largest_cluster_pcd = get_largest_cluster(pcd, viz=False, eps=eps)
                #
                # largest_cluster_pcd = pcd #get_largest_cluster(pcd, viz=False, eps=eps)
                #
                # # further clustering with euclidian distance based thresholding
                # if largest_cluster_pcd is not None:
                #     rgbFromObj = save_rgbFromObj(target_object, viewpoint_camera, viewpoint_angle, only_return=False) #save gt poses!
                #     R_true = rgbFromObj[:3, :3]
                #     t_true = np.expand_dims(rgbFromObj[:3, 3], axis=1)
                #
                #     model_pcd = model_mesh.sample_points_uniformly(number_of_points=3000)
                #     model_pcd.points = o3d.utility.Vector3dVector(
                #         (R_true @ np.asarray(model_pcd.points).transpose() + t_true).transpose())
                #     largest_cluster_pcd = get_closest_cluster(largest_cluster_pcd, model_pcd, viz=False)
                #     if largest_cluster_pcd is not None:
                #         if not os.path.exists(ycb_data_folder + "/" + target_object + "/clouds/largest_cluster"):
                #             os.makedirs(ycb_data_folder + "/" + target_object + "/clouds/largest_cluster")
                #         o3d.io.write_point_cloud(ycb_data_folder + target_object + "/clouds/largest_cluster/pc_" + viewpoint_camera + "_" + referenceCamera + "_" + viewpoint_angle + "_masked.ply", largest_cluster_pcd)
