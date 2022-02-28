import copy
import cv2
import datetime
import json
import matplotlib.pyplot as plt
from PIL import Image
import h5py as h5
import numpy as np
import os
import pickle
import sys
import time
import trimesh
from tqdm import tqdm
from scipy.spatial import ConvexHull
import pickle as pkl
import open3d as o3d
import random

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','ycb-tools'))
# Define folders
default_ycb_folder = os.path.join("models", "ycb")
referenceCamera = "NP5"

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

from get_3d_keypoints import project2d, plot_pts_on_image
from depth_to_pcl_processing import depth_img_to_pcl, save_pcl, img_to_pcl
import learning_objects.utils.general as gu

ycb_data_folder = "../../third_party/ycb-tools/models/ycb/"  # Folder that contains the ycb data.


def load_mesh(target_object, viz=True, calculate_data=False):
    mesh_filename = os.path.join(ycb_data_folder + target_object, "google_16k", "nontextured.ply")
    # mesh_filename = os.path.join(ycb_data_folder + target_object, "poisson", "nontextured.ply")
    #pitcher google_16k is fine, drill is not


    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh.compute_vertex_normals()
    if calculate_data:
        edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = mesh.is_vertex_manifold()
        self_intersecting = mesh.is_self_intersecting()
        watertight = mesh.is_watertight()
        orientable = mesh.is_orientable()

        print(f"  edge_manifold:          {edge_manifold}")
        print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
        print(f"  vertex_manifold:        {vertex_manifold}")
        print(f"  self_intersecting:      {self_intersecting}")
        print(f"  watertight:             {watertight}")
        print(f"  orientable:             {orientable}")
    if viz:
        keypoint_radius = 0.005

        keypoint_markers = []
        keypoints_xyz = [[.1,0,0], [0,.1,0], [0,0,.1], [0,0,0]]
        for xyz in [keypoints_xyz[0]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in [keypoints_xyz[1]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.0, 0.8, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in [keypoints_xyz[2]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.0, 0.0, 0.8])
            keypoint_markers.append(new_mesh)
        for xyz in [keypoints_xyz[3]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0.8, 0.8])
            keypoint_markers.append(new_mesh)
        o3d.visualization.draw_geometries([mesh] + keypoint_markers)
    return mesh

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def save_kpts(kpt_idxs, points_xyz, target_oject):
    """
    Takes list of kpt indices in the points_xyz np.ndarray and saves into npy file
    :param kpt_idxs:
    :param points_xyz:
    :return:
    """
    kpt_xyz = np.asarray(points_xyz)[kpt_idxs]
    kpt_filename = os.path.join(ycb_data_folder + target_object, "kpts_xyz.npy")
    np.save(kpt_filename, kpt_xyz)
    loaded_array = np.load(kpt_filename)
    print("saved and loaded array", loaded_array)

def load_obj_from_ref_H(target_object, camera, viewpoint_angle):
    calibrationFilename = os.path.join(ycb_data_folder + target_object, "calibration.h5")
    calibration = h5.File(calibrationFilename)
    rgbKey = "H_{0}_from_{1}".format(camera, referenceCamera)
    rgbFromRef = calibration[rgbKey][:]

    objFromRefFilename = os.path.join(ycb_data_folder + target_object, 'poses',
                                      '{0}_{1}_pose.h5'.format(referenceCamera, viewpoint_angle))
    objFromRef = h5.File(objFromRefFilename)['H_table_from_reference_camera'][:]
    print("objFromRef", objFromRef)
    print("rgbFromRef", rgbFromRef)

    return np.asarray(rgbFromRef), np.asarray(objFromRef)

def load_intrinsics(target_object, viewpoint_camera):
    calibrationFilename = os.path.join(ycb_data_folder + target_object, "calibration.h5")
    calibration = h5.File(calibrationFilename)
    rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
    return rgbK


def get_largest_cluster(pcd, viz=False):
    #downsample pcd
    if len(pcd.points) > 70000: #maybe cap at 80000?
        print("DOWNSAMPLING to do clustering")
        pcd = pcd.voxel_down_sample(voxel_size=0.001)
        print("pcd now has reduced number of points", len(pcd.points))

    cluster = np.array(pcd.cluster_dbscan(eps=.2, min_points=300, print_progress=True))
    max_label = cluster.max()
    print("number of clusters", max_label + 1)
    if max_label + 1 == 0:
        print("No clusters found")
        return None

    counts = np.bincount(np.where(cluster > 0, cluster, 0))
    largest_cluster_label = np.argmax(counts)
    bool_indices_of_largest_cluster = (cluster == largest_cluster_label)
    largest_cluster_points = np.array(pcd.points)[bool_indices_of_largest_cluster]


    cmap = plt.get_cmap("tab20")
    colors = cmap(cluster / (max_label if max_label > 0 else 1))
    colors[cluster < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    primary_cluster_pcd = o3d.geometry.PointCloud()
    primary_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

    if len(pcd.points) < 500:
        print("less than 500 points in this point cloud, skipping")
        return None

    o3d.geometry.PointCloud.estimate_normals(primary_cluster_pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                                                    max_nn=30))
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(primary_cluster_pcd, camera_location=np.array([0., 0., 0.]))
    primary_cluster_pcd.paint_uniform_color([.5, .5, .5])
    if viz:
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([primary_cluster_pcd])
    return primary_cluster_pcd

def load_image_and_model2(target_object, viewpoint_camera, viewpoint_angle):
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
    if not os.path.isfile(rgbFilename):
        print(
            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
        exit(1)
    # pts = 2d projection of 3d mesh model
    mesh = load_mesh(target_object)
    points = mesh.vertices #these are mesh model points (we assume 0,0,0 is at the center bottom)?
    points_array = np.asarray(points)

    rgbFromRef, objFromRef = load_obj_from_ref_H(target_object, viewpoint_camera, viewpoint_angle)
    table_from_camera = objFromRef @ np.linalg.inv(rgbFromRef) #we want transformation from camera to ref, then ref to table

    rgbK = load_intrinsics(target_object, viewpoint_camera)


    # we need to transform the mesh to world coordinates wrt the rgb camera
    print("points.shape", points_array.shape)
    #mesh points in the rgb coordinates are H_world_to_camera @ points_in_world
    camera_from_table = rgbFromRef @ np.linalg.inv(objFromRef)
    rmat_cam_from_table = camera_from_table[:3,:3]
    rvect_cam_from_table, _ = cv2.Rodrigues(rmat_cam_from_table)
    tvect_cam_from_table = camera_from_table[:3,3]

    ### real pcl
    # pcd = o3d.io.read_point_cloud(ycb_data_folder+target_object+"/clouds/largest_cluster/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+str(viewpoint_angle)+"_masked.ply")
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # points_array = np.asarray(pcd.points)
    # print(points_array[:3])
    # points_homo = np.hstack((points_array, np.ones((points_array.shape[0], 1))))
    # print("points_homo.shape", points_homo.shape)
    # image_pts = rgbK @ points_array.T  # transform the world points to camera coordinates
    # image_pts /= image_pts[-1, :]
    # print("image_pts.shape", image_pts.shape)
    # plot_pts_on_image(image_pts, rgbFilename)
    ###
    #####
    # points_array = np.asarray([[0,0,0], [.1,0,0], [0,.1,0], [0,-.1,0]])
    print(points_array[:3])
    points_homo = np.hstack((points_array, np.ones((points_array.shape[0],1))))
    print("points_homo.shape", points_homo.shape)
    image_pts = rgbK @ camera_from_table[:3,:] @ points_homo.T #transform the world points to camera coordinates
    image_pts /= image_pts[-1, :]
    print("image_pts.shape", image_pts.shape)
    plot_pts_on_image(image_pts, rgbFilename)
    #####

    ##I have gt coordinates of the poses in objFromRef, I need to save instead a npy file 4x4 transformation
    ## of obj from rgbCamera to compare against points on pcl

def save_rgbFromObj(target_object, viewpoint_camera, viewpoint_angle):
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
    if not os.path.isfile(rgbFilename):
        print(
            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
        exit(1)
    rgbFromRef, objFromRef = load_obj_from_ref_H(target_object, viewpoint_camera, viewpoint_angle)
    objFromRGB = objFromRef @ np.linalg.inv(rgbFromRef)  # we want transformation from camera to ref, then ref to table
    rgbFromObj = rgbFromRef @ np.linalg.inv(objFromRef)
    if not os.path.exists(ycb_data_folder + "/" + target_object + "/poses/gt_wrt_rgb"):
        os.makedirs(ycb_data_folder + "/" + target_object + "/poses/gt_wrt_rgb")

    rgbFromObj_filename = os.path.join(ycb_data_folder + target_object, "poses/gt_wrt_rgb/", '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
    np.save(rgbFromObj_filename, rgbFromObj)
    loaded_array = np.load(rgbFromObj_filename)
    print("saved and loaded array", loaded_array)
    return rgbFromObj


if __name__=="__main__":
    target_object = "021_bleach_cleanser"#"021_bleach_cleanser" #"019_pitcher_base"#"035_power_drill"	# Full name of the target object.
    # mesh = load_mesh(target_object, viz=False)
    # points = mesh.vertices
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = points
    # picked_points_idx = pick_points(pcd)
    # save_kpts(picked_points_idx, points, target_object)

    #### testing transformations
    # load_image_and_model2(target_object, "NP1", 30)

    #### saving ground truth transformations wrt saved point cloud frame of reference
    # for viewpoint_camera in ["NP1", "NP2", "NP3", "NP4", "NP5"]:
    #     for viewpoint_angle in range(358):
    #         if viewpoint_angle%3 != 0:
    #             continue
    #         save_rgbFromObj(target_object, viewpoint_camera, viewpoint_angle)

    #### making train/val/testing splits:
    pcd_filepath = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster")

    saved_point_clouds = []
    for filename in os.listdir(pcd_filepath):
        saved_point_clouds.append(filename)
    random.shuffle(saved_point_clouds)
    num_test = int(.15 * len(saved_point_clouds))
    num_val = num_test
    num_train = len(saved_point_clouds) - num_test - num_val


    train_split = saved_point_clouds[:num_train]
    val_split = saved_point_clouds[num_train:num_train+num_val]
    test_split = saved_point_clouds[num_train+num_val:]
    #save splits in npy
    split_path = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster")
    np.save(split_path + '/train_split.npy', train_split)
    np.save(split_path + '/val_split.npy', val_split)
    np.save(split_path + '/test_split.npy', test_split)
