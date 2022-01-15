import numpy as np
import os
import open3d as o3d
import csv
import common
import sys

from tsdf_fusion import Fusion


sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','dataset-api','utils'))
from car_instance.car_models import *


'''
Run this script to test the averaging of two TSDFs from two different cars.
'''
def average_tsdf(tsdf_list, weights):
    """Function to average the tsdf
    """
    #assert weights are all nonnegative
    assert(len(tsdf_list) == len(weights))
    avg_tsdf = np.zeros(tsdf_list[0].shape)
    weights_sum = 0
    for tsdf, weight in zip(tsdf_list, weights):
        avg_tsdf += tsdf * weight
        weights_sum += weight
    avg_tsdf /= weights_sum
    return avg_tsdf

def average_kpts(kpt_list, weights):
    assert (len(kpt_list) == len(weights))
    avg_kpts = np.zeros(kpt_list[0].shape)
    weights_sum = 0
    for kpts, weight in zip(kpt_list, weights):
        avg_kpts += kpts * weight
        weights_sum += weight
    avg_kpts /= weights_sum
    return avg_kpts

def save_models_as_o3d_mesh(dir_location_src, dir_location_dest, cluster=True, viz=True):
    model_names = car_name2id.keys()
    model_locations = ["{}/{}.off".format(dir_location_src, model_name) for model_name in model_names]
    o3d_model_locations = ["{}/{}.ply".format(dir_location_dest, model_name) for model_name in model_names]
    for loc in range(len(model_locations)):
        model_mesh = common.Mesh.from_off(model_locations[loc])
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(model_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(model_mesh.faces)
        if cluster:
            triangle_clusters, cluster_n_triangles, cluster_area = (o3d_mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            print("getting largest cluster")
            print("loc", model_locations[loc])
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            o3d_mesh.remove_triangles_by_mask(triangles_to_remove)

        o3d_mesh.compute_vertex_normals()
        if viz:
            o3d_mesh.paint_uniform_color([1, 0.706, 0])
            o3d.visualization.draw_geometries([o3d_mesh])
        o3d.io.write_triangle_mesh(o3d_model_locations[loc], o3d_mesh)

if __name__ == "__main__":
    # ---------------------------------------------------#
    # save_models_as_o3d_mesh("./car_models/2_watertight/", "./car_models/2_watertight_o3d_clustered/")
    # ---------------------------------------------------#


    print("Testing script for averaging two cars")

    # load off file
    suv_name = "aodi-Q7-SUV"
    sedan_name = "aodi-a6"
    mazida_name = "mazida-6-2015"
    test_car_name = "dihao-EV"
    depths_dir = "./car_models/2_depth/"
    scaled_kpts = "./car_models/1_scaled_kpts/"
    tsdf_dir = "./car_models/fused_tsdf/"
    mesh_dir_unscaled = "./car_models/2_watertight_frame_fixed/" #weird internal artifacts don't use
    mesh_dir_scaled = "./car_models/2_watertight/"
    file_ext = ".off.h5"

    meshes = []

    # load rendered depths
    suv_fpath = os.path.join(depths_dir, suv_name+file_ext)
    sedan_fpath = os.path.join(depths_dir, sedan_name+file_ext)
    mazida_fpath = os.path.join(depths_dir, mazida_name+file_ext)
    test_car_fpath = os.path.join(depths_dir, test_car_name+file_ext)
    model_paths = [suv_fpath, sedan_fpath, test_car_fpath]
    tsdf_paths = [os.path.join(tsdf_dir, suv_name+'.npy'), os.path.join(tsdf_dir, sedan_name+'.npy'), os.path.join(tsdf_dir, test_car_name+'.npy')]
    # model_path_suv = [suv_fpath]
    # model_path_sedan = [sedan_fpath]
    weights = [1, .25, .25]

    # fuse to generate tsdf
    app = Fusion()
    # tsdf_list = app.get_tsdf(model_paths)
    tsdf_list = app.load_tsdf(tsdf_paths)

    # average two tsdf
    avg_tsdf = average_tsdf(tsdf_list, weights)

    # averaged tsdf to mesh
    avg_mesh = app.tsdf_to_mesh([avg_tsdf])
    avg_mesh = avg_mesh[0]
    meshes.append(avg_mesh)

    kpt_list = []
    for path in model_paths:
        # add keypoints
        keypoints_xyz = []
        with open(os.path.join(scaled_kpts, os.path.basename(path).split('.')[0] + '.csv'), newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                keypoints_xyz.append([float(x) for x in row])

        # rotation fix
        # (note: this is a rotation + reflection)
        # 90 deg around y, then a reflection across yz (x->-x)
        keypoints_xyz = np.array(keypoints_xyz)
        R = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        for i in range(keypoints_xyz.shape[0]):
            keypoints_xyz[i, :] = np.transpose(R @ keypoints_xyz[i, :].T)

        kpt_list.append(keypoints_xyz)

    keypoints_xyz = average_kpts(kpt_list, weights)

    keypoint_markers = []
    for xyz in keypoints_xyz:
        kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.005)
        kpt_mesh.translate(xyz)
        kpt_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(kpt_mesh)

    for avg_mesh in meshes:

        #visualize mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(avg_mesh['vertices'])
        o3d_mesh.triangles = o3d.utility.Vector3iVector(avg_mesh['triangles'])
        o3d_mesh.compute_vertex_normals()

        # #use clustering to remove artifacts
        # triangle_clusters, cluster_n_triangles, cluster_area = (o3d_mesh.cluster_connected_triangles())
        # triangle_clusters = np.asarray(triangle_clusters)
        # cluster_n_triangles = np.asarray(cluster_n_triangles)
        # cluster_area = np.asarray(cluster_area)
        # print("show largest cluster")
        # largest_cluster_idx = cluster_n_triangles.argmax()
        # triangles_to_remove = triangle_clusters != largest_cluster_idx
        # o3d_mesh.remove_triangles_by_mask(triangles_to_remove)


        model_pcd = o3d_mesh.sample_points_uniformly(10000)
        model_pcd.paint_uniform_color([0, 0.5, 0.5])

        #outlier removal to try and remove internal artifacts
        o3d.visualization.draw_geometries([o3d_mesh] + [] + keypoint_markers)
        # breakpoint()

    # ---------------------------------------------------#
    # # original cad models that have interior
    # models_with_interiors = ['MG-GT-2015', 'Skoda_Fabia-2011', 'beiqi-huansu-H3', 'bentian-fengfan', 'biaozhi-408',
    #                          'biyadi-qin', 'biyadi-tang', 'changan-CS35-2012', 'changan-cs5', 'changcheng-H6-2016',
    #                          'dazhong', 'dihao-EV', 'feiyate', 'fengtian-MPV', 'jili-boyue', 'jipu-3', 'kaidilake-CTS',
    #                          'lingmu-SX4-2012', 'linken-SUV', 'rongwei-RX5', 'sanling-oulande', 'supai-2016',
    #                          ' xiandai-i25-2016']
    # # new mesh models that have interiors after tsdf averaging and clustering
    # new_models_with_interiors = ['036-CAR01', '037-CAR02', 'aodi-a6', 'baoma-X5', 'baoshijie-kayan',
    #                              'benchi-GLK-300', 'benchi-ML500', 'benchi-SUR', 'biyadi-F3',
    #                              'dongfeng-fengguang-S560', 'jili-boyue',
    #                              'feiyate', 'fengtian-liangxiang', 'fengtian-SUV-gai',
    #                              'kaidilake-CTS',
    #                              'mazida-6-2015', 'sikeda-jingrui', 'Skoda_Fabia-2011', 'yingfeinidi-SUV']
    # #kinda messy = [dazhong]
    # #quick stats
    # prev_models = set(models_with_interiors)
    # print("prev_model len", len(prev_models))
    # new_models = set(new_models_with_interiors)
    # print("new_model len", len(new_models))
    # print("old corrupted that aren't in new corrupted", prev_models.difference(new_models))
    # print("new corrupted that aren't in old corrupted", new_models.difference(prev_models))