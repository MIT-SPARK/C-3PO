import numpy as np
import time
import os, math
import open3d as o3d
from pathlib import Path

from tsdf_fusion import Fusion


def average_tsdf(tsdf_list, weights):
    """Function to average the tsdf
    """
    assert (len(tsdf_list) == len(weights))
    avg_tsdf = np.zeros(tsdf_list[0].shape)
    weights_sum = 0
    for tsdf, weight in zip(tsdf_list, weights):
        avg_tsdf += tsdf * weight
        weights_sum += weight
    avg_tsdf /= weights_sum
    return avg_tsdf


if __name__ == "__main__":
    print("Animation script to generate rotating car video")
    depths_dir = "./car_models/2_depth/"
    file_ext = ".off.h5"

    car_names_list = ["xiandai-suonata", "lingmu-aotuo-2009", "aodi-Q7-SUV"]
    car_names_list = ["aodi-Q7-SUV"]

    # load rendered depths
    model_paths = [os.path.join(depths_dir, car_name + file_ext) for car_name in car_names_list]

    # fuse to generate tsdf
    app = Fusion()
    all_cars_tsdf_list = app.get_tsdf(model_paths)

    # animation
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # parameters
    save_image = True
    for k in range(len(car_names_list)):
        N = 500  # 250 frames
        o3d_mesh = o3d.geometry.TriangleMesh()
        tsdf = all_cars_tsdf_list[k]
        car_dir = os.path.join("./output_rotating_cars_frames", car_names_list[k])
        Path(car_dir).mkdir(parents=True, exist_ok=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()

        for i in range(N):

            avg_mesh = app.tsdf_to_mesh([tsdf])
            avg_mesh = avg_mesh[0]

            o3d_mesh.vertices = o3d.utility.Vector3dVector(avg_mesh['vertices'])
            o3d_mesh.triangles = o3d.utility.Vector3iVector(avg_mesh['triangles'])
            o3d_mesh.compute_vertex_normals()

            if i == 0:
                vis.add_geometry(o3d_mesh)
                vis.get_render_option().load_from_json("./render_options/vid_option.json")
                # ctr.set_zoom(0.5)

            ctr.rotate(-5, 0)
            vis.update_geometry(o3d_mesh)
            vis.poll_events()
            vis.update_renderer()
            if save_image:
                image_path = os.path.join(car_dir, "car_rotating_%04d.jpg" % i)
                vis.capture_screen_image(image_path)

        vis.close()
