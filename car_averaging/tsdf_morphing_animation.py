import numpy as np
import time
import os, math
import open3d as o3d

from tsdf_fusion import Fusion, average_tsdf


'''
Run this script to generate an animation of the car morphing from one to another.
The cars that you wish to appear in the animation can be modified by changing the car_names_list variable in the main 
function below.
The animation is dumped to a folder as individual frames.
Use ffmpeg to turn frames into a video:
ffmpeg -r 1 -f image2 -i image%04d.jpg -frames:v 25 -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
'''
if __name__ == "__main__":
    print("Animation script for morphing/averaging cars")
    # load off file
    depths_dir = "./car_models/2_depth/"
    file_ext = ".off.h5"

    car_names_list = ["aodi-Q7-SUV", "xiandai-suonata", "lingmu-aotuo-2009", "aodi-Q7-SUV"]

    # load rendered depths
    model_paths = [os.path.join(depths_dir, car_name + file_ext) for car_name in car_names_list]

    # fuse to generate tsdf
    app = Fusion()
    all_cars_tsdf_list = app.get_tsdf(model_paths)

    # animation
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    N = 100 * (len(car_names_list)-1)
    weights = np.linspace(0, len(car_names_list)-1-0.01, N)
    save_image = True
    o3d_mesh = o3d.geometry.TriangleMesh()
    for i in range(N):
        print("Weight = {}".format(weights[i]))
        unnormalized_weight = weights[i]
        weight = unnormalized_weight - math.floor(unnormalized_weight)

        current_car_pos = int(math.floor(unnormalized_weight))
        assert current_car_pos+1 < len(all_cars_tsdf_list)
        tsdf_list = all_cars_tsdf_list[current_car_pos:current_car_pos+2]

        avg_tsdf = average_tsdf(tsdf_list, [1 - weight, weight])
        avg_mesh = app.tsdf_to_mesh([avg_tsdf])
        avg_mesh = avg_mesh[0]

        o3d_mesh.vertices = o3d.utility.Vector3dVector(avg_mesh['vertices'])
        o3d_mesh.triangles = o3d.utility.Vector3iVector(avg_mesh['triangles'])
        o3d_mesh.compute_vertex_normals()

        if i == 0:
            vis.add_geometry(o3d_mesh)
            vis.get_render_option().load_from_json("./render_options/vid_option.json")
            ctr.rotate(-150, 100)
            ctr.set_zoom(0.5)

        vis.update_geometry(o3d_mesh)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image("./output_frames/car_morphing_%04d.jpg" % i)

    vis.destroy_window()