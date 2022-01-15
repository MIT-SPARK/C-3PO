import os, sys
import open3d as o3d

if __name__ == "__main__":
    print("Converting .ply model files to .off files")
    ply_dir = "../car_models/"
    off_dir = "./car_models/0_in"
    ply_files = [f for f in os.listdir(ply_dir) if os.path.isfile(os.path.join(ply_dir, f))]
    for ply_fname in ply_files:
        ply_path = os.path.join(ply_dir, ply_fname)
        mesh_data = o3d.io.read_triangle_mesh(ply_path)

        off_fname = ply_fname.split(".")[0] + ".off"
        off_path = os.path.join(off_dir, off_fname)
        o3d.io.write_triangle_mesh(off_path, mesh_data)
