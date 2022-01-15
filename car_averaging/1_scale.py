import os
import pickle
import csv
import sys

import common
import argparse
import numpy as np
import meshlab_pickedpoints


TOTAL_NUM_KPTS = 66

class Scale:
    """
    Scales a bunch of meshes.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--kpt_in_dir', type=str, help='Path to input keypoint directory to scale points')
        parser.add_argument('--kpt_out_dir', type=str, help='Path to output keypoint directory of scaled points')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
        parser.add_argument('--transform_out_dir', type=str,
                            help='Path to scale & translation parameters output directory; will overwrite files!')
        parser.add_argument('--use_max_scale', type=bool, default=True,
                            help='Set True to find the largest scale possible first, then use this scale to scale every model.')
        parser.add_argument('--padding', type=float, default=0.1, help='Relative padding applied on each side.')
        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def load_kpt_lib(self, dir_root):
        """Load directory of PP files

        :param dir_root:
        :return:
        """
        kpt_db = []
        avaialble_cads = set()
        # load keypoints - vehicle dicts only for cars with full number of keypoints
        for file in sorted(os.listdir(dir_root)):
            if file.endswith(".pp"):
                with open(os.path.join(dir_root, file), 'r') as f:
                    pts = meshlab_pickedpoints.load(f)
                    car_name = file.split(".")[0]
                    flag = len(pts) == TOTAL_NUM_KPTS
                    if flag:
                        kpt_list = [None] * TOTAL_NUM_KPTS
                        for kpt, pt in pts.items():
                            kpt_list[int(kpt)] = pt
                            if pt[0] == 0 and pt[1] == 0:
                                flag = False
                                break
                    else:
                        print("WARNING: {} missing points".format(file))
                    if flag:
                        kpt_db.append({'kpts': np.transpose(np.array(kpt_list)), 'name': car_name})
                        avaialble_cads.add(car_name)
            else:
                print("WARNING: {} unknown file".format(file))
        return kpt_db, avaialble_cads

    def load_all_kpts(self):
        """
        :return: cad_db_dict where keys are model names and values are a list of xyz keypoints
        """
        cad_db, available_cads = self.load_kpt_lib(self.options.kpt_in_dir)
        cad_db_dict = {x["name"]: x["kpts"] for x in cad_db}
        for name, kpts in cad_db_dict.items():
            keypoints_xyz = np.transpose(cad_db_dict[name][:3, :])
            cad_db_dict[name] = keypoints_xyz
        return cad_db_dict

    def run(self):
        """
        Run the tool, i.e. scale all found OFF files.
        """

        assert os.path.exists(self.options.in_dir)
        common.makedir(self.options.out_dir)
        #lisa added
        common.makedir(self.options.transform_out_dir)
        common.makedir(self.options.kpt_out_dir)
        files = self.read_directory(self.options.in_dir)

        cad_kpt_dict = self.load_all_kpts()

        max_size = None
        if self.options.use_max_scale:
            max_size = 0
            for filepath in files:
                mesh = common.Mesh.from_off(filepath)
                # Get extents of model.
                min, max = mesh.extents()
                total_min = np.min(np.array(min))
                total_max = np.max(np.array(max))
                c_size = total_max.item() - total_min.item(),
                if c_size[0] > max_size:
                    max_size = c_size[0]

        for filepath in files:
            mesh = common.Mesh.from_off(filepath)

            model_name = os.path.basename(filepath).split(".")[0]
            print(model_name)
            keypoints_xyz = cad_kpt_dict[model_name]

            # Get extents of model.
            min, max = mesh.extents()
            total_min = np.min(np.array(min))
            total_max = np.max(np.array(max))

            # Set the center (although this should usually be the origin already).
            centers = (
                (min[0] + max[0]) / 2,
                (min[1] + max[1]) / 2,
                (min[2] + max[2]) / 2
            )
            # Scales all dimensions equally.
            if self.options.use_max_scale:
                sizes = (max_size, max_size, max_size)
            else:
                sizes = (
                    total_max - total_min,
                    total_max - total_min,
                    total_max - total_min
                )
            translation = (
                -centers[0],
                -centers[1],
                -centers[2]
            )
            scales = (
                1 / (sizes[0] + 2 * self.options.padding * sizes[0]),
                1 / (sizes[1] + 2 * self.options.padding * sizes[1]),
                1 / (sizes[2] + 2 * self.options.padding * sizes[2])
            )
            #translate keypoints
            assert len(translation) == 3
            for i in range(3):
                keypoints_xyz[:, i] += translation[i]
            #scale keypoints
            assert len(scales) == 3
            for i in range(3):
                keypoints_xyz[:, i] *= scales[i]

            mesh.translate(translation)
            mesh.scale(scales)

            print('[Data] %s extents before %f - %f, %f - %f, %f - %f' % (
                os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))
            min, max = mesh.extents()
            print('[Data] %s extents after %f - %f, %f - %f, %f - %f' % (
                os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

            # May also switch axes if necessary.
            # mesh.switch_axes(1, 2)

            mesh.to_off(os.path.join(self.options.out_dir, os.path.basename(filepath)))

            #dump the scaled and transformed keypoints
            kpt_dump_path = os.path.join(self.options.kpt_out_dir,
                                         model_name + ".csv")
            with open(kpt_dump_path, 'w') as f:
                field_names = ['x', 'y', 'z']
                writer = csv.writer(f)
                writer.writerow(field_names)
                writer.writerows(keypoints_xyz)

            # dump the scale and translation to csv files
            print("filepath", filepath)
            print("self.options.transform_out_dir", self.options.transform_out_dir)
            tr_dump_path = os.path.join(self.options.transform_out_dir,
                                        os.path.basename(filepath).split(".")[0] + ".csv")
            with open(tr_dump_path, 'w') as f:
                field_names = ['scale_x', 'scale_y', 'scale_z', 'translation_x', 'translation_y', 'translation_z']
                c_dict = {'scale_x': scales[0], 'scale_y': scales[1], 'scale_z': scales[2],
                          'translation_x': translation[0], 'translation_y': translation[1],
                          'translation_z': translation[2]}

                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
                writer.writerow(c_dict)


if __name__ == '__main__':
    app = Scale()
    app.run()

