import os
import pickle
import csv

from tqdm import tqdm
import common
import argparse
import numpy as np


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

        parser = argparse.ArgumentParser(description='Revert the scaling and translation applied')
        parser.add_argument('--in_models_dir', type=str, help='Path to input models directory.')
        parser.add_argument('--in_transform_params_dir', type=str,
                            help='Path to directory containing scaling and translation parameters.')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
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

    def run(self):
        """
        Run the tool, i.e. scale all found OFF files.
        """

        assert os.path.exists(self.options.in_models_dir)
        common.makedir(self.options.out_dir)
        files = self.read_directory(self.options.in_models_dir)

        for filepath in tqdm(files):
            # load transform parameters
            car_name = os.path.basename(filepath).split(".")[0]
            param_path = os.path.join(self.options.in_transform_params_dir, car_name + ".csv")
            scales = None
            translation = None

            with open(param_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scales = (float(row['scale_x']), float(row['scale_y']), float(row['scale_z']))
                    translation = (
                        float(row['translation_x']), float(row['translation_y']), float(row['translation_z']))

            mesh = common.Mesh.from_off(filepath)

            # rotation fix
            # (note: this is a rotation + reflection)
            # 90 deg around y, then a reflection across yz (x->-x)
            R = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            for i in range(mesh.vertices.shape[0]):
                mesh.vertices[i, :] = np.transpose(R @ mesh.vertices[i, :].T)

            # descaling and translate back
            for i in range(3):
                mesh.vertices[:, i] /= scales[i]
            for i in range(3):
                mesh.vertices[:, i] -= translation[i]

            mesh.to_off(os.path.join(self.options.out_dir, os.path.basename(filepath)))


if __name__ == '__main__':
    app = Scale()
    app.run()
