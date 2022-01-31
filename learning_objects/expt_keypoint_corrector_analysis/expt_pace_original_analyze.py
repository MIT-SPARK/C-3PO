"""
This code analyzes the data generated by the experiment expt_with_pace_se3isopc.py

"""
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from datetime import datetime
plt.style.use('seaborn-whitegrid')

import os
import sys
sys.path.append("../../")
from learning_objects.utils.general import generate_filename
from learning_objects.datasets.keypointnet import CLASS_NAME

COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


if __name__ == '__main__':

    file_names = ["./expt_pace_original/20220127_202653_experiment.pickle"]

    for name in file_names:

        fp = open(name, 'rb')
        parameters, data = pickle.load(fp)
        fp.close()

        print("-" * 80)

        Rerr_naive = data['rotation_err_naive']
        terr_naive = data['translation_err_naive']
        shape_err_naive = data['shape_err_naive']
        shape_true_1 = data['shape_true_1']#.transpose(0,1)
        print(shape_true_1.shape)
        shape_true_2 = data['shape_true_2']#.transpose(0,1)
        shape_pace_1 = data['shape_pace_1']#.transpose(0,1)
        shape_pace_2 = data['shape_pace_2']#.transpose(0,1)
        print(shape_true_1)
        print(shape_true_2)


        class_id = parameters['class_id']
        model_id = parameters['model_id']
        cad_model_name = CLASS_NAME[class_id]

        fig = plt.figure()
        true_color = [(0, str(item / len(shape_true_1)), 0) for item in range(len(shape_true_1))]
        pace_color = [(str(item / len(shape_true_1)), 0, 0) for item in range(len(shape_true_1))]
        plt.scatter(x=shape_true_1, y=shape_true_2, s=20.0, label='shape_true', c='green')
        plt.scatter(x=shape_pace_1, y=shape_pace_2, s=20.0, label='shape_pace', c='red')
        plt.xlabel('ratio shape 1')
        plt.ylabel('ratio shape 2')
        plt.title("Category: " + str(CLASS_NAME[class_id]) + ", Model: " + str(model_id))
        plt.legend(loc='upper right')


        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_pace_shape_plot_' + timestamp + '_' + rand_string + '.jpg'
        fig.savefig(filename)
        plt.close(fig)
        #
        # fig = plt.figure()
        # pace_color = [(0, str(item / len(shape_true_1)), 0) for item in range(len(shape_true_1))]
        # plt.scatter(x=shape_pace_1, y=shape_pace_2, s=20.0, label='shape_pace', c=pace_color)
        #
        # plt.show()
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # rand_string = generate_filename()
        # # filename = name[:-7] + '_rotation_error_plot_' + timestamp + '_' + rand_string + '.jpg'
        # # fig.savefig(filename)
        # plt.close(fig)



