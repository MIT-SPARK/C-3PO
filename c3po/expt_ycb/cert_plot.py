
import argparse
import io
import matplotlib.pyplot as plt
import os
import pickle
import sys
import torch
import yaml
from matplotlib import colors as mcolors

plt.style.use('seaborn-whitegrid')


sys.path.append("../..")
from c3po.datasets.shapenet import CLASS_NAME, CLASS_ID

COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        if module == 'learning_objects.utils.general':
            return super().find_class('c3po.utils.general', name)
        else:
            return super().find_class(module, name)


def cert_at_train(detector_type, model_id):

    base_folder = './'
    data_folder = base_folder + '/' + model_id + '/'
    file_name = data_folder + '_certi_all_batches_' + detector_type + '.pkl'

    with open(file_name, 'rb') as f:
        # x = pickle.load(f)
        # x = torch.load(f, map_location=torch.device('cpu'))
        x = CPU_Unpickler(f).load()

    return x


def plot_cert_at_train(model_ids, only_categories):

    detector_type = ["point_transformer"]
    base_folder = './cert_plots/'
    only_one = False

    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    if only_categories[0] == only_categories[1]:
        model_id = only_categories[0]

        fig_save_file_name = base_folder + "cert_plot_" + model_id + ".jpg"
        only_categories = [only_categories[0]]
        only_one=True
    else:
        fig_save_file_name = base_folder + "cert_plot"
        for model_id in only_categories:

            fig_save_file_name = fig_save_file_name + "_" + model_id

        fig_save_file_name = fig_save_file_name + ".jpg"

    plot_dict = dict()
    index = 0
    for model_id in model_ids:
        if model_id not in only_categories:
            continue

        len_max = 0.0
        for detector in detector_type:
            # print(detector)
            x = cert_at_train(detector_type=detector, model_id=model_id)
            # if detector == "point_transformer":
            #     print(x)
            x = x.val
            # print(x)
            # print("x[0]", x[0])

            if isinstance(x[0], list):
                x_flat = [u for sublist in x for u in sublist]
            else:
                x_flat = x
            # print(x_flat)
            len_max = max(len_max, len(x_flat))
            plot_dict[detector] = 100 * torch.tensor(x_flat)

        for detector in detector_type:
            if len(plot_dict[detector]) < len_max:
                z = torch.ones(len_max-len(plot_dict[detector]))
                z = z * torch.tensor([float("inf")])
                plot_dict[detector] = torch.cat([plot_dict[detector], z])

        if index == 0:
            fig = plt.figure()
            iter_range = torch.arange(len_max)
            # plt.plot(iter_range, plot_dict["pointnet"], '-', label=key + ': pointnet', color='grey')
            plt.plot(iter_range, plot_dict["point_transformer"], '-', label=model_id + ': point transformer', color='orangered')
            plt.xlabel('Number of SGD iterations')
            plt.ylabel('Percent certifiable')
            if only_one:
                plt.legend(loc="lower right")
                plt.show()
                fig.savefig(fig_save_file_name)
                plt.close(fig)
        if index == 1:
            # fig = plt.figure()
            # iter_range = torch.arange(len_max)
            # plt.plot(iter_range, plot_dict["pointnet"], '--', label=key + ': pointnet', color='grey')
            plt.plot(iter_range, plot_dict["point_transformer"], '--', label=model_id + ': point transformer', color='orangered')
            # plt.xlabel('number of SGD iterations')
            # plt.ylabel('% certifiable')
            plt.legend(loc="lower right")
            plt.show()
            fig.savefig(fig_save_file_name)
            plt.close(fig)

        index += 1


if __name__ == "__main__":
    """
    usage: 
    >> python cert_plot.py "chair"
    >> python cert_plot.py" "airplane"
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id1", help="specify the ycb class name.", type=str)
    parser.add_argument("model_id2", help="specify the ycb class name.", type=str)

    args = parser.parse_args()

    # class_name = args.class_name
    only_categories = [args.model_id1, args.model_id2]

    stream = open("model_ids.yml", "r")
    model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']

    plot_cert_at_train(model_ids=model_ids, only_categories=only_categories)