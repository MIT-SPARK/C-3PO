"""
This code will compare the compute time of corrector with algo='torch' and algo='scipy'.

"""
import matplotlib.pyplot as plt
import numpy as np
# import os
import pickle
import sys
# import time
# import torch
import yaml
# from tqdm import tqdm
# from matplotlib import colors as mcolors

from analyze import EXPT_NAME, DEFAULT_OBJECT

sys.path.append("../../")
from c3po.datasets.shapenet import CLASS_ID


def analyze_results(class_id, model_id):

    location = './runs/' + str(class_id) + '/' + str(model_id) + '/'
    filemane = EXPT_NAME + '_.pickle'
    file = open(location + filemane, 'rb')
    parameters, data = pickle.load(file)

    # print(data)
    # print(data['time_algo_torch'])
    # print(data['time_algo_scipy'])
    # print(parameters['batch_range'])
    batch_range = np.asarray(parameters['batch_range'][0:-1])
    time_algo_torch = np.asarray(data['time_algo_torch'][0:-1]) / 1000000
    time_algo_scipy = np.asarray(data['time_algo_scipy'][0:-1]) / 1000000

    fig = plt.figure()
    plt.plot(batch_range, time_algo_torch, 'o--',
             label='batch gradient descent', color='orangered')
    plt.plot(batch_range, time_algo_scipy, 'o--',
             label='non batch: trust region', color='grey')
    plt.xlabel("Batch size")
    plt.ylabel("Compute time per input (sec)")
    plt.legend(loc='upper left')
    plt.xlim([batch_range[0], batch_range[-1]])
    plt.show()
    # filename = 'plot.pdf'
    # fig.savefig(location + filename)
    # plt.close(fig)


if __name__ == "__main__":

    class_name = DEFAULT_OBJECT

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    class_id = CLASS_ID[class_name]
    model_id = model_class_ids[class_name]

    analyze_results(class_id, model_id)