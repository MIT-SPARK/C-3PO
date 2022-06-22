"""
This code will compare the compute time of corrector with algo='torch' and algo='scipy'.

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
import torch
import yaml
from matplotlib import colors as mcolors

sys.path.append("../../")

COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from learning_objects.datasets.keypointnet import DepthPC, CLASS_ID
from learning_objects.models.keypoint_corrector import kp_corrector_reg, keypoint_perturbation
from learning_objects.models.point_set_registration import PointSetRegistration
from learning_objects.utils.ddn.node import ParamDeclarativeFunction


EXPT_NAME = "corrector_time_analysis"

class Experiment:
    def __init__(self, class_id, model_id, num_points, batch_range,
                 num_iterations=10, kp_noise_var=0.6,
                 kp_noise_fra=0.8,
                 theta=50.0, kappa=10.0, device='gpu'):
        super().__init__()

        # experiment name
        self.name = EXPT_NAME

        # model parameters
        self.class_id = class_id
        self.model_id = model_id
        self.num_points = num_points

        # averaging over
        self.num_iterations = num_iterations

        # keypoint noise parameters
        self.kp_noise_fra = kp_noise_fra
        self.kp_noise_var = kp_noise_var

        # loss function parameters
        self.theta = theta
        self.kappa = kappa

        # batch range
        self.batch_range = batch_range

        # device
        if device == 'gpu':
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = 'cpu'

        # setting up experiment parameters and data for saving
        self.data = dict()
        self.parameters = dict()

        self.parameters['class_id'] = self.class_id
        self.parameters['model_id'] = self.model_id
        self.parameters['num_points'] = self.num_points
        self.parameters['num_iterations'] = self.num_iterations
        self.parameters['kp_noise_type'] = 'sporadic'
        self.parameters['kp_noise_fra'] = self.kp_noise_fra
        self.parameters['kp_noise_var'] = self.kp_noise_var
        self.parameters['theta'] = self.theta
        self.parameters['kappa'] = self.kappa
        self.parameters['name'] = self.name
        self.parameters['batch_range'] = self.batch_range

    def _single_loop(self, batch_size, algo):

        # setting up data
        # n is the desired number of points in the pcl,
        dataset = DepthPC(class_id=self.class_id, model_id=self.model_id,
                          num_of_points_to_sample=self.num_points, dataset_len=self.num_iterations * batch_size)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model_keypoints = dataset._get_model_keypoints().to(device=self.device_)  # (1, 3, N)
        cad_models = dataset._get_cad_models().to(device=self.device_)  # (1, 3, m)
        diameter = dataset._get_diameter()

        # defining the point set registration
        point_set_registration = PointSetRegistration(source_points=model_keypoints)

        # defining the keypoint corrector
        corrector_node = kp_corrector_reg(cad_models=cad_models, model_keypoints=model_keypoints,
                                          theta=self.theta, kappa=self.kappa, algo=algo)
        corrector = ParamDeclarativeFunction(problem=corrector_node)

        time_taken = 0.0

        # experiment loop
        for i, data in enumerate(dataset_loader):
            # This runs for self.num_iterations
            print("Algo: ",  algo, " :: Batch size: ", batch_size, ":: Iteration: ", i)

            # extracting data
            input_point_cloud, keypoints_true, rotation_true, translation_true = data
            input_point_cloud = input_point_cloud.to(device=self.device_)
            keypoints_true = keypoints_true.to(device=self.device_)
            rotation_true = rotation_true.to(device=self.device_)
            translation_true = translation_true.to(device=self.device_)

            detected_keypoints = keypoint_perturbation(keypoints_true, self.kp_noise_var, self.kp_noise_fra).to(device=self.device_)

            # estimate model: using the keypoint corrector

            torch.cuda.synchronize()
            # start = time.perf_counter()
            start = time.process_time_ns()
            #start = time.time_ns()

            correction = corrector.forward(detected_keypoints, input_point_cloud)

            torch.cuda.synchronize()
            # end = time.perf_counter()
            end = time.process_time_ns()
            
            #end = time.time_ns()
            # correction = torch.zeros_like(correction)
            R, t = point_set_registration.forward(target_points=detected_keypoints + correction)
            #model_estimate = R @ cad_models + t
            #keypoint_estimate = R @ model_keypoints + t

            # corrector time
            time_taken += (end - start)

        time_taken = time_taken / (batch_size * (i + 1))
        time_taken = time_taken / 1000      # to get time in msec

        return time_taken

    def execute(self):

        time_algo_torch = []
        time_algo_scipy = []

        for b in self.batch_range:

            print("Analyzing Compute Times for Batch Size: ", b)

            time_taken_torch = self._single_loop(batch_size=b, algo='torch')
            time_algo_torch.append(time_taken_torch)

            time_taken_scipy = self._single_loop(batch_size=b, algo='scipy')
            time_algo_scipy.append(time_taken_scipy)

            # saving the experiment and data
            location = './runs/' + str(self.class_id) + '/' + str(self.model_id) + '/'
            if not os.path.isdir(location):
                os.makedirs(location)

            filemane = self.name + 'batch_size_' + str(b) + '.pickle'

            file = open(location + filemane, 'wb')
            data = {'time_algo_torch': time_taken_torch,
                    'time_algo_scipy': time_taken_scipy}
            pickle.dump(data, file)
            file.close()

        return time_algo_torch, time_algo_scipy

    def execute_n_save(self):

        # execute the experiment
        compute_times = self.execute()
        self.data = {'time_algo_torch': compute_times[0],
                     'time_algo_scipy': compute_times[1]}

        # saving the experiment and data
        location = './runs/' + str(self.class_id) + '/' + str(self.model_id) + '/'
        if not os.path.isdir(location):
            os.makedirs(location)

        filemane = self.name + '_.pickle'

        file = open(location + filemane, 'wb')
        pickle.dump([self.parameters, self.data], file)
        file.close()

        return None


def run_experiments_on(class_id, model_id):

    # model parameters
    num_points = 500

    # averaging over
    num_iterations = 10

    # kp_noise parameters
    kp_noise_fra = 0.8
    kp_noise_var = 0.6

    # loss function parameters
    theta = 50.0
    kappa = 10.0

    # batch range
    # batch_range = [1, 5, 10]
    batch_range = [1, 5, 10, 25, 50, 75, 100, 150, 200]
    # batch_range = [1, 5, 10, 100, 500]

    print("-" * 40)
    print("Experiment: ")
    print("class_id: ", class_id)
    print("model_id: ", model_id)
    print("kp_noise_fra: ", kp_noise_fra)
    print("-" * 40)

    expt = Experiment(class_id=class_id, model_id=model_id, num_points=num_points,
                      batch_range=batch_range,
                      num_iterations=num_iterations, kp_noise_var=kp_noise_var,
                      kp_noise_fra=kp_noise_fra,
                      theta=theta, kappa=kappa, device='gpu')

    expt.execute_n_save()


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
    # plt.show()
    filename = 'plot.pdf'
    fig.savefig(location + filename)
    plt.close(fig)


if __name__ == "__main__":

    class_name = 'chair'

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    class_id = CLASS_ID[class_name]
    model_id = model_class_ids[class_name]

    run_experiments_on(class_id, model_id)
    analyze_results(class_id, model_id)

