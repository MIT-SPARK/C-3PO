
import torch
from pytorch3d import ops

from datetime import datetime
import pickle
import csv

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud

from learning_objects.models.keypoint_corrector import kp_corrector_reg
from learning_objects.models.keypoint_corrector import keypoint_perturbation, registration_eval, display_two_pcs
from learning_objects.models.point_set_registration import point_set_registration
from learning_objects.models.certifiability import certifiability

from learning_objects.utils.general import translation_error, rotation_error


def get_sq_distances(X, Y):
    """
    inputs:
    X   : torch.tensor of shape (B, 3, n)
    Y   : torch.tensor of shape (B, 3, m)

    outputs:
    sq_dist_xy  : torch.tensor of shape (B, n)  : for every point in X, the sq. distance to the closest point in Y
    sq_dist_yz  : torch.tensor of shape (B, m)  : for every point in Y, the sq. distance to the closest point in X
    """

    sq_dist_xy, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist_yx, _, _ = ops.knn_points(torch.transpose(Y, -1, -2), torch.transpose(X, -1, -2), K=1)
    # dist (B, n, 1): distance from point in Y to the nearest point in X

    return sq_dist_xy, sq_dist_yx


class experiment():
    def __init__(self, class_id, model_id, num_points, num_iterations, kp_noise_var_range,
                 kp_noise_type='sporadic', kp_noise_fra=0.2,
                 certify=certifiability(epsilon=0.8, delta=0.5, radius=0.3),
                 theta=50.0, kappa=10.0):
        super().__init__()

        # model parameters
        self.class_id = class_id
        self.model_id = model_id
        self.num_points = num_points

        # averaging over
        self.num_iterations = num_iterations

        # keypoint noise parameters
        self.kp_noise_type = kp_noise_type
        self.kp_noise_fra = kp_noise_fra
        self.kp_noise_var_range = kp_noise_var_range

        # certification parameters
        self.certify = certify

        # loss function parameters
        self.theta = theta
        self.kappa = kappa

        # experiment name
        self.name = 'Analyzing keypoint corrector with simple registration on DepthPointCloud dataset'


        # setting up data
        self.se3_dataset = SE3PointCloud(class_id=self.class_id, model_id=self.model_id, num_of_points=self.num_points,
                                    dataset_len=self.num_iterations)
        self.se3_dataset_loader = torch.utils.data.DataLoader(self.se3_dataset, batch_size=1, shuffle=False)

        self.model_keypoints = self.se3_dataset._get_model_keypoints()  # (1, 3, N)
        self.cad_models = self.se3_dataset._get_cad_models()  # (1, 3, m)

        # defining the keypoint corrector
        self.corrector = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints,
                                     theta=self.theta, kappa=self.kappa)

        # setting up experiment parameters and data for saving
        self.data = dict()
        self.parameters = dict()

        self.parameters['class_id'] = self.class_id
        self.parameters['model_id'] = self.model_id
        self.parameters['num_points'] = self.num_points
        self.parameters['num_iterations'] = self.num_iterations
        self.parameters['kp_noise_type'] = self.kp_noise_type
        self.parameters['kp_noise_fra'] = self.kp_noise_fra
        self.parameters['kp_noise_var_range'] = self.kp_noise_var_range
        self.parameters['certify'] = self.certify
        self.parameters['theta'] = self.theta
        self.parameters['kappa'] = self.kappa
        self.parameters['name'] = self.name


    def _single_loop(self, kp_noise_var):

        # experiment data
        rotation_err_naive = torch.zeros(self.num_iterations, 1)
        rotation_err_corrector = torch.zeros(self.num_iterations, 1)
        translation_err_naive = torch.zeros(self.num_iterations, 1)
        translation_err_corrector = torch.zeros(self.num_iterations, 1)

        certi_naive = torch.zeros((self.num_iterations, 1), dtype=torch.bool)
        certi_corrector = torch.zeros((self.num_iterations, 1), dtype=torch.bool)

        sqdist_input_naiveest = []
        sqdist_input_correctorest = []

        # experiment loop
        for i, data in enumerate(self.se3_dataset_loader):

            print("Testing at kp_noise_var: ", kp_noise_var, ". Iteration: ", i)

            # extracting data
            input_point_cloud, rotation_true, translation_true = data

            # generating perturbed keypoints
            keypoints_true = rotation_true @ self.model_keypoints + translation_true
            # detected_keypoints = keypoints_true
            detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type=self.kp_noise_type,
                                                       fra=self.kp_noise_fra, var=kp_noise_var)

            # estimate model: using point set registration on perturbed keypoints
            R_naive, t_naive = point_set_registration(source_points=self.model_keypoints, target_points=detected_keypoints)
            model_estimate_naive = R_naive @ self.cad_models + t_naive
            # display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate.squeeze(0))

            # estimate model: using the keypoint corrector
            correction = self.corrector.forward(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud)
            # correction = torch.zeros_like(correction)
            R, t = point_set_registration(source_points=self.model_keypoints, target_points=detected_keypoints + correction)
            model_estimate = R @ self.cad_models + t
            # display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate.squeeze(0))

            # evaluate the two metrics
            rotation_err_naive[i] = rotation_error(rotation_true, R_naive)
            rotation_err_corrector[i] = rotation_error(rotation_true, R)
            translation_err_naive[i] = translation_error(translation_true, t_naive)
            translation_err_corrector[i] = translation_error(translation_true, t)

            # saving sq distances for certification analysis
            sq_dist_input_naive = get_sq_distances(X=input_point_cloud, Y=model_estimate_naive)
            sq_dist_input_corrector = get_sq_distances(X=input_point_cloud, Y=model_estimate)
            sqdist_input_naiveest.append(sq_dist_input_naive)
            sqdist_input_correctorest.append(sq_dist_input_corrector)

            # certification
            certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate_naive)
            certi_naive[i] = certi

            certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate)
            certi_corrector[i] = certi

        return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
               certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest



    def execute(self):

        rotation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        translation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        rotation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        translation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        certi_naive = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool)
        certi_corrector = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool)
        sqdist_input_naiveest = []
        sqdist_input_correctorest = []

        for i, kp_noise_var in enumerate(self.kp_noise_var_range):

            print("-"*40)
            print("Testing at kp_noise_var: ", kp_noise_var)
            print("-"*40)

            Rerr_naive, Rerr_corrector, terr_naive, terr_corrector, \
            c_naive, c_corrector, sqdist_in_naive, sq_dist_in_corrector = self._single_loop(kp_noise_var=kp_noise_var)

            rotation_err_naive[i, ...] = Rerr_naive.squeeze(-1)
            rotation_err_corrector[i, ...] = Rerr_corrector.squeeze(-1)

            translation_err_naive[i, ...] = terr_naive.squeeze(-1)
            translation_err_corrector[i, ...] = terr_corrector.squeeze(-1)

            certi_naive[i, ...] = c_naive.squeeze(-1)
            certi_corrector[i, ...] = c_corrector.squeeze(-1)

            sqdist_input_naiveest.append(sqdist_in_naive)
            sqdist_input_correctorest.append(sq_dist_in_corrector)


        self.data['rotation_err_naive'] = rotation_err_naive
        self.data['rotation_err_corrector'] = rotation_err_corrector
        self.data['translation_err_naive'] = translation_err_naive
        self.data['translation_err_corrector'] = translation_err_corrector
        self.data['certi_naive'] = certi_naive
        self.data['certi_corrector'] = certi_corrector
        self.data['sqdist_input_naiveest'] = sqdist_input_naiveest
        self.data['sqdist_input_correctorest'] = sqdist_input_correctorest

        return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
               certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest

    def execute_n_save(self):

        # execute the experiment
        self.execute()

        # saving the experiment and data
        location = './expt_with_reg_se3pc/'
        if not os.path.isdir(location):
            os.mkdir(location)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filemane = timestamp + '_experiment.pickle'

        file = open(location + filemane, 'wb')
        pickle.dump([self.parameters, self.data], file)
        file.close()

        return location + filemane


def run_experiments_on(class_id, model_id, kp_noise_type, kp_noise_fra=0.2):

    # model parameters
    num_points = 500

    # averaging over
    num_iterations = 100

    # kp_noise parameters
    kp_noise_var_range = torch.arange(0.1, 0.9, 0.1)

    # certification parameters
    epsilon = 0.98
    delta = 0.98
    radius = 0.1
    certify = certifiability(epsilon=epsilon, delta=delta, radius=radius)

    # loss function parameters
    theta = 50.0
    kappa = 10.0

    print("-" * 40)
    print("Experiment: ")
    print("class_id: ", class_id)
    print("model_id: ", model_id)
    print("kp_noise_type: ", kp_noise_type)
    print("kp_noise_fra: ", kp_noise_fra)
    print("-" * 40)

    expt = experiment(class_id=class_id, model_id=model_id, num_points=num_points,
                      num_iterations=num_iterations, kp_noise_var_range=kp_noise_var_range,
                      kp_noise_type=kp_noise_type, kp_noise_fra=kp_noise_fra,
                      certify=certify, theta=theta, kappa=kappa)
    filename = expt.execute_n_save()

    # experiment data
    expt = dict()
    expt['class_id'] = class_id
    expt['model_id'] = model_id
    expt['kp_noise_type'] = kp_noise_type
    expt['kp_noise_fra'] = kp_noise_fra
    expt['filename'] = filename

    expt_filename = './expt_with_reg_depthpc/experiments.csv'
    field_names = ['class_id', 'model_id', 'kp_noise_type', 'kp_noise_fra', 'filename']

    fp = open(expt_filename, 'a')
    dict_writer = csv.DictWriter(fp, field_names)
    dict_writer.writerow(expt)
    fp.close()



if __name__ == "__main__":


    # model parameters
    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='sporadic', kp_noise_fra=0.2)
    run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='sporadic', kp_noise_fra=0.8)
    run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='uniform')



