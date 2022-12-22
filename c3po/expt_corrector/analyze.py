import csv
import os
import pickle
import random
import sys
import time
import torch
from datetime import datetime
from pytorch3d import ops, transforms

sys.path.append("../../")

from c3po.datasets.shapenet import SE3PointCloud, DepthPC
from c3po.datasets.shapenet import PCD_FOLDER_NAME as KEYPOINTNET_PCD_FOLDER_NAME, \
    CLASS_NAME as KEYPOINTNET_ID2NAME, \
    CLASS_ID as KEYPOINTNET_NAME2ID

from c3po.models.keypoint_corrector import kp_corrector_reg
from c3po.models.point_set_registration import PointSetRegistration
from c3po.models.certifiability import certifiability
from c3po.models.keypoint_corrector import keypoint_perturbation

from c3po.utils.ddn.node import ParamDeclarativeFunction
from c3po.utils.visualization_utils import display_two_pcs, visualize_torch_model_n_keypoints, \
    temp_expt_1_viz
from c3po.utils.evaluation_metrics import chamfer_dist, translation_error, rotation_error


def get_sq_distances(X, Y):
    """
    inputs:
    X   : torch.tensor of shape (B, 3, n)
    Y   : torch.tensor of shape (B, 3, m)

    outputs:
    sq_dist_xy  : torch.tensor of shape (B, n)  : for every point in X, the sq. distance to the closest point in Y
    sq_dist_yz  : torch.tensor of shape (B, m)  : for every point in Y, the sq. distance to the closest point in X
    """

    sq_dist_xy, _, _ = ops.knn_points(torch.transpose(X, -1, -2), torch.transpose(Y, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist_yx, yx_nn_idxs, _ = ops.knn_points(torch.transpose(Y, -1, -2), torch.transpose(X, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in Y to the nearest point in X

    return sq_dist_xy, sq_dist_yx, yx_nn_idxs


def get_kp_sq_distances(kp, kp_):
    sq_dist = ((kp-kp_)**2).sum(dim=1)
    return sq_dist #check output dimensions


class Experiment:
    def __init__(self, class_id, model_id, num_points, num_iterations, kp_noise_var_range,
                 kp_noise_fra=0.2,
                 certify=certifiability(epsilon=0.8, delta=0.5, radius=0.3),
                 theta=50.0, kappa=10.0, device='cpu', do_certification=False):
        super().__init__()

        # model parameters
        self.class_id = class_id
        self.model_id = model_id
        self.num_points = num_points

        # averaging over
        self.num_iterations = num_iterations

        # keypoint noise parameters
        self.kp_noise_fra = kp_noise_fra
        self.kp_noise_var_range = kp_noise_var_range

        # certification parameters
        self.certify = certify

        # loss function parameters
        self.theta = theta
        self.kappa = kappa

        # experiment name
        self.name = 'Analyzing keypoint corrector with simple registration on SE3PointCloud dataset'

        # device
        self.device_ = device

        self.do_certification = do_certification

        # setting up data
        # n is the desired number of points in the pcl,
        self.se3_dataset = DepthPC(class_id=self.class_id, model_id=self.model_id,
                                   num_of_points_to_sample=self.num_points, dataset_len=self.num_iterations)
        self.se3_dataset_loader = torch.utils.data.DataLoader(self.se3_dataset, batch_size=self.num_iterations, shuffle=False, num_workers=4)

        self.model_keypoints = self.se3_dataset._get_model_keypoints().to(device=self.device_)  # (1, 3, N)
        self.cad_models = self.se3_dataset._get_cad_models().to(device=self.device_)  # (1, 3, m)
        self.diameter = self.se3_dataset._get_diameter()

        # defining the point set registration
        self.point_set_registration = PointSetRegistration(source_points=self.model_keypoints)

        # defining the keypoint corrector
        corrector_node = kp_corrector_reg(cad_models=self.cad_models, model_keypoints=self.model_keypoints,
                                           theta=self.theta, kappa=self.kappa)
        self.corrector = ParamDeclarativeFunction(problem=corrector_node)

        # setting up experiment parameters and data for saving
        self.data = dict()
        self.parameters = dict()

        self.parameters['class_id'] = self.class_id
        self.parameters['model_id'] = self.model_id
        self.parameters['num_points'] = self.num_points
        self.parameters['num_iterations'] = self.num_iterations
        self.parameters['kp_noise_type'] = 'sporadic'
        self.parameters['kp_noise_fra'] = self.kp_noise_fra
        self.parameters['kp_noise_var_range'] = self.kp_noise_var_range
        self.parameters['certify'] = self.certify
        self.parameters['theta'] = self.theta
        self.parameters['kappa'] = self.kappa
        self.parameters['name'] = self.name

    def _single_loop(self, kp_noise_var, visualization=False):

        # experiment data
        rotation_err_naive = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        rotation_err_corrector = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        translation_err_naive = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        translation_err_corrector = torch.zeros(self.num_iterations, 1).to(device=self.device_)
        if self.do_certification:
            certi_naive = torch.zeros((self.num_iterations, 1), dtype=torch.bool).to(device=self.device_)
            certi_corrector = torch.zeros((self.num_iterations, 1), dtype=torch.bool).to(device=self.device_)

        sqdist_input_naiveest = []
        sqdist_input_correctorest = []

        sqdist_kp_naiveest = []
        sqdist_kp_correctorest = []

        pc_padding_masks = []

        chamfer_pose_naive_to_gt_pose_list = []
        chamfer_pose_corrected_to_gt_pose_list = []

        # experiment loop
        for i, data in enumerate(self.se3_dataset_loader):
            if i % 10 == 0:
                print("Testing at kp_noise_var: ", kp_noise_var, ". Iteration: ", i)

            # extracting data
            input_point_cloud, keypoints_true, rotation_true, translation_true = data
            input_point_cloud = input_point_cloud.to(device=self.device_)
            keypoints_true = keypoints_true.to(device=self.device_)
            rotation_true = rotation_true.to(device=self.device_)
            translation_true = translation_true.to(device=self.device_)

            # generating perturbed keypoints
            # keypoints_true = rotation_true @ self.model_keypoints + translation_true
            # detected_keypoints = keypoints_true
            detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true,
                                                       fra=self.kp_noise_fra, var=kp_noise_var*self.diameter)
            if visualization:
                temp_expt_1_viz(cad_models=input_point_cloud, model_keypoints=detected_keypoints, gt_keypoints = keypoints_true)
                # visualize_torch_model_n_keypoints(cad_models=input_point_cloud, model_keypoints=detected_keypoints)

            # estimate model: using point set registration on perturbed keypoints
            R_naive, t_naive = self.point_set_registration.forward(target_points=detected_keypoints)
            model_estimate_naive = R_naive @ self.cad_models + t_naive
            keypoint_estimate_naive = R_naive @ self.model_keypoints + t_naive
            if visualization:
                print("Displaying input and naive model estimate: ")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate_naive.squeeze(0))

            # estimate model: using the keypoint corrector
            correction = self.corrector.forward(detected_keypoints, input_point_cloud)
            # correction = torch.zeros_like(correction)
            R, t = self.point_set_registration.forward(target_points=detected_keypoints + correction)
            model_estimate = R @ self.cad_models + t
            keypoint_estimate = R @ self.model_keypoints + t
            if visualization:
                print("Displaying input and corrector model estimate: ")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate.squeeze(0))

            # evaluate the two metrics
            # rotation_err_naive[i] = rotation_error(rotation_true, R_naive)
            # rotation_err_corrector[i] = rotation_error(rotation_true, R)
            # translation_err_naive[i] = translation_error(translation_true, t_naive)
            # translation_err_corrector[i] = translation_error(translation_true, t)
            rotation_err_naive = rotation_error(rotation_true, R_naive)
            rotation_err_corrector = rotation_error(rotation_true, R)
            translation_err_naive = translation_error(translation_true, t_naive)
            translation_err_corrector = translation_error(translation_true, t)

            # saving sq distances for certification analysis
            sq_dist_input_naive = get_sq_distances(X=input_point_cloud, Y=model_estimate_naive)
            sq_dist_input_corrector = get_sq_distances(X=input_point_cloud, Y=model_estimate)
            sqdist_input_naiveest.append(sq_dist_input_naive)
            sqdist_input_correctorest.append(sq_dist_input_corrector)

            sq_dist_kp_naive = get_kp_sq_distances(kp = keypoint_estimate_naive, kp_=detected_keypoints)
            sq_dist_kp_corrector = get_kp_sq_distances(kp = keypoint_estimate, kp_ = detected_keypoints + correction)
            sqdist_kp_naiveest.append(sq_dist_kp_naive)
            sqdist_kp_correctorest.append(sq_dist_kp_corrector)

            pc_padding = ((input_point_cloud == torch.zeros(3, 1).to(device=self.device_)).sum(dim=1) == 3)
            pc_padding_masks.append(pc_padding)

            model_true = rotation_true @ self.cad_models + translation_true
            #save mean chamfer loss between model_estimate_naive and model_true
            #save mean chamfer loss between model_estimate and model_true
            chamfer_pose_naive_to_gt_pose = chamfer_dist(model_estimate_naive, model_true, max_loss=False)
            chamfer_pose_corrected_to_gt_pose = chamfer_dist(model_estimate, model_true, max_loss=False)
            chamfer_pose_naive_to_gt_pose_list.append(chamfer_pose_naive_to_gt_pose)
            chamfer_pose_corrected_to_gt_pose_list.append(chamfer_pose_corrected_to_gt_pose)

            # certification
            if self.do_certification:
                certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate_naive,
                                                kp=keypoint_estimate_naive, kp_=detected_keypoints)
                # certi_naive[i] = certi
                certi_naive = certi

                certi, _ = self.certify.forward(X=input_point_cloud, Z=model_estimate,
                                                kp=keypoint_estimate, kp_=detected_keypoints + correction)
                # certi_corrector[i] = certi
                certi_corrector = certi

            if visualization and i >= 5:
                break

        if self.do_certification:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
               certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest, sqdist_kp_naiveest, \
               sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, chamfer_pose_corrected_to_gt_pose_list
        else:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                   sqdist_input_naiveest, sqdist_input_correctorest, sqdist_kp_naiveest, \
                   sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, chamfer_pose_corrected_to_gt_pose_list

    def execute(self):

        rotation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        translation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        rotation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        translation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations).to(device=self.device_)
        if self.do_certification:
            certi_naive = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool).to(device=self.device_)
            certi_corrector = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool).to(device=self.device_)
        sqdist_input_naiveest = []
        sqdist_input_correctorest = []
        sqdist_kp_naiveest = []
        sqdist_kp_correctorest = []
        pc_padding_masks = []
        chamfer_pose_naive_to_gt_pose_list = []
        chamfer_pose_corrected_to_gt_pose_list = []

        for i, kp_noise_var in enumerate(self.kp_noise_var_range):

            print("-"*40)
            print("Testing at kp_noise_var: ", kp_noise_var)
            print("-"*40)

            start = time.perf_counter()
            if self.do_certification:
                Rerr_naive, Rerr_corrector, terr_naive, terr_corrector, \
                c_naive, c_corrector, sqdist_in_naive, sq_dist_in_corrector, sqdist_kp_naive, \
                sqdist_kp_corrector, pc_padding_mask,  chamfer_pose_naive_to_gt_pose, \
                chamfer_pose_corrected_to_gt_pose = self._single_loop(kp_noise_var=kp_noise_var)
            else:
                Rerr_naive, Rerr_corrector, terr_naive, terr_corrector, \
                sqdist_in_naive, sq_dist_in_corrector, sqdist_kp_naive, \
                sqdist_kp_corrector, pc_padding_mask, chamfer_pose_naive_to_gt_pose, \
                chamfer_pose_corrected_to_gt_pose = self._single_loop(kp_noise_var=kp_noise_var)

            end = time.perf_counter()
            print("Time taken: ", (end-start)/60, ' min')

            rotation_err_naive[i, ...] = Rerr_naive.squeeze(-1)
            rotation_err_corrector[i, ...] = Rerr_corrector.squeeze(-1)

            translation_err_naive[i, ...] = terr_naive.squeeze(-1)
            translation_err_corrector[i, ...] = terr_corrector.squeeze(-1)

            if self.do_certification:
                certi_naive[i, ...] = c_naive.squeeze(-1)
                certi_corrector[i, ...] = c_corrector.squeeze(-1)

            sqdist_input_naiveest.append(sqdist_in_naive)
            sqdist_input_correctorest.append(sq_dist_in_corrector)

            sqdist_kp_naiveest.append(sqdist_kp_naive)
            sqdist_kp_correctorest.append(sqdist_kp_corrector)

            pc_padding_masks.append(pc_padding_mask)

            chamfer_pose_naive_to_gt_pose_list.append(chamfer_pose_naive_to_gt_pose)
            chamfer_pose_corrected_to_gt_pose_list.append(chamfer_pose_corrected_to_gt_pose)

        self.data['rotation_err_naive'] = rotation_err_naive
        self.data['rotation_err_corrector'] = rotation_err_corrector
        self.data['translation_err_naive'] = translation_err_naive
        self.data['translation_err_corrector'] = translation_err_corrector
        if self.do_certification:
            self.data['certi_naive'] = certi_naive
            self.data['certi_corrector'] = certi_corrector
        self.data['sqdist_input_naiveest'] = sqdist_input_naiveest
        self.data['sqdist_input_correctorest'] = sqdist_input_correctorest
        self.data['sqdist_kp_naiveest'] = sqdist_kp_naiveest
        self.data['sqdist_kp_correctorest'] = sqdist_kp_correctorest
        self.data['pc_padding_masks'] = pc_padding_masks
        self.data['chamfer_pose_naive_to_gt_pose_list'] = chamfer_pose_naive_to_gt_pose_list
        self.data['chamfer_pose_corrected_to_gt_pose_list'] = chamfer_pose_corrected_to_gt_pose_list

        if self.do_certification:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                   certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest, \
                   sqdist_kp_naiveest, sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, \
                   chamfer_pose_corrected_to_gt_pose_list
        else:
            return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
                   sqdist_input_naiveest, sqdist_input_correctorest, \
                   sqdist_kp_naiveest, sqdist_kp_correctorest, pc_padding_masks, chamfer_pose_naive_to_gt_pose_list, \
                   chamfer_pose_corrected_to_gt_pose_list

    def execute_n_save(self):

        # execute the experiment
        self.execute()

        # saving the experiment and data
        location = './runs/' + str(self.class_id) + '/' + str(self.model_id) + '_wchamfer/'
        if not os.path.isdir(location):
            os.makedirs(location)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = timestamp + '_experiment.pickle'

        file = open(location + filename, 'wb')
        pickle.dump([self.parameters, self.data], file)
        file.close()

        return location + filename


def run_experiments_on(class_id, model_id, kp_noise_fra=0.2, only_visualize=False, do_certification=False):

    # model parameters
    num_points = 500

    # averaging over
    num_iterations = 100

    # kp_noise parameters
    kp_noise_var_range = torch.arange(0.1, 1.55, 0.1)

    # certification parameters
    epsilon = 0.995
    delta = 0.5
    radius = 0.01
    certify = certifiability(epsilon=epsilon, delta=delta, radius=radius)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is: ", device)

    # loss function parameters
    theta = 50.0
    kappa = 10.0

    print("-" * 40)
    print("Experiment: ")
    print("class_id: ", class_id)
    print("model_id: ", model_id)
    print("kp_noise_fra: ", kp_noise_fra)
    print("-" * 40)

    expt = Experiment(class_id=class_id, model_id=model_id, num_points=num_points,
                      num_iterations=num_iterations, kp_noise_var_range=kp_noise_var_range,
                      kp_noise_fra=kp_noise_fra,
                      certify=certify, theta=theta, kappa=kappa, device=device, do_certification=do_certification)

    if only_visualize:
        while True:
            kp_noise_var = float(input('Enter noise variance parameter: '))
            expt._single_loop(kp_noise_var=kp_noise_var, visualization=True)
            flag = input('Do you want to try another variance? (y/n): ')
            if flag == 'n':
                break
    else:
        filename = expt.execute_n_save()

        # experiment data
        expt = dict()
        expt['class_id'] = class_id
        expt['model_id'] = model_id
        expt['kp_noise_type'] = 'sporadic'
        expt['kp_noise_fra'] = kp_noise_fra
        expt['filename'] = filename
        expt['num_iterations'] = num_iterations

        expt_filename = './runs/' + str(class_id) + '/' + str(model_id) + '_wchamfer/' + 'experiments.csv'
        field_names = ['class_id', 'model_id', 'kp_noise_type', 'kp_noise_fra', 'filename', 'num_iterations']

        fp = open(expt_filename, 'a')
        dict_writer = csv.DictWriter(fp, field_names)
        dict_writer.writerow(expt)
        fp.close()


def choose_models(num_models=10, pcd_path = KEYPOINTNET_PCD_FOLDER_NAME, use_random=False):
    """
    For each class_id in pcd_path, choose num_models models randomly from each class.
    :param num_models: the number of models to sample from each class_id category
    :return: class_id_to_model_id_samples: dict: maps class_id to a list of sampled model_ids
    """
    class_id_to_model_id_samples = {}
    folder_contents = os.listdir(pcd_path)
    # hardcoded:
    if not use_random:
        return {'02691156': ['3db61220251b3c9de719b5362fe06bbb'],
                '02808440': ['90b6e958b359c1592ad490d4d7fae486'],
                '02818832': ['7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f'],
                '02876657': ['41a2005b595ae783be1868124d5ddbcb'],  # bottle
                '02954340': ['3dec0d851cba045fbf444790f25ea3db'],
                '02958343': ['ad45b2d40c7801ef2074a73831d8a3a2'],
                '03001627': ['1cc6f2ed3d684fa245f213b8994b4a04'],
                '03467517': ['5df08ba7af60e7bfe72db292d4e13056'],
                '03513137': ['3621cf047be0d1ae52fafb0cab311e6a'],
                '03624134': ['819e16fd120732f4609e2d916fa0da27'],
                '03642806': ['519e98268bee56dddbb1de10c9529bf7'],
                '03790512': ['481f7a57a12517e0fe1b9fad6c90c7bf'],
                '03797390': ['f3a7f8198cc50c225f5e789acd4d1122'],
                '04225987': ['98222a1e5f59f2098745e78dbc45802e'],
                '04379243': ['3f5daa8fe93b68fa87e2d08958d6900c'],
                '04530566': ['5c54100c798dd681bfeb646a8eadb57']
                }

    for class_id in folder_contents:
        models = os.listdir(pcd_path + str(class_id) + '/')

        # choose random num_models from models without replacement

        model_id_samples = random.sample(models, num_models)
        model_id_samples = [path[:-4] for path in model_id_samples]
        class_id_to_model_id_samples[class_id] = model_id_samples
    return class_id_to_model_id_samples


def run_full_experiment(kp_noise_fra=0.8, do_certification=False):
    class_id_to_model_id_samples = choose_models(num_models=1, use_random=False)
    for class_id, model_id_samples in class_id_to_model_id_samples.items():
        for model_id in model_id_samples:
            run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_fra=kp_noise_fra, do_certification=do_certification)


if __name__ == "__main__":
    run_full_experiment()

    # # model parameters
    # class_id = "03001627"  # chair
    # model_id = "1cc6f2ed3d684fa245f213b8994b4a04"  # a particular chair model

    # # # model parameters
    # class_id = "02876657"
    # model_id = "41a2005b595ae783be1868124d5ddbcb" # a particular bottle model
    #
    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_fra=0.2)
    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_fra=0.8)
    #

    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_fra=0.2,
    #                    only_visualize=True)
    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_fra=0.8,
    #                    only_visualize=True)
