
import torch
from pytorch3d import ops, transforms

from datetime import datetime
import pickle
import csv

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3nIsotorpicShapePointCloud, visualize_torch_model_n_keypoints, \
    DepthAndIsotorpicShapePointCloud

from learning_objects.models.keypoint_corrector import kp_corrector_reg, kp_corrector_pace
from learning_objects.models.point_set_registration import point_set_registration
# from learning_objects.models.pace_ddn import PACEbp
from learning_objects.models.pace_altern_ddn import PACEbp
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.models.certifiability import certifiability

from learning_objects.utils.general import display_two_pcs

#ToDo: This code works, but is still using solve_algo1() in kp_corrector_pace. This is because autograd backprop()
# is not imlemented for PACEbp.But, this code now uses PACEbp from pace_altern_ddn.

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


def keypoint_perturbation(keypoints_true, var=0.8, type='uniform', fra=0.2):
    """
    inputs:
    keypoints_true  :  torch.tensor of shape (B, 3, N)
    var             :  float
    type            : 'uniform' or 'sporadic'
    fra             :  float    : used if type == 'sporadic'

    output:
    detected_keypoints  : torch.tensor of shape (B, 3, N)
    """

    if type=='uniform':
        # detected_keypoints = keypoints_true + var*torch.randn_like(keypoints_true)
        detected_keypoints = keypoints_true + var * (torch.rand(size=keypoints_true.shape) - 0.5)

    elif type=='sporadic':
        mask = (torch.rand(size=keypoints_true.shape) < fra).int().float()
        # detected_keypoints = keypoints_true + var*torch.randn_like(keypoints_true)*mask
        detected_keypoints = keypoints_true + var * (torch.rand(size=keypoints_true.shape) - 0.5) * mask
    else:
        return ValueError

    return detected_keypoints


def shape_error(c, c_):
    """
    inputs:
    c: torch.tensor of shape (K, 1) or (B, K, 1)
    c_: torch.tensor of shape (K, 1) or (B, K, 1)

    output:
    c_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if c.dim() == 2:
        return torch.norm(c - c_, p=2)/c.shape[0]
    elif c.dim() == 3:
        return torch.norm(c - c_, p=2, dim=1)/c.shape[1]
    else:
        return ValueError


def translation_error(t, t_):
    """
    inputs:
    t: torch.tensor of shape (3, 1) or (B, 3, 1)
    t_: torch.tensor of shape (3, 1) or (B, 3, 1)

    output:
    t_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if t.dim() == 2:
        return torch.norm(t - t_, p=2)/3.0
    elif t.dim() == 3:
        return torch.norm(t-t_, p=2, dim=1)/3.0
    else:
        return ValueError


def rotation_error(R, R_):
    """
    inputs:
    R: torch.tensor of shape (3, 3) or (B, 3, 3)
    R_: torch.tensor of shape (3, 3) or (B, 3, 3)

    output:
    R_err: torch.tensor of shape (1, 1) or (B, 1)
    """

    if R.dim() == 2:
        return transforms.matrix_to_euler_angles(torch.matmul(R.T, R_), "XYZ").abs().sum()/3.0
        # return torch.abs(0.5*(torch.trace(R.T @ R_) - 1).unsqueeze(-1))
        # return 1 - 0.5*(torch.trace(R.T @ R_) - 1).unsqueeze(-1)
        # return torch.norm(R.T @ R_ - torch.eye(3, device=R.device), p='fro')
    elif R.dim() == 3:
        return transforms.matrix_to_euler_angles(torch.transpose(R, 1, 2) @ R_, "XYZ").abs().mean(1).unsqueeze(1)
        # return torch.abs(0.5*(torch.einsum('bii->b', torch.transpose(R, 1, 2) @ R_) - 1).unsqueeze(-1))
        # return 1 - 0.5 * (torch.einsum('bii->b', torch.transpose(R, 1, 2) @ R_) - 1).unsqueeze(-1)
        # return torch.norm(R.transpose(-1, -2) @ R_ - torch.eye(3, device=R.device), p='fro', dim=[1, 2])
    else:
        return ValueError





class experiment():
    def __init__(self, class_id, model_id, num_points, num_iterations, kp_noise_var_range,
                 kp_noise_type='sporadic', kp_noise_fra=0.2,
                 certify=certifiability(epsilon=0.8, delta=0.5, radius=0.3),
                 theta=50.0, kappa=10.0, shape_scaling=torch.tensor([0.5, 2.0])):
        super().__init__()

        # model parameters
        self.class_id = class_id
        self.model_id = model_id
        self.num_points = num_points
        self.shape_scaling = shape_scaling

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
        self.name = 'Analyzing keypoint corrector with simple registration on SE3nIsotropicShapePointCloud dataset'


        # setting up data
        self.se3_dataset = DepthAndIsotorpicShapePointCloud(class_id=self.class_id, model_id=self.model_id,
                                                            num_of_points=self.num_points,
                                                            dataset_len=self.num_iterations,
                                                            shape_scaling=self.shape_scaling)
        self.se3_dataset_loader = torch.utils.data.DataLoader(self.se3_dataset, batch_size=1, shuffle=False)

        self.model_keypoints = self.se3_dataset._get_model_keypoints()  # (2, 3, N)
        self.cad_models = self.se3_dataset._get_cad_models()  # (2, 3, m)
        self.diameter = self.se3_dataset._get_diameter()

        # setting up pace
        # pace parameters
        self.N = self.model_keypoints.shape[-1]
        self.K = self.model_keypoints.shape[0]
        self.weights = torch.ones(self.N, 1)
        self.pace = PACEbp(weights=self.weights,model_keypoints=self.model_keypoints,batch_size=1)

        # setting up keypoint corrector
        self.corrector = kp_corrector_pace(cad_models=self.cad_models, model_keypoints=self.model_keypoints,
                                           theta=self.theta, kappa=self.kappa, batch_size=1)

        # setting up model generator
        self.modelgen = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)


        # setting up experiment parameters and data for saving
        self.data = dict()
        self.parameters = dict()

        self.parameters['class_id'] = self.class_id
        self.parameters['model_id'] = self.model_id
        self.parameters['num_points'] = self.num_points
        self.parameters['shape_scaling'] = self.shape_scaling
        self.parameters['num_iterations'] = self.num_iterations
        self.parameters['kp_noise_type'] = self.kp_noise_type
        self.parameters['kp_noise_fra'] = self.kp_noise_fra
        self.parameters['kp_noise_var_range'] = self.kp_noise_var_range
        self.parameters['certify'] = self.certify
        self.parameters['theta'] = self.theta
        self.parameters['kappa'] = self.kappa
        self.parameters['name'] = self.name


    def _single_loop(self, kp_noise_var, visualization=False):

        # experiment data
        rotation_err_naive = torch.zeros(self.num_iterations, 1)
        rotation_err_corrector = torch.zeros(self.num_iterations, 1)
        translation_err_naive = torch.zeros(self.num_iterations, 1)
        translation_err_corrector = torch.zeros(self.num_iterations, 1)
        shape_err_naive = torch.zeros(self.num_iterations, 1)
        shape_err_corrector = torch.zeros(self.num_iterations, 1)

        certi_naive = torch.zeros((self.num_iterations, 1), dtype=torch.bool)
        certi_corrector = torch.zeros((self.num_iterations, 1), dtype=torch.bool)

        sqdist_input_naiveest = []
        sqdist_input_correctorest = []

        # experiment loop
        for i, data in enumerate(self.se3_dataset_loader):

            print("Testing at kp_noise_var: ", kp_noise_var, ". Iteration: ", i)

            # extracting data
            input_point_cloud, keypoints_true, rotation_true, translation_true, shape_true = data

            # generating perturbed keypoints
            detected_keypoints = keypoint_perturbation(keypoints_true=keypoints_true, type=self.kp_noise_type,
                                                       fra=self.kp_noise_fra, var=kp_noise_var*self.diameter)
            if visualization:
                visualize_torch_model_n_keypoints(cad_models=input_point_cloud, model_keypoints=detected_keypoints)

            # estimate model: using point set registration on perturbed keypoints
            R_naive, t_naive, c_naive = self.pace.forward(y=detected_keypoints)
            _, model_estimate_naive = self.modelgen.forward(shape=c_naive)
            model_estimate_naive = R_naive @ model_estimate_naive + t_naive
            if visualization:
                print("Displaying input and naive model estimate: ")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate_naive.squeeze(0))

            # estimate model: using the keypoint corrector
            correction = self.corrector.forward(detected_keypoints=detected_keypoints, input_point_cloud=input_point_cloud)
            # correction = torch.zeros_like(correction)
            R, t, c = self.pace.forward(y=detected_keypoints+correction)
            _, model_estimate = self.modelgen.forward(shape=c)
            model_estimate = R @ model_estimate + t
            if visualization:
                print("Displaying input and corrector model estimate: ")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate.squeeze(0))

            # evaluate the two metrics
            rotation_err_naive[i] = rotation_error(rotation_true, R_naive)
            rotation_err_corrector[i] = rotation_error(rotation_true, R)
            translation_err_naive[i] = translation_error(translation_true, t_naive)
            translation_err_corrector[i] = translation_error(translation_true, t)
            shape_err_naive[i] = shape_error(shape_true, c_naive)
            shape_err_corrector[i] = shape_error(shape_true, c)

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

            if visualization and i >= 5:
                break

        return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
               shape_err_naive, shape_err_corrector, \
               certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest



    def execute(self):

        rotation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        translation_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        shape_err_naive = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        rotation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        translation_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        shape_err_corrector = torch.zeros(len(self.kp_noise_var_range), self.num_iterations)
        certi_naive = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool)
        certi_corrector = torch.zeros(size=(len(self.kp_noise_var_range), self.num_iterations), dtype=torch.bool)
        sqdist_input_naiveest = []
        sqdist_input_correctorest = []

        for i, kp_noise_var in enumerate(self.kp_noise_var_range):

            print("-"*40)
            print("Testing at kp_noise_var: ", kp_noise_var)
            print("-"*40)

            Rerr_naive, Rerr_corrector, terr_naive, terr_corrector, shapeerr_naive, shapeerr_corrector, \
            c_naive, c_corrector, sqdist_in_naive, sq_dist_in_corrector = self._single_loop(kp_noise_var=kp_noise_var)

            rotation_err_naive[i, ...] = Rerr_naive.squeeze(-1)
            rotation_err_corrector[i, ...] = Rerr_corrector.squeeze(-1)

            translation_err_naive[i, ...] = terr_naive.squeeze(-1)
            translation_err_corrector[i, ...] = terr_corrector.squeeze(-1)

            shape_err_naive[i, ...] = shapeerr_naive.squeeze(-1)
            shape_err_corrector[i, ...] = shapeerr_corrector.squeeze(-1)

            certi_naive[i, ...] = c_naive.squeeze(-1)
            certi_corrector[i, ...] = c_corrector.squeeze(-1)

            sqdist_input_naiveest.append(sqdist_in_naive)
            sqdist_input_correctorest.append(sq_dist_in_corrector)


        self.data['rotation_err_naive'] = rotation_err_naive
        self.data['rotation_err_corrector'] = rotation_err_corrector
        self.data['translation_err_naive'] = translation_err_naive
        self.data['translation_err_corrector'] = translation_err_corrector
        self.data['shape_err_naive'] = shape_err_naive
        self.data['shape_err_corrector'] = shape_err_corrector
        self.data['certi_naive'] = certi_naive
        self.data['certi_corrector'] = certi_corrector
        self.data['sqdist_input_naiveest'] = sqdist_input_naiveest
        self.data['sqdist_input_correctorest'] = sqdist_input_correctorest

        return rotation_err_naive, rotation_err_corrector, translation_err_naive, translation_err_corrector, \
               shape_err_naive, shape_err_corrector, \
               certi_naive, certi_corrector, sqdist_input_naiveest, sqdist_input_correctorest

    def execute_n_save(self):

        # execute the experiment
        self.execute()

        # saving the experiment and data
        location = './expt_with_pace_depthisopc/'
        if not os.path.isdir(location):
            os.mkdir(location)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filemane = timestamp + '_experiment.pickle'

        file = open(location + filemane, 'wb')
        pickle.dump([self.parameters, self.data], file)
        file.close()

        return location + filemane


def run_experiments_on(class_id, model_id, kp_noise_type, kp_noise_fra=0.2, only_visualize=False):

    # model parameters
    num_points = 500
    shape_scaling = torch.tensor([0.5, 2.0])

    # averaging over
    num_iterations = 100

    # kp_noise parameters
    kp_noise_var_range = torch.arange(0.1, 1.55, 0.1)

    # certification parameters
    epsilon = 0.995
    delta = 0.5
    radius = 0.01
    certify = certifiability(epsilon=epsilon, delta=delta, radius=radius)

    # loss function parameters
    theta = 10.0
    kappa = 50.0

    print("-" * 40)
    print("Experiment: ")
    print("class_id: ", class_id)
    print("model_id: ", model_id)
    print("kp_noise_type: ", kp_noise_type)
    print("kp_noise_fra: ", kp_noise_fra)
    print("-" * 40)

    expt = experiment(class_id=class_id, model_id=model_id, num_points=num_points, shape_scaling=shape_scaling,
                      num_iterations=num_iterations, kp_noise_var_range=kp_noise_var_range,
                      kp_noise_type=kp_noise_type, kp_noise_fra=kp_noise_fra,
                      certify=certify, theta=theta, kappa=kappa)

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
        expt['kp_noise_type'] = kp_noise_type
        expt['kp_noise_fra'] = kp_noise_fra
        expt['filename'] = filename
        expt['num_iterations'] = num_iterations

        expt_filename = 'expt_with_pace_depthisopc/experiments.csv'
        field_names = ['class_id', 'model_id', 'kp_noise_type', 'kp_noise_fra', 'filename', 'num_iterations']

        fp = open(expt_filename, 'a')
        dict_writer = csv.DictWriter(fp, field_names)
        dict_writer.writerow(expt)
        fp.close()


#ToDo: Write the code to run this experiment for 10 models in each of the 16 model categories. The result will be the average error.

if __name__ == "__main__":

    # model parameters
    class_id = "03001627"  # chair
    model_id = "1e3fba4500d20bb49b9f2eb77f5e247e"  # a particular chair model

    run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='sporadic', kp_noise_fra=0.2)
    run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='sporadic', kp_noise_fra=0.8)
    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='uniform')

    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='sporadic', kp_noise_fra=0.2,
    #                    only_visualize=True)
    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='sporadic', kp_noise_fra=0.8,
    #                    only_visualize=True)
    # run_experiments_on(class_id=class_id, model_id=model_id, kp_noise_type='uniform', only_visualize=True)
