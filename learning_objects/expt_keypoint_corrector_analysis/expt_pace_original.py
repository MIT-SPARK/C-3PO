
import torch
from pytorch3d import ops, transforms

from datetime import datetime
import pickle
import csv

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3nIsotorpicShapePointCloud, SE3nAnisotropicScalingPointCloud, \
    ScaleAxis, visualize_torch_model_n_keypoints

from learning_objects.models.keypoint_corrector import kp_corrector_reg, kp_corrector_pace
from learning_objects.models.point_set_registration import point_set_registration
from learning_objects.models.pace import PACEmodule
# from learning_objects.test.test_pace import PACEAltern
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.models.certifiability import certifiability

from learning_objects.utils.ddn.node import ParamDeclarativeFunction
from learning_objects.utils.general import display_two_pcs

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

    sq_dist_yx, _, _ = ops.knn_points(torch.transpose(Y, -1, -2), torch.transpose(X, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in Y to the nearest point in X

    return sq_dist_xy, sq_dist_yx




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
    def __init__(self, class_id, model_id, num_points, num_iterations,
                 certify=certifiability(epsilon=0.8, delta=0.5, radius=0.3),
                 theta=50.0, kappa=10.0, shape_scaling=torch.tensor([0.5, 2.0]), dataset_weights = None):
        super().__init__()

        # model parameters
        self.class_id = class_id
        self.model_id = model_id
        self.num_points = num_points
        self.shape_scaling = shape_scaling
        self.shape_dataset = dataset_weights

        # averaging over
        self.num_iterations = num_iterations

        # loss function parameters
        self.theta = theta
        self.kappa = kappa

        # experiment name
        self.name = 'Analyzing original PACE implementation on SE3nAnisotropicScalingPointCloud dataset'


        # setting up data
        self.se3_dataset = SE3nAnisotropicScalingPointCloud(class_id=self.class_id, model_id=self.model_id,
                                                        num_of_points=self.num_points, dataset_len=self.num_iterations,
                                                        shape_scaling=self.shape_scaling, scale_direction=ScaleAxis.X,
                                                            shape_dataset=self.shape_dataset)
        self.se3_dataset_loader = torch.utils.data.DataLoader(self.se3_dataset, batch_size=1, shuffle=False)

        self.model_keypoints = self.se3_dataset._get_model_keypoints()  # (2, 3, N)
        self.cad_models = self.se3_dataset._get_cad_models()  # (2, 3, m)
        self.diameter = self.se3_dataset._get_diameter()

        # setting up pace
        # pace parameters
        self.N = self.model_keypoints.shape[-1]
        self.K = self.model_keypoints.shape[0]
        self.weights = torch.ones(self.N, 1)
        self.pace = PACEmodule(model_keypoints=self.model_keypoints, weights=self.weights, use_optimized_lambda_constant=True,
                               class_id=self.class_id)


        self.modelgen = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)


        # setting up experiment parameters and data for saving
        self.data = dict()
        self.parameters = dict()

        self.parameters['class_id'] = self.class_id
        self.parameters['model_id'] = self.model_id
        self.parameters['num_points'] = self.num_points
        # self.parameters['shape_scaling'] = self.shape_scaling
        self.parameters['shape_weights'] = self.shape_dataset
        self.parameters['num_iterations'] = self.num_iterations
        self.parameters['theta'] = self.theta
        self.parameters['kappa'] = self.kappa
        self.parameters['name'] = self.name


    def _single_loop(self, visualization=False):

        # experiment data
        rotation_err_naive = torch.zeros(len(self.shape_dataset), 1)
        translation_err_naive = torch.zeros(len(self.shape_dataset), 1)
        shape_err_naive = torch.zeros(len(self.shape_dataset), 1)
        shape_true_1 = torch.zeros(len(self.shape_dataset), 1)
        shape_true_2 = torch.zeros(len(self.shape_dataset), 1)
        shape_pace_1 = torch.zeros(len(self.shape_dataset), 1)
        shape_pace_2 = torch.zeros(len(self.shape_dataset), 1)




        # experiment loop
        for i, data in enumerate(self.se3_dataset_loader):

            # extracting data
            input_point_cloud, keypoints_true, rotation_true, translation_true, shape_true = data

            # estimate model: using point set registration on perturbed keypoints
            R_pace, t_pace, c_pace = self.pace.forward(y=keypoints_true)
            _, model_estimate_naive = self.modelgen.forward(shape=c_pace)
            #also see output of modelgen from ground truth shape
            # use pace module (no alternating minimization and initilize with ground truth)
            model_estimate_naive = R_pace @ model_estimate_naive + t_pace
            if visualization:
                print("Displaying input and naive model estimate: ")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate_naive.squeeze(0))


            # evaluate the two metrics
            rotation_err_naive[i] = rotation_error(rotation_true, R_pace)
            translation_err_naive[i] = translation_error(translation_true, t_pace)
            shape_err_naive[i] = shape_error(shape_true, c_pace)
            shape_true = shape_true.squeeze()
            c_pace = c_pace.squeeze()
            print("c_true", shape_true)
            print("c_pace", c_pace)
            shape_true_1[i] = shape_true[0]
            shape_true_2[i] = shape_true[1]
            shape_pace_1[i] = c_pace[0]
            shape_pace_2[i] = c_pace[1]
            if i == len(self.shape_dataset) - 1 :
                break


        return rotation_err_naive, translation_err_naive, \
               shape_err_naive, shape_true_1, shape_true_2, shape_pace_1, shape_pace_2



    def execute(self):

        rotation_err_naive = torch.zeros(self.num_iterations, len(self.shape_dataset))
        translation_err_naive = torch.zeros(self.num_iterations, len(self.shape_dataset))
        shape_err_naive = torch.zeros(self.num_iterations, len(self.shape_dataset))
        shape_true_1 = torch.zeros(self.num_iterations, len(self.shape_dataset))
        shape_true_2 = torch.zeros(self.num_iterations, len(self.shape_dataset))

        shape_pace_1 = torch.zeros(self.num_iterations, len(self.shape_dataset))
        shape_pace_2 = torch.zeros(self.num_iterations, len(self.shape_dataset))

        for i in range(self.num_iterations):

            Rerr_naive, terr_naive, shapeerr_naive, c_true1, c_true2, c_pace1, c_pace2 = self._single_loop()

            rotation_err_naive[i, ...] = Rerr_naive.squeeze(-1)

            translation_err_naive[i, ...] = terr_naive.squeeze(-1)

            shape_err_naive[i, ...] = shapeerr_naive.squeeze(-1)

            shape_true_1[i, ...] = c_true1.squeeze(-1)
            shape_true_2[i, ...] = c_true2.squeeze(-1)
            shape_pace_1[i, ...] = c_pace1.squeeze(-1)
            shape_pace_2[i, ...] = c_pace2.squeeze(-1)





        self.data['rotation_err_naive'] = rotation_err_naive
        self.data['translation_err_naive'] = translation_err_naive
        self.data['shape_err_naive'] = shape_err_naive
        self.data['shape_true_1'] = shape_true_1
        self.data['shape_true_2'] = shape_true_2
        self.data['shape_pace_1'] = shape_pace_1
        self.data['shape_pace_2'] = shape_pace_2




        return rotation_err_naive, translation_err_naive, \
               shape_err_naive, shape_true_1, shape_true_2, shape_pace_1, shape_pace_2

    def execute_n_save(self):

        # execute the experiment
        self.execute()

        # saving the experiment and data
        location = './expt_pace_original/' + str(self.class_id) + '/' + str(self.model_id) + '/'
        if not os.path.isdir(location):
            os.makedirs(location)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = timestamp + '_experiment.pickle'

        file = open(location + filename, 'wb')
        pickle.dump([self.parameters, self.data], file)
        file.close()

        return location + filename


def run_experiments_on(class_id, model_id, only_visualize=False):

    # model parameters
    num_points = 2048
    shape_scaling = torch.tensor([0.5, 2.0])
    #input shapes to try
    shape_1_weight = torch.arange(0, 1.1, 0.1)
    shape_2_weight = torch.flip(torch.arange(0, 1.1, 0.1), [0])
    shape_dataset_weights = torch.vstack((shape_1_weight, shape_2_weight)).transpose(0, 1)
    # tensor([[0.0000, 1.0000],
    #     [0.1000, 0.9000],
    #     [0.2000, 0.8000],
    #     [0.3000, 0.7000],
    #     [0.4000, 0.6000],
    #     [0.5000, 0.5000],
    #     [0.6000, 0.4000],
    #     [0.7000, 0.3000],
    #     [0.8000, 0.2000],
    #     [0.9000, 0.1000],
    #     [1.0000, 0.0000]])


    # averaging over
    num_iterations = 100

    # loss function parameters
    theta = 10.0
    kappa = 50.0

    print("-" * 40)
    print("Experiment: ")
    print("class_id: ", class_id)
    print("model_id: ", model_id)
    print("-" * 40)

    expt = experiment(class_id=class_id, model_id=model_id, num_points=num_points, shape_scaling=shape_scaling,
                      num_iterations=num_iterations, theta=theta, kappa=kappa, dataset_weights = shape_1_weight)

    if only_visualize:
        while True:
            expt._single_loop(visualization=True)
    else:
        filename = expt.execute_n_save()

        # experiment data
        expt = dict()
        expt['class_id'] = class_id
        expt['model_id'] = model_id
        expt['filename'] = filename
        expt['num_iterations'] = num_iterations

        expt_filename = 'expt_pace_original/' + str(class_id) + '/' + str(model_id) + '/experiments.csv'
        field_names = ['class_id', 'model_id', 'filename', 'num_iterations']

        fp = open(expt_filename, 'a')
        dict_writer = csv.DictWriter(fp, field_names)
        dict_writer.writerow(expt)
        fp.close()


#ToDo: Write the code to run this experiment for 10 models in each of the 16 model categories. The result will be the average error.

if __name__ == "__main__":

    # model parameters
    class_id = "03001627"
    model_id = "1f0bfd529b33c045b84e887edbdca251"  # a particular model

    # class_id = "04379243"
    # model_id = "2a88f66d5e09e502581fd77200548509"

    run_experiments_on(class_id=class_id, model_id=model_id, only_visualize=False)

