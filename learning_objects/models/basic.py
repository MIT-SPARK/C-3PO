import torch
import torch.nn as nn

import sys
sys.path.append("../../")


from learning_objects.utils.ddn.node import ParamDeclarativeFunction

from learning_objects.utils.general import generate_random_keypoints

from learning_objects.models.keypoint_detector import RegressionKeypoints
from learning_objects.models.pace_ddn import PACEbp
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.discard.keypoint_corrector_old import PACEwKeypointCorrection, from_y



class ProposedModel(nn.Module):
    def __init__(self, model_keypoints, cad_models, keypoint_type='regression',
                 keypoint_method='pointnet', weights=None, lambda_constant=None, keypoint_correction=False):
        super().__init__()
        """
        keypoint_type   : 'heatmap' or 'regression'
        keypoint_method : 'pointnet' or 'point_transformer'
        model_keypoints : torch.tensor of shape (K, 3, N)
        cad_models      : torch.tensor of shape (K, 3, n)
        weights         : torch.tensor of shape (N, 1) or None
        lambda_constant : torch.tensor of shape (1, 1) or None
        keypoint_correction     : True or False
        """
        self.keypoint_type = keypoint_type
        self.keypoint_correction = keypoint_correction
        self.keypoint_method = keypoint_method
        self.model_keypoints = model_keypoints
        self.N = self.model_keypoints.shape[-1]  # (1, 1)
        self.K = self.model_keypoints.shape[0]  # (1, 1)
        self.cad_models = cad_models

        self.weights = weights
        if weights == None:
            self.weights = torch.ones(self.N, 1)

        self.lambda_constant = lambda_constant
        if lambda_constant == None:
            self.lambda_constant = torch.sqrt(torch.tensor([self.N/self.K]))

        # Keypoint Detector
        self.keypoint_detector = RegressionKeypoints(k=self.N, method=keypoint_method)

        # PACE
        self.pace_fn = PACEbp(weights=self.weights, model_keypoints=self.model_keypoints)

        # PACE + Keypoint Corrector
        if self.keypoint_correction:
            self.pace_and_correction_node = PACEwKeypointCorrection(model_keypoints=self.model_keypoints,
                                                                         cad_models=self.cad_models,
                                                                         weights=self.weights,
                                                                         lambda_constant=self.lambda_constant)
            self.pace_and_correction_fn = ParamDeclarativeFunction(self.pace_and_correction_node)

        # Model Generator
        self.generate_model_fn = ModelFromShape(cad_models=self.cad_models, model_keypoints=self.model_keypoints)


    def forward(self, input_point_cloud, train=True):
        """
        input:
        input_point_cloud: torch.tensor of shape (B, 3, m)

        where
        B = batch size
        m = number of points in each point cloud

        output:
        keypoints: torch.tensor of shape (B, 3, N)
        predicted_model: torch.tensor of shape (B, 3, n)
        """

        # detect keypoints
        detected_keypoints = torch.transpose(
            self.keypoint_detector(torch.transpose(input_point_cloud, -1, -2)), -1, -2)

        print(detected_keypoints.shape)

        if train or (not self.keypoint_correction):
            # During training or when not using keypoint_correction
            #           keypoints = detected_keypoints
            #
            R, t, c = self.pace_fn.forward(y=detected_keypoints)
            keypoints = detected_keypoints
        else:
            # During testing and when using keypoint_correction
            #           keypoints = corrected_keypoints from the bi-level optimization.
            #
            y = self.pace_and_correction_fn.forward(input_point_cloud, detected_keypoints)
            R, t, c, correction = from_y(y=y, K=self.K, N=self.N)
            keypoints = detected_keypoints + correction

        return R, t, c, keypoints












if __name__ == '__main__':

    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device is ', device)
    print('-' * 20)


    #
    B = 2   # batch size
    N = 8   # number of keypoints
    K = 2   # number of cad models in the shape category
    n = 10  # number of points in the model_point_cloud
    m = 10  # number of points in the input_point_cloud

    cad_models = torch.rand(K, 3, n).to(device=device)
    model_keypoints = cad_models[:, :, 0:N]
    weights = torch.rand(N, 1).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)

    # initializing model
    proposed_model = ProposedModel(model_keypoints=model_keypoints,
                                   cad_models=cad_models,
                                   weights=weights,
                                   lambda_constant=lambda_constant,
                                   keypoint_correction=False)

    detected_keypoints, rotation, translation, shape = generate_random_keypoints(batch_size=B,
                                                                                 model_keypoints=model_keypoints.cpu())

    model_gen_for_data = ModelFromShape(cad_models=cad_models.cpu(), model_keypoints=model_keypoints.cpu())
    _, input_point_cloud = model_gen_for_data.forward(shape=shape)
    input_point_cloud = rotation @ input_point_cloud + translation

    # transferring generated data to device
    input_point_cloud = input_point_cloud.to(device=device)
    detected_keypoints = detected_keypoints.to(device=device)
    rotation = rotation.to(device=device)
    translation = translation.to(device=device)
    shape = shape.to(device=device)


    # applying model
    R, t, c, kp = proposed_model(input_point_cloud)






    

