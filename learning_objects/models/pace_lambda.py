"""
This code implements a function that tunes the lambda_constant for PACE

"""

import torch
from scipy import optimize
import numpy as np
import random

import sys
sys.path.append("../../")

from learning_objects.models.pace import PACEmodule
from learning_objects.models.modelgen import ModelFromShape
from learning_objects.datasets.keypointnet import SE3nAnisotropicScalingPointCloud, \
    DepthAndAnisotropicScalingPointCloud, ScaleAxis
from learning_objects.utils.general import display_two_pcs


class GetLambda():
    def __init__(self, model_keypoints):
        super().__init__()

        self.model_keypoints = model_keypoints

    def loss(self, c, c_):
        """
        c   : torch.tensor of shape (B, K, 1)
        c_  : torch.tensor of shape (B, K, 1)

        """

        return ((c - c_)**2).mean(dim=1).mean(dim=1)

    def get_shapes(self):
        """
        model_keypoints : torch.tensor of shape (K, 3, N)

        output:
        keypoints   : torch.tensor of shape (B, 3, N)
            where B = len(eta) * K
        shape       : torch.tensor of shape (B, K, 1)
        """
        model_keypoints = self.model_keypoints

        K, _, N = model_keypoints.shape
        device_ = model_keypoints.device
        eta_range = [0.8, 0.9, 0.99]

        model_keypoints = model_keypoints.unsqueeze(0)  # (1, K, 3, N)
        keypoints = torch.zeros(K * len(eta_range), 3, N)
        shape = torch.zeros(K * len(eta_range), K, 1)
        idx = 0
        for eta in eta_range:

            # c of shape (K, K)
            c = ((1 - eta) / (K - 1)) * torch.ones(K, K).to(device=device_)
            c += (eta - (1 - eta) / (K - 1)) * torch.eye(K).to(device=device_)

            c = c.unsqueeze(-1).unsqueeze(-1)   # (K, K, 1, 1)

            kp = torch.einsum('akii,jkdn->adn', c, model_keypoints)
            keypoints[idx * K:(idx + 1) * K, ...] = kp
            shape[idx * K:(idx + 1) * K, ...] = c.squeeze(-1)

            idx += 1

        return keypoints, shape


    def get(self):

        keypoints, shape = self.get_shapes()  # keypoitns: (B, 3, N), shape: (B, K, 1)

        def fun(lam):
            lam = torch.from_numpy(lam).to(torch.float)

            pace = PACEmodule(model_keypoints=self.model_keypoints, lambda_constant=lam)
            _, _, c = pace(keypoints)

            return self.loss(c, shape).mean()

        # Optimize fun as a function of lambda
        lam0 = np.array([1.0])
        loss_now = 100000
        result = optimize.minimize(fun, lam0, method='trust-constr', bounds=((0, 10.0),))

        print("loss before optimization: ", fun(lam0))
        print("loss after optimization: ", result.fun)
        print("opt status: ", result.status)
        print("num of steps: ", result.nit)
        print("corrector optimization successful: ", result.success)
        lambda_constant = torch.from_numpy(result.x).to(torch.float)

        # print("The lambda constant is: ", lambda_constant)

        return lambda_constant

def test_isotropic_cube():
    """
    Tests lambda optimization on hardcoded keypoints of two isotropic scalings of a cube
    :return:
    """
    B = torch.zeros(3, 8).to(torch.float)
    B[:, 0] = torch.tensor([1.0, 1.0, 1.0]).to(torch.float)
    B[:, 1] = torch.tensor([1.0, 1.0, 0.0]).to(torch.float)
    B[:, 2] = torch.tensor([1.0, 0.0, 1.0]).to(torch.float)
    B[:, 3] = torch.tensor([1.0, 0.0, 0.0]).to(torch.float)
    B[:, 4] = torch.tensor([0.0, 1.0, 1.0]).to(torch.float)
    B[:, 5] = torch.tensor([0.0, 1.0, 0.0]).to(torch.float)
    B[:, 6] = torch.tensor([0.0, 0.0, 1.0]).to(torch.float)
    B[:, 7] = torch.tensor([0.0, 0.0, 0.0]).to(torch.float)

    # centering pointcloud
    B = B - torch.tensor([0.5, 0.5, 0.5]).unsqueeze(-1).to(torch.float)

    # print(B)
    # isotropic scaling
    B0 = B
    B1 = 2*B

    model_keypoints = torch.cat([B0.unsqueeze(0), B1.unsqueeze(0)], dim=0)  # (K, 3, N)
    lambda_opt = GetLambda(model_keypoints=model_keypoints)
    lambda_constant = lambda_opt.get()

    print("The lambda constant is: ", lambda_constant)

def generate_lambda(class_id, model_id, shape_scaling= torch.tensor([0.5, 2.0])):
    """
    Optimizes lambda_constant on the SE3nAnistropicScalingPointCloud
    dataset for a particular class_id and model_id, returns the optimal lambda_constant
    for use in pace. Optimization for lambda_constant happens with no rotation or translation.
    Ground truth keypoints are used.
    :param class_id: the id of the category used for the dataset
    :param model_id: the id of the model used for the dataset. This model is anistropically scaled
                        in the dataset
    :param shape_scaling: the minimum and maximum amounts to scale the object in the dataset
    :return:
    """
    se3_dataset = SE3nAnisotropicScalingPointCloud(class_id=class_id, model_id=model_id,
                                                        num_of_points=2048, dataset_len=10,
                                                        shape_scaling=shape_scaling, scale_direction=ScaleAxis.X)
    keypoints = se3_dataset._get_model_keypoints() #(2, 3, num_keypoints)
    lambda_opt = GetLambda(model_keypoints=keypoints)
    lambda_constant = lambda_opt.get()

    print("The lambda constant is: ", lambda_constant)


    return lambda_constant, se3_dataset


def test_lambda(lambda_constant, dataset, viz=False):
    if viz:
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    model_keypoints = dataset._get_model_keypoints()

    modelgen = ModelFromShape(cad_models=dataset._get_cad_models(), model_keypoints=model_keypoints)

    pace = PACEmodule(model_keypoints=model_keypoints, lambda_constant=lambda_constant)
    for i, data in enumerate(dataset_loader):
        input_point_cloud, keypoints_true, rotation_true, translation_true, shape_true = data
        R, t, c = pace.forward(y=keypoints_true) #test pace for input keypoints without perturbation
        if viz:
            _, model_estimate_naive = modelgen.forward(shape=c)
            model_estimate_naive = R @ model_estimate_naive + t
            display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=model_estimate_naive.squeeze(0))

        print("mean batch loss: " + str(i) + str(((c - shape_true) ** 2).mean(dim=1).mean(dim=1).mean())) #mean loss per batch
        print("------------------------------------")


if __name__ == "__main__":
    #test:
    # generate a lambda optimized for a model inside of a category
    # test that lambda on different models within that category
    # uncomment the category you want to tune lambda for:

    class_id = "03001627"  # chair
    chair_models = ["1e3fba4500d20bb49b9f2eb77f5e247e", "1a6f615e8b1b5ae4dbbc9440457e303e",
                    "1a38407b3036795d19fb4103277a6b93", "1b7bef12c554c1244c686b8271245d1b",
                    "1b92525f3945f486fe24b6f1cb4a9319", "1e0580f443a9e6d2593ebeeedbff73b",
                    "1f0bfd529b33c045b84e887edbdca251", "2b1af04045c8c823f51f77a6d7299806",
                    "2bd6800d64c01d677721fafb59ea099", "2bd6800d64c01d677721fafb59ea099"]
    random.shuffle(chair_models) #shuffles in place
    for model_id in chair_models:
        lambda_constant, dataset = generate_lambda(class_id, model_id)
        break
    for model_id in chair_models:
        print("testing on new model_id within category", class_id)
        # if the depth dataset is uncommented, viz must be set to true in test_lambda
        # dataset = DepthAndAnisotropicScalingPointCloud(class_id=class_id, model_id=model_id,
        #                                                num_of_points=2048,
        #                                                dataset_len=10,
        #                                                shape_scaling=torch.tensor([0.5, 2.0]),
        #                                                scale_direction=ScaleAxis.X)

        test_lambda(lambda_constant, dataset)

    class_id = "03467517" #guitar
    # guitar_models = ["1a96f73d0929bd4793f0194265a9746c", "1c8c6874c0cb9bc73429c1c21d77499d",
    #                 "1e019147e99ddf90d6e28d388c951ca4", "1e56954ca36bbfdd6b05157edf4233a3",
    #                 "1f08ecadd3cb407dcf45555a18e7351a", "1fbbf4450cb813faed5f5791584afe61",
    #                 "2c491c5718322fc4849457db85ec22c6", "2d31ac290daab56a4e96698ef01e7360",
    #                 "2e2d2d6dbb316502872341b062fa57a9", "2e4ec0874ea34a50812ca0ac90db1c07"]
    # random.shuffle(guitar_models) #shuffles in place
    # for model_id in guitar_models:
    #     lambda_constant, dataset = generate_lambda(class_id, model_id)
    #     break
    # for model_id in guitar_models:
    #     print("testing on new model_id within category", class_id)
    #     test_lambda(lambda_constant, dataset)
    #

    # class_id = "02876657" #bottle
    # bottle_models = ["1ae823260851f7d9ea600d1a6d9f6e07", "1c38ca26c65826804c35e2851444dc2f",
    #                 "3e7d7a9637e385f2fd1efdcc788bb066", "3f91158956ad7db0322747720d7d37e8",
    #                 "5ad47181a9026fc728cc22dce7529b69", "5c1253ae61a16e306871591246ec74dc",
    #                 "6c24c027831790516f79a03aa42f27b9", "6e57c665a47d7d7730612f5c0ef21eb8",
    #                 "6ebe74793197919e93f2361527e0abe5", "9ba6291a60113dbfeb6d856318cd1a7e"]
    # random.shuffle(bottle_models) #shuffles in place
    # for model_id in bottle_models:
    #     lambda_constant, dataset = generate_lambda(class_id, model_id)
    #     break
    # for model_id in bottle_models:
    #     print("testing on new model_id within category", class_id)
    #     test_lambda(lambda_constant, dataset)

    # class_id = "03790512" #motorcycle
    # motorcycle_models = ["1e664c94e932ece9883158c0007ed9ba", "2d655fc4ecb2df2a747c19778aa6cc0",
    #                 "2f32164c5a422fe8b44d82a74ebc14c0", "2ff252507b18ce60cd4bec905df51d1d",
    #                 "3a94134ec9698b74ec5901a85efd6b67", "4df789715eea8fa552ee90e577613070",
    #                 "5be7e00fe279ac927cb7616d82cf9fae", "5de5e9abd5f49ad7468bac13e007a6e9",
    #                 "6a9db96b3ed76f956dc3fd80522499a5", "6b6cf38ed305dc11ab13c10f92a32fc1"]
    # random.shuffle(motorcycle_models) #shuffles in place
    # for model_id in motorcycle_models:
    #     lambda_constant, dataset = generate_lambda(class_id, model_id)
    #     break
    # for model_id in motorcycle_models:
    #     print("testing on new model_id within category", class_id)
    #     #if the depth dataset is uncommented, viz must be set to true in test_lambda
    #     dataset = DepthAndAnisotropicScalingPointCloud(class_id=class_id, model_id=model_id,
    #                                                    num_of_points=2048,
    #                                                    dataset_len=10,
    #                                                    shape_scaling=torch.tensor([0.5, 2.0]),
    #                                                    scale_direction=ScaleAxis.X)
    #
    #     test_lambda(lambda_constant, dataset)




