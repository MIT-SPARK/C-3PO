"""
This implements keypoint detector modules.

Note:
    There are two kinds of keypoint detectors: heatmap based and regression based.

    Current implementations use Point Transformer and PointNet++ architectures for the keypoint detectors,
    and can be chosen by specifying the "method" variable in the input.

"""

import torch
import torch.nn as nn

import os
import sys
sys.path.append("../../")

import numpy as np

from learning_objects.models.point_transformer import PointTransformerSegment, PointTransformerCls
from learning_objects.models.pointnet import PointNetDenseCls, PointNetCls
from learning_objects.models.rsnet import RSNet

CORRESPONDENCE_LOG_FOLER: str = '../../KeypointNet/benchmark_scripts/correspondence_log/'


class ModelWrapper(torch.nn.Module):
    def __init__(self, model_impl) -> None:
        super().__init__()
        self.model_impl = model_impl

    def forward(self, data):
        pc = data[0]
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()
        res = self.model_impl(pc.transpose(1, 2).cuda())
        return res

class CorrespondenceCriterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, outputs):
        print("---------------------------------------------------")
        print("inside CorrespondenceCriterion")
        print("inputs.shape", inputs.shape)
        print("outputs.shape", outputs.shape)
        kp_indexs = inputs[1]

        loss = []
        for b, kp_index in enumerate(kp_indexs):
            loss_rot = []
            for rot_kp_index in kp_index:
                loss_rot.append(F.cross_entropy(outputs[b][None], rot_kp_index[None].long().cuda(), ignore_index=-1))
            loss.append(torch.min(torch.stack(loss_rot)))
        loss = torch.mean(torch.stack(loss))
        return loss


class HeatmapKeypoints(nn.Module):
    """
    This module generates keypoints of an input point cloud using heatmaps.

    Note:
    The output keypoints, by definition, are a subset of the input point cloud.
    This method uses point cloud segmentation architecture.
    """
    def __init__(self, N=20, method='point_transformer', dim=[3, 32, 64, 128, 256, 512]):
        super().__init__()
        """
        Inputs: 
        ------
            N       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
            dim     : list(int)   : dimensions of various layers, in the case of point transformer architecture. 
        """
        self.N = N
        self.method = method
        self.dim = dim

        if self.method == 'point_transformer':
            self.keypoint_heatmap = PointTransformerSegment(output_dim=self.N, dim=self.dim)
        elif self.method == 'pointnet':
            self.keypoint_heatmap = PointNetDenseCls(k=self.N)
        else:
            raise NotImplementedError

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
        pointcloud: (B, 3+d, self.N) torch.Tensor
                        B = batch size
                        self.N = number of points per point cloud
                        d = input feature dimension
                        pointcloud[B, 0:3, self.N] = locations
                        pointcloud[B, 3:, self.N] = input features

        Outputs:
        -------
            y: (B, 3, self.N) torch.Tensor
                        B = batch size
                        self.N = number of keypoints
                        y[B, 0:3, self.N] = location of keypoints
        """

        pointcloud = torch.transpose(pointcloud, -1, -2)    # (B, N, 3+d)

        if self.method == 'point_transformer':

            heatmap = self.keypoint_heatmap(pointcloud)
            _, idx = torch.max(heatmap, dim=1)

        elif self.method == 'pointnet':

            pos, _ = self._break_up_pc(pointcloud)
            heatmap, _, _ = self.keypoint_heatmap(pos.transpose(1, 2))
            _, idx = torch.max(heatmap, dim=1)

        else:
            raise ValueError


        y = torch.zeros(size=(pointcloud.size(0), self.N, 3))
        for i in range(idx.size(0)):
            y[i, :, :] = pointcloud[i, idx[i], :3]

        return y.transpose(-1, -2)      # (B, 3, self.N)


class RegressionKeypoints(nn.Module):
    """
    This module generates keypoints as a regression of the input point cloud.

    Note:
    The output keypoints may not be a subset of the input point cloud.
    This method uses point cloud classification architecture.
    """
    def __init__(self, N=20, method='point_transformer', dim=[3, 16, 32, 64, 128, 256]):
        super().__init__()
        """
        Inputs: 
        ------
            N       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
        """
        self.N = N
        self.method = method

        if self.method == 'point_transformer':
            self.keypoint = PointTransformerCls(output_dim=3*self.N, channels=dim, sampling_ratio=0.25)
        elif self.method == 'pointnet':
            self.keypoint = PointNetCls(k=3*self.N)
        else:
            raise NotImplementedError

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
            pointcloud: (B, 3+d, m) torch.Tensor
                    B = batch size
                    m = number of points per point cloud
                    d = input feature dimension
                    pointcloud[..., 0:3] = locations
                    pointcloud[..., 3:] = input features

        Outputs:
        -------
            y: (B, 3, self.N) torch.Tensor
                    B = batch size
                    self.N = number of keypoints
                    y[..., 0:3] = location of keypoints
        """
        batch_size = pointcloud.shape[0]
        pointcloud = torch.transpose(pointcloud, -1, -2)  # (B, N, 3+d)

        if self.method == 'point_transformer':
            keypoints = self.keypoint(pointcloud)

        elif self.method == 'pointnet':
            pos, _ = self._break_up_pc(pointcloud)
            keypoints, _, _ = self.keypoint(pos.transpose(1, 2))

        else:
            raise ValueError
        return torch.reshape(keypoints, (batch_size, 3, self.N))

class RSNetKeypoints(nn.Module):
    """
    Generates keypoints from input point cloud
    Output keypoints are a subset of the input point cloud
    """
    def __init__(self, class_name="bed" , N=66, method='rsnet'):
        super().__init__()
        """
        Inputs: 
        ------
            N       : int32   : number of keypoints 
            method  : string  : takes values: {'point_transformer', 'pointnet'}
        """
        self.N = N
        self.method = method
        self.class_name = class_name
        self.best_model_checkpoint = os.path.join(CORRESPONDENCE_LOG_FOLER, self.class_name + '/',
                                                  self.method + '/best.pth')

        if not os.path.isfile(self.best_model_checkpoint):
            print("ERROR: CAN'T LOAD PRETRAINED RSNET MODEL, PATH DOESN'T EXIST")
            print(self.best_model_checkpoint)

        if self.method == 'rsnet':
            self.model = ModelWrapper(RSNet(network_res=.01, network_rg=2.0, num_classes=66))#.cuda()
            checkpoint = torch.load(self.best_model_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.train()
        else:
            raise NotImplementedError

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
        pointcloud: (B, 3+d, self.N) torch.Tensor
                        B = batch size
                        self.N = number of points per point cloud
                        d = input feature dimension
                        pointcloud[B, 0:3, self.N] = locations
                        pointcloud[B, 3:, self.N] = input features

        Outputs:
        -------
            y: (B, 3, self.N) torch.Tensor
                        B = batch size
                        self.N = number of keypoints (66)
                        y[B, 0:3, self.N] = location of keypoints
        """

        pointcloud = torch.transpose(pointcloud, -1, -2)    # (B, N, 3+d)
        if self.method == 'rsnet':
            pos, _ = self._break_up_pc(pointcloud) #torch.Size([B, 2048, 3])
            pos = [pos]
            outputs = self.model(pos) #(B, 2048, 66)
            pred_index = torch.argmax(outputs, dim=1)#.cpu().numpy() #(5,66) batch, idxs of pts of pcl
            #take only the first relevant keypoints for class
            pred_index = pred_index[...,:self.N]

        else:
            raise ValueError


        y = torch.zeros(size=(pointcloud.size(0), self.N, 3), requires_grad=True).to(self.device)
        for i in range(pred_index.shape[0]):
            y[i, :, :] = pointcloud[i, pred_index[i], :3]

        return y.transpose(-1, -2)      # (B, 3, self.N)


if __name__ == "__main__":

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    # Parameters:
    N = 44      # number of keypoints
    m = 100     # number of points in the input point cloud
    b = 10      # batch size

    print('-'*20)
    print('Testing Heatmap Models')
    print('-'*20)
    # Test: HeatmapKeypoints() with method='pointnet'
    print('Test: HeatmapKeypoints() with method=\'pointnet\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)
    # pc = torch.rand(4, 1000, 11)
    pc = pc.to(device=device)
    kp = HeatmapKeypoints(method='pointnet', N=N).to(device=device)
    # kp = HeatmapKeypoints(method='point_transformer', dim=[8, 32, 64], k=63).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)



    # Test: HeatmapKeypoints() with method='point_transformer'
    print('Test: HeatmapKeypoints() with method=\'point_transformer\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)
    # pc = torch.rand(4, 1000, 6)
    pc = pc.to(device=device)
    # kp = HeatmapKeypoints(method='pointnet', k=63).to(device=device)
    kp = HeatmapKeypoints(method='point_transformer', dim=[8, 16, 24], N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)



    print('-'*20)
    print('Testing Regression Models')
    print('-'*20)
    # Test: RegressionKeypoints() with method='pointnet'
    print('Test: RegressionKeypoints() with method=\'pointnet\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)

    pc = pc.to(device=device)
    kp = RegressionKeypoints(method='pointnet', N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)



    # Test: RegressionKeypoints() with method='point_transformer'
    print('Test: RegressionKeypoints() with method=\'point_transformer\'')

    pc = torch.rand(b, 3+8, m)
    print("Shape of input point cloud: ", pc.shape)
    # pc = torch.rand(4, 1000, 6)
    pc = pc.to(device=device)
    kp = RegressionKeypoints(method='point_transformer', dim=[8, 16, 24], N=N).to(device=device)
    y = kp(pointcloud=pc)
    print("Shape of keypoint output: ", y.shape)
    print('-' * 20)

