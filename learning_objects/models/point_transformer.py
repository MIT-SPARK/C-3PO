"""
This is an  implementation of point transformer.

Source: https://github.com/lucidrains/point-transformer-pytorch

Paper: https://arxiv.org/abs/2012.09164
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import torch.utils.dlpack

from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.pool import fps
# from torch_cluster import fps (use torch_geometric.nn.pool.fps)
# import point_transformer_ops.point_transformer_utils as pt_utils

#### Some functions taken from point-transformer.point_transformer_lib.point_transformer_utils


def three_interpolate(p, p_old, x):
    """
    Inputs:
    -------
        p:      (B, N, 3)  torch.Tensor
        p_old:  (B, M, 3)  torch.Tensor
        x:      (B, M, channels)  torch.Tensor

    Outputs:
    --------
        interpolated_feats: (B, M, channels) torch.Tensor

    Notes:
    -----
    * This function operates entirely on the cpu
    """

    device = 'cpu'
    p = p.to(device=device)
    p_old = p_old.to(device=device)
    x = x.to(device=device)

    _p = torch.vstack(tuple(p[i, ...] for i in range(p.size(0))))
    _p_old = torch.vstack(tuple(p_old[i, ...] for i in range(p_old.size(0))))
    _x = torch.vstack(tuple(x[i, ...] for i in range(x.size(0))))

    batch = torch.kron(torch.arange(start=0, end=p.size(0)), torch.ones(p.size(1)))
    batch_old = torch.kron(torch.arange(start=0, end=p_old.size(0)), torch.ones(p_old.size(1)))
    batch = batch.long()
    batch_old = batch_old.long()
    _interpolated_feats = knn_interpolate(x=_x, pos_x=_p, pos_y=_p_old, batch_x=batch, batch_y=batch_old, k=3)

    interpolated_feats = torch.reshape(_interpolated_feats, (p_old.size(0), -1, x.size(-1)))
    interpolated_feats = interpolated_feats.cuda()

    return interpolated_feats

    # # important code pieces
    # a = torch.rand(5, 10, 4)
    # b = torch.rand(5, 5, 4)
    # c = torch.stack(tuple(torch.vstack((a[i, ...], b[i, ...])) for i in range(a.size(0))))




def farthest_point_sampling(xyz, npoints):
    """
    Inputs:
    ----------
    xyz     : torch.Tensor
              (B, N, 3) tensor where N > npoints
    npoints : int32
              number of features in the sampled set

    Outputs:
    -------
    out     : torch.Tensor
              (B, npoints, 3) tensor containing the set

    Note
    -----
    * This function operates entirely on the cpu
    """

    device = 'cpu'
    xyz = xyz.to(device=device)

    _xyz = torch.vstack(tuple(xyz[i, ...] for i in range(xyz.size(0))))
    batch = torch.kron(torch.arange(start=0, end=xyz.size(0)), torch.ones(xyz.size(1)))
    batch = batch.long()
    ratio = npoints/xyz.size(-2)
    index = fps(_xyz, batch, ratio=ratio, random_start=True)
    _xyz_out = _xyz[index]
    out = torch.reshape(_xyz_out, (xyz.size(0), -1, 3))
    out = out.cuda()
    return out




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def kNN(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    outputs
        neighbors: (B * N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
                   It is flattened along batch dimension.
    """
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert (
        query.shape[0] == dataset.shape[0]
    ), "Input tensors should have same batch size."
    assert (
        query.shape[2] == dataset.shape[2]
    ), "Input tensors should have same dimension."

    B, N1, _ = dataset.shape

    query_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query))
    dataset_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(dataset))

    indices = []
    for i in range(query_o3d.shape[0]):
        _query = query_o3d[i]
        _dataset = dataset_o3d[i]
        nns = o3d.core.nns.NearestNeighborSearch(_dataset)
        status = nns.knn_index()
        if not status:
            raise Exception("Index failed.")
        neighbors, _ = nns.knn_search(_query, k)
        # calculate prefix sum of indices
        # neighbors += N1 * i
        indices.append(torch.utils.dlpack.from_dlpack(neighbors.to_dlpack()))

    # flatten indices
    indices = torch.stack(indices)
    return indices


def kNN_torch(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    outputs
        neighbors: (B * N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
                   It is flattened along batch dimension.
    """
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert (
        query.shape[0] == dataset.shape[0]
    ), "Input tensors should have same batch size."
    assert (
        query.shape[2] == dataset.shape[2]
    ), "Input tensors should have same dimension."

    dists = square_distance(query, dataset)  # dists: [B, N0, N1]
    neighbors = dists.argsort()[:, :, :k]  # neighbors: [B, N0, k]
    torch.cuda.empty_cache()
    return neighbors


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def idx_pt(pts, idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(pts, 1, idx[..., None].expand(-1, -1, pts.size(-1)))
    return res.reshape(*raw_size, -1)


class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k):
        super().__init__()

        self.prev_linear = nn.Linear(dim, dim)

        self.k = k

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim)
        )

        self.final_linear = nn.Linear(dim, dim)

    def forward(self, x, pos):
        # queries, keys, values

        x_pre = x

        knn_idx = kNN_torch(pos, pos, self.k)
        knn_xyz = index_points(pos, knn_idx)

        q = self.to_q(x)
        k = idx_pt(self.to_k(x), knn_idx)
        v = idx_pt(self.to_v(x), knn_idx)

        pos_enc = self.pos_mlp(pos[:, :, None] - knn_xyz)

        attn = self.attn_mlp(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)

        agg = einsum('b i j d, b i j d -> b i d', attn, v + pos_enc)
        agg = self.final_linear(agg) + x_pre

        return agg


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio, fast=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        self.fast = fast
        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, x, p):
        """
        inputs
            x: (B, N, in_channels) shaped torch Tensor (A set of feature vectors)
            p: (B, N, 3) shaped torch Tensor (3D coordinates)
        outputs
            y: (B, M, out_channels) shaped torch Tensor
            p_out: (B, M, 3) shaped torch Tensor
        M = N * sampling ratio
        """
        B, N, _ = x.shape
        M = int(N * self.sampling_ratio)

        # 1: Farthest Point Sampling
        # p_flipped = p.transpose(1, 2).contiguous()
        p_out = farthest_point_sampling(xyz=p, npoints=M)
        # p_out = (
        #     gather_operation(
        #         p_flipped, farthest_point_sample(p, M)
        #     )
        #         .transpose(1, 2)
        #         .contiguous()
        # )  # p_out: (B, M, 3)

        # 2: kNN & MLP
        knn_fn = kNN_torch if self.fast else kNN
        neighbors = knn_fn(p_out, p, self.k)  # neighbors: (B, M, k)

        # 2-1: Apply MLP onto each feature
        x_flipped = x.transpose(1, 2).contiguous()
        mlp_x = (
            self.mlp(x_flipped).transpose(1, 2).contiguous()
        )  # mlp_x: (B, N, out_channels)

        # 2-2: Extract features based on neighbors
        features = index_points(mlp_x, neighbors)  # features: (B, M, k, out_channels)

        # 3: Local Max Pooling
        y = torch.max(features, dim=2)[0]  # y: (B, M, out_channels)

        return y, p_out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )
        self.lateral_mlp = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x, p, x_old, p_old):
        """
        Inputs:
            x:     (B, N, in_channels) torch.Tensor
            p:     (B, N, 3) torch.Tensor
            x_old: (B, M, out_channels) torch.Tensor
            p_old: (B, M, 3) torch.Tensor

        Outputs:
            y:     (B, M, out_channels) torch.Tensor
            p_old: (B, M, 3) torch.Tensor

        Note: N is smaller than M.
        """
        #
        # x = self.up_mlp(x.transpose(1, 2).contiguous())
        # dist, idx = pt_utils.three_nn(p2, p)
        # dist_recip = 1.0 / (dist + 1e-8)
        # norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # weight = dist_recip / norm
        # interpolated_feats = pt_utils.three_interpolate(
        #     x, idx, weight
        # )
        # x2 = self.lateral_mlp(x2.transpose(1, 2).contiguous())
        x = self.up_mlp(x.transpose(1, 2).contiguous()).transpose(1, 2)
        interpolated_feats = three_interpolate(p=p, p_old=p_old, x=x)

        _x_old = self.lateral_mlp(x_old.transpose(1, 2).contiguous()).transpose(1, 2)
        y = interpolated_feats + _x_old

        return y.contiguous(), p_old


class PointTransformerSegment(nn.Module):
    def __init__(self, dim=[6, 32, 64, 128, 256, 512], output_dim=20, k=16, sampling_ratio=0.25):
        super().__init__()

        self.Encoder = nn.ModuleList()
        for i in range(len(dim) - 1):
            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i + 1], bias=False))
            else:
                self.Encoder.append(TransitionDown(in_channels=dim[i], out_channels=dim[i + 1], k=k, sampling_ratio=sampling_ratio, fast=True))
            self.Encoder.append(PointTransformerBlock(dim=dim[i + 1], k=k))
        self.Decoder = nn.ModuleList()

        for i in range(len(dim) - 1, 0, -1):
            if i == len(dim) - 1:
                self.Decoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.Decoder.append(TransitionUp(in_channels=dim[i + 1], out_channels=dim[i]))

            self.Decoder.append(PointTransformerBlock(dim=dim[i], k=k))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        Inputs:
        ------
            pointscloud: (B, N, 3+d)
                B = batch size
                N = number of points in a batch
                d = input feature dimension
                pointcloud[..., 0:3] = positions
                pointcloud[..., 3:]  = feature
        Outputs: (B, N, d_out)
        -------

        """
        # timer = Timer("forward")
        # timer.tic()
        xyz, features = self._break_up_pc(pc=pointcloud)
        features = features.transpose(1, 2).contiguous()

        l_xyz, l_features = [xyz], [features]

        for i in range(int(len(self.Encoder) / 2)):
            if i == 0:
                li_features = self.Encoder[2 * i](l_features[i])
                li_xyz = l_xyz[i]
            else:
                li_features, li_xyz = self.Encoder[2 * i](l_features[i], l_xyz[i])
            li_features = self.Encoder[2 * i + 1](li_features, li_xyz)

            l_features.append(li_features)
            l_xyz.append(li_xyz)
            del li_features, li_xyz
        D_n = int(len(self.Decoder) / 2)

        for i in range(D_n):
            if i == 0:
                l_features[D_n - i] = self.Decoder[2 * i](l_features[D_n - i])
                l_features[D_n - i] = self.Decoder[2 * i + 1](l_features[D_n - i], l_xyz[D_n - i])
            else:
                l_features[D_n - i], l_xyz[D_n - i] = self.Decoder[2 * i](l_features[D_n - i + 1], l_xyz[D_n - i + 1],
                                                                          l_features[D_n - i], l_xyz[D_n - i])
                l_features[D_n - i] = self.Decoder[2 * i + 1](l_features[D_n - i], l_xyz[D_n - i])

        del l_features[0], l_features[1:], l_xyz
        out = self.fc_layer(l_features[0].transpose(1, 2).contiguous())
        # timer.toc()

        return out.transpose(1, 2)


class PointTransformerCls(nn.Module):
    def __init__(self, output_dim=20, channels=[6, 32, 64, 128, 256, 512], k=16, sampling_ratio=0.25):
        super().__init__()

        channels.append(output_dim)
        assert len(channels) > 3

        self.prev_block = nn.Sequential(
            nn.Linear(3, channels[0]),
            nn.ReLU(True),
            nn.Linear(channels[0], channels[0]),
        )
        self.prev_transformer = PointTransformerBlock(channels[0], k)

        self.trans_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(1, len(channels) - 2):
            self.trans_downs.append(
                TransitionDown(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    k=k,
                    sampling_ratio=sampling_ratio,
                )
            )
            self.transformers.append(PointTransformerBlock(channels[i], k))

        self.final_block = nn.Sequential(
            nn.Linear(channels[-3], channels[-2]),
            nn.ReLU(True),
            nn.Linear(channels[-2], channels[-1]),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
        Forward pass of the network

        Parameters
        ---------
        pointcloud: Variable(torch.cuda.FloatTensor)
            (B, N, 3 + input_channels) tensor
            Point cloud to run predicts on
            Each point in the point-cloud MUST
            be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        # # Timers
        # t_prev = Timer("prev_block")
        # t_prev.tic()
        features = self.prev_block(xyz)
        # t_prev.toc()

        # t_prev_trs = Timer("prev_transformer")
        # t_prev_trs.tic()
        features = self.prev_transformer(features, xyz)
        # t_prev_trs.toc()

        # t_td = Timer("transition_down")
        # t_trs = Timer("transformer")
        for trans_down_layer, transformer_layer in zip(self.trans_downs, self.transformers):
            # t_td.tic()
            features, xyz = trans_down_layer(features, xyz)
            # t_td.toc()

            # t_trs.tic()
            features = transformer_layer(features, xyz)
            # t_trs.toc()

        # t_final = Timer("final_block")
        # t_final.tic()
        out = self.final_block(features.mean(1))
        # t_final.toc()
        return out


if __name__ == '__main__':

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    print('-'*20)

    # Test: farthest_point_sampling()
    print('Test: farthest_point_sampling()')
    pos = torch.rand(4, 10, 3)
    ratio = 0.7
    npoints = int(ratio*pos.size(1))
    print(pos.is_cuda)
    pos_sampled = farthest_point_sampling(xyz=pos, npoints=npoints)
    print(pos_sampled.size())
    print(pos_sampled.is_cuda)

    print(pos[0, ...])
    print(pos_sampled[0, ...])
    print('-'*20)

    # Test: PointTransformerBlock()
    print('Test: PointTransformerBlock()')
    f = torch.rand(4, 10, 8)
    pt_transformer = PointTransformerBlock(dim=8, k=4).to(device=device)
    f = f.to(device=device)
    pos = pos.to(device=device)
    print('Is f on CUDA:', f.is_cuda)
    print('Is pos on CUDA:', pos.is_cuda)
    f_out = pt_transformer(f, pos)
    print('Is f_out on CUDA:', f_out.is_cuda)
    print(f_out.size())
    print('-'*20)

    # Test: TransitionDown()
    print('Test: TransitionDown()')
    pt_down = TransitionDown(in_channels=8, out_channels=5, k=8, sampling_ratio=0.5)
    pt_down.to(device=device)
    f_out, p_out = pt_down(x=f, p=pos)
    print(f_out.size())
    print(p_out.size())
    print(f_out.is_cuda)
    print(p_out.is_cuda)
    print('-'*20)


    # Test: three_interpolate()
    print('Test: three_interpolate()')

    p_old = torch.rand(4, 10, 3)
    x_old = torch.rand(4, 10, 16)
    p = p_old[:, 1:4, :]
    x = x_old[:, 1:4, :]
    p = p.cuda()
    x = x.cuda()
    p_old = p_old.cuda()
    x_old = x_old.cuda()

    print(p.is_cuda)
    print(x.is_cuda)
    print(p_old.is_cuda)
    print(x_old.is_cuda)
    interpolated_features = three_interpolate(p=p, p_old=p_old, x=x)

    print(interpolated_features.size())
    print(interpolated_features.is_cuda)
    print('-'*20)

    # Test: TransitionUp()
    print('Test: TransitionUp()')

    p_old = torch.rand(4, 10, 3)
    x_old = torch.rand(4, 10, 16)
    p = p_old[:, 1:4, :]
    x = x_old[:, 1:4, :]
    p = p.cuda()
    x = x.cuda()
    p_old = p_old.cuda()
    x_old = x_old.cuda()

    print(p.is_cuda)
    print(x.is_cuda)
    print(p_old.is_cuda)
    print(x_old.is_cuda)

    pt_up = TransitionUp(in_channels=16, out_channels=16).to(device=device)

    y, p_old = pt_up(x=x, p=p, x_old=x_old, p_old=p_old)

    print(y.size())
    print(p_old.size())
    print(y.is_cuda)
    print(p_old.is_cuda)
    print("-"*20)

    # Test: PointTransformerSegment()
    print('Test: PointTransformerSegment()')

    pointcloud = torch.rand(4, 2000, 11).cuda()
    print(pointcloud.is_cuda)

    pt_segment = PointTransformerSegment(dim=[8, 32], output_dim=64).to(device=device)

    y = pt_segment(pointcloud)
    print(y.size())
    print(y.is_cuda)

    print('-'*20)



