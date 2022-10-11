import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature
from util.misc import extract_features

from model.resunet import ResUNetBN2C

import torch

if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
  print('Downloading weights...')
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
      'ResUNetBN2C-16feat-3conv.pth')


def fcgf(source_points, target_points):
  """
  source_points : torch.tensor of shape (3, n)
  target_points : torch.tensor of shape (3, m)

  source_down   : torch.tensor of shape (3, l)
  target_down   : torch.tensor of shape (3, k)
  source_feat   : torch.tensor of shape (d, l)
  target_feat   : torch.tensor of shape (d, k)
  """
  # setup device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # load pre-trained model
  checkpoint = torch.load('ResUNetBN2C-16feat-3conv.pth')
  model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()
  model = model.to(device)

  # apply model
  #ToDo: modify this.
  # pcd = o3d.io.read_point_cloud(config.input)
  # - figuring out how to ger correspondances between the two point clouds.
  # - working on teaser, instead.
  src = source_points.transpose(-1, -2).numpy()     # (n, 3) np.array
  tar = target_points.transpose(-1, -2).numpy()     # (m, 3) np.array
  src_down, src_feature = extract_features(
      model,
      xyz=src,
      voxel_size=config.voxel_size,
      device=device,
      skip_check=True)

  tar_down, tar_feature = extract_features(
      model,
      xyz=tar,
      voxel_size=config.voxel_size,
      device=device,
      skip_check=True)




  vis_pcd = o3d.geometry.PointCloud()
  vis_pcd.points = o3d.utility.Vector3dVector(xyz_down)

  vis_pcd = get_colored_point_cloud_feature(vis_pcd,
                                            feature.detach().cpu().numpy(),
                                            config.voxel_size)
  o3d.visualization.draw_geometries([vis_pcd])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--input',
      default='redkitchen-20.ply',
      type=str,
      help='path to a pointcloud file')
  parser.add_argument(
      '-m',
      '--model',
      default='ResUNetBN2C-16feat-3conv.pth',
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.025,
      type=float,
      help='voxel size to preprocess point cloud')

  config = parser.parse_args()
  demo(config)
