# -*- coding: future_fstrings -*-
import open3d as o3d  # prevent loading error

import numpy as np
import sys
import json
import logging
import torch
from easydict import EasyDict as edict

sys.path.append("../..")

from c3po.baselines.fcgf.liby.data_loaders import make_data_loader
from c3po.baselines.fcgf.config import get_config

from c3po.baselines.fcgf.liby.trainer import ContrastiveLossTrainer, HardestContrastiveLossTrainer, \
    TripletLossTrainer, HardestTripletLossTrainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def get_trainer(trainer):
  if trainer == 'ContrastiveLossTrainer':
    return ContrastiveLossTrainer
  elif trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'TripletLossTrainer':
    return TripletLossTrainer
  elif trainer == 'HardestTripletLossTrainer':
    return HardestTripletLossTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):
  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_threads=config.train_num_thread)

  if config.test_valid:
    val_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        num_threads=config.val_num_thread)
  else:
    val_loader = None

  ds = train_loader.dataset
  (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans) = ds[0]
  # (n, 3), (m, 3), (n, 3), (m, 3), (n, 1), (m, 1), list, (4, 4)

  zzz0 = np.hstack([xyz0, np.ones((xyz0.shape[0], 1))])
  zzz1 = np.hstack([xyz1, np.ones((xyz1.shape[0], 1))])
  zzz2 = (trans @ zzz0.T).T
  xyz2 = zzz2[:, :3]


  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
  )

  trainer.train()


if __name__ == "__main__":
  logger = logging.getLogger()
  config = get_config()

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)
