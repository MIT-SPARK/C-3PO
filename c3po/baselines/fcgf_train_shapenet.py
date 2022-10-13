# -*- coding: future_fstrings -*-

import numpy as np
import sys
import argparse
import logging
import torch

sys.path.append("../..")

from c3po.baselines.fcgf.config import get_config
from c3po.baselines.fcgf.liby.data_loaders_shapenet import make_data_loader
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

#TODO:
# - verify the extraction of feature correspondences.


def main(config):

    config.dataset_length = 2000
    train_dl = make_data_loader(type=config.type,
                                dataset_length=config.dataset_length,
                                batch_size=config.batch_size,
                                voxel_size=config.voxel_size,
                                positive_pair_search_voxel_size_multiplier=config.positive_pair_search_voxel_size_multiplier
                                )

    config.dataset_length = 50
    val_dl = make_data_loader(type=config.type,
                              dataset_length=config.dataset_length,
                              batch_size=config.val_batch_size,
                              voxel_size=config.voxel_size,
                              positive_pair_search_voxel_size_multiplier=config.positive_pair_search_voxel_size_multiplier
                              )

    Trainer = get_trainer(config.trainer)
    trainer = Trainer(
      config=config,
      data_loader=train_dl,
      val_data_loader=val_dl,
    )

    trainer.train()


if __name__ == "__main__":

    config = get_config()
    config.out_dir = config.out_dir + "/" + str(config.type)
    # config.type = 'sim' # 'real'
    # config.max_epoch = 2
    # config.voxel_size = 0.025
    # config.batch_size = 4
    # config.positive_pair_search_voxel_size_multiplier = 1.5
    # config.dataset_length = 2000
    # breakpoint()

    main(config)
