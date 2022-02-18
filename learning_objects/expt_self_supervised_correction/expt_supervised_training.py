"""
This code uses files from supervised_training.py to generate trained keypoint detectors for our experiments.

"""

import torch
import yaml
from pytorch3d import ops
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID
from learning_objects.expt_self_supervised_correction.supervised_training import train_detector, visualize_detector


def train_kp_detectors(detector_type, model_class_ids):

    for key, value in model_class_ids.items():
        class_id = CLASS_ID[key]
        model_id = str(value)

        stream = open("supervised_training.yml", "r")
        hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)

        print(">>"*40)
        print("Training: ", key, "; Model ID:", str(model_id))
        train_detector(hyper_param=hyper_param,
                       detector_type=detector_type,
                       class_id=class_id,
                       model_id=model_id)
        torch.cuda.empty_cache()




def visualize_kp_detectors(detector_type, model_class_ids):

    for key, value in model_class_ids.items():
        class_id = CLASS_ID[key]
        model_id = str(value)

        stream = open("supervised_training.yml", "r")
        hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)

        print(">>"*40)
        print("Visualizing: ", key, "; Model ID:", str(model_id))

        visualize_detector(detector_type=detector_type,
                           class_id=class_id,
                           model_id=model_id,
                           hyper_param=hyper_param)
        torch.cuda.empty_cache()


if __name__ == "__main__":

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    # train_kp_detectors(detector_type='pointnet', model_class_ids=model_class_ids)
    train_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids)





