"""
This code uses files from self_supervised_training.py to generate trained keypoint detectors on depth pc.

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
from learning_objects.expt_self_supervised_correction.self_supervised_training import train_detector, visualize_detector



def train_kp_detectors(detector_type, model_class_ids, only_categories=None):

    for key, value in model_class_ids.items():
        if key in only_categories:
            class_id = CLASS_ID[key]
            model_id = str(value)

            hyper_param_file = "self_supervised_training_" + detector_type + ".yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)

            print(">>"*40)
            print("Training: ", key, "; Model ID:", str(model_id))
            train_detector(detector_type=detector_type,
                           class_id=class_id,
                           model_id=model_id,
                           hyper_param=hyper_param)


if __name__ == "__main__":

    # only_categories = ["airplane", "bathtub", "car", "chair", "guitar", "knife", "motorcycle", "skateboard", "table"]
    only_categories = ["car", "chair", "guitar", "motorcycle"]

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    train_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids, only_categories=only_categories)
    # train_kp_detectors(detector_type='pointnet', model_class_ids=model_class_ids, only_categories=only_categories)

