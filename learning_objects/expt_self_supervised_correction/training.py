import torch
import yaml
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
import sys
sys.path.append("../../")

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPointCloud2, DepthPC, CLASS_NAME, \
    FixedDepthPC, CLASS_ID
from learning_objects.utils.general import display_results, TrackingMeter

# loss functions
from learning_objects.expt_self_supervised_correction.loss_functions import certify
from learning_objects.expt_self_supervised_correction.loss_functions import self_supervised_training_loss \
    as self_supervised_loss
from learning_objects.expt_self_supervised_correction.loss_functions import self_supervised_validation_loss \
    as validation_loss
from learning_objects.expt_self_supervised_correction.evaluation_metrics import evaluation_error, add_s_error
from learning_objects.expt_self_supervised_correction.evaluation import evaluate
from learning_objects.expt_self_supervised_correction.proposed_model import ProposedRegressionModel as ProposedModel
from learning_objects.expt_self_supervised_correction.self_supervised_training import train_detector as train_detector_self_supervised
from learning_objects.expt_self_supervised_correction.supervised_training import train_detector as train_detector_supervised
from learning_objects.expt_self_supervised_correction.train_baseline import train_detector as train_detector_baseline


def train_kp_detectors(detector_type, class_name, model_id, use_corrector=True, train_mode="self_supervised"):
    class_id = CLASS_ID[class_name]

    hyper_param_file = train_mode + "_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    if not train_mode == "supervised":
        hyper_param = hyper_param[detector_type]
        hyper_param['epsilon'] = hyper_param['epsilon'][class_name]

    print(">>"*40)
    print("Training: ", class_name, "; Model ID:", str(model_id))
    kwargs = {'hyper_param': hyper_param, 'detector_type': detector_type, 'class_id': class_id,
              'model_id': model_id, 'use_corrector': use_corrector}
    if train_mode == "self_supervised":
        train_detector_self_supervised(**kwargs)
    elif train_mode == "supervised":
        train_detector_supervised(**kwargs)
    elif train_mode == "baseline":
        train_detector_baseline(**kwargs)

if __name__ == "__main__":

    """
    usage: 
    >> python training.py "point_transformer" "chair" "self_supervised"
    >> python training.py "pointnet" "chair" "self_supervised"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("train_mode", help="specify the training mode, self_supervised, supervised, or baseline", type=str)

    args = parser.parse_args()

    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.class_name)
    detector_type = args.detector_type
    class_name = args.class_name
    train_mode = args.train_mode
    assert train_mode in ["self_supervised", "supervised", "baseline"]
    only_categories = [class_name]

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    if class_name not in model_class_ids:
        raise Exception('Invalid class_name')
    else:
        model_id = model_class_ids[class_name]

    train_kp_detectors(detector_type=detector_type, class_name=class_name, model_id=model_id,
                       use_corrector=True, train_mode=train_mode)