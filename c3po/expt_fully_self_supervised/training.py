import argparse
import os
import sys
import yaml
from datetime import datetime

sys.path.append("../../")

from c3po.datasets.shapenet import CLASS_ID

from c3po.expt_fully_self_supervised.full_self_supervised_training import train_detector \
    as train_detector_shapenet
from c3po.expt_fully_self_supervised.full_self_supervised_training_ycb import train_detector \
    as train_detector_ycb

def train_kp_detectors(detector_type, model_id, class_name=None, use_corrector=True, dataset="shapenet"):
    hyper_param_file = "full_self_supervised_training"
    if dataset == "ycb":
        hyper_param_file = hyper_param_file + "_ycb.yml"
    elif dataset == "shapenet":
        class_id = CLASS_ID[class_name]
        hyper_param_file = hyper_param_file + ".yml"
    else:
        raise Exception("Invalid dataset specified. Please specify shapenet or ycb.")
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]
    if dataset == "shapenet":
        hyper_param['epsilon'] = hyper_param['epsilon'][class_name]
    else:
        hyper_param['epsilon'] = hyper_param['epsilon'][model_id]

    kwargs = {'hyper_param': hyper_param, 'detector_type': detector_type,
              'model_id': model_id, 'use_corrector': use_corrector}
    print(">>"*40)
    if dataset == "shapenet":
        print("Training: ", class_name)
        kwargs['class_id'] = class_id
    print("Model ID:", str(model_id))

    if dataset == "shapenet":
        train_detector_shapenet(**kwargs)
    elif dataset == "ycb":
        train_detector_ycb(**kwargs)

if __name__ == "__main__":

    """
    usage: 
    >> python training.py "point_transformer" "chair" "self_supervised"
    >> python training.py "pointnet" "chair" "self_supervised"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("model_id", help="specify the ycb model id.", type=str)
    parser.add_argument("dataset", help="specify the dataset, shapenet or ycb", type=str)

    args = parser.parse_args()

    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.class_name)
    detector_type = args.detector_type
    dataset = args.dataset
    assert dataset in ["shapenet", "ycb"]
    if dataset == "shapenet":
        class_name = args.class_name
        stream = open("class_model_ids.yml", "r")
        model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
        if class_name not in model_class_ids:
            raise Exception('Invalid class_name')
        else:
            model_id = model_class_ids[class_name]
    elif dataset == "ycb":
        class_name = None
        stream = open("class_model_ids_ycb.yml", "r")
        model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']
        if model_id not in model_ids:
            raise Exception('Invalid model_id')
        model_id = args.model_id

    train_kp_detectors(detector_type=detector_type, class_name=class_name, model_id=model_id,
                       use_corrector=True, dataset=dataset)
