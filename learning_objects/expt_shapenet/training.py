import argparse
import os
import sys
import yaml
from datetime import datetime

sys.path.append("../../")

from learning_objects.datasets.shapenet import CLASS_NAME, CLASS_ID

from learning_objects.expt_shapenet.self_supervised_training import train_detector as train_detector_self_supervised
from learning_objects.expt_shapenet.supervised_training import train_detector as train_detector_supervised
from learning_objects.expt_shapenet.train_baseline import train_detector as train_detector_baseline


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
    use_corrector = True if train_mode == "self_supervised" else False

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    if class_name not in model_class_ids:
        raise Exception('Invalid class_name')
    else:
        model_id = model_class_ids[class_name]

    train_kp_detectors(detector_type=detector_type, class_name=class_name, model_id=model_id,
                       use_corrector=use_corrector, train_mode=train_mode)