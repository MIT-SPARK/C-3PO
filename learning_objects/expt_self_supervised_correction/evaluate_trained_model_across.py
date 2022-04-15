"""
The goal here is to evaluate the trained model for object A, on other object inputs.

"""

import torch
import yaml
import argparse
import sys

sys.path.append('../..')

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID
from learning_objects.utils.general import display_two_pcs
# from learning_objects.expt_self_supervised_correction.expt_supervised_training import visualize_kp_detectors
from learning_objects.expt_self_supervised_correction.self_supervised_training import visualize_kp_detectors

if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_trained_model_across.py "point_transformer" "chair" "pre" "table"   # evaluates chair model pre(sim)-trained on table eval_dataset
    >> python evaluate_trained_model_across.py "pointnet" "chair" "post" "airplane"

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("models_to_analyze", help="pre/post, for pre-trained or post-training models.", type=str)
    parser.add_argument("cross_class_name", help="specify ShapeNet class name on which you want to evaluate the model.", type=str)

    args = parser.parse_args()

    detector_type = args.detector_type
    class_name = args.class_name
    models_to_analyze = args.models_to_analyze
    cross_class_name = args.cross_class_name
    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.class_name)
    only_categories = [class_name]

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    cross_class_id = CLASS_ID[cross_class_name]
    cross_model_id = model_class_ids[cross_class_name]
    # print("Evaluating on: ", cross_class_name)
    # print("Cross class id: ", cross_class_id)
    # print("Cross model id: ", cross_model_id)

    visualize_kp_detectors(detector_type=detector_type,
                           model_class_ids=model_class_ids,
                           only_categories=only_categories,
                           visualize=False, evaluate_models=True,
                           models_to_analyze=models_to_analyze,
                           cross=True,
                           cross_model_id=cross_model_id,
                           cross_class_id=cross_class_id)