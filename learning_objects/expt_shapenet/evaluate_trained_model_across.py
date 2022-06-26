"""
The goal here is to evaluate the trained model for object A, on other object inputs.

"""
import argparse
import sys
import torch
import yaml

sys.path.append('../..')

from learning_objects.datasets.shapenet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID
from learning_objects.utils.visualization_utils import display_two_pcs
from learning_objects.expt_shapenet.self_supervised_training import evaluate_model

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
    if class_name not in model_class_ids:
        raise Exception('Invalid class_name')
    else:
        model_id = model_class_ids[class_name]

    evaluate_model(detector_type=detector_type,
                   class_name=class_name, model_id=model_id,
                   visualize=False, use_corrector=True, evaluate_models=True,
                   models_to_analyze=models_to_analyze,
                   cross=True,
                   cross_model_id=cross_model_id,
                   cross_class_id=cross_class_id)