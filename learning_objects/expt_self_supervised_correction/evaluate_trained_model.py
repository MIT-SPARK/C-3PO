import argparse
import sys
import torch
import yaml

sys.path.append('../..')

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID
from learning_objects.utils.general import display_two_pcs
from learning_objects.expt_self_supervised_correction.self_supervised_training import evaluate_model

if __name__ == "__main__":

    """
    usage: 
    >> python evaluate_trained_model.py "point_transformer" "chair" "pre"
    >> python evaluate_trained_model.py "pointnet" "chair" "post"
    
    """


    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("models_to_analyze", help="pre/post, for pre-trained or post-training models.", type=str)

    args = parser.parse_args()

    detector_type = args.detector_type
    class_name = args.class_name
    models_to_analyze = args.models_to_analyze
    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.class_name)
    only_categories = [class_name]

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    if class_name not in model_class_ids:
        raise Exception('Invalid class_name')
    else:
        model_id = model_class_ids[class_name]

    evaluate_model(detector_type=detector_type,
                   class_name=class_name,
                   model_id=model_id,
                   visualize=False, evaluate_models=True,
                   models_to_analyze=models_to_analyze,
                   use_corrector=True)
