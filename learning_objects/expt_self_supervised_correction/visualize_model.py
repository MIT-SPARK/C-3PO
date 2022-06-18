
import torch
import yaml
import argparse
import sys

sys.path.append('../..')

from learning_objects.expt_self_supervised_correction.self_supervised_training import visualize_kp_detectors

if __name__ == "__main__":
    """
    usage: 
    >> python visualize_model.py "point_transformer" "chair" "pre"
    >> python visualize_model.py "pointnet" "chair" "post"
    
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

    visualize_kp_detectors(detector_type=detector_type,
                           model_class_ids=model_class_ids,
                           only_categories=only_categories,
                           visualize=True,
                           evaluate_models=False,
                           use_corrector=False,
                           models_to_analyze=models_to_analyze)