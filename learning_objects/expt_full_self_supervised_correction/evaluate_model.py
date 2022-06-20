import argparse
import sys
import torch
import yaml

sys.path.append('../..')

from learning_objects.expt_full_self_supervised_correction.full_self_supervised_training import visualize_kp_detectors
from learning_objects.expt_full_self_supervised_correction.full_self_supervised_training_ycb import visualize_kp_detectors \
    as visualize_kp_detectors_ycb

if __name__ == "__main__":
    """
    usage: 
    shapenet
    >> python evaluate_trained_model.py "point_transformer" "chair" "pre" "table" "shapenet"
    >> python evaluate_trained_model.py "pointnet" "chair" "post" "chair" "shapenet"
    
    ycb
    >> python full_self_supervised_training.py "point_transformer" "003_cracker_box" "post" "003_cracker_box" "ycb"


    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name or YCB model id.", type=str)
    parser.add_argument("models_to_analyze", help="pre/post, for pre-trained or post-training models.", type=str)
    parser.add_argument("test_class_name", help="specify the ShapeNet class name or YCB model id, on whom the model will be tested",
                        type=str)
    parser.add_argument("dataset_name", help="specify the dataset, shapenet or ycb",
                        type=str)

    args = parser.parse_args()

    detector_type = args.detector_type
    class_name = args.class_name
    models_to_analyze = args.models_to_analyze
    test_class_name = args.test_class_name
    dataset = args.dataset_name
    if dataset == "shapenet":
        only_categories = [class_name]

        stream = open("class_model_ids.yml", "r")
        model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

        visualize_kp_detectors(detector_type=detector_type,
                               model_class_ids=model_class_ids,
                               only_categories=only_categories,
                               test_class_name=test_class_name,
                               visualize=False,
                               evaluate_models=True,
                               use_corrector=True,
                               models_to_analyze=models_to_analyze)
    elif dataset == "ycb":
        only_models = [class_name]
        with open("class_model_ids_ycb.yml", 'r') as stream:
            model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']
        visualize_kp_detectors_ycb(detector_type=detector_type,
                                   model_ids=model_ids,
                                   only_models=only_models,
                                   test_model_name=test_class_name,
                                   visualize=False,
                                   evaluate_models=True,
                                   use_corrector=True,
                                   models_to_analyze=models_to_analyze)
    else:
        raise Exception("Invalid dataset_name, please choose either shapenet or ycb")
