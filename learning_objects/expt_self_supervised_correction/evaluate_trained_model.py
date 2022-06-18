
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

    visualize_kp_detectors(detector_type=detector_type,
                           model_class_ids=model_class_ids,
                           only_categories=only_categories,
                           visualize=False, evaluate_models=True,
                           models_to_analyze=models_to_analyze,
                           use_corrector=True)

    ############# OLD

    # only_categories = ["bottle", "chair", "guitar", "car"]
    # visualize_kp_detectors(detector_type='pointnet', model_class_ids=model_class_ids, only_categories=only_categories,
    #                        evaluate_models=True, models_to_analyze='both', visualize=False)


    # only_categories = ["bottle"]
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='pre')
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='post')

    # only_categories = ["chair"]
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='pre')
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='post')

    # only_categories = ["guitar"]
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='pre')
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='post')

    # only_categories = ["car"]
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='pre')
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='post')

    # only_categories = ["airplane"]
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='pre')
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='post')

    # only_categories = ["motorcycle"]
    # visualize_kp_detectors(detector_type='pointnet', model_class_ids=model_class_ids,
    #                        only_categories=only_categories, visualize=False, evaluate_models=True,
    #                        models_to_analyze='pre')