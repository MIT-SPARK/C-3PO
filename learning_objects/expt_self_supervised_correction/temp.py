
import torch
import yaml
import sys
sys.path.append('../..')

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID
from learning_objects.utils.general import display_two_pcs
# from learning_objects.expt_self_supervised_correction.expt_supervised_training import visualize_kp_detectors
from learning_objects.expt_self_supervised_correction.expt_self_supervised_training import visualize_kp_detectors

if __name__ == "__main__":

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    only_categories = ["bottle", "chair", "guitar", "car"]

    visualize_kp_detectors(detector_type='pointnet', model_class_ids=model_class_ids, only_categories=only_categories)
    # visualize_kp_detectors(detector_type='point_transformer', model_class_ids=model_class_ids,
    #                        only_categories=only_categories)