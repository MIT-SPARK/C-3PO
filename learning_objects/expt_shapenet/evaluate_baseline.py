import argparse
import sys
import yaml

sys.path.append('../..')

from learning_objects.datasets.shapenet import CLASS_NAME
from learning_objects.expt_shapenet.train_baseline import evaluate_model

if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_baseline.py "point_transformer" "chair"
    >> python evaluate_baseline.py "pointnet" "chair"

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)
    # parser.add_argument("models_to_analyze", help="pre/post, for pre-trained or post-training models.", type=str)

    args = parser.parse_args()

    detector_type = args.detector_type
    class_name = args.class_name
    # models_to_analyze = args.models_to_analyze
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
                   visualize=True, evaluate_models=True,
                   use_corrector=False,
                   visualize_before=True,
                   visualize_after=False)