import argparse
import sys
import yaml

sys.path.append('../..')

from c3po.datasets.shapenet import CLASS_NAME
from c3po.expt_shapenet.train_baseline import evaluate_model

if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_baseline.py \
    --detector "pointnet" \
    --object "chair" \
    --dataset "shapenet.sim.real" 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", help="specify the detector type.", type=str)
    parser.add_argument("--object", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("--dataset",
                        choices=["shapenet",
                                 "shapenet.sim.easy", "shapenet.sim.hard",
                                 "shapenet.real.easy", "shapenet.real.hard"], type=str)
    # parser.add_argument("models_to_analyze", help="pre/post, for pre-trained or post-training models.", type=str)

    args = parser.parse_args()

    detector_type = args.detector
    class_name = args.object
    dataset = args.dataset
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
                   visualize_before=False,
                   visualize_after=False,
                   dataset=dataset)