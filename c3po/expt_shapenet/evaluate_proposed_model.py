import argparse
import sys
import torch
import yaml

sys.path.append('../..')

from c3po.datasets.shapenet import CLASS_NAME, CLASS_ID
from c3po.utils.visualization_utils import display_two_pcs
from c3po.expt_shapenet.self_supervised_training import evaluate_model

if __name__ == "__main__":

    """
    usage: 
    >> python evaluate_proposed_model.py \
    --detector "point_transformer" \
    --object "chair" \
    --model "post" \
    --dataset "shapenet.sim.hard"
    """


    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", help="specify the detector type.", type=str)
    parser.add_argument("--object", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("--model", help="pre/post, for pre-trained or post-training models.", type=str)
    parser.add_argument("--dataset", default="shapenet",
                        choices=["shapenet",
                                 "shapenet.sim.easy", "shapenet.sim.hard",
                                 "shapenet.real.easy", "shapenet.real.hard"], type=str)

    args = parser.parse_args()

    detector_type = args.detector
    class_name = args.object
    models_to_analyze = args.model
    dataset = args.dataset
    # print("KP detector type: ", args.detector_type)
    # print("CAD Model class: ", args.class_name)

    # breakpoint()
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
                   use_corrector=True,
                   dataset=dataset)
