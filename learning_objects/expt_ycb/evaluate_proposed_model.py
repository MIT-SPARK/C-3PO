import argparse
import sys
import torch
import yaml

sys.path.append('../..')

from learning_objects.utils.visualization_utils import display_two_pcs
from learning_objects.expt_ycb.self_supervised_training import evaluate_model

if __name__ == "__main__":

    """
    usage: 
    >> python evaluate_proposed_model.py "point_transformer" "021_bleach_cleanser" "pre"
    >> python evaluate_proposed_model.py "point_transformer" "021_bleach_cleanser" "post"
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("model_id", help="specify the ycb model id.", type=str)
    parser.add_argument("models_to_analyze", help="pre/post, for pre-trained or post-training models.", type=str)


    args = parser.parse_args()

    print("KP detector type: ", args.detector_type)
    print("CAD Model class: ", args.model_id)
    detector_type = args.detector_type
    model_id = args.model_id
    models_to_analyze = args.models_to_analyze


    # keeping for code monkey param happiness
    with open("model_ids.yml", 'r') as stream:
        model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']
        if model_id not in model_ids:
            raise Exception('Invalid model_id')

    evaluate_model(detector_type=detector_type,
                   model_id=model_id,
                   evaluate_models=True,
                   visualize=False,
                   use_corrector=True,
                   models_to_analyze=models_to_analyze,
                   degeneracy_eval=False,
                   average_metrics=False)


