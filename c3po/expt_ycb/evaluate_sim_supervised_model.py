import argparse
import sys
import torch
import yaml
import os
import pickle
import numpy as np

sys.path.append('../..')

from c3po.utils.visualization_utils import display_two_pcs
from c3po.expt_ycb.self_supervised_training import evaluate_model
from c3po.expt_ycb.proposed_model import ProposedRegressionModel as ProposedModel
from c3po.datasets.ycb import DepthYCB
from c3po.datasets.ycb_eval import YCB
from c3po.datasets.utils_dataset import toFormat
from c3po.expt_shapenet.evaluation import evaluate
from c3po.expt_shapenet.self_supervised_training import visual_test


def evaluate_model(detector_type, model_id,
                   evaluate_models=True,
                   visualize=True,
                   use_corrector=True,
                   models_to_analyze='post',
                   degeneracy_eval=False,
                   average_metrics=False,
                   dataset=None):

    if models_to_analyze == 'pre':
        evaluate_pretrained = True
        evaluate_trained = False
    elif models_to_analyze == 'post':
        evaluate_pretrained = False
        evaluate_trained = True
    else:
        return NotImplementedError

    hyper_param_file = "self_supervised_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]
    hyper_param['epsilon'] = hyper_param['epsilon'][model_id]
    print(">>" * 40)
    print("Analyzing Trained Model for Object: " + str(model_id))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + '/' + model_id + '/'
    best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '_data_augment.pth'
    best_post_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + 'data_augment.pth'
    hyper_param_file = best_model_save_location + '_self_supervised_kp_point_transformer_hyperparams.pth'
    if os.path.exists(hyper_param_file):
        with open(hyper_param_file, "rb") as f:
            hyper_param = pickle.load(f)

    metrics_file_base = best_model_save_location + "_averaged_eval_metrics"
    if evaluate_pretrained:
        metrics_file_base = metrics_file_base + "_sim_trained"
    if degeneracy_eval:
        metrics_file_base = metrics_file_base + "_with_degeneracy"
    if not use_corrector:
        metrics_file_base = metrics_file_base + "_no_corrector"
    metrics_file = metrics_file_base + ".pkl"

    if average_metrics:
        if os.path.exists(metrics_file):
            with open(metrics_file, "rb") as f:
                averaged_metrics = pickle.load(f)
        else:
            averaged_metrics = [0] * 6 if not degeneracy_eval else [0] * 14
        print("LOADING AVERAGE OF " + str(averaged_metrics[-1]) + " RUNS!")
        print(averaged_metrics)
        averaged_metrics = np.array(averaged_metrics)

    # Evaluation
    # test dataset:
    if dataset is None or dataset == "ycb":
        eval_dataset = DepthYCB(model_id=model_id,
                                split='test',
                                only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                                num_of_points=hyper_param['num_of_points_to_sample'])
        eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else hyper_param['eval_batch_size'][model_id]

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)
        data_type = "ycb"
    else:

        type = dataset.split('.')[1]
        eval_dataset = YCB(type=type, object=model_id, length=50, num_points=1024, split="test")
        eval_dataset = toFormat(eval_dataset)

        eval_batch_size = len(eval_dataset) if hyper_param['only_load_nondegenerate_pcds'] else \
            hyper_param['eval_batch_size'][model_id]

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                                  pin_memory=True)
        data_type = dataset

    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    model = ProposedModel(model_id=model_id, model_keypoints=model_keypoints, cad_models=cad_models,
                          keypoint_detector=detector_type, local_max_pooling=False, correction_flag=use_corrector,
                          need_predicted_keypoints=True).to(device)

    if evaluate_pretrained:
        if not os.path.isfile(best_pre_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
        state_dict = torch.load(best_pre_model_save_file)
    elif evaluate_trained:
        if not os.path.isfile(best_post_model_save_file):
            print("ERROR: CAN'T LOAD TRAINED REGRESSION MODEL, PATH DOESN'T EXIST")
        state_dict = torch.load(best_post_model_save_file, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)
    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters: ", num_parameters)

    # Evaluation:
    if evaluate_models:
        print(">>"*40)
        log_info = "PRE-TRAINED MODEL:" if evaluate_pretrained else "(SELF-SUPERVISED) TRAINED MODEL:"
        print(log_info)
        print(">>" * 40)
        if evaluate_trained:
            raise ValueError("Running incorrect file.")
            # log_dir = "eval/c3po/" + detector_type
        elif evaluate_pretrained:
            log_dir = "eval/KeyPoSim/" + detector_type
        else:
            raise ValueError("Incorrectly specified model type.")
        # eval_metrics = evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param, certification=True,
        #          device=device, normalize_adds=False, degeneracy=degeneracy_eval)
        evaluate(eval_loader=eval_loader, model=model, hyper_param=hyper_param,
                 device=device, log_dir=log_dir, data_type=data_type)
    # # Visual Test
    dataset_batch_size = 1
    dataset = DepthYCB(model_id=model_id,
                            split='test',
                            only_load_nondegenerate_pcds= hyper_param['only_load_nondegenerate_pcds'],
                            num_of_points=hyper_param['num_of_points_to_sample'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False, pin_memory=True)

    if visualize:
        print(">>" * 40)
        log_info = "VISUALIZING PRE-TRAINED MODEL:" if evaluate_pretrained else "VISUALIZING (SELF-SUPERVISED) TRAINED MODEL:"
        print(log_info)
        print(">>" * 40)
        visual_test(test_loader=loader, model=model, hyper_param=hyper_param, degeneracy_eval=degeneracy_eval)

    del state_dict, model
    # if average_metrics and evaluate_models:
    #     eval_metrics = eval_metrics + [1]
    #     averaged_metrics = averaged_metrics + np.array(eval_metrics)
    #     for elt_idx in range(len(averaged_metrics)):
    #         if np.isnan(averaged_metrics[elt_idx]):
    #             averaged_metrics[elt_idx] = 0
    #     print("NEW AVERAGED METRICS: ", averaged_metrics)
    #     with open(metrics_file, 'wb+') as f:
    #         pickle.dump(averaged_metrics, f)
    # elif evaluate_models:
    #     return eval_metrics
    # else:
    #     return None

    return None


if __name__ == "__main__":

    """
    usage:  
    >> python evaluate_sim_supervised_model.py \
    --detector "point_transformer" \
    --object "021_bleach_cleanser" \
    --dataset "ycb.real" 

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", help="specify the detector type.", type=str)
    parser.add_argument("--object", help="specify the ycb model id.", type=str)
    parser.add_argument("--dataset", choices=["ycb", "ycb.sim", "ycb.real"], type=str)

    args = parser.parse_args()

    print("KP detector type: ", args.detector)
    print("CAD Model class: ", args.object)
    detector_type = args.detector
    model_id = args.object
    dataset = args.dataset

    # keeping for code monkey param happiness
    with open("model_ids.yml", 'r') as stream:
        model_ids = yaml.load(stream=stream, Loader=yaml.Loader)['model_ids']
        if model_id not in model_ids:
            raise Exception('Invalid model_id')

    evaluate_model(detector_type=detector_type,
                   model_id=model_id,
                   evaluate_models=True,
                   visualize=False,
                   use_corrector=False,
                   models_to_analyze="pre",
                   degeneracy_eval=False,
                   average_metrics=False,
                   dataset=dataset)