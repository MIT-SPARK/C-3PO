import argparse
import os
import sys
import torch
import yaml

sys.path.append('../..')

from learning_objects.datasets.shapenet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID, FixedDepthPC
from learning_objects.expt_shapenet.evaluation import evaluate
from learning_objects.utils.visualization_utils import display_results
from learning_objects.utils.loss_functions import certify
from learning_objects.utils.evaluation_metrics import add_s_error
from learning_objects.expt_shapenet.proposed_model import ProposedRegressionModel as ProposedModel


def visual_test(test_loader, model, hyper_param, device=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cad_models = test_loader.dataset._get_cad_models()
    cad_models = cad_models.to(device)

    for i, vdata in enumerate(test_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        input_point_cloud = input_point_cloud.to(device)
        keypoints_target = keypoints_target.to(device)
        R_target = R_target.to(device)
        t_target = t_target.to(device)

        # Make predictions for this batch
        model.eval()
        predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, _, predicted_model_keypoints \
            = model(input_point_cloud)

        # certification
        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=predicted_keypoints,
                        predicted_model_keypoints=predicted_model_keypoints,
                        epsilon=hyper_param['epsilon'])

        print("Certifiable: ", certi)

        # add-s
        pc_t = R_target @ cad_models + t_target
        add_s = add_s_error(predicted_point_cloud=predicted_point_cloud,
                            ground_truth_point_cloud=pc_t,
                            threshold=hyper_param['adds_threshold'])
        print("ADD-S: ", add_s)

        pc = input_point_cloud.clone().detach().to('cpu')
        pc_p = predicted_point_cloud.clone().detach().to('cpu')
        pc_t = pc_t.clone().detach().to('cpu')
        kp = keypoints_target.clone().detach().to('cpu')
        kp_p = predicted_keypoints.clone().detach().to('cpu')
        print("DISPLAY: INPUT PC")
        display_results(input_point_cloud=pc, detected_keypoints=kp, target_point_cloud=pc,
                        target_keypoints=None)
        print("DISPLAY: INPUT AND PREDICTED PC")
        display_results(input_point_cloud=pc, detected_keypoints=kp_p, target_point_cloud=pc_p,
                        target_keypoints=kp)
        print("DISPLAY: TRUE AND PREDICTED PC")
        display_results(input_point_cloud=pc_p, detected_keypoints=kp_p, target_point_cloud=pc_t,
                        target_keypoints=kp)

        del pc, pc_p, kp, kp_p, pc_t
        del input_point_cloud, keypoints_target, R_target, t_target, \
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        if i >= 9:
            break


def visualize_detector(hyper_param, detector_type, class_id, model_id,
                       evaluate_models=True, models_to_analyze='post',
                       use_corrector=False,
                       visualize=False, device=None):
    """

    """
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if models_to_analyze == 'pre':
        pre_ = True
        post_ = False
    elif models_to_analyze == 'post':
        pre_ = False
        post_ = True
    else:
        return NotImplementedError



    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'
    best_post_model_save_file = best_model_save_location + '_best_self_supervised_kp_' + detector_type + '.pth'

    # Evaluation
    # validation dataset:
    eval_dataset_len = hyper_param['eval_dataset_len']
    eval_batch_size = hyper_param['eval_batch_size']
    eval_dataset = FixedDepthPC(class_id=class_id, model_id=model_id,
                                n=hyper_param['num_of_points_selfsupervised'],
                                num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                                dataset_len=eval_dataset_len,
                                rotate_about_z=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)


    # model
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    if pre_:
        model_before = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                                     keypoint_detector=detector_type, correction_flag=use_corrector,
                                     need_predicted_keypoints=True).to(device)

        if not os.path.isfile(best_pre_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

        state_dict_pre = torch.load(best_pre_model_save_file, map_location=device)
        model_before.load_state_dict(state_dict_pre)

        num_parameters = sum(param.numel() for param in model_before.parameters() if param.requires_grad)
        print("Number of trainable parameters: ", num_parameters)

    if post_:
        model_after = ProposedModel(class_name=class_name, model_keypoints=model_keypoints, cad_models=cad_models,
                                    keypoint_detector=detector_type, correction_flag=use_corrector,
                                    need_predicted_keypoints=True).to(device)

        if not os.path.isfile(best_post_model_save_file):
            print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

        state_dict_post = torch.load(best_post_model_save_file, map_location=device)
        model_after.load_state_dict(state_dict_post)

        num_parameters = sum(param.numel() for param in model_after.parameters() if param.requires_grad)
        print("Number of trainable parameters: ", num_parameters)

    # Evaluation:
    if evaluate_models:
        if pre_:
            print(">>"*40)
            print("PRE-TRAINED MODEL:")
            print(">>" * 40)
            evaluate(eval_loader=eval_loader, model=model_before, hyper_param=hyper_param, certification=True,
                     device=device)
        if post_:
            print(">>" * 40)
            print("(SELF-SUPERVISED) TRAINED MODEL:")
            print(">>" * 40)
            evaluate(eval_loader=eval_loader, model=model_after, hyper_param=hyper_param, certification=True,
                     device=device)

    # # Visual Test
    dataset_len = 20
    dataset_batch_size = 1
    dataset = DepthPC(class_id=class_id,
                      model_id=model_id,
                      n=hyper_param['num_of_points_selfsupervised'],
                      num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                      dataset_len=dataset_len,
                      rotate_about_z=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch_size, shuffle=False)

    if visualize and pre_:
        print(">>" * 40)
        print("VISUALIZING PRE-TRAINED MODEL:")
        print(">>" * 40)
        visual_test(test_loader=loader, model=model_before, hyper_param=hyper_param)

    if visualize and post_:
        print(">>" * 40)
        print("(SELF-SUPERVISED) TRAINED MODEL:")
        print(">>" * 40)
        visual_test(test_loader=loader, model=model_after, hyper_param=hyper_param)

    if pre_:
        del model_before, state_dict_pre
    if post_:
        del model_after, state_dict_post

    return None


def evaluate_model(detector_type, class_name, model_id,
                   evaluate_models=True,
                   models_to_analyze='post',
                   visualize=True,
                   use_corrector=False):

    class_id = CLASS_ID[class_name]

    hyper_param_file = "self_supervised_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]
    hyper_param['epsilon'] = hyper_param['epsilon'][class_name]

    print(">>"*40)
    print("Analyzing Trained Model for Object: ", class_name, "; Model ID:", str(model_id))
    visualize_detector(detector_type=detector_type,
                       class_id=class_id,
                       model_id=model_id,
                       hyper_param=hyper_param,
                       evaluate_models=evaluate_models,
                       models_to_analyze=models_to_analyze,
                       visualize=visualize)


if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_sim_supervised_model.py "point_transformer" "chair"
    >> python evaluate_sim_supervised_model.py "pointnet" "chair"

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
                   models_to_analyze='pre',
                   visualize=False, evaluate_models=True, use_corrector=False)