
import torch
import yaml
import argparse
import sys

sys.path.append('../..')

from learning_objects.datasets.keypointnet import FixedDepthPC
from learning_objects.datasets.keypointnet import CLASS_NAME, CLASS_ID, DepthPC
from learning_objects.expt_self_supervised_correction.loss_functions import certify
from learning_objects.expt_self_supervised_correction.evaluation_metrics import evaluation_error, add_s_error
from learning_objects.expt_self_supervised_correction.proposed_model import ICP


def eval_icp(class_id, model_id, hyper_param, evaluate_models=True, visualize=False):

    # define dataset and dataloader
    # validation dataset:
    eval_dataset_len = hyper_param['eval_dataset_len']
    eval_batch_size = hyper_param['eval_batch_size']
    eval_dataset = FixedDepthPC(class_id=class_id, model_id=model_id,
                                n=hyper_param['num_of_points_selfsupervised'],
                                num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                                dataset_len=eval_dataset_len,
                                rotate_about_z=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    # get cad models
    cad_models = eval_dataset._get_cad_models().to(torch.float)

    # initialize the ICP model with the cad_models
    icp = ICP(cad_models=cad_models)

    # for the data batch evaluate icp output from ICP
    pc_err = 0.0
    kp_err = 0.0
    R_err = 0.0
    t_err = 0.0
    adds_err = 0.0

    pc_err_cert = 0.0
    kp_err_cert = 0.0
    R_err_cert = 0.0
    t_err_cert = 0.0
    adds_err_cert = 0.0

    num_cert = 0.0
    num_batches = len(eval_loader)

    for i, vdata in enumerate(eval_loader):
        input_point_cloud, keypoints_target, R_target, t_target = vdata
        # input_point_cloud = input_point_cloud.to(device)
        # keypoints_target = keypoints_target.to(device)
        # R_target = R_target.to(device)
        # t_target = t_target.to(device)
        batch_size = input_point_cloud.shape[0]

        # Make predictions for this batch
        predicted_point_cloud, R_predicted, t_predicted = icp.forward(input_point_cloud)
        # predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints \
        #     = model(input_point_cloud, correction_flag=correction_flag, need_predicted_keypoints=True)

        certi = certify(input_point_cloud=input_point_cloud,
                        predicted_point_cloud=predicted_point_cloud,
                        corrected_keypoints=keypoints_target,
                        predicted_model_keypoints=keypoints_target,
                        epsilon=hyper_param['epsilon'],
                        is_symmetric=hyper_param['is_symmetric'])

        # fraction certifiable
        # error of all objects
        # error of certified objects

        pc_err_, _, R_err_, t_err_ = \
            evaluation_error(input=(input_point_cloud, keypoints_target, R_target, t_target),
                             output=(predicted_point_cloud, keypoints_target, R_predicted, t_predicted))

        ground_truth_point_cloud = R_target @ cad_models + t_target
        adds_err_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                threshold=hyper_param["adds_threshold"])

        # error for all objects
        pc_err += pc_err_.sum()
        # kp_err += kp_err_.sum()
        R_err += R_err_.sum()
        t_err += t_err_.sum()
        adds_err += adds_err_.sum()

        # fraction certifiable
        num_cert += certi.sum()

        # error for certifiable objects
        pc_err_cert += (pc_err_ * certi).sum()
        # kp_err_cert += (kp_err_ * certi).sum()
        R_err_cert += (R_err_ * certi).sum()
        t_err_cert += (t_err_ * certi).sum()
        adds_err_cert += (adds_err_ * certi).sum()

        del input_point_cloud, keypoints_target, R_target, t_target, \
            predicted_point_cloud, R_predicted, t_predicted

    # avg_vloss = running_vloss / (i + 1)
    ave_pc_err = pc_err / ((i + 1) * batch_size)
    # ave_kp_err = kp_err / ((i + 1) * batch_size)
    ave_R_err = R_err / ((i + 1) * batch_size)
    ave_t_err = t_err / ((i + 1) * batch_size)
    ave_adds_err = 100 * adds_err / ((i + 1) * batch_size)

    ave_pc_err_cert = pc_err_cert / num_cert
    # ave_kp_err_cert = kp_err_cert / num_cert
    ave_R_err_cert = R_err_cert / num_cert
    ave_t_err_cert = t_err_cert / num_cert
    ave_adds_err_cert = 100 * adds_err_cert / num_cert

    fra_cert = 100 * num_cert / ((i + 1) * batch_size)

    print(">>>>>>>>>>>>>>>> EVALUATING MODEL >>>>>>>>>>>>>>>>>>>>")
    print("Evaluating performance across all objects:")
    print("pc error: ", ave_pc_err.item())
    # print("kp error: ", ave_kp_err.item())
    print("R error: ", ave_R_err.item())
    print("t error: ", ave_t_err.item())
    print("ADD-S (%): ", ave_adds_err.item())

    print("Evaluating certification: ")
    print("epsilon parameter: ", hyper_param['epsilon'])
    print("% certifiable: ", fra_cert.item())
    print("Evaluating performance for certifiable objects: ")
    print("pc error: ", ave_pc_err_cert.item())
    # print("kp error: ", ave_kp_err_cert.item())
    print("R error: ", ave_R_err_cert.item())
    print("t error: ", ave_t_err_cert.item())
    print("ADD-S (%): ", ave_adds_err_cert.item())

    # compute rotation, translation, ADD-S errors
    # compute %certifiable

    return None

def evaluate_icp(model_class_ids, only_categories, evaluate_models=True, visualize=False):

    for key, value in model_class_ids.items():
        if key in only_categories:
            class_id = CLASS_ID[key]
            model_id = str(value)
            class_name = CLASS_NAME[class_id]

            hyper_param_file = "self_supervised_training.yml"
            stream = open(hyper_param_file, "r")
            hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
            hyper_param = hyper_param['pointnet']   # we only use the evaluation dataset parameters, which are the same
            hyper_param['epsilon'] = hyper_param['epsilon'][key]

            if class_name == 'bottle':
                hyper_param["is_symmetric"] = True
            else:
                hyper_param["is_symmetric"] = False

            print(">>"*40)
            print("Analyzing Baseline for Object: ", key, "; Model ID:", str(model_id))

            eval_icp(class_id=class_id,
                     model_id=model_id,
                     hyper_param=hyper_param,
                     evaluate_models=evaluate_models,
                     visualize=visualize)

    return None


if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_icp.py "chair"
    >> python evaluate_icp.py "table"

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)

    args = parser.parse_args()

    class_name = args.class_name
    only_categories = [class_name]

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    evaluate_icp(model_class_ids=model_class_ids,
                 only_categories=only_categories,
                 visualize=False, evaluate_models=True)