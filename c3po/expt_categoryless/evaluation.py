import os
import pickle
import sys
import torch
import yaml
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../..")

from c3po.utils.loss_functions import certify
from c3po.utils.evaluation_metrics import evaluation_error, add_s_error, is_pcd_nondegenerate
from c3po.utils.evaluation_metrics import adds_error, rotation_error, translation_error, EvalData
from c3po.datasets.ycb import MODEL_TO_KPT_GROUPS as MODEL_TO_KPT_GROUPS_YCB
from c3po.datasets.shapenet import MODEL_TO_KPT_GROUPS as MODEL_TO_KPT_GROUPS_SHAPENET
from c3po.datasets.shapenet import OBJECT_CATEGORIES as SHAPENET_OBJECTS
from c3po.datasets.ycb import MODEL_IDS as YCB_OBJECTS


def evaluate(eval_loader, model, hyper_param, certification=True, device=None, normalize_adds=False,
             new_eval=True, data_type=None, log_dir='eval/'):

    model.eval()

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "model_id" in model.__dict__.keys():
        if model.model_id in YCB_OBJECTS:
            if data_type is None:
                data_type = "ycb"
            object_name = model.model_id
        else:
            raise ValueError("model_id not correct.")
    elif "class_name" in model.__dict__.keys():
        if model.class_name in SHAPENET_OBJECTS:
            if data_type is None:
                data_type = "shapenet"
            object_name = model.class_name
        else:
            raise ValueError("class_name not correct.")
    else:
        raise ValueError("model is neither suited for ycb or shapenet.")

    with torch.no_grad():

        # pc_err = 0.0
        # kp_err = 0.0
        # R_err = 0.0
        # t_err = 0.0
        adds_err = 0.0
        auc = 0.0

        # pc_err_cert = 0.0
        # kp_err_cert = 0.0
        # R_err_cert = 0.0
        # t_err_cert = 0.0
        adds_err_cert = 0.0
        auc_cert = 0.0

        num_cert = 0.0
        num_batches = len(eval_loader)

        model_keypoints = eval_loader.dataset._get_model_keypoints()
        model_keypoints = model_keypoints.to(device=device)

        if normalize_adds:
            print("normalizing adds thresholds")
            model_diameter = eval_loader.dataset._get_diameter()
            print("model diameter is", model_diameter)
            hyper_param["adds_auc_threshold"] = hyper_param["adds_auc_threshold"]*model_diameter
            print(hyper_param["adds_auc_threshold"])
            hyper_param["adds_threshold"]= hyper_param["adds_threshold"]*model_diameter
            print(hyper_param["adds_threshold"])

        if new_eval:

            # log_dir = log_dir + '/' + data_type + '/' + object_name
            writer = SummaryWriter(log_dir)

            adds_list = []
            rerr_list = []
            terr_list = []
            oc_list = []
            nd_list = []
            for i, vdata in enumerate(tqdm(eval_loader)):

                # breakpoint()
                input_point_cloud, R_target, t_target = vdata
                input_point_cloud = input_point_cloud.to(device)
                # keypoints_target = keypoints_target.to(device)
                R_target = R_target.to(device)
                t_target = t_target.to(device)
                batch_size = input_point_cloud.shape[0]

                # Make predictions for this batch
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints \
                    = model(input_point_cloud)
                ground_truth_point_cloud = R_target @ model.cad_models + t_target

                # if certification:
                certi = certify(input_point_cloud=input_point_cloud,
                                predicted_point_cloud=predicted_point_cloud,
                                corrected_keypoints=predicted_keypoints,
                                predicted_model_keypoints=predicted_model_keypoints,
                                epsilon=hyper_param['epsilon'])

                # if degeneracy:
                # if ycb
                if "ycb" in data_type:
                    nondeg = is_pcd_nondegenerate(model.model_id, input_point_cloud, predicted_keypoints,
                                                  MODEL_TO_KPT_GROUPS_YCB)
                    # deg = nondeg < 1
                # if shapenet
                elif "shapenet" in data_type:
                    nondeg = is_pcd_nondegenerate(model.class_name, input_point_cloud, predicted_model_keypoints,
                                                  MODEL_TO_KPT_GROUPS_SHAPENET)
                    # deg = nondeg < 1
                else:
                    raise ValueError("data_type not specified correctly.")

                T_gt = torch.eye(4).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
                T_est = torch.eye(4).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
                T_gt[:, :3, :3] = R_target
                T_gt[:, :3, 3:] = t_target
                T_est[:, :3, :3] = R_predicted
                T_est[:, :3, 3:] = t_predicted

                # breakpoint()

                adds_ = adds_error(model.cad_models, T_gt, T_est)
                rerr_ = rotation_error(R_predicted, R_target)
                terr_ = translation_error(t_predicted, t_target)

                adds_x = [x.item() for x in adds_]
                rerr_x = [x.item() for x in rerr_]
                terr_x = [x.item() for x in terr_]
                oc_x = [int(x.item()) for x in certi]
                nd_x = [int(x.item()) for x in nondeg]

                adds_list = [*adds_list, *adds_x]
                rerr_list = [*rerr_list, *rerr_x]
                terr_list = [*terr_list, *terr_x]
                oc_list = [*oc_list, *oc_x]
                nd_list = [*nd_list, *nd_x]

                del input_point_cloud, R_target, t_target, \
                    predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

            eval_data = EvalData()
            eval_data.set_adds(np.asarray(adds_list))
            eval_data.set_rerr(np.asarray(rerr_list))
            eval_data.set_terr(np.asarray(terr_list))
            eval_data.set_oc(np.asarray(oc_list))
            eval_data.set_nd(np.asarray(nd_list))
            save_file = writer.log_dir + '/' + 'eval_data.pkl'
            eval_data.save(save_file)

        else:
            for i, vdata in enumerate(tqdm(eval_loader)):
                input_point_cloud, R_target, t_target = vdata
                input_point_cloud = input_point_cloud.to(device)
                R_target = R_target.to(device)
                t_target = t_target.to(device)
                # keypoints_target = R_target @ model_keypoints + t_target
                batch_size = input_point_cloud.shape[0]

                # Make predictions for this batch
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints\
                    = model(input_point_cloud)

                if certification:
                    certi = certify(input_point_cloud=input_point_cloud,
                                    predicted_point_cloud=predicted_point_cloud,
                                    corrected_keypoints=predicted_keypoints,
                                    predicted_model_keypoints=predicted_model_keypoints,
                                    epsilon=hyper_param['epsilon'])

                # fraction certifiable
                # error of all objects
                # error of certified objects

                # pc_err_, kp_err_, R_err_, t_err_ = \
                #     evaluation_error(input=(input_point_cloud, keypoints_target, R_target, t_target),
                #                      output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))

                ground_truth_point_cloud = R_target @ model.cad_models + t_target
                # adds_err_, auc_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                #                              threshold=hyper_param["adds_threshold"])
                adds_err_, _ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                           threshold=hyper_param["adds_threshold"])
                _, auc_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                      threshold=hyper_param["adds_auc_threshold"])

                # gt_cert_all = certify(input_point_cloud=ground_truth_point_cloud,
                #                       predicted_point_cloud=predicted_point_cloud,
                #                       corrected_keypoints=predicted_keypoints,
                #                       predicted_model_keypoints=predicted_model_keypoints,
                #                       epsilon=hyper_param['epsilon'])


                # error for all objects
                # pc_err += pc_err_.sum()
                # kp_err += kp_err_.sum()
                # R_err += R_err_.sum()
                # t_err += t_err_.sum()
                adds_err += adds_err_.sum()
                auc += auc_

                if certification:
                    # fraction certifiable
                    num_cert += certi.sum()

                    # error for certifiable objects
                    # pc_err_cert += (pc_err_ * certi).sum()
                    # kp_err_cert += (kp_err_ * certi).sum()
                    # R_err_cert += (R_err_ * certi).sum()
                    # t_err_cert += (t_err_ * certi).sum()
                    adds_err_cert += (adds_err_ * certi).sum()

                    _, auc_cert_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                              threshold=hyper_param['adds_auc_threshold'], certi=certi)
                    auc_cert += auc_cert_

                del input_point_cloud, R_target, t_target, \
                    predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

            # avg_vloss = running_vloss / (i + 1)
            # ave_pc_err = pc_err / ((i + 1)*batch_size)
            # ave_kp_err = kp_err / ((i + 1)*batch_size)
            # ave_R_err = R_err / ((i + 1)*batch_size)
            # ave_t_err = t_err / ((i + 1)*batch_size)
            ave_adds_err = 100 * adds_err / ((i + 1) * batch_size)
            ave_auc = 100 * auc / (i + 1)

            if certification:
                # ave_pc_err_cert = pc_err_cert / num_cert
                # ave_kp_err_cert = kp_err_cert / num_cert
                # ave_R_err_cert = R_err_cert / num_cert
                # ave_t_err_cert = t_err_cert / num_cert
                ave_adds_err_cert = 100 * adds_err_cert / num_cert
                ave_auc_cert = 100 * auc_cert / (i + 1)

                fra_cert = 100 * num_cert / ((i + 1)*batch_size)

            print(">>>>>>>>>>>>>>>> EVALUATING MODEL >>>>>>>>>>>>>>>>>>>>")
            print("Evaluating performance across all objects:")
            # print("pc error: ", ave_pc_err.item())
            # print("kp error: ", ave_kp_err.item())
            # print("R error: ", ave_R_err.item())
            # print("t error: ", ave_t_err.item())
            print("ADD-S (", int(hyper_param["adds_threshold"]*100), "%): ", ave_adds_err.item())
            print("ADD-S AUC (", int(hyper_param["adds_auc_threshold"]*100), "%): ", ave_auc.item())
            # print("GT-certifiable: ")

            print("Evaluating certification: ")
            print("epsilon parameter: ", hyper_param['epsilon'])
            print("% certifiable: ", fra_cert.item())
            print("Evaluating performance for certifiable objects: ")
            # print("pc error: ", ave_pc_err_cert.item())
            # print("kp error: ", ave_kp_err_cert.item())
            # print("R error: ", ave_R_err_cert.item())
            # print("t error: ", ave_t_err_cert.item())
            print("ADD-S (", int(hyper_param["adds_threshold"]*100), "%): ", ave_adds_err_cert.item())
            print("ADD-S AUC (", int(hyper_param["adds_auc_threshold"]*100), "%): ", ave_auc_cert.item())
            # print("GT-certifiable: ")

    return None


if __name__ == "__main__":

    print("test")
    # print("THIS CODE WILL GENERATE AND STORE DATA FOR EVALUATION")
    #
    # stream = open("class_model_ids.yml", "r")
    # model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    # stream = open("evaluation_datagen.yml", "r")
    # param = yaml.load(stream=stream, Loader=yaml.Loader)
    #
    # # run this to generate evaluation data
    # generate_depthpc_eval_data(model_class_ids=model_class_ids, param=param)
    #
    # print("THIS CODE WILL GENERATE AND STORE DATA FOR SUPERVISED BASELINE TRAINING")
    #
    # stream = open("class_model_ids.yml", "r")
    # model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    # stream = open("self_supervised_training.yml", "r")
    # param = yaml.load(stream=stream, Loader=yaml.Loader)
    #
    # # run this to generate evaluation data
    # generate_depthpc_eval_data(model_class_ids=model_class_ids, param=param)

