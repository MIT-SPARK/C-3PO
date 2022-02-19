

import torch
import sys
sys.path.append("../..")

from learning_objects.expt_self_supervised_correction.loss_functions import certify
from learning_objects.expt_self_supervised_correction.evaluation_metrics import evaluation_error

from learning_objects.expt_self_supervised_correction.loss_functions import chamfer_loss

def add_s_error(predicted_point_cloud, ground_truth_point_cloud):
    """
    predicted_point_cloud       : torch.tensor of shape (B, 3, m)
    ground_truth_point_cloud    : torch.tensor of shape (B, 3, m)

    """

    # compute chamfer loss between the two
    #ToDo: Verify that this is indeed the ADD-S metric. We have used half-chamfer loss here.

    return chamfer_loss(predicted_point_cloud, ground_truth_point_cloud)


def evaluate(eval_loader, model, hyper_param, certification=True, device=None):

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

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
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)
            batch_size = input_point_cloud.shape[0]

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints\
                = model(input_point_cloud, correction_flag=True, need_predicted_keypoints=True)

            if certification:
                certi = certify(input_point_cloud=input_point_cloud,
                                predicted_point_cloud=predicted_point_cloud,
                                corrected_keypoints=predicted_keypoints,
                                predicted_model_keypoints=predicted_model_keypoints,
                                epsilon=hyper_param['epsilon'],
                                is_symmetric=hyper_param['is_symmetric'])

            # fraction certifiable
            # error of all objects
            # error of certified objects

            pc_err_, kp_err_, R_err_, t_err_ = \
                evaluation_error(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                 output=(predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted))

            ground_truth_point_cloud = R_target @ model.cad_models + t_target
            adds_err_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud)

            # error for all objects
            pc_err += pc_err_.sum()
            kp_err += kp_err_.sum()
            R_err += R_err_.sum()
            t_err += t_err_.sum()
            adds_err += adds_err_.sum()

            if certification:
                # fraction certifiable
                num_cert += certi.sum()

                # error for certifiable objects
                pc_err_cert += (pc_err_ * certi).sum()
                kp_err_cert += (kp_err_ * certi).sum()
                R_err_cert += (R_err_ * certi).sum()
                t_err_cert += (t_err_ * certi).sum()
                adds_err_cert += (adds_err_ * certi).sum()

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        # avg_vloss = running_vloss / (i + 1)
        ave_pc_err = pc_err / ((i + 1)*batch_size)
        ave_kp_err = kp_err / ((i + 1)*batch_size)
        ave_R_err = R_err / ((i + 1)*batch_size)
        ave_t_err = t_err / ((i + 1)*batch_size)
        ave_adds_err = adds_err / ((i + 1) * batch_size)

        if certification:
            ave_pc_err_cert = pc_err_cert / num_cert
            ave_kp_err_cert = kp_err_cert / num_cert
            ave_R_err_cert = R_err_cert / num_cert
            ave_t_err_cert = t_err_cert / num_cert
            ave_adds_err_cert = adds_err_cert / num_cert

            fra_cert = num_cert / ((i + 1)*batch_size)

        print(">>>>>>>>>>>>>>>> EVALUATING MODEL >>>>>>>>>>>>>>>>>>>>")
        print("Evaluating performance across all objects:")
        print("pc error: ", ave_pc_err.item())
        print("kp error: ", ave_kp_err.item())
        print("R error: ", ave_R_err.item())
        print("t error: ", ave_t_err.item())
        print("ADD-S error: ", ave_adds_err.item())

        print("Evaluating certification: ")
        print("fraction certifiable: ", fra_cert.item())
        print("Evaluating performance for certifiable objects: ")
        print("pc error: ", ave_pc_err_cert.item())
        print("kp error: ", ave_kp_err_cert.item())
        print("R error: ", ave_R_err_cert.item())
        print("t error: ", ave_t_err_cert.item())
        print("ADD-S error: ", ave_adds_err_cert.item())

    return None


if __name__ == "__main__":

    print("test")
