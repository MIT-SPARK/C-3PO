"""
This code analyzes the data generated by the experiment expt_with_reg_se3pc.py

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
from datetime import datetime
from matplotlib import colors as mcolors

sys.path.append("../../")
from c3po.utils.general import generate_filename
from c3po.utils.visualization_utils import scatter_bar_plot
from c3po.datasets.shapenet import CLASS_NAME
from c3po.models.certifiability import certifiability

plt.style.use('seaborn-whitegrid')
COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def masked_varul_mean(data, mask):
    """
    inputs:
    data    : torch.tensor of shape (B, n)
    mask    : torch.tensor of shape (B, n). dtype=torch.bool

    outputs:
    var     : torch.tensor of shape (B, 2)  :
        var[:, 0] = lower variance
        var[:, 1] = upper variance

    mean    : torch.tensor of shape (B,)

    """
    device_ = data.device
    batch_size = data.shape[0]

    var = torch.zeros(batch_size, 2).to(device_)
    mean = torch.zeros(batch_size).to(device_)

    for batch, (d, m) in enumerate(zip(data, mask)):
        dm = torch.masked_select(d, m)

        dm_mean = dm.mean()
        dm_centered = dm - dm_mean
        dm_centered_up = dm_centered*(dm_centered >= 0)
        dm_centered_lo = dm_centered*(dm_centered < 0)
        len = dm_centered.shape[0]

        dm_var_up = torch.sum(dm_centered_up**2)/(len + 0.001)
        dm_var_lo = torch.sum(dm_centered_lo**2)/(len + 0.001)

        mean[batch] = dm_mean
        var[batch, 0] = dm_var_lo
        var[batch, 1] = dm_var_up

    return var, mean


def varul_mean(data):
    """
    inputs:
    data    : torch.tensor of shape (B, n)

    outputs:
    var     : torch.tensor of shape (B, 2)  :
        var[:, 0] = lower variance
        var[:, 1] = upper variance

    mean    : torch.tensor of shape (B,)

    """

    mean = data.mean(dim=1).unsqueeze(-1)

    data_centered = data - mean
    data_pos = data_centered*(data_centered >= 0)
    data_neg = data_centered*(data_centered < 0)
    len = data_centered.shape[1]

    var_up = torch.sum(data_pos**2, dim=1)/(len + 0.001)
    var_low = torch.sum(data_neg**2, dim=1)/(len + 0.001)

    var_up = var_up.unsqueeze(-1)
    var_low = var_low.unsqueeze(-1)
    var = torch.cat([var_low, var_up], dim=1)

    return var, mean.squeeze(-1)



def certification(data, epsilon, delta, num_iterations=100, full_batch=False):
    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    certify=certifiability(epsilon=epsilon, delta=delta, radius=0.3)
    ###
    certi_naive = torch.zeros(size=(15, num_iterations), dtype=torch.bool).to(
        device=device_)
    certi_corrector = torch.zeros(size=(15, num_iterations), dtype=torch.bool).to(
        device=device_)

    ###
    sqdist_input_naiveest = data['sqdist_input_naiveest']
    sqdist_input_correctorest = data['sqdist_input_correctorest']
    sqdist_kp_naiveest = data['sqdist_kp_naiveest']
    sqdist_kp_correctorest = data['sqdist_kp_correctorest']
    pc_padding_masks = data['pc_padding_masks']

    for kp_noise_var_i in range(len(sqdist_input_naiveest)):
        print("kp_noise_var_i", kp_noise_var_i)
        c_naive = torch.zeros((num_iterations, 1), dtype=torch.bool).to(device=device_)
        c_corrector = torch.zeros((num_iterations, 1), dtype=torch.bool).to(device=device_)
        # if experiments were full batch, just set output of certify.forward_with_distances
        # to c_naive and c_corrector
        for batch_i in range(len(sqdist_input_naiveest[kp_noise_var_i])):
            print("batch_i", batch_i)

            #len 100 or batch size 100
            sqdist_input_naive = sqdist_input_naiveest[kp_noise_var_i][batch_i]
            sqdist_input_corrector = sqdist_input_correctorest[kp_noise_var_i][batch_i]
            sqdist_kp_naive = sqdist_kp_naiveest[kp_noise_var_i][batch_i]
            sqdist_kp_corrector = sqdist_kp_correctorest[kp_noise_var_i][batch_i]
            pc_padding = pc_padding_masks[kp_noise_var_i][batch_i]
            certi_naive_batch, _ = certify.forward_with_distances(
                sqdist_input_naive[0], sqdist_input_naive[1], sqdist_input_naive[2], sqdist_kp_naive, pc_padding)
            certi_corrector_batch, _ = certify.forward_with_distances(
                sqdist_input_corrector[0], sqdist_input_corrector[1], sqdist_input_naive[2], sqdist_kp_corrector, pc_padding)
            if full_batch: #full batch
                c_naive = certi_naive_batch
                c_corrector = certi_corrector_batch
            else:
                print("certi_naive_batch.shape", certi_naive_batch.shape)
                c_naive[batch_i] = certi_naive_batch
                c_corrector[batch_i] = certi_corrector_batch
        certi_naive[kp_noise_var_i, ...] = c_naive.squeeze(-1)
        certi_corrector[kp_noise_var_i, ...] = c_corrector.squeeze(-1)

    return certi_naive, certi_corrector

if __name__ == '__main__':
    use_adds_metric = True
    file_names = ["./expt_with_reg_depthpc/02876657/41a2005b595ae783be1868124d5ddbcb_wchamfer/20220227_170722_experiment.pickle"]
    # the following pickle files are the experiment metrics to generate plots from the paper
    # file_names = ["./expt_with_reg_depthpc/02691156/3db61220251b3c9de719b5362fe06bbb_wchamfer/20220610_185655_experiment.pickle",
    #               "./expt_with_reg_depthpc/02808440/90b6e958b359c1592ad490d4d7fae486_wchamfer/20220610_194647_experiment.pickle",
    #               "./expt_with_reg_depthpc/02818832/7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f_wchamfer/20220610_203643_experiment.pickle",
    #               "./expt_with_reg_depthpc/02876657/41a2005b595ae783be1868124d5ddbcb_wchamfer/20220227_170722_experiment.pickle",
    #               "./expt_with_reg_depthpc/02954340/3dec0d851cba045fbf444790f25ea3db_wchamfer/20220610_212958_experiment.pickle",
    #               "./expt_with_reg_depthpc/02958343/ad45b2d40c7801ef2074a73831d8a3a2_wchamfer/20220610_222513_experiment.pickle",
    #               "./expt_with_reg_depthpc/03001627/1cc6f2ed3d684fa245f213b8994b4a04_wchamfer/20220610_232015_experiment.pickle",
    #               "./expt_with_reg_depthpc/03467517/5df08ba7af60e7bfe72db292d4e13056_wchamfer/20220611_001510_experiment.pickle",
    #               "./expt_with_reg_depthpc/03513137/3621cf047be0d1ae52fafb0cab311e6a_wchamfer/20220611_010956_experiment.pickle",
    #               "./expt_with_reg_depthpc/03624134/819e16fd120732f4609e2d916fa0da27_wchamfer/20220611_020429_experiment.pickle",
    #               "./expt_with_reg_depthpc/03642806/519e98268bee56dddbb1de10c9529bf7_wchamfer/20220611_025907_experiment.pickle",
    #               "./expt_with_reg_depthpc/03790512/481f7a57a12517e0fe1b9fad6c90c7bf_wchamfer/20220611_035351_experiment.pickle",
    #               "./expt_with_reg_depthpc/03797390/f3a7f8198cc50c225f5e789acd4d1122_wchamfer/20220611_044817_experiment.pickle",
    #               "./expt_with_reg_depthpc/04225987/98222a1e5f59f2098745e78dbc45802e_wchamfer/20220611_054239_experiment.pickle",
    #               "./expt_with_reg_depthpc/04379243/3f5daa8fe93b68fa87e2d08958d6900c_wchamfer/20220611_063732_experiment.pickle",
    #               "./expt_with_reg_depthpc/04530566/5c54100c798dd681bfeb646a8eadb57_wchamfer/20220611_073255_experiment.pickle",
    #
    #               ]

    for name in file_names:

        fp = open(name, 'rb')
        parameters, data = pickle.load(fp)
        fp.close()

        if use_adds_metric:
            for noise_idx in range(len(data['chamfer_pose_naive_to_gt_pose_list'])):
                data['chamfer_pose_naive_to_gt_pose_list'][noise_idx] = np.asarray(data['chamfer_pose_naive_to_gt_pose_list'][noise_idx][0].squeeze().to('cpu'))
                data['chamfer_pose_corrected_to_gt_pose_list'][noise_idx] = np.asarray(data['chamfer_pose_corrected_to_gt_pose_list'][noise_idx][0].squeeze().to('cpu'))

        print("-" * 80)

        Rerr_naive = data['rotation_err_naive'].to('cpu')
        Rerr_corrector = data['rotation_err_corrector'].to('cpu')
        terr_naive = data['translation_err_naive'].to('cpu')
        terr_corrector = data['translation_err_corrector'].to('cpu')
        # certi_naive = data['certi_naive'].to('cpu')
        # certi_corrector = data['certi_corrector'].to('cpu')
        # CALCULATE DYNAMICALLY
        epsilon = .99
        delta = .7
        fig_save_folder = '/'.join(name.split('/')[:-1] + ['eps' + str(epsilon)[2:]])
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        fig_save_prefix = fig_save_folder + '/' + name.split('/')[-1]
        print(fig_save_prefix)
        certi_naive, certi_corrector = certification(data, epsilon=epsilon, delta=delta, full_batch=True)
        certi_naive = certi_naive.to('cpu')
        certi_corrector = certi_corrector.to('cpu')

        sqdist_input_naiveest = data['sqdist_input_naiveest']
        sqdist_input_correctorest = data['sqdist_input_correctorest']
        sqdist_kp_naiveest = data['sqdist_kp_naiveest']
        sqdist_kp_correctorest = data['sqdist_kp_correctorest']

        if use_adds_metric:
            chamfer_pose_naive_to_gt_pose_list = torch.from_numpy(np.asarray(data['chamfer_pose_naive_to_gt_pose_list']))
            chamfer_pose_corrected_to_gt_pose_list = torch.from_numpy(np.asarray(data['chamfer_pose_corrected_to_gt_pose_list']))


        kp_noise_var_range = parameters['kp_noise_var_range'].to('cpu')
        kp_noise_type = parameters['kp_noise_type']
        kp_noise_fra = parameters['kp_noise_fra']
        class_id = parameters['class_id']
        model_id = parameters['model_id']
        cad_model_name = CLASS_NAME[class_id]

        # Plotting rotation distribution
        fig = plt.figure()

        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive, label='naive', color='lightgray')
        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive*certi_naive, label='naive + certification', color='royalblue')
        plt.show()
        plt.close(fig)

        fig = plt.figure()
        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector, label='corrector', color='lightgray')
        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector * certi_corrector, label='corrector + certification',
                               color='orangered')
        plt.show()
        plt.close(fig)

        if use_adds_metric:
            chamfer_metric_naive_var, chamfer_metric_naive_mean = varul_mean(chamfer_pose_naive_to_gt_pose_list)
            chamfer_metric_corrected_var, chamfer_metric_corrected_mean = varul_mean(chamfer_pose_corrected_to_gt_pose_list)
            chamfer_metric_naive_certi_var, chamfer_metric_naive_certi_mean = masked_varul_mean(chamfer_pose_naive_to_gt_pose_list, mask=certi_naive)
            chamfer_metric_corrected_certi_var, chamfer_metric_corrected_certi_mean = masked_varul_mean(chamfer_pose_corrected_to_gt_pose_list, mask=certi_corrector)

        Rerr_naive_var, Rerr_naive_mean = varul_mean(Rerr_naive)
        Rerr_corrector_var, Rerr_corrector_mean = varul_mean(Rerr_corrector)
        terr_naive_var, terr_naive_mean = varul_mean(terr_naive)
        terr_corrector_var, terr_corrector_mean = varul_mean(terr_corrector)

        Rerr_naive_certi_var, Rerr_naive_certi_mean = masked_varul_mean(Rerr_naive, mask=certi_naive)
        Rerr_corrector_certi_var, Rerr_corrector_certi_mean = masked_varul_mean(Rerr_corrector, mask=certi_corrector)
        terr_naive_certi_var, terr_naive_certi_mean = masked_varul_mean(terr_naive, mask=certi_naive)
        terr_corrector_certi_var, terr_corrector_certi_mean = masked_varul_mean(terr_corrector, mask=certi_corrector)

        fraction_not_certified_naive_var, fraction_not_certified_naive_mean = \
            varul_mean(1 - certi_naive.float())
        fraction_not_certified_corrector_var, fraction_not_certified_corrector_mean = \
            varul_mean(1 - certi_corrector.float())


        Rerr_naive_var = torch.sqrt(Rerr_naive_var).T
        Rerr_corrector_var = torch.sqrt(Rerr_corrector_var).T
        terr_naive_var = torch.sqrt(terr_naive_var).T
        terr_corrector_var = torch.sqrt(terr_corrector_var).T
        Rerr_naive_certi_var = torch.sqrt(Rerr_naive_certi_var).T
        Rerr_corrector_certi_var = torch.sqrt(Rerr_corrector_certi_var).T
        terr_naive_certi_var = torch.sqrt(terr_naive_certi_var).T
        terr_corrector_certi_var = torch.sqrt(terr_corrector_certi_var).T
        fraction_not_certified_corrector_var = torch.sqrt(fraction_not_certified_corrector_var).T
        fraction_not_certified_naive_var = torch.sqrt(fraction_not_certified_naive_var).T

        if use_adds_metric:
            chamfer_metric_naive_var = torch.sqrt(chamfer_metric_naive_var).T
            chamfer_metric_corrected_var = torch.sqrt(chamfer_metric_corrected_var).T
            chamfer_metric_naive_certi_var = torch.sqrt(chamfer_metric_naive_certi_var).T
            chamfer_metric_corrected_certi_var = torch.sqrt(chamfer_metric_corrected_certi_var).T

        # plotting chamfer metric
        if use_adds_metric:

            fig = plt.figure()
            plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_naive_mean, yerr=chamfer_metric_naive_var,
                         fmt='-x', color='black', ecolor='gray', elinewidth=1, capsize=3, label='naive')
            plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_naive_certi_mean, yerr=chamfer_metric_naive_certi_var,
                         fmt='--o', color='grey', ecolor='lightgray', elinewidth=3, capsize=0, label='naive + certification')
            plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_corrected_mean, yerr=chamfer_metric_corrected_var, fmt='-x', color='red',
                         ecolor='salmon', elinewidth=1, capsize=3, label='corrector')
            plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_corrected_certi_mean, yerr=chamfer_metric_corrected_certi_var, fmt='--o',
                         color='orangered', ecolor='salmon', elinewidth=3, capsize=0, label='corrector + certification')
            # alternate colors
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_naive_mean, yerr=chamfer_metric_naive_var,
            #              fmt='-', color='red', ecolor='red', elinewidth=1, capsize=3, label='naive')
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_naive_certi_mean, yerr=chamfer_metric_naive_certi_var,
            #              fmt='--o', color='red', ecolor=(1.0,0,0,0.3), elinewidth=3, capsize=3, label='naive + certification')
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_corrected_mean, yerr=chamfer_metric_corrected_var,
            #              fmt='-', color='green', ecolor='green', elinewidth=1, capsize=3, label='corrector')
            # plt.errorbar(x=kp_noise_var_range, y=chamfer_metric_corrected_certi_mean, yerr=chamfer_metric_corrected_certi_var,
            #              fmt='--o', color='green', ecolor=(0,.5,0,0.3), elinewidth=3, capsize=3, label='corrector + certification')
            plt.legend(loc='upper left')
            # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
            # plt.title(title)
            plt.xlabel('Noise variance parameter $\sigma$')
            plt.ylabel('Normalized ADD-S')
            plt.show()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rand_string = generate_filename()
            filename = fig_save_prefix[:-7] + '_chamfer_metric_plot_' + timestamp + '_' + rand_string + '.jpg'
            fig.savefig(filename)
            plt.close(fig)



        # plotting rotation errors
        fig = plt.figure()
        plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_mean, yerr=Rerr_naive_var, fmt='-x', color='black', ecolor='gray',
                     elinewidth=1, capsize=3, label='naive')
        plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_certi_mean, yerr=Rerr_naive_certi_var, fmt='--o', color='grey',
                     ecolor='lightgray', elinewidth=3, capsize=0, label='naive + certification')
        plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_mean, yerr=Rerr_corrector_var, fmt='-x', color='red',
                     ecolor='salmon', elinewidth=1, capsize=3, label='corrector')
        plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_certi_mean, yerr=Rerr_corrector_certi_var, fmt='--o',
                     color='orangered', ecolor='salmon', elinewidth=3, capsize=0, label='corrector + certification')

        # alternate colors
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_mean, yerr=Rerr_naive_var, fmt='-', color='red', ecolor='red', elinewidth=1, capsize=3, label='naive')
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_certi_mean, yerr=Rerr_naive_certi_var, fmt='--o', color='red', ecolor=(1.0,0,0,0.3), elinewidth=3, capsize=3, label='naive + certification')
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_mean, yerr=Rerr_corrector_var, fmt='-', color='green', ecolor='green', elinewidth=1, capsize=3, label='corrector')
        # plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_certi_mean, yerr=Rerr_corrector_certi_var, fmt='--o', color='green', ecolor=(0,.5,0,0.3), elinewidth=3, capsize=3, label='corrector + certification')
        plt.legend(loc='upper left')
        # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        # plt.title(title)
        plt.xlabel('Noise variance parameter $\sigma$')
        plt.ylabel('Rotation error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = fig_save_prefix[:-7] + '_rotation_error_plot_' + timestamp + '_' + rand_string +'.jpg'
        fig.savefig(filename)
        plt.close(fig)






        # Plotting translation errors
        fig = plt.figure()
        plt.errorbar(x=kp_noise_var_range, y=terr_naive_mean, yerr=terr_naive_var, fmt='-x', color='black', ecolor='gray',
                     elinewidth=1, capsize=3, label='naive')
        plt.errorbar(x=kp_noise_var_range, y=terr_naive_certi_mean, yerr=terr_naive_certi_var, fmt='--o', color='grey',
                     ecolor='lightgray', elinewidth=3, capsize=0, label='naive + certification')
        plt.errorbar(x=kp_noise_var_range, y=terr_corrector_mean, yerr=terr_corrector_var, fmt='-x', color='red',
                     ecolor='salmon', elinewidth=1, capsize=3, label='corrector')
        plt.errorbar(x=kp_noise_var_range, y=terr_corrector_certi_mean, yerr=terr_corrector_certi_var, fmt='--o',
                     color='salmon', ecolor='orangered', elinewidth=3, capsize=0, label='corrector + certification')

        #alternate colors
        # plt.errorbar(x=kp_noise_var_range, y=terr_naive_mean, yerr=terr_naive_var, fmt='-', color='red', ecolor='red', elinewidth=1, capsize=3, label='naive')
        # plt.errorbar(x=kp_noise_var_range, y=terr_naive_certi_mean, yerr=terr_naive_certi_var, fmt='--o', color='red', ecolor=(1.0,0,0,0.3), elinewidth=3, capsize=3, label='naive + certification')
        # plt.errorbar(x=kp_noise_var_range, y=terr_corrector_mean, yerr=terr_corrector_var, fmt='-', color='green', ecolor='green', elinewidth=1, capsize=3, label='corrector')
        # plt.errorbar(x=kp_noise_var_range, y=terr_corrector_certi_mean, yerr=terr_corrector_certi_var, fmt='--o', color='green', ecolor=(0,.5,0,0.3), elinewidth=3, capsize=3, label='corrector + certification')
        plt.legend(loc='upper left')
        # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        # plt.title(title)
        plt.xlabel('Noise variance parameter $\sigma$')
        plt.ylabel('Translation error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = fig_save_prefix[:-7] + '_translation_error_plot_' + timestamp + '_' + rand_string + '.jpg'
        fig.savefig(filename)
        plt.close(fig)


        # Plotting fraction not certified
        fig = plt.figure()
        plt.bar(x=kp_noise_var_range-0.01, width=0.02, height=fraction_not_certified_naive_mean,
                color='grey', align='center', label='naive')
        # plt.errorbar(x=kp_noise_var_range-0.01, y=fraction_not_certified_naive_mean,
        #              yerr=fraction_not_certified_naive_var,
        #              fmt='o', color='black', ecolor='darkgray', elinewidth=1, capsize=3)
        plt.bar(x=kp_noise_var_range+0.01, width=0.02, height=fraction_not_certified_corrector_mean,
                color='salmon', align='center', label='corrector')
        # plt.errorbar(x=kp_noise_var_range+0.01, y=fraction_not_certified_corrector_mean,
        #              yerr=fraction_not_certified_corrector_var,
        #              fmt='o', color='red', ecolor='orangered', elinewidth=1, capsize=3)
        plt.legend(loc='upper left')
        plt.xlabel('Noise variance parameter $\sigma$')
        plt.ylabel('Fraction not certifiable')
        # title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        # plt.title(title)
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = fig_save_prefix[:-7] + '_fraction_not_certifiable_plot_' + timestamp + '_' + rand_string + '.jpg'
        fig.savefig(filename)
        plt.close(fig)

