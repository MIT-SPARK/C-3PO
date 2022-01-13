"""
This code analyzes the data generated by the experiment expt_with_reg_se3pc.py

"""
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from datetime import datetime
plt.style.use('seaborn-whitegrid')

import os
import sys
sys.path.append("../../")
from learning_objects.utils.general import generate_filename, scatter_bar_plot
from learning_objects.datasets.keypointnet import CLASS_NAME

COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


# def masked_var_mean(data, mask):
#     """
#     inputs:
#     data    : torch.tensor of shape (B, n)
#     mask    : torch.tensor of shape (B, n). dtype=torch.bool
#
#     outputs:
#     mean    : torch.tensor of shape (B, 1)
#     var     : torch.tensor of shape (B, 1)
#     """
#
#     mean = (torch.sum(data*mask.float(), dim=1)/torch.sum(mask.float(), dim=1)).unsqueeze(-1)
#
#     data_centered = data - mean
#     var = (torch.sum((data_centered**2)*mask.float(), dim=1)/(torch.sum(mask.float(), dim=1)-1)).unsqueeze(-1)
#
#     return var.squeeze(-1), mean.squeeze(-1)

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



if __name__ == '__main__':

    file_names = ["./expt_with_reg_depthpc/20220112_205609_experiment.pickle",
                  "./expt_with_reg_depthpc/20220112_214100_experiment.pickle"]

    for name in file_names:

        fp = open(name, 'rb')
        parameters, data = pickle.load(fp)
        fp.close()

        print("-" * 80)

        Rerr_naive = data['rotation_err_naive']
        Rerr_corrector = data['rotation_err_corrector']
        terr_naive = data['translation_err_naive']
        terr_corrector = data['translation_err_corrector']
        certi_naive = data['certi_naive']
        certi_corrector = data['certi_corrector']
        sqdist_input_naiveest = data['sqdist_input_naiveest']
        sqdist_input_correctorest = data['sqdist_input_correctorest']

        kp_noise_var_range = parameters['kp_noise_var_range']
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

        # Rerr_naive_var, Rerr_naive_mean = torch.var_mean(Rerr_naive, dim=1, unbiased=False)
        # Rerr_corrector_var, Rerr_corrector_mean = torch.var_mean(Rerr_corrector, dim=1, unbiased=False)
        # terr_naive_var, terr_naive_mean = torch.var_mean(terr_naive, dim=1, unbiased=False)
        # terr_corrector_var, terr_corrector_mean = torch.var_mean(terr_corrector, dim=1, unbiased=False)
        #
        # Rerr_naive_certi_var, Rerr_naive_certi_mean = masked_var_mean(Rerr_naive, mask=certi_naive)
        # Rerr_corrector_certi_var, Rerr_corrector_certi_mean = masked_var_mean(Rerr_corrector, mask=certi_corrector)
        # terr_naive_certi_var, terr_naive_certi_mean = masked_var_mean(terr_naive, mask=certi_naive)
        # terr_corrector_certi_var, terr_corrector_certi_mean = masked_var_mean(terr_corrector, mask=certi_corrector)
        #
        # fraction_not_certified_naive_var, fraction_not_certified_naive_mean = \
        #     torch.var_mean(1 - certi_naive.float(), dim=1, unbiased=True)
        # fraction_not_certified_corrector_var, fraction_not_certified_corrector_mean = \
        #     torch.var_mean(1 - certi_corrector.float(), dim=1, unbiased=True)

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



        # plotting rotation errors
        fig = plt.figure()
        plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_mean, yerr=Rerr_naive_var, fmt='-x', color='black',
                     ecolor='gray', elinewidth=1, capsize=3, label='naive')
        plt.errorbar(x=kp_noise_var_range, y=Rerr_naive_certi_mean, yerr=Rerr_naive_certi_var, fmt='--o', color='grey',
                     ecolor='lightgray', elinewidth=3, capsize=0, label='naive + certification')
        plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_mean, yerr=Rerr_corrector_var, fmt='-x', color='red',
                     ecolor='salmon', elinewidth=1, capsize=3, label='corrector')
        plt.errorbar(x=kp_noise_var_range, y=Rerr_corrector_certi_mean, yerr=Rerr_corrector_certi_var, fmt='--o', color='orangered',
                     ecolor='salmon', elinewidth=3, capsize=0, label='corrector + certification')
        plt.legend(loc='upper left')
        if kp_noise_type=='sporadic':
            title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
            plt.title(title)
        elif kp_noise_type=='uniform':
            title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type
            plt.title(title)
        plt.xlabel('noise variance parameter $\sigma$')
        plt.ylabel('rotation error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_rotation_error_plot_' + timestamp + '_' + rand_string +'.pdf'
        fig.savefig(filename)
        plt.close(fig)






        # Plotting translation errors
        fig = plt.figure()
        plt.errorbar(x=kp_noise_var_range, y=terr_naive_mean, yerr=terr_naive_var, fmt='-x', color='black',
                     ecolor='gray', elinewidth=1, capsize=3, label='naive')
        plt.errorbar(x=kp_noise_var_range, y=terr_naive_certi_mean, yerr=terr_naive_certi_var, fmt='--o', color='grey',
                     ecolor='lightgray', elinewidth=3, capsize=0, label='naive + certification')
        plt.errorbar(x=kp_noise_var_range, y=terr_corrector_mean, yerr=terr_corrector_var, fmt='-x', color='red',
                     ecolor='salmon', elinewidth=1, capsize=3, label='corrector')
        plt.errorbar(x=kp_noise_var_range, y=terr_corrector_certi_mean, yerr=terr_corrector_certi_var, fmt='--o',
                     color='orangered',
                     ecolor='salmon', elinewidth=3, capsize=0, label='corrector + certification')
        plt.legend(loc='upper left')
        if kp_noise_type == 'sporadic':
            title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
            plt.title(title)
        elif kp_noise_type == 'uniform':
            title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type
            plt.title(title)
        plt.xlabel('noise variance parameter $\sigma$')
        plt.ylabel('translation error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_translation_error_plot_' + timestamp + '_' + rand_string + '.pdf'
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
        plt.xlabel('noise variance parameter $\sigma$')
        plt.ylabel('fraction not ($\epsilon$, $\delta$)-certifiable')
        if kp_noise_type == 'sporadic':
            title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
            plt.title(title)
        elif kp_noise_type == 'uniform':
            title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type
            plt.title(title)
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_fraction_not_certifiable_plot_' + timestamp + '_' + rand_string + '.pdf'
        fig.savefig(filename)
        plt.close(fig)

