"""
This code analyzes the data generated by the experiment expt_with_pace_se3isopc.py

"""
import matplotlib.pyplot as plt
import os
import pickle
import sys
import torch
from datetime import datetime
from matplotlib import colors as mcolors

sys.path.append("../../")
from learning_objects.utils.general import generate_filename, scatter_bar_plot
from learning_objects.datasets.keypointnet import CLASS_NAME
from learning_objects.models.certifiability import certifiability
plt.style.use('seaborn-whitegrid')

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
                sqdist_input_corrector[0], sqdist_input_corrector[1], sqdist_input_corrector[2], sqdist_kp_corrector, pc_padding)
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

    file_names = ["./expt_with_pace_depthanisopc/02818832/7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f/20220208_204249_experiment.pickle",
                  "./expt_with_pace_depthanisopc/03001627/1cc6f2ed3d684fa245f213b8994b4a04/20220208_194609_experiment.pickle"]

    for name in file_names:

        fp = open(name, 'rb')
        parameters, data = pickle.load(fp)
        fp.close()

        print("-" * 80)

        Rerr_naive = data['rotation_err_naive']
        Rerr_corrector = data['rotation_err_corrector']
        terr_naive = data['translation_err_naive']
        terr_corrector = data['translation_err_corrector']
        shape_err_naive = data['shape_err_naive']
        shape_err_corrector = data['shape_err_corrector']
        # certi_naive = data['certi_naive']
        # certi_corrector = data['certi_corrector']
        # CALCULATE DYNAMICALLY
        certi_naive, certi_corrector = certification(data, epsilon=.99, delta=.7)
        certi_naive = certi_naive.to('cpu')
        certi_corrector = certi_corrector.to('cpu')
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
        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_naive * certi_naive, label='naive + certification',
                               color='royalblue')
        plt.show()
        plt.close(fig)

        fig = plt.figure()
        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector, label='corrector', color='lightgray')
        plt = scatter_bar_plot(plt, x=kp_noise_var_range, y=Rerr_corrector * certi_corrector,
                               label='corrector + certification',
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
        cerr_naive_var, cerr_naive_mean = varul_mean(shape_err_naive)
        cerr_corrector_var, cerr_corrector_mean = varul_mean(shape_err_corrector)

        Rerr_naive_certi_var, Rerr_naive_certi_mean = masked_varul_mean(Rerr_naive, mask=certi_naive)
        Rerr_corrector_certi_var, Rerr_corrector_certi_mean = masked_varul_mean(Rerr_corrector, mask=certi_corrector)
        terr_naive_certi_var, terr_naive_certi_mean = masked_varul_mean(terr_naive, mask=certi_naive)
        terr_corrector_certi_var, terr_corrector_certi_mean = masked_varul_mean(terr_corrector, mask=certi_corrector)
        cerr_naive_certi_var, cerr_naive_certi_mean = masked_varul_mean(shape_err_naive, mask=certi_naive)
        cerr_corrector_certi_var, cerr_corrector_certi_mean = masked_varul_mean(shape_err_corrector, mask=certi_corrector)

        fraction_not_certified_naive_var, fraction_not_certified_naive_mean = \
            varul_mean(1 - certi_naive.float())
        fraction_not_certified_corrector_var, fraction_not_certified_corrector_mean = \
            varul_mean(1 - certi_corrector.float())


        Rerr_naive_var = torch.sqrt(Rerr_naive_var).T
        Rerr_corrector_var = torch.sqrt(Rerr_corrector_var).T
        Rerr_naive_certi_var = torch.sqrt(Rerr_naive_certi_var).T
        Rerr_corrector_certi_var = torch.sqrt(Rerr_corrector_certi_var).T

        terr_naive_var = torch.sqrt(terr_naive_var).T
        terr_corrector_var = torch.sqrt(terr_corrector_var).T
        terr_naive_certi_var = torch.sqrt(terr_naive_certi_var).T
        terr_corrector_certi_var = torch.sqrt(terr_corrector_certi_var).T

        cerr_naive_var = torch.sqrt(cerr_naive_var).T
        cerr_corrector_var = torch.sqrt(cerr_corrector_var).T
        cerr_naive_certi_var = torch.sqrt(cerr_naive_certi_var).T
        cerr_corrector_certi_var = torch.sqrt(cerr_corrector_certi_var).T

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
        title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        plt.title(title)
        plt.xlabel('noise variance parameter $\sigma$')
        plt.ylabel('rotation error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_rotation_error_plot_' + timestamp + '_' + rand_string +'.jpg'
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
        title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        plt.title(title)
        plt.xlabel('noise variance parameter $\sigma$')
        plt.ylabel('translation error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_translation_error_plot_' + timestamp + '_' + rand_string + '.jpg'
        fig.savefig(filename)
        plt.close(fig)


        # Plotting shape errors
        fig = plt.figure()
        plt.errorbar(x=kp_noise_var_range, y=cerr_naive_mean, yerr=cerr_naive_var, fmt='-x', color='black',
                     ecolor='gray', elinewidth=1, capsize=3, label='naive')
        plt.errorbar(x=kp_noise_var_range, y=cerr_naive_certi_mean, yerr=cerr_naive_certi_var, fmt='--o', color='grey',
                     ecolor='lightgray', elinewidth=3, capsize=0, label='naive + certification')
        plt.errorbar(x=kp_noise_var_range, y=cerr_corrector_mean, yerr=cerr_corrector_var, fmt='-x', color='red',
                     ecolor='salmon', elinewidth=1, capsize=3, label='corrector')
        plt.errorbar(x=kp_noise_var_range, y=cerr_corrector_certi_mean, yerr=cerr_corrector_certi_var, fmt='--o',
                     color='orangered',
                     ecolor='salmon', elinewidth=3, capsize=0, label='corrector + certification')
        plt.legend(loc='upper left')
        title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        plt.title(title)
        plt.xlabel('noise variance parameter $\sigma$')
        plt.ylabel('shape error')
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_shape_error_plot_' + timestamp + '_' + rand_string + '.jpg'
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
        title = 'CAD model: ' + cad_model_name + ', Noise: ' + kp_noise_type + f' with fra={kp_noise_fra:.2f}'
        plt.title(title)
        plt.show()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rand_string = generate_filename()
        filename = name[:-7] + '_fraction_not_certifiable_plot_' + timestamp + '_' + rand_string + '.jpg'
        fig.savefig(filename)
        plt.close(fig)

