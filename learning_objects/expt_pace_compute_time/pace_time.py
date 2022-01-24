"""
This code will compare the compute time of PACE (w. SDP solver) and PACE (w. Alten solver).

"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("../../")

from learning_objects.utils.general import generate_random_keypoints, generate_filename
from learning_objects.models.pace import PACEmodule
from learning_objects.models.pace_altern_ddn import PACEbp



if __name__ == "__main__":

    # Test: PACEmodule
    print('Testing PACEmodule(torch.nn.Module)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is ', device)
    print('-' * 20)

    N = 44
    K = 69
    n = 100
    weights = torch.rand(N, 1).to(device=device)
    model_keypoints = torch.rand(K, 3, N).to(device=device)
    lambda_constant = torch.tensor([1.0]).to(device=device)
    cad_models = torch.rand(K, 3, n).to(device=device)

    pace_sdp_cvxpy = PACEmodule(model_keypoints=model_keypoints).to(device=device)
    pace_altern_gpu = PACEbp(model_keypoints=model_keypoints)

    time_sdp_cvxpy = []
    time_altern_gpu = []
    Brange = [1, 5, 10, 100, 500]
    for B in Brange:

        temp_sdp = 0
        temp_altern = 0

        for iter in range(20):

            print("Batch: ", B, "Iter: ", iter)

            keypoints, _, _, _ = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints.to('cpu'))
            keypoints = keypoints.to(device=device)
            # rotations = rotations.to(device=device)
            # translations = translations.to(device=device)
            # shape = shape.to(device=device)

            keypoints.requires_grad = True
            a = time.perf_counter()
            b = time.perf_counter()

            start = time.perf_counter()
            rot_est1, trans_est1, shape_est1 = pace_sdp_cvxpy(keypoints)
            end = time.perf_counter()
            print("SDP time (per problem): ", (end - start) / B)
            temp_sdp += (end-start)/B

            start = time.perf_counter()
            rot_est2, trans_est2, shape_est2 = pace_altern_gpu.forward(keypoints)
            end = time.perf_counter()
            print("Altern time (per problem): ", (end - start) / B)
            temp_altern += (end - start) / B


        time_sdp_cvxpy.append(temp_sdp/10.0)
        time_altern_gpu.append(temp_altern/10.0)


    # plotting
    name = "./runs/expt_pace_compute_time"
    fig = plt.figure()
    plt.plot(Brange, time_sdp_cvxpy, 'o--', color='r', label='PACE')
    plt.plot(Brange, time_altern_gpu, 'o--', color='b', label='Altern')
    plt.xlabel("Number of batches")
    plt.ylabel("Compute time per batch (ms)")
    plt.legend(loc='upper right')
    plt.xlim([Brange[0], Brange[-1]])
    plt.show()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rand_string = generate_filename()
    filename = name + '_' + timestamp + '_' + rand_string + '.pdf'
    fig.savefig(filename)
    plt.close(fig)







