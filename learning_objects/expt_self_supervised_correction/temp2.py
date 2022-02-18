
import torch
import pickle
import sys
sys.path.append("../..")


if __name__ == "__main__":

    print("test")
    # plotting %certifiable during training
    filename = './chair/1e3fba4500d20bb49b9f2eb77f5e247e/_certi_all_batches_pointnet.pkl'

    with open(filename, 'rb') as f:
        fra_cert = pickle.load(f)

    print(fra_cert.val)
