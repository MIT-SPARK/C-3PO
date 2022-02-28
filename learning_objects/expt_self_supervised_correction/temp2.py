
import torch
import pickle
import sys
import argparse
import yaml

sys.path.append("../..")

def VOCap(rec, threshold):

    rec = torch.sort(rec)[0]
    rec = torch.where(rec <= threshold, rec, torch.tensor([float("inf")]))

    print(rec)

    n = rec.shape[0]
    prec = torch.cumsum(torch.ones(n)/n, dim=0)

    index = torch.isfinite(rec)
    rec = rec[index]
    prec = prec[index]

    mrec = torch.zeros(rec.shape[0] + 2)
    mrec[0] = 0
    mrec[-1] = threshold
    mrec[1:-1] = rec

    mpre = torch.zeros(prec.shape[0]+2)
    mpre[1:-1] = prec
    mpre[-1] = prec[-1]

    for i in range(1, mpre.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])

    ap = 0
    for i in range(mrec.shape[0]-1):
        if mrec[i+1] != mrec[i]:
            ap += (mrec[i+1] - mrec[i]) * mpre[i+1] * 100 * (1/threshold)

    return ap


if __name__ == "__main__":

    print("test")
    n = 10000
    d = torch.rand(n)
    d = torch.sort(d)[0]
    # accuracy = torch.cumsum(torch.ones(n)/n, dim=0)

    ap = VOCap(d, threshold=1.0)
    print(ap)
