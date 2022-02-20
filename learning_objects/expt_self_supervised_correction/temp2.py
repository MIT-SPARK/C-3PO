
import torch
import pickle
import sys
import argparse
import yaml

sys.path.append("../..")

if __name__ == "__main__":

    hyper_param_file = "self_supervised_training.yml"
    detector_type = "point_transformer"
    key = "guitar"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]
    hyper_param['epsilon'] = hyper_param['epsilon'][key]

    print(hyper_param)


