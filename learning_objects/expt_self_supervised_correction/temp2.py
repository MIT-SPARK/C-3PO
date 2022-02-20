
import torch
import pickle
import sys
import argparse

sys.path.append("../..")

if __name__ == "__main__":

    print("test")

    parser = argparse.ArgumentParser()
    parser.add_argument("detector_type", help="specify the detector type.", type=str)
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)

    args = parser.parse_args()

    print("KP detector type: ", args.detector_type)
    print("CAD Model class: ", args.class_name)

