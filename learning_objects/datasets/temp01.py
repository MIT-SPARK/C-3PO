ANNOTATIONS_FOLDER: str = '../../data/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = '../../data/KeypointNet/KeypointNet/pcds/'
MESH_FOLDER_NAME: str = '../../data/KeypointNet/ShapeNetCore.v2.ply/'
OBJECT_CATEGORIES: list = ['airplane', 'bathtub', 'bed', 'bottle',
                           'cap', 'car', 'chair', 'guitar',
                           'helmet', 'knife', 'laptop', 'motorcycle',
                           'mug', 'skateboard', 'table', 'vessel']
CLASS_ID: dict = {'airplane': "02691156",
                  'bathtub': "02808440",
                  'bed': "02818832",
                  'bottle': "02876657",
                  'cap': "02954340",
                  'car': "02958343",
                  'chair': "03001627",
                  'guitar': "03467517",
                  'helmet': "03513137",
                  'knife': "03624134",
                  'laptop': "03642806",
                  'motorcycle': "03790512",
                  'mug': "03797390",
                  'skateboard': "04225987",
                  'table': "04379243",
                  'vessel': "04530566"}

CLASS_NAME: dict = {"02691156": 'airplane',
                    "02808440": 'bathtub',
                    "02818832": 'bed',
                    "02876657": 'bottle',
                    "02954340": 'cap',
                    "02958343": 'car',
                    "03001627": 'chair',
                    "03467517": 'guitar',
                    "03513137": 'helmet',
                    "03624134": 'knife',
                    "03642806": 'laptop',
                    "03790512": 'motorcycle',
                    "03797390": 'mug',
                    "04225987": 'skateboard',
                    "04379243": 'table',
                    "04530566": 'vessel'}

import copy
import csv
import torch
import pandas as pd
import open3d as o3d
import json
import numpy as np

import os
import sys
sys.path.append("../../")

import learning_objects.utils.general as gu




def analyze_models_in_category(category_name='chair'):
    """
    This is to analyze all the models in a given ShapeNet category.
    This is to see if all the models can be compressed with a single shape parameter.

    Note:
        class = category
    """

    if category_name not in set(CLASS_ID.keys()):
        return ValueError


    category_id = CLASS_ID[category_name]
    annotation_file = ANNOTATIONS_FOLDER + CLASS_NAME[str(category_id)] + '.json'
    file_temp = open(str(annotation_file))
    annotation_data = json.load(file_temp)

    # len(annotation_data) = number of cad models in that category
    num_keypoints = set()

    for idx_model in range(len(annotation_data)):
        num_keypoints.add(len(annotation_data[idx_model]['keypoints']))


    return num_keypoints



if __name__ == '__main__':


    for category_name in list(CLASS_ID.keys()):
        print("For category: " + str(category_name))
        num_keypoints = analyze_models_in_category(category_name=str(category_name))
        print(num_keypoints)


