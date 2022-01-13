import copy

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api','utils'))
import apollo_utils as uts
from car_instance.car_models import *
import learning_objects.utils.general as gu



# ANNOTATIONS_FOLDER: str = '../../data/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = '../../datasets/apollo_car_3d/pointclouds/apolloscape_pointclouds_09072021/'
DEPTH_PCD_FOLDER_NAME: str = '../../datasets/apollo_car_3d/pointclouds/apolloscape_pointclouds_depth/'
MESH_FOLDER_NAME: str = '../../datasets/apollo_car_3d/3d_car_instance_sample/car_models/'
OBJECT_CATEGORIES: list = ['car']
