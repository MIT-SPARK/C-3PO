import json

def main():

    annotations_folder = '../../../datasets/KeypointNet/KeypointNet/annotations/'
    pcd_folder_name = '../../../datasets/KeypointNet/KeypointNet/pcds/'
    mesh_folder_name = '../../../datasets/KeypointNet/ShapeNetCore.v2.ply/'
    object_categories = ['airplane', 'all', 'bathtub', 'bed',
                         'bottle', 'cap', 'car', 'chair', 'guitar',
                         'helmet', 'knife', 'laptop', 'motorcycle',
                         'mug', 'skateboard', 'table', 'vessel']


    annotations_file = json.load(annotations_folder)

if __name__ == "__main__":
    main()

