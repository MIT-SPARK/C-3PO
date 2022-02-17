
import torch
import yaml
import sys
sys.path.append('../..')

from learning_objects.datasets.keypointnet import SE3PointCloud, DepthPC, CLASS_NAME, CLASS_ID
from learning_objects.utils.general import display_two_pcs



if __name__ == "__main__":

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)

    for key, value in model_class_ids.items():

        class_id = CLASS_ID[key]
        class_name = CLASS_NAME[class_id]
        model_id = value

        print(">>" * 40)
        print("Object: ", key, "; Model ID:", str(model_id))

        dataset = SE3PointCloud(class_id=class_id,
                                model_id=model_id,
                                num_of_points=1000,
                                dataset_len=2)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for i, data in enumerate(loader):

            pc, kp, R, t = data
            display_two_pcs(pc1=pc[0, ...], pc2=pc[0, ...])

        dataset = DepthPC(class_id=class_id,
                          model_id=model_id,
                          n=2048,
                          num_of_points_to_sample=1000,
                          dataset_len=5, rotate_about_z=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for i, data in enumerate(loader):

            pc, kp, R, t = data
            display_two_pcs(pc1=pc[0, ...], pc2=pc[0, ...])



