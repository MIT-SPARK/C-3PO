
import torch
import open3d as o3d
import sys
import pandas as pd
from pathlib import Path

sys.path.append("../..")
from c3po.datasets.shapenet import FixedDepthPC, CLASS_ID, CLASS_MODEL_ID
from c3po.datasets.shapenet import ShapeNet
from c3po.datasets.utils_dataset import toFormat
from c3po.analyze_data.utils import analyze_registration_dataset, plot_cdf
from c3po.utils.visualization_utils import visualize_torch_model_n_keypoints, display_two_pcs

num_of_points_to_sample = 1000
num_of_points_selfsupervised = 2048
eval_dataset_len = 50
eval_batch_size = 25  # 50


class dataWrapper:
    def __init__(self):
        self.none = 1.0

    def __call__(self, x):
        pc2, kp2, R, t = x

        T = torch.eye(4).to(device=R.device)
        T[:3, :3] = R
        T[:3, 3:] = t

        return None, pc2, T


def analyze_data(dataset, class_name):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    class_id = CLASS_ID[class_name]
    model_id = CLASS_MODEL_ID[class_name]

    if dataset is None or dataset == "shapenet":

        eval_dataset = FixedDepthPC(class_id=class_id, model_id=model_id,
                                    n=num_of_points_selfsupervised,
                                    num_of_points_to_sample=num_of_points_to_sample,
                                    dataset_len=eval_dataset_len,
                                    rotate_about_z=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

        data_type = "shapenet"
        object_name = class_name

    else:

        if dataset not in ["shapenet.sim.easy", "shapenet.sim.hard", "shapenet.real.easy", "shapenet.real.hard"]:
            raise ValueError("dataset not specified correctlry.")
            # return None

        base_folder = str(Path(__file__).parent.parent.parent) + '/data'
        dataset_path = base_folder + '/' + dataset + '/' + class_name + ".pkl"
        type = dataset.split('.')[1]
        adv_option = dataset.split('.')[2]
        eval_dataset = ShapeNet(type=type, object=class_name,
                                length=50, num_points=1024, adv_option=adv_option,
                                from_file=True,
                                filename=dataset_path)
        eval_dataset = toFormat(eval_dataset)

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

        data_type = dataset
        object_name = class_name

    # get cad models
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # visualize
    pc2, kp2, R, t = eval_dataset[0]
    # breakpoint()
    est = R @ cad_models.squeeze(0) + t
    display_two_pcs(cad_models, pc2.unsqueeze(0))
    display_two_pcs(est.unsqueeze(0), pc2.unsqueeze(0))

    # analyze rotation/translation errors
    rerr, terr = analyze_registration_dataset(eval_dataset, data_type, transform=dataWrapper())

    plot_cdf(data=rerr, label="rotation", filename='./data/' + str(data_type) + "rerr_test")
    plot_cdf(data=terr, label="translation", filename= './data/' + str(data_type) + "terr_test")

    # saving
    data_ = dict()
    data_["rerr"] = rerr
    data_["terr"] = terr

    df = pd.DataFrame.from_dict(data_)
    filename = './data/' + data_type + '_test.csv'
    df.to_csv(filename)


if __name__ == "__main__":

    class_name = "bed"
    datasets = ["shapenet.sim.easy", "shapenet.sim.hard",
                "shapenet.real.easy", "shapenet.real.hard"]
    # datasets = ["shapenet.real.easy"]

    for dataset in datasets:
        analyze_data(dataset, class_name)















