
import yaml
import math

ycb_object_labels = {
                    "001_chips_can": "chips can",
                    "002_master_chef_can": "master chef can",
                    "003_cracker_box": "cracker box",
                    "004_sugar_box": "sugar box",
                    "005_tomato_soup_can": "tomato soup can",
                    "006_mustard_bottle": "mustard bottle",
                    "007_tuna_fish_can": "tuna fish can",
                    "008_pudding_box": "pudding box",
                    "009_gelatin_box": "gelatin box",
                    "010_potted_meat_can": "potted meat can",
                    "011_banana": "banana",
                    "019_pitcher_base": "pitcher base",
                    "021_bleach_cleanser": "bleach cleanser",
                    "035_power_drill": "power drill",
                    "036_wood_block": "wood block",
                    "037_scissors": "scissors",
                    "040_large_marker": "large marker",
                    "051_large_clamp": "large clamp",
                    "052_extra_large_clamp": "extra large clamp",
                    "061_foam_brick": "foam brick"
}

base_folder = "../../"
files = [
    "c3po/expt_shapenet/self_supervised_training.yml",
    "c3po/expt_ycb/self_supervised_training.yml"
]


def get_epsilon(experiment):

    if experiment == "shapenet":
        file_ = files[0]
        filename = "runs/epsilon_expt_shapenet.tex"
    elif experiment == "ycb":
        filename = "runs/epsilon_expt_ycb.tex"
        file_ = files[1]
    else:
        raise ValueError("experiment incorrectly specified.")

    data = dict()
    data['object'] = []
    data['epsilon'] = []

    stream = open(base_folder + file_, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)

    d_ = hyper_param['point_transformer']['epsilon']
    for key, val in d_.items():

        ep_ = 100 * math.sqrt(math.log(1/val))
        data['object'].append(key)
        data['epsilon'].append(ep_)

    # creating latex table
    lines = []
    if experiment == "shapenet":
        items_per_row = 6
    else:
        items_per_row = 5

    data_size = len(data['object'])
    num_rows = data_size // items_per_row + 1

    line_ = "\\begin{tabular}{|"
    for idx in range(items_per_row):
        line_ += "l|r|"
    line_ += "}"
    lines.append(line_)

    lines.append("\\toprule")

    line_ = ""
    for idx in range(items_per_row):
        line_ += "  object & $\\epsilon_\\ocx$  "

        if idx < items_per_row - 1:
            line_ += " & "
        else:
            line_ += " \\\\"
    lines.append(line_)

    lines.append("\\midrule")
    idx_current = 0

    for row_ in range(num_rows):
        for col_ in range(items_per_row):

            if idx_current < data_size:
                if "ycb" in experiment:
                    line_ = ycb_object_labels[data['object'][idx_current]]
                else:
                    line_ = data['object'][idx_current]
                line_ += " & "
                line_ += f" {data['epsilon'][idx_current]:.2f}"
            else:
                line_ = " & "

            idx_current += 1
            if col_ < items_per_row - 1:
                line_ += " & "
            else:
                line_ += " \\\\"

            lines.append(line_)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # saving
    with open(filename, "w") as f:
        f.write('\n'.join(lines))

    return data


if __name__ == "__main__":

    get_epsilon("shapenet")
    get_epsilon("ycb")