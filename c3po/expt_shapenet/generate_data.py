

import yaml
import sys
sys.path.append("../..")
from c3po.expt_shapenet.evaluation import generate_depthpc_eval_data

if __name__ == "__main__":

    print("Do not run this. As the data has already been generated.")

    # #ToDo: Do not run this. The data has already been generated.
    #
    # print("GENERATING DATA FOR SELF-SUPERVISED TRAINING")
    #
    # stream = open("class_model_ids.yml", "r")
    # model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    # stream = open("self_supervised_training.yml", "r")
    # param = yaml.load(stream=stream, Loader=yaml.Loader)
    #
    # # run this to generate evaluation data
    # generate_depthpc_eval_data(model_class_ids=model_class_ids, param=param)
    #
    print("GENERATING DATA FOR EVALUATION")

    stream = open("class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    stream = open("evaluation_datagen.yml", "r")
    param = yaml.load(stream=stream, Loader=yaml.Loader)

    # run this to generate evaluation data
    generate_depthpc_eval_data(model_class_ids=model_class_ids, param=param)


