<div align="center">
  <a href="http://mit.edu/sparklab/">
    <img align="left" src="docs/media/sparklab_logo.png" width="80" alt="sparklab">
  </a> 
  <a href="https://mit.edu"> 
    <img align="right" src="docs/media/mit.png" width="80" alt="mit">
  </a>
</div>

<br>
<br>
<br>

# C3PO: A New Approach to Self-Supervised 3D Object Perception

**Authors:** Rajat Talak, Lisa Peng, [Luca Carlone](https://lucacarlone.mit.edu/)

## Introduction
**todo[lisa]: put some pretty gifs here**

C3PO is a new keypoint-based self-supervised object pose estimation method that uses keypoint correction, a certificate of correctness and a certificate of nondegeneracy to predict and verify object poses from input depth point clouds. 

## Citation
We kindly ask to cite our paper if you find this library useful:

- R. Talak, L. Peng, L. Carlone, "Correct and Certify: A New Approach to Self-Supervised 3D-Object Perception,". [arXiv:2206.11215](https://arxiv.org/abs/2206.11215) [cs.CV], Jun. 2022.


```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.11215,
  doi = {10.48550/ARXIV.2206.11215},
  url = {https://arxiv.org/abs/2206.11215},
  author = {Talak, Rajat and Peng, Lisa and Carlone, Luca},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Correct and Certify: A New Approach to Self-Supervised 3D-Object Perception},
  publisher = {arXiv},
  year = {2022}
}

```
## Datasets

Our experiments rely on the [ShapeNet](https://shapenet.org/), [KeypointNet](https://github.com/qq456cvb/KeypointNet), [YCB](https://www.ycbbenchmarks.com/object-models/) and our processed versions of these datasets. Please view ShapeNet's terms of use [here](https://shapenet.org/terms). There's no need to download the datasets seperately. Follow the steps below to download and save the relevant data for this project. 

1. Download our processed dataset files on Google Drive [here](https://drive.google.com/drive/folders/1IPHZ-KiuT42ugZQ27-CZuiZfvy5LLvIa?usp=sharing) and move all the files to the same directory this README is in (the C3PO repo). We've provided the dataset as a zip archive split into 1GB chunks of the format ```c3po-data.z**```.
2. Combine the archives into one zip file: 
	- ```zip -F c3po-data.zip --out data.zip```
3. Unzip the file:
	- ```unzip data.zip```
4. Verify your directory structure looks as follows:

```
C3PO
│   README.md
│   c3po   
│   setup.py
└───data
│   │   learning-objects
│   |   └───...
│   │
│   │   KeypointNet
│   |   └───...
│   │
│   └───ycb
│       │   models
│       └───...
│   
└───...
```
### Dataset Citations

```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}

@article{you2020keypointnet,
  title={KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations},
  author={You, Yang and Lou, Yujing and Li, Chengkun and Cheng, Zhoujun and Li, Liangwei and Ma, Lizhuang and Lu, Cewu and Wang, Weiming},
  journal={arXiv preprint arXiv:2002.12687},
  year={2020}
}

@article{Calli_2015,
	doi = {10.1109/mra.2015.2448951},
	url = {https://doi.org/10.1109%2Fmra.2015.2448951},
	year = 2015,
	month = {sep},
	publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
	volume = {22},
	number = {3},
	pages = {36--52},
	author = {Berk Calli and Aaron Walsman and Arjun Singh and Siddhartha Srinivasa and Pieter Abbeel and Aaron M. Dollar},
	title = {Benchmarking in Manipulation Research: Using the Yale-{CMU}-Berkeley Object and Model Set},
	journal = {{IEEE} Robotics {\&}amp$\mathsemicolon$ Automation Magazine}
}
```

## Installation

### Environment Setup
We use *anaconda environments* to manage our dependencies. C3PO has been tested on Ubuntu 18.04 with python 3.8 and python 3.9.

We've provided two yml environment files to clone from. This will be the easiest way to replicate our environment and install all dependencies. `environment_learning_objects_38.yml` uses python 3.8 and cuda 11.1. `environment_learning_objects_39.yml` uses python 3.9 and cuda 10.2. Both virtual environments are compatible with our code.

```
# Clone the environment. This line will take a while to execute. 
conda env create -f environment_learning_objects_3*.yml

# Activate the environment
# The first line of the yml file sets the environment's name:
# learning-objects-00 or learning-objects-01
conda activate <NAME OF ENVIRONMENT>

# Clone C3PO
git clone https://github.com/MIT-SPARK/C3PO.git

# Install C3PO
python setup.py develop

# Verify environment installed
conda env list

```
### Verify the following libraries are installed:

- cudatoolkit
- pytorch
- pytorch-geometric
- fvcore
- iopath
- bottler
- pytorch3d
- scipy
- pymanopt
- yaml
- open3d
- jupyterlab
- cvxpy
- cvxpylayers 



# Usage
Quick Links:

- [Proposed Model](##proposed-model)

- [Experiments Overview](##experiments-overview)
	- [Experiment 1](##i-experiment-1)
	- [Experiment 2](##ii-experiment-2)
	- [Experiment 3](##iii-experiment-3)
 	- [Experiment 4](##iv-experiment-4)

## Proposed Model
Our proposed model is in `c3po/expt_shapenet/proposed_model.py` for use with the shapenet dataset and `c3po/expt_ycb/proposed_model.py` for use with the ycb dataset. A brief description of parameters is below:

```
- class_name/model_id: The category of the object.
- model_keypoints: torch.tensor of shape (K, 3, N). The keypoints of our cad model where N is number of keypoints, specified per model.
- cad_models: torch.tensor of shape (K, 3, n). Sampled point cloud of the cad model with number of points specified in the dataloader.
- keypoint_detector: Specifies which keypoint detector architecture to use. Supports 'pointnet' and 'point_transformer' out of the box. See RegressionKeypoints class to customize your own detector.
- local_max_pooling: Boolean to specify whether to take the max or mean of local features in the forward pass of the 'point_transformer' keypoint detector (if used).
- correction_flag: Boolean to specify whether to use the corrector or not.
- need_predicted_keypoints: Boolean to specify whether to return predicted model keypoints (ground truth model keypoints transformed by predicted R and t) in the forward pass.
```

## Experiments Overview
Our repository is organized into experiments inside our c3po folder. The numbering corresponds to the order of appearance in our paper. Read descriptions under each experiment for details.

## i. Experiment 1: 
`c3po/expt_keypoint_corrector_analysis/`
### Description
This experiment aims to show the effectiveness of our keypoint corrector module. It uses shapenet dataset models. For each input point cloud, we perturb 80% of the the keypoints with varying amounts of noise and then pass the input through the corrector module and then the registration module. Averaged errors for 100 iterations of the corrector forward pass per noise variance parameter are saved for plot generation.

### Results
Our plots from the paper are saved at filepath: `c3po/expt_keypoint_corrector_analysis/expt_with_reg_depthpc/<CLASS_ID>/<MODEL_ID>_wchamfer/`

### Replication
To run our full experiment and save metrics for plot generation, run: 

```
cd c3po/expt_keypoint_corrector_analysis/
python expt_with_reg_depthpc.py
```

The experiment will save metrics for plot generation in the filepath `c3po/expt_keypoint_corrector_analysis/expt_with_reg_depthpc/<CLASS_ID>/<MODEL_ID>_wchamfer/<TIMESTAMP>_experiment.pickle`

To regenerate plots, change the `file_names` parameter inside `expt_with_reg_depthpc_analyze.py` to the pickle filepath containing saved metrics from the previous step, and run `python expt_with_reg_depthpc_analyze.py`

## ii. Experiment 2: 
`c3po/expt_shapenet/`
### Description

This folder contains our proposed model as well as supervised training, self-supervised training, various ICP baseline training, and evaluation code for ***simulated*** depth point clouds using shapenet models. 

### Results
Saved models are saved at filepath: `c3po/expt_shapenet/<CLASS_NAME>/<MODEL_ID>/`

### Replication
To run training and save models for evaluation (***this will overwrite existing models***), run: 

```
cd c3po/expt_shapenet/

# to run supervised training, edit and run
bash handy_supervised_train.sh

# to run self-supervised training, edit and run
bash handy_self_supervised_train.sh

# to run baseline training, edit and run
bash handy_train_baseline.sh

```

To evaluate trained models, run:

```
cd c3po/expt_shapenet/

# for models trained with supervision, edit and run
bash handy_evaluate_sim_trained.sh

# for models trained with self-supervision, edit and run
bash handy_evaluate.sh

# for baseline models, edit and run
bash handy_evaluate_baseline.sh
```

To run ICP and/or RANSAC baselines, run:

```
cd c3po/expt_shapenet/

# edit and run
bash handy_evaluate_icp.sh
```


## iii. Experiment 3: 
`c3po/expt_ycb/`
### Description

This folder contains our proposed model as well as supervised training, self-supervised training, various ICP baseline training, and evaluation code for ***real*** depth point clouds using ycb models. 

### Results
Saved models are saved at filepath: `c3po/expt_ycb/<MODEL_ID>/`

### Replication
To run training and save models for evaluation (***this will overwrite existing models***), run: 

```
cd c3po/expt_ycb/

# to run supervised training, edit and run
bash handy_supervised_train.sh

# to run self-supervised training, edit and run
bash handy_self_supervised_train.sh

# to run baseline training, edit and run
bash handy_train_baseline.sh

```

To evaluate trained models, run:

```
cd c3po/expt_ycb/

# for models trained with supervision, edit and run
bash handy_evaluate_sim_trained.sh

# for models trained with self-supervision, edit and run
bash handy_evaluate.sh

# for baseline models, edit and run
bash handy_evaluate_baseline.sh
```

To run ICP and/or RANSAC baselines, run:

```
cd c3po/expt_ycb/

# edit and run
bash handy_evaluate_icp.sh
```

## iv. Experiment 4: 
`c3po/expt_fully_self_supervised/`
### Description
This folder contains our training and evaluation code for our proposed model using multiple object types as input data for ***simulated*** and ***real*** depth point clouds using shapenet and ycb models respectively. 

### Results
Saved models are saved at filepath: `c3po/expt_fully_self_supervised/<MODEL_ID>/` for ycb objects and `c3po/expt_fully_self_supervised/<CLASS_NAME>/<MODEL_ID>/` for shapenet objects.

### Replication
To run training and save models for evaluation (***this will overwrite existing models***), run: 

```
cd c3po/expt_fully_self_supervised/

bash handy_train_<shapenet/ycb>.py

```

To evaluate trained models, run:

```
cd c3po/expt_fully_self_supervised/

bash handy_evaluate_<shapenet/ycb>.py
```


# License
Our C3PO project is released under MIT license.
