# Certifiable 3D Object Pose Estimation 

Authors: Rajat Talak and Lisa Peng

Certifiable 3D Object Pose Estimation (C-3PO) is an open-source implementation of our paper: 
"*Correct and Certify: A New Approach to Self-Supervised 3D Object Perception*".
This repository helps reproduce the experimental results reported in the paper and provides trained models for use.

## Paper 

Our work solves the certifiable object pose estimation problem, where -- given a partial point cloud of an object -- the goal 
is to estimate the object pose, fit a CAD model to the sensor data, and provide certification guarantees.

#### R. Talak, L. Peng, L. Carlone, Correct and Certify: A New Approach to Self-Supervised 3D-Object Perception, June 2022 [[arXiv](https://arxiv.org/abs/2206.11215)]

**Abstract:** We consider a certifiable object pose estimation problem, where -- given a partial point cloud of an 
object -- the goal is to estimate the object pose, fit a CAD model to the sensor data, and provide certification 
guarantees. We solve this problem by combining (i) a novel self-supervised training approach, and (ii) a certification 
procedure, that not only verifies whether the output produced by the model is correct or not (i.e. *certifiability*), 
but also flags uniqueness of the produced solution (i.e. *strong certifiability*). We use a semantic keypoint-based 
pose estimation model, that is initially trained in simulation and does not perform well on real-data due to the 
domain gap. Our self-supervised training procedure uses a *corrector* and a *certification* module to improve the 
detector. The corrector module corrects the detected keypoints to compensate for the domain gap, and is implemented 
as a declarative layer, for which we develop a simple differentiation rule. The certification module declares 
whether the corrected output produced by the model is certifiable (i.e. correct) or not. At each iteration, the 
approach optimizes over the loss induced only by the certifiable input-output pairs. As training progresses, we see 
that the fraction of outputs that are certifiable increases, eventually reaching near 100% in many cases. We conduct 
extensive experiments to evaluate the performance of the corrector, the certification, and the proposed self-supervised 
training using the ShapeNet and YCB datasets, and show the proposed approach achieves performance comparable to fully 
supervised baselines while not using any annotation for supervision on real data. 

If you find this repository useful, do cite our work:

```bibtex
@article{Talak22arxiv-correctAndCertify,
  title = {Correct and {{Certify}}: {{A New Approach}} to {{Self-Supervised 3D-Object Perception}}},
  author = {Talak, Rajat and Peng, Lisa and Carlone, Luca},
  year = {2022},
  month = {Jun.},
  journal = {arXiv preprint arXiv: 2206.11215},
  eprint = {2206.11215},
  note = {\linkToPdf{https://arxiv.org/pdf/2206.11215.pdf}},
  pdf={https://arxiv.org/pdf/2206.11215.pdf},
  Year = {2022}
}

```


## Installation 

Clone the repository and install a conda environment from the yml file:
```bash
git clone https://github.com/MIT-SPARK/C-3PO.git 
cd C-3PO/
conda env create -f environment.yml
conda activate c3po
```
Setup the repository with dataset and downloaded pre-trained models. See instructions [here](docs/repository-setup.md)

## Experiments

### Data Analysis
This analysis is to show the distribution of pose transformation error --namely, rotation and translation error-- 
induced in the pose estimation dataset. The performance of several baselines critically depend on this distribution.

To see this distribution:
```bash
cd results/data_analysis
jupyter notebook data_analysis.ipynb
```

### Keypoint Corrector Analysis

#### Description 
This experiment aims to show the effectiveness of our keypoint corrector module. It uses ShapeNet dataset models. 
For each input point cloud, we perturb 80% of the the keypoints with varying amounts of noise and then pass the 
input through the corrector module and then the registration module. Averaged ADD-S errors for 100 iterations of 
the corrector forward pass per noise variance parameter are saved for plot generation. 

#### Replication
[//]: # (To replicate our results do the following. )
To run experiments and save performance metrics for plot generation.
```bash
cd scripts/expt_keypoint_corrector_analysis
bash analyze.sh
```

Generate plots from saved data: 
```bash
cd results/expt_keypoint_corrector_analysis
jupyter notebook results.ipynb
```

[//]: # (|<img src="docs/media/table-adds.jpg" width="100%">|<img src="docs/media/vessel-adds.jpg" width="100%">|<img src="docs/media/skateboard-adds.jpg" width="100%">|)

[//]: # (|:---:|:---:|:---:|)

[//]: # (| corrector results on table model | corrector results on vessel model | corrector results on skateboard model |)

[//]: # ()

### The ShapeNet Experiment

#### Description 
This experiment shows the success of the proposed self-supervised training on a dataset of simulated depth point clouds using ShapeNet models. We are able to generate data across various object categories in ShapeNet and show the power of our proposed model in matching a supervised baseline, without using any annotation on the generated training data.

#### Replication
[//]: # (The proposed model requires one to specify the object category and the architecture used for the keypoint detector. We show how to train and evaluate the proposed model for **object**: *chair* and **keypoint detector**: *point transformer*. )

Trained models are saved in the repository. Evaluate and visualize the results with:
```bash
cd scripts/expt_shapenet
bash evaluate_real.sh
bash evaluate_sim.sh

cd ../../
cd results/expt_shapenet_ycb
jupyter notebook results.ipynb
```
For training models see instructions [here](docs/training-models.md).


### The YCB Experiment 

#### Description 
This experiment shows that the proposed self-supervised training method also works on a real-world dataset comprised of RGB-D images. We see that the proposed model -- after self-supervised training -- is able to match or exceed the performance of a supervised baseline, without using any annotations for training.

#### Replication
[//]: # (The proposed model requires one to specify the object category and the architecture used for the keypoint detector. We show how to train and evaluate the proposed model for **object**: *002\_master\_chef\_can* and **keypoint detector**: *point transformer*. )

Trained models are saved in the repository. Evaluate and visualize the results with:
```bash
cd scripts/expt_ycb
bash evaluate.sh

cd ../../
cd results/expt_shapenet_ycb
jupyter notebook results.ipynb
```
For training models see instructions [here](docs/training-models.md).


### Categoryless Self-Supervised Training Experiment

#### Description
The proposed self-supervised training works even when the unannotated data does not have category labels.
This experiment validates it. 

#### Replication


### Corrector Compute Time Analysis
We implement a constant step size batch gradient descent to solve the corrector optimization problem in the forward pass. 
We show that this results in faster forward compute time, in training. The analysis here validates it.

To analyze compute time --time to solve the corrector optimization problem-- per data point in a batch,
as a function of the batch size, run:
```bash
cd scripts/expt_corrector_compute_time_analysis
bash analyze.sh
```

To plot the results, run:
```bash
cd results/expt_corrector_compute_time_analysis
bash plot.sh
```


## License
Our C-3PO project is released under MIT license.


## Acknowledgement
This work was partially funded by ARL DCIST CRA W911NF-17-2-0181, ONR RAIDER N00014-18-1-2828, and NSF CAREER award "Certifiable Perception for Autonomous Cyber-Physical Systems".
