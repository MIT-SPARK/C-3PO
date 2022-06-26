#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
 

# Loading the required module
source /etc/profile
module unload anaconda
module unload cuda
module load anaconda/2020b
module load cuda/10.2

# loading environment
export PYTHONNOUSERSITE=True
which python
source activate learning-objects-02
which python
 

# Run the script
python self_supervised_training.py "pointnet" "mug" >> logs/stdout_self_supervised_mug.log
