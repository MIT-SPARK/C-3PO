# Conda Environment "env-c3po"

This describes a way to install all the required packages into a conda environment for our C-3PO project.

This is now successfull and I am using this conda environment to write, run, and test my code.

```
conda create --name env-c3po python=3.8
```

With this, I got the conda virtual environment with python version 3.8.12. 

This environment does not have any torch or cuda installed. I, therefore, have to install the following packages:
- pytorch (preferably 1.8.2 with cuda 11.1)
- pytorch-geometric (2.0.2)
- open3d 
- pytorch3d (be careful how you install this, as it broke my conda environment the last time)
- matplotlib
- cvxpy
- cvxpylayers
- pymanopt
- jupyterlab
- pandas


Installing torch LTS(1.8.2) with cuda 11.1
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

Installing torch-geometric
```bash
conda install pyg -c pyg -c conda-forge
```

Installing open3d
```bash
conda install -c open3d-admin -c conda-forge open3d
```

Installing pytorch3d dependencies
```bash
conda install -c bottler nvidiacub
conda install jupyter
```

Installing pip and git on conda
```bash
conda install pip git 
```

Installing pytorch3d, by building from source
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Installing matplotlib
```bash
conda install matplotlib  
```

Installing pandas
```bash
conda install pandas
```

Installing tensorboard
```bash
conda install -c conda-forge tensorboard
```

## Optional:
Installing cvxpy
```bash
conda install -c conda-forge cvxpy
```

Installing cvxpylayers
```bash
pip install cvxpylayers
```

Installing jupyterlab
```bash
conda install -c conda-forge jupyterlab
```

Installing pymanopt
```bash
pip install pymanopt
```



## Problems:
1. **pytorch3d installation gave me a problem.** I had installed from pre-built wheels. However, my requirement was an older version of torch and python. The compatible older version of pytorch3d had some bugs. I therefore then built pytorch3d from source using pip in conda. This problem is now resolved.

