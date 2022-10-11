# Manually Creating Conda Environment

This describes a way to manually create a conda environment for C-3PO project; with python 3.8.
You can run the following as a shell script:

```bash
conda create --name c3po python=3.8
conda activate c3po
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install pyg -c pyg -c conda-forge
conda install -c open3d-admin -c conda-forge open3d
conda install -c bottler nvidiacub
conda install jupyter
conda install pip git 
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
conda install matplotlib  
conda install pandas
conda install -c conda-forge tensorboard
conda install -c conda-forge jupyterlab
```
