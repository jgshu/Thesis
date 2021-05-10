##!/bin/bash
#set -ex

# install pyflakes to do code error checking
echo "pip3 install pyflakes --cache-dir $HOME/.pip-cache"
pip3 install pyflakes --cache-dir $HOME/.pip-cache

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

echo "conda create -n fedml python=3.7.4"
conda create -n fedml python=3.7.4

echo "conda activate fedml"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fedml

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install pytorch torchvision cudatoolkit -c pytorch

# Install MPI
conda install -c anaconda mpi4py

# Install Wandb
pip install --upgrade wandb

# Install other required package
conda install scikit-learn
conda install numpy
conda install h5py
conda install setproctitle
conda install networkx
pip install -r requirements.txt