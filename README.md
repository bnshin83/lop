# Loss of Plasticity in Deep Continual Learning

## Contents

- [Overview](#overview)
- [Repository Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](./LICENSE)
- [Citation](./citation.bib)

## Overview

This repository contains the code to reproduce the experiments present in our paper titled [Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7).
A talk about this work can be found [here](https://www.youtube.com/watch?v=p_zknyfV9fY).

### Abstract
Artificial neural networks, deep-learning methods and the backpropagation algorithm form the foundation of modern machine learning and artificial intelligence.
These methods are almost always used in two phases, one in which the weights of the network are updated and one in which the weights are held constant while the network is used or evaluated. 
This contrasts with natural learning and many applications, which require continual learning. 
It has been unclear whether deep-learning methods work in continual-learning settings. 
Here we show that they do not—that standard deep-learning methods gradually lose plasticity in continual-learning settings until they learn no better than a shallow network. 
We show such loss of plasticity using the classic ImageNet dataset and reinforcement-learning problems across a wide range of variations in the network and the learning algorithm. 
Plasticity is maintained indefinitely only by algorithms that continually inject diversity into the network, such as our continual backpropagation algorithm, a variation of backpropagation in which a small fraction of less-used units are continually and randomly reinitialized. 
Our results indicate that methods based on gradient descent are not enough—that sustained deep learning requires a random, non-gradient component to maintain variability and plasticity.

### Citation
Please cite our work if you find it useful:

```latex
@article{dohare2024loss,
  title={Loss of Plasticity in Deep Continual Learning},
  author={Dohare, Shibhansh and Hernandez-Garcia, J. Fernando and Lan, Qingfeng and Rahman, Parash and Mahmood, A. Ruapm and Sutton, Richard S.},
  journal={Nature},
  volume={632},
  pages={768---774},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```


## Repository Contents
- [lop/algos](./lop/algos): All the algorithms used in the paper, including our new continual backpropagation algorithm.
- [lop/nets](./lop/nets): The network architectures used in the paper.
- [lop/imagenet](./lop/imagenet): Demonstration and mitigation of loss of plasticity in a task-incremental problem using ImageNet.
- [lop/incremental_cifar](./lop/incremental_cifar): Demonstration and mitigation of loss of plasticity in a class-incremental problem.
- [lop/slowly_changing_regression](./lop/slowly_changing_regression): A small problem for quick demonstration of loss of plasticity.
- [lop/rl](./lop/rl): Loss of plasticity in standard reinforcement learning problems using the PPO algorithm[1].

The README files in each subdirectory contains further information on the contents of the subdirectory.

## System Requirements

This package only requires a standard computed with sufficient RAM (8GB+) to reproduce the experimental results.
However, a GPU can significantly speed up experiments with larger networks such as the residual networks in [lop/incremental_cifar](./lop/incremental_cifar).
Internet connection is required to download many of the datasets and packages.


The package has been tested on Ubuntu 20.04 and python3.8. We expect this package to work on all machines that support all the packages listed in [`requirements.txt`](requirements.txt)


## Installation Guide

### Option 1: Using virtualenv (Original Method)

Create a virtual environment
```sh
mkdir ~/envs
virtualenv --no-download --python=/usr/bin/python3.8 ~/envs/lop
source ~/envs/lop/bin/activate
pip3 install --no-index --upgrade pip
```

Download the repository and install the requirements
```sh
git clone https://github.com/shibhansh/loss-of-plasticity.git
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
```

Add this lines in your `~/.zshrc` or `~/.bashrc`
```sh
source ~/envs/lop/bin/activate
```

### Option 2 (Boonam): Using Conda

1. Install Miniconda or Anaconda if you haven't already

2. Create a conda environment with Python 3.8 in your scratch directory
```sh
conda create -p /scratch/gautschi/shin283/conda_envs/lop python=3.8 -y
conda activate /scratch/gautschi/shin283/conda_envs/lop
```

3. Download the repository and install the requirements
```sh
git clone https://github.com/shibhansh/loss-of-plasticity.git
cd loss-of-plasticity
```

4. Install PyTorch and torchvision with conda (recommended for better CUDA support)
```sh
conda install pytorch=2.1.0 torchvision=0.16.0 -c pytorch
```

5. Install the remaining dependencies
```sh
pip install free-mujoco-py==2.1.6 gym==0.23.1 pycparser==2.21 zipp==3.16.2 sympy==1.9 setuptools==44.0.0 numpy==1.24.1 mlproj-manager==0.0.29 matplotlib==3.7.2 PyYAML==6.0.1 tqdm==4.66.1 scipy==1.10.1
pip install -e .
```

6. Add this line to your `~/.zshrc` or `~/.bashrc` to automatically activate the environment
```sh
conda activate lop
```

Installation on a normal laptop with good internet connection should only take a few minutes

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.