#!/bin/bash

conda create -n dibs_env python=3.8
conda activate dibs_env
pip install dibs-lib
pip install "jax[cuda]==0.3.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install scikit-learn wandb