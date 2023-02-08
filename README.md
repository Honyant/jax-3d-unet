# JAX 3D U-Net

This repository contains code for training a 3D U-Net with JAX, using equinox as a deep learning library.
This code has been tested on TPUv3-8 and 8xV100 GPUs.
This repository offers a standard U-Net, a ViT-based U-Net, and a novel U-Net with a tier-like architecture, allowing users to choose the model that best suits their needs.

## Prerequisites

To run this code, you will need clone:
`git clone https://github.com/honyant/jax-3d-unet.git`

You will also need to install the following dependencies:
`pip install -r requirements.txt`

wandb is used for logging and visualization. To sign up for a free account, visit https://wandb.ai/site.
Follow the instructions to install the wandb package and login.

## Training

To train the model, run the following command:

`python unet3d-jax/train.py`

## Results

You can view the results using wandb.