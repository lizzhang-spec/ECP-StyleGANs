# ECP-StyleGANs
This is the official code for Evolutionary Channel Pruning for Style-Based Generative Adversarial Networks

The code is heavily based on the [StyleGAN](https://github.com/sangwoomo/FreezeD) and [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) codes.


# Usage
This section of the README walks through how to search, fine-tune, and evaluate the ECP-StyleGANs.


## Preparing data
For both experiments on StyleGAN and StyleGAN2, first create lmdb datasets:
> python prepare_data.py


## StyleGAN

### Download pre-traind GAN models
Please follow the codebase at [StyleGAN](https://github.com/sangwoomo/FreezeD)

### Pre-compute FID activations
> python precompute_acts.py --dataset DATASET

### Run experiments
> python search.py

> python finetune.py

> python test.py  ## this is for the sample and evaluation

### Comparison experiments: L1 pruning and random pruning
> python L1.py

> python RC.py

### Results on Animal Face and AFHQ dataset
See the published paper



## StyleGAN2

### Convert weight from official checkpoints
Please follow the codebase at [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch)

### Run experiments
> python search.py

> python train.py ## this is for finetuning

> python generate.py  ## this is for the sample and evaluation

### Results on FFHQ-256 dataset
See the published paper

## Citation
If you use this code for your research, please cite our papers.
```
To be updated...
```
