# CAPTRA: CAtegory-level Pose Tracking for Rigid and Articulated Objects from Point Clouds

![teaser](images/teaser.png)


## Introduction

This is the official PyTorch implementation of our paper [CAPTRA: CAtegory-level Pose Tracking for Rigid and Articulated Objects from Point Clouds](https://arxiv.org/abs/2104.03437). This repository is still under construction.

For more information, please visit our [project page](https://yijiaweng.github.io/CAPTRA).

<span class="center"><img src="images/bmvc_ours.gif" width="45%"> <img src="images/real_drawers_ours.gif" width="45%"></span>
<p style="text-align: left; width: 90%; margin-left: 0%"><b>Result visualization on real data.</b> Our models, trained on synthetic data only, can directly generalize to real data, assuming the availability of object masks but not part masks. Left: results on a laptop trajectory from BMVC dataset. Right: results on a real drawers trajectory we captured, where a Kinova Jaco2 arm pulls out the top drawer.</p>



## Citation

If you find our work useful in your research, please consider citing:

```
@article{weng2021captra,
	title={CAPTRA: CAtegory-level Pose Tracking for Rigid and Articulated Objects from Point Clouds},
	author={Weng, Yijia and Wang, He and Zhou, Qiang and Qin, Yuzhe and Duan, Yueqi and Fan, Qingnan and Chen, Baoquan and Su, Hao and Guibas, Leonidas J},
	journal={arXiv preprint arXiv:2104.03437},
	year={2021}
```

## Updates

+ [2021/04/14] Released code, data, and pretrained models for testing & evaluation.
+ [2021/04/22] Released code and data for training.

## Installation

+ Our code has been tested with
  + Ubuntu 16.04, 20.04, and macOS(CPU only)
  + CUDA 11.0
  + Python 3.7.7
  + PyTorch 1.6.0

+ We recommend using [Anaconda](https://www.anaconda.com/) to create an environment named `captra` dedicated to this repository, by running the following:

  ```bash
  conda env create -n captra python=3.7
  conda activate captra
  ```

+ Create a directory for code, data, and experiment checkpoints. 

  ```bash
  mkdir captra && cd captra
  ```
  
+ Clone the repository

  ```bash
  git clone https://github.com/HalfSummer11/CAPTRA.git
  cd CAPTRA
  ```

+ Install dependencies.

  ```bash
  pip install -r requirements.txt
  ```

+ Compile the CUDA code for PointNet++ backbone.

  ```bash
  cd network/models/pointnet_lib
  python setup.py install
  ```

## Datasets

+ Create a directory for all datasets under `captra`

  ```bash
  mkdir data && cd data
  ```

  + Make sure to point `basepath` in `CAPTRA/configs/obj_config/obj_info_*.yml` to your dataset if you put it at a different location.

#### NOCS-REAL275

```bash
mkdir nocs_data && cd nocs_data
```

#### Test

+ Download and unzip [nocs_model_corners.tar](http://download.cs.stanford.edu/orion/captra/nocs_model_corners.tar), where the 3D bounding boxes of normalized object models are saved.

  ```bash
  wget http://download.cs.stanford.edu/orion/captra/nocs_model_corners.tar
  tar -xzvf nocs_real_corners.tar
  ```

+ Create `nocs_full` to hold original NOCS data. Download and unzip "Real Dataset - Test" from [the original NOCS dataset](https://github.com/hughw19/NOCS_CVPR2019#datasets), which contains 6 real test trajectories.

  ```bash
  mkdir nocs_full && cd nocs_full
  wget http://download.cs.stanford.edu/orion/nocs/real_test.zip
  unzip real_test.zip
  ```

+ Generate and run the pre-processing script

  ```bash
  cd CAPTRA/datasets/nocs_data/preproc_nocs
  python generate_all.py --data_path ../../../../data/nocs_data --data_type=test_only --parallel --num_proc=10 > nocs_preproc.sh # generate the script for data preprocessing
  # parallel & num_proc specifies the number of parallel processes in the following procedure
  bash nocs_preproc.sh # the actual data preprocessing
  ```

+ After the steps above, the folder should look like [File Structure - Dataset Folder Structure](#File Structure).

### Train

+ Download and unzip "CAMERA Dataset - Training/Test" and "Real Dataset - Training" from [the original NOCS dataset](https://github.com/hughw19/NOCS_CVPR2019#datasets) under `nocs_data/nocs_full`

  ```bash
  # current path relative to project root (captra): data/nocs_data/nocs_full
  wget http://download.cs.stanford.edu/orion/nocs/camera_train.zip
  unzip camera_train.zip
  wget http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip
  unzip camera_val25K.zip
  wget http://download.cs.stanford.edu/orion/nocs/real_train.zip
  unzip real_train.zip
  ```

  + By now, `nocs_full` should be structured as follows. Note that the depth image (`*_depth.png`) only contains the synthetic foreground objects. For our purpose, we need a complete depth image composing both the synthetic foreground and the real background.

    ```bash
    nocs_full
    ├── real_test
    ├── real_train 
    ├── train	
    │   ├── 00000
    │   │   ├── 0000_color.png, 0000_coord.png, 0000_depth.png, 0000_mask.png, 0000_meta.txt
    │   │   ├── 0001_color.png, ...
    │   │   └── ...
    │   ├── 00001
    │   └── ...
    └── val				# same structure as train
    ```
  
+ Download and unzip "CAMERA Dataset - Composed_depths" from [the original NOCS dataset](https://github.com/hughw19/NOCS_CVPR2019#datasets) under `nocs_data`.  

  ```bash
  cd ../ # current path relative to project root (captra): data/nocs_data
  wget http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip
  unzip camera_composed_depth.zip
  ```

  This will result in a folder named `camera_full_depths`, structured as follows.

  ```bash
  camera_full_depths
  ├── train	
  │   ├── 00000
  │   │   ├── 0000_composed.png # depth image containing both synthetic foreground objects 
  │   │   │									 # and the real background
  │   │   ├── 0001_composed.png # rendered object normalized coordinates
  │   │   └── ...
  │   ├── 00001
  │   └── ...
  └── val				# same structure as train
  ```
  
  Then copy-merge `camera_full_depths` with `nocs_full`.
  
  ```bash
  # merge camera_full_depth/train/????? to nocs_full/train/?????
  rsync -arv camera_full_depths/ nocs_full/
rm -r camera_full_depths
  ```
  
+ Generate and run the pre-processing script

  ```bash
  cd CAPTRA/datasets/nocs_data/preproc_nocs
  python generate_all.py --data_path ../../../../data/nocs_data --data_type=all --parallel --num_proc=10 > nocs_preproc_all.sh # generate the script for data preprocessing
  # parallel & num_proc specifies the number of parallel processes in the following procedure
  bash nocs_preproc_all.sh # the actual data preprocessing
  ```

+ After the steps above, the folder should look like [File Structure - Dataset Folder Structure](#File Structure).

### SAPIEN Synthetic Articulated Object Dataset

```bash
mkdir sapien_data && cd sapien_data
```

#### Test 

+ Download and unzip [object URDF models](http://download.cs.stanford.edu/orion/captra/sapien_urdf.tar) and [testing trajectories](http://download.cs.stanford.edu/orion/captra/sapien_test.tar) 

  ```bash
  wget http://download.cs.stanford.edu/orion/captra/sapien_urdf.tar
  wget http://download.cs.stanford.edu/orion/captra/sapien_test.tar
  tar -xzvf sapien_urdf.tar # urdf
  tar -xzvf sapien_test.tar # render_seq
  ```

### Train

+ Download and unzip [training data](http://download.cs.stanford.edu/orion/captra/sapien_train.tar).

  ```bash
  wget http://download.cs.stanford.edu/orion/captra/sapien_train.tar
  tar -xzvf sapien_train.tar # render
  ```

  

## Testing & Evaluation

### Download Pretrained Model Checkpoints

+ Create a folder `runs` under `captra` for experiments 

  ```bash
  mkdir runs && cd runs
  ```

+ Download our pretrained model checkpoints for

  + NOCS-REAL275:  [nocs_ckpt.tar](http://download.cs.stanford.edu/orion/captra/nocs_ckpt.tar)
  + SAPIEN synthetic articulated object dataset: [sapien_ckpt.tar](http://download.cs.stanford.edu/orion/captra/sapien_ckpt.tar)

+ Unzip them in `runs`

  ```bash
  tar -xzvf nocs_ckpt.tar  
  ```

  which should give

  ```bash
  runs
  ├── 1_bottle_rot 		#	RotationNet for the bottle category
  ├── 1_bottle_coord 	# CoordinateNet for the bottle category
  ├── 2_bowl_rot 
  └── ...
  ```

### Testing

+ To generate pose predictions for a certain category, run the corresponding script in `CAPTRA/scripts/track` (without further specification, all scripts are run from `CAPTRA`), e.g. for the bottle category from NOCS-REAL275,

  ```bash
  bash scripts/track/nocs/1_bottle.sh
  ```

+ The predicted pose will be saved under the experiment folder `1_bottle_rot` (see [File Structure - Experiment Folder Structure](#File Structure)).

+ *To test the tracking speed for articulated objects in SAPIEN, make sure to set `--batch_size=1` in the script. You may use `--dataset_length=500` to avoid running through the whole test set.*

### Evaluation

+ To evaluate the pose predictions produced in the previous step, uncomment and run the corresponding line in `CAPTRA/scripts/eval.sh`, e.g. for the bottle category from NOCS-REAL275, the corresponding line is

  ```bash
  python misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=1 --experiment_dir=../runs/1_bottle_rot
  ```

  

## Training

+ To train the CoordinateNet and RotationNet for a certain category, run the corresponding script in `CAPTRA/scripts/train`, e.g. for the bottle category from NOCS-REAL275, scripts can be found in `CAPTRA/scripts/train/nocs/1_bottle.sh`.

  ```bash
  # RotationNet
  python network/train.py --config=config_rotnet.yml --obj_config=obj_info_nocs.yml \
                          --pose_perturb/r=5.0 --pose_perturb/t=0.03 --pose_perturb/s=0.02 \
                          --batch_size=12 \
                          --obj_category=1 \
                          --experiment_dir=../runs/1_bottle_rot_new \
                          --use_val=real_test \  
                          --num_workers=2
  # CoordinateNet
  python network/train.py --config=config_coordnet.yml --obj_config=obj_info_nocs.yml \
                          --pose_perturb/r=5.0 --pose_perturb/t=0.03 --pose_perturb/s=0.02 \
                          --batch_size=12 \
                          --obj_category=1 \
                          --experiment_dir=../runs/1_bottle_coord_new \
                          --use_val=real_test \
                          --num_workers=2
  ```

+ The output logs and checkpoints will be saved at `1_bottle_rot_new/log`, `1_bottle_rot_new/ckpt`, respectively. (see [File Structure - Experiment Folder Structure](#File Structure)).

+ For NOCS dataset, after training the models using synthetic data, we can further finetune them with real training data. To do this, simply replace `train.py` with `train_nocs_finetune.py` in the training script. Use `--syn_n` and `--real_only` to adjust the proportion of real/synthetic training data used in each epoch.


## File Structure

### Overall Structure

The working directory should be organized as follows.

```bash
captra
├── CAPTRA		# this repository
├── data			# datasets
│   ├── nocs_data		# NOCS-REAL275
│   └── sapien_data	# synthetic dataset of articulated objects from SAPIEN
└── runs			# folders for individual experiments
    ├── 1_bottle_coord
    ├── 1_bottle_rot
    └── ...
```

### Code Structure

Below is an overview of our code. Only the most relevant folders/files are shown.

```bash
CAPTRA
├── configs		# configuration files
│   ├── all_config			# experiment configs
│   ├── pointnet_config 	# pointnet++ configs (radius, etc)
│   ├── obj_config			# dataset configs
│   └── config.py			# parser
├── datasets	# data preprocessing & dataset definitions
│   ├── arti_data			# articulated data
│   │   └── ...
│   ├── nocs_data			# NOCS-REAL275 data
│   │   ├── ...
│   │   └── preproc_nocs	# prepare nocs data
│   └── ...					# utility functions
├── pose_utils	# utility functions for pose/bounding box computation
├── utils.py
├── misc		# evaluation and visualization
│   ├── eval
│   └── visualize
├── scripts		# scripts for training/testing
└── network		# main part
    ├── data		# torch dataloader definitions
    ├── models		# model definition
    │   ├── pointnet_lib
    │   ├── pointnet_utils.py
    │   ├── backbones.py
    │   ├── blocks.py		# the above defines backbone/building blocks
    │   ├── loss.py
    │   ├── networks.py		# defines CoordinateNet and RotationNet
    │   └── model.py		# defines models for training/tracking
    ├── trainer.py			# training agent
    ├── parse_args.py		# parse arguments for train/test
    ├── test.py		# test
    ├── train.py	# train
    └── train_nocs_mix.py	# finetune with a mixture of synthetic/real data
```



### Experiment Folder Structure

For each experiment, a dedicated folder in `captra/runs` is organized as follows.

```bash
1_bottle_rot
├── log		# training/testing log files
│   └── log.txt
├── ckpt	# model checkpoints
│   ├── model_0001.pt
│   └── ...
└── results
    ├── data*		# per-trajectory raw network outputs 
    │   ├── bottle_shampoo_norm_scene_4.pkl
    │   └── ...
    ├── err.csv**	# per-frame error	
    └── err.pkl**	# per-frame error
*: generated after testing with --save
**: generated after running misc/eval/eval.py
```

### Dataset Folder Structure

```bash
nocs_data
├── nocs_model_corners	# instance bounding box information	
├── nocs_full		 	# original NOCS data, organized in frames (not object-centric)
│   ├── real_test
│   │   ├── scene_1
│   │   └── ...
│   ├── real_train
│   ├── train	  		  # see the following
│   └── val			
├── instance_list*		# collects each instance's occurences in nocs_full/*/
├── render*				# per-instance segmented data for training
├── preproc**			# cashed data 	
└── splits**			# data lists for train/test	
*: generated after data-preprocessing
**: generated during training/testing

# Specifically, nocs_data/nocs_full/train (and val) should be structured as follows:
train	
├── 00000
│   ├── 0000_coord.png 		# rendered object normalized coordinates
│   ├── 0000_depth.png 		# depth image containing synthetic foreground objects only
│   ├── 0000_mask.png  		# object mask
│   ├── 0000_meta.txt  		# meta information
│   ├── 0000_composed.png* # depth image containing both synthetic foreground objects 
│   │	  							  # and the real background
│   ├── 0000_pose.pkl**		 # object poses computed from *_coord.png and *_depth.png
│   └── ...
├── 00001
└── ...
*: generated after copy-merging camera_full_depths with nocs_full
**: generated after data-preprocessing

sapien_data
├── urdf				# instance URDF models
├── render_seq			# testing trajectories
├── render**			# single-frame training/validation data
├── preproc_seq*		# cashed testing trajectory data	
├── preproc**			# cashed testing trajectory data
└── splits*				# data lists for train/test	
*: generated during training/testing
**: training
```



##Acknowledgements

This implementation is based on the following repositories. We thank the authors for open sourcing their great works! 

+ PointNet++: [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) and [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

+ [Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation: hughw19/NOCS_CVPR2019](https://github.com/hughw19/NOCS_CVPR2019)

+ [Category-Level Articulated Object Pose Estimation: dragonlong/articulated-pose](https://github.com/dragonlong/articulated-pose)


