"""
Description:
Train 3D diffusion model.
 
Run as: 
python DiffusionModel_3D_Training.py

For multi-GPU:
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu DiffusionModel_3D_Training.py

Contributors:
Originally adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import inspect
import dask.array as da
import h5py
import time
import json

from video_diffusion_pytorch import Unet3D, GaussianDiffusion
from argparse import ArgumentParser
from skimage.transform import resize

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

from guidedfwi.diffusion3d import Trainer
from guidedfwi.plots import plot_slices
from guidedfwi.utils import log_experiment, pad_dask_array_to_target_size, extract_cubes_fast
from guidedfwi.diffusion2d import normalize_to_zero_to_one


# Define a block‑wise normalizer: each block has shape (b, X, Y, Z)
def normalize_per_cube(block):
    # block.ndim == 4, block.shape == (b, cube_size, cube_size, cube_size)
    # compute min/max over each cube
    mins = block.min(axis=(1,2,3), keepdims=True)
    maxs = block.max(axis=(1,2,3), keepdims=True)
    # avoid divide‑by‑zero
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    return (block - mins) / denom


if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument(
    "--training_data",
    type=str,
    default='Combined',
    )
    parser.add_argument(
    "--cube_size",
    type=int,
    default=64, # or velocity
    )
    parser.add_argument(
    "--unet_size",
    type=int,
    default=128,
    )
    parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    )
    
    ##################################################################
    # Experiment logging
    ##################################################################

    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = (
        '../results/DiffusionModel_3D_Training_'+str(run_id)
    )
    
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        
    # Log experiment
    log_experiment(
        results_folder, script_path=os.path.abspath(inspect.getfile(inspect.currentframe())), run_id=run_id
    )
    with open(os.path.join(results_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ##################################################################
    # Data Generation
    ##################################################################

    if args.training_data == 'Combined':
    
        if not os.path.exists(data_path): # Needs update

            print('Creating Dataset...')

            # Configuration for each model
            model_configs = [
                {
                    "name": "segsalt",
                    "loader": lambda: resize(
                        np.fromfile(
                            "../data/velocities/salt",
                            dtype=np.float32
                        ).reshape(676,676,210),
                        (768,768,768),
                        order=3,
                        mode='reflect'
                    ),
                },
                {
                    "name": "seamarid",
                    "loader": lambda: resize(
                        np.fromfile(
                            "../data/velocities/Arid_vp_LR",
                            dtype=np.float32
                        ).reshape(400,400,600),
                        (768,768,768),
                        order=3,
                        mode='reflect'
                    ),
                },
                {
                    "name": "segoverthrust",
                    "loader": lambda: resize(
                        np.fromfile(
                            "../data/velocities/overthrust",
                            dtype=np.float32
                        ).reshape(801,801,187),
                        (768,768,768),
                        order=3,
                        mode='reflect'
                    ),
                },
                {
                    "name": "seamtimelapse",
                    "loader": lambda: resize(
                        np.load("../data/velocities/vp_f16.npy"),
                        (768,768,768),
                        order=3,
                        mode='reflect'
                    ),
                },
            ]

            cube_size = args.cube_size
            
            if args.cube_size == 64:
                down_factors = [2, 5, 7, 9]
            elif args.cube_size == 128:
                down_factors = [1, 2, 3, 4]

            all_model_cubes = []

            for cfg in model_configs:
                print(f"Loading & resizing {cfg['name']} …")
                vp3d = cfg["loader"]()
                # wrap in a Dask array for lazy concatenation
                vp3d_da = da.from_array(vp3d, chunks=(256,256,256))

                model_cubes = []
                for f in down_factors:
                    stride_size = (64, 64, 64)
                    # extract overlapping cubes of size (cube_size,)*3 from the full volume
                    cubes = extract_cubes_fast(
                        vp3d,
                        (cube_size*f, cube_size*f, cube_size*f),
                        stride_size
                    )
                    # down‐sample each cube by factor f (if f==1 this is a no‑op)
                    if f > 1:
                        cubes = cubes[:, ::f, ::f, ::f]
                    # convert to Dask for memory efficiency
                    cubes_da = da.from_array(cubes, chunks=(128, cube_size, cube_size, cube_size))
                    model_cubes.append(cubes_da)

                # concatenate all scales for this model
                model_allscales = da.concatenate(model_cubes, axis=0)

                # normalize to [0,1] per‐model
                # apply it to every chunk (chunking should have first dim <= full cubes)
                normed = model_allscales.map_blocks(
                    normalize_per_cube,
                    dtype=model_allscales.dtype,
                )

                all_model_cubes.append(normed)
                print(f" — extracted {normed.shape[0]} cubes from {cfg['name']}")

            # finally, concatenate *all* models into one big training set
            training_set = da.concatenate(all_model_cubes, axis=0)
            
            # flip each cube along its last axis (depth) and append to the dataset
            training_set_flipped_depth = training_set[:, :, :, ::-1]
            training_set_flipped_inline = training_set[:, :, ::-1, :]
        
            # add the augmented data    
            training_set = da.concatenate([training_set, training_set_flipped_depth, training_set_flipped_inline], axis=0)
            
            print(f"Total cubes in dataset: {training_set.shape[0]}")

            # persist to disk as NumPy (or use .to_zarr for out‑of‑core)
            np.save(data_path, training_set.compute())
            print(f"Saved dataset to {data_path}")
    
    ##################################################################
    # Diffusion model initialization and training
    ##################################################################
    
    model = Unet3D(
        dim = args.unet_size,
        dim_mults = (1, 2, 4, 8, 16),
        channels=1
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size = args.cube_size,
        num_frames = args.cube_size,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1'
    ).cuda()

    trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size = args.batch_size,
        train_lr = 1e-5,
        train_num_steps = 200000,              
        gradient_accumulate_every = 2,   
        ema_decay = 0.995,               
        amp = True,                     
        save_and_sample_every = 10000,
        num_sample_rows = 1,
        results_folder = results_folder,
        cube_size = args.cube_size,
    )

    trainer.train()