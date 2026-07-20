"""
Description:
Train 2D diffusion model.
 
Run as: 
python DiffusionModel_2D_Training.py

For multi-GPU:
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu DiffusionModel_2D_Training.py

Contributors:
Originally adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

# pip3 install git+https://github.com/lucidrains/video-diffusion-pytorch.git

import torch
import math
import numpy as np
import os
import glob
import json
import time
import inspect

from skimage.transform import resize
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import cpu_count
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torchvision import transforms as T, utils
from argparse import ArgumentParser

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, exists, has_int_squareroot, cycle, divisible_by, num_to_groups
from denoising_diffusion_pytorch.version import __version__

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

from guidedfwi.diffusion2d import Trainer
from guidedfwi.utils import log_experiment
from guidedfwi.openfwi import build_training_velocity_tensor, TRAIN_FILES
        
def main():
    
    parser = ArgumentParser()
    
    parser.add_argument(
    "--training_data",
    type=str,
    default='seg',
    )
    parser.add_argument(
    "--results_folder",
    type=str,
    default=None,
    )
    parser.add_argument(
    "--load_model",
    type=int,
    default=None,
    )
    parser.add_argument(
    "--unet_dim",
    type=int,
    default=256,
    )
    parser.add_argument(
    "--input_dim",
    type=int,
    default=256,
    )
    parser.add_argument(
    "--unet_objective",
    type=str,
    default='pred_v',
    )
    parser.add_argument(
    "--openfwi_root",
    type=str,
    default='../data/FlatVel_A',
    help="Root folder of the OpenFWI FlatVel-A dataset (contains 'model/' and 'data/'). Only used when --training_data=flatvel_a.",
    )
    parser.add_argument(
    "--train_num_steps",
    type=int,
    default=700000,
    help="Total optimizer steps to train for. Checkpoints/sample grids are still written every save_and_sample_every (5000) steps regardless of this total, so you can stop early and load any milestone.",
    )

    ##################################################################
    # Experiment logging
    ##################################################################

    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = (
        '../results/DiffusionModel_2D_Training_'+str(run_id)
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
    # Diffusion model loading
    ##################################################################

    model = Unet(
        dim = args.unet_dim,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.input_dim,
        timesteps = 1000,
        objective = args.unet_objective
    ).cuda()

    if args.training_data == 'seg' or args.training_data == 'combined':
        vp_seg = np.load('/home/taufikmh/Datasets/3D/combined_patches/vp_indiv_normed.npy', mmap_mode="r").reshape(-1,1,256,256)
        vs_seg = np.load('/home/taufikmh/Datasets/3D/combined_patches/vs_indiv_normed.npy', mmap_mode="r").reshape(-1,1,256,256)
        rho_seg = np.load('/home/taufikmh/Datasets/3D/combined_patches/rho_indiv_normed.npy', mmap_mode="r").reshape(-1,1,256,256)

    if args.training_data == 'openfwi' or args.training_data == 'combined':
        parent_dir = "/home/taufikmh/Datasets/openfwi/"
        X = 3
        selected_files = []

        top_folders = [
            os.path.join(parent_dir, d)
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d)) and not d.endswith("_A")
        ]

        for folder in top_folders:
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.isdir(subfolder_path):
                    npy_files = sorted(glob.glob(os.path.join(subfolder_path, "*.npy")))
                    selected_files.extend(npy_files[:X])

        arrays = []
        for f in selected_files:
            data = np.load(f)[:, 0]
            resized = resize(data, (500, 256, 256), order=1, preserve_range=True, anti_aliasing=True)
            arrays.append(resized)

        vp_openfwi = np.stack(arrays, axis=0).reshape(-1, 1, 256, 256)
        vs_openfwi = vp_openfwi / np.sqrt(3)
        rho_openfwi = 0.31 * np.power(vp_openfwi * 1e3, 0.25)
        
    if args.training_data == 'random' or args.training_data == 'combined':
        vp_random = np.load('../data/random_layers_vp_1ksamples_in_meters.npy', mmap_mode="r")
        vs_random = vp_random / np.sqrt(3)
        rho_random = 0.31 * np.power(vp_random * 1e3, 0.25)

    if args.training_data == 'flatvel_a':
        # OpenFWI FlatVel-A replaces the proprietary SEG/in-house volumes used by
        # the 'seg' branch above. Only the 55 training files (model1..model55)
        # are read here -- the 5 held-out test files and the seismic gathers
        # (data*.npy) are never touched during prior training.
        vp_flatvel_native = build_training_velocity_tensor(args.openfwi_root, TRAIN_FILES).numpy()
        n_flatvel = vp_flatvel_native.shape[0]

        # Native FlatVel-A resolution is 70x70. The Unet/GaussianDiffusion below
        # operate at a fixed 256x256 resolution (so that the resulting prior
        # stays loadable by DiffusionModel_2D_FWIGuidedSamplingSynthetic.py),
        # so we resize exactly like the existing 'openfwi' branch above already
        # does for its own (also non-256-native) source volumes.
        vp_flatvel = resize(
            vp_flatvel_native.reshape(n_flatvel, 70, 70), (n_flatvel, 256, 256),
            order=1, preserve_range=True, anti_aliasing=True
        ).reshape(n_flatvel, 1, 256, 256)

        # FlatVel-A ships vp only (already in m/s, ~1500-4500). vs/rho are
        # derived the same way DiffusionModel_2D_FWIGuidedSamplingSynthetic.py
        # already derives them for other vp-only velocity types (Gardner's
        # relation, no unit conversion needed since vp is already in m/s).
        vs_flatvel = vp_flatvel / np.sqrt(2)
        rho_flatvel = 0.31 * vp_flatvel ** 0.25

    # Combine data
    if args.training_data == 'seg':
        vp = vp_seg
        vs = vs_seg
        rho = rho_seg
    elif args.training_data == 'openfwi':
        vp = vp_openfwi
        vs = vs_openfwi
        rho = rho_openfwi
    elif args.training_data == 'random':
        vp = vp_random
        vs = vs_random
        rho = rho_random
    elif args.training_data == 'flatvel_a':
        vp = vp_flatvel
        vs = vs_flatvel
        rho = rho_flatvel
    elif args.training_data == 'combined':
        vp = np.concatenate((vp_seg, vp_openfwi, vp_random), axis=0)
        vs = np.concatenate((vs_seg, vs_openfwi, vs_random), axis=0)
        rho = np.concatenate((rho_seg, rho_openfwi, rho_random), axis=0)

    # Concatenate vp vs and rho
    training_images = np.concatenate((vp,vs,rho), axis=1)

    # Compute min and max along spatial dimensions only (keep sample and channel dimensions)
    min_vals = training_images.min(axis=(2, 3), keepdims=True)  # shape (N, 3, 1, 1)
    max_vals = training_images.max(axis=(2, 3), keepdims=True)  # shape (N, 3, 1, 1)

    # Normalize each channel independently for each sample
    training_images = (training_images - min_vals) / (max_vals - min_vals + 1e-8)
    
    if args.input_dim == 128:
        training_images = training_images[:, :, ::2, ::2]

    # Convert to torch tensor (Trainer wraps this in a TensorDataset, which
    # requires an actual torch.Tensor; previously this conversion only
    # happened for the input_dim==128 branch, leaving a plain numpy array
    # otherwise -- fixed here since it applies regardless of --training_data).
    training_images = torch.from_numpy(training_images).float()

    trainer = Trainer(
        diffusion,
        training_images,
        train_batch_size = 1,
        train_lr = 2e-6, # 1e-5
        save_and_sample_every = 5000,
        num_samples = 16,
        save_best_and_latest_only = False,
        results_folder=results_folder if args.results_folder is None else args.results_folder,
        train_num_steps = args.train_num_steps,  # total training steps
        gradient_accumulate_every = 16,   # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False              # whether to calculate fid during training
    )

    print("Accelerator device:", trainer.accelerator.device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())

    if args.load_model is not None:
        trainer.load(args.load_model)
        
    ##################################################################
    # Training and inference
    ##################################################################
    
    trainer.train()

    sampled_images = diffusion.sample(batch_size = 4)
    sampled_images.shape # (4, 3, 128, 128)
    
if __name__ == "__main__":
    main()