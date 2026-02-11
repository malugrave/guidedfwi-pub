"""
Description:
Migration-guided 3D diffusion model sampling.
 
Run as: 
python DiffusionModel_3D_MigrationGuidedSamplingSynthetic.py

For multi-GPU:
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu DiffusionModel_3D_MigrationGuidedSamplingSynthetic.py

Contributors:
Originally adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

import math
import numpy as np
import torch
import random
import h5py
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import os
import segyio
import time
import json

from skimage.transform import resize
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

from guidedfwi.diffusion2d import normalize_to_zero_to_one, unnormalize_from_zero_to_one
from guidedfwi.diffusion3d import Trainer
from guidedfwi.plots import plot_slices
from guidedfwi.utils import extract_cubes, combine_cubes, numpy_to_cuda, cuda_to_numpy, set_seed, log_experiment

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

def migration_guided_reconstruction(vp, dvp_migration, vp0, weight, imaging=False, vp_range=(1.5, 4.5)):
    """
    Compute gradient between the diffusion-derived perturbation and measured migration perturbation

    Parameters
    ----------
    vp : :obj:`torch.tensor`
        Normalized P-velocity (-1 to 1)
    dvp_migration : :obj:`torch.tensor`
        P-velocity perturbation
    vp0 : :obj:`torch.tensor`
        Migration P-velocity in km/s
    weight : :obj:`float`
        Weighting scalar for the computed gradient
    imaging : :obj:`bool`, optional
        Option to use the image instead of perturbation
    vp_range : :obj:`tuple`
        Minimum and maximum P-wave velocity in km/s
        
    Returns
    -------
    grad : :obj:`torch.tensor`
        Gradient computed by taking an L2 norm between the diffusion sample and the migration image
    """
    
    with torch.enable_grad():
        
        # Transform from 0-1 to 1-2 value range images
        # dvp_diffusion = (((vp+1)**(-2) - vp0**(-2)).clone().detach())
        dvp_diffusion = (((unnormalize_from_zero_to_one((vp+1)*0.5, vp_range[0], vp_range[1]))**(-2) - (vp0)**(-2)).clone().detach())
        dvp_diffusion.requires_grad_(True)  # Ensure this is true for your input tensors
        
        if imaging:
            rec_loss = F.mse_loss(torch.diff(dvp_diffusion, dim=-1), torch.diff(dvp_migration, dim=-1), reduction='none').sum()
        else:
            rec_loss = F.mse_loss(dvp_diffusion, dvp_migration, reduction='none').sum()

        return torch.autograd.grad(rec_loss, dvp_diffusion)[0] * weight

def p_sample_migration(trainer, x, dvp_migration, vp0, t, cond=None, cond_scale=1., clip_denoised=True, debug=False, imaging=False, weight=1, noise_weight=1, vp_range=(1.5, 4.5)):
    """
    Migration-guided reverse diffusion sampling

    Parameters
    ----------
    x : :obj:`torch.tensor`
        Normalized P-velocity (-1 to 1)
    dvp_migration : :obj:`torch.tensor`
        P-velocity perturbation
    vp0 : :obj:`torch.tensor`
        Migration P-velocity in km/s
    t : :obj:`torch.tensor`
        Time vector
    classes : :obj:`torch.tensor`
        Velocity class index
    debug : :obj:`bool`, optional
        Option to plot the gradient for debugging
    imaging : :obj:`bool`, optional
        Option to use the image instead of perturbation
    weight : :obj:`float`
        Weighting scalar for the computed gradient, 
        the larger the more influence coming from the migration image
    noise_weight : :obj:`float`
        Weighting scalar for the diffusion update, 
        the larger the more influence coming from the diffusion
    vp_range : :obj:`tuple`
        Minimum and maximum P-wave velocity in km/s
        
    Returns
    -------
    """
    
    b, *_, device = *x.shape, x.device
    model_mean, model_variance, model_log_variance = trainer.model.p_mean_variance(
        x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale
    )
    
    with torch.no_grad():
        cond_grad = migration_guided_reconstruction(
            vp=x, dvp_migration=dvp_migration, vp0=vp0, weight=model_variance*weight, imaging=imaging, vp_range=vp_range
        )
        
        model_mean = model_mean + cond_grad
            
    noise = torch.randn_like(x)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

    torch.cuda.empty_cache()
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * noise_weight, model_mean


def main():
    
    parser = ArgumentParser()
    
    parser.add_argument(
    "--start_timestep",
    type=int,
    default=0,
    )
    parser.add_argument(
    "--gradient_weight",
    type=int,
    default=1,
    )
    parser.add_argument(
    "--noise_weight",
    type=float,
    default=0.05,
    )
    parser.add_argument(
    "--cube_size",
    type=int,
    default=64,
    )
    parser.add_argument(
    "--stride",
    type=int,
    default=64,
    )
    parser.add_argument(
    "--nx",
    type=int,
    default=64,
    )
    parser.add_argument(
    "--ny",
    type=int,
    default=64,
    )
    parser.add_argument(
    "--nz",
    type=int,
    default=64,
    )
    parser.add_argument(
    "--velocity_name",
    type=str,
    default='Compass',
    )

    ##################################################################
    # Experiment logging
    ##################################################################
    
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = (
        '../results/DiffusionModel_3D_MigrationGuidedSamplingSynthetic_'+str(run_id)
    )
    
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        
    # Log experiment
    log_experiment(
        results_folder, script_path=os.path.abspath(inspect.getfile(inspect.currentframe())), run_id=run_id
    )
    with open(os.path.join(results_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Appendix for figures based on diffusion parameters
    difpar = (
        str(args.start_timestep)+'starttime_'
        +str(args.gradient_weight)+'weight_'+str(args.noise_weight)+'noise_'
        +str(args.stride)+'stride_'+str(args.cube_size)+'square.png'
    )
        
    ##################################################################
    # Input Data Loading
    ##################################################################

    # LSRTM image
    lsrtm_cube = np.load('/home/taufikmh/KAUST/summer_2024/uqdiffefwi-dev/scripts/experiments/TigrisLSRTM3D_Custom_64sou_4096rec_25freq_20dx_20dy_10dz/Image_19.npy')

    # Load the true model
    data_dir = '/home/taufikmh/Datasets/3D/01_C_tigris/download/'
    vel_tigris_3d = resize(segyio.tools.cube(data_dir+'Vel.2018_TIGRIS_StagSeis_CGG_finalmodel_P.p1.segy')[10:,10:850,::4], (256, 512, 1024))/1e3
    vp_true = 0.75*(1+vel_tigris_3d[:,:,:768][::4, ::8, ::12])

    m_vmin, m_vmax = np.percentile(vp_true, (1, 99))
    _, i_vmax = np.percentile(lsrtm_cube, (0.5, 99.5))

    mask = np.ones_like(vp_true)
    mask[vp_true<1.51] = 0

    # Initial model for FWI by smoothing the true model
    vp_init = gaussian_filter(vp_true, sigma=[3, 3, 3])

    # Plot the relevant data and image
    plot_slices(
        vp_true.T,  aspect='auto', interpolation='bicubic', cmap='rainbow', 
        vmin=m_vmin,vmax=m_vmax, 
        fig_name=os.path.join(results_folder, 'VpTrue.png'), size=8
    )

    plot_slices(
        vp_init.T,  aspect='auto', interpolation='bicubic', cmap='rainbow', 
        vmin=m_vmin,vmax=m_vmax,
        fig_name=os.path.join(results_folder, 'VpInit.png'), size=8
    )

    plot_slices(
        lsrtm_cube.T,  aspect='auto', interpolation='bicubic', cmap='gray', 
        vmin=-i_vmax,vmax=i_vmax,
        fig_name=os.path.join(results_folder, 'LSRTM.png'), size=8
    )
    
    # Patching properties
    cube_size = (args.cube_size, args.cube_size, args.cube_size)
    stride = (args.stride, args.stride, args.stride)

    # Resample for debugging purposes only
    vp_true = resize(vp_true, (args.nx, args.ny, args.nz)).T
    vp_init = resize(vp_init, (args.nx, args.ny, args.nz)).T
    lsrtm_cube = resize(lsrtm_cube, (args.nx, args.ny, args.nz)).T

    # Spatial dimensions
    nx, ny, nz = vp_true.shape

    # Apply normalization
    vp_init_cuda = torch.from_numpy(vp_init).view(1,nx,ny,nz).float().cuda()
    vp_init_normed = 2*normalize_to_zero_to_one(vp_init_cuda, vp_true.min(), vp_true.max())-1

    # Start from reshaped initial velocity model
    diffusion_cube = vp_init_normed
    lsrtm_cubes = extract_cubes(lsrtm_cube, cube_size, stride)
    vp0_cubes = extract_cubes(vp_init, cube_size, stride)

    # Debug patching process
    lsrtm_recon = combine_cubes(lsrtm_cubes, vp_true.shape, cube_size, stride)
    plot_slices(
        lsrtm_recon-lsrtm_cube,
        aspect='auto', interpolation='bicubic', cmap='gray', 
        vmin=-1e-12,vmax=1e-12,
        fig_name=os.path.join(results_folder, 'LSRTMReconError_'+difpar), size=8
    )
    vp0_recon = combine_cubes(vp0_cubes, vp_true.shape, cube_size, stride)
    plot_slices(
        vp0_recon-vp_init,
        aspect='auto', interpolation='bicubic', cmap='rainbow', 
        # vmin=1e-8*m_vmin,vmax=1e-8*m_vmax
        vmin=-1e-12,vmax=1e-12,
        fig_name=os.path.join(results_folder, 'VpReconError_'+difpar), size=8
    )

    ##################################################################
    # Diffusion model initialization and training
    ##################################################################

    # The model was trained using 7 different classes as labels but they are not used for sampling
    set_seed(12315019)

    print('Creating Unet...')

    model = Unet3D(
        dim=128,
        dim_mults=(1, 2, 4, 8, 16),
        channels=1
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        num_frames=64,
        channels=1,
        timesteps=1000,   # number of steps
        loss_type='l1'    # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        '/home/taufikmh/Datasets/3D/combined_cubes/vp_combined_0plus1_070924.npy',
        train_batch_size=2,
        train_lr=8e-6,
        train_num_steps=200000,              
        gradient_accumulate_every=2,   
        ema_decay=0.995,               
        amp=False,                     
        save_and_sample_every=2000,
        num_sample_rows=2,
        results_folder='/home/taufikmh/KAUST/summer_2024/uqdiffefwi-dev/scripts/video-diffusion'
    )

    # Load model
    trainer.load(26)
    
    print('Loaded...')
    
    ##################################################################
    # Migration-guided Conditional Sampling
    ##################################################################

    start_timestep = args.start_timestep
    timesteps = 1000
    repetitions = 1
    debug_i = False # If True, no debugging
    gradient_weight = args.gradient_weight

    diffusion_samples = []
    diffusion_means = []

    diffusion_i = vp_init_normed[0].clone()

    for _ in range(repetitions):

        for i in tqdm(reversed(range(0, timesteps-start_timestep)), total=timesteps-start_timestep):
            
            diffusion_cubes = extract_cubes(cuda_to_numpy(diffusion_i), cube_size, stride)
            
            small_cubes = []
            small_means = []
            
            for j in range(len(lsrtm_cubes)):
                
                diffusion_j = numpy_to_cuda(diffusion_cubes[j]).unsqueeze(0).unsqueeze(0)
                lsrtm_j = numpy_to_cuda(lsrtm_cubes[j]).unsqueeze(0).unsqueeze(0)
                vp0_j = numpy_to_cuda(vp0_cubes[j]).unsqueeze(0).unsqueeze(0)
                
                cube_j, mean_j = p_sample_migration(
                    trainer, x=diffusion_j, t=torch.full((1,), i, device='cuda', dtype=torch.long), weight=gradient_weight,
                    dvp_migration=lsrtm_j, vp0=vp0_j, noise_weight=args.noise_weight, vp_range=(vp_true.min(), vp_true.max())
                )
                    
                small_cubes.append(cuda_to_numpy(cube_j.squeeze(0).squeeze(0)))
                small_means.append(cuda_to_numpy(mean_j.squeeze(0).squeeze(0)))
                
            diffusion_i = numpy_to_cuda(combine_cubes(small_cubes, lsrtm_cube.shape, cube_size, stride)).float()
            means_i = numpy_to_cuda(combine_cubes(small_means, lsrtm_cube.shape, cube_size, stride)).float()
            
            diffusion_samples.append(cuda_to_numpy(diffusion_i))
            diffusion_means.append(cuda_to_numpy(means_i))
    
    plot_slices(
        unnormalize_from_zero_to_one(0.5*(1+diffusion_samples[-1]), vp_true.min(), vp_true.max()), 
        aspect='auto', cmap='rainbow', vmin=m_vmin,vmax=m_vmax,
        fig_name=os.path.join(results_folder, 'Diffusion_'+difpar), size=8, interpolation='bicubic'
    )

if __name__ == "__main__":
    main()