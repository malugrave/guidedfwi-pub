"""
Description:
Migration-guided 2D diffusion model sampling.
 
Run as: 
python DiffusionModel_2D_MigrationGuidedSamplingSynthetic.py

For multi-GPU:
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu DiffusionModel_2D_MigrationGuidedSamplingSynthetic.py

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
import time
import json

from skimage.transform import resize
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

from guidedfwi.diffusion2d import Trainer, normalize_to_zero_to_one, unnormalize_from_zero_to_one
from guidedfwi.plots import plot_modulus
from guidedfwi.utils import extract_squares, combine_squares, numpy_to_cuda, cuda_to_numpy, set_seed, log_experiment

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

def migration_guided_reconstruction(vp, dvp_migration, vp0, weight, debug=False, masks=None, vpw=None, vp_range=(1.5, 4.5)):  
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
    debug : :obj:`bool`, optional
        Option to plot the gradient for debugging
    masks : :obj:`torch.tensor`, optional
        Masks for the well locations
    vpw : :obj:`torch.tensor`
        Normalized wireline (wells) P-velocity
    vp_range : :obj:`tuple`
        Minimum and maximum P-wave velocity in km/s
        
    Returns
    -------
    grad : :obj:`torch.tensor`
        Gradient computed by taking an L2 norm between the diffusion sample and the migration image
    """
    
    with torch.enable_grad():
        
        # Transform from -1to1 to 1to2 value range images
        dvp_diffusion = (((unnormalize_from_zero_to_one((vp+1)*0.5, vp_range[0], vp_range[1]))**(-2) - (vp0)**(-2)).clone().detach())
        dvp_diffusion.requires_grad_(True)  # Ensure this is true for your input tensors
        
        if masks is None:
            rec_loss = F.mse_loss(dvp_diffusion, dvp_migration, reduction='none').sum()
        else:
            img_loss = F.mse_loss(dvp_diffusion, dvp_migration, reduction='none').sum()
            vp_t = (dvp_diffusion + (vp0)**(-2))**(-0.5)
            well_loss = F.mse_loss(masks*vp_t, masks*vpw, reduction='none').sum()
            
            rec_loss = img_loss + well_loss
        
        grad = torch.autograd.grad(rec_loss, dvp_diffusion)[0]
        if debug:
            plot_modulus(dvp_diffusion[0,0].detach().cpu().numpy(), aspect='auto', cmap='gray')
            plot_modulus(dvp_migration[0,0].detach().cpu().numpy(), aspect='auto', cmap='gray')

        return grad * weight

def p_sample_migration(trainer, x, dvp_migration, vp0, t, classes, cond_scale=1., clip_denoised=True, debug=False, masks=None, vpw=None, vp_range=(1.5, 4.5), weight=1, noise_weight=0.5):
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
    masks : :obj:`torch.tensor`, optional
        Masks for the well locations
    vpw : :obj:`torch.tensor`
        Normalized wireline (wells) P-velocity
    vp_range : :obj:`tuple`
        Minimum and maximum P-wave velocity in km/s
    weight : :obj:`float`
        Weighting scalar for the computed gradient, 
        the larger the more influence coming from the migration image
    noise_weight : :obj:`float`
        Weighting scalar for the diffusion update, 
        the larger the more influence coming from the diffusion
        
    Returns
    -------
    img: :obj:`torch.tensor`
        Normalized diffusion sample (-1 to 1)
    x_start: :obj:`torch.tensor`
        Predicted x_start for the diffusion process
    """
    batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
    model_mean, model_variance, model_log_variance, x_start = trainer.ema.ema_model.p_mean_variance(
        x=x, t=batched_times, classes=classes, cond_scale=cond_scale, clip_denoised=clip_denoised
    )
    noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    
    # Compute gradient
    cond_grad = migration_guided_reconstruction(
        vp=x, dvp_migration=dvp_migration, vp0=vp0, 
        weight=model_variance*weight, debug=debug, masks=masks, vpw=vpw, vp_range=vp_range
    )
    model_mean += cond_grad
    
    # Add weighted noise to the diffusion sample
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise * noise_weight
    return pred_img.clamp_(-1., 1.), x_start

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
    default=0.5,
    )
    parser.add_argument(
    "--square_size",
    type=int,
    default=256,
    )
    parser.add_argument(
    "--stride",
    type=int,
    default=256,
    )
    parser.add_argument(
    "--nx",
    type=int,
    default=1024,
    )
    parser.add_argument(
    "--nz",
    type=int,
    default=512,
    )
    parser.add_argument(
    "--velocity_name",
    type=str,
    default='Compass',
    )

    args = parser.parse_args()
    
    # Appendix for figures based on diffusion parameters
    difpar = (
        str(args.start_timestep)+'starttime_'
        +str(args.gradient_weight)+'weight_'+str(args.noise_weight)+'noise_'
        +str(args.stride)+'stride_'+str(args.square_size)+'square.png'
    )
            
    ##################################################################
    # Experiment logging
    ##################################################################
    
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = (
        '../results/DiffusionModel_2D_MigrationGuidedSamplingSynthetic_'+str(run_id)
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
    # Input Data Loading
    ##################################################################

    # LSRTM image
    lsrtm_image = np.load('../results/SEAMNoSalt_LSRTM_Acoustic2DIso_39sou_875rec_8freq_0.02dx_0.007dz/Image_19.npy')

    # Load the true model
    vp_true = np.load('../data/seam_salt.npy')/1e3
    vp_wrong = np.load('../data/seam_nosalt.npy')/1e3

    m_vmin, m_vmax = np.percentile(vp_true, (1, 99))
    _, i_vmax = np.percentile(lsrtm_image, (0.5, 99.5))

    mask = np.ones_like(vp_true)
    mask[vp_true<1.51] = 0

    # Initial model for FWI by smoothing the true model
    vp_init = gaussian_filter(vp_wrong, sigma=[25,25])

    # Plot the relevant data and image
    plot_modulus(
        vp_true.T,  aspect='auto', interpolation='bicubic', cmap='rainbow', 
        vmin=m_vmin,vmax=m_vmax,
        fig_name=os.path.join(results_folder, 'VpTrue.png')
    )

    plot_modulus(
        vp_init.T,  aspect='auto', interpolation='bicubic', cmap='rainbow', 
        vmin=m_vmin,vmax=m_vmax,
        fig_name=os.path.join(results_folder, 'VpInit.png')
    )

    plot_modulus(
        lsrtm_image.T,  aspect='auto', interpolation='bicubic', cmap='gray', 
        vmin=-i_vmax,vmax=i_vmax,
        fig_name=os.path.join(results_folder, 'LSRTM.png')
    )
    
    # Patching properties
    square_size = (args.square_size, args.square_size)
    stride = (args.stride, args.stride)

    # Resample for debugging purposes only
    vp_true = resize(vp_true, (args.nx, args.nz)).T
    vp_init = resize(vp_init, (args.nx, args.nz)).T
    lsrtm_image = resize(lsrtm_image, (args.nx, args.nz)).T

    # Spatial dimensions
    nx, nz = vp_true.shape[1],vp_true.shape[0]

    # Apply normalization
    vp_init_cuda = torch.from_numpy(vp_init).view(1,1,nz,nx).repeat(1,3,1,1).float().cuda()
    vp_init_normed = 2*normalize_to_zero_to_one(vp_init_cuda, vp_true.min(), vp_true.max())-1

    # Custom p_sample_loop
    classes = torch.tensor([0]).cuda()                                                       

    # Start from reshaped initial velocity model
    diffusion_image = vp_init_normed
    lsrtm_patches = extract_squares(lsrtm_image, square_size, stride)
    vp0_patches = extract_squares(vp_init, square_size, stride)

    # Debug patching process
    lsrtm_recon = combine_squares(lsrtm_patches, vp_true.shape, square_size, stride)
    plot_modulus(
        lsrtm_recon-lsrtm_image,
        aspect='auto', interpolation='bicubic', cmap='gray', 
        vmin=-1e-12,vmax=1e-12,
        fig_name=os.path.join(results_folder, 'LSRTMReconError_'+difpar)
    )
    vp0_recon = combine_squares(vp0_patches, vp_true.shape, square_size, stride)
    plot_modulus(
        vp0_recon-vp_init,
        aspect='auto', interpolation='bicubic', cmap='rainbow', 
        # vmin=1e-8*m_vmin,vmax=1e-8*m_vmax
        vmin=-1e-12,vmax=1e-12,
        fig_name=os.path.join(results_folder, 'VpReconError_'+difpar)
    )
    
    ##################################################################
    # Diffusion model initialization and training
    ##################################################################

    # The model was trained using 7 different classes as labels but they are not used for sampling
    set_seed(12315019)
    num_classes = 7
    classes = ['salt', 'volve', 'crust', 'arid', 'tigris', 'prism', 'over']
    train_x, train_y = torch.rand([311, 3, 128, 128]), torch.rand([311])

    print('Creating Unet...')

    model = Unet(
        dim = 256,
        dim_mults = (1, 2, 4, 8, 16),
        num_classes = num_classes,
        cond_drop_prob = 0.5,
        resnet_block_groups = 8,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 256,
        timesteps = 1000
    ).cuda()

    trainer = Trainer(
        train_x, train_y, diffusion,
        train_batch_size = 1,
        train_lr = 5e-6,
        train_num_steps = 200000,              
        gradient_accumulate_every = 2,   
        ema_decay = 0.995,               
        amp = False,                     
        save_and_sample_every = 1000,
        results_folder = '../results/DiffusionModel_2D_Combined_256',
        num_samples = 25,
        classes=classes
    )

    # Load model
    trainer.load(200)
    
    ##################################################################
    # Migration-guided Conditional Sampling
    ##################################################################

    start_timestep = args.start_timestep
    timesteps = 1000
    repetitions = 1
    debug_i = False # If True, no debugging
    gradient_weight = args.gradient_weight

    for _ in tqdm(range(repetitions)):

        for t in tqdm(reversed(range(0, timesteps-start_timestep)), desc = 'Sampling loop time step', total = (timesteps-start_timestep)):
            
            diffusion_patches = extract_squares(diffusion_image[0,0].detach().cpu().numpy(), square_size, stride)

            diffusion_image_patches = []
            
            for i in range(len(diffusion_patches)):

                diffusion_image_i, _ = p_sample_migration(trainer,
                    numpy_to_cuda(diffusion_patches[i]).view(1,1,256,256).repeat(1,3,1,1).float(), 
                    numpy_to_cuda(lsrtm_patches[i]).view(1,1,256,256).repeat(1,3,1,1).float(), 
                    numpy_to_cuda(vp0_patches[i]).view(1,1,256,256).repeat(1,3,1,1).float(), 
                    t, classes, vp_range=(vp_true.min(), vp_true.max()), 
                    cond_scale=1, weight=gradient_weight, debug=debug_i, noise_weight=args.noise_weight)
                
                diffusion_image_patches.append(cuda_to_numpy(diffusion_image_i[0,0]))

                if debug_i == True:
                    if i % (len(diffusion_patches)*24) == 0:
                        debug_i=True
                    else:
                        debug_i=False
                
            diffusion_image = numpy_to_cuda(combine_squares(diffusion_image_patches, vp_init_normed[0,0].shape, square_size, stride)).float().view(1,1,nz,nx)

    plot_modulus(
        unnormalize_from_zero_to_one(0.5*(1+diffusion_image[0,0].detach().cpu().numpy()), vp_true.min(), vp_true.max()), 
        aspect='auto', cmap='rainbow',
        fig_name=os.path.join(results_folder, 'Diffusion_'+difpar)
    )

if __name__ == "__main__":
    main()