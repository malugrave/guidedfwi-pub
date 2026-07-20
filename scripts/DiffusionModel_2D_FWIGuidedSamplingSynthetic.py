"""
Description:
FWI-guided 2D diffusion model sampling.
 
Run as: 
python DiffusionModel_2D_FWIGuidedSamplingSynthetic.py

For multi-GPU:
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu DiffusionModel_2D_FWIGuidedSamplingSynthetic.py

Contributors:
Originally adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

# pip3 install git+https://github.com/lucidrains/denoising-diffusion-pytorch.git

import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import deepwave
import inspect
import matplotlib.pyplot as plt
import json
import time

from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from tqdm.auto import tqdm
from ema_pytorch import EMA
from argparse import ArgumentParser

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, divisible_by
from denoising_diffusion_pytorch.version import __version__

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

from guidedfwi.diffusion2d import Trainer
from guidedfwi.plots import plot_modulus, plot_diffusion_evolution, plot_flipped_data
from guidedfwi.utils import log_experiment, normalize_to_minusone_and_one, denormalize_from_minusone_and_one
from guidedfwi.openfwi import load_test_sample

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

def extract_squares_with_gaussian_weights(array, square_size, stride, device='cpu', sigma=1.):
    """
    Extract overlapping square patches and associate Gaussian weights per patch.

    Parameters
    -----------
        array (Tensor): Input tensor of shape (B, C, H, W)
        square_size (tuple): Patch size (dx, dy)
        stride (tuple): Stride (sx, sy)
        sigma (float): Gaussian width

    Returns
    --------
        patches (Tensor): Shape (B, N, C, dx, dy)
        weights (Tensor): Shape (1, N, 1, dx, dy) — same weights across batch and channel
        padded_shape (tuple): (H_pad, W_pad)
        grid_shape (tuple): (nH, nW)
    """
    B, C, H, W = array.shape
    dx, dy = square_size
    sx, sy = stride

    pad_H = (0, (dx - H % dx) % dx)
    pad_W = (0, (dy - W % dy) % dy)

    padded = F.pad(array, (0, pad_W[1], 0, pad_H[1]), mode='replicate')
    H_pad, W_pad = padded.shape[-2:]

    patches = padded.unfold(2, dx, sx).unfold(3, dy, sy)  # (B, C, nH, nW, dx, dy)
    nH, nW = patches.shape[2], patches.shape[3]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, nH * nW, C, dx, dy)

    # Create 2D Gaussian weight patch
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, dx), torch.linspace(-1, 1, dy), indexing='ij')
    sigma = sigma
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.max()  # Normalize to [0, 1]
    gaussian = gaussian.to(device)

    # Broadcast to match patch shape: (1, N, 1, dx, dy)
    weights = gaussian.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, nH * nW, 1, 1, 1)

    return patches, weights, (H_pad, W_pad), (nH, nW)

def combine_squares_with_gaussian_weights(patches, weights, original_shape, stride, grid_shape):
    """
    Combine overlapping patches using Gaussian weights.

    Parameters
    -----------
        patches (Tensor): (B, N, C, dx, dy)
        weights (Tensor): (1, N, 1, dx, dy)
        original_shape (tuple): (H, W)
        stride (tuple): (sx, sy)
        grid_shape (tuple): (nH, nW)

    Returns
    --------
        combined (Tensor): (B, C, H, W)
    """
    B, N, C, dx, dy = patches.shape
    sx, sy = stride
    H, W = original_shape
    nH, nW = grid_shape

    padded_H = H + (dx - H % dx) % dx
    padded_W = W + (dy - W % dy) % dy
    output_size = (padded_H, padded_W)

    patches = patches * weights  # Apply Gaussian weights
    patches = patches.view(B, nH, nW, C, dx, dy).permute(0, 3, 4, 5, 1, 2)
    patches = patches.reshape(B * C, dx * dy, nH * nW)

    weight_maps = weights.expand(B, N, C, dx, dy).view(B, nH, nW, C, dx, dy).permute(0, 3, 4, 5, 1, 2)
    weight_maps = weight_maps.reshape(B * C, dx * dy, nH * nW)

    fold = torch.nn.Fold(output_size=output_size, kernel_size=(dx, dy), stride=(sx, sy))

    combined = fold(patches)
    weight_sum = fold(weight_maps)

    weight_sum[weight_sum == 0] = 1
    combined = combined / weight_sum

    combined = combined.view(B, C, padded_H, padded_W)[..., :H, :W]

    return combined

def normalize_patches_individually(patches):
    """
    Normalize each patch to [-1, 1] individually.

    Parameters
    -----------
        patches (Tensor): shape (B, N, C, dx, dy)

    Returns
    -------
        norm_patches: normalized patches
        min_vals: shape (B, N, C, 1, 1)
        max_vals: shape (B, N, C, 1, 1)
    """
    
    min_vals = patches.amin(dim=(-2, -1), keepdim=True)
    max_vals = patches.amax(dim=(-2, -1), keepdim=True)
    norm_patches = 2 * (patches - min_vals) / (max_vals - min_vals + 1e-16) - 1
    return norm_patches, min_vals, max_vals

def denormalize_patches(norm_patches, min_vals, max_vals):
    """
    Denormalize patches from [-1, 1] back to original scale.

    Parameters
    -----------
        norm_patches: normalized patches
        min_vals, max_vals: saved min and max for each patch

    Returns
    -------
        denorm_patches: in original scale
    """
    
    return (norm_patches + 1) * 0.5 * (max_vals - min_vals) + min_vals

def forward_propagator(
    vp,
    multi_source=True,
    total_sources=64,
    selected_sources=16,
    freq=4,
    nt=2000,
    dt=0.004,
    dz=6.25,
    dx=25,
    device=None,
    source_locations=None,
    receiver_locations=None,
    return_src_info=False,
    seed=None,
):
    """
    Forward modeling using Deepwave for either simultaneous or individual source experiments.

    Parameters
    -----------
    vp : Tensor
        Velocity model (C, H, W) or (1, C, H, W)
    multi_source : bool
        Whether to use simultaneous-source FWI.
    total_sources : int
        Total source positions available across the model.
    selected_sources : int
        Number of sources to activate per shot in simultaneous case.
    freq : float
        Frequency of source wavelet.
    nt : int
        Number of time steps.
    dt : float
        Time step (s).
    dx, dz : float
        Grid spacing in x and z.
    return_src_info : bool
        If True, return source/receiver info (for FWI loop).

    Returns
    --------
    data : Tensor
        Simulated seismic data.
    """
    if device is None:
        device = vp.device

    num_dims = 2
    nx = vp.shape[-1]
    num_shots = 1 if multi_source else total_sources
    num_sources_per_shot = selected_sources if multi_source else 1
    num_receivers_per_shot = nx

    # Generate default source/receiver positions
    if source_locations is None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x_src = dx
        y_positions = torch.linspace(0, (nx - 1) * dx, total_sources)

        if multi_source:
            sampled_idx = np.random.choice(total_sources, size=selected_sources, replace=False)
            y_srcs = y_positions[sampled_idx]
            source_locations = torch.zeros(num_shots, selected_sources, num_dims)
            source_locations[0, :, 0] = x_src
            source_locations[0, :, 1] = y_srcs
        else:
            source_locations = torch.zeros(num_shots, 1, num_dims)
            source_locations[:, 0, 0] = x_src
            source_locations[:, 0, 1] = y_positions[:num_shots]
            
        source_locations[:, :, 0] /= dz  # z-direction (usually 0 index)
        source_locations[:, :, 1] /= dx  # x-direction (usually 1 index)

    if receiver_locations is None:
        
        receiver_locations = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        receiver_locations[:, :, 0] = dx
        receiver_locations[:, :, 1] = torch.arange(num_receivers_per_shot) * dx
        
        # Divide by dx and dz to convert to grid units
        receiver_locations[:, :, 0] /= dz
        receiver_locations[:, :, 1] /= dx

    # Generate source amplitudes
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, 1 / freq).to(device)
    source_amplitudes = wavelet.repeat(num_shots, num_sources_per_shot, 1)
    
    # Run Deepwave
    data = deepwave.scalar(
        vp,
        grid_spacing=[dz, dx],
        dt=dt,
        source_amplitudes=source_amplitudes.to(device),
        source_locations=source_locations.to(device),
        receiver_locations=receiver_locations.to(device),
        accuracy=8,
        pml_freq=freq,
        pml_width=[0, 10, 10, 10],
    )[-1]

    if return_src_info:
        return data, source_amplitudes, source_locations, receiver_locations
    else:
        return data

def p_sample_loop_with_fwi_guidance(
    trainer,
    shape,
    forward_propagator=None,
    inject_every=2,
    use_fwi_guidance=False,
    self_condition=False,
    return_all_timesteps=False,
    t_start=0,
    x_smooth=None,
    fwi_only=False,
    clip_input=False,
    vmax=4500,
    vmin=1500,
    fwi_loop=1,
    run_fwi_under=1000,
    debug=False,
    freq=4,
    dx=40, dz=20,
    window_size=(256, 256),
    stride=(128, 128),
    model_input_size=(256, 256),
    normalize_patch=False,
    vp_true=None,
    total_sources=64,
    selected_sources=16
):
    """
    Sampling loop that combines reverse diffusion with FWI gradient updates.

    Parameters
    -----------
    trainer : object
        Trainer containing the EMA diffusion model.
    shape : tuple
        Shape of the sample to generate (B, C, H, W).
    inject_every : int
        Interval of timesteps to inject FWI gradients.
    use_fwi_guidance : bool
        Enable FWI-based gradient guidance.
    return_all_timesteps : bool
        Whether to return all intermediate samples.
    t_start : int
        Starting timestep for reverse diffusion.
    x_smooth : Tensor or None
        Optional smoothing initialization.
    fwi_only : bool
        If True, initialize with small noise.
    clip_input : bool
        Clamp input to [-1, 1] range.
    vmax, vmin : float
        Max/min values for velocity clipping and normalization.
    fwi_loop : int
        Number of gradient descent steps for FWI.
    run_fwi_under : int
        Only run FWI guidance if timestep < this value.
    debug : bool
        Enable debug visualizations.
    freq : int
        Frequency for PML and wavelet.
    window_size, stride, model_input_size : tuple
        Patch extraction, overlap, and model input shape.

    Returns
    --------
    output_img : Tensor
        Final or all generated images.
    fwi_loss : list
        List of FWI loss values (if applicable).
    """
    
    freq =  freq # Hz
    nt = 2000
    dt = 0.004 # seconds
    dz = dz # meters
    dx = dx # meters
    
    diffusion = trainer.ema.ema_model
    batch, device = shape[0], diffusion.device

    if x_smooth is not None:
        img = 1e-16 * torch.randn(shape, device=device) + x_smooth if fwi_only else 0.5 * torch.randn(shape, device=device) + x_smooth
    else:
        img = torch.randn(shape, device=device)
        
    imgs = [denormalize_from_minusone_and_one(img, vmin, vmax)]
    x_start = None

    fwi_loss = []

    vp_param = None                    # torch.nn.Parameter holding velocity in physical units
    optimizer = None                   # persistent Adam tracking momentum across levels
    fwi_lr = 50.0                      # keep your previous learning rate
    max_grad = None                    # set e.g. 1e3 to clip, or leave as None
    nz, nx = shape[2:]
    
    for i in tqdm(reversed(range(0, diffusion.num_timesteps - t_start)), desc='sampling loop time step', total=diffusion.num_timesteps - t_start):
        t = torch.tensor(i, device=device)
        if clip_input:
            img = img.clamp(-1, 1)

        self_cond = x_start if self_condition else None

        # Patch-based diffusion with interpolation to model_input_size
        input_img = img.detach().cpu().clone()
        # patches, padded_shape, grid_shape = extract_squares_with_gaussian_weights(img.to(torch.float64), window_size, stride)
        
        patches, weights, padded_shape, grid_shape = extract_squares_with_gaussian_weights(img.clone().detach().to(torch.float64).reshape(shape), window_size, stride)   
        
        if normalize_patch:     
            patches, min_patch_vals, max_patch_vals = normalize_patches_individually(patches.to(torch.float64))
        
        noise_patches, _, _, _ = extract_squares_with_gaussian_weights(torch.randn(shape).to(torch.float64).cuda(), window_size, stride)
        B, N, C, dx1, dx2 = patches.shape
        patch_outputs = []

        for b in range(B):
            sample_patches = []
            for n in range(N):
                x_patch = patches[b, n:n+1]
                noise_patch = noise_patches[b, n:n+1]
                
                if model_input_size != window_size: 
                    x_patch = torch.nn.functional.interpolate(x_patch, size=model_input_size, mode='bilinear', align_corners=False)
                    noise_patch = torch.nn.functional.interpolate(noise_patch, size=model_input_size, mode='bilinear', align_corners=False)

                # # Built-in p_sample()
                # x_out, _ = diffusion.p_sample(upscaled, t, self_cond)
                
                # Custom p_sample()
                preds = diffusion.model_predictions(x_patch.to(torch.float32), torch.full((shape[0],), t, dtype = torch.long).cuda(), self_cond)
                x_start = preds.pred_x_start
                x_start.clamp_(-1., 1.) # This is true in the original p_mean_variance()
                model_mean, _, model_log_variance = diffusion.q_posterior(x_start = x_start, x_t = x_patch.to(torch.float32), t = torch.full((shape[0],), t, dtype = torch.long).cuda())

                # model_mean, _, model_log_variance, x_start = diffusion.p_mean_variance(upscaled.to(torch.float32), torch.full((shape[0],), t, dtype = torch.long).cuda(), self_cond) # Built-in p_mean_variance()
                noise = noise_patch if t > 0 else 0. # no noise if t == 0
                x_out = model_mean + (0.5 * model_log_variance).exp() * noise
                
                # x_out = upscaled # Debugging the patching/re-patching process
                
                downscaled = torch.nn.functional.interpolate(x_out, size=(dx1, dx2), mode='bilinear', align_corners=False)
                sample_patches.append(downscaled)
                
            patch_outputs.append(torch.cat(sample_patches, dim=0).unsqueeze(0))

        patches = torch.cat(patch_outputs, dim=0)  # (B, N, C, dx, dy)
        
        if normalize_patch:
            patches = denormalize_patches(patches.to(torch.float64), min_patch_vals, max_patch_vals)
        
        # img = combine_squares_with_gaussian_weights(patches.to(torch.float64), shape[-2:], stride, grid_shape)
        
        img = combine_squares_with_gaussian_weights(patches.to(torch.float64), weights.to(vp_true.device).to(torch.float64), shape[-2:], stride, grid_shape)

        if debug:
            plot_modulus((img.detach().cpu().numpy()-input_img.numpy())[0,0], aspect='auto', cmap='terrain', vmin=-1e-15, vmax=1e-15)
        
        # FWI guidance on full image
        if use_fwi_guidance and (i % inject_every == 0) and (i < run_fwi_under):
                
            x0 = img.detach().clone()
            # vp = denormalize_from_minusone_and_one(x0[0, 0], vmin, vmax).clone().detach().cuda().float().requires_grad_(True)

            vp_current = denormalize_from_minusone_and_one(
                x0[0, 0], vmin, vmax
            ).detach().to(device=device, dtype=torch.float32)

            # optimizer = torch.optim.Adam([{'params': [vp], 'lr': 10}])

            # (NEW) initialize persistent Parameter + optimizer once; then only copy values
            if vp_param is None:
                vp_param = torch.nn.Parameter(vp_current)  # requires_grad=True
                optimizer = torch.optim.Adam([{'params': [vp_param], 'lr': fwi_lr}])
            else:
                with torch.no_grad():
                    vp_param.copy_(vp_current)

            # for epoch in range(fwi_loop):
                
            #     optimizer.zero_grad()

            #     # Bounds projection and smoothing
            #     vp.data[vp.data < vmin] = vmin
            #     vp.data[vp.data > vmax] = vmax 
                
            #     running_loss = 0
                
            #     # Ensures the source locations are consistent when doing random sampling for both observed and synthetic data
            #     seed = torch.randint(0, 1234567, (1,)).item()
                
            #     # Simultaneous-source
            #     syn_data = forward_propagator(vp, multi_source=True, freq=freq, seed=seed, nt=nt, dt=dt, dz=dz, dx=dx, total_sources=total_sources)
            #     obs_data = forward_propagator(vp_true, multi_source=True, freq=freq, seed=seed, nt=nt, dt=dt, dz=dz, dx=dx, total_sources=total_sources) + 1e1 * torch.randn_like(syn_data)
            #     loss = torch.nn.functional.mse_loss(obs_data, syn_data)
            #     loss.backward(retain_graph=True)
            #     running_loss = loss.item()
                
            #     fwi_loss.append(running_loss)
                
            #     if (loss.isnan().sum()>0):
            #         print('Discovered NaNs in the data.')
                    
            #     optimizer.step() 
            
            for epoch in range(fwi_loop):
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    vp_param.clamp_(vmin, vmax)                     # keep bounds before sim
                    
                x_src = dx
                sampled_idx = np.random.choice(total_sources, size=total_sources, replace=False)
                y_src = torch.linspace(0, (nx - 1) * dx, total_sources)[sampled_idx]
                    
                for sou in range(0, total_sources, selected_sources):

                    source_locations = torch.zeros(1, selected_sources, 2)
                    source_locations[0, :, 0] = dx
                    source_locations[0, :, 1] = y_src[sou:sou+selected_sources]
                    
                    source_locations[:, :, 0] /= dz  # z-direction (usually 0 index)
                    source_locations[:, :, 1] /= dx  # x-direction (usually 1 index)

                    # your existing forward_propagator calls, but use 'vp_param'
                    syn_data = forward_propagator(vp_param, multi_source=True, freq=freq, nt=nt, dt=dt, dz=dz, dx=dx, total_sources=total_sources, source_locations=source_locations, selected_sources=selected_sources)
                    obs_data = forward_propagator(vp_true, multi_source=True, freq=freq, nt=nt, dt=dt, dz=dz, dx=dx, total_sources=total_sources, source_locations=source_locations, selected_sources=selected_sources) + 1e1 * torch.randn_like(syn_data)
                    
                    # (TIP: cache or precompute obs_data; do not re-noise it inside this loop)
                    loss = torch.nn.functional.mse_loss(obs_data, syn_data)
                    loss.backward()

                if max_grad is not None:
                    torch.nn.utils.clip_grad_norm_([vp_param], max_grad)
                optimizer.step()

                with torch.no_grad():
                    vp_param.clamp_(vmin, vmax)                     # project back to bounds

                img = normalize_to_minusone_and_one(vp_param.detach().clone().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1), vmin, vmax)
                
        imgs.append(denormalize_from_minusone_and_one(img, vmin, vmax))

    return (torch.stack(imgs, dim=1), fwi_loss) if return_all_timesteps else (imgs[-1], fwi_loss)

def main():
    
    parser = ArgumentParser()

    parser.add_argument(
    "--sigma",
    type=int,
    nargs='+',        # Accepts one or more integers
    default=[10, 10], # Default list if not provided
    )    
    parser.add_argument(
    "--velocity_type",
    type=str,
    default='seam_arid',
    )
    parser.add_argument(
    "--resize_model",
    type=str,
    default='n',
    )
    parser.add_argument(
    "--use_fwi",
    type=str,
    default='y',
    )
    parser.add_argument(
    "--training_data",
    type=str,
    default='seg',
    )
    parser.add_argument(
    "--start_guidance_from",
    type=int,
    default=700,
    )
    parser.add_argument(
    "--run_fwi_under",
    type=int,
    default=700,
    )
    parser.add_argument(
    "--inject_guidance_every",
    type=int,
    default=1,
    )
    parser.add_argument(
    "--guidance_loop",
    type=int,
    default=1,
    )
    parser.add_argument(
    "--window_size",
    type=int,
    default=256,
    )
    parser.add_argument(
    "--stride",
    type=int,
    default=128,
    )
    parser.add_argument(
    "--num_samples",
    type=int,
    default=1,
    )
    parser.add_argument(
    "--num_sources",
    type=int,
    default=64,
    )
    parser.add_argument(
    "--selected_sources",
    type=int,
    default=16,
    )
    parser.add_argument(
    "--frequency",
    type=int,
    default=5,
    )
    parser.add_argument(
    "--input_dim",
    type=int,
    default=256,
    help="Working resolution of the diffusion Unet (must match --input_dim used in DiffusionModel_2D_Training.py for the loaded checkpoint).",
    )
    parser.add_argument(
    "--openfwi_root",
    type=str,
    default='../data/FlatVel_A',
    help="Root folder of the OpenFWI FlatVel-A dataset (contains 'model/' and 'data/'). Only used when --velocity_type=flatvel_a.",
    )
    parser.add_argument(
    "--test_file",
    type=int,
    default=56,
    help="OpenFWI FlatVel-A file number used for inference (must be one of 56-60, the held-out test files).",
    )
    parser.add_argument(
    "--test_sample",
    type=int,
    default=0,
    help="Sample index (0-499) inside --test_file used for inference.",
    )
    parser.add_argument(
    "--openfwi_dx",
    type=float,
    default=10.0,
    help="Lateral grid spacing (m) of the OpenFWI FlatVel-A models (70 px over a 700 m domain by default).",
    )
    parser.add_argument(
    "--openfwi_dz",
    type=float,
    default=10.0,
    help="Depth grid spacing (m) of the OpenFWI FlatVel-A models.",
    )
    parser.add_argument(
    "--diffusion_results_folder",
    type=str,
    default=None,
    help="Folder holding the diffusion prior checkpoint trained on FlatVel-A (required when --training_data=flatvel_a).",
    )
    parser.add_argument(
    "--diffusion_checkpoint",
    type=int,
    default=None,
    help="Checkpoint milestone to load from --diffusion_results_folder (required when --training_data=flatvel_a).",
    )

    ##################################################################
    # Experiment logging
    ##################################################################
    
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = (
        '../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_'+str(run_id)
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
        dim = 256,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.input_dim,
        timesteps = 1000
    ).cuda()

    # Convert to torch tensor
    training_images = torch.randn((100,3,args.input_dim,args.input_dim)).float()#.cuda()

    # Resolve which pretrained diffusion prior to load. 'seg' and the other
    # (non-flatvel_a) original option keep their original hardcoded
    # results folder/checkpoint. 'flatvel_a' points at whatever run the user
    # produced with `DiffusionModel_2D_Training.py --training_data flatvel_a`,
    # whose folder name/checkpoint number are not known ahead of time.
    if args.training_data == 'flatvel_a':
        assert args.diffusion_results_folder is not None and args.diffusion_checkpoint is not None, (
            "--diffusion_results_folder and --diffusion_checkpoint must be provided "
            "when --training_data=flatvel_a (point them at your own "
            "DiffusionModel_2D_Training.py --training_data flatvel_a run)."
        )
        diffusion_results_folder = args.diffusion_results_folder
        diffusion_checkpoint = args.diffusion_checkpoint
    elif args.training_data == 'seg':
        diffusion_results_folder = '../results/DiffusionModel_2D_Training_20250623-092200'
        diffusion_checkpoint = 52
    else:
        diffusion_results_folder = '../results/DiffusionModel_2D_Training_20250723-005200'
        diffusion_checkpoint = 32

    trainer = Trainer(
        diffusion,
        training_images,
        train_batch_size = 4,
        train_lr = 2e-6, # 1e-5
        save_and_sample_every = 1000,
        num_samples = 16,
        results_folder = diffusion_results_folder,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 4,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False             # whether to calculate fid during training
    )

    # Load model
    trainer.load(diffusion_checkpoint)

    # Parameters
    resize_model = False if args.resize_model == 'n' else True# Set False to keep original size
    nz, nx = 256, 256       # Model input size if resizing is enabled
    device = 'cuda'

    # Velocity loading logic
    if args.velocity_type == 'seam_arid':
        vp_raw = np.fromfile('../data/velocities/Arid_vp_LR', np.float32).reshape(400, 400, 600)[:, 200, :]
        vs_raw = np.fromfile('../data/velocities/Arid_vs_LR', np.float32).reshape(400, 400, 600)[:, 200, :]
        rho_raw = np.fromfile('../data/velocities/Arid_rho_LR', np.float32).reshape(400, 400, 600)[:, 200, :]
        dx, dz = 25, 6.25

    elif args.velocity_type == 'seg_overthrust':
        vp_raw = np.fromfile('../data/velocities/overthrust', np.float32).reshape(801, 801, 187)[400, :, :]
        dx, dz = 25, 25

    elif args.velocity_type == 'bp_tiber':
        vp_raw = np.load('/home/taufikmh/Datasets/3D/01_C_tigris/vp_cropped.npy')[10, :, :]
        dx, dz = 25, 25

    elif args.velocity_type == 'seg_salt':
        vp_raw = np.fromfile('../data/velocities/salt', np.float32).reshape(676, 676, 210)[:, 330, :]
        dx, dz = 20, 20

    elif args.velocity_type == 'flatvel_a':
        # Replaces the proprietary SEAM/SEG/BP binaries above with a single
        # held-out OpenFWI FlatVel-A test sample (files 56-60, never mixed
        # with the 1-55 training range). v_true below plays exactly the same
        # role vp_raw already plays for the other (vp-only) velocity types.
        openfwi_sample = load_test_sample(args.openfwi_root, args.test_file, args.test_sample)
        vp_raw = openfwi_sample["velocity"].numpy()[0]  # (1, 70, 70) -> (70, 70)
        dx, dz = args.openfwi_dx, args.openfwi_dz

        # The real recorded gather (d_obs) is kept only for reference/QC: the
        # FWI guidance loop below stays a self-consistent synthetic experiment
        # (it forward-models its own "observed" data from vp_true with
        # Deepwave, exactly like every other velocity_type here). OpenFWI's
        # precomputed data{n}.npy uses a different, undocumented acquisition
        # (5 individual shots, 1000 time samples, unknown dt/wavelet/source
        # positions) that is inconsistent with the Deepwave acquisition
        # hardcoded a few lines below (nt=2000, dt=0.004s, --frequency Hz
        # Ricker, --num_sources/--selected_sources simultaneous shots).
        # Wiring data_obs directly into the gradient computation would
        # require rebuilding that acquisition geometry to match OpenFWI's,
        # which is a separate, unverified physics assumption -- so it is not
        # done implicitly here.
        openfwi_d_obs_reference = openfwi_sample["seismic"].numpy()  # (5, 1000, 70), for reference only

    else:
        raise ValueError(f"Unknown velocity type: {args.velocity_type}")

    if args.velocity_type != 'seam_arid':
        vs_raw = vp_raw/np.sqrt(2)
        rho_raw = 0.31 * vp_raw ** 0.25

    # Resize if needed
    if resize_model:
        vp_true_np = resize(vp_raw, (nz, nx))
        vs_true_np = resize(vs_raw, (nz, nx))
        rho_true_np = resize(rho_raw, (nz, nx))
        dz, dx = 20, 40
    else:
        nx, nz = vp_raw.shape  # use actual size
        vp_true_np = vp_raw
        vs_true_np = vs_raw
        rho_true_np = rho_raw

    # Create tensors
    vp_true = torch.from_numpy(vp_true_np).float().to(device).T
    vs_true = torch.from_numpy(vs_true_np).float().to(device).T
    rho_true = torch.from_numpy(rho_true_np).float().to(device).T
    
    vp_init = torch.from_numpy(gaussian_filter(vp_true_np, args.sigma)).float().to(device).T
    vs_init = torch.from_numpy(gaussian_filter(vs_true_np, args.sigma)).float().to(device).T
    rho_init = torch.from_numpy(gaussian_filter(rho_true_np, args.sigma)).float().to(device).T

    if args.velocity_type == 'flatvel_a':
        # Kept only as a reference artifact -- see the note above on why it is
        # not fed into the FWI guidance loop.
        np.save(os.path.join(results_folder, 'openfwi_d_obs_reference.npy'), openfwi_d_obs_reference)

    x_sharp = torch.cat((vp_true.unsqueeze(0).unsqueeze(0), vs_true.unsqueeze(0).unsqueeze(0), rho_true.unsqueeze(0).unsqueeze(0)), 1).detach().cpu().numpy()
    x_smooth = torch.cat((vp_init.unsqueeze(0).unsqueeze(0), vs_init.unsqueeze(0).unsqueeze(0), rho_init.unsqueeze(0).unsqueeze(0)), 1).detach().cpu().numpy()
    
    # Compute min and max along spatial dimensions only (keep sample and channel dimensions)
    min_vals = x_sharp.min(axis=(2, 3), keepdims=True)  # shape (N, 3, 1, 1)
    max_vals = x_sharp.max(axis=(2, 3), keepdims=True)  # shape (N, 3, 1, 1)
    
    # Normalize each channel independently for each sample
    x_smooth = 2 * (x_smooth - min_vals) / (max_vals - min_vals + 1e-8) - 1
    
    plot_modulus(
        vp_init.detach().cpu().numpy()/1e3,
        aspect='auto', cmap='rainbow', vmin=vp_true_np.min()/1e3, vmax=vp_true_np.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/smooth.pdf'
    )
    plot_modulus(
        vp_true.detach().cpu().numpy()/1e3, 
        aspect='auto', cmap='rainbow', vmin=vp_true_np.min()/1e3, vmax=vp_true_np.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/true.pdf'
    )
    
    ##################################################################
    # FWI-guided sampling
    ##################################################################
    
    samples, losses = [], []
    
    for _ in range(args.num_samples):
        
        sample, loss = p_sample_loop_with_fwi_guidance(
            trainer=trainer,
            shape=x_smooth.shape,
            return_all_timesteps=False,
            t_start=args.start_guidance_from,
            inject_every=args.inject_guidance_every,
            x_smooth=torch.from_numpy(x_smooth).float().cuda(),
            use_fwi_guidance=True if args.use_fwi=='y' else False,
            fwi_only=True,
            forward_propagator=forward_propagator,
            fwi_loop=args.guidance_loop,
            run_fwi_under=args.run_fwi_under,
            window_size=(args.window_size, args.window_size),
            stride=(args.stride, args.stride),
            model_input_size=(args.input_dim, args.input_dim),
            vmin=vp_true.min(), vmax=vp_true.max(),
            dx=dx, dz=dz, freq=args.frequency,
            vp_true=vp_true,
            normalize_patch=True,
            total_sources=args.num_sources,
            selected_sources=args.selected_sources
        )
        
        samples.append(sample)
        losses.append(loss)

    plot_modulus(
        sample[0, 0].detach().cpu().numpy()/1e3, 
        aspect='auto', cmap='rainbow', vmin=vp_true_np.min()/1e3, vmax=vp_true_np.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/sample.pdf'
    )
    
    # Compute statistics
    stacked_samples = torch.cat(samples, dim=0)[:, 0]

    # Compute mean and std over sample dimension (dim=0)
    mean = stacked_samples.mean(dim=0)               # shape: (256, 256)
    std = stacked_samples.std(dim=0)                 # shape: (256, 256)

    plot_modulus(
        mean.detach().cpu().numpy()/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true_np.min()/1e3, vmax=vp_true_np.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/mean.pdf'
    )
    plot_modulus(
        std.detach().cpu().numpy()/1e3, 
        aspect='auto', cmap='Reds',
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/std.pdf'
    )

    # Save diffusion samples
    np.save(results_folder+'/samples.npy', torch.cat(samples, dim=0).detach().cpu().numpy())
    np.save(results_folder+'/losses.npy', np.array(losses))
    
if __name__ == "__main__":
    main()