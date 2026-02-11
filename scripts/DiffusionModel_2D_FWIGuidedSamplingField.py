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
import time
import json

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
from guidedfwi.utils import log_experiment, normalize_to_minusone_and_one, denormalize_from_minusone_and_one, denormalize_from_zero_and_one

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

def extract_squares_vectorized(array, square_size, stride):
    """
    Extract overlapping square patches from a 4D tensor using stride.

    Parameters
    -----------
        array (Tensor): Input tensor of shape (B, C, H, W)
        square_size (tuple): Size of each square patch (dx, dy)
        stride (tuple): Stride between patches (sx, sy)

    Returns
    --------
        patches (Tensor): Extracted patches of shape (B, N, C, dx, dy)
        padded_shape (tuple): Padded shape of the input (H_pad, W_pad)
        grid_shape (tuple): Number of patches in height and width (nH, nW)
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
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, nH * nW, C, dx, dy)  # (B, N, C, dx, dy)

    return patches, (H_pad, W_pad), (nH, nW)

def combine_squares_vectorized(patches, original_shape, stride, grid_shape):
    """
    Reconstruct the full image from overlapping patches using weighted averaging.

    Parameters
    -----------
        patches (Tensor): Input patches of shape (B, N, C, dx, dy)
        original_shape (tuple): Original image shape (H, W)
        stride (tuple): Stride used during patch extraction (sx, sy)
        grid_shape (tuple): Grid size (nH, nW)

    Returns
    --------
        combined (Tensor): Reconstructed tensor of shape (B, C, H, W)
    """
    
    B, N, C, dx, dy = patches.shape
    sx, sy = stride
    H, W = original_shape
    nH, nW = grid_shape

    padded_H = H + (dx - H % dx) % dx
    padded_W = W + (dy - W % dy) % dy
    output_size = (padded_H, padded_W)

    patches = patches.view(B, nH, nW, C, dx, dy).permute(0, 3, 4, 5, 1, 2)
    patches = patches.reshape(B * C, dx * dy, nH * nW)

    fold = torch.nn.Fold(output_size=output_size, kernel_size=(dx, dy), stride=(sx, sy))
    unfold = torch.nn.Unfold(kernel_size=(dx, dy), stride=(sx, sy))

    combined = fold(patches)
    weight = fold(unfold(torch.ones_like(combined)))

    weight[weight == 0] = 1
    combined = combined / weight
    combined = combined.view(B, C, padded_H, padded_W)[..., :H, :W]

    return combined

def xcorr_loss(x,y):
    """
    Compute normalized cross-correlation loss between two vectors.

    Parameters
    -----------
        x (Tensor): Input tensor of shape (N,)
        y (Tensor): Target tensor of shape (N,)

    Returns
    --------
        loss (Tensor): Negative cosine similarity scalar value
    """
    
    x = x/torch.norm(x)
    y = y/torch.norm(y)
    loss = -torch.sum(torch.mul(x,y))
    return loss

def global_xcorr_loss(pred, obs, eps=1e-8):
    """
    Compute global normalized cross-correlation loss across traces.

    Parameters
    -----------
        pred (Tensor): Predicted data of shape (B, R, T)
        obs (Tensor): Observed data of shape (B, R, T)
        eps (float): Small value to prevent division by zero

    Returns
    --------
        loss (Tensor): Mean negative normalized cross-correlation across traces
    """
    
    pred = pred - pred.mean(dim=-1, keepdim=True)
    obs = obs - obs.mean(dim=-1, keepdim=True)

    num = torch.sum(pred * obs, dim=-1)
    denom = torch.sqrt(torch.sum(pred**2, dim=-1) * torch.sum(obs**2, dim=-1) + eps)

    ncc = num / denom.clamp(min=eps)
    loss = -torch.mean(ncc)  # maximize similarity

    return loss

def p_sample_loop_with_fwi_guidance(
    trainer,
    shape,
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
    window_size=(256, 256),
    stride=(64, 64),
    model_input_size=(256, 256),
    normalize_patch=False
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
    
    diffusion = trainer.ema.ema_model
    batch, device = shape[0], diffusion.device

    if x_smooth is not None:
        img = 1e-16 * torch.randn(shape, device=device) + x_smooth if fwi_only else 0.5 * torch.randn(shape, device=device) + x_smooth
    else:
        img = torch.randn(shape, device=device)

    imgs = [img]
    x_start = None
    debug_counter = 0

    # Compute min and max along spatial dimensions only (keep sample and channel dimensions)
    m_min = 1500
    m_max = 4500
    
    device = torch.device('cuda')
    
    source_amplitudes = torch.from_numpy(
    np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/source_amplitude.npy')
    ).float().to(device)

    obs_data = torch.from_numpy(
    np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/observed_data.npy')
    ).float().to(device)

    # Load initial velocity model
    vp_init = torch.from_numpy(
    np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/initial_velocity.npy')
    ).float().to(device)

    # Load velocity model mask
    vp_mask = torch.from_numpy(
    np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/velocity_mask.npy')
    ).float().to(device)

    # Load receiver mask
    receiver_mask = torch.from_numpy(
    np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/receiver_mask.npy')
    ).bool().to(device)
    source_locations = torch.from_numpy(
        np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/source_locations.npy').reshape(-1, 1, 2)
    ).float().to(device)

    receiver_locations = torch.from_numpy(
        np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/receiver_locations.npy').reshape(source_locations.shape[0], -1, 2)
    ).float().to(device)

    # Set parameters
    multi_source = False
    
    obs_scaler = torch.max(torch.abs(obs_data))

    dx = 25
    dz = 25
    nz, nx = vp_init.shape
    freq = 6
    dt = 0.002
    nt = 3525
    num_dims = 2
    num_batches = 8 # 4 worked on A100
    num_shots = 1 if multi_source else source_locations.shape[0]
    num_sources_per_shot = source_locations.shape[0] if multi_source else 1
    num_receivers_per_shot = receiver_locations.shape[1]

    obs_weight, syn_weight = 1/obs_scaler, 1/obs_scaler
    
    # Counter to trackt the first gradient
    fwi_counter = 0
    fwi_loss = []
    
    for i in tqdm(reversed(range(0, diffusion.num_timesteps - t_start)), desc='sampling loop time step', total=diffusion.num_timesteps - t_start):
        t = torch.tensor(i, device=device)
        if clip_input:
            img = img.clamp(-1, 1)

        self_cond = x_start if self_condition else None

        # Patch-based diffusion with interpolation to model_input_size
        input_img = img.detach().cpu().clone()
        
        # patches, padded_shape, grid_shape = extract_squares_vectorized(img.to(torch.float64), window_size, stride)
        
        patches, weights, padded_shape, grid_shape = extract_squares_with_gaussian_weights(img.clone().detach().to(torch.float64).reshape(shape), window_size, stride)
        
        if normalize_patch:
            patches, min_patch_vals, max_patch_vals = normalize_patches_individually(patches.to(torch.float64))
        
        # noise_patches, _, _ = extract_squares_vectorized(torch.randn_like(img).to(torch.float64), window_size, stride)
        
        noise_patches, _, _, _ = extract_squares_with_gaussian_weights(torch.randn(shape).to(torch.float64).cuda(), window_size, stride)
        
        B, N, C, dx1, dx2 = patches.shape
        patch_outputs = []

        for b in range(B):
            sample_patches = []
            for n in range(N):
                x_patch = patches[b, n:n+1]
                noise_patch = noise_patches[b, n:n+1]
                
                upscaled = torch.nn.functional.interpolate(x_patch, size=model_input_size, mode='bilinear', align_corners=False)
                noise_upscaled = torch.nn.functional.interpolate(noise_patch, size=model_input_size, mode='bilinear', align_corners=False)

                # # Built-in p_sample()
                # x_out, _ = diffusion.p_sample(upscaled, t, self_cond)
                
                # Custom p_sample()
                preds = diffusion.model_predictions(upscaled.to(torch.float32), torch.full((shape[0],), t, dtype = torch.long).cuda(), self_cond)
                x_start = preds.pred_x_start
                # x_start.clamp_(-1., 1.) # This is true in the original p_mean_variance()
                model_mean, _, model_log_variance = diffusion.q_posterior(x_start = x_start, x_t = upscaled.to(torch.float32), t = torch.full((shape[0],), t, dtype = torch.long).cuda())

                # model_mean, _, model_log_variance, x_start = diffusion.p_mean_variance(upscaled.to(torch.float32), torch.full((shape[0],), t, dtype = torch.long).cuda(), self_cond) # Built-in p_mean_variance()
                noise = noise_upscaled if t > 0 else 0. # no noise if t == 0
                x_out = model_mean # + (0.5 * model_log_variance).exp() * noise
                
                # x_out = upscaled # Debugging the patching/re-patching process
                
                downscaled = torch.nn.functional.interpolate(x_out, size=(dx1, dx2), mode='bilinear', align_corners=False)
                sample_patches.append(downscaled)
                
            patch_outputs.append(torch.cat(sample_patches, dim=0).unsqueeze(0))

        patches = torch.cat(patch_outputs, dim=0)  # (B, N, C, dx, dy)
        
        if normalize_patch:
            patches = denormalize_patches(patches.to(torch.float64), min_patch_vals, max_patch_vals)
        
        # img = combine_squares_vectorized(patches.to(torch.float64), shape[-2:], stride, grid_shape)
        
        img = combine_squares_with_gaussian_weights(patches.to(torch.float64), weights.to(upscaled.device).to(torch.float64), shape[-2:], stride, grid_shape)
        
        if debug:
            plot_modulus((img.detach().cpu().numpy()-input_img.numpy())[0,0], aspect='auto', cmap='terrain', vmin=-1e-15, vmax=1e-15)
        
        x_start = img.clone()
        
        # FWI guidance on full image
        if use_fwi_guidance and (i % inject_every == 0) and (i < run_fwi_under):
                
            x0 = img.detach().clone()
            vp = denormalize_from_minusone_and_one(x0[0, 0], vmin, vmax).clone().detach().cuda().float().requires_grad_(True)
            vp.data[vp_mask==0] = vmin

            optimizer = torch.optim.Adam([{'params': [vp], 'lr': 5}])

            for epoch in range(fwi_loop):
                
                running_loss = 0
                
                optimizer.zero_grad()
                
                # Bounds projection and smoothing
                vp.data[vp.data < 1500] = 1500
                vp.data[vp.data > 4500] = 4500  
                
                for it in range(0, num_shots, num_batches):
                    batch_src_amps = source_amplitudes[it:it+num_batches, :, :]
                    batch_rcv_amps_true = obs_data[it:it+num_batches, :, :].to(device)
                    batch_source_locations = source_locations[it:it+num_batches, :, :].to(device)
                    batch_receiver_locations = receiver_locations[it:it+num_batches, :, :].to(device)

                    batch_rcv_amps_pred = deepwave.scalar(
                        vp, 
                        dt=dt, 
                        grid_spacing=[dz, dx],
                        source_amplitudes=batch_src_amps,
                        source_locations=batch_source_locations,
                        receiver_locations=batch_receiver_locations,
                        accuracy=8,
                        pml_freq=freq,
                        pml_width=[0, 10, 10, 10],
                        max_vel=4500
                    )[-1]

                    loss = global_xcorr_loss(
                        obs_weight*batch_rcv_amps_pred[receiver_mask[it:it+num_batches, :]], 
                        syn_weight*batch_rcv_amps_true[receiver_mask[it:it+num_batches, :]]
                    )

                    loss.backward(retain_graph=True)
                    
                    running_loss += loss.item()
                
                fwi_loss.append(running_loss / num_shots)
                
                if (loss.isnan().sum()>0):
                    print('Discovered NaNs in the data.')
                    
                if (epoch==0) and (fwi_counter==0): 
                    gmax = torch.abs(vp.grad).max()
                    fwi_counter+=1
                
                # Normalize with the first max gradient
                vp.grad = vp.grad / gmax  #* v_mask.to(device)
                
                # Smooth gradient
                vp.grad = torch.tensor(gaussian_filter(vp.grad.cpu().numpy(), 2)).to(device)
                
                # Mask water layer
                vp.grad *= vp_mask
                
                assert vp_mask.shape == vp.shape
                    
                optimizer.step()

                img = normalize_to_minusone_and_one(vp.detach().clone().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1), vmin, vmax)
                
                if debug and debug_counter % 20 == 0:
                    plot_modulus(vp.detach().cpu().numpy()[:,:500]/1e3, aspect='auto', cmap='terrain', vmin=1.5, vmax=4.5, extent=[0,499*0.025,149*0.025,0])                    
                    plot_flipped_data(batch_rcv_amps_pred, batch_rcv_amps_true, batch_idx=-1, cmap='gray',vmin=-.1, vmax=.1)
                    
                debug_counter += 1

        imgs.append(denormalize_from_minusone_and_one(img, vmin, vmax))

    return (torch.stack(imgs, dim=1), fwi_loss) if return_all_timesteps else (imgs[-1], fwi_loss)

def main():
    
    parser = ArgumentParser()
    
    parser.add_argument(
    "--training_data",
    type=str,
    default='seg-256-pred-v',
    )
    parser.add_argument(
    "--start_guidance_from",
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
    "--model_size",
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
    "--sigma",
    type=int,
    default=10,
    )
    parser.add_argument(
    "--run_fwi_under",
    type=int,
    default=100,
    )
    
    ##################################################################
    # Experiment logging
    ##################################################################
    
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = '../results/DiffusionModel_2D_FWIGuidedSamplingField_'+str(run_id)

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
        dim = args.model_size,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.model_size,
        timesteps = 1000
    ).cuda()

    # Convert to torch tensor
    training_images = torch.randn(100,3,256,256).float()#.cuda()
    
    if args.training_data=='seg-256-pred-v':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250616-235700' 
        model_idx = 52
                
    elif args.training_data=='openfwi-256-pred-v':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250617-222000'
        model_idx = 32

    elif args.training_data=='seg-128-pred-noise':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250706-134054'
        model_idx = 8

    elif args.training_data=='seg-128-pred-x0':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250706-134056'
        model_idx = 8

    elif args.training_data=='seg-128-pred-v':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250706-134134'
        model_idx = 62
        
    elif args.training_data=='random-128-pred-noise':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250706-132748'
        model_idx = 8

    elif args.training_data=='random-128-pred-x0':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250706-133454'
        model_idx = 8

    elif args.training_data=='random-128-pred-v':
        diffusion_folder = '../results/DiffusionModel_2D_Training_20250706-132750'
        model_idx = 8
    
    trainer = Trainer(
        diffusion,
        training_images,
        train_batch_size = 1,
        train_lr = 2e-6, # 1e-5
        save_and_sample_every = 1000,
        num_samples = 16,
        results_folder = diffusion_folder,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 16,   # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False             # whether to calculate fid during training
    )

    # Load model
    trainer.load(model_idx)
    
    vp_init = torch.from_numpy(
    np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/initial_velocity.npy')
    ).float().to(diffusion.device)

    # Compute min and max along spatial dimensions only (keep sample and channel dimensions)
    min_vals = 1500
    max_vals = 4500

    x_init = 2 * (vp_init - min_vals) / (max_vals - min_vals + 1e-8) - 1

    plot_modulus(
        denormalize_from_minusone_and_one(x_init.detach().cpu().numpy()[:,:500], 1.5, 4.5), 
        aspect='auto', cmap='terrain', vmin=1.5, vmax=4.5, extent=[0,499*0.025,149*0.025,0],
        fig_name=results_folder+'/init.pdf'
    )
    
    ##################################################################
    # FWI-guided sampling
    ##################################################################
    
    samples, losses = [], []
    
    for _ in range(args.num_samples):
    
        sample, loss = p_sample_loop_with_fwi_guidance(
            trainer=trainer,
            shape=(1, 3, 150, 770),
            return_all_timesteps=False,
            t_start=args.start_guidance_from,
            inject_every=args.inject_guidance_every,
            x_smooth=x_init.float().cuda(),
            use_fwi_guidance=True,
            fwi_only=True,
            fwi_loop=args.guidance_loop,
            run_fwi_under=args.run_fwi_under, #1000-args.start_guidance_from,
            vmin=1500, vmax=4500,
            debug=False,
            freq=6,
            window_size=(args.window_size, args.window_size),
            stride=(args.stride, args.stride),
            model_input_size=(args.model_size, args.model_size)
            # clip_input=True
        )

        samples.append(sample)
        losses.append(loss)

    plot_modulus(
        sample[0].detach().cpu().numpy()[0][:,:499]/1e3, 
        aspect='auto', cmap='terrain', vmin=1.5, vmax=4.5, extent=[0,499*0.025,149*0.025,0],
        fig_name=results_folder+'/sample.pdf'
    )
    
    # Compute statistics
    stacked_samples = torch.cat(samples, dim=0)[:, 0]

    # Compute mean and std over sample dimension (dim=0)
    mean = stacked_samples.mean(dim=0)         
    std = stacked_samples.std(dim=0)              
    
    # Save diffusion samples
    np.save(results_folder+'/samples.npy', torch.cat(samples, dim=0).detach().cpu().numpy())
    np.save(results_folder+'/losses.npy', np.array(losses))

    plot_modulus(
        mean.detach().cpu().numpy()[:,:499]/1e3, 
        aspect='auto', cmap='terrain', vmin=1.5, vmax=4.5, extent=[0,499*0.025,149*0.025,0],
        fig_name=results_folder+'/mean.pdf'
    )
    plot_modulus(
        std.detach().cpu().numpy()[:,:499]/1e3, 
        aspect='auto', cmap='Reds', extent=[0,499*0.025,149*0.025,0],
        fig_name=results_folder+'/std.pdf'
    )

if __name__ == "__main__":
    main()