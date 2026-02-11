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
import gstools as gs
import pickle
import gc

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
from guidedfwi.svgd import RadialBasisFunction, SteinVariationalGradientDescent, compute_gradient, compute_gradient_per_batch, compute_max_gradient_per_batch
from guidedfwi.utils import log_experiment, normalize_to_minusone_and_one, denormalize_from_minusone_and_one, denormalize_from_zero_and_one
from guidedfwi.plots import plot_modulus, plot_diffusion_evolution, plot_flipped_data

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

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

def main():
    
    parser = ArgumentParser()
    
    parser.add_argument(
    "--num_samples",
    type=int,
    default=1,
    )
    parser.add_argument(
    "--perturbation_size",
    type=float,
    default=2.5,
    )
    parser.add_argument(
    "--perturbation_amplitude",
    type=float,
    default=1.25,
    )
    parser.add_argument(
    "--num_fwis",
    type=int,
    default=200,
    )
    parser.add_argument(
    "--sigma",
    type=int,
    default=10,
    )

    ##################################################################
    # Experiment logging
    ##################################################################
    
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = '../results/SVGDFWI_2D_Field_'+str(run_id)

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        
        # Log experiment
        log_experiment(
            results_folder, script_path=os.path.abspath(inspect.getfile(inspect.currentframe())), run_id=run_id
        )
        with open(os.path.join(results_folder, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    ##################################################################
    # Acquisition setup
    ##################################################################

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
    
    ##################################################################
    # Gaussian random field perturbations
    ##################################################################

    # Create Gaussian random field perturbations    
    vp_inits = np.repeat(vp_init.detach().cpu().numpy().reshape(1, nz, nx), args.num_samples, axis=0)
    masks = np.repeat(vp_mask.detach().cpu().numpy().reshape(1, nz, nx), args.num_samples, axis=0)
    
    # Create perturbations
    zax = np.arange(0, nz)
    xax = np.arange(0, nx)
    seed = 12315019
    
    # Control the spatial sizes proportional to the perturbation size
    len_scale = args.perturbation_size
    nu = args.perturbation_amplitude

    # Setup random fields with gstools
    grf_seed = gs.random.MasterRNG(seed)
    grf = np.zeros((args.num_samples, nz * nx))

    for i in range(args.num_samples):
        rf = gs.Matern(dim=2, var=2.5e-2, len_scale=len_scale, nu=nu)
        srf = gs.SRF(rf, seed=grf_seed(), store=f"better_field{i}")
        srf.set_pos([zax, xax], "structured")
        grf[i, :] = srf().reshape(1, nz * nx)

    perturbations = 500 * (grf.reshape(-1, nz, nx) * masks)

    # Add perturbations to the initial model
    vp_inits_perturb = vp_inits + perturbations

    _, ps_max = np.percentile(
        perturbations.std(0).reshape(nz, nx), [2, 98]
    )

    pm_min, pm_max = np.percentile(
        perturbations.mean(0).reshape(nz, nx), [2, 98]
    )
    ps_min = 0

    plot_modulus(perturbations.std(0)[:, :500] / 1e3, 
                interpolation='bicubic', cmap='Reds',
                vmin=ps_min / 1e3, vmax=ps_max / 1e3, 
                extent=[0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_perturb_std.pdf')
    )

    plot_modulus(perturbations.mean(0)[:, :500] / 1e3, 
                interpolation='bicubic', cmap='terrain',
                vmin=pm_min / 1e3, vmax=pm_max / 1e3, 
                extent=[0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_perturb_mean.pdf')
    )

    plot_modulus(perturbations[0][:, :500] / 1e3, 
                interpolation='bicubic', cmap='terrain', 
                extent=[0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0],
                vmin=pm_min * 5 / 1e3, vmax=pm_max * 5 / 1e3, 
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_perturb_sample.pdf')
    )

    plot_modulus(vp_inits_perturb.std(0)[:, :500] / 1e3, 
                interpolation='bicubic', cmap='Reds', 
                vmin=ps_min / 1e3, vmax=ps_max / 1e3, 
                extent=[0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_vp_perturb_std.pdf')
    )

    plot_modulus(vp_inits_perturb.mean(0)[:, :500] / 1e3, 
                interpolation='bicubic', cmap='terrain',
                vmin=m_min / 1e3, vmax=m_max / 1e3,
                extent=[0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_vp_perturb_mean.pdf')
    )

    plot_modulus(vp_inits_perturb[0][:, :500] / 1e3, 
                interpolation='bicubic', cmap='terrain',
                vmin=m_min / 1e3, vmax=m_max / 1e3,
                extent=[0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_vp_perturb_sample.pdf')
    )
    
    ##################################################################
    # SVGD loop
    ##################################################################
    
    # Counter to trackt the first gradient
    samples, losses = [], []
    
    vp_svgd = torch.from_numpy(vp_inits_perturb.reshape(args.num_samples, -1)).to(device).float()

    optimizer = torch.optim.Adam([vp_svgd], lr=5)
    loss_function = global_xcorr_loss
    scheduler = None
    alpha = torch.ones(args.num_fwis)
    svgd_kernel = RadialBasisFunction(power=1.5)
    
    gradient_function = lambda x: compute_gradient(
        x,
        source_locations,
        receiver_locations,
        nz,
        nx,
        [dz, dx],
        dt,
        freq,
        obs_data,
        source_amplitudes,
        loss_function,
        num_batches,
        None,
        device,
        [1/obs_scaler, 1/obs_scaler],
        grad_max_ref=1,
        vp_mask=vp_mask,
        smooth_sigma=2,
        vmin=1500,
        vmax=4500,
        receiver_mask=receiver_mask
    )
    
    # Initialize the  kernel and SVGD_FWI class
    svgd = SteinVariationalGradientDescent(vp_svgd, svgd_kernel, alpha, optimizer, scheduler, device='cuda')

    # Dictionary to save all variables
    svgd_results = {
        "epoch_loss": [],
        "updates": [],
        "updates_mean": [],
        "updates_std": [],
        "kernels": [],
        "sigmas": [],
    }

    # Inversion parameters
    loss_history = []
    total_iter = 0
    
    fwi_loop = tqdm(range(args.num_fwis))
    
    for iteration in fwi_loop:
        
        # Convert vp_svgd to CPU numpy array once and reuse
        vp_svgd_np = vp_svgd.detach().clone().cpu().numpy()

        # FWI model update and gradient computation
        running_loss = 0
        running_loss, grad = compute_gradient_per_batch(vp_svgd, gradient_function)
        
        grad_collection = torch.zeros_like(vp_svgd)
        
        # SVGD loop
        for ig, sgrad in enumerate(grad):
            smooth_grad = sgrad.reshape(nz, nx)
            smooth_grad = torch.from_numpy(smooth_grad).to(device)
            grad_collection[ig] = smooth_grad.ravel()

        if iteration == 0:
            gmax = compute_max_gradient_per_batch(grad)
            gmax = torch.tensor(gmax).to(device=device)

        svgd.step(vp_svgd, grad_collection, m_min, m_max, iteration, gmax, EMA=None)
        
        # Collect results
        svgd_results["updates"].append(vp_svgd_np.astype(np.float16))
        svgd_results["updates_mean"].append(vp_svgd_np.mean(0).astype(np.float16))
        svgd_results["updates_std"].append(vp_svgd_np.std(0).astype(np.float16))
        svgd_results["sigmas"].append(np.float16(svgd.sigma)) 
        svgd_results["epoch_loss"].append(np.float16(running_loss))
        svgd_results["kernels"].append(svgd.K_XX.detach().cpu().numpy().astype(np.float16))

        if torch.isnan(vp_svgd).any():
            print("NaN detected in vp_svgd")

        # Save every 5 iterations
        if (iteration % 5 == 0) or ((iteration+1) == args.num_fwis):
            save_path = os.path.join(results_folder, f'svgd_results_iter_{iteration:04d}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(svgd_results, f)

            # Plot and save most recent sample, mean, std
            latest_sample = svgd_results["updates"][-1][-1].reshape(nz, nx)[:, :500] / 1e3  # km/s
            latest_mean = svgd_results["updates_mean"][-1].reshape(nz, nx)[:, :500] / 1e3
            latest_std = svgd_results["updates_std"][-1].reshape(nz, nx)[:, :500] / 1e3

            extent_km = [0, (499) * dx/1e3, (nz - 1) * dz/1e3, 0]

            plot_modulus(
                latest_sample, aspect='auto', cmap='terrain', vmin=1.5, vmax=4.5,
                extent=extent_km, fig_name=os.path.join(results_folder, f'sample_iter_{iteration:04d}.pdf')
            )
            plot_modulus(
                latest_mean, aspect='auto', cmap='terrain', vmin=1.5, vmax=4.5,
                extent=extent_km, fig_name=os.path.join(results_folder, f'mean_iter_{iteration:04d}.pdf')
            )
            plot_modulus(
                latest_std * 3, aspect='auto', cmap='Reds',
                extent=extent_km, fig_name=os.path.join(results_folder, f'std_iter_{iteration:04d}.pdf')
            )

            gc.collect()
            torch.cuda.empty_cache()

        # Update tqdm postfix
        fwi_loop.set_postfix(iter=iteration, loss=running_loss)
        total_iter += 1
        loss_history.append(running_loss)

if __name__ == "__main__":
    main()