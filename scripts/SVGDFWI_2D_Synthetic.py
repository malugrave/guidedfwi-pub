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
import yaml

from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from tqdm.auto import tqdm
from ema_pytorch import EMA
from argparse import ArgumentParser

# Remove when guidedfwi is already installed
import sys

sys.path.append('../src/')
from guidedfwi.svgd import RadialBasisFunction, SteinVariationalGradientDescent, compute_gradient, compute_gradient_per_batch, compute_max_gradient_per_batch
from guidedfwi.utils import log_experiment
from guidedfwi.plots import plot_modulus, plot_diffusion_evolution, plot_flipped_data

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

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
    if source_locations is None or receiver_locations is None:
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

        receiver_locations = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
        receiver_locations[:, :, 0] = dx
        receiver_locations[:, :, 1] = torch.arange(num_receivers_per_shot) * dx
        
        # Divide by dx and dz to convert to grid units
        source_locations[:, :, 0] /= dz  # z-direction (usually 0 index)
        source_locations[:, :, 1] /= dx  # x-direction (usually 1 index)
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

def main():
    
    parser = ArgumentParser()
    
    parser.add_argument(
    "--num_samples",
    type=int,
    default=5,
    )
    parser.add_argument(
    "--frequency",
    type=int,
    default=5,
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
    default=100,
    )
    parser.add_argument(
    "--velocity_type",
    type=str,
    default='seam_arid',
    )
    parser.add_argument(
    "--config_file",
    type=str,
    default="../configs/synthetic-acquisition2.yaml",
    )
    parser.add_argument(
    "--multi_source",
    type=str,
    default="n",
    )
    parser.add_argument(
    "--sigma",
    type=int,
    nargs='+',        # Accepts one or more integers
    default=[10, 10], # Default list if not provided
    ) 
    parser.add_argument(
    "--num_sources",
    type=int,
    default=64,
    )

    ##################################################################
    # Experiment logging
    ##################################################################
    
    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = '../results/SVGDFWI_2D_Synthetic_'+str(run_id)

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

    # Parameters
    resize_model = False  # Set False to keep original size
    nz, nx = 256, 256        # Model input size if resizing is enabled
    device = 'cuda'

    # Velocity loading logic
    if args.velocity_type == 'seam_arid':
        vp_raw = np.fromfile('../data/velocities/Arid_vp_LR', np.float32).reshape(400, 400, 600)[:, 200, :]
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

    else:
        raise ValueError(f"Unknown velocity type: {args.velocity_type}")

    # Resize if needed
    if resize_model:
        vp_true_np = resize(vp_raw, (nz, nx))
        dz, dx = 20, 40
    else:
        nx, nz = vp_raw.shape  # use actual size
        vp_true_np = vp_raw

    # Create tensors
    vp_true = torch.from_numpy(vp_true_np).float().to(device).T
    vp_init = torch.from_numpy(gaussian_filter(vp_true_np, args.sigma)).float().to(device).T
    vp_mask = torch.ones_like(vp_init)
    
    m_min, m_max = vp_true.min(), vp_true.max()
    
    # FWI loop parameters
    multi_source = True if args.multi_source=='y' else False
    
    freq = args.frequency
    total_sources = args.num_sources
    num_batches = 16

    # if not multi_source:
    obs_data, source_amplitudes, source_locations, receiver_locations = forward_propagator(
        vp_true, multi_source=multi_source, return_src_info=True,
        total_sources=total_sources, freq=freq, dz=dz, dx=dx
    )
    
    obs_data += 1e1 * torch.randn_like(obs_data)
    
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

    plot_modulus(perturbations.std(0) / 1e3, 
                interpolation='bicubic', cmap='Reds',
                vmin=ps_min / 1e3, vmax=ps_max / 1e3, 
                extent=[0, nx * dx/1e3, nz * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_perturb_std.pdf')
    )

    plot_modulus(perturbations.mean(0) / 1e3, 
                interpolation='bicubic', cmap='rainbow',
                vmin=pm_min / 1e3, vmax=pm_max / 1e3, 
                extent=[0, nx * dx/1e3, nz * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_perturb_mean.pdf')
    )

    plot_modulus(perturbations[0] / 1e3, 
                interpolation='bicubic', cmap='rainbow', 
                extent=[0, nx * dx/1e3, nz * dz/1e3, 0],
                vmin=pm_min * 5 / 1e3, vmax=pm_max * 5 / 1e3, 
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_perturb_sample.pdf')
    )

    plot_modulus(vp_inits_perturb.std(0) / 1e3, 
                interpolation='bicubic', cmap='Reds', 
                vmin=ps_min / 1e3, vmax=ps_max / 1e3, 
                extent=[0, nx * dx/1e3, nz * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_vp_perturb_std.pdf')
    )

    plot_modulus(vp_inits_perturb.mean(0) / 1e3, 
                interpolation='bicubic', cmap='rainbow',
                vmin=m_min / 1e3, vmax=m_max / 1e3,
                extent=[0, nx * dx/1e3, nz * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_vp_perturb_mean.pdf')
    )

    plot_modulus(vp_inits_perturb[0] / 1e3, 
                interpolation='bicubic', cmap='rainbow',
                vmin=m_min / 1e3, vmax=m_max / 1e3,
                extent=[0, nx * dx/1e3, nz * dz/1e3, 0],
                aspect='auto',
                fig_name=os.path.join(results_folder, 'initial_vp_perturb_sample.pdf')
    )
    
    ##################################################################
    # SVGD loop
    ##################################################################
    
    # Counter to trackt the first gradient
    samples, losses = [], []
    
    vp_svgd = torch.from_numpy(vp_inits_perturb.reshape(args.num_samples, -1)).to(device).float()
    
    # Bounds projection and smoothing
    vp_svgd.data[vp_svgd.data < m_min] = m_min
    vp_svgd.data[vp_svgd.data > m_max] = m_max  

    optimizer = torch.optim.Adam([vp_svgd], lr=50)
    loss_function = torch.nn.functional.mse_loss
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
        0.004,
        freq,
        obs_data,
        source_amplitudes,
        loss_function,
        num_batches,
        None,
        device,
        grad_max_ref=1,
        vp_mask=vp_mask,
        vmin=m_min,
        vmax=m_max
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
        if iteration % 5 == 0:
            save_path = os.path.join(results_folder, f'svgd_results_iter_{iteration:04d}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(svgd_results, f)

            # Plot and save most recent sample, mean, std
            latest_sample = svgd_results["updates"][-1][-1].reshape(nz, nx) / 1e3  # km/s
            latest_mean = svgd_results["updates_mean"][-1].reshape(nz, nx) / 1e3
            latest_std = svgd_results["updates_std"][-1].reshape(nz, nx) / 1e3

            extent_km = [0, (nx - 1) * dx / 1e3, (nz - 1) * dz / 1e3, 0]

            plot_modulus(
                latest_sample, aspect='auto', cmap='rainbow', vmin=m_min/1e3, vmax=m_max/1e3,
                extent=extent_km, fig_name=os.path.join(results_folder, f'sample_iter_{iteration:04d}.pdf')
            )
            plot_modulus(
                latest_mean, aspect='auto', cmap='rainbow', vmin=m_min/1e3, vmax=m_max/1e3,
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