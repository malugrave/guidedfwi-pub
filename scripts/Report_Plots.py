"""
Description:
Accumulation of different plots used in the report.
 
Run as: 
python Report_Reports.py

Contributors:
Mohammad Hasyim Taufik (taufikmh)
"""

import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from scipy.ndimage import gaussian_filter

sys.path.append('../src/')
from guidedfwi.plots import plot_modulus

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
    
    results_folder = '../results/Reports'
    
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    ##################################################################
    # 2D Synthetic: SEAM Arid
    ##################################################################
    vp_true = np.fromfile('../data/velocities/Arid_vp_LR', np.float32).reshape(400, 400, 600)[:, 200, :].T
    dx, dz = 25, 6.25
    vp_init = gaussian_filter(vp_true, [10, 10])
    
    # Diffusion model
    vp_mean = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_20250810-230638/samples.npy')[:, 0].mean(0)    
    vp_std = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_20250810-230638/samples.npy')[:, 0].std(0)   
    
    # SVGD
    path = "../results/SVGDFWI_2D_Synthetic_20250806-002142/svgd_results_iter_0045.pkl"
    
    with open(path, "rb") as f:
        svgd_results = pickle.load(f)  # dict with "updates", "updates_mean", "updates_std" 
        
    svgd_mean   = svgd_results["updates_mean"][-1].reshape(vp_true.shape)
    svgd_std    = svgd_results["updates_std"][-1].reshape(vp_true.shape)

    plot_modulus(
        vp_true/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_arid_true.pdf'
    )
    
    plot_modulus(
        vp_init/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_arid_init.pdf'
    )

    plot_modulus(
        vp_mean/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_arid_mean.pdf'
    )

    plot_modulus(
        vp_std/1e3, 
        aspect='auto', cmap='Reds',
        vmin=0, vmax=0.25,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_arid_std.pdf'
    )

    plot_modulus(
        svgd_mean/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_arid_mean_svgd.pdf'
    )

    plot_modulus(
        svgd_std/1e3, 
        aspect='auto', cmap='Reds',
        vmin=0, vmax=0.25,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_arid_std_svgd.pdf'
    )
    
    ##################################################################
    # 2D Synthetic: SEG Salt
    ##################################################################
    vp_true = np.fromfile('../data/velocities/salt', np.float32).reshape(676, 676, 210)[:, 330, :].T
    dz, dx = 20, 20
    vp_init = gaussian_filter(vp_true, [5, 5])
    vp_mean = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_20250811-033618/samples.npy')[:, 0].mean(0)    
    vp_std = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_20250811-033618/samples.npy')[:, 0].std(0)    
    
    # SVGD
    path = "../results/SVGDFWI_2D_Synthetic_20250806-165410/svgd_results_iter_0045.pkl"
    
    with open(path, "rb") as f:
        svgd_results = pickle.load(f)  # dict with "updates", "updates_mean", "updates_std" 
        
    svgd_mean   = svgd_results["updates_mean"][-1].reshape(vp_true.shape)
    svgd_std    = svgd_results["updates_std"][-1].reshape(vp_true.shape)
    
    plot_modulus(
        vp_true/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_salt_true.pdf'
    )
    
    plot_modulus(
        vp_init/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_salt_init.pdf'
    )

    plot_modulus(
        vp_mean/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_salt_mean.pdf'
    )

    plot_modulus(
        vp_std/1e3, 
        aspect='auto', cmap='Reds',
        vmin=0, vmax=0.11,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_salt_std.pdf'
    )

    plot_modulus(
        svgd_mean/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_salt_mean_svgd.pdf'
    )

    plot_modulus(
        svgd_std/1e3, 
        aspect='auto', cmap='Reds',
        vmin=0, vmax=0.11,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_salt_std_svgd.pdf'
    )
    
    ##################################################################
    # 2D Synthetic: SEG/EAGE Overthrust
    ##################################################################
    vp_true = np.fromfile('../data/velocities/overthrust', np.float32).reshape(801, 801, 187)[400, :, :].T
    dx, dz = 25, 25
    vp_init = gaussian_filter(vp_true, [15, 5])
    vp_mean = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_20250811-051911/samples.npy')[:, 0].mean(0)    
    vp_std = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingSynthetic_20250811-051911/samples.npy')[:, 0].std(0)    
    
    # SVGD
    path = "../results/SVGDFWI_2D_Synthetic_20250806-215053/svgd_results_iter_0045.pkl"
    
    with open(path, "rb") as f:
        svgd_results = pickle.load(f)  # dict with "updates", "updates_mean", "updates_std" 
        
    svgd_mean   = svgd_results["updates_mean"][-1].reshape(vp_true.shape)
    svgd_std    = svgd_results["updates_std"][-1].reshape(vp_true.shape)

    plot_modulus(
        vp_true/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_overthrust_true.pdf'
    )

    plot_modulus(
        vp_init/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_overthrust_init.pdf'
    )

    plot_modulus(
        vp_mean/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_overthrust_mean.pdf'
    )

    plot_modulus(
        vp_std/1e3, 
        aspect='auto', cmap='Reds',
        vmin=0, vmax=0.13,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_overthrust_std.pdf'
    )

    plot_modulus(
        svgd_mean/1e3, 
        aspect='auto', cmap='rainbow',
        vmin=vp_true.min()/1e3, vmax=vp_true.max()/1e3,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_overthrust_mean_svgd.pdf'
    )

    plot_modulus(
        svgd_std/1e3, 
        aspect='auto', cmap='Reds',
        vmin=0, vmax=0.13,
        extent=[0, vp_true.shape[1] * dx/1e3, vp_true.shape[0] * dz/1e3, 0],
        fig_name=results_folder+'/synthetic2d_overthrust_std_svgd.pdf'
    )

    # ##################################################################
    # # 2D Field: The NW Australia (CGG)
    # ##################################################################
    
    # vp_mean = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250807-173601/samples.npy')[:, 0].mean(0)[:, :499]
    # vp_init = np.load('/home/taufikmh/KAUST/fall_2024/guidedifwi-dev/data/initial_velocity.npy')
    # vp_std = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250807-173601/samples.npy')[:, 0].std(0)[:, :499]
    # dz, dx = 25, 25
    
    # losses = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250807-173601/losses.npy')
    
    # plot_modulus(
    #     vp_init[:, :499]/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_init.pdf'
    # )
    
    # plot_modulus(
    #     vp_mean/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_mean_random.pdf'
    # )

    # plot_modulus(
    #     vp_std/1e3, 
    #     aspect='auto', cmap='Reds',
    #     vmin=np.percentile(vp_std, 3)/1e3, vmax=np.percentile(vp_std, 97)/1e3,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_std_random.pdf'
    # ) 
    # plot_modulus(
    #     np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250807-173601/samples.npy')[:, 0][0][:, :499]/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_sample_random.pdf'
    # ) 

    # p_low, p_high = 1, 99
    # low, high = np.percentile(losses, [p_low, p_high], axis=0)  # (2, 200)
    # x = np.arange(losses.shape[1])

    # plt.figure(figsize=(8,6))
    # plt.plot(x, np.mean(losses, axis=0) , label="Mean loss")
    # plt.fill_between(x, low, high, alpha=0.25, label=f"{p_low}–{p_high} percentile")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(results_folder+'/field2d_losses_random.pdf')
    
    # vp_mean = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250808-005336/samples.npy')[:, 0].mean(0)[:, :499]
    # vp_std = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250808-005336/samples.npy')[:, 0].std(0)[:, :499]
    # dz, dx = 25, 25
    
    # losses = np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250808-005336/losses.npy')
    
    # plot_modulus(
    #     vp_mean/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_mean_seg.pdf'
    # )

    # plot_modulus(
    #     vp_std/1e3, 
    #     aspect='auto', cmap='Reds',
    #     vmin=np.percentile(vp_std, 3)/1e3, vmax=np.percentile(vp_std, 97)/1e3,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_std_seg.pdf'
    # ) 
    # plot_modulus(
    #     np.load('../results/DiffusionModel_2D_FWIGuidedSamplingField_20250808-005336/samples.npy')[:, 0][0][:, :499]/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_sample_seg.pdf'
    # ) 
    
    # low, high = np.percentile(losses, [p_low, p_high], axis=0)  # (2, 200)
    # x = np.arange(losses.shape[1])

    # plt.figure(figsize=(8,6))
    # plt.plot(x, np.mean(losses, axis=0) , label="Mean loss")
    # plt.fill_between(x, low, high, alpha=0.25, label=f"{p_low}–{p_high} percentile")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(results_folder+'/field2d_losses_seg.pdf')
    
    # # SVGD
    # path = "../results/SVGDFWI_2D_Field_20250807-143638/svgd_results_iter_0195.pkl"
    
    # with open(path, "rb") as f:
    #     svgd_results = pickle.load(f)  # dict with "updates", "updates_mean", "updates_std" 
        
    # svgd_mean   = svgd_results["updates_mean"][-1].reshape(vp_init.shape)[:, :499]
    # svgd_sample   = svgd_results["updates"][-1].reshape(-1, 150,770)[0][:, :499]
    # svgd_std    = svgd_results["updates_std"][-1].reshape(vp_init.shape)[:, :499]
    # losses_svgd = np.array(svgd_results["epoch_loss"])
    
    
    # plot_modulus(
    #     svgd_mean/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_mean_svgd.pdf'
    # )

    # plot_modulus(
    #     svgd_sample/1e3, 
    #     aspect='auto', cmap='rainbow',
    #     vmin=1.5, vmax=4.5,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_sample_svgd.pdf'
    # )

    # plot_modulus(
    #     svgd_std/1e3, 
    #     aspect='auto', cmap='Reds',
    #     vmin=np.percentile(svgd_std, 3)/1e3, vmax=np.percentile(svgd_std, 97)/1e3,
    #     extent=[0, vp_mean.shape[1] * dx/1e3, vp_mean.shape[0] * dz/1e3, 0],
    #     fig_name=results_folder+'/field2d_std_svgd.pdf'
    # )

if __name__ == "__main__":
    main()