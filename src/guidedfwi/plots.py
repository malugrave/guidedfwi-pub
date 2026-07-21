import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import imageio
import os
import numpy as np
import matplotlib.animation as animation
import torch

from matplotlib.ticker import FormatStrFormatter

from scipy.interpolate import interp1d

def plot_shot_gather(
    data, receiver_locations, dt=0.004, dx=25, bin_dx=5, **kwargs
):
    """
    Plot seismic data using imshow, binning overlapping receiver locations.

    Parameters
    -----------
    data : Tensor [R, T]
        Simulated seismic data.
    receiver_locations : Tensor [R, 2]
        Receiver locations in normalized grid units (z, x).
    dt : float
        Time sampling interval (s).
    dx : float
        Grid spacing (used to convert from grid to meters).
    bin_dx : float
        Bin size in meters for aggregating overlapping receivers.
    """
    data = data.squeeze().cpu().numpy()  # [R, T]
    x_rcv = receiver_locations.squeeze()[:, 1].cpu().numpy() * dx  # [R] in meters
    t = np.arange(data.shape[1]) * dt  # [T] time axis in seconds

    # Create bins along receiver axis
    x_min, x_max = x_rcv.min(), x_rcv.max()
    x_edges = np.arange(x_min, x_max + bin_dx, bin_dx)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    num_bins = len(x_centers)

    # Initialize binned output
    binned_data = np.zeros((num_bins, data.shape[1]))
    bin_counts = np.zeros((num_bins, 1))

    # Digitize receiver positions into bins
    bin_idx = np.digitize(x_rcv, x_edges) - 1  # convert to 0-based

    for i in range(len(x_rcv)):
        b = bin_idx[i]
        if 0 <= b < num_bins:
            binned_data[b] += data[i]
            bin_counts[b] += 1

    # Normalize by number of receivers per bin
    bin_counts[bin_counts == 0] = 1  # prevent divide-by-zero
    # binned_data /= bin_counts

    # Plot
    extent = [x_centers[0]/1e3, x_centers[-1]/1e3, t[-1], t[0]]  # km, s
    plt.figure(figsize=(12, 6))
    plt.imshow(
        binned_data.T, aspect='auto', cmap='gray', extent=extent, **kwargs
    )
    plt.xlabel("Receiver Position (km)")
    plt.ylabel("Time (s)")
    plt.title("Shot Gather (binned over receiver positions)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
    
def plot_interleaved_data(
    pred,
    obs,
    receiver_locations=None,
    batch_idx=0,
    normalize=True,
    interleave_every=10,
    dx=25,
    dt=0.004,
    bin_dx=5,
    fig_name=None,
    **kwargs
):
    """
    Plot a wavefield combining alternating blocks from observed and predicted data.
    Uses binning to handle overlapping receiver locations.

    Parameters
    -----------
    pred: Tensor [R, T] or [B, R, T]
    obs: Tensor [R, T] or [B, R, T]
    receiver_locations: Tensor [1, R, 2] or [R, 2]
    dx: spatial sampling (m) for converting from grid to real-world
    dt: time sampling (s)
    bin_dx: spatial resolution of receiver bin (m)
    interleave_every: number of traces per block
    """
    # Handle batch
    if pred.ndim == 3:
        pred = pred[batch_idx]
    if obs.ndim == 3:
        obs = obs[batch_idx]
    assert pred.shape == obs.shape, "Prediction and observation shape mismatch"
    
    # Normalize
    if normalize:
        max_val = torch.max(torch.abs(torch.stack([pred, obs])))
        pred = pred / max_val
        obs = obs / max_val

    if receiver_locations is not None:
        x_rcv = receiver_locations.squeeze()[:, 1].cpu().numpy() * dx
        t = np.arange(pred.shape[1]) * dt
        R, T = pred.shape

        # Binning coordinates
        x_min, x_max = x_rcv.min(), x_rcv.max()
        x_edges = np.arange(x_min, x_max + bin_dx, bin_dx)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        num_bins = len(x_centers)

        # Digitize receivers into bins
        bin_idx = np.digitize(x_rcv, x_edges) - 1

        def bin_wavefield(wave):
            binned = np.zeros((num_bins, T))
            counts = np.zeros((num_bins, 1))
            wave_np = wave.detach().cpu().numpy()
            for i in range(R):
                b = bin_idx[i]
                if 0 <= b < num_bins:
                    binned[b] += wave_np[i]
                    counts[b] += 1
            counts[counts == 0] = 1
            # return torch.tensor(binned / counts, dtype=torch.float32)
            return torch.tensor(binned, dtype=torch.float32)

        pred_binned = bin_wavefield(pred)
        obs_binned = bin_wavefield(obs)

        # Interleave
        output = obs_binned.clone()
        toggle = True
        for i in range(0, num_bins, interleave_every):
            if not toggle:
                output[i:i+interleave_every] = pred_binned[i:i+interleave_every]
            toggle = not toggle

        extent = [x_centers[0]/1e3, x_centers[-1]/1e3, t[-1], t[0]]  # km, s
        x_labels = x_centers / 1e3
    else:
        # Fallback to trace index mode
        output = obs.clone()
        toggle = True
        for i in range(0, pred.shape[0], interleave_every):
            if not toggle:
                output[i:i+interleave_every] = pred[i:i+interleave_every]
            toggle = not toggle
        extent = None
        x_labels = np.arange(pred.shape[0])

    # vmin/vmax
    vmin = kwargs.pop("vmin", float(output.min()))
    vmax = kwargs.pop("vmax", float(output.max()))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(output.T, aspect='auto', extent=extent, **kwargs)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.0125, pad=0.005)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['−', '+'])

    # Axis labels
    ax.set_xlabel("Receiver Position (km)" if receiver_locations is not None else "Receiver Index")
    ax.set_ylabel("Time (s)")

    # Vertical interleave markers
    for x in range(interleave_every, output.shape[0], interleave_every):
        x_val = x_labels[x]
        ax.axvline(x=x_val, color='white', linestyle='--', linewidth=1)

    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name)
    plt.show()


# def plot_interleaved_data(
#     pred,
#     obs,
#     receiver_locations=None,
#     batch_idx=0,
#     normalize=True,
#     interleave_every=10,
#     dx=25,
#     dt=0.004,
#     interp_dx=5,
#     fig_name=None,
#     **kwargs
# ):
#     """
#     Plot a wavefield combining alternating blocks from observed and predicted data.
#     Meant to visualize FWI data fitting by replacement, not duplication.

#     Parameters
#     -----------
#     pred: Tensor [R, T] or [B, R, T]
#     obs: Tensor [R, T] or [B, R, T]
#     receiver_locations: Tensor [1, R, 2] or [R, 2]
#     dx: spatial sampling (m) for converting from grid to real-world
#     dt: time sampling (s)
#     interp_dx: interpolation resolution along receiver axis (m)
#     interleave_every: number of traces per block
#     """
#     # Prepare input shape
#     if pred.ndim == 3:
#         pred = pred[batch_idx]
#     if obs.ndim == 3:
#         obs = obs[batch_idx]
#     assert pred.shape == obs.shape, "Prediction and observation shape mismatch"
    
#     # Normalize before interpolation
#     if normalize:
#         max_val = torch.max(torch.abs(torch.stack([pred, obs])))
#         pred = pred / max_val
#         obs = obs / max_val

#     if receiver_locations is not None:
#         # Extract x receiver coordinates in meters
#         x_rcv = receiver_locations.squeeze()[:, 1].detach().cpu().numpy() * dx
#         t = np.arange(pred.shape[1]) * dt

#         # Create a uniform grid for interpolation
#         x_uniform = np.arange(np.min(x_rcv), np.max(x_rcv) + interp_dx, interp_dx)
#         T = pred.shape[1]

#         def interp_to_uniform(data_tensor):
#             data_np = data_tensor.detach().cpu().numpy()  # [R, T]
#             interp_data = np.zeros((len(x_uniform), T))
#             for i in range(T):
#                 f = interp1d(x_rcv, data_np[:, i], bounds_error=False, fill_value=0.0)
#                 interp_data[:, i] = f(x_uniform)
#             return torch.tensor(interp_data)

#         pred_interp = interp_to_uniform(pred)
#         obs_interp = interp_to_uniform(obs)

#         # Interleave after interpolation
#         R_interp = pred_interp.shape[0]
#         output = obs_interp.clone()
#         toggle = True
#         for i in range(0, R_interp, interleave_every):
#             if not toggle:
#                 output[i:i+interleave_every] = pred_interp[i:i+interleave_every]
#             toggle = not toggle

#         extent = [x_uniform[0]/1e3, x_uniform[-1]/1e3, t[-1], t[0]]  # km, seconds
#         x_labels = x_uniform / 1e3

#     else:
#         # No receiver locations: use raw trace index
#         output = obs.clone()
#         toggle = True
#         for i in range(0, pred.shape[0], interleave_every):
#             if not toggle:
#                 output[i:i+interleave_every] = pred[i:i+interleave_every]
#             toggle = not toggle
#         extent = None
#         x_labels = np.arange(pred.shape[0])

#     # Get vmin/vmax for consistent color scaling
#     vmin = kwargs.pop("vmin", float(output.min()))
#     vmax = kwargs.pop("vmax", float(output.max()))

#     # Plotting
#     fig, ax = plt.subplots(figsize=(12, 5))
#     im = ax.imshow(output.T, aspect='auto', extent=extent, **kwargs)

#     # Colorbar formatting
#     cbar = fig.colorbar(im, ax=ax, fraction=0.0125, pad=0.005)
#     cbar.set_ticks([vmin, vmax])
#     cbar.set_ticklabels(['−', '+'])

#     # Axis labels
#     ax.set_xlabel("Receiver Position (km)" if receiver_locations is not None else "Receiver Index")
#     ax.set_ylabel("Time (s)")

#     # Vertical dashed lines at interleave boundaries
#     for x in range(interleave_every, output.shape[0], interleave_every):
#         x_val = x_labels[x]
#         ax.axvline(x=x_val, color='white', linestyle='--', linewidth=1)

#     plt.tight_layout()
#     if fig_name:
#         plt.savefig(fig_name)
#     plt.show()
    
def plot_flipped_data(pred, obs, batch_idx=0, normalize=True, fig_name=None, **kwargs):
    """
    Plot a wavefield combining alternating blocks from observed and predicted data.
    Meant to visualize FWI data fitting by replacement, not duplication.

    Parameters
        pred: Tensor of shape [R, T] or [B, R, T]
        obs: Tensor of shape [R, T] or [B, R, T]
        interleave_every: number of receiver traces per block
    """
    if pred.ndim == 3:
        pred = pred[batch_idx]
    if obs.ndim == 3:
        obs = obs[batch_idx]

    assert pred.shape == obs.shape, "Shape mismatch between prediction and observation"
    R, T = pred.shape

    if normalize:
        max_val = torch.max(torch.abs(torch.stack([obs, pred])))
        obs = obs / max_val
        pred = pred / max_val

    # Create output by interleaving values
    output = torch.cat((torch.flipud(pred), obs, torch.flipud(pred))).cpu()
        
    # Get vmin and vmax from kwargs if provided, else fallback to data range
    vmin = kwargs.get("vmin", float(output.min()))
    vmax = kwargs.get("vmax", float(output.max()))

    # Plot with vertical lines
    fig, ax = plt.subplots(figsize=(12, 5))
    im = plt.imshow(output.detach().cpu().numpy().T, aspect='auto', **kwargs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.0125, pad=0.005)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['−', '+'])  # Use en dash for minus
    plt.xlabel("Receiver Index")
    plt.ylabel("Time Sample")

    # Draw vertical boundaries
    for x in range(0, output.shape[0], obs.shape[0]):
        if x!=0:
            plt.axvline(x=x, color='white', linestyle='--', linewidth=1)

    plt.tight_layout()

    if fig_name is not None: 
        plt.savefig(fig_name)

    plt.show()

def plot_diffusion_evolution(
    samples,
    show_m=10,
    start_step=None,
    end_step=None,
    fig_name=None,
    **kwargs
):
    """
    Plots the evolution of diffusion samples over selected timesteps.

    Parameters
        samples (torch.Tensor): Tensor of shape (1, T, C, H, W)
        every_n (int): Interval between plotted timesteps
        show_m (int): Number of images to show (maximum)
        start_step (int): Starting timestep (inclusive). If None, defaults to 0.
        end_step (int): Ending timestep (inclusive). If None, defaults to T-1.

    Returns
        None (displays matplotlib plot)
    """
    T, C, H, W = samples.shape
    assert C >= 1, "Expected at least one channel"

    # Handle default timestep range
    if end_step is None:
        end_step = T - 1
    if start_step is None:
        start_step = 0

    # Clamp range and generate indices
    start_step = max(0, start_step)
    end_step = min(T - 1, end_step)
    
    every_n = (end_step - start_step) // (show_m - 1)
    
    indices = list(range(start_step, end_step + 1, every_n))[:show_m]

    fig, axs = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
    if len(indices) == 1:
        axs = [axs]

    for i, idx in enumerate(indices):
        img = samples[idx, 0].cpu().detach().numpy()
        axs[i].imshow(img, **kwargs)
        axs[i].set_title(f"Step {idx}")
        axs[i].axis('off')

    plt.tight_layout()
    
    if fig_name is not None: 
        plt.savefig(fig_name)
        
    plt.show()

def plot_several_modulus(array, n, m, size_n=4, size_m=4, fig_name=None, 
                          recs=None, sous=None, logs=None, logs_thickness=10, cbar_label=None, **kwargs):
    """
    Generate a series of images from a 3D numpy array (N, M, W, H) representing multiple 2D datasets.
    
    Parameters
    ----------
    array: 3D numpy array of shape (N, W, H).
    n: Number of rows for the subplot configuration.
    m: Number of columns for the subplot configuration.
    size_n: Height of the figure.
    size_m: Width of the figure.
    fig_name: Name of the output file for saving.
    recs: 2D array of receiver coordinates for scatter plot.
    sous: 2D array of source coordinates for scatter plot.
    logs: List of log information for plotting.
    logs_thickness: Line thickness for log plots.
    **kwargs: Additional keyword arguments for imshow (e.g., colormap).
    """
    
    # Create subplots with shared y-axis
    fig, axs = plt.subplots(n, m, figsize=(size_m * m, size_n * n), sharey=True, sharex=True)
    plt.tight_layout()

    # Ensure axs is a 2D array for easy indexing
    if n == 1:
        axs = [axs]  # Make axs iterable if there's only one row

    # Flatten axs for easier iteration if there are multiple rows
    if n > 1 or m > 1:
        axs = axs.flatten()

    # Plot each modulus image based on adjusted window sizes
    for idx in range(n * m):

        if idx < array.shape[0]:  # Ensure index is within bounds
            ax = axs[idx]
            ax.imshow(array[idx], **kwargs)
            
            if recs is not None:
                ax.scatter(recs[:, 0], recs[:, 1], c='k')
            
            if sous is not None:
                ax.scatter(sous[:, 0], sous[:, 1], c='r')
            
            if logs is not None:
                for log in logs:
                    ax.plot(log[0], log[1], log[2], linewidth=logs_thickness)
        
        else:
            axs[idx].axis('off')  # Turn off any remaining subplots

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', pad=0.04, shrink=0.5)
    cbar.set_label(cbar_label)  # Label the colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format the colorbar ticks

    # Save the figure if a filename is provided
    if fig_name is not None: 
        plt.savefig(fig_name)

    plt.show()

def plot_shot_gathers_animated(data: np.ndarray, fig_name: str, fps: int, **kwargs):
    """
    Create a GIF animation from a 4D numpy array of size (N, ns, nt, nr).
    
    Parameters
    ----------
    - data: 4D numpy array of size (N, ns, nt, nr).
    - fig_name: Name of the output GIF file.
    - fps: Frame per second, the lower, the slower.
    """
    N = data.shape[0]  # Number of images to animate

    # Create subplots with shared y-axis
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    
    # Initialize the first frame for colorbar
    im = axs[0].imshow(data[0, 0], aspect='auto', cmap='gray', **kwargs)
    
    # Create a colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Value')  # Label the colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format the colorbar ticks

    # Animation update function
    def update(frame):
        for ax, ishot in zip(axs, [0, 1, 2]):
            ax.clear()  # Clear the previous frame
            im = ax.imshow(data[frame, ishot], aspect='auto', cmap='gray',
                           **kwargs)
            ax.set_xlabel('Receiver')
            if ishot == 0:
                ax.set_ylabel('nt')
        return axs

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=N, interval=100)

    # Save the animation as a GIF
    ani.save(fig_name, writer='pillow', fps=fps)
    
    # Clean up the figure
    plt.close(fig)

def plot_shot_gathers(data, fig_name=None, **kwargs):
    """
    Create a subplot from a 3D numpy array of size (ns, nt, nr).
    
    Parameters
    ----------
    - data: 3D numpy array of size (ns, nt, nr).
    - fig_name: Name of the output image file.
    """
    # Create subplots with shared y-axis
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    # Plotting
    for ax, ishot in zip(axs, [0, 1, 2]):
        im = ax.imshow(data[ishot], aspect='auto', cmap='gray',
                    **kwargs)
        ax.set_xlabel('Receiver')
        
        if ishot == 0:
            ax.set_ylabel('nt')

    # Add a single colorbar on the right side
    cbar = plt.colorbar(im, ax=axs, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if fig_name is not None: 
        plt.savefig(fig_name)

    plt.show()

def plot_statistics(data, fig_name=None):
    """
    Plot the mean, 5th, and 95th percentiles for each element in the data.

    Parameters
    ----------
    - data: A list of lists, where each inner list contains numerical values (M items).
    """
    # Convert to a numpy array for easier calculations
    data_array = np.array(data)

    # Calculate statistics
    means = np.mean(data_array, axis=1)
    percentiles_5 = np.percentile(data_array, 5, axis=1)
    percentiles_95 = np.percentile(data_array, 95, axis=1)

    # Create indices for the N elements
    indices = np.arange(len(means))

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(indices, means, label='Mean', marker='o')
    plt.fill_between(indices, percentiles_5, percentiles_95, color='lightblue', alpha=0.5, label='5th and 95th Percentiles')
    
    # Adding labels and title
    plt.legend()
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    if fig_name is not None: 
        plt.savefig(fig_name)
        
    plt.show()

def plot_modulus(m, fig_name=None, 
                recs=None, sous=None, logs=None, logs_thickness=10, **kwargs):
    """
    Generate image from a 2D acoustic modulus.

    Parameters
    ----------
    m: 2D numpy arrays to plot.
    fig_name: Name of the output GIF file.
    recs: 2D array of receiver coordinates for scatter plot.
    sous: 2D array of source coordinates for scatter plot.
    logs: List of log information for plotting.
    logs_thickness: Line thickness for log plots.
    **kwargs: Additional keyword arguments for imshow.
    """
    
    plt.figure(figsize=(12, 5))
    plt.imshow(m, **kwargs)
    cbar = plt.colorbar()
    cbar.set_label('km/s')
    
    if 'extent' in kwargs:
        plt.xlabel('km')
        plt.ylabel('km')
    else:
        plt.xlabel('nx')
        plt.ylabel('nz')
        
    if recs is not None:
        plt.scatter(recs[:,0], recs[:,1], c='k')
    
    if sous is not None:
        plt.scatter(sous[:,0], sous[:,1], c='r')
        
    if logs is not None:
        for i in range(len(logs)):
            plt.plot(logs[i][0], logs[i][1], logs[i][2], linewidth=logs_thickness)
        
    if fig_name is not None:
        plt.savefig(fig_name)

    plt.show()

def plot_stagewise_diffusion(stage_data, vmin, vmax, outpath, n_show=6, channel=0, cmap='rainbow'):
    """
    Grid of n_show diffusion stages, one row per stage, showing (left to
    right): the noisy state entering the step, the network's denoised (x0)
    estimate, the DDPM ancestral-sampling output (which already re-injects
    the posterior noise for the next, less noisy timestep), and the state
    after FWI guidance for that step (identical to the DDPM output on steps
    where guidance did not fire).

    Parameters
    ----------
    stage_data : dict
        As returned by p_sample_loop_with_fwi_guidance(..., save_stage_plots=True)
        (its last returned element), with keys 't', 'x_before', 'x0_hat',
        'x_ddpm_step', 'x_after' -- each a list of (1, C, H, W) tensors, one
        entry per saved diffusion timestep.
    vmin, vmax : float
        Color scale limits (physical units), shared across all panels.
    outpath : str
        Output file path.
    n_show : int
        Number of stages (rows) to display, evenly spaced across the saved steps.
    channel : int
        Which channel to plot (0=vp, 1=vs, 2=rho).
    cmap : str
        Colormap for the velocity panels.
    """
    from matplotlib.gridspec import GridSpec

    n_saved = len(stage_data["t"])
    if n_saved < 1:
        raise ValueError("No saved stages available.")

    n_show = int(min(max(n_show, 1), n_saved))
    idxs = np.unique(np.linspace(0, n_saved - 1, n_show).round().astype(int))

    ncols = 4
    nrows = len(idxs)

    fig = plt.figure(figsize=(5.0 * ncols + 0.8, 4.0 * nrows))
    gs = GridSpec(
        nrows, ncols + 1, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.055], wspace=0.12, hspace=0.25,
    )
    cax = fig.add_subplot(gs[:, -1])
    im = None

    for row, idx in enumerate(idxs):
        t_current = stage_data["t"][idx]
        fields = [
            stage_data["x_before"][idx].numpy()[0, channel],
            stage_data["x0_hat"][idx].numpy()[0, channel],
            stage_data["x_ddpm_step"][idx].numpy()[0, channel],
            stage_data["x_after"][idx].numpy()[0, channel],
        ]
        titles = [
            f"x_t (noisy input)\nt={t_current}",
            "denoised estimate\n(x0_hat)",
            "DDPM step\n(re-noised for t-1)",
            "after FWI guidance",
        ]

        for col, (field, title) in enumerate(zip(fields, titles)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title, fontsize=12, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"stage {idx}", fontsize=11, labelpad=12)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("velocity", fontsize=12)
    fig.suptitle(f"Diffusion + FWI guidance stagewise evolution: {len(idxs)} stages", fontsize=16, y=0.995)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_modulus_animated(arrays, gif_name='output.gif', recs=None, sous=None, logs=None, logs_thickness=10, **kwargs):
    """
    Generate a GIF from a list of 2D numpy arrays.

    Parameters
    ----------
    arrays: 3D numpy array (N x W x D) to plot.
    gif_name: Name of the output GIF file.
    recs: 2D array of receiver coordinates for scatter plot.
    sous: 2D array of source coordinates for scatter plot.
    logs: List of log information for plotting.
    logs_thickness: Line thickness for log plots.
    **kwargs: Additional keyword arguments for imshow.
    """
    
    # Check if the input array is 3D
    if arrays.ndim != 3:
        raise ValueError("Input arrays must be a 3D numpy array (N x W x D)")

    N, W, D = arrays.shape  # Unpack dimensions

    # List to hold the filenames of saved frames
    filenames = []

    for i in range(N):
        m = arrays[i]  # Get the 2D slice for frame i
        plt.figure(figsize=(12, 5))
        plt.imshow(m, **kwargs)
        cbar = plt.colorbar()
        cbar.set_label('km/s')

        if 'extent' in kwargs:
            plt.xlabel('km')
            plt.ylabel('km')
        else:
            plt.xlabel('nx')
            plt.ylabel('nz')
        
        if recs is not None:
            # Ensure recs are valid for the current frame plotting
            plt.scatter(recs[:, 0], recs[:, 1], c='k')
        
        if sous is not None:
            # Ensure sous are valid for the current frame plotting
            plt.scatter(sous[:, 0], sous[:, 1], c='r')
        
        if logs is not None:
            for log in logs:
                plt.plot(log[0], log[1], log[2], linewidth=logs_thickness)

        # Save the current figure to a file
        filename = f'temp_frame_{i}.png'
        plt.savefig(filename)
        filenames.append(filename)

        plt.close()  # Close the figure to save memory

    # Create a GIF from the saved frames
    with imageio.get_writer(gif_name, mode='I', duration=2.5, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Optionally, remove temporary files (clean-up)
    for filename in filenames:
        os.remove(filename)
        
def plot_moduli(m1, m2, m3, vmin=[None, None, None], vmax=[None, None, None], 
                fig_name=None, save_dir='./', 
                recs=None, sous=None, **kwargs):
    """
    Generate image from 2D elastic moduli.

    Parameters
    ----------
    m1: 2D numpy array to plot.
    m2: 2D numpy array to plot.
    m3: 2D numpy array to plot.
    vmin: Minimum values for the color scale for each plot.
    vmax: Maximum values for the color scale for each plot.
    fig_name: Name of the output file.
    save_dir: Directory to save the figure.
    recs: 2D array of receiver coordinates for scatter plot.
    sous: 2D array of source coordinates for scatter plot.
    **kwargs: Additional keyword arguments for imshow.
    """
    
    # Create subplots with shared y-axis
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # First plot
    im0 = axs[0].imshow(m1, aspect='auto', vmin=vmin[0], vmax=vmax[0], **kwargs)
    cbar0 = plt.colorbar(im0, ax=axs[0])
    cbar0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar0.set_label('km/s')
    axs[0].set_xlabel('nx')  # Set x-label for the first plot
    
    if recs is not None:
        axs[0].scatter(recs[:, 0], recs[:, 1], c='k')
        
    if sous is not None:
        axs[0].scatter(sous[:, 0], sous[:, 1], c='r')
    
    # Second plot
    im1 = axs[1].imshow(m2, aspect='auto', vmin=vmin[1], vmax=vmax[1], **kwargs)
    cbar1 = plt.colorbar(im1, ax=axs[1])
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar1.set_label('km/s')
    axs[1].set_xlabel('nx')  # Set x-label for the second plot
    
    if recs is not None:
        axs[1].scatter(recs[:, 0], recs[:, 1], c='k')
        
    if sous is not None:
        axs[1].scatter(sous[:, 0], sous[:, 1], c='r')
    
    # Third plot
    im2 = axs[2].imshow(m3, aspect='auto', vmin=vmin[2], vmax=vmax[2], **kwargs)
    cbar2 = plt.colorbar(im2, ax=axs[2])
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar2.set_label('g/cc')
    axs[2].set_xlabel('nx')  # Set x-label for the third plot
    
    if recs is not None:
        axs[2].scatter(recs[:, 0], recs[:, 1], c='k')
        
    if sous is not None:
        axs[2].scatter(sous[:, 0], sous[:, 1], c='r')
        
    # Set common y-axis label for all subplots
    axs[0].set_ylabel('nz')

    # Save the figure if fig_name is provided
    if fig_name is not None: 
        plt.savefig(save_dir + fig_name)
        
    plt.show()
    
def plot_slices(array, ids=None, size=5, width=1, ax=None, fig=None, recs=None, sous=None, fig_name=None, cbar_label='km/s', horizons=None, bounds=None, **kwargs):
    """
    Generate image slices from a 3D array.
    """
    if ids is None:
        ids = [array.shape[2]//2, array.shape[1]//2, array.shape[0]//2]

    if bounds is None:
        x = np.arange(array.shape[2])
        y = np.arange(array.shape[1])
        z = np.arange(array.shape[0])
    else:
        x = np.linspace(bounds[0], bounds[1], array.shape[2])
        y = np.linspace(bounds[0], bounds[2], array.shape[1])
        z = np.linspace(bounds[0], bounds[3], array.shape[0])
   
    
    grids = [x,y,z]

    if fig is None:
        fig = plt.figure(figsize=(1.5*size, size))
        
    if ax:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(size, size, subplot_spec=ax, wspace=0.05, hspace=0.05)
    else:
        gs = matplotlib.gridspec.GridSpec(size, size, wspace=0.3, hspace=0.3)

    if fig is None:
        ax = fig.add_subplot(gs[0:(size-width), 0:(size-width)])
    
    # Upper left image
    ax0 = fig.add_subplot(gs[0:(size-width-2), 0:(size-width-2)])

    if bounds is None:
        im0 = ax0.imshow(array[ids[2]], **kwargs)
    else:
        im0 = ax0.imshow(array[ids[2]], extent=[bounds[0], bounds[1], bounds[2], bounds[0]], **kwargs)
    if horizons is not None:
        ax0.imshow(horizons[ids[2]], cmap='jet')
    ax0.hlines(grids[1][ids[1]],xmin=grids[0].min(),xmax=grids[0].max(), linestyles='--', color='black')
    ax0.text(grids[0].max(), grids[1][ids[1]], 'X1', horizontalalignment='right')
    ax0.text(grids[0].min(), grids[1][ids[1]], 'X0')
    ax0.text(grids[0][ids[0]], grids[1].min(), 'Y0', verticalalignment='top')
    ax0.text(grids[0][ids[0]], grids[1].max(), 'Y1')
    ax0.vlines(grids[0][ids[0]],ymin=grids[1].min(),ymax=grids[1].max(), linestyles='--', color='black')
    
    if bounds is not None:
        ax0.set_xlabel('km')
        ax0.set_ylabel('km')
    else:
        ax0.set_xlabel('nx')
        ax0.set_ylabel('ny')
    ax0.xaxis.set_ticks_position('top')
    ax0.xaxis.set_label_position('top')

    # Right image
    ax1 = fig.add_subplot(gs[0:(size-width-2), (size-width-2):])
    if bounds is None:
        ax1.imshow(array[:, :, ids[0]].T, **kwargs)
    else:
        ax1.imshow(array[:, :, ids[0]].T, extent=[bounds[0], bounds[3], bounds[1], bounds[0]], **kwargs)
    if horizons is not None:
        ax1.imshow(horizons[:, :, ids[0]], cmap='jet')
    ax1.hlines(grids[1][ids[1]],xmin=grids[2].min(),xmax=grids[2].max(), linestyles='--', color='black')
    ax1.vlines(grids[2][ids[2]],ymin=grids[1].min(),ymax=grids[1].max(), linestyles='--', color='black')
    ax1.text(grids[2][ids[2]], grids[1].max(), 'Y1', horizontalalignment='left', verticalalignment='bottom')
    ax1.text(grids[2][ids[2]], grids[1].min(), 'Y0', verticalalignment='top')
    ax1.yaxis.set_ticks_position('right')
    # ax1.xaxis.set_ticks_position('top')
    
    # Bottom image
    ax2 = fig.add_subplot(gs[(size-width-2):, 0:(size-width-2)])
    
    if bounds is None:
        ax2.imshow(array[:, ids[1], :], **kwargs)
    else:
        ax2.imshow(array[:, ids[1], :], extent=[bounds[0], bounds[1], bounds[3], bounds[0]], **kwargs)
    if horizons is not None:
        ax2.imshow(horizons[:, ids[1], :], cmap='jet')
    ax2.hlines(grids[2][ids[2]],xmin=grids[0].min(),xmax=grids[0].max(), linestyles='--', color='black')
    ax2.vlines(grids[0][ids[0]],ymin=grids[2].min(),ymax=grids[2].max(), linestyles='--', color='black')
    ax2.text(grids[0].max(), grids[2][ids[2]], 'X1', horizontalalignment='right', verticalalignment='top')
    ax2.text(grids[0].min(), grids[2][ids[2]], 'X0', verticalalignment='top')
    if bounds is not None:
        ax2.set_ylabel('km')
    else:
        ax2.set_ylabel('nz')
    

    # Add colorbar for the first image
    cbar = fig.colorbar(im0, ax=[ax0, ax1, ax2], orientation='vertical', fraction=0.05, pad=0.075)
    cbar.set_label(cbar_label, rotation=270-180, labelpad=15)
    
    if recs is not None:
        ax0.scatter(recs[0], recs[1], c='yellow', marker='v')
        
    if sous is not None:
        ax0.scatter(sous[0], sous[1], c='black', marker='*')
        
    if fig_name is not None:
        plt.savefig(fig_name)
        
    plt.show()