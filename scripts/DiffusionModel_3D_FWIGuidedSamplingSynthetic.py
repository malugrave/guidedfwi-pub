"""
Description:
FWI-guided 3D diffusion model sampling.
 
Run as: 
python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py

For multi-GPU:
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=13; export MKL_NUM_THREADS=13; export NUMBA_NUM_THREADS=13; CUDA_VISIBLE_DEVICES=0 mpiexec -n 4 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py

Contributors:
Originally adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

# pip3 install git+https://github.com/lucidrains/denoising-diffusion-pytorch.git

import torch
import torch.nn.functional as F
import numpy as np
import os
import inspect
import matplotlib.pyplot as plt
import time
import h5py
import json

from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from tqdm.auto import tqdm
from ema_pytorch import EMA
from argparse import ArgumentParser
from mpi4py import MPI
from tqdm import tqdm
from matplotlib import pyplot as plt
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from devito import configuration
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

from guidedfwi.diffusion3d import Trainer
from guidedfwi.utils import extract_cubes, combine_cubes, numpy_to_cuda, cuda_to_numpy, set_seed, log_experiment, denormalize_from_minusone_and_one, normalize_to_minusone_and_one, clear_devito_cache, highpass_filter
from guidedfwi.plots import plot_slices, plot_modulus, plot_diffusion_evolution
from guidedfwi.acoustic3disowrapper import AcousticWave3D
from guidedfwi.loss import L2MultiSource
from guidedfwi.postprocessing import PostProcessX
from guidedfwi.torchoperator import TorchOperator
from guidedfwi.encoding import Encoding

# # Comment out if LateX is not present
# plt.style.use('../asset/plots.mplstyle')
    
# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'ERROR'
clear_devito_cache()

from mpi4py import MPI
import torch
from tqdm.auto import tqdm

def load_selected_sources(ids, dirname=".", prefix="source_", suffix=".npy"):
    memmaps = []
    for i in ids:
        path = f"{dirname}/{prefix}{i}{suffix}"
        arr = np.load(path, mmap_mode="r")  # memory-map, shape (T, R)
        memmaps.append(arr)
    # Stack into a view; this will not copy all data immediately if you operate carefully
    stacked = np.stack(memmaps, axis=0)  # shape (len(ids), T, R)
    return stacked

def p_sample_loop_with_fwi_guidance(
    trainer,
    shape,
    forward_propagator,
    observed_data,
    inject_every=2,
    use_fwi_guidance=False,
    eta=1e-3,
    self_condition=False,
    return_all_timesteps=False,
    t_start=0,
    x_smooth=None,
    clip_input=False,
    vmin=1500.0,
    vmax=4500.0,
    fwi_loop=3,
    run_fwi_under=1000,
    cube_size=0,
    stride=None,
    debug=False,
    figure_path='./',
    mask=None
):
    """
    3D diffusion sampling with Devito‐FWI guidance, fully MPI‐aware.

    Parameters
    ----------
    trainer : object
        Holds `trainer.ema.ema_model`: the diffusion network on GPU (only for rank 0).
    shape : tuple (B, C, D, H, W)
        Diffusion latent shape.
    forward_propagator : TorchOperator
        Wraps AcousticWave3D.loss_grad, does its own MPI all‐reduce.
    observed_data : torch.Tensor
        Ground‐truth seismograms (ignored inside this function but shown for API consistency).
    inject_every : int
        Inject FWI every this many reverse timesteps.
    use_fwi_guidance : bool
        Enable FWI gradient injection.
    eta : float
        Step size for FWI gradient updates.
    self_condition : bool
        Use self‐conditioning in p_sample.
    return_all_timesteps : bool
        If True, returns the full trajectory of samples.
    t_start : int
        Start the reverse loop at t = num_timesteps−t_start.
    x_smooth : torch.Tensor or None
        Optional smoother initializer.
    clip_input : bool
        Clamp sample to [−1,1] before each p_sample.
    vmin, vmax : float
        Bounds for denormalizing latent → velocity.
    fwi_loop : int
        Number of inner gradient steps per injection.
    run_fwi_under : int
        Only inject when t < run_fwi_under.

    Returns
    -------
    torch.Tensor or None
        On rank 0: the final (or trajectory) sample(s), unnormalized to [0,1].
        On other ranks: None.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Only rank 0 actually holds the diffusion model
    device = next(trainer.model.denoise_fn.parameters()).device if rank == 0 else None
        
    # Initialize the sample *only* on rank 0
    if rank == 0:
        if x_smooth is not None:
            img = 1e-16 * torch.randn(shape, device=device) + x_smooth.to(device)
        else:
            img = torch.randn(shape, device=device)
        imgs = [img]
        x_start = None
    else:
        imgs = None
        x_start = None

    diffusion = trainer.model if rank == 0 else None
    total_steps = diffusion.num_timesteps - t_start if rank == 0 else None
    
    total_steps = comm.bcast(total_steps, root=0)
    
    # Rank 0 extracts & broadcasts vp_np
    if rank == 0:
        vp_np  = denormalize_from_minusone_and_one(img[0,0].clone().detach(), vmin, vmax).cpu().numpy()
    else:
        vp_np = None
        
    # broadcast updated img (and implicitly vp_param) to other ranks
    img = comm.bcast(img if rank == 0 else None, root=0)
    vp = comm.bcast(vp_np if rank == 0 else None, root=0)

    # Every rank builds vp and computes its gradient
    vp = torch.from_numpy(vp).requires_grad_()
    
    optimizer = torch.optim.Adam([{'params': [vp], 'lr': 5e-2}])

    for i in tqdm(reversed(range(0, total_steps)), desc='sampling loop time step', total=total_steps):
        
        # On rank 0, do the diffusion step
        if rank == 0:
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            if clip_input:
                img = img.clamp(-1, 1)
            self_cond = x_start if self_condition else None
            # img, x_start = diffusion.p_sample(img, t, self_cond)
            
            # with torch.no_grad():
            #     img, _, _ = diffusion.p_mean_variance(x = img.view(shape), t = t, clip_denoised = True, cond = None, cond_scale = 1.)
            
            # cube‑batching: split full volume into smaller blocks
            if cube_size > 0:
                # bring current sample to CPU numpy
                vol_np = img[0,0].detach().cpu().numpy()
                noise_np = torch.randn_like(img[0,0].detach().cpu()).numpy()
                # extract overlapping cubes of size cube_size with given stride
                cubes = extract_cubes(vol_np.astype(np.float64), (cube_size,)*3, (stride,)*3)
                noises = extract_cubes(noise_np.astype(np.float64), (cube_size,)*3, (stride,)*3)
                out_cubes = []
                out_xstarts = []

                # patches = torch.cat(patch_outputs, dim=0)  # (B, N, C, dx, dy)
                # process each cube independently
                for idx_t, cube_np in enumerate(cubes):
                    cube_t = torch.from_numpy(cube_np).unsqueeze(0).unsqueeze(0).to(device)
                    noise_t = torch.from_numpy(noises[idx_t]).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        model_mean, _, model_log_variance = diffusion.p_mean_variance(
                            cube_t.to(torch.float32), t = t, clip_denoised = True, cond = None, cond_scale = 1.
                        )
                        # no noise when t == 0
                        nonzero_mask = (1 - (t == 0).float()).reshape(cube_t.shape[0], *((1,) * (len(cube_t.shape) - 1)))
                        cube_out = model_mean # + nonzero_mask * (0.5 * model_log_variance).exp() * noise_t.to(torch.float32)

                    out_cubes.append(
                        cube_out.squeeze(0).squeeze(0).detach().cpu().numpy()
                    )

                # reassemble full volume from processed cubes
                vol_np = combine_cubes(out_cubes, shape[2:], (cube_size,)*3, (stride,)*3)
                
                # back to torch tensors
                img = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device).to(torch.float32)
        
            else:
                # single‐volume reverse step
                with torch.no_grad():
                    img, _, _ = diffusion.p_mean_variance(x = img.to(torch.float32).view(shape), t = t, clip_denoised = True, cond = None, cond_scale = 1.)
                    
            
            # copy the new diffusion sample into vp_param.data
            vp.data.copy_(
                denormalize_from_minusone_and_one(img[0,0], vmin, vmax)
            )
                    
            # if debug:
                            
            #     plot_slices(denormalize_from_minusone_and_one(img[0,0], vmin, vmax).detach().cpu().numpy().T, 
            #         aspect='auto', interpolation='bicubic', cmap='rainbow', 
            #         vmin=vmin, vmax=vmax, 
            #         size=8,
            #         fig_name=os.path.join(figure_path, 'Sample_'+str(i)+'.png')
            #     )
            #     _, e_max = np.percentile(denormalize_from_minusone_and_one(img[0,0], vmin, vmax).detach().cpu().numpy().T-denormalize_from_minusone_and_one(x_smooth[0,0], vmin, vmax).detach().cpu().numpy().T, [2,98]) 
            #     plot_slices(denormalize_from_minusone_and_one(img[0,0], vmin, vmax).detach().cpu().numpy().T-denormalize_from_minusone_and_one(x_smooth[0,0], vmin, vmax).detach().cpu().numpy().T, 
            #         aspect='auto', interpolation='bicubic', cmap='rainbow', 
            #         vmin=-e_max, vmax=e_max, 
            #         size=8,
            #         fig_name=os.path.join(figure_path, 'Update_'+str(i)+'.png')
            #     )
            
        # # broadcast updated img (and implicitly vp_param) to other ranks
        # img = comm.bcast(img, root=0)
        # vp = comm.bcast(vp, root=0)

        # FWI guidance (all ranks)
        if use_fwi_guidance and (i % inject_every == 0) and (i < run_fwi_under):
                
            for epoch in range(fwi_loop):
                loss_fwi = forward_propagator.apply(vp)
                optimizer.zero_grad()
                loss_fwi.backward()
                # grad_np = vp.grad.detach().cpu().numpy()
                optimizer.step()
                
                if mask is not None:
                    vp[mask==0].data = torch.tensor(vmin).float()
                    
                if (rank == 0) and debug:
                    
                    _, g_max = np.percentile(vp.grad.detach().cpu().numpy(), [2,98]) 
                    
                    plot_slices(vp.detach().cpu().numpy().T, 
                        aspect='auto', interpolation='bicubic', cmap='rainbow', 
                        vmin=vmin, vmax=vmax, 
                        size=8,
                        fig_name=os.path.join(figure_path, 'Velocity_'+str(epoch)+'_'+str(i)+'.png')
                    )
                    plot_slices(vp.grad.detach().cpu().numpy().T, 
                        aspect='auto', interpolation='bicubic', cmap='seismic', 
                        vmin=-g_max, vmax=g_max, 
                        size=8,
                        fig_name=os.path.join(figure_path, 'Gradient_'+str(epoch)+'_'+str(i)+'.png')
                    )
                
                img = normalize_to_minusone_and_one(vp.detach().clone().unsqueeze(0).unsqueeze(0), vmin, vmax)

        # Collect trajectory on rank 0
        if rank == 0:
            imgs.append(denormalize_from_minusone_and_one(img, vmin, vmax))

    if rank == 0:
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        else:
            return imgs[-1]
    else:
        return None

def main():
    
    ##################################################################
    # Experiment logging
    ##################################################################

    args = parser.parse_args()
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    results_folder = (
        '../results/DiffusionModel_3D_FWIGuidedSamplingSynthetic_'+str(run_id)
    )
        
    ##################################################################
    # Diffusion model initialization and training
    ##################################################################

    # The model was trained using 7 different classes as labels but they are not used for sampling
    set_seed(12315019)
    
    if rank == 0:

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
            folder=None,
            train_batch_size=2,
            train_lr=8e-6,
            train_num_steps=200000,              
            gradient_accumulate_every=2,   
            ema_decay=0.995,               
            amp=False,                     
            save_and_sample_every=2000,
            num_sample_rows=2,
            results_folder='../results/DiffusionModel_3D_Training_20250728-165119'
        )

        # Load model
        trainer.load(18)
        
    else:
        model, diffusion, trainer = None, None, None
    
    ##################################################################
    # Parameters
    ##################################################################

    # Model and aquisition parameters
    par = {
        'nx':args.velocity_size,        'dx':80/1e3,                                                            'ox':0,
        'ny':args.velocity_size,        'dy':80/1e3,                                                            'oy':0,
        'nz':args.velocity_size,        'dz':40/1e3,                                                            'oz':0,
        'ns':args.total_sources,        'ds':80*(args.velocity_size-1)/(np.sqrt(args.total_sources)-1)/1e3,     'os':0,  'sz':0,
        'nr':args.total_receivers,      'dr':80*(args.velocity_size-1)/(np.sqrt(args.total_receivers)-1)/1e3,   'or':0,  'rz':0,
        'nt':5000 if args.velocity_size == 384 else 4000,                      
        'dt':0.004 if args.velocity_size == 384 else 0.003,                                                             
        'ot':0,
        'freq':args.data_frequency,
        'niter':1,  
        'num_particles':10,
        'lr':'5e-2',
        'factor':24,
        'sigma':10
    }

    datapath = (
        '../data/seismic_compass_'
        +str(par['ns'])+'sou_'+str(par['nr'])+'rec_'+par['freq']+'freq_'
        +str(par['dx'])+'dx_'+str(par['dy'])+'dy_'+str(par['dz'])+'dz_'
        +str(par['nt'])+'nt_'+str(par['dt'])+'dt_'
        +str(par['nx'])+'nx_'+str(par['ny'])+'ny_'+str(par['nz'])+'nz'
    )
    
    tokenpath = datapath + '/completion_token.json'
    
    if rank == 0:
        print(datapath)
        
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
            
        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        
        # Log experiment
        log_experiment(
            results_folder, script_path=os.path.abspath(inspect.getfile(inspect.currentframe())), run_id=run_id
        )

    # Modelling parameters
    shape = (par['nx'], par['ny'], par['nz'])
    spacing = (par['dx'], par['dy'], par['dz'])
    origin = (par['ox'], par['oy'], par['oz'])
    space_order = 12
    nbl = 20
            
    ##################################################################
    # Acquisition set-up
    ##################################################################

    # Sampling frequency
    fs = 1 / par['dt'] 

    # Axes
    x = np.arange(par['nx']) * par['dx'] + par['ox']
    y = np.arange(par['ny']) * par['dy'] + par['oy']
    z = np.arange(par['nz']) * par['dz'] + par['oz']
    t = np.arange(par['nt']) * par['dt'] + par['ot']
    tmax = t[-1] # in s

    # Grid
    X,Y,Z = np.meshgrid(x,y,z)

    # Sources
    xs, ys = np.meshgrid(np.arange(np.sqrt(par['ns'])) * par['ds'] + par['os'], np.arange(np.sqrt(par['ns'])) * par['ds'] + par['os'])
    x_s = np.zeros((par['ns'], 3))
    x_s[:, 0] = xs.reshape(-1)
    x_s[:, 1] = ys.reshape(-1)
    x_s[:, 2] = par['sz']

    # Receivers
    xr, yr = np.meshgrid(np.arange(np.sqrt(par['nr'])) * par['dr'] + par['or'], np.arange(np.sqrt(par['nr'])) * par['dr'] + par['or'])
    x_r = np.zeros((par['nr'], 3))
    x_r[:, 0] = xr.reshape(-1)
    x_r[:, 1] = yr.reshape(-1)
    x_r[:, 2] = par['rz']

    # Randomize the source locations
    nsrc = par['ns']

    # on rank 0, generate a random permutation of [0..nsrc-1]
    if rank == 0:
        perm = np.random.permutation(nsrc)
    else:
        perm = None

    # broadcast that perm to all ranks
    perm = comm.bcast(perm, root=0)

    # split it into `size` roughly‐equal blocks and pick ours
    blocks = np.array_split(perm, size)
    local_ids = blocks[rank]

    # slice your shot coords accordingly
    xs_local = x_s[local_ids, 0]
    ys_local = x_s[local_ids, 1]
    zs_local = x_s[local_ids, 2]

    ##################################################################
    # Velocity model
    ##################################################################

    # Load the true model
    vp_true = resize(np.load('../data/velocities/compass_vp3d.npy'), (par['nx'], par['ny'], par['nz']))

    m_min, m_max = vp_true.min(), vp_true.max()

    mask = np.ones_like(vp_true)
    mask[vp_true<1.51] = 0

    # Initial model for FWI by smoothing the true model
    vp_init = gaussian_filter(vp_true, sigma=[par['sigma'], par['sigma'], par['sigma']])

    # Replace water velocity
    vp_init[mask==0] = 1.5 
    vp_true[mask==0] = 1.5 

    if rank == 0:
        plot_slices(mask.T, 
            aspect='auto', interpolation='bicubic', cmap='rainbow', 
            recs=[x_r[:, 0], x_r[:, 1], x_r[:, 2]], 
            sous=[x_s[:, 0], x_s[:, 1], x_s[:, 2]],  
            bounds=[0,x.max(),y.max(),z.max()],
            vmin=0,vmax=1, 
            size=8,
            fig_name=os.path.join(results_folder, 'Mask.png')
        )
        plot_slices(vp_true.T, 
            aspect='auto', interpolation='bicubic', cmap='rainbow', 
            vmin=m_min,vmax=m_max, 
            bounds=[0,x.max(),y.max(),z.max()],
            size=8,
            fig_name=os.path.join(results_folder, 'TrueVp.png')
        )
        plot_slices(vp_init.T, 
            aspect='auto', interpolation='bicubic', cmap='rainbow', 
            vmin=m_min,vmax=m_max, 
            bounds=[0,x.max(),y.max(),z.max()],
            size=8,
            fig_name=os.path.join(results_folder, 'InitialVp.png')
        )

    ##################################################################
    # FWI
    ##################################################################
        
    freqs = [int(x) for x in par['freq'].split(',')]
    lrs = [float(x) for x in par['lr'].split(',')]

    total_iter = 0

    for fi, freq in enumerate(freqs):
        
        if rank == 0:
            plot_slices(vp_init.T, 
                aspect='auto', interpolation='bicubic', cmap='rainbow', 
                vmin=m_min,vmax=m_max, 
                bounds=[0,x.max(),y.max(),z.max()],
                size=8,
                fig_name=os.path.join(results_folder, 'InitialVp_'+str(freq)+'freq.png')
            )

        ##################################################################
        # Data
        ##################################################################

        # Choose how to split sources to ranks
        ns_rank = local_split((par['ns'], ), MPI.COMM_WORLD, Partition.SCATTER, 0)
        ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
        isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
        isend_rank = np.cumsum(ns_ranks)[rank]

        print(f'After randomize rank: {rank}, ns: {ns_rank}, indices: {local_ids}')

        # Define modelling engine
        amod = AcousticWave3D(shape, origin, spacing, 
            # x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1],  x_s[isin_rank:isend_rank, 2],
            xs_local, ys_local, zs_local,
            x_r[:, 0], x_r[:, 1], x_r[:, 2],
            0., tmax,  
            vp=vp_true,
            src_type="Ricker", f0=freq,
            space_order=space_order, nbl=nbl,
            factor=par['factor'],
            base_comm=comm
        )
        
        if rank == 0:
            dobs_mid, _ = amod._mod_oneshot(
                amod._create_model(
                    amod.shape, amod.origin, amod.spacing, 
                    amod.vp, amod.space_order, amod.nbl, amod.fs), 
                isend_rank-1, None
            )
            d_min, d_max = np.percentile(dobs_mid, (5,95))

            plot_slices(
                np.swapaxes(dobs_mid.reshape(-1, int(np.sqrt(par['nr'])), int(np.sqrt(par['nr']))), 0, 2).T, 
                aspect='auto', interpolation='bicubic', cmap='gray', 
                vmin=-d_max,vmax=d_max, 
                size=8,
                fig_name=os.path.join(results_folder, 'Dobs_'+str(freq)+'freq.png')
            )

        # Remove frequency below 3 Hz
        model, geometry = amod.model_and_geometry()
        wav = highpass_filter(geometry.src.wavelet, dt=geometry.dt/1e3, cutoff_freq=0.5)
        
        # Define modelling engine
        amod = AcousticWave3D(shape, origin, spacing, 
            xs_local, ys_local, zs_local,
            x_r[:, 0], x_r[:, 1], x_r[:, 2],
            0., tmax,  
            vp=vp_true,
            src_type="Ricker", f0=freq,
            space_order=space_order, nbl=nbl,
            factor=par['factor'],
            wav=wav,
            base_comm=comm
        )
        
        # Model data
        if rank == 0:
            print('Model data...')
            
        if not os.path.exists(tokenpath):
            dobs, _ = amod.mod_allshots()
            
            for idx, source_idx in enumerate(local_ids):
                print('Saving source:', source_idx)
                
                np.save(datapath+'/source_'+str(source_idx)+'.npy', dobs[idx])
            
            # Wait for other ranks (if used) to finish generating data before creating the completion token  
            comm.Barrier()
            
            if rank==0:
                token_dict = {
                    "created_by_rank": 0,
                    "created_using": os.path.join(results_folder, f"{run_id}.log"),
                }
                # Atomic write
                with open(tokenpath, "w") as f:
                    json.dump(token_dict, f)
                    f.flush()
                    os.fsync(f.fileno())
        else:
            dobs = load_selected_sources(local_ids, dirname=datapath, prefix="source_", suffix=".npy")

            
        # Add noise to data``
        sigman = 0 #1e-6
        dobs = dobs + np.random.normal(0, sigman, dobs.shape)

        ##################################################################
        # Gradient scaling
        ##################################################################
        
        encoding_type = None #'random_time_delay'
        encoding_params = {'nt': geometry.nt, 'delay': 0.4}
        encoding = Encoding(encoding_type, xs_local.shape[0], encoding_params) if encoding_type is not None else None
        
        # Define loss     
        l2loss = L2MultiSource(dobs, encoder=encoding)

        ainv = AcousticWave3D(shape, origin, spacing, 
            # x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1],  x_s[isin_rank:isend_rank, 2],
            xs_local, ys_local, zs_local,
            x_r[:, 0], x_r[:, 1], x_r[:, 2],
            0., tmax,
            vprange=(vp_true.min(), vp_true.max()),
            src_type="Ricker", f0=freq,
            space_order=space_order, nbl=nbl,
            factor=par['factor'],
            loss=l2loss, clearcache=True,
            multisource_batch_size=par['ns']//9,
            source_encoding=encoding_type, #'random_polarity', #None,
            base_comm=comm,
            encoding_params=encoding_params
            )
        if rank == 0:
            print('Compute gradient...')
            
        postproc = PostProcessX(scaling=1, mask=mask)
        
        _, direction = ainv.loss_grad(vp_init, postprocess=postproc.apply)

        scaling = abs(direction).max()

        _, g_max = np.percentile(direction.T/scaling, [2,98]) 

        if rank == 0:
            plot_slices(direction.reshape(vp_true.shape).T/scaling, aspect='auto', 
                interpolation='bicubic', cmap='seismic', 
                vmin=-g_max, vmax=g_max,
                bounds=[0,x.max(),y.max(),z.max()],
                size=8,
                fig_name=os.path.join(results_folder, 'InitialGradient_'+str(freq)+'freq.png'))
            
            # Save gradient
            np.save(os.path.join(results_folder, 'InitialGradient_'+str(freq)+'freq.npy'), direction/scaling)

        # Compute first gradient and find scaling
        postproc = PostProcessX(scaling=scaling, mask=mask, scale_loss=True)
        ainv_torch = TorchOperator(ainv.loss_grad, kwargs_prop=dict(postprocess=postproc.apply))
        
    ##################################################################
    # FWI-guided sampling
    ##################################################################

    samples = []
    
    for s in range(args.num_samples):

        # Call your 3D diffusion sampler
        sample = p_sample_loop_with_fwi_guidance(
            trainer,
            shape=(1,1,par['nz'],par['ny'],par['nx']),
            forward_propagator=ainv_torch,
            observed_data=torch.from_numpy(dobs),
            use_fwi_guidance=True,
            x_smooth=normalize_to_minusone_and_one(
                torch.from_numpy(vp_init).clone().unsqueeze(0).unsqueeze(0), 
                vp_true.min(), vp_true.max()
            ),
            eta=1e-4,
            inject_every=2,
            fwi_loop=1,
            run_fwi_under=1000,
            t_start=900,
            vmin=vp_true.min(), vmax=vp_true.max(),
            cube_size=64, 
            stride=64,
            debug=True,
            figure_path=results_folder,
            mask=mask
        )

        if rank == 0:
            
            plot_slices(
                sample[0, 0].detach().cpu().numpy().T, 
                aspect='auto', cmap='rainbow', size=8, vmin=m_min, vmax=m_max,
                fig_name=f"{results_folder}/sample_"+str(s)+".png"
            )

            samples.append(sample)

            # Compute statistics
            stacked_samples = torch.cat(samples, dim=0)[:, 0]
            
            # Save diffusion samples
            torch.save(stacked_samples.detach().cpu(), results_folder+'/samples_'+str(args.num_samples)+'.pt')
            
            if s > 0:

                # Compute mean and std over sample dimension (dim=0)
                mean = stacked_samples.mean(dim=0)
                std = stacked_samples.std(dim=0)

                plot_slices(
                    mean.detach().cpu().numpy().T, 
                    aspect='auto', cmap='rainbow', size=8, vmin=m_min, vmax=m_max,
                    fig_name=f"{results_folder}/mean_"+str(s)+".png"
                )

                plot_slices(
                    std.detach().cpu().numpy().T, 
                    aspect='auto', cmap='Reds', size=8, vmin=m_min, vmax=m_max,
                    fig_name=f"{results_folder}/std_"+str(s)+".png"
        )

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_frequency",
        type=str,
        default='4',
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default='seg',
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--total_sources",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--total_receivers",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--velocity_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--cube-size",  
        type=int, 
        default=0,
        help="If >0, chop vp_true into cubes of this size and sample each."
    )
    parser.add_argument(
        "--cube-stride",
        type=int, 
        default=None,
        help="Stride for cube extraction (defaults to cube_size)."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./results",
        help="Where to save per‑cube outputs."
    )
    
    args = parser.parse_args()
    
    main()