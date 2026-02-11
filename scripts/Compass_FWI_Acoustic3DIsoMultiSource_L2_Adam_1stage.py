"""
Description:
Acoustic 3D Multi-source Isotropic Full Waveform Inversion on the Compass model.
 
Run as: 
Adjust the XXX_NUM_THREADS below such that the product between this number on number of processes ("-n") does not exceed logical cores!

export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=56; export MKL_NUM_THREADS=56; export NUMBA_NUM_THREADS=56; mpiexec -n 2 python Compass_FWI_Acoustic3DIsoMultiSource_L2_Adam_1stage.py

Contributors:
Originally adapted from https://github.com/DIG-Kaust/Devito-fwi version 0.1.0
Modified by Mohammad Hasyim Taufik (taufikmh)
"""

import os
import numpy as np
import time
import torch
import gstools as gs
import yaml
import inspect
import h5py
import matplotlib.pyplot as plt
import os, segyio, json

# Comment out if LateX is not present
# plt.style.use('../asset/plots.mplstyle')

from mpi4py import MPI
from tqdm import tqdm
from matplotlib import pyplot as plt
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from devito import configuration
from argparse import ArgumentParser

# See install.sh to install the following package
import sys
sys.path.append('../src/')

# See install.sh to install the following package
from guidedfwi.plots import plot_slices
from guidedfwi.utils import clear_devito_cache, highpass_filter, log_experiment
from guidedfwi.acoustic3disowrapper import AcousticWave3D
from guidedfwi.loss import L2MultiSource
from guidedfwi.postprocessing import PostProcessX
from guidedfwi.torchoperator import TorchOperator
from guidedfwi.encoding import Encoding

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# Callback to track model error
def fwi_callback(vp_i, vp, vp_error, iter, save_file=True, results_folder='./', **kwargs):
    vp_error.append(np.linalg.norm((vp_i.reshape(-1) - vp.reshape(-1))/vp.reshape(-1)))
    
    if rank == 0:
        plot_slices(vp_i.T,
            aspect='auto', interpolation='bicubic', 
            size=8, 
            cmap='rainbow', fig_name=os.path.join(results_folder, 'InvertedVP_'+str(iter)+'.png'), **kwargs)

        if save_file:
            np.save(os.path.join(results_folder, 'InvertedVp_'+str(iter)+'.npy'), vp_i.reshape(vp.shape))
            
def load_selected_sources(ids, dirname=".", prefix="source_", suffix=".npy"):
    memmaps = []
    for i in ids:
        path = f"{dirname}/{prefix}{i}{suffix}"
        arr = np.load(path, mmap_mode="r")  # memory-map, shape (T, R)
        memmaps.append(arr)
    # Stack into a view; this will not copy all data immediately if you operate carefully
    stacked = np.stack(memmaps, axis=0)  # shape (len(ids), T, R)
    return stacked

configuration['log-level'] = 'ERROR'

clear_devito_cache()

def main():

    if rank == 0:
        print(f'Distributed FWI ({size} ranks)')

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
        'nt':5000,                      'dt':0.004,                                                             'ot':0,
        'freq':args.data_frequency,
        'niter':50,  
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

    # Modelling parameters
    shape = (par['nx'], par['ny'], par['nz'])
    spacing = (par['dx'], par['dy'], par['dz'])
    origin = (par['ox'], par['oy'], par['oz'])
    space_order = 12
    nbl = 20

    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Path to save figures
    results_folder = (
        '../results/Compass_FWI_Acoustic3DIsoMultiSource_'+str(run_id)
    )

    if rank==0:
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
            
        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        
        # Log experiment
        log_experiment(
            results_folder, script_path=os.path.abspath(inspect.getfile(inspect.currentframe())), run_id=run_id
        )
            
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
            multisource_batch_size=par['ns'],
            source_encoding=encoding_type, #'random_polarity', #None,
            base_comm=comm,
            encoding_params=encoding_params
            )
        if rank == 0:
            print('Compute gradient...')
            
        postproc = PostProcessX(scaling=1, mask=mask)
        loss, direction = ainv.loss_grad(vp_init, postprocess=postproc.apply)

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
        # Inversion
        ##################################################################

        # Inversion parameters
        loss_history = []
        vp_error_history = []
        maxiter = par['niter']

        # SVGD Parameters
        vp_init_torch = torch.from_numpy(np.copy(vp_init))
        vp_init_torch.requires_grad = True
        optimizer = torch.optim.Adam([vp_init_torch], lr=lrs[fi])
        scheduler = None

        if rank == 0:
            print('Run FWI for frequency '+str(freq)+'Hz...')
            tstart = time.time()

        for iter in tqdm(range(maxiter)):
            
            optimizer.zero_grad()
            
            ##################################################################
            # FWI Step
            ##################################################################

            # Compute gradient
            loss = ainv_torch.apply(vp_init_torch)
            loss.backward()
            
            # Update model
            optimizer.step()
            # scheduler_vp.step(loss)

            # Store history of loss and model error
            loss_history.append(loss.item())

            # Callback for plotting and saving
            if ((iter==0) or ((iter+1) % 2 == 0) or (iter+1 == maxiter)) and (rank==0):
                
                fwi_callback(
                    vp_init_torch.clone().detach().numpy().reshape(vp_true.shape), 
                    vp_error=vp_error_history,
                    vp=vp_true, iter=total_iter, results_folder=results_folder, 
                    vmin=m_min, vmax=m_max, bounds=[0,x.max(),y.max(),z.max()]
                )
                
                print(f'Frequency {freq}, Iteration {iter}, Loss {loss_history[-1]}')
                
            if torch.isnan(vp_init_torch).any():
                print("NaN detected in vp_inits")
                break

            if rank == 0:    
                plt.figure(figsize=(10, 5))
                plt.semilogy(loss_history, 'k')
                plt.savefig(os.path.join(results_folder, 'DataError_'+str(freq)+'freq.png'))
                
                plt.figure(figsize=(10, 5))
                plt.semilogy(vp_error_history, 'b')
                plt.savefig(os.path.join(results_folder, 'ModelError_'+str(freq)+'freq.png'))
            
            total_iter += 1
        
        # Use the inverted moduli from current frequency as the next initial moduli
        vp_init = vp_init_torch.detach().clone().numpy()
        np.save(os.path.join(results_folder, 'Loss_freq'+str(freq)+'.npy'), np.array(loss_history))
            
    if rank == 0:    
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
        
if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "--total_sources",
        type=int,
        default=100,
    )
    
    parser.add_argument(
        "--data_frequency",
        type=str,
        default='4',
    )
    
    parser.add_argument(
        "--total_receivers",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--velocity_size",
        type=int,
        default=128,
    )
    
    args = parser.parse_args()
    
    main()