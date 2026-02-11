"""
Description:
Acoustic 2D Isotropic Full Waveform Inversion on the Compass model.
 
Run as: 
Adjust the XXX_NUM_THREADS below such that the product between this number on number of processes ("-n") does not exceed logical cores!

export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export NUMBA_NUM_THREADS=4; mpiexec -n 16 python Compass_FWI_Acoustic2DIso_L2_Adam_1stage.py

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
import matplotlib.pyplot as plt
import json

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

from mpi4py import MPI
from tqdm import tqdm
from matplotlib import pyplot as plt
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from devito import configuration

# See install.sh to install the following package
import sys
sys.path.append('../src/')

from guidedfwi.plots import plot_modulus
from guidedfwi.utils import clear_devito_cache, highpass_filter, log_experiment
from guidedfwi.acoustic2disowrapper import AcousticWave2D
from guidedfwi.loss import L2
from guidedfwi.postprocessing import PostProcessX
from guidedfwi.torchoperator import TorchOperator

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


# Callback to track model error
def fwi_callback(vp_i, vp, vp_error, iter, save_file=False):
    vp_error.append(np.linalg.norm((vp_i.reshape(-1) - vp.reshape(-1))/vp.reshape(-1)))
    
    if rank == 0:
        plot_modulus(vp_i.T, 
            aspect='auto', interpolation='bicubic', 
            recs=x_r, sous=x_s, 
            extent=[0,x.max(),z.max(),0],
            vmin=m_min,vmax=m_max, 
            cmap='rainbow', fig_name=os.path.join(results_folder, 'InvertedVP_'+str(iter)+'.png'))
        
        if save_file:
            np.save(os.path.join(results_folder, 'InvertedVp_'+str(iter)+'.npy'), vp_i.reshape(vp.shape))

configuration['log-level'] = 'ERROR'

clear_devito_cache()

if rank == 0:
    print(f'Distributed FWI ({size} ranks)')

##################################################################
# Parameters
##################################################################

# Model and aquisition parameters
par = {
    'nx':512,   'dx':16/1e3,     'ox':0,
    'nz':256,   'dz':10/1e3,     'oz':0,
    'ns':64,    'ds':128/1e3,    'os':0,    'sz':10/1e3,
    'nr':128,   'dr':64/1e3,     'or':0,    'rz':250/1e3,
    'nt':2000,  'dt':0.002,      'ot':0,
    'freq':'12',     
    'niter':150,
    'factor':8,  
    'lr':'5e-2',
    'sigma':10 # for 512 previously 8
}

# Modelling parameters
shape = (par['nx'], par['nz'])
spacing = (par['dx'], par['dz'])
origin = (par['ox'], par['oz'])
space_order = 8
nbl = 20

run_id = time.strftime("%Y%m%d-%H%M%S")

# Path to save figures
results_folder = (
    '../results/Compass_FWI_Acoustic2DIso_'+str(run_id)
)

if rank==0:
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
        
    # Log experiment
    log_experiment(
        results_folder, script_path=os.path.abspath(inspect.getfile(inspect.currentframe()), run_id=run_id)
    )
        
##################################################################
# Acquisition set-up
##################################################################

# Sampling frequency
fs = 1 / par['dt'] 

# Axes
x = np.arange(par['nx']) * par['dx'] + par['ox']
z = np.arange(par['nz']) * par['dz'] + par['oz']
t = np.arange(par['nt']) * par['dt'] + par['ot']
tmax = t[-1] # in s

# Source locations in m
x_s = np.zeros((par['ns'], 2))
x_s[:, 0] = np.arange(par['ns']) * par['ds'] + par['os']
x_s[:, 1] = par['sz']

# Receiver locations in m
x_r = np.zeros((par['nr'], 2))
x_r[:, 0] = np.arange(par['nr']) * par['dr'] + par['or']
x_r[:, 1] = par['rz']

##################################################################
# Velocity model
##################################################################

# Load the true model
vel_compass_3d = np.load('../data/velocities/compass_vp3d.npy')
vp_true = resize(vel_compass_3d[:,490,:], (par['nx'], par['nz']))

m_min = vp_true.T.min()
m_max = vp_true.T.max()

mask = np.ones_like(vp_true)
mask[vp_true<1.51] = 0

# Initial model for FWI by smoothing the true model
vp_init = gaussian_filter(vp_true, sigma=[par['sigma'], par['sigma']])

# Replace water velocity
vp_init[mask==0] = 1.5 
vp_true[mask==0] = 1.5 

if rank == 0:
    plot_modulus(mask.T, 
        aspect='auto', interpolation='bicubic', cmap='rainbow', 
        vmin=0,vmax=1, 
        extent=[0,x.max(),z.max(),0],
        fig_name=os.path.join(results_folder, 'Mask.png')
    )
    plot_modulus(vp_true.T, 
        aspect='auto', interpolation='bicubic', cmap='rainbow', 
        recs=x_r, sous=x_s, 
        vmin=m_min,vmax=m_max, 
        extent=[0,x.max(),z.max(),0],
        fig_name=os.path.join(results_folder, 'TrueVp.png')
    )
    plot_modulus(vp_init.T, 
        aspect='auto', interpolation='bicubic', cmap='rainbow', 
        recs=x_r, sous=x_s, 
        extent=[0,x.max(),z.max(),0],
        vmin=m_min,vmax=m_max, 
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
        plot_modulus(
            vp_init.T, 
            aspect='auto', interpolation='bicubic', cmap='rainbow', 
            recs=x_r, sous=x_s, 
            vmin=m_min,vmax=m_max,
            extent=[0,x.max(),z.max(),0],
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
    print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

    # Define modelling engine
    amod = AcousticWave2D(shape, origin, spacing, 
        x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1], 
        x_r[:, 0], x_r[:, 1], 
        0., tmax,  
        vp=vp_true,
        src_type="Ricker", f0=freq,
        space_order=space_order, nbl=nbl,
        base_comm=comm,
        factor=par['factor']
    )

    # Remove frequency below 3 Hz
    model, geometry = amod.model_and_geometry()
    wav = highpass_filter(geometry.src.wavelet, dt=geometry.dt/1e3, cutoff_freq=3)
    
    # Define modelling engine
    amod = AcousticWave2D(shape, origin, spacing, 
        x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1], 
        x_r[:, 0], x_r[:, 1], 
        0., tmax,  
        vp=vp_true,
        src_type="Ricker", f0=freq,
        space_order=space_order, nbl=nbl,
        wav=wav,
        base_comm=comm,
        factor=par['factor']
    )
    
    # Model data
    if rank == 0:
        print('Model data...')
        
    dobs, dtobs = amod.mod_allshots()

    # Add noise to data
    sigman = 0 #1e-6
    dobs = dobs + np.random.normal(0, sigman, dobs.shape)

    ##################################################################
    # Gradient scaling
    ##################################################################

    # Define loss 
    l2loss = L2(Identity(int(np.prod(dobs.shape[1:]))), dobs.reshape(ns_rank[0], -1))

    ainv = AcousticWave2D(shape, origin, spacing, 
        x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1], 
        x_r[:, 0], x_r[:, 1], 
        0., tmax,  
        vprange=(vp_true.min(), vp_true.max()),
        src_type="Ricker", f0=freq,
        space_order=space_order, nbl=nbl,
        wav=wav, loss=l2loss,
        base_comm=comm,
        factor=par['factor']
    )

    if rank == 0:
        print('Compute gradient...')
        
    postproc = PostProcessX(scaling=1, mask=mask)
    loss, direction = ainv._loss_grad(vp_init, postprocess=postproc.apply)

    scaling = abs(direction).max()

    _, g_vmax = np.percentile(direction.T/scaling, [2,98]) 

    if rank == 0:
        plot_modulus(direction.T/scaling, aspect='auto', 
            interpolation='bicubic', cmap='seismic', 
            extent=[0,x.max(),z.max(),0],
            vmin=-g_vmax, vmax=g_vmax,
            fig_name=os.path.join(results_folder, 'InitialGradient_'+str(freq)+'freq.png'))

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
        if ((iter==0) or ((iter+1) % 25 == 0) or (iter+1 == maxiter)) and (rank==0):
            
            fwi_callback(vp_init_torch.clone().detach().numpy().reshape(vp_true.shape), 
                        vp_error=vp_error_history,
                        vp=vp_true, iter=total_iter)
            
            print(f'Frequency {freq}, Iteration {iter}, Loss {loss_history[-1]}')
            
        if torch.isnan(vp_init_torch).any():
            print("NaN detected in vp_inits")
            break
        
        total_iter += 1
    
    # Use the inverted moduli from current frequency as the next initial moduli
    vp_init = vp_init_torch.detach().clone().numpy()
    np.save(os.path.join(results_folder, 'Loss_freq'+str(freq)+'.npy'), np.array(loss_history))
    
    if rank == 0:    
        plt.figure(figsize=(10, 5))
        plt.semilogy(loss_history, 'k')
        plt.savefig(os.path.join(results_folder, 'DataError_'+str(freq)+'freq.png'))
        
        plt.figure(figsize=(10, 5))
        plt.semilogy(vp_error_history, 'b')
        plt.savefig(os.path.join(results_folder, 'ModelError_'+str(freq)+'freq.png'))
        
if rank == 0:    
    print('\nTotal time (s) = %.2f' % (time.time() - tstart))
    print('---------------------------------------------------------\n')