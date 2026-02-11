"""
Description:
Acoustic 3D Isotropic Reverse Time Migration (RTM) on the BG Compass model.
 
Run as: 
Adjust the XXX_NUM_THREADS below such that the product between this number on number of processes ("-n") does not exceed logical cores!

These are working example using ml.m5.12xlarge (48 CPUs and 192 GiB RAM):
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=3; export MKL_NUM_THREADS=3; export NUMBA_NUM_THREADS=3; mpiexec -n 13 python Compass_RTM_Acoustic3DIso.py

Contributors:
Originally adapted from https://github.com/DIG-Kaust/Devito-fwi version 0.1.0
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

import os
import numpy as np
import time
import json

from mpi4py import MPI
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from devito import configuration, mmax, Function, gaussian_smooth, Operator, Eq
from examples.seismic import AcquisitionGeometry, Model, Receiver, TimeAxis
from pylops.basicoperators import Identity, Laplacian

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

# See install.sh to install the following package
from guidedfwi.plots import plot_slices
from guidedfwi.utils import save_dict_as_compressed_npz, clear_devito_cache, highpass_filter, log_experiment
from guidedfwi.acoustic3disowrapper import AcousticWave3D
from guidedfwi.acousticisosolver import AcousticWaveSolver
from guidedfwi.postprocessing import PostProcessX
from guidedfwi.loss import Empty

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'WARNING'
configuration['log-level'] = 'ERROR'
        
if rank == 0:
    print(f'Distributed RTM ({size} ranks)')

##################################################################
# Parameters
##################################################################

# Model and aquisition parameters
par = {
    'nx':192,       'dx':8*5/1e3,     'ox':0,
    'ny':192,       'dy':8*5/1e3,     'oy':0,
    'nz':192,       'dz':4*5/1e3,     'oz':0,
    'ns':6*6,       'ds':8*5*191/5e3, 'os':0,  'sz':10/1e3,
    'nr':96*96,     'dr':2*8*5/1e3,   'or':0,  'rz':12*20/1e3,
    'nt':2000,      'dt':0.004,       'ot':0,
    'freq':10,
    'niter':20, 
    'sigma':25 
}

# Modelling parameters
shape = (par['nx'], par['ny'], par['nz'])
spacing = (par['dx'], par['dy'], par['dz'])
origin = (par['ox'], par['oy'], par['oz'])
space_order = 20 #8
nbl = 20

run_id = time.strftime("%Y%m%d-%H%M%S")

# Path to save figures
results_folder = (
    '../results/Compass_RTM_Acoustic3DIso_'+str(run_id)
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
y = np.arange(par['ny']) * par['dy'] + par['oy']
z = np.arange(par['nz']) * par['dz'] + par['oz']
t = np.arange(par['nt']) * par['dt'] + par['ot']
tmax = t[-1] # * 1e3 # in ms

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

##################################################################
# Velocity model
##################################################################

# Load the true model
vel_compass_3d = np.load('../data/velocities/compass_vp3d.npy')
vp_true = resize(vel_compass_3d, (par['nx'], par['ny'], par['nz']))

m_vmin, m_vmax = np.percentile(vp_true, (2,98))

mask = np.ones_like(vp_true)
mask[vp_true<1.51] = 0

# Initial model for FWI by smoothing the true model
vp_init = gaussian_filter(vp_true, sigma=[par['sigma'], par['sigma'], par['sigma']])

# Replace water velocity
vp_init[mask==0] = 1.5 

# Make 1D
vp_init[:,:,:] = vp_init[-1,-1,:]

dm_true = vp_true**(-2) - vp_init**(-2)
dm_vmin, dm_vmax = np.percentile(dm_true, (3,97))

if rank == 0:
    plot_slices(dm_true.T,  aspect='auto', size=8, interpolation='bicubic', cmap='gray', 
                vmin=dm_vmin,vmax=dm_vmax,
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'TrueldVel.png'))
    plot_slices(vp_true.T,  aspect='auto', size=8, interpolation='bicubic', cmap='rainbow', 
                vmin=m_vmin,vmax=m_vmax,
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'TruelVel.png'))
    plot_slices(vp_init.T,  aspect='auto', size=8, interpolation='bicubic', cmap='rainbow', 
                recs=[xr,yr], sous=[xs,ys],  
                vmin=m_vmin,vmax=m_vmax,
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'InitialVel.png'))

##################################################################
# RTM
##################################################################

# Choose how to split sources to ranks
ns_rank = local_split((par['ns'], ), MPI.COMM_WORLD, Partition.SCATTER, 0)
ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
isend_rank = np.cumsum(ns_ranks)[rank]
print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

if rank == 0:
    print('Model data...')
    tstart = time.time()

# Define modelling engine
amod = AcousticWave3D(shape, origin, spacing, 
    x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1],  x_s[isin_rank:isend_rank, 2],
    x_r[:, 0], x_r[:, 1], x_r[:, 2],
    0., tmax,  
    vp=vp_true,
    src_type="Ricker", f0=float(par['freq']),
    space_order=space_order, nbl=nbl,
    factor=24,
    base_comm=comm
)

# Model observed synthetic data
dobs, _ = amod.mod_allshots()

# Create operator
rtmloss = Empty(dobs.reshape(ns_rank[0], -1))

if rank == 0:
    print('Run RTM...')

ainv = AcousticWave3D(shape, origin, spacing, 
    x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1],  x_s[isin_rank:isend_rank, 2],
    x_r[:, 0], x_r[:, 1], x_r[:, 2],
    0., tmax,  
    vprange=(vp_true.min(), vp_true.max()),
    src_type="Ricker", f0=float(par['freq']),
    space_order=space_order, nbl=nbl,
    factor=24,
    loss=rtmloss,
    base_comm=comm
)

# Compute image
postproc = PostProcessX(scaling=1, mask=mask)
_, rtm = ainv._loss_grad(vp_init, postprocess=postproc.apply)
rtm /= rtm.max()

# Apply post-processing with Laplacian
Dop = Laplacian((par['nx'], par['ny'], par['nz']))
rtm_laplacian = Dop @ rtm
rtm_laplacian /= rtm_laplacian.max()

# Apply gain
gain = np.power(z, 3)
    
if  rank == 0:
    np.save(os.path.join(results_folder, 'RTM.npy'), rtm * gain)
    np.save(os.path.join(results_folder, 'RTMLap.npy'), rtm_laplacian * gain)
    
    plot_slices((rtm * gain).T,  aspect='auto', size=8, interpolation='bicubic', cmap='gray', 
                vmin=-1e-1, vmax=1e-1,
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'RTM.png'))
    
    plot_slices((rtm_laplacian * gain).T,  aspect='auto', size=8, interpolation='bicubic', cmap='gray', 
                vmin=-1e-3, vmax=1e-3,
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'RTMLap.png'))
    
    print('\nTotal time (s) = %.2f' % (time.time() - tstart))
    print('---------------------------------------------------------\n')