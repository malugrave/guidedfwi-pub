"""
Description:
Acoustic 3D Isotropic Least-Square Reverse Time Migration (LSRTM) on the BG Compass model.
 
Run as: 
Adjust the XXX_NUM_THREADS below such that the product between this number on number of processes ("-n") does not exceed logical cores!

These are working example using ml.m5.12xlarge (48 CPUs and 192 GiB RAM):
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=3; export MKL_NUM_THREADS=3; export NUMBA_NUM_THREADS=3; mpiexec -n 32 python Compass_LSRTM_Acoustic3DIso.py
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=2; export NUMBA_NUM_THREADS=2; mpiexec -n 64 python Compass_LSRTM_Acoustic3DIso.py

Contributors:
Originally adapted from https://github.com/DIG-Kaust/Devito-fwi version 0.1.0
and https://github.com/devitocodes/devito/blob/master/examples/seismic/tutorials/13_LSRTM_acoustic.ipynb
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

import os
import time
import matplotlib
import numpy as np
import torch
import json

from matplotlib import pyplot as plt
from mpi4py import MPI
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from devito import configuration
from devito import Operator,Eq,solve,Grid,SparseFunction,norm
from devito import TimeFunction,Function
from devito import gaussian_smooth
from devito import mmax
from devito.logger import info
from devito import configuration

from examples.seismic import AcquisitionGeometry, Model, Receiver
from examples.seismic import Model
from examples.seismic import Receiver
from examples.seismic import TimeAxis
from examples.seismic.self_adjoint import (setup_w_over_q)

from devitofwi.deep.torchoperator import TorchOperator

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'WARNING'
configuration['log-level'] = 'ERROR'

import sys
sys.path.append('../src/')

# See install.sh to install the following package
from guidedfwi.plots import plot_slices
from guidedfwi.utils import log_experiment
from guidedfwi.acousticisosolver import AcousticWaveSolver
# from examples.seismic.acoustic import AcousticWaveSolver

# Comment out if LateX is not present
plt.style.use('../asset/plots.mplstyle')
  
# LSRTM operator
def lsrtm_gradient(dm, source_locations):
    
    residual = Receiver(name='residual', grid=model.grid, time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    
    d_obs = Receiver(name='d_obs', grid=model.grid,time_range=geometry.time_axis,
                         coordinates=geometry.rec_positions)

    d_syn = Receiver(name='d_syn', grid=model.grid,time_range=geometry.time_axis,
                         coordinates=geometry.rec_positions)
    
    grad_full = Function(name='grad_full', grid=model.grid)
    
    grad_illum = Function(name='grad_illum', grid=model.grid)
    
    src_illum = Function (name ="src_illum", grid = model.grid)

    # Using devito's reference of virtual source
    dm_true =  (solver.model.vp.data**(-2) - model0.vp.data**(-2))
    
    objective = 0.
    for i in range(source_locations.shape[0]):
        
        # Observed Data using Born's operator
        geometry.src_positions[0, :] = source_locations[i, :]

        # _, u0, _ = solver.forward(vp=model0.vp, save=True)
        _, u0, usnaps, _ = solver.forward(vp=model0.vp, save=False, autotune=True, factor=16)
        
        # _, _, _,_ = solver.jacobian(dm_true, vp=model0.vp, rec = d_obs)
        _, _, _, _ = solver.jacobian(dm_true, vp=model0.vp, rec=d_obs, autotune=True)

        # Calculated Data using Born's operator
        solver.jacobian(dm, vp=model0.vp, rec = d_syn)
        
        residual.data[:] = d_syn.data[:]- d_obs.data[:]
     
        # grad_shot,_ = solver.gradient(rec=residual, u=u0, vp=model0.vp)
        grad_shot,_ = solver.gradient(rec=residual, u=u0, usnaps=usnaps, vp=model0.vp, autotune=True, factor=16)
        
        # Imaging condition
        src_illum_upd =  Eq(src_illum, src_illum + u0**2)
        
        op_src = Operator([src_illum_upd])
        op_src.apply(**{'time_M':0})
        # op_src.apply()
              
        grad_sum = Eq(grad_full, grad_full  + grad_shot)
        op_grad = Operator([grad_sum])
        op_grad.apply()
        
        objective += .5*np.linalg.norm(residual.data)**2
        # objective += .5*norm(residual)**2
        
    grad_f = Eq(grad_illum, grad_full/(src_illum+10**-9))
    op_gradf = Operator([grad_f])
    op_gradf.apply()
     
    # return objective, grad_illum, d_obs, d_syn
    return objective, grad_illum

# Step size update
def get_alfa(grad_iter,grad_full,image_iter,niter_lsrtm):
     
    term1 = np.dot(image_iter.reshape(-1), image_iter.reshape(-1))
    term2 = np.dot(image_iter.reshape(-1), grad_iter.reshape(-1))
    term3 = np.dot(grad_iter.reshape(-1), grad_iter.reshape(-1))
    
    if niter_lsrtm == 0:
        alfa = .05 / mmax(grad_full)
    else:
        abb1 = term1 / term2
        abb2 = term2 / term3
        abb3 = abb2 / abb1
        
        if abb3 > 0 and abb3 < 1:
            alfa = abb2
        else:
            alfa = abb1
            
    return alfa  

# Callback to track model error
def lsrtm_callback(xk, iter, **kwargs):
    
    if rank == 0:
        plot_slices(xk.reshape((par['nx'], par['ny'], par['nz'])).T, size=8, 
            aspect='auto', interpolation='bicubic', 
            vmin=dm_vmin,vmax=dm_vmax,
            bounds=[0,x.max(),y.max(),z.max()],
            cmap='gray', fig_name=os.path.join(results_folder, 'Img_'+str(iter)+'.png'), **kwargs)
        
        np.save(os.path.join(results_folder, 'Image_'+str(iter)+'.npy'), xk.reshape((par['nx'], par['ny'], par['nz'])))

if rank == 0:
    print(f'Distributed LSRTM ({size} ranks)')

##################################################################
# Parameters
##################################################################

# Model and aquisition parameters
# par = {
#     'nx':64,      'dx':20/1e3,     'ox':0,
#     'ny':64,      'dy':20/1e3,     'oy':0,
#     'nz':64,      'dz':10/1e3,     'oz':0,
#     'ns':8*8,     'ds':20*9/1e3,   'os':0,  'sz':0,
#     'nr':64*64,   'dr':20/1e3,     'or':0,  'rz':0,
#     'nt':1500,    'dt':0.001,      'ot':0,
#     'freq':35, 
#     'sigma':5
# }

par = {
    'nx':64,      'dx':80/1e3,     'ox':0,
    'ny':64,      'dy':80/1e3,     'oy':0,
    'nz':64,      'dz':40/1e3,     'oz':0,
    'ns':8*8,     'ds':80*9/1e3,   'os':0,  'sz':0,
    'nr':64*64,   'dr':80/1e3,     'or':0,  'rz':0,
    'nt':3000,    'dt':0.002,      'ot':0,
    'freq':10, 
    'sigma':5
}

# par = {
#     'nx':192,       'dx':8*5/1e3,     'ox':0,
#     'ny':192,       'dy':8*5/1e3,     'oy':0,
#     'nz':192,       'dz':4*5/1e3,     'oz':0,
#     'ns':6*6,       'ds':8*5*191/5e3, 'os':0,  'sz':10/1e3,
#     'nr':96*96,     'dr':2*8*5/1e3,   'or':0,  'rz':12*20/1e3,
#     'nt':2000,      'dt':0.004,       'ot':0,
#     'freq':10,
#     'niter':20, 
#     'sigma':25 
# }

# Modelling parameters
shape = (par['nx'], par['ny'], par['nz'])
spacing = (par['dx'], par['dy'], par['dz'])
origin = (par['ox'], par['oy'], par['oz'])
space_order = 20 #8
nbl = 20

run_id = time.strftime("%Y%m%d-%H%M%S")

# Path to save figures
results_folder = (
    '../results/Compass_LSRTM_Acoustic3DIso_'+str(run_id)
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
    
fpeak = par['freq']/1e3
t0w = 1.0 / fpeak
omega = 2.0 * np.pi * fpeak
qmin = 0.1
qmax = 100000
npad = nbl
dtype = np.float32

nshots = par['ns']
nreceivers = par['nr']
t0 = 0.
tn = tmax  # Simulation last 1 second (1000 ms)
filter_sigma = (3, 3, 3) # Filter's length

init_damp = lambda func, nbl: setup_w_over_q(func, omega, qmin, qmax, npad, sigma=0)
model = Model(vp=vp_true, origin=origin, shape=shape, spacing=spacing,
              space_order=8, bcs=init_damp,nbl=npad,dtype=dtype)
model0 = Model(vp=vp_true, origin=origin, shape=shape, spacing=spacing,
              space_order=8, bcs=init_damp,nbl=npad,dtype=dtype)

dt = model.critical_dt 
s = model.grid.stepping_dim.spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
nt=time_range.num

model0.vp.data[nbl:(-nbl),nbl:(-nbl),nbl:(-nbl)] = vp_init


if rank == 0:
    plot_slices(dm_true.T, size=8, aspect='auto', interpolation='bicubic', cmap='gray', 
                vmin=dm_vmin,vmax=dm_vmax, 
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'TruedVel.png'))

    plot_slices(model.vp.data[nbl:(-nbl),nbl:(-nbl),nbl:(-nbl)].T, size=8, aspect='auto', interpolation='bicubic', cmap='rainbow', 
                vmin=m_vmin,vmax=m_vmax, 
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'TrueVel.png'))

    plot_slices(model0.vp.data[nbl:(-nbl),nbl:(-nbl),nbl:(-nbl)].T, size=8, aspect='auto', interpolation='bicubic', cmap='rainbow', 
                recs=[xr,yr], sous=[xs,ys], 
                vmin=m_vmin,vmax=m_vmax, 
                bounds=[0,x.max(),y.max(),z.max()],
                fig_name=os.path.join(results_folder, 'InitialVel.png'))
    
src_coordinates = np.empty((1, 3))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = par['rz']

rec_coordinates = x_r

geometry = AcquisitionGeometry(
    model,
    rec_coordinates,
    src_coordinates,
    t0,
    tn,
    src_type='Ricker',
    f0=fpeak * 1e3,
    fs=model.fs,
)

solver = AcousticWaveSolver(model, geometry, space_order=8)

##################################################################
# LSRTM
##################################################################

# Choose how to split sources to ranks
ns_rank = local_split((par['ns'], ), MPI.COMM_WORLD, Partition.SCATTER, 0)
ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
isend_rank = np.cumsum(ns_ranks)[rank]
print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

niter=20 # Number of iterations of the LSRTM
history = np.zeros((niter, 1)) # Objective function

image_up_dev = np.zeros((model0.vp.shape[0],model0.vp.shape[1],model0.vp.shape[2]),dtype)
image = np.zeros((model0.vp.shape[0],model0.vp.shape[1],model0.vp.shape[2]))

image_prev = np.zeros((model0.vp.shape[0],model0.vp.shape[1],model0.vp.shape[2]))    
grad_prev  = np.zeros((model0.vp.shape[0],model0.vp.shape[1],model0.vp.shape[2]))

yk  = np.zeros((model0.vp.shape[0],model0.vp.shape[1],model0.vp.shape[2]))
sk = np.zeros((model0.vp.shape[0],model0.vp.shape[1],model0.vp.shape[2]))

if rank == 0:
    print('Run LSRTM...')
    tstart = time.time()

for k in tqdm(range(niter)):
    dm =  image_up_dev
    
    objective, grad_full = lsrtm_gradient(dm, source_locations=x_s[isin_rank:isend_rank])
    
    # Gather loss and gradient from all ranks
    objective = comm.allreduce(objective, op=MPI.SUM)
    grad_full_data = comm.allreduce(grad_full.data, op=MPI.SUM)
    grad_full.data[:] = grad_full_data
    
    history[k] = objective
    
    yk = grad_full.data - grad_prev
    sk = image_up_dev - image_prev
    alfa = get_alfa(yk, grad_full, sk, k)

    grad_prev = grad_full.data
    image_prev = image_up_dev
    image_up_dev = image_up_dev - alfa*grad_full.data
      
    # Saving the first migration using Born operator
    if k == 0:
        image = image_up_dev

    if rank == 0:
        print(f'Iteration {k}, Loss {objective}')
        lsrtm_callback(image_up_dev[nbl:(-nbl),nbl:(-nbl),nbl:(-nbl)], iter=k)
    
if  rank == 0:
    print('\nTotal time (s) = %.2f' % (time.time() - tstart))
    print('---------------------------------------------------------\n')

    plt.figure(figsize=(14, 5))
    plt.plot(history, 'k')
    plt.title('Loss history')
    plt.savefig(os.path.join(results_folder, 'Loss.png'))