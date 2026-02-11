"""
Description:
Training data generation via Acoustic 3D Isotropic Reverse Time Migration (RTM).

This script mirrors Compass_RTM_Acoustic3DIso.py but iterates over models
from a combined dataset and writes per-index compressed outputs containing
the velocity cube and its corresponding RTM image.
 
Run as:
Adjust the XXX_NUM_THREADS below such that the product between this number
and the number of processes ("-n") does not exceed logical cores.

Example (CPU with MPI):
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=7; export MKL_NUM_THREADS=7; export NUMBA_NUM_THREADS=7; \
mpiexec -n 16 python TrainingGeneration_RTM_Acoustic3DIso.py --index_start 0 --index_end 1 --save_plots

Contributors:
Originally adapted from Compass_RTM_Acoustic3DIso.py and internal wrappers
Modified by Mohammad Hasyim Taufik (hatsyim)
"""

import os
import time
import json
import inspect
import warnings
import numpy as np

from argparse import ArgumentParser
from mpi4py import MPI
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from pylops_mpi.DistributedArray import local_split, Partition
from pylops.basicoperators import Laplacian
from matplotlib import pyplot as plt

# Devito
from devito import configuration

# Remove when guidedfwi is already installed
import sys
sys.path.append('../src/')

# Internal utilities
from guidedfwi.plots import plot_slices
from guidedfwi.utils import save_dict_as_compressed_npz, clear_devito_cache, log_experiment
from guidedfwi.acoustic3disowrapper import AcousticWave3D
from guidedfwi.postprocessing import PostProcessX
from guidedfwi.loss import Empty

# # Comment out if LaTeX is not present
# plt.style.use('../asset/plots.mplstyle')

# Suppress all warnings
warnings.filterwarnings("ignore")


def main():

    ##################################################################
    # Arguments
    ##################################################################
    parser = ArgumentParser()

    # Dataset and indexing
    parser.add_argument(
        "--dataset_path", type=str,
        default='/data/taufikmh/Datasets/3D/Combined_vp3d_64.npy',
    )
    parser.add_argument(
        "--index_start", type=int, default=0,
    )
    parser.add_argument(
        "--index_end", type=int, default=-1,
    )
    parser.add_argument(
        "--index_step", type=int, default=1,
    )

    # Grid and acquisition parameters (close to Compass_RTM_Acoustic3DIso)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--dx", type=float, default=8*5/1e3)  # km
    parser.add_argument("--dy", type=float, default=8*5/1e3)  # km
    parser.add_argument("--dz", type=float, default=4*5/1e3)  # km
    parser.add_argument("--nt", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--freq", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=6.0)
    parser.add_argument("--space_order", type=int, default=12)
    parser.add_argument("--nbl", type=int, default=20)
    parser.add_argument("--src_grid", type=int, default=8, help="sqrt(ns) layout")
    parser.add_argument("--rec_grid", type=int, default=64, help="sqrt(nr) layout")
    parser.add_argument("--sz", type=float, default=10/1e3, help="source depth (km)")
    parser.add_argument("--rz", type=float, default=20/1e3, help="receiver depth (km)")
    parser.add_argument("--factor", type=int, default=4, help="snapshot subsampling for gradient")

    # I/O and plotting
    parser.add_argument("--results_root", type=str, default="../results")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_first_only", action="store_true")
    parser.add_argument("--clear_cache_end", action="store_true",
                        help="Clear Devito JIT cache once at the end (rank 0 only)")

    args = parser.parse_args()

    ##################################################################
    # MPI and Devito config
    ##################################################################
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    configuration['log-level'] = 'ERROR'

    if rank == 0:
        print(f'Distributed RTM training generation ({size} ranks)')

    ##################################################################
    # Experiment logging
    ##################################################################
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_folder = os.path.join(
        args.results_root, f'TrainingGeneration_RTM_Acoustic3DIso_{run_id}'
    )

    if rank == 0:
        os.makedirs(results_folder, exist_ok=True)
        # Log experiment
        log_experiment(
            results_folder,
            script_path=os.path.abspath(inspect.getfile(inspect.currentframe())),
            run_id=run_id,
        )
        with open(os.path.join(results_folder, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    ##################################################################
    # Load dataset (memory-mapped)
    ##################################################################
    if rank == 0:
        print('Opening combined dataset (memmap)...')
    vp_combined = np.load(args.dataset_path, mmap_mode='r')/1e3  # shape (N, 64, 64, 64)
    N = vp_combined.shape[0]

    if args.index_end < 0 or args.index_end >= N:
        index_end = N - 1
    else:
        index_end = args.index_end

    indices = list(range(args.index_start, index_end + 1, args.index_step))
    if rank == 0:
        print(f'Total models to process: {len(indices)} (from {indices[0]} to {indices[-1]})')

    ##################################################################
    # Acquisition set-up (mirrors Compass script, scaled to grid)
    ##################################################################
    # Spatial grid
    nx, ny, nz = args.nx, args.ny, args.nz
    dx, dy, dz = args.dx, args.dy, args.dz
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    # Time axis
    nt, dt = args.nt, args.dt
    t = np.arange(nt) * dt
    tmax = t[-1]

    # Source and receiver layouts
    ns_sqrt = args.src_grid
    nr_sqrt = args.rec_grid
    ns = ns_sqrt * ns_sqrt
    nr = nr_sqrt * nr_sqrt

    # Spacing across the full domain width (close to Compass style)
    Lx = (nx - 1) * dx
    Ly = (ny - 1) * dy
    dsx = Lx / (ns_sqrt - 1)
    dsy = Ly / (ns_sqrt - 1)
    drx = Lx / (nr_sqrt - 1)
    dry = Ly / (nr_sqrt - 1)

    xs, ys = np.meshgrid(np.arange(ns_sqrt) * dsx, np.arange(ns_sqrt) * dsy)
    x_s = np.zeros((ns, 3))
    x_s[:, 0] = xs.reshape(-1)
    x_s[:, 1] = ys.reshape(-1)
    x_s[:, 2] = args.sz

    xr, yr = np.meshgrid(np.arange(nr_sqrt) * drx, np.arange(nr_sqrt) * dry)
    x_r = np.zeros((nr, 3))
    x_r[:, 0] = xr.reshape(-1)
    x_r[:, 1] = yr.reshape(-1)
    x_r[:, 2] = args.rz

    ##################################################################
    # Source distribution across ranks (as in Compass)
    ##################################################################
    ns_rank = local_split((ns, ), MPI.COMM_WORLD, Partition.SCATTER, 0)
    ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
    isin_rank = np.insert(np.cumsum(ns_ranks)[:-1], 0, 0)[rank]
    isend_rank = np.cumsum(ns_ranks)[rank]
    if rank == 0:
        print(f'Source partition: ns_total={ns}, per-rank={ns_ranks.tolist()}')

    ##################################################################
    # Loop over requested models
    ##################################################################
    for k, idx in enumerate(indices):
        if rank == 0:
            print(f"\n[{k+1}/{len(indices)}] Processing model index: {idx}")
                            # Velocity

        tstart = time.time() if rank == 0 else None

        # Load velocity (km/s)
        vp_true = np.array(vp_combined[idx], dtype=np.float32)  # shape (64,64,64)

        # If input grids differ, resize to (nx, ny, nz)
        if vp_true.shape != (nx, ny, nz):
            vp_true = resize(vp_true, (nx, ny, nz), preserve_range=True).astype(np.float32)

        # Basic stats and water mask
        m_vmin, m_vmax = np.percentile(vp_true, (2, 98))
        mask = np.ones_like(vp_true, dtype=np.float32)
        # mask[vp_true < 1.51] = 0.0

        # Smooth true model to form initial model (FWI-style)
        vp_init = gaussian_filter(vp_true, sigma=[args.sigma, args.sigma, args.sigma]).astype(np.float32)
        # vp_init[mask == 0] = 1.5  # water replacement
        
        if rank == 0:
            # Bounds for plotting in km
            bounds = [0, x.max(), y.max(), z.max()]
            
            # Velocity
            plot_slices(vp_true.T, aspect='auto', size=8, interpolation='bicubic',
                        cmap='rainbow', vmin=m_vmin, vmax=m_vmax,
                        bounds=bounds,
                        fig_name=os.path.join(results_folder, f'VpTrue_{idx:05d}.png'))
            plot_slices(vp_init.T, aspect='auto', size=8, interpolation='bicubic',
                        cmap='rainbow', vmin=m_vmin, vmax=m_vmax,
                        bounds=bounds,
                        fig_name=os.path.join(results_folder, f'VpInit_{idx:05d}.png'))

        ##################################################################
        # RTM
        ##################################################################
        if rank == 0:
            print('Modeling data (per-rank forward)...')

        # Modeling operator with true model on local shot subset
        amod = AcousticWave3D(
            shape=(nx, ny, nz),
            origin=(0.0, 0.0, 0.0),
            spacing=(dx, dy, dz),
            src_x=x_s[isin_rank:isend_rank, 0],
            src_y=x_s[isin_rank:isend_rank, 1],
            src_z=x_s[isin_rank:isend_rank, 2],
            rec_x=x_r[:, 0],
            rec_y=x_r[:, 1],
            rec_z=x_r[:, 2],
            t0=0.0,
            tn=tmax,
            vp=vp_true,
            src_type="Ricker",
            f0=float(args.freq),
            space_order=args.space_order,
            nbl=args.nbl,
            factor=args.factor,
            base_comm=comm,
        )

        # Local forward modeling for assigned shots
        dobs, _ = amod.mod_allshots()

        # Empty loss for RTM (adjoint equals data)
        rtmloss = Empty(dobs.reshape(ns_rank[0], -1))

        if rank == 0:
            print('Computing RTM (adjoint imaging)...')

        # Inversion operator using vprange for consistent time axis
        ainv = AcousticWave3D(
            shape=(nx, ny, nz),
            origin=(0.0, 0.0, 0.0),
            spacing=(dx, dy, dz),
            src_x=x_s[isin_rank:isend_rank, 0],
            src_y=x_s[isin_rank:isend_rank, 1],
            src_z=x_s[isin_rank:isend_rank, 2],
            rec_x=x_r[:, 0],
            rec_y=x_r[:, 1],
            rec_z=x_r[:, 2],
            t0=0.0,
            tn=tmax,
            vprange=(vp_true.min(), vp_true.max()),
            src_type="Ricker",
            f0=float(args.freq),
            space_order=args.space_order,
            nbl=args.nbl,
            factor=args.factor,
            loss=rtmloss,
            base_comm=comm,
        )

        # Post-process: scale, mask, etc.
        postproc = PostProcessX(scaling=1.0, mask=mask)
        _, rtm = ainv._loss_grad(vp_init, postprocess=postproc.apply)

        # Normalize and enhance image similarly to Compass code
        rtm = rtm.astype(np.float32)
        rtm /= (np.max(np.abs(rtm)) + 1e-12)

        # Optional Laplacian enhancement (not saved, but used for plots if requested)
        Dop = Laplacian((nx, ny, nz))
        rtm_lap = Dop @ rtm
        rtm_lap = rtm_lap.astype(np.float32)
        rtm_lap /= (np.max(np.abs(rtm_lap)) + 1e-12)

        # Depth gain
        gain = np.power(z, 3, dtype=np.float64).astype(np.float32)
        rtm_gain = rtm * gain  # broadcasting along z
        rtm_lap_gain = rtm_lap * gain

        # Save compressed outputs: velocity and RTM image
        if rank == 0:
            out_path = os.path.join(results_folder, f'RTM_Acoustic3DIso_{idx:05d}.npz')
            save_dict_as_compressed_npz(out_path, {
                'vp': vp_true,                          # (nx, ny, nz)
                'vp_init': vp_init,                     # (nx, ny, nz)
                'rtm': rtm,                             # (nx, ny, nz)
                'rtm_gain': rtm_gain,                   # (nx, ny, nz)
                'rtm_lap_gain': rtm_lap_gain,           # (nx, ny, nz)
            })

            if args.save_plots and (not args.plot_first_only or k == 0):

                # Images
                m_vmin, m_vmax = np.percentile(rtm_gain, (2, 98))
                plot_slices(rtm_gain.T, aspect='auto', size=8, interpolation='bicubic',
                            cmap='gray', vmin=m_vmin, vmax=m_vmax,
                            bounds=bounds,
                            fig_name=os.path.join(results_folder, f'RTM_{idx:05d}.png'))
                m_vmin, m_vmax = np.percentile(rtm_lap_gain, (2, 98))
                plot_slices(rtm_lap_gain.T, aspect='auto', size=8, interpolation='bicubic',
                            cmap='gray', vmin=m_vmin, vmax=m_vmax,
                            bounds=bounds,
                            fig_name=os.path.join(results_folder, f'RTMLap_{idx:05d}.png'))

            # Timing
            if tstart is not None:
                print('Total time (s) = %.2f' % (time.time() - tstart))

        # Synchronize ranks between indices to prevent overlapping JIT work on the cache
        comm.Barrier()

        # Do NOT clear Devito cache inside the loop; it can race with ongoing JIT compilation
        # across ranks and iterations. Optionally clear once at the end.

    if rank == 0:
        print('\nAll requested models processed.')
        if args.clear_cache_end:
            try:
                clear_devito_cache()
                print('Devito cache cleared at end of run.')
            except Exception as e:
                print(f'Warning: could not clear Devito cache: {e}')


if __name__ == "__main__":
    main()