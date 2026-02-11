#!/bin/bash -l 
#SBATCH --mem=1000G
#SBATCH --time=24:00:00 
#SBATCH --ntasks=48
#SBATCH --tasks-per-node=48
#SBATCH --constraint="intel&cascadelake"

# Load environment
module load openmpi
module load gcc
# module load cuda/11.2.2
source ~/miniconda3/bin/activate taufimo

# Run : standard FWI
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIso_L2_Adam_1stage.py --total_receivers=16384 --total_sources=2500 --data_frequency=3 --velocity_size=512
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIso_L2_Adam_1stage.py --total_receivers=9216 --total_sources=1296 --data_frequency=3 --velocity_size=384
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIso_L2_Adam_1stage.py --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=256
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIso_L2_Adam_1stage.py --total_receivers=1024 --total_sources=576 --data_frequency=3 --velocity_size=128

# Run : multi-source FWI
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIsoMultiSource_L2_Adam_1stage.py --total_receivers=6400 --total_sources=1024 --data_frequency=3 --velocity_size=512
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=16; export MKL_NUM_THREADS=16; export NUMBA_NUM_THREADS=16; mpiexec -n 3 python Compass_FWI_Acoustic3DIsoMultiSource_L2_Adam_1stage.py --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=384
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIsoMultiSource_L2_Adam_1stage.py --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=256
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python Compass_FWI_Acoustic3DIsoMultiSource_L2_Adam_1stage.py --total_receivers=1024 --total_sources=576 --data_frequency=3 --velocity_size=128