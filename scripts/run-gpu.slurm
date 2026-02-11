#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --gpus=1
#SBATCH --mem=0                       # grab ALL host RAM on the node
#SBATCH --constraint="a100&8gpus"       # Ibex A100 8-GPU nodes (≈1 TB host RAM)

# Load environment
module load openmpi
module load gcc
module load cuda/11.8
source ~/miniconda3/bin/activate taufimo

# Run job
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 4 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py --num_samples=10 --total_receivers=1024 --total_sources=576 --data_frequency=3 --velocity_size=128
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 4 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py --num_samples=10 --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=256
export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=21; export MKL_NUM_THREADS=21; export NUMBA_NUM_THREADS=21; mpiexec -n 6 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py --num_samples=10 --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=384