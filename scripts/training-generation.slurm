#!/bin/bash -l
#SBATCH --job-name=rtm-train
# NOTE: write logs to submission directory to avoid mkdir race/permissions
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --tasks-per-node=16
#SBATCH --constraint="intel&cascadelake"

# Purpose:
# - Spray RTM generation across indices of Combined_vp3d_64.npy
# - One SLURM array task processes one or more indices (configurable via CHUNK)
# - Within each task, MPI is used to split sources across ranks (nprocs)

# Load environment (adapt to your cluster)
module load openmpi
module load gcc
source ~/miniconda3/bin/activate taufimo

############################################
# Tuning knobs (override with env if needed)
############################################

# Number of MPI ranks per task
# Default to use all allocated tasks (16)
: "${NPROCS:=${SLURM_NTASKS:-16}}"

# Threads per rank (OpenMP/BLAS/Numba)
# 1 thread per rank to fully occupy cores with MPI ranks
: "${THREADS:=3}"

# Dataset (default matches the script)
: "${DATASET_PATH:=/ibex/scratch/taufikmh/Datasets/3D/Combined_vp3d_64.npy}"

# Chunk size per array task (1 = one index per task)
: "${CHUNK:=1}"

# Optional offset for array-based indexing
: "${OFFSET:=0}"

# Index override (if you want to bypass array math)
# export INDEX_START=10 INDEX_END=12

############################################
# Compute indices for this array task
############################################

if [[ -n "$INDEX_START" && -n "$INDEX_END" ]]; then
  START="$INDEX_START"
  END="$INDEX_END"
else
  if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set; set INDEX_START/INDEX_END explicitly or submit with --array"
    exit 2
  fi
  START=$(( OFFSET + SLURM_ARRAY_TASK_ID * CHUNK ))
  END=$(( START + CHUNK - 1 ))
fi

echo "Running RTM training generation for indices: ${START}..${END}"
echo "MPI ranks: ${NPROCS}, Threads per rank: ${THREADS} on node ${SLURM_NODELIST}"

# Devito/OpenMP threading
export DEVITO_LANGUAGE=openmp
export DEVITO_MPI=0
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export NUMBA_NUM_THREADS=${THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Launch via Slurm's srun so slots match the allocation
mpiexec -n ${NPROCS} \
  python TrainingGeneration_RTM_Acoustic3DIso.py \
    --dataset_path "${DATASET_PATH}" \
    --index_start ${START} \
    --index_end ${END} \
    --clear_cache_end
