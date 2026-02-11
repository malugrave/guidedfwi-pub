# # Synthetic Example 3D: BG Compass model
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=28; export MKL_NUM_THREADS=28; export NUMBA_NUM_THREADS=28; CUDA_VISIBLE_DEVICES=0 mpiexec -n 4 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py --total_receivers=1024 --total_sources=576 --data_frequency=3 --velocity_size=128 --num_samples=2
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=28; export MKL_NUM_THREADS=28; export NUMBA_NUM_THREADS=28; CUDA_VISIBLE_DEVICES=0 mpiexec -n 4 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=256 --num_samples=2
# export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=37; export MKL_NUM_THREADS=37; export NUMBA_NUM_THREADS=37; CUDA_VISIBLE_DEVICES=0 mpiexec -n 3 python DiffusionModel_3D_FWIGuidedSamplingSynthetic.py --total_receivers=4096 --total_sources=576 --data_frequency=3 --velocity_size=384 --num_samples=2

# Synthetic Example 2D DAPS: SEG/EAGE Overthrust
for l in 1e-8 1e-6 1e-4 1e-2 1;
   do CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_DAPSFWISamplingSynthetic.py --start_guidance_from=700 --guidance_loop=5 --inject_guidance_every=5 --window_size=256 --stride=128 --training_data=seg --num_samples=2 --run_fwi_under=700 --use_fwi=y --resize_model=n --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 5 15 --langevin_eta=$l;
done