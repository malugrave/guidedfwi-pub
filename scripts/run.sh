# # Synthetic Example 1: SEAM Arid model
# CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_FWIGuidedSamplingSynthetic.py --start_guidance_from=700 --guidance_loop=5 --inject_guidance_every=5 --window_size=256 --stride=128 --training_data=seg --num_samples=20 --run_fwi_under=700 --use_fwi=y --resize_model=n --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 10 10
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 10 10 --num_fwis=50
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 10 10 --num_fwis=300
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 20 20 --num_fwis=50
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 20 20 --num_fwis=300
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 31 31 --num_fwis=50
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seam_arid --frequency=6 --num_sources=128 --sigma 31 31 --num_fwis=300

# # Synthetic Example 2: SEG Salt model
# CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_FWIGuidedSamplingSynthetic.py --start_guidance_from=700 --guidance_loop=5 --inject_guidance_every=5 --window_size=256 --stride=128 --training_data=seg --num_samples=20 --run_fwi_under=700 --use_fwi=y --resize_model=n --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 5 5
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 5 5 --num_fwis=50
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 5 5 --num_fwis=300
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 10 10 --num_fwis=50
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 10 10 --num_fwis=300
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 16 16 --num_fwis=50
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seg_salt --frequency=6 --num_sources=128 --sigma 16 16 --num_fwis=300

# # Synthetic Example 3: SEG/EAGE Overthrust model
# CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_FWIGuidedSamplingSynthetic.py --start_guidance_from=700 --guidance_loop=5 --inject_guidance_every=5 --window_size=256 --stride=128 --training_data=seg --num_samples=20 --run_fwi_under=700 --use_fwi=y --resize_model=n --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 5 15
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 5 15 --num_fwis=300
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 5 15 --num_fwis=50
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 10 15 --num_fwis=300
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 10 15 --num_fwis=50
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=y --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 21 21 --num_fwis=300
CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Synthetic.py --num_samples=20 --multi_source=n --velocity_type=seg_overthrust --frequency=5 --num_sources=256 --sigma 21 21 --num_fwis=50

# # Field Example 1: SVGD
# CUDA_VISIBLE_DEVICES=0 python SVGDFWI_2D_Field.py --num_samples=20 --num_fwis=200 --perturbation_amplitude=0.5

# # Field Example 2: Diffusion sampling
# CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_FWIGuidedSamplingField.py --start_guidance_from=900 --guidance_loop=2 --inject_guidance_every=1 --window_size=128 --stride=64 --model_size=128 --training_data=random-128-pred-v --num_samples=20
# CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_FWIGuidedSamplingField.py --start_guidance_from=900 --guidance_loop=2 --inject_guidance_every=1 --window_size=128 --stride=64 --model_size=128 --training_data=seg-128-pred-v --num_samples=20
# CUDA_VISIBLE_DEVICES=0 python DiffusionModel_2D_FWIGuidedSamplingField.py --start_guidance_from=900 --guidance_loop=2 --inject_guidance_every=1 --window_size=256 --stride=64 --model_size=256 --training_data=openfwi-256-pred-v --num_samples=20