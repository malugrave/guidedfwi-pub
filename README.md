
Reproducible material for **Diffusion model-based posterior sampling in full waveform inversion** and **Accelerating Stein variational gradient descent in full waveform inversion with diffusion models** - **Mohammad Taufik and Tariq Alkhalifah**


# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: contains logo and Matplotlib plotting style file.
* :open_file_folder: **data**: contains synhtetic velocity models.
* :open_file_folder: **results**: contains the experiments.
* :open_file_folder: **scripts**: contains Python codes to run the jobs.
* :open_file_folder: **notebooks**: contains jupyter notebooks experiments.
* :open_file_folder: **src**: contains source code.

## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate guidedfwi
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
