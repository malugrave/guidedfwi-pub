#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh
# 

#!/bin/bash

# Check if MPI is installed
if command -v mpirun &> /dev/null
then
    echo "MPI is already installed."
else
    echo "MPI is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y openmpi-bin libopenmpi-dev
    if [ $? -eq 0 ]; then
        echo "MPI installation successful."
    else
        echo "MPI installation failed."
        exit 1
    fi
fi

# Automatically locate conda.sh
if [ -n "$CONDA_PREFIX" ]; then
    # If in a conda environment, use the environment's base directory
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
elif [ -n "$CONDA_HOME" ]; then
    # If CONDA_HOME is set, this could point to the base installation
    source "$CONDA_HOME/etc/profile.d/conda.sh"
elif [ -d "$HOME/miniconda3" ]; then
    # If you know the installation path (Miniconda)
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -d "$HOME/anaconda3" ]; then
    # If you have Anaconda installed
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda installation not found."
    exit 1
fi

# Create conda environment
conda env create -f environment.yml
conda activate guidedfwi
conda env list

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusion model repositories as python packages
pip3 install git+https://github.com/lucidrains/denoising-diffusion-pytorch.git # 2D
pip3 install git+https://github.com/lucidrains/video-diffusion-pytorch.git # 3D

# Wave propagation packages
pip3 install deepwave
pip3 install devito==4.8.10
pip3 install numpy==1.26.4

# Install the guidedfwi package (source codes under the src/guidedfwi folder)
# Run the following line if you want to install the guidedfwi package only
pip3 install --use-pep517 -e .
echo 'Created and activated environment:' $(which python)

# Check cupy works as expected
echo 'Checking torch version and GPU'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'
echo 'Done!'