#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Installing Miniforge
echo "Installing Miniforge..." >> /tmp/init.log
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh || echo "Failed to download Miniforge" >> /tmp/init.log
sha256sum miniforge.sh
bash miniforge.sh -b -p "$HOME/miniforge3" || echo "Failed to install Miniforge" >> /tmp/init.log

# Initialize Conda for bash
~/miniforge3/bin/conda init bash || echo "Failed to initialize Conda" >> /tmp/init.log

# Reload bash configuration to ensure conda is available
source ~/.bashrc
source ~/miniforge3/etc/profile.d/conda.sh

# Configure Conda not to auto-activate the base environment
conda config --set auto_activate_base false || echo "Failed to configure Conda" >> /tmp/init.log

# Create the ray_env Conda environment with Python 3.12
echo "Creating Conda environment ray_env..." >> /tmp/init.log
conda create -n ray_env python=3.12 -y || echo "Failed to create Conda environment" >> /tmp/init.log
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate ray_env && \

# Ensure install dependencies
conda install -y pip || echo "Failed to install pip in ray_env" >> /tmp/init.log
conda install -y -c conda-forge libgl >> /tmp/init.log 2>&1 || echo "Failed to install libgl" >> /tmp/init.log

# Activate ray_env and pip install repository dependencies
echo "Activating ray_env and installing dependencies..." >> /tmp/init.log
pip install -r /home/ray/img-classif/pipelines/vllm_app/requirements.txt >> /tmp/init.log 2>&1 || \
echo "Failed to install repository dependencies" >> /tmp/init.log

echo "Setup env and dependencies completed successfully!" >> /tmp/init.log