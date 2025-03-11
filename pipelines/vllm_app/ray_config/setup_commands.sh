#!/bin/bash

# Log the start of the script
echo "Starting setup_all.sh" >> /tmp/init.log

# Install system dependencies
echo "Installing system dependencies..." >> /tmp/init.log
sudo apt-get update && sudo apt-get install -y wget bzip2 curl software-properties-common || { echo "Failed to install dependencies" >> /tmp/init.log; exit 1; }

# Clone the repository
echo "Cloning repository..." >> /tmp/init.log
git clone https://github.com/mangohehe/img-classif.git /home/ray/img-classif || { echo "Failed to clone repository" >> /tmp/init.log; exit 1; }

# Set Conda installation directory
CONDA_DIR="/home/ray/miniforge3"

# Source Conda initialization script
CONDA_SH="$CONDA_DIR/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH" || { echo "Failed to initialize Conda" >> /tmp/init.log; exit 1; }
else
  echo "Conda initialization script not found at $CONDA_SH. Please ensure Conda is properly installed." >> /tmp/init.log
  exit 1
fi

# Create and activate the Conda environment
echo "Setting up Ray environment..." >> /tmp/init.log
bash /home/ray/img-classif/pipelines/vllm_app/ray_config/setup_ray_env.sh >> /tmp/init.log 2>&1 || { echo "Failed to setup Ray environment" >> /tmp/init.log; exit 1; }
conda activate ray_env || { echo "Failed to activate ray_env" >> /tmp/init.log; exit 1; }

# Setup VLLM
echo "Setting up VLLM..." >> /tmp/init.log
bash /home/ray/img-classif/pipelines/vllm_app/ray_config/setup_vllm.sh >> /tmp/init.log 2>&1 || { echo "Failed to setup VLLM" >> /tmp/init.log; exit 1; }

# Install the vllm_app package
echo "Installing vllm_app package..." >> /tmp/init.log
cd /home/ray/img-classif/pipelines/vllm_app && pip install -e . >> /tmp/init.log 2>&1 || { echo "Failed to install the vllm_app package" >> /tmp/init.log; exit 1; }

# Export the Conda environment
echo "Exporting Conda environment..." >> /tmp/init.log
conda env export > /home/ray/environment.yml || { echo "Failed to export Conda environment" >> /tmp/init.log; exit 1; }

# Log the completion of the script
echo "Completed setup_all.sh" >> /tmp/init.log