#!/bin/bash
set -e

LOG_FILE="/tmp/init.log"

echo "Starting head setup commands" >> "$LOG_FILE"

# Source the bash profile and activate the ray_env Conda environment
echo "Sourcing ~/.bashrc and activating ray_env..." >> "$LOG_FILE"
source ~/.bashrc
source /home/ray/miniforge3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh" >> "$LOG_FILE"; exit 1; }
conda activate ray_env || { echo "Failed to activate ray_env" >> "$LOG_FILE"; exit 1; }

# Re-create environment on head node using the provided environment.yml
echo "Re-creating environment on head node..." >> "$LOG_FILE"
conda env create -f /home/ray/environment.yml >> "$LOG_FILE" 2>&1 || \
  echo "Failed to create Conda environment from environment.yml" >> "$LOG_FILE"

# Install additional head dependencies via pip
echo "Installing additional head dependencies..." >> "$LOG_FILE"
pip install -r /home/ray/img-classif/pipelines/vllm_app/requirements.txt >> "$LOG_FILE" 2>&1 || \
  echo "Failed to install dependencies from requirements.txt" >> "$LOG_FILE"

# Install system-level prerequisites (run with sudo)
echo "Installing prerequisites and configuring Grafana repositories..." >> "$LOG_FILE"
sudo apt-get install -y apt-transport-https software-properties-common wget || \
  echo "Failed to install prerequisites" >> "$LOG_FILE"

sudo mkdir -p /etc/apt/keyrings/ || \
  echo "Failed to create /etc/apt/keyrings/" >> "$LOG_FILE"

wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | \
  sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null || \
  echo "Failed to download or convert Grafana GPG key" >> "$LOG_FILE"

echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | \
  sudo tee -a /etc/apt/sources.list.d/grafana.list || \
  echo "Failed to add Grafana stable repo" >> "$LOG_FILE"

echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com beta main" | \
  sudo tee -a /etc/apt/sources.list.d/grafana.list || \
  echo "Failed to add Grafana beta repo" >> "$LOG_FILE"

sudo apt-get update || \
  echo "Failed to update APT repositories" >> "$LOG_FILE"

sudo apt-get install -y grafana || \
  echo "Failed to install Grafana OSS" >> "$LOG_FILE"

sudo chown -R ray:ray /usr/share/grafana || \
  echo "Failed to update ownership of /usr/share/grafana" >> "$LOG_FILE"

sudo chmod -R 755 /usr/share/grafana || \
  echo "Failed to set permissions for /usr/share/grafana" >> "$LOG_FILE"

echo "Grafana installation and service start complete" >> "$LOG_FILE"
echo "Completed head setup commands" >> "$LOG_FILE"