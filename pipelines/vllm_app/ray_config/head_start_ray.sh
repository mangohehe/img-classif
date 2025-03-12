#!/bin/bash
set -e

# Log start time
echo "Starting head_start_ray.sh at $(date)" >> /tmp/init.log

# Source bash profile and activate the ray_env Conda environment
echo "Sourcing ~/.bashrc and activating ray_env..." >> /tmp/init.log
source ~/.bashrc
source /home/ray/miniforge3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh" >> "$LOG_FILE"; exit 1; }
conda activate ray_env || { echo "Failed to activate ray_env" >> /tmp/init.log; exit 1; }

# Stop any existing Ray cluster
echo "Stopping existing Ray cluster..." >> /tmp/init.log
ray stop

# Set environment variables for Ray Dashboard + Grafana integration
echo "Setting environment variables for Ray integration..." >> /tmp/init.log
export RAY_GRAFANA_HOST="http://localhost:3000"
export RAY_GRAFANA_IFRAME_HOST="http://localhost:3000"
export RAY_PROMETHEUS_HOST="http://localhost:9090"
export RAY_PROMETHEUS_NAME="Prometheus"

# Increase file descriptor limit
echo "Setting ulimit to 65536..." >> /tmp/init.log
ulimit -n 65536

# Start the Ray head node with specified options
echo "Starting Ray head node..." >> /tmp/init.log
ray start \
  --head \
  --port=6379 \
  --dashboard-port=8265 \
  --object-manager-port=8076 \
  --autoscaling-config=~/ray_bootstrap_config.yaml \
  --dashboard-host=0.0.0.0 \
  --metrics-export-port=8080 || { echo "Failed to start Ray head node" >> /tmp/init.log; exit 1; }

# Launch the Ray Prometheus metrics exporter
echo "Launching Ray Prometheus metrics exporter..." >> /tmp/init.log
ray metrics launch-prometheus || { echo "Failed to launch Prometheus metrics exporter" >> /tmp/init.log; exit 1; }

# Create necessary Grafana provisioning directories
echo "Setting up Grafana provisioning directories..." >> /tmp/init.log
mkdir -p /tmp/ray/session_latest/metrics/grafana/provisioning/plugins || { echo "Failed to create Grafana plugins directory" >> /tmp/init.log; exit 1; }
mkdir -p /tmp/ray/session_latest/metrics/grafana/provisioning/alerting || { echo "Failed to create Grafana alerting directory" >> /tmp/init.log; exit 1; }

# Start Grafana server with custom configuration
echo "Starting Grafana server..." >> /tmp/init.log
grafana-server --homepath /usr/share/grafana --config /tmp/ray/session_latest/metrics/grafana/grafana.ini || { echo "Failed to start Grafana server" >> /tmp/init.log; exit 1; }

echo "Completed head_start_ray.sh at $(date)" >> /tmp/init.log