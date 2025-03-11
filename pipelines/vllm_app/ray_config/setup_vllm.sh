#!/bin/bash
set -e

echo "Step 1: Updating package lists..."
sudo apt-get update -y

echo "Step 2: Installing GCC 12, G++ 12, and libnuma-dev..."
sudo apt-get install -y gcc-12 g++-12 libnuma-dev

echo "Step 3: Configuring GCC alternatives..."
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

echo "Step 4: Cloning the vLLM source code..."
# Check if directory exists to avoid re-cloning
if [ ! -d "vllm_source" ]; then
    git clone https://github.com/vllm-project/vllm.git vllm_source
else
    echo "vllm_source already exists. Skipping clone."
fi
cd vllm_source

echo "Step 5: Upgrading pip and installing required Python packages..."
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

echo "Step 6: Building and installing the vLLM CPU backend with explicit flags..."
VLLM_TARGET_DEVICE=cpu FORCE_CMAKE=1 pip install -e . --no-cache-dir --no-build-isolation

echo "Step 7: Setting environment variables for CPU execution..."
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_KVCACHE_SPACE=40   # Adjust based on available RAM
export OMP_NUM_THREADS=8           # Match the number of CPU cores
export VLLM_CPU_OMP_THREADS_BIND=0-7

echo "Setup completed successfully!"