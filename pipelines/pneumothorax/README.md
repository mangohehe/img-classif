# Image Classif: Pneumothorax Inference Pipeline

**Image Classif** is a model reference service for Pneumothorax inference running on a Ray-based cluster on Google Cloud Platform (GCP). It provides modular components for model deployment via Ray Serve, dataset preprocessing, and inference execution—all configured through external YAML and JSON files.

## Features

- **Ray Serve Deployment:** Seamlessly deploy your model with Ray Serve.
- **GCP Integration:** Load model checkpoints and datasets directly from Google Cloud Storage.
- **Configurable Pipeline:** Customize model parameters, dataset paths, and image transformations via configuration files.
- **Modular Code Structure:** Clean separation between inference, deployment, and dataset management.
- **CLI Interface:** Run the entire inference pipeline with a simple command-line tool.

## Requirements

- Python 3.7+
- [Ray](https://docs.ray.io/en/latest/) (with Ray Serve)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/), [Requests](https://docs.python-requests.org/en/latest/), [tqdm](https://tqdm.github.io/)
- [Starlette](https://www.starlette.io/), [gcsfs](https://github.com/fsspec/gcsfs), [opencv-python](https://pypi.org/project/opencv-python/)
- [Albumentations](https://albumentations.ai/), [PyYAML](https://pyyaml.org/), [Cryptography](https://cryptography.io/)
- [Google Cloud Storage](https://googleapis.dev/python/storage/latest/), [Google API Python Client](https://github.com/googleapis/google-api-python-client)

For the complete list of dependencies, see the [requirements.txt](requirements.txt) file.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/img-classif.git
   cd img-classif

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

4. **Install the Package in Editable Mode:**

   ```bash
   pip install -e .

5. **Run the Tests (Optional):**

   ```bash
   pytest
   
6. **Running the Pipeline:**

   ```bash
   pneumothorax-pipeline --model ./configs/inferencer_gcp.yaml --transformer ./configs/transformer_1024.json

#### Configuration 
- **Model Configuration:** Create a model configuration file in YAML (e.g., ./configs/inferencer_gcp.yaml) that specifies the model architecture, checkpoint paths, and other parameters.

- **Transformer Configuration:** Create a transformer configuration file in JSON (e.g., ./configs/transformer_1024.json) to define image transformation settings using Albumentations.

## Accessing Grafana and the Ray Dashboard via SSH Tunnels

By default, ports 3000 (Grafana) and 8265 (Ray Dashboard) are only accessible within the cluster network. To securely connect from your local machine without opening these ports publicly, you can use SSH tunneling with the `gcloud` command.

### Steps

## Spin Up the Ray Cluster on GCP and Access Grafana & Ray Dashboard via SSH Tunnels

1. **Start the Ray Cluster**  
   From your local machine, run:
   ```bash
   ray up pneumothorax/ray_config/minimal-ray-cluster-cpus-only.yaml -y

2. **Create the SSH Tunnel:**
   Open a terminal on local machine and run:
   ```bash
   gcloud compute ssh \
       ray-arm-minimal-head-8e93b970-compute \
       --zone asia-southeast1-b \
       -- -L 3000:localhost:3000 -L 8265:localhost:8265

3. **Access Grafana & the Ray Dashboard**
   - Grafana: Navigate to http://localhost:3000 in your local browser.
   - Ray Dashboard: Navigate to http://localhost:8265 in your local browser.
