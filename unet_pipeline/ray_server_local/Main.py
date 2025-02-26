# python main.py --cfg /Users/fenggao/Library/CloudStorage/GoogleDrive-2023isanewjourney@gmail.com/My\ Drive/kaggle/img-classif/unet_pipeline/experiments/albunet_valid/inference_local.yaml
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import argparse
from pathlib import Path
import logging
from typing import Dict
import yaml
from ServeDeployment import start_serve
from InferenceClient import run_inference, save_results
from PneumothoraxRayDataset import PneumothoraxRayDataset as PneumothoraxDataset
from utils.Helpers import init_logger
import albumentations as albu

# Initialize logger
SCRIPT_DIR = Path(__file__).parent
logs_dir = SCRIPT_DIR / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "inference.log"
logger = init_logger(__name__, log_file_name=str(log_file_path))

def argparser() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Pneumothorax pipeline')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--local', action='store_true', help='Run in local mode (skip GCP and Ray)')
    return parser.parse_args()

def main() -> None:
    """Main function to run the inference pipeline."""
    args = argparser()
    config_path = Path(args.cfg).resolve()  # Resolve to absolute path
    logger.info(f"Using config file: {config_path}")

    # Load the config file
    with open(config_path, "r") as f:
        inference_config = yaml.safe_load(f)
    logger.info(f"Loaded inference config: {inference_config}")

    # Load configuration parameters
    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']  # Use CPU or GPU based on config
    num_workers = inference_config['NUM_WORKERS']
    transform = albu.load(inference_config['TEST_TRANSFORMS'])
    use_flip = inference_config['FLIP']
    local_data_dir = Path(inference_config['LOCAL_DATA_DIR'])
    local_checkpoint_path = Path(inference_config['CHECKPOINTS']['LOCAL_CHECKPOINT_DIR'])
    result_path = local_data_dir / inference_config['RESULT']

    # Start Ray Serve deployment
    if not args.local:
        logger.info("Starting Ray Serve deployment...")
        start_serve(inference_config, str(local_checkpoint_path / inference_config['CHECKPOINTS']['CHECKPOINT_NAME']), device)

    # Load dataset
    dataset = PneumothoraxDataset(data_folder=local_data_dir, mode='test', transform=transform)
    ray_dataset = dataset.get_dataset()

    # Run inference
    logger.info("Running inference...")
    mask_dict = run_inference(ray_dataset, batch_size, use_flip, num_workers)

    # Save results
    logger.info("Saving results...")
    save_results(mask_dict, result_path)
    logger.info("Inference complete.")

if __name__ == "__main__":
    main()