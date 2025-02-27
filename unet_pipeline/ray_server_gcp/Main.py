import sys
import os
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from pathlib import Path
import argparse
import logging
import yaml
import gcsfs  # For GCS support
import albumentations as albu
from ServeDeployment import start_serve
from InferenceClient import run_inference, save_results
from PneumothoraxRayDatasetOnGCP import PneumothoraxRayDatasetOnGCP as PneumothoraxDataset
from utils.Helpers import init_logger

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
    parser.add_argument('--transform', type=str, type=str, help='Path to the transformation YAML file')
    return parser.parse_args()

def load_config(config_path: Path) -> Dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_path(base_path: str, sub_path: str, use_gcs: bool) -> str:
    """Resolve a path, either local or GCS."""
    if use_gcs:
        return f"gs://{base_path}/{sub_path}"
    return str(Path(base_path) / sub_path)

def main() -> None:
    """Main function to run the inference pipeline."""
    args = argparser()
    config_path = Path(args.cfg).resolve()  # Resolve to absolute path
    transform = albu.load(Path(args.transform).resolve())
    logger.info(f"Using config file: {config_path}")

    # Load the config file
    inference_config = load_config(config_path)
    logger.info(f"Loaded inference config: {inference_config}")

    # Determine if running in GCP mode
    use_gcs = not args.local

    # Load configuration parameters
    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']  # Use CPU or GPU based on config
    num_workers = inference_config['NUM_WORKERS']
    use_flip = inference_config['FLIP']

    # Resolve data and checkpoint paths
    data_dir = inference_config['GCS_DATA_DIR'] if use_gcs else inference_config['LOCAL_DATA_DIR']
    checkpoint_dir = inference_config['CHECKPOINTS']['GCS_CHECKPOINT_DIR']
    checkpoint_name = inference_config['CHECKPOINTS']['CHECKPOINT_NAME']
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"

    # Result path
    result_path = resolve_path(inference_config['RESULT_FOLDER'], inference_config['RESULT'], use_gcs)

    # Initialize Ray (if not in local mode)
    if not args.local:
        import ray
        if not ray.is_initialized():
            ray.init(address="auto")  # Connect to the Ray cluster
        logger.info("Ray initialized.")

    # Start Ray Serve deployment (if not in local mode)
    if not args.local:
        logger.info("Starting Ray Serve deployment...")
        start_serve(inference_config, checkpoint_path, device)

    # Load dataset
    logger.info("Loading dataset and processing...")
    dataset = PneumothoraxDataset(data_folder=data_dir, transform=transform)
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