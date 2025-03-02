# python ray_server_gcp/Main.py --cfg ray_server_gcp/inference_gcp.yaml --transform ray_server_gcp/valid_transforms_1024_old.json
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
import yaml
import albumentations as albu
from .ServeDeployment import start_serve
from .InferenceClient import run_inference, save_results
from .PneumothoraxRayDatasetOnGCP import PneumothoraxRayDatasetOnGCP as PneumothoraxDataset
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
    parser.add_argument('--transform', type=str, help='Path to the transformation YAML file')
    return parser.parse_args()

def load_config(config_path: Path) -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_gcp_path(base_path: str, sub_path: str) -> str:
    """Resolve a path on GCS."""
    return f"gs://{base_path}/{sub_path}"

def main() -> None:
    """Main function to run the inference pipeline."""
    args = argparser()
    
    # Load configuration
    config_path = Path(args.cfg).resolve()  # Resolve to absolute path
    inference_config = load_config(config_path)
    logger.info(f"Loaded inference config: {inference_config}")
    
    # Load transformation
    transform = albu.load(Path(args.transform).resolve())
    logger.info(f"Loaded transformation: {transform}")   
    

    # Load parameters
    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']  
    num_workers = inference_config['NUM_WORKERS']
    use_flip = inference_config['FLIP']

    # Resolve directories
    dataset_path = resolve_gcp_path(inference_config['DATASET']['DATASET_DIR'], inference_config['DATASET']['DATASET_NAME'])
    model_path = resolve_gcp_path(inference_config['MODEL']['CHECKPOINTS']['GCS_CHECKPOINT_DIR'], inference_config['MODEL']['CHECKPOINTS']['CHECKPOINT_NAME'])
    result_path = resolve_gcp_path(inference_config['RESULTS']['RESULTS_DIR'], inference_config['RESULTS']['RESULTS_NAME'])
    logger.info(f"Resolved dataset path: {dataset_path}")
    logger.info(f"Resolved model path: {model_path}")
    logger.info(f"Resolved result path: {result_path}")

    # Start Ray Serve deployment 
    logger.info("Starting Ray Serve deployment...")
    start_serve(inference_config, model_path, device)

    # Load dataset
    logger.info("Loading dataset and processing...")
    dataset = PneumothoraxDataset(data_folder=dataset_path, transform=transform)
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