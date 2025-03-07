import sys
import os
import argparse
import yaml
import logging
import albumentations as albu
from pathlib import Path
import ray 

# Relative imports from the current package
from .serve_deployment import start_serve
from .inference_client import run_inference, save_results
from .dataset import PneumothoraxRayDatasetOnGCP as PneumothoraxDataset

# Optionally, add the parent directory to sys.path for relative imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Pneumothorax pipeline')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the configuration YAML file')
    parser.add_argument('--transformer', type=str,
                        help='Path to the transformation file')
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    """Load the YAML configuration file."""
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def load_transform(transform_path: Path):
    """Load transformation configuration using Albumentations."""
    return albu.load(transform_path.resolve())


def resolve_gcp_path(base: str, sub: str) -> str:
    """Resolve a Google Cloud Storage path."""
    return f"gs://{base}/{sub}"


def create_logger(log_dir: Path, log_file_name: str) -> logging.Logger:
    """Create and configure a logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_file_name
    if log_file.exists():
        log_file.unlink()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'
    )
    file_handler = logging.FileHandler(filename=str(log_file))
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def main() -> None:
    """Run the Pneumothorax inference pipeline."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logger
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"
    logger = create_logger(logs_dir, "inference.log")

    # Load configuration file
    config_path = Path(args.model).resolve()
    config = load_yaml_config(config_path)
    logger.info(f"Loaded configuration: {config}")

    # **Initialize Ray and connect to the existing cluster**
    ray.init(address="auto")

    # Load transformation if provided
    if args.transformer:
        transform = load_transform(Path(args.transformer).resolve())
        logger.info(f"Loaded transformation configuration from {args.transformer}")
    else:
        transform = None
        logger.info("No transformation configuration provided; proceeding without transformation.")

    # Extract parameters from configuration with safe defaults
    batch_size = config.get('BATCH_SIZE', 1)
    device = config.get('DEVICE', 'cpu')
    num_workers = config.get('NUM_WORKERS', 1)
    use_flip = config.get('FLIP', False)

    # Resolve GCP paths for dataset, model, and results
    dataset_cfg = config.get('DATASET', {})
    model_cfg = config.get('MODEL', {}).get('CHECKPOINTS', {})
    results_cfg = config.get('RESULTS', {})

    dataset_path = resolve_gcp_path(dataset_cfg.get('DATASET_DIR', ''), dataset_cfg.get('DATASET_NAME', ''))
    model_path = resolve_gcp_path(model_cfg.get('GCS_CHECKPOINT_DIR', ''), model_cfg.get('CHECKPOINT_NAME', ''))
    result_path = resolve_gcp_path(results_cfg.get('RESULTS_DIR', ''), results_cfg.get('RESULTS_NAME', ''))

    logger.info(f"Resolved dataset path: {dataset_path}")
    logger.info(f"Resolved model path: {model_path}")
    logger.info(f"Resolved result path: {result_path}")

    # Start Ray Serve deployment
    logger.info("Starting Ray Serve deployment...")
    start_serve(config, model_path, device)
    # Note: start_serve may block execution. If non-blocking behavior is desired,
    # consider running it in a separate thread or process.

    # Load dataset and prepare for inference
    logger.info("Loading and processing dataset...")
    dataset = PneumothoraxDataset(data_folder=dataset_path, transform=transform)
    ray_dataset = dataset.get_dataset()

    # Run inference and save results
    logger.info("Running inference...")
    mask_dict = run_inference(ray_dataset, batch_size, use_flip, num_workers)
    logger.info("Inference completed.")

    logger.info("Saving results...")
    save_results(mask_dict, result_path)
    logger.info("Results saved successfully. Pipeline complete.")


if __name__ == "__main__":
    main()