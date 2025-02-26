# python /Users/fenggao/Library/CloudStorage/GoogleDrive-2023isanewjourney@gmail.com/My\ Drive/kaggle/img-classif/unet_pipeline/RayInference.py 
# --cfg /Users/fenggao/Library/CloudStorage/GoogleDrive-2023isanewjourney@gmail.com/My\ Drive/kaggle/img-classif/unet_pipeline/experiments/albunet_valid/inference_local.yaml 
# --local
from typing import Optional, Tuple
import argparse
import os
import importlib
import pickle
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm
import albumentations as albu
import torch
from torch.utils.data import DataLoader, Subset
import ray

from Pneumadataset import PneumothoraxDataset
# Assuming these are defined
from utils.helpers import load_yaml, init_seed, init_logger

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the logs directory exists in the script's directory
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Initialize logger with a log file
log_file_path = logs_dir / "inference.log"
logger = init_logger(__name__, log_file_name=str(log_file_path))


def argparser() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Pneumothorax pipeline')
    parser.add_argument(
        '--cfg', type=str, help='Path to the configuration YAML file')
    parser.add_argument(
        '--local', action='store_true', help='Run in local mode (skip GCP and Ray)')
    return parser.parse_args()


def inference_image(model: torch.nn.Module, images: torch.Tensor, device: str) -> np.ndarray:
    """Run inference on a batch of images."""
    images = images.to(device)
    predicted = model(images)
    masks = torch.sigmoid(predicted)
    masks = masks.squeeze(1).cpu().detach().numpy()
    return masks


def check_local_directory_exists(local_data_dir, remote_data_dir):
    """Check if the remote data directory already exists in the local path."""
    local_remote_data_dir = local_data_dir
    if local_remote_data_dir.exists() and any(local_remote_data_dir.iterdir()):
        print(f"Local directory {local_remote_data_dir} already exists.")
        return True
    else:
        print(f"Local directory {local_remote_data_dir} does not exist.")
        return False


def check_remote_directory_exists(bucket_name, remote_data_dir):
    """Check if the remote data directory exists in the GCP bucket."""
    remote_path = f"gs://{bucket_name}/{remote_data_dir}"
    try:
        # Run gsutil to check if the remote directory exists
        result = subprocess.run(['gsutil', 'ls', remote_path],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stdout:
            print(f"Remote directory {remote_path} exists.")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking remote directory {remote_path}: {e}")
        return False
    return False


def copy_folder_from_gcp(bucket_name: str, remote_data_dir: str, local_data_dir: Path) -> None:
    """Copy an entire folder from GCP bucket to local directory using gsutil."""
    if check_local_directory_exists(local_data_dir, remote_data_dir):
        logger.info(
            f"Skipping copy: The directory {remote_data_dir} already exists locally.")
        return

    if not check_remote_directory_exists(bucket_name, remote_data_dir):
        logger.error(
            f"Remote directory {remote_data_dir} does not exist in the bucket.")
        return

    local_data_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"gs://{bucket_name}/{remote_data_dir}"
    local_path = str(local_data_dir)

    try:
        subprocess.run(['gsutil', '-m', 'cp', '-r',
                       remote_path + '/', local_path], check=True)
        logger.info(f"Successfully copied {remote_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error copying {remote_path} to {local_path}: {e}")


@ray.remote(num_cpus=1)
def inference_model_ray(
    model: torch.nn.Module,
    dataset: PneumothoraxDataset,
    data_subset_indices: List[int],
    batch_size: int,
    use_flip: bool,
    num_workers: int,
    device: str
) -> Dict[str, np.ndarray]:
    """Run inference on a subset of data using Ray."""
    dataset_subset = Subset(dataset, data_subset_indices)
    dataloader = DataLoader(
        dataset_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    mask_dict = {}
    subset_size = len(dataset_subset)
    logger.info(f"[Ray Worker] Processing subset of size: {subset_size}")
    for i, (image_ids, images) in enumerate(tqdm(dataloader, desc="Inference on subset")):
        masks = inference_image(model, images, device)
        if use_flip:
            flipped_imgs = torch.flip(images, dims=(3,))
            flipped_masks = inference_image(model, flipped_imgs, device)
            flipped_masks = np.flip(flipped_masks, axis=2)
            masks = (masks + flipped_masks) / 2
        for name, mask in zip(image_ids, masks):
            mask_dict[name] = mask.astype(np.float32)
        logger.info(
            f"[Ray Worker] Processed batch {i + 1} of {len(dataloader)}")
    return mask_dict


def is_directory_writable(directory: str) -> bool:
    try:
        test_file = os.path.join(directory, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except (OSError, IOError):
        return False


def save_results(mask_dict: Dict[str, np.ndarray], result_path: Path) -> None:
    """Save the inference results to a file."""
    logger.info(f"Attempting to save results to: {result_path}")

    if not mask_dict:
        logger.warning("Warning: mask_dict is empty! No results to save.")
        return

    logger.info(f"mask_dict contains {len(mask_dict)} entries.")
    result_dir = result_path.parent

    try:
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Successfully created directory: {result_dir}")
    except Exception as e:
        logger.error(
            f"Error: Failed to create directory {result_dir}. Exception: {e}")
        return

    if not os.access(result_dir, os.W_OK):
        logger.error(f"Error: Directory '{result_dir}' is not writable!")
        return

    try:
        with open(result_path, 'wb') as handle:
            pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Successfully saved results to {result_path}")
    except Exception as e:
        logger.error(f"Error: Failed to save results. Exception: {e}")


def create_output_directory(config: dict, experiment_folder: Path, logger) -> Optional[Tuple[Path, Path]]:
    """
    Create the output directory specified in the config and return the result path.

    Args:
        config (dict): The configuration dictionary.
        experiment_folder (Path): The default folder to use if RESULT_FOLDER is not specified.
        logger: The logger instance for logging messages.

    Returns:
        Optional[Tuple[Path, Path]]: A tuple containing (output_dir, result_path) if successful, None otherwise.
    """
    # Get the output directory from the config, defaulting to experiment_folder
    output_dir = Path(config.get('RESULT_FOLDER', experiment_folder))
    result_path = output_dir / config['RESULT']

    # Ensure the output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return None

    # Check if the directory is writable
    if not is_directory_writable(output_dir):
        logger.error(f"The directory '{output_dir}' is NOT writable.")
        return None

    return output_dir, result_path


def main() -> None:
    """Main function to run the inference pipeline."""
    args = argparser()
    config_path = Path(args.cfg).resolve()
    experiment_folder = config_path.parents[0]
    inference_config = load_yaml(config_path)
    logger.info(f"Loaded inference config: {inference_config}")

    # Create the output directory and get the result path
    output_info = create_output_directory(
        inference_config, experiment_folder, logger)
    if output_info is None:
        return  # Exit if the output directory could not be created or is not writable
    _, result_path = output_info

    # Load configuration parameters
    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']
    num_workers = inference_config['NUM_WORKERS']
    transform = albu.load(inference_config['TEST_TRANSFORMS'])
    use_flip = inference_config['FLIP']
    local_data_dir = Path(inference_config['LOCAL_DATA_DIR'])
    local_checkpoint_path = Path(
        inference_config['CHECKPOINTS']['LOCAL_CHECKPOINT_DIR'])

    if args.local:
        logger.info("Running in local mode. Skipping GCP and Ray.")
    else:
        logger.info("Running in cloud mode.")
        bucket_name = inference_config['GCP_BUCKET_NAME']
        remote_data_dir = inference_config['GCP_DATA_DIR']
        remote_checkpoint_path = inference_config['GCP_CHECKPOINT_PATH']

        # Copy test data and checkpoint from GCP
        copy_folder_from_gcp(bucket_name, remote_data_dir, local_data_dir)
        copy_folder_from_gcp(
            bucket_name, remote_checkpoint_path, local_checkpoint_path)

    # Load model
    module = importlib.import_module(inference_config['MODEL']['PY'])
    model_class = getattr(module, inference_config['MODEL']['CLASS'])
    model = model_class(
        **inference_config['MODEL'].get('ARGS', None)).to(device)
    model.eval()

    ray.init(
        address="auto",
        runtime_env={
            "working_dir": SCRIPT_DIR,
            "excludes": [
                "data/*",       # Exclude the data directory
                "logs/*",       # Exclude the logs directory
                "results/*",    # Exclude the results directory
                "*.pth",        # Exclude model checkpoints
                "*.pt",         # Exclude PyTorch files
                "*.zip",        # Exclude zip files
                "*.tar.gz",     # Exclude compressed files
                "fold."
            ]
        }
    )

    # Load dataset
    dataset = PneumothoraxDataset(
        data_folder=local_data_dir, mode='test', transform=transform)

    if args.local:
        # Run inference locally without Ray
        logger.info("Running inference locally.")
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        mask_dict = {}

        for i, (image_ids, images) in enumerate(tqdm(dataloader, desc="Inference")):
            masks = inference_image(model, images, device)
            if use_flip:
                flipped_imgs = torch.flip(images, dims=(3,))
                flipped_masks = inference_image(model, flipped_imgs, device)
                flipped_masks = np.flip(flipped_masks, axis=2)
                masks = (masks + flipped_masks) / 2
            for name, mask in zip(image_ids, masks):
                mask_dict[name] = mask.astype(np.float32)
            logger.info(f"Processed batch {i + 1} of {len(dataloader)}")
    else:
        # Split dataset for parallel processing with Ray
        num_splits = inference_config['NUM_SPLITS']
        split_indices = np.array_split(range(len(dataset)), num_splits)

        # Load model checkpoint
        checkpoint = local_checkpoint_path + "/" + \
            inference_config['CHECKPOINTS']['CHECKPOINT_NAME']
        model.load_state_dict(torch.load(
            checkpoint, map_location=torch.device(device)))
        model.eval()

        # Submit Ray tasks
        futures = [
            inference_model_ray.remote(
                model, dataset, indices, batch_size, use_flip, num_workers, device)
            for indices in split_indices
        ]

        # Collect results
        mask_dict = defaultdict(int)
        for i, future in enumerate(futures):
            current_mask_dict = ray.get(future)
            mask_dict.update(current_mask_dict)
            logger.info(f"Processed subset {i + 1} of {len(futures)}")

        ray.shutdown()

    # Save results
    save_results(mask_dict, result_path)
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
