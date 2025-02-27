import requests
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from typing import Dict
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

def send_inference_request(images: np.ndarray, use_flip: bool) -> np.ndarray:
    """Send a batch of images to the Ray Serve deployment for inference."""
    response = requests.post(
        "http://localhost:8000/PneumothoraxModel",
        json={"images": images.tolist(), "use_flip": use_flip}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Inference request failed: {response.text}")
    return np.array(response.json())

def run_inference(
    dataset,
    batch_size: int,
    use_flip: bool,
    num_workers: int
) -> Dict[str, np.ndarray]:
    """Run inference on the dataset using the Ray Serve deployment."""
    mask_dict = {}

    # Calculate total number of batches
    total_batches = (dataset.count() + batch_size - 1) // batch_size

    # Iterate over the dataset in batches with a progress bar
    for batch in tqdm(dataset.iter_batches(batch_size=batch_size), total=total_batches, desc="Inference", unit="batch"):
        print(batch)  # Debug: Check if batches are being yielded
        images = batch["image"]
        image_ids = [f"image_{i}" for i in range(len(images))]
        masks = send_inference_request(np.array(images), use_flip)
        for name, mask in zip(image_ids, masks):
            mask_dict[name] = mask.astype(np.float32)
    return mask_dict

def save_results(mask_dict: Dict[str, np.ndarray], result_path: Path) -> None:
    """Save the inference results to a file."""
    logger.info(f"Saving results to {result_path}")
    with open(result_path, 'wb') as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)