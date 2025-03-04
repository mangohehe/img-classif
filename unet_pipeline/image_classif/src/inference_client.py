import requests
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)
DEFAULT_INFERENCE_URL = "http://localhost:8000/PneumothoraxModel"

def send_inference_request(
    images: np.ndarray, 
    use_flip: bool, 
    url: str = DEFAULT_INFERENCE_URL, 
    timeout: int = 30
) -> np.ndarray:
    """Send a batch of images to the Ray Serve deployment for inference.

    Args:
        images (np.ndarray): Batch of images.
        use_flip (bool): Flag to indicate whether to apply horizontal flip.
        url (str, optional): Endpoint URL for the inference service.
        timeout (int, optional): Timeout for the HTTP request in seconds.

    Returns:
        np.ndarray: Inference results as a NumPy array.

    Raises:
        RuntimeError: If the HTTP request fails.
    """
    try:
        response = requests.post(
            url,
            json={"images": images.tolist(), "use_flip": use_flip},
            timeout=timeout
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to send inference request: {e}")
        raise RuntimeError(f"Inference request failed: {e}")
    
    return np.array(response.json())

def run_inference(
    dataset,
    batch_size: int,
    use_flip: bool,
    num_workers: int
) -> Dict[str, np.ndarray]:
    """Run inference on the dataset using the Ray Serve deployment."""
    mask_dict = {}
    total_batches = (dataset.count() + batch_size - 1) // batch_size
    global_index = 0  # Initialize a global counter for image IDs

    # Iterate over the dataset in batches with a progress bar
    for batch in tqdm(
        dataset.iter_batches(batch_size=batch_size),
        total=total_batches,
        desc="Inference",
        unit="batch"
    ):
        logger.debug(f"Processing batch: {batch}")
        images = batch["image"]
        # Generate unique IDs using the global counter
        image_ids = [f"image_{global_index + i}" for i in range(len(images))]
        global_index += len(images)
        masks = send_inference_request(np.array(images), use_flip)
        for name, mask in zip(image_ids, masks):
            mask_dict[name] = mask.astype(np.float32)
    return mask_dict

def save_results(mask_dict: Dict[str, np.ndarray], result_path: Path) -> None:
    """Save the inference results to a file.

    Args:
        mask_dict (Dict[str, np.ndarray]): Dictionary of inference results.
        result_path (Path): File path where the results will be saved.
    """
    logger.info(f"Saving results to {result_path}")
    with open(result_path, 'wb') as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)