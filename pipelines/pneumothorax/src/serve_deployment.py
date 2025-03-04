from ray import serve
import torch
import numpy as np
from typing import Dict, Any, List
import importlib
from pathlib import Path
import logging
from starlette.requests import Request
import gcsfs

logger = logging.getLogger(__name__)

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2})
class PneumothoraxModel:
    def __init__(self, model_config: Dict[str, Any], model_path: str, device: str = "cpu"):
        """
        Initialize the PneumothoraxModel deployment.

        Args:
            model_config (Dict[str, Any]): Configuration dictionary for the model.
            model_path (str): Path to the model checkpoint (local or GCS).
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model_config = model_config
        self.model = self.load_model(model_path).to(self.device)
        self.model.eval()

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the model from the given checkpoint path.

        Supports both local filesystem and GCS paths.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: The loaded model.

        Raises:
            FileNotFoundError: If the local model checkpoint does not exist.
        """
        logger.info(f"Loading model from: {model_path}")

        # Load checkpoint from GCS or local filesystem
        if model_path.startswith("gs://"):
            try:
                fs = gcsfs.GCSFileSystem()
                with fs.open(model_path, "rb") as f:
                    checkpoint = torch.load(f, map_location=self.device)
            except Exception as e:
                logger.error(f"Failed to load model from GCS: {e}")
                raise e
        else:
            checkpoint_path = Path(model_path)
            if not checkpoint_path.exists():
                logger.error(f"Model checkpoint not found at: {model_path}")
                raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

        # Load model architecture based on configuration
        try:
            module = importlib.import_module(self.model_config['MODEL']['PY'])
            model_class = getattr(module, self.model_config['MODEL']['CLASS'])
            model_args = self.model_config['MODEL'].get('ARGS', {})
            model = model_class(**model_args)
        except Exception as e:
            logger.error(f"Failed to initialize model architecture: {e}")
            raise e

        # Load model weights
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

        return model

    async def __call__(self, request: Request) -> List[List[List[float]]]:
        """
        Handle inference requests.

        Expects a JSON payload with 'images' and 'use_flip' keys.

        Args:
            request (Request): The incoming HTTP request.

        Returns:
            List[List[List[float]]]: Inference results as a nested list.
        """
        try:
            request_data = await request.json()
            images = request_data["images"]
            use_flip = request_data["use_flip"]
        except Exception as e:
            logger.error(f"Invalid request format: {e}")
            return []

        # Convert images to a tensor and move to the designated device
        images_tensor = torch.tensor(images, dtype=torch.float32).to(self.device)
        masks = self.inference_image(images_tensor)

        # Apply horizontal flip augmentation if enabled
        if use_flip:
            flipped_imgs = torch.flip(images_tensor, dims=(3,))
            flipped_masks = self.inference_image(flipped_imgs)
            flipped_masks_np = np.flip(flipped_masks.cpu().numpy(), axis=2)
            masks_np = (masks.cpu().numpy() + flipped_masks_np) / 2
        else:
            masks_np = masks.cpu().numpy()

        return masks_np.tolist()

    def inference_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of images.

        Args:
            images (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Inference results.
        """
        with torch.no_grad():
            predicted = self.model(images)
            masks = torch.sigmoid(predicted)
            masks = masks.squeeze(1)
        return masks

def start_serve(model_config: Dict[str, Any], model_path: str, device: str = "cpu"):
    """
    Start the Ray Serve deployment.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model.
        model_path (str): Path to the model checkpoint (local or GCS).
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    serve.start(http_options={"host": "0.0.0.0"})
    deployment = PneumothoraxModel.bind(model_config, model_path, device)
    serve.run(deployment)