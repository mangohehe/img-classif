from ray import serve
import torch
import numpy as np
from typing import Dict, List
import importlib
from pathlib import Path
import logging
from starlette.requests import Request

logger = logging.getLogger(__name__)

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 4})  # Adjust num_cpus as needed
class PneumothoraxModel:
    def __init__(self, model_config: Dict, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_config = model_config
        self.model = self.load_model(model_path).to(self.device)
        self.model.eval()

    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load the model from the given path."""
        logger.info(f"Loading model from: {model_path}")
        if not Path(model_path).exists():
            logger.error(f"Model checkpoint not found at: {model_path}")
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        module = importlib.import_module(self.model_config['MODEL']['PY'])
        model_class = getattr(module, self.model_config['MODEL']['CLASS'])
        model = model_class(**self.model_config['MODEL'].get('ARGS', None))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    async def __call__(self, request: Request) -> List[List[List[float]]]:
        """Handle inference requests."""
        # Parse the JSON body of the request
        request_data = await request.json()
        images = request_data["images"]  # Batch of images
        use_flip = request_data["use_flip"]

        # Convert to tensor and move to device
        images = torch.tensor(images, dtype=torch.float32).to(self.device)
        masks = self.inference_image(images)

        # Apply flip augmentation if enabled
        if use_flip:
            flipped_imgs = torch.flip(images, dims=(3,))
            flipped_masks = self.inference_image(flipped_imgs)
            flipped_masks = np.flip(flipped_masks.cpu().numpy(), axis=2)
            masks = (masks.cpu().numpy() + flipped_masks) / 2
        else:
            masks = masks.cpu().numpy()

        return masks.tolist()

    def inference_image(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of images."""
        with torch.no_grad():
            predicted = self.model(images)
            masks = torch.sigmoid(predicted)
            masks = masks.squeeze(1)
        return masks

def start_serve(model_config: Dict, model_path: str, device: str = "cpu"):
    """Start the Ray Serve deployment."""
    serve.start(http_options={"host": "0.0.0.0"})
    deployment = PneumothoraxModel.bind(model_config, model_path, device)
    serve.run(deployment)