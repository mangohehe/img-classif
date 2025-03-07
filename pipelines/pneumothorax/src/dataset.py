import os
import numpy as np
import cv2
import ray
from ray.data import Dataset
from albumentations.pytorch import ToTensorV2
import gcsfs  # For GCS support
import logging
from typing import Callable, Optional, Union, Dict

logger = logging.getLogger(__name__)

#   Overall, this class creates a pipeline to:
	#	Load image paths from a GCS bucket.
	#	Create a Ray Dataset from these paths.
	#	Preprocess images (read, decode, normalize, and reformat).
	#	Apply Albumentations transformations.
        # Resize
        # Bounding Box
        # Augmentation
	#	Allow iteration over a preprocessed and optionally limited dataset.

class PneumothoraxRayDatasetOnGCP:
    def __init__(
        self, 
        data_folder: str, 
        transform: Optional[Callable] = None, 
        max_samples: Optional[int] = None
    ):
        """
        Initialize the dataset loader.

        Args:
            data_folder (str): Path to the data folder (local or GCS).
            transform (Callable, optional): Transformation function to apply to images.
            max_samples (int, optional): Maximum number of samples to load (for debugging or smaller runs).
        """
        self.transform = transform
        self.to_tensor = ToTensorV2()
        self.max_samples = max_samples
        self.fs = gcsfs.GCSFileSystem()

        self.test_image_path = data_folder
        logger.info(f"Loading image list from data folder: {self.test_image_path}")

        try:
            self.image_list = sorted(self.fs.ls(self.test_image_path))
        except Exception as e:
            logger.error(f"Failed to list images in {self.test_image_path}: {e}")
            raise e

        self.num_data = len(self.image_list)
        logger.info(f"Found {self.num_data} images in {self.test_image_path}")

        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dataset:
        """
        Load test images into a Ray Dataset.

        Returns:
            Dataset: A Ray Dataset containing image paths.
        """
        # Ensure each path is a proper GCS path.
        test_image_paths = [
            fname if fname.startswith("gs://") else f"gs://{fname}" 
            for fname in self.image_list
        ]

        dataset = ray.data.from_items(test_image_paths)
        sample = dataset.take(1)
        logger.debug(f"Dataset sample: {sample}")
        return dataset

    def preprocess_image(self, image_path: Union[str, Dict]) -> np.ndarray:
        """
        Preprocess an image by reading, normalizing, and reshaping it.

        Args:
            image_path (str or dict): Path to the image or a dictionary containing the path.

        Returns:
            np.ndarray: Preprocessed image in CHW format.
        """
        # If image_path is a dictionary, extract the actual path.
        if isinstance(image_path, dict) and 'item' in image_path:
            image_path = image_path['item']
        
        if not isinstance(image_path, str):
            error_msg = f"Invalid image path: {image_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Processing image: {image_path}")
        
        try:
            with self.fs.open(image_path, 'rb') as f:
                image_data = f.read()
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            raise e

        if image is None:
            error_msg = f"Failed to load image at path: {image_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert BGR to RGB, normalize, and convert to float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Transpose from HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        logger.debug("Image processed successfully")
        return image

    def transform_sample(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply transformations to a sample.

        Args:
            sample (dict): A dictionary containing the image with key 'image'.

        Returns:
            dict: Transformed sample.
        """
        logger.debug("Transforming sample")
        if self.transform:
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
        logger.debug("Sample transformed")
        return sample

    def get_dataset(self) -> Dataset:
        """
        Preprocess test images and apply transformations.

        Returns:
            Dataset: A Ray Dataset containing preprocessed and transformed images.
        """
        def process_item(item: str) -> Dict[str, np.ndarray]:
            image = self.preprocess_image(item)
            transformed = self.transform_sample({"image": image})
            return {"image": transformed["image"]}

        dataset = self.dataset.map(process_item)

        # Optionally limit the dataset to max_samples
        if self.max_samples is not None:
            dataset = dataset.limit(self.max_samples)
        return dataset

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        total = self.dataset.count()
        return min(total, self.max_samples) if self.max_samples is not None else total

    def __getitem__(self, index: int):
        """
        This method is not needed for Ray Dataset, but kept for compatibility.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use .get_dataset() and iterate over the Ray Dataset instead.")