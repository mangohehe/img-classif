import os
import numpy as np
import cv2
import ray
from ray.data import Dataset
from albumentations.pytorch import ToTensorV2
import gcsfs  # For GCS support

class PneumothoraxRayDatasetOnGCP:
    def __init__(self, data_folder, transform=None, max_samples=None, use_gcs=False):
        """
        Args:
            data_folder (str): Path to the data folder (local or GCS).
            transform (callable, optional): Transformations to apply to the images.
            max_samples (int, optional): Maximum number of samples to load (for debugging).
            use_gcs (bool): Whether to use Google Cloud Storage (GCS) for data access.
        """
        self.transform = transform
        self.to_tensor = ToTensorV2()
        self.max_samples = max_samples  # Optional: Limit the number of samples for debugging
        self.use_gcs = use_gcs  # Whether to use GCS

        # Initialize GCS filesystem if using GCS
        if self.use_gcs:
            self.fs = gcsfs.GCSFileSystem()
        else:
            self.fs = None  # Use local filesystem

        # Path definitions
        self.test_image_path = os.path.join(data_folder, 'test')

        # Load data into Ray Dataset
        if self.use_gcs:
            self.image_list = sorted(self.fs.ls(self.test_image_path))
        else:
            self.image_list = sorted(os.listdir(self.test_image_path))
        self.num_data = len(self.image_list)
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        """Load test images into a Ray Dataset."""
        if self.use_gcs:
            test_image_paths = [f"gs://{fname}" for fname in self.image_list]
        else:
            test_image_paths = [os.path.join(self.test_image_path, fname) for fname in self.image_list]
        dataset = ray.data.from_items(test_image_paths)
        
        print("Dataset sample:", dataset.take(1))  # Inspect the dataset structure
        return dataset

    def preprocess_image(self, image_path):
        """
        Preprocess an image (read, normalize, and reshape).

        Args:
            image_path (str or dict): Path to the image or a dictionary containing the path.

        Returns:
            np.ndarray: Preprocessed image in CHW format.
        """
        # If image_path is a dictionary, extract the actual path
        if isinstance(image_path, dict) and 'item' in image_path:
            image_path = image_path['item']
        
        if not isinstance(image_path, str):
            raise ValueError(f"Invalid image path: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Read image from GCS or local filesystem
        if self.use_gcs:
            with self.fs.open(image_path, 'rb') as f:
                image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
        
        # Convert image to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Reshape to [height, width, channels] -> [channels, height, width]
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW format
        print(f"Processed image")
        return image

    def transform_sample(self, sample):
        """
        Apply transformations to a sample.

        Args:
            sample (dict): A dictionary containing the image.

        Returns:
            dict: Transformed sample.
        """
        print(f"Transforming sample")
        if self.transform:
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
        print(f"Transformed sample")
        return sample

    def get_dataset(self):
        """
        Preprocess test images and apply transformations.

        Returns:
            ray.data.Dataset: A Ray Dataset containing preprocessed and transformed images.
        """
        return self.dataset.map(lambda x: {
            "image": self.transform_sample({"image": self.preprocess_image(x)})["image"]
        })

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        if self.max_samples:
            return min(self.dataset.count(), self.max_samples)
        return self.dataset.count()

    def __getitem__(self, index):
        """
        This method is not needed for Ray Dataset, but kept for compatibility.
        """
        raise NotImplementedError("Use .get_dataset() and iterate over the Ray Dataset instead.")