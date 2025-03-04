import os
import numpy as np
import cv2
import ray
from ray.data import Dataset
from albumentations.pytorch import ToTensorV2

# Initialize Ray
if not ray.is_initialized():
    ray.init()

class PneumothoraxRayDataset:
    def __init__(self, data_folder, transform=None, max_samples=None):
        self.transform = transform
        self.to_tensor = ToTensorV2()
        self.max_samples = max_samples  # Optional: Limit the number of samples for debugging

        # Path definitions
        self.test_image_path = os.path.join(data_folder, 'test')

        # Load data into Ray Dataset
        self.image_list = sorted(os.listdir(self.test_image_path))
        self.num_data = len(self.image_list)
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        # Load test images
        test_image_paths = [os.path.join(self.test_image_path, fname) for fname in self.image_list]
        dataset = ray.data.from_items(test_image_paths)
        
        print("Dataset sample:", dataset.take(1))  # Inspect the dataset structure
        return dataset

    def preprocess_image(self, image_path):
        # If image_path is a dictionary, extract the actual path
        if isinstance(image_path, dict) and 'item' in image_path:
            image_path = image_path['item']
        
        if not isinstance(image_path, str) or not os.path.exists(image_path):
            raise ValueError(f"Invalid image path: {image_path}")
        
        print(f"Processing image: {image_path}")
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
        # Apply transformations to the sample
        print(f"Transforming sample")
        if self.transform:
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
        print(f"Transformed sample")
        return sample

    def get_dataset(self):
        # Preprocess test images and apply transformations

        #return self.dataset.map(lambda x: {"image": self.preprocess_image(x)})
        return self.dataset.map(lambda x: {
            "image": self.transform_sample({"image": self.preprocess_image(x)})["image"]
        })

    def __len__(self):
        # Use .count() to get the number of items in the dataset
        if self.max_samples:
            return min(self.dataset.count(), self.max_samples)
        return self.dataset.count()

    def __getitem__(self, index):
        # This method is not needed for Ray Dataset, but kept for compatibility
        raise NotImplementedError("Use .get_dataset() and iterate over the Ray Dataset instead.")