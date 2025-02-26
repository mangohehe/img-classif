import os
import numpy as np
import cv2
import pandas as pd
import ray
from ray.data import Dataset
from albumentations.pytorch import ToTensorV2

# Initialize Ray
if not ray.is_initialized():
    ray.init()

class PneumothoraxRayDataset:
    def __init__(self, data_folder, mode, transform=None, fold_index=None, folds_distr_path=None, max_samples=None):
        self.transform = transform
        self.mode = mode
        self.fold_index = fold_index
        self.folds_distr_path = folds_distr_path
        self.to_tensor = ToTensorV2()
        self.max_samples = max_samples  # Optional: Limit the number of samples for debugging

        # Path definitions
        self.train_image_path = os.path.join(data_folder, 'train')
        self.train_mask_path = os.path.join(data_folder, 'mask')
        self.test_image_path = os.path.join(data_folder, 'test')

        # Load data into Ray Dataset
        self.set_mode(mode, fold_index)
        self.dataset = self._load_dataset()

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds_df = pd.read_csv(self.folds_distr_path)
            folds_df.fold = folds_df.fold.astype(str)
            folds = folds_df[folds_df.fold != fold_index]
            self.image_list = folds.fname.values.tolist()
            self.exist_labels = folds.exist_labels.values.tolist()
        elif self.mode == 'val':
            folds_df = pd.read_csv(self.folds_distr_path)
            folds_df.fold = folds_df.fold.astype(str)
            folds = folds_df[folds_df.fold == fold_index]
            self.image_list = folds.fname.values.tolist()
            self.exist_labels = [1] * len(self.image_list)  # All validation images have masks
        elif self.mode == 'test':
            self.image_list = sorted(os.listdir(self.test_image_path))
            self.exist_labels = None  # Test images don't have labels
        self.num_data = len(self.image_list)

    def _load_dataset(self):
        if self.mode == 'test':
            # Load test images
            test_image_paths = [os.path.join(self.test_image_path, fname) for fname in self.image_list]
            dataset = ray.data.from_items(test_image_paths)
        else:
            # Load train/val data
            image_paths = [os.path.join(self.train_image_path, fname) for fname in self.image_list]
            mask_paths = [os.path.join(self.train_mask_path, fname) for fname in self.image_list]
            exist_labels = self.exist_labels

            # Create a Ray Dataset from image paths, mask paths, and labels
            dataset = ray.data.from_items(list(zip(image_paths, mask_paths, exist_labels)))
        
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
        
        return image

    def preprocess_mask(self, mask_path, label_exists):
        # If mask_path is a dictionary, extract the actual path
        if isinstance(mask_path, dict) and 'item' in mask_path:
            mask_path = mask_path['item']
        
        if label_exists == 0:
            mask = np.zeros((1024, 1024), dtype=np.float32)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if mask.shape != (1024, 1024):
                mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)  # Convert (H, W) → (H, W, 1)
        return mask

    def transform_sample(self, sample):
        # Apply transformations to the sample
        if self.transform:
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
        return sample

    def get_dataset(self):
        if self.mode == 'test':
            # Preprocess test images
            return self.dataset.map(lambda x: {"image": self.preprocess_image(x)})
        else:
            # Preprocess train/val images and masks
            return self.dataset.map(lambda x: {
                "image": self.preprocess_image(x[0]),
                "mask": self.preprocess_mask(x[1], x[2])
            })

    def __len__(self):
        # Use .count() to get the number of items in the dataset
        if self.max_samples:
            return min(self.dataset.count(), self.max_samples)
        return self.dataset.count()

    def __getitem__(self, index):
        # This method is not needed for Ray Dataset, but kept for compatibility
        raise NotImplementedError("Use .get_dataset() and iterate over the Ray Dataset instead.")