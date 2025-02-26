import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import Sampler
import albumentations as albu
from google.cloud import storage
from PIL import Image
import io

class PneumothoraxDatasetonGCP(Dataset):  # Renamed for consistency
    def __init__(self, data_folder, mode, transform=None, fold_index=None, folds_distr_path=None, bucket=None, test_run=False):
        self.transform = transform
        self.mode = mode
        self.data_folder = data_folder  # Store data folder (for GCS paths)
        self.bucket = bucket  # GCS Bucket (can be None for local files)
        self.max_samples = 20 if test_run else None  # For test runs
        self.folds_distr_path = folds_distr_path
        self.fold_index = fold_index
        self.set_mode(mode, fold_index)
        self.to_tensor = ToTensorV2()

        # Local path definitions (only used if bucket is None)
        self.train_image_path = os.path.join(data_folder, 'train')
        self.train_mask_path = os.path.join(data_folder, 'mask')
        self.test_image_path = os.path.join(data_folder, 'test')

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        folds_df = pd.read_csv(self.folds_distr_path)
        folds_df.fold = folds_df.fold.astype(str)

        if self.mode == 'train':
            folds = folds_df[folds_df.fold != fold_index]
            self.image_list = folds.fname.values.tolist()
            self.exist_labels = folds.exist_labels.values.tolist()
        elif self.mode == 'val':
            folds = folds_df[folds_df.fold == fold_index]
            self.image_list = folds.fname.values.tolist()
            self.exist_labels = [1] * len(self.image_list)  # All validation images have masks
        elif self.mode == 'test':
            if self.bucket: #If bucket exists, load from GCS
                self.image_list = sorted([blob.name for blob in self.bucket.list_blobs(self.data_folder + '/test')])
            else: #Load from local path
                self.image_list = sorted(os.listdir(self.test_image_path))
            self.exist_labels = None  # Test images don't have labels
        self.num_data = len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]

        if self.mode == 'test':
            image = self._load_image(image_name, is_test=True) #Load test image
            if self.transform:
                sample = {"image": image}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image = sample['image']

            image_id = image_name.replace('.png', '')
            return image_id, image

        elif self.mode in ['train', 'val']:
            image = self._load_image(image_name) #Load train/val image
            mask = self._load_mask(image_name) #Load train/val mask

            if self.transform:
                sample = {"image": image, "mask": mask}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image, mask = sample['image'], sample['mask']

            return image, mask

    def _load_image(self, image_name, is_test=False):  # Helper function to load images (local or GCS)
        if self.bucket:  # Load from GCS
            folder = "test" if is_test else "train"
            image_blob = self.bucket.blob(f"{self.data_folder}/{folder}/{image_name}")
            image_data = image_blob.download_as_bytes()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
        else:  # Load from local path
            image_path = os.path.join(self.test_image_path if is_test else self.train_image_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return image

    def _load_mask(self, image_name):  # Helper function to load masks (local or GCS)
        if self.bucket:  # Load from GCS
            mask_blob = self.bucket.blob(f"{self.data_folder.replace('dataset', 'mask')}/train/{image_name}")
            mask_data = mask_blob.download_as_bytes()
            mask = Image.open(io.BytesIO(mask_data))
            mask = np.array(mask).astype(np.uint8)
        else:  # Load from local path
            mask_path = os.path.join(self.train_mask_path, image_name.replace('image', 'mask'))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        mask = np.expand_dims(mask, axis=-1)
        return mask

    def __len__(self):
        if self.max_samples:
            return min(self.num_data, self.max_samples)
        return self.num_data

class PneumoSampleronGCP(Sampler):  # Renamed for consistency
    _folds_df = None  # Class-level attribute

    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequency of non-empty images must be greater than zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba
        self.folds_distr_path = folds_distr_path #Store the folds distribution path

        if PneumoSampler._folds_df is None:  # Read CSV only once
            PneumoSampler._folds_df = pd.read_csv(self.folds_distr_path)
            PneumoSampler._folds_df.fold = PneumoSampler._folds_df.fold.astype(str)

        self.folds = PneumoSampler._folds_df[PneumoSampler._folds_df.fold != fold_index].reset_index(drop=True)

        self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        self.n_positive = len(self.positive_idxs)  # Use len() for clarity
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative, replace=False)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative