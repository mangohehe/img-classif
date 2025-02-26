import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from albumentations.pytorch import ToTensorV2

class PneumothoraxDataset(Dataset):
    def __init__(self, data_folder, mode, transform=None,
                 fold_index=None, folds_distr_path=None):
        
        self.transform = transform
        self.mode = mode
        
        # Path definitions
        self.train_image_path = os.path.join(data_folder, 'train')
        self.train_mask_path = os.path.join(data_folder, 'mask')
        self.test_image_path = os.path.join(data_folder, 'test')
        
        self.fold_index = None
        self.folds_distr_path = folds_distr_path
        self.set_mode(mode, fold_index)
        self.to_tensor = ToTensorV2()

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds = pd.read_csv(self.folds_distr_path)
            folds.fold = folds.fold.astype(str)
            folds = folds[folds.fold != fold_index]
            
            self.train_list = folds.fname.values.tolist()
            self.exist_labels = folds.exist_labels.values.tolist()
            self.num_data = len(self.train_list)

        elif self.mode == 'val':
            folds = pd.read_csv(self.folds_distr_path)
            folds.fold = folds.fold.astype(str)
            folds = folds[folds.fold == fold_index]
            
            self.val_list = folds.fname.values.tolist()
            self.num_data = len(self.val_list)

        elif self.mode == 'test':
            self.test_list = sorted(os.listdir(self.test_image_path))
            self.num_data = len(self.test_list)

    def __getitem__(self, index):
        if self.fold_index is None and self.mode != 'test':
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return
    
        if self.mode == 'test':
            image_path = os.path.join(self.test_image_path, self.test_list[index])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # Normalize
    
            if self.transform:
                sample = {"image": image}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image = sample['image']
            
            image_id = self.test_list[index].replace('.png', '')
            return image_id, image
    
        elif self.mode in ['train', 'val']:
            if self.mode == 'train':
                image_path = os.path.join(self.train_image_path, self.train_list[index])
                mask_path = os.path.join(self.train_mask_path, self.train_list[index])
                label_exists = self.exist_labels[index]
            else:
                image_path = os.path.join(self.train_image_path, self.val_list[index])
                mask_path = os.path.join(self.train_mask_path, self.val_list[index])
                label_exists = 1  # Validation should always have labels
    
            # Load Image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # Normalize
    
            # Load Mask
            if label_exists == 0:
                mask = np.zeros((1024, 1024), dtype=np.float32)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
            # Ensure mask shape consistency
            if mask.shape != (1024, 1024):
                mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    
            # Fix Shape Mismatch - Expand mask dimensions to match the image
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)  # Convert (H, W) → (H, W, 1)
    
            # Apply Transformations
            if self.transform:
                sample = {"image": image, "mask": mask}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image, mask = sample['image'], sample['mask']
    
            return image, mask
             
    def __len__(self):
        return self.num_data


class PneumoSampler(Sampler):
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba
        
        self.folds = pd.read_csv(folds_distr_path)
        self.folds.fold = self.folds.fold.astype(str)
        self.folds = self.folds[self.folds.fold != fold_index].reset_index(drop=True)

        self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        
    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
