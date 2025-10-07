#!/usr/bin/env python3
"""
Create a matched dataset with only images that have corresponding masks.
This matches the competition notebook approach.
"""

import os
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = '/home/fenggao/github/img-classif/input/dataset1024'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
MASK_DIR = os.path.join(BASE_DIR, 'mask')

# New matched dataset directory
MATCHED_DIR = '/home/fenggao/github/img-classif/input/dataset1024_matched'
MATCHED_TRAIN_DIR = os.path.join(MATCHED_DIR, 'train')
MATCHED_MASK_DIR = os.path.join(MATCHED_DIR, 'mask')

# Create directories
os.makedirs(MATCHED_TRAIN_DIR, exist_ok=True)
os.makedirs(MATCHED_MASK_DIR, exist_ok=True)

print("="*70)
print("CREATING MATCHED DATASET (Competition Approach)")
print("="*70)

# Get all mask files (these are the positive cases)
mask_files = glob(f"{MASK_DIR}/*.png")
mask_ids = set([Path(f).stem for f in mask_files])

print(f"\nFound {len(mask_ids)} masks")

# Find corresponding train images
train_files = glob(f"{TRAIN_DIR}/*.png")
train_dict = {Path(f).stem: f for f in train_files}

print(f"Found {len(train_files)} train images")

# Create matched pairs
matched_count = 0
missing_train = []

print("\nCreating matched pairs...")
for mask_id in tqdm(sorted(mask_ids)):
    mask_path = os.path.join(MASK_DIR, f"{mask_id}.png")

    if mask_id in train_dict:
        train_path = train_dict[mask_id]

        # Copy files to new directory
        shutil.copy2(train_path, os.path.join(MATCHED_TRAIN_DIR, f"{mask_id}.png"))
        shutil.copy2(mask_path, os.path.join(MATCHED_MASK_DIR, f"{mask_id}.png"))

        matched_count += 1
    else:
        missing_train.append(mask_id)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"✅ Successfully matched: {matched_count} pairs")

if missing_train:
    print(f"⚠️  Missing train images: {len(missing_train)}")
    print(f"   First few: {missing_train[:5]}")
else:
    print(f"✅ All masks have corresponding train images")

print(f"\nNew dataset location: {MATCHED_DIR}")
print(f"  - Train: {MATCHED_TRAIN_DIR}")
print(f"  - Mask: {MATCHED_MASK_DIR}")

# Verify counts
final_train_count = len(glob(f"{MATCHED_TRAIN_DIR}/*.png"))
final_mask_count = len(glob(f"{MATCHED_MASK_DIR}/*.png"))

print(f"\nFinal verification:")
print(f"  Train images: {final_train_count}")
print(f"  Mask images: {final_mask_count}")

if final_train_count == final_mask_count:
    print(f"  ✅ Perfect match! Ready for training.")
else:
    print(f"  ❌ Mismatch detected!")

print("="*70)
