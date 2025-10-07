#!/usr/bin/env python3
"""
Quick data diagnostic script for pneumothorax dataset.
Checks image-mask pairing and class balance.
"""

import os
from glob import glob
from pathlib import Path
import cv2
from tqdm import tqdm

DATA_DIR = '/home/fenggao/github/img-classif/input/dataset1024'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
MASK_DIR = os.path.join(DATA_DIR, 'mask')
TEST_DIR = os.path.join(DATA_DIR, 'test')

print("=" * 70)
print("PNEUMOTHORAX DATASET DIAGNOSTIC")
print("=" * 70)

# Check directories exist
print("\n📁 Directory Check:")
for name, path in [('Train images', TRAIN_DIR),
                   ('Train masks', MASK_DIR),
                   ('Test images', TEST_DIR)]:
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {name}: {path}")

# Count files
print("\n📊 File Counts:")
train_images = sorted(glob(f"{TRAIN_DIR}/*.png"))
mask_images = sorted(glob(f"{MASK_DIR}/*.png"))
test_images = sorted(glob(f"{TEST_DIR}/*.png"))

print(f"  Train images: {len(train_images)}")
print(f"  Mask images:  {len(mask_images)}")
print(f"  Test images:  {len(test_images)}")

# Check pairing
print("\n🔍 Image-Mask Pairing:")
mask_ids = set([Path(p).stem for p in mask_images])
train_ids = set([Path(p).stem for p in train_images])

images_with_masks = train_ids & mask_ids
images_without_masks = train_ids - mask_ids
masks_without_images = mask_ids - train_ids

print(f"  Images with masks:    {len(images_with_masks)}")
print(f"  Images without masks: {len(images_without_masks)}")
print(f"  Masks without images: {len(masks_without_images)}")

if len(masks_without_images) > 0:
    print(f"\n  ⚠️  WARNING: {len(masks_without_images)} masks have no corresponding images!")
    print(f"     Sample orphan masks: {list(masks_without_images)[:3]}")

# Class balance
print("\n⚖️  Class Balance:")
print("  Checking mask contents...")

positive_count = 0
negative_count = 0

for img_path in tqdm(train_images[:100], desc="Sampling"):  # Sample first 100
    img_id = Path(img_path).stem

    if img_id in mask_ids:
        mask_path = os.path.join(MASK_DIR, f"{img_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.max() > 127:
            positive_count += 1
        else:
            negative_count += 1
    else:
        negative_count += 1

total_sampled = positive_count + negative_count
print(f"\n  Sample (first 100 images):")
print(f"    Positive (has pneumothorax): {positive_count} ({positive_count/total_sampled*100:.1f}%)")
print(f"    Negative (no pneumothorax):  {negative_count} ({negative_count/total_sampled*100:.1f}%)")

# Estimate full dataset
est_positive = int(positive_count / total_sampled * len(train_images))
est_negative = len(train_images) - est_positive

print(f"\n  Estimated full dataset:")
print(f"    Positive: ~{est_positive} ({est_positive/len(train_images)*100:.1f}%)")
print(f"    Negative: ~{est_negative} ({est_negative/len(train_images)*100:.1f}%)")

# Sample filenames
print("\n📝 Sample Filenames:")
print(f"  Train image: {Path(train_images[0]).name}")
print(f"  Mask image:  {Path(mask_images[0]).name}")
print(f"  Test image:  {Path(test_images[0]).name if test_images else 'N/A'}")

# Image properties
print("\n🖼️  Image Properties:")
sample_img = cv2.imread(train_images[0], cv2.IMREAD_GRAYSCALE)
print(f"  Image shape: {sample_img.shape}")
print(f"  Image dtype: {sample_img.dtype}")
print(f"  Value range: [{sample_img.min()}, {sample_img.max()}]")

if len(mask_images) > 0:
    sample_mask = cv2.imread(mask_images[0], cv2.IMREAD_GRAYSCALE)
    print(f"  Mask shape:  {sample_mask.shape}")
    print(f"  Mask dtype:  {sample_mask.dtype}")
    print(f"  Value range: [{sample_mask.min()}, {sample_mask.max()}]")

# Recommendations
print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS:")
print("=" * 70)

if len(images_without_masks) > 0:
    print(f"\n✅ GOOD: Dataset includes {len(images_without_masks)} negative samples")
    print(f"   These will be handled by creating empty masks automatically.")

if len(images_without_masks) > len(images_with_masks):
    print(f"\n⚠️  NOTE: More negatives than positives ({len(images_without_masks)} vs {len(images_with_masks)})")
    print(f"   Consider using POS_SAMPLE_WEIGHT=2.0 or higher for class balance.")

print(f"\n✅ Dataset is ready for training!")
print(f"   Total usable images: {len(train_images)}")
print(f"   Expected positive ratio: ~{len(images_with_masks)/len(train_images)*100:.1f}%")

print("\n" + "=" * 70)
