#!/usr/bin/env python3
"""
Quick script to check if the dataset is properly set up for local training/inference.
"""

import os
from pathlib import Path

def check_dataset_setup():
    """Check if the dataset directory structure is correct."""

    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "input" / "dataset1024"
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    print("=" * 60)
    print("Dataset Setup Checker")
    print("=" * 60)
    print(f"\nProject root: {project_root}")
    print(f"Expected dataset location: {dataset_path}")

    # Check if dataset directory exists
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset directory not found!")
        print(f"   Expected: {dataset_path}")
        print(f"\n   To fix this, please:")
        print(f"   1. Create the directory: mkdir -p {dataset_path}")
        print(f"   2. Download/copy your dataset to this location")
        print(f"   3. See DATASET_SETUP.md for detailed instructions")
        return False
    else:
        print(f"\n✓ Dataset directory exists: {dataset_path}")

    # Check train directory
    if not train_path.exists():
        print(f"\n❌ ERROR: Training images directory not found!")
        print(f"   Expected: {train_path}")
        return False
    else:
        train_images = list(train_path.glob("*.png")) + list(train_path.glob("*.jpg"))
        print(f"✓ Training directory exists: {train_path}")
        print(f"  Found {len(train_images)} training images")

        if len(train_images) == 0:
            print(f"  ⚠️  WARNING: No images found in training directory!")

    # Check test directory
    if not test_path.exists():
        print(f"\n❌ ERROR: Test images directory not found!")
        print(f"   Expected: {test_path}")
        return False
    else:
        test_images = list(test_path.glob("*.png")) + list(test_path.glob("*.jpg"))
        print(f"✓ Test directory exists: {test_path}")
        print(f"  Found {len(test_images)} test images")

        if len(test_images) == 0:
            print(f"  ⚠️  WARNING: No images found in test directory!")

    # Check CSV files
    csv_files = {
        "train-rle.csv": project_root / "input" / "train-rle.csv",
        "stage_2_train.csv": project_root / "input" / "stage_2_train.csv",
    }

    print("\n" + "-" * 60)
    print("CSV Files:")
    for name, path in csv_files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✓ {name} ({size_kb:.1f} KB)")
        else:
            print(f"⚠️  {name} not found (optional)")

    # Summary
    print("\n" + "=" * 60)
    if train_path.exists() and test_path.exists():
        if len(train_images) > 0 and len(test_images) > 0:
            print("✅ Dataset setup looks good! You can proceed with training/inference.")
        else:
            print("⚠️  Dataset directories exist but no images found.")
            print("   Please add images to the train/ and test/ directories.")
    else:
        print("❌ Dataset setup incomplete. See DATASET_SETUP.md for instructions.")
    print("=" * 60)

    return True

if __name__ == "__main__":
    check_dataset_setup()
