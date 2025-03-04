import pytest
import numpy as np
import torch
import pickle
import cv2
from io import BytesIO
from pathlib import Path

# Import functions from your source modules
from src.inference_client import send_inference_request, run_inference, save_results
from src.dataset import PneumothoraxRayDatasetOnGCP

# ----------------------------
# Dummy classes and functions for testing
# ----------------------------

class DummyResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP error with status code {}".format(self.status_code))

    def json(self):
        return self._data


def dummy_post(url, json, timeout):
    # For testing, simply echo back the images as "inference results"
    images = json.get("images", [])
    return DummyResponse(200, images)


class DummyDataset:
    """A dummy dataset to simulate Ray Dataset behavior for inference testing."""
    def __init__(self, num_items, batch_size):
        self.num_items = num_items
        self.batch_size = batch_size

    def count(self):
        return self.num_items

    def iter_batches(self, batch_size):
        # Yield batches of dummy images (randomly generated)
        for i in range(0, self.num_items, batch_size):
            batch_size_actual = min(batch_size, self.num_items - i)
            # Each batch is a dict with key "image": list of images.
            batch = {"image": [np.random.rand(3, 224, 224) for _ in range(batch_size_actual)]}
            yield batch


class DummyGCSFS:
    """A dummy GCS filesystem for testing the dataset module."""
    def ls(self, path):
        # Return dummy file paths (simulate two images)
        return [f"{path}/image1.jpg", f"{path}/image2.jpg"]

    def open(self, path, mode):
        # Create a dummy image: black image 100x100, encode as JPEG.
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        ret, buf = cv2.imencode(".jpg", dummy_image)
        if not ret:
            raise ValueError("Failed to encode image")
        return BytesIO(buf.tobytes())

# ----------------------------
# Test Functions
# ----------------------------

def test_send_inference_request(monkeypatch):
    # Monkey-patch requests.post in the inference_client module
    monkeypatch.setattr("src.inference_client.requests.post", dummy_post)
    
    # Create a dummy images array
    dummy_images = np.random.rand(2, 3, 224, 224)
    use_flip = False

    # Call the function; it should return an array equal to dummy_images
    result = send_inference_request(dummy_images, use_flip)
    np.testing.assert_allclose(result, dummy_images)


def test_run_inference(monkeypatch):
    # Use the same monkey-patching as in test_send_inference_request
    monkeypatch.setattr("src.inference_client.requests.post", dummy_post)

    # Create a dummy dataset with 4 items and a batch size of 2
    batch_size = 2
    num_items = 4
    dummy_dataset = DummyDataset(num_items, batch_size)

    # Run inference on the dummy dataset
    mask_dict = run_inference(dummy_dataset, batch_size, use_flip=False, num_workers=1)
    
    # Verify that we have the expected keys
    expected_keys = [f"image_{i}" for i in range(num_items)]
    assert sorted(mask_dict.keys()) == expected_keys
    # Also check that each mask has the same shape as a dummy image (assumed from send_inference_request)
    for mask in mask_dict.values():
        assert isinstance(mask, np.ndarray)
        

def test_save_results(tmp_path):
    # Create a dummy mask dictionary
    mask_dict = {
        "image_0": np.array([[0.5, 0.5]], dtype=np.float32),
        "image_1": np.array([[0.7, 0.7]], dtype=np.float32)
    }
    # Define a temporary file path for saving results
    file_path = tmp_path / "results.pkl"
    
    # Call save_results to pickle the dictionary
    save_results(mask_dict, file_path)
    
    # Load the file and verify that contents match
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    for key in mask_dict:
        np.testing.assert_allclose(mask_dict[key], loaded[key])


def test_preprocess_image(monkeypatch):
    # Monkey-patch gcsfs.GCSFileSystem in the dataset module to use DummyGCSFS
    monkeypatch.setattr("src.dataset.gcsfs.GCSFileSystem", lambda: DummyGCSFS())
    
    # Initialize the dataset with a dummy GCS path and no transformation
    dummy_data_folder = "gs://dummy_data"
    dataset = PneumothoraxRayDatasetOnGCP(data_folder=dummy_data_folder, transform=None)
    
    # Get the first image path from the image list
    image_path = dataset.image_list[0]
    processed_image = dataset.preprocess_image(image_path)
    
    # Check that the processed image is a numpy array in CHW format:
    # Expecting 3 channels and non-zero dimensions for height and width.
    assert isinstance(processed_image, np.ndarray)
    assert processed_image.ndim == 3
    assert processed_image.shape[0] == 3  # channels