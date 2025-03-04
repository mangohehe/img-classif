"""
src package for the Pneumothorax inference pipeline.

Modules:
    - inference_client: Functions for sending inference requests and processing responses.
    - serve_deployment: Ray Serve deployment logic for the Pneumothorax model.
    - dataset: Contains the PneumothoraxRayDatasetOnGCP class for loading and preprocessing data.
    - main: Entry point for running the complete pipeline.
"""

from .inference_client import send_inference_request, run_inference, save_results
from .serve_deployment import start_serve, PneumothoraxModel
from .dataset import PneumothoraxRayDatasetOnGCP

__all__ = [
    "send_inference_request",
    "run_inference",
    "save_results",
    "start_serve",
    "PneumothoraxModel",
    "PneumothoraxRayDatasetOnGCP",
]