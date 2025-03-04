"""
Models package for the image classification service.

This package includes various model architectures:
- ternausnets: Implementation of TernausNet models.
- selim_zoo: A collection of additional architectures, including:
    - densenet (DenseNet)
    - dpn (DPN)
    - resnet (ResNet)
    - senet (SENet)
    - unet (UNet)

Usage:
    from models import ternausnets, densenet, dpn, resnet, senet, unet
"""

# Import ternausnets module
from . import ternausnets

# Import submodules from selim_zoo
from .selim_zoo import densenet, dpn, resnet, senet, unet

__all__ = [
    "ternausnets",
    "densenet",
    "dpn",
    "resnet",
    "senet",
    "unet",
]