"""
PRSM Distillation Backends
Support for multiple ML frameworks and distillation methodologies

This module provides concrete implementations for different ML frameworks:
- PyTorch (primary)
- TensorFlow/Keras
- Hugging Face Transformers
- ONNX (for cross-platform deployment)
- JAX/Flax (for research and high-performance computing)
"""

from .pytorch_backend import PyTorchDistillationBackend
from .transformers_backend import TransformersDistillationBackend
from .tensorflow_backend import TensorFlowDistillationBackend

__all__ = [
    'PyTorchDistillationBackend',
    'TransformersDistillationBackend', 
    'TensorFlowDistillationBackend'
]