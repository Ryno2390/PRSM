"""
PRSM Distillation Backends
Support for multiple ML frameworks and distillation methodologies

This module provides concrete implementations for different ML frameworks:
- PyTorch (primary)
- TensorFlow/Keras
- Hugging Face Transformers
- ONNX (for cross-platform deployment)
- JAX/Flax (for research and high-performance computing)

Teacher Model Loading:
- teacher_loader: Unified interface for loading teacher models from various sources
"""

from .pytorch_backend import PyTorchDistillationBackend
from .transformers_backend import TransformersDistillationBackend
from .tensorflow_backend import TensorFlowDistillationBackend
from .teacher_loader import (
    TeacherSource,
    TeacherModelWrapper,
    classify_teacher_source,
    load_teacher_model,
)

__all__ = [
    'PyTorchDistillationBackend',
    'TransformersDistillationBackend',
    'TensorFlowDistillationBackend',
    # Teacher loading
    'TeacherSource',
    'TeacherModelWrapper',
    'classify_teacher_source',
    'load_teacher_model',
]