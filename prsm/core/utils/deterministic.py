"""
PRSM Deterministic Verification Utility
=======================================

Provides tools for ensuring identical execution results across different 
network nodes. This is critical for achieving blockchain consensus on 
distributed AI tasks without requiring full re-computation.
"""

import hashlib
import json
import random
import os
from typing import Any, Dict, Optional
from decimal import Decimal
import structlog

logger = structlog.get_logger(__name__)

class DeterministicRNG:
    """
    A globally consistent and seedable random number generator.
    Ensures that stochastic operations produce identical results across 
    different platforms and hardware.
    """
    def __init__(self, seed: int):
        self.seed = seed
        self.state = seed
        # Standard LCG parameters (similar to glibc)
        self.m = 2**31
        self.a = 1103515245
        self.c = 12345

    def next_float(self) -> float:
        """Generate a deterministic float between 0 and 1"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def next_int(self, min_val: int, max_val: int) -> int:
        """Generate a deterministic integer within a range"""
        f = self.next_float()
        return min_val + int(f * (max_val - min_val + 1))

def get_local_generator(seed: int) -> DeterministicRNG:
    """Returns a local deterministic RNG instance"""
    return DeterministicRNG(seed)

def force_determinism(seed: int = 42):
    """
    Force deterministic behavior across common AI libraries.
    """
    logger.info(f"Forcing global determinism with seed: {seed}")
    
    # 1. Standard Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. NumPy (if available)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
        
    # 3. PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms (note: might be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

def generate_verification_hash(output_data: Any, model_id: str, input_hash: str) -> str:
    """
    Generate a cryptographic signature for a model's output.
    Used by the blockchain layer to verify consensus.
    
    Includes quantization to handle floating-point non-determinism.
    """
    import torch
    import numpy as np

    def quantize(data):
        """Recursively round floats to ensure deterministic hashing"""
        if isinstance(data, (float, Decimal)):
            return round(float(data), 4)
        if isinstance(data, dict):
            return {k: quantize(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [quantize(v) for v in data]
        if hasattr(data, 'detach'): # PyTorch Tensor
            return quantize(data.detach().cpu().numpy().tolist())
        if isinstance(data, np.ndarray):
            return quantize(data.tolist())
        return data

    stable_output = quantize(output_data)
        
    verification_payload = {
        "model_id": model_id,
        "input_hash": input_hash,
        "output": stable_output
    }
    
    payload_str = json.dumps(verification_payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()
