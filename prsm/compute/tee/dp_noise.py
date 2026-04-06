"""
Differential Privacy Noise Injector.

Calibrated Gaussian noise injection for activation privacy
in TEE-backed confidential compute pipelines.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

from prsm.compute.tee.models import DPConfig, PrivacyLevel


class DPNoiseInjector:
    """Inject calibrated Gaussian noise into tensors for differential privacy."""

    def __init__(self, config: Optional[DPConfig] = None) -> None:
        self._config = config or DPConfig()
        self._epsilon_spent: float = 0.0

    @classmethod
    def from_privacy_level(cls, level: PrivacyLevel) -> DPNoiseInjector:
        """Create an injector from a privacy level preset."""
        return cls(config=PrivacyLevel.config_for_level(level))

    @property
    def epsilon_spent(self) -> float:
        """Cumulative privacy budget consumed so far."""
        return self._epsilon_spent

    @property
    def config(self) -> DPConfig:
        return self._config

    def _clip(self, tensor: np.ndarray) -> np.ndarray:
        """Clip tensor so its L2 norm does not exceed config.clip_norm."""
        norm = np.linalg.norm(tensor)
        if norm > self._config.clip_norm:
            tensor = tensor * (self._config.clip_norm / norm)
        return tensor

    def inject(self, tensor: np.ndarray) -> np.ndarray:
        """Clip and add calibrated Gaussian noise to a tensor.

        If epsilon is inf (no privacy), returns a copy without noise.
        """
        clipped = self._clip(tensor.copy())

        if math.isinf(self._config.epsilon):
            return clipped

        sigma = self._config.noise_scale
        noise = np.random.normal(loc=0.0, scale=sigma, size=clipped.shape)
        self._epsilon_spent += self._config.epsilon
        return clipped + noise

    def inject_bytes(self, data: bytes) -> bytes:
        """Parse JSON bytes, apply DP noise to numeric arrays, re-encode.

        Expects a JSON object. Any top-level key whose value is a list of
        numbers will have noise applied.
        """
        obj = json.loads(data)

        for key, value in obj.items():
            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                arr = np.array(value, dtype=np.float64)
                noised = self.inject(arr)
                obj[key] = noised.tolist()

        return json.dumps(obj).encode("utf-8")
