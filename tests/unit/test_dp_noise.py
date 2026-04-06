"""Tests for DPNoiseInjector (Ring 7)."""

import json
import math

import numpy as np
import pytest

from prsm.compute.tee.dp_noise import DPNoiseInjector
from prsm.compute.tee.models import DPConfig, PrivacyLevel


class TestDPNoiseInjector:
    def test_noise_changes_data(self):
        injector = DPNoiseInjector(DPConfig(epsilon=1.0))
        tensor = np.array([1.0, 2.0, 3.0])
        result = injector.inject(tensor)
        assert not np.array_equal(result, tensor)

    def test_preserves_shape(self):
        injector = DPNoiseInjector(DPConfig(epsilon=4.0))
        tensor = np.random.randn(5, 3)
        result = injector.inject(tensor)
        assert result.shape == tensor.shape

    def test_lower_epsilon_more_noise(self):
        np.random.seed(42)
        tensor = np.ones(1000)

        high_eps = DPNoiseInjector(DPConfig(epsilon=100.0, clip_norm=100.0))
        low_eps = DPNoiseInjector(DPConfig(epsilon=0.1, clip_norm=100.0))

        high_result = high_eps.inject(tensor)
        np.random.seed(43)
        low_result = low_eps.inject(tensor)

        high_diff = np.mean(np.abs(high_result - tensor))
        low_diff = np.mean(np.abs(low_result - tensor))
        assert low_diff > high_diff

    def test_clipping_enforced(self):
        cfg = DPConfig(epsilon=float("inf"), clip_norm=1.0)
        injector = DPNoiseInjector(cfg)
        tensor = np.array([10.0, 10.0, 10.0])
        result = injector.inject(tensor)
        assert np.linalg.norm(result) <= 1.0 + 1e-7

    def test_bytes_roundtrip(self):
        injector = DPNoiseInjector(DPConfig(epsilon=1.0))
        payload = json.dumps({"activations": [1.0, 2.0, 3.0], "label": "test"}).encode()
        result = injector.inject_bytes(payload)
        obj = json.loads(result)
        assert "activations" in obj
        assert len(obj["activations"]) == 3
        assert obj["label"] == "test"  # non-numeric unchanged

    def test_no_noise_when_epsilon_inf(self):
        cfg = DPConfig(epsilon=float("inf"), clip_norm=100.0)
        injector = DPNoiseInjector(cfg)
        tensor = np.array([1.0, 2.0, 3.0])
        result = injector.inject(tensor)
        np.testing.assert_array_almost_equal(result, tensor)

    def test_privacy_budget_tracking(self):
        cfg = DPConfig(epsilon=2.0)
        injector = DPNoiseInjector(cfg)
        assert injector.epsilon_spent == 0.0

        injector.inject(np.array([1.0]))
        assert injector.epsilon_spent == 2.0

        injector.inject(np.array([1.0]))
        assert injector.epsilon_spent == 4.0

    def test_from_privacy_level(self):
        injector = DPNoiseInjector.from_privacy_level(PrivacyLevel.HIGH)
        assert injector.config.epsilon == 4.0

        injector_max = DPNoiseInjector.from_privacy_level(PrivacyLevel.MAXIMUM)
        assert injector_max.config.epsilon == 1.0
