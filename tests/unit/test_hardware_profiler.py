"""Tests for hardware profiler and compute tier classification."""

import pytest
from unittest.mock import patch, MagicMock

from prsm.compute.wasm.profiler_models import (
    HardwareProfile,
    ComputeTier,
    ThermalClass,
)


class TestComputeTier:
    def test_tier_from_tflops_t1(self):
        assert ComputeTier.from_tflops(3.0) == ComputeTier.T1

    def test_tier_from_tflops_t2(self):
        assert ComputeTier.from_tflops(15.0) == ComputeTier.T2

    def test_tier_from_tflops_t3(self):
        assert ComputeTier.from_tflops(50.0) == ComputeTier.T3

    def test_tier_from_tflops_t4(self):
        assert ComputeTier.from_tflops(100.0) == ComputeTier.T4

    def test_tier_boundary_t1_t2(self):
        assert ComputeTier.from_tflops(5.0) == ComputeTier.T2

    def test_tier_boundary_t2_t3(self):
        assert ComputeTier.from_tflops(30.0) == ComputeTier.T3

    def test_tier_boundary_t3_t4(self):
        assert ComputeTier.from_tflops(80.0) == ComputeTier.T4

    def test_tier_zero_tflops(self):
        assert ComputeTier.from_tflops(0.0) == ComputeTier.T1


class TestHardwareProfile:
    def test_profile_creation(self):
        profile = HardwareProfile(
            cpu_cores=8,
            cpu_freq_mhz=3200.0,
            ram_total_gb=16.0,
            ram_available_gb=10.0,
            gpu_name="NVIDIA RTX 4070",
            gpu_vram_gb=12.0,
            tflops_fp32=29.1,
            tflops_fp16=58.2,
            gpu_api="cuda",
            storage_available_gb=500.0,
            upload_mbps=50.0,
            download_mbps=200.0,
            thermal_class=ThermalClass.SUSTAINED,
        )
        assert profile.compute_tier == ComputeTier.T2
        assert profile.tflops_fp32 == 29.1

    def test_profile_tier_derived_from_tflops(self):
        profile = HardwareProfile(
            cpu_cores=4,
            cpu_freq_mhz=2400.0,
            ram_total_gb=8.0,
            ram_available_gb=4.0,
            tflops_fp32=2.0,
            thermal_class=ThermalClass.BURST,
        )
        assert profile.compute_tier == ComputeTier.T1

    def test_profile_to_dict(self):
        profile = HardwareProfile(
            cpu_cores=8,
            cpu_freq_mhz=3200.0,
            ram_total_gb=16.0,
            ram_available_gb=10.0,
            tflops_fp32=29.1,
            thermal_class=ThermalClass.SUSTAINED,
        )
        d = profile.to_dict()
        assert d["cpu_cores"] == 8
        assert d["compute_tier"] == "t2"
        assert d["thermal_class"] == "sustained"
        assert "tflops_fp32" in d

    def test_profile_from_dict(self):
        d = {
            "cpu_cores": 8,
            "cpu_freq_mhz": 3200.0,
            "ram_total_gb": 16.0,
            "ram_available_gb": 10.0,
            "tflops_fp32": 29.1,
            "tflops_fp16": 0.0,
            "gpu_name": "",
            "gpu_vram_gb": 0.0,
            "gpu_api": "",
            "storage_available_gb": 0.0,
            "upload_mbps": 0.0,
            "download_mbps": 0.0,
            "thermal_class": "sustained",
        }
        profile = HardwareProfile.from_dict(d)
        assert profile.cpu_cores == 8
        assert profile.compute_tier == ComputeTier.T2


from prsm.compute.wasm.profiler import HardwareProfiler


class TestHardwareProfiler:
    def test_detect_cpu(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.cpu_cores >= 1
        assert profile.ram_total_gb > 0

    @patch("prsm.compute.wasm.profiler.subprocess.run")
    def test_detect_nvidia_gpu(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA RTX 4070, 12288, 29150\n",
        )
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.gpu_name == "NVIDIA RTX 4070"
        assert profile.gpu_vram_gb == pytest.approx(12.0, abs=0.1)
        assert profile.tflops_fp32 == pytest.approx(29.1, abs=0.1)
        assert profile.gpu_api == "cuda"

    def test_detect_returns_valid_tier(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.compute_tier in [
            ComputeTier.T1, ComputeTier.T2, ComputeTier.T3, ComputeTier.T4,
        ]

    def test_detect_thermal_class(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.thermal_class in [
            ThermalClass.SUSTAINED, ThermalClass.BURST, ThermalClass.THROTTLED,
        ]

    def test_tflops_estimate_cpu_fallback(self):
        profiler = HardwareProfiler()
        tflops = profiler._estimate_cpu_tflops(cores=8, freq_mhz=3200.0)
        assert tflops > 0
        assert tflops < 5.0

    def test_profile_serialization_roundtrip(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        d = profile.to_dict()
        restored = HardwareProfile.from_dict(d)
        assert restored.cpu_cores == profile.cpu_cores
        assert restored.compute_tier == profile.compute_tier


class TestGossipHardwareProfile:
    def test_gossip_constant_exists(self):
        from prsm.node.gossip import GOSSIP_HARDWARE_PROFILE
        assert GOSSIP_HARDWARE_PROFILE == "hardware_profile"

    def test_gossip_retention_is_24h(self):
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS
        assert GOSSIP_RETENTION_SECONDS.get("hardware_profile") == 86400
