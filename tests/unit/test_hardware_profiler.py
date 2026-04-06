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
