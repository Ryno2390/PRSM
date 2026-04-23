"""
Hardware Profile Data Models
============================

Dataclasses for hardware capability reporting and compute tier classification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class ComputeTier(str, Enum):
    """Hardware compute tiers based on TFLOPS."""
    T1 = "t1"  # < 5 TFLOPS: Mobile, IoT, old laptops
    T2 = "t2"  # 5-30 TFLOPS: Consoles, mid-range GPUs
    T3 = "t3"  # 30-80 TFLOPS: High-end desktops, M-series
    T4 = "t4"  # 80+ TFLOPS: Datacenter GPUs

    @classmethod
    def from_tflops(cls, tflops: float) -> "ComputeTier":
        if tflops >= 80.0:
            return cls.T4
        elif tflops >= 30.0:
            return cls.T3
        elif tflops >= 5.0:
            return cls.T2
        else:
            return cls.T1


class ThermalClass(str, Enum):
    """Thermal headroom classification."""
    SUSTAINED = "sustained"
    BURST = "burst"
    THROTTLED = "throttled"


@dataclass
class HardwareProfile:
    """Complete hardware capability profile for a PRSM node."""
    cpu_cores: int = 1
    cpu_freq_mhz: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    tflops_fp32: float = 0.0
    tflops_fp16: float = 0.0
    gpu_api: str = ""  # "cuda", "metal", "rocm", ""
    storage_available_gb: float = 0.0
    upload_mbps: float = 0.0
    download_mbps: float = 0.0
    thermal_class: ThermalClass = ThermalClass.SUSTAINED

    # TEE (Ring 7)
    tee_available: bool = False
    tee_type: str = ""  # "sgx", "tdx", "sev", "trustzone", "secure_enclave", ""

    @property
    def compute_tier(self) -> ComputeTier:
        return ComputeTier.from_tflops(self.tflops_fp32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "tflops_fp32": self.tflops_fp32,
            "tflops_fp16": self.tflops_fp16,
            "gpu_api": self.gpu_api,
            "storage_available_gb": self.storage_available_gb,
            "upload_mbps": self.upload_mbps,
            "download_mbps": self.download_mbps,
            "thermal_class": self.thermal_class.value,
            "compute_tier": self.compute_tier.value,
            "tee_available": self.tee_available,
            "tee_type": self.tee_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareProfile":
        thermal = d.get("thermal_class", "sustained")
        if isinstance(thermal, str):
            thermal = ThermalClass(thermal)
        return cls(
            cpu_cores=d.get("cpu_cores", 1),
            cpu_freq_mhz=d.get("cpu_freq_mhz", 0.0),
            ram_total_gb=d.get("ram_total_gb", 0.0),
            ram_available_gb=d.get("ram_available_gb", 0.0),
            gpu_name=d.get("gpu_name", ""),
            gpu_vram_gb=d.get("gpu_vram_gb", 0.0),
            tflops_fp32=d.get("tflops_fp32", 0.0),
            tflops_fp16=d.get("tflops_fp16", 0.0),
            gpu_api=d.get("gpu_api", ""),
            storage_available_gb=d.get("storage_available_gb", 0.0),
            upload_mbps=d.get("upload_mbps", 0.0),
            download_mbps=d.get("download_mbps", 0.0),
            thermal_class=thermal,
            tee_available=d.get("tee_available", False),
            tee_type=d.get("tee_type", ""),
        )
