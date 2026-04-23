"""
Hardware Profiler
=================

Detects hardware capabilities of the current node including CPU, GPU,
memory, storage, and thermal classification. Used to assign compute tiers
for WASM job scheduling.
"""

import logging
import os
import platform
import shutil
import subprocess
from typing import Optional, Tuple

from prsm.compute.wasm.profiler_models import (
    HardwareProfile,
    ThermalClass,
)

logger = logging.getLogger(__name__)

# Known GPU TFLOPS (FP32) lookup table
GPU_TFLOPS_TABLE = {
    "RTX 4090": 82.6, "RTX 4080": 48.7, "RTX 4070 Ti": 40.1, "RTX 4070": 29.1,
    "RTX 4060 Ti": 22.1, "RTX 4060": 15.1,
    "RTX 3090": 35.6, "RTX 3080": 29.8, "RTX 3070": 20.3, "RTX 3060": 12.7,
    "Apple M1": 2.6, "Apple M1 Pro": 5.3, "Apple M1 Max": 10.6,
    "Apple M2": 3.6, "Apple M2 Pro": 6.8, "Apple M2 Max": 13.6,
    "Apple M3": 4.1, "Apple M3 Pro": 8.4, "Apple M3 Max": 16.8,
    "Apple M4": 4.6, "Apple M4 Pro": 9.2, "Apple M4 Max": 18.4,
    "PS5 RDNA 2": 10.3, "Xbox Series X RDNA 2": 12.1,
}


class HardwareProfiler:
    """Detects hardware capabilities and builds a HardwareProfile."""

    def detect(self) -> HardwareProfile:
        """Run full hardware detection and return a HardwareProfile."""
        cpu_cores, cpu_freq_mhz = self._detect_cpu()
        ram_total_gb, ram_available_gb = self._detect_memory()
        gpu_name, gpu_vram_gb, tflops_fp32, tflops_fp16, gpu_api = self._detect_gpu()
        storage_available_gb = self._detect_storage()
        thermal_class = self._detect_thermal_class()
        tee_available, tee_type = self._detect_tee()

        # If no GPU was detected, estimate TFLOPS from CPU
        if tflops_fp32 == 0.0:
            tflops_fp32 = self._estimate_cpu_tflops(cpu_cores, cpu_freq_mhz)

        return HardwareProfile(
            cpu_cores=cpu_cores,
            cpu_freq_mhz=cpu_freq_mhz,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            tflops_fp32=tflops_fp32,
            tflops_fp16=tflops_fp16,
            gpu_api=gpu_api,
            storage_available_gb=storage_available_gb,
            thermal_class=thermal_class,
            tee_available=tee_available,
            tee_type=tee_type,
        )

    def _detect_cpu(self) -> Tuple[int, float]:
        """Detect CPU cores and frequency via psutil."""
        try:
            import psutil
            cores = psutil.cpu_count(logical=True) or 1
            freq = psutil.cpu_freq()
            freq_mhz = freq.current if freq else 0.0
            return cores, freq_mhz
        except ImportError:
            logger.warning("psutil not available, using os.cpu_count()")
            import os
            return os.cpu_count() or 1, 0.0

    def _detect_memory(self) -> Tuple[float, float]:
        """Detect total and available RAM via psutil."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 ** 3)
            available_gb = vm.available / (1024 ** 3)
            return total_gb, available_gb
        except ImportError:
            logger.warning("psutil not available, cannot detect memory")
            return 0.0, 0.0

    def _detect_gpu(self) -> Tuple[str, float, float, float, str]:
        """Detect GPU: try NVIDIA first, then Apple Silicon.

        Returns:
            (gpu_name, gpu_vram_gb, tflops_fp32, tflops_fp16, gpu_api)
        """
        # Try NVIDIA first
        result = self._detect_nvidia()
        if result is not None:
            return result

        # Try Apple Silicon
        result = self._detect_apple_silicon()
        if result is not None:
            return result

        return ("", 0.0, 0.0, 0.0, "")

    def _detect_nvidia(self) -> Optional[Tuple[str, float, float, float, str]]:
        """Parse nvidia-smi output for GPU name, VRAM, and TFLOPS."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,clocks.max.sm",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            stdout = result.stdout
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            stdout = stdout.strip()
            if not stdout:
                return None

            line = stdout.split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                return None

            gpu_name = parts[0]
            vram_mb = float(parts[1])
            vram_gb = vram_mb / 1024.0

            # Look up TFLOPS from table
            tflops_fp32 = 0.0
            for key, val in GPU_TFLOPS_TABLE.items():
                if key in gpu_name:
                    tflops_fp32 = val
                    break

            # If we have clock speed, estimate TFLOPS if not in table
            if tflops_fp32 == 0.0 and len(parts) >= 3:
                try:
                    clock_mhz = float(parts[2])
                    # Very rough: assume ~5000 CUDA cores for unknown GPU
                    tflops_fp32 = (5000 * clock_mhz * 2) / 1e6
                except (ValueError, IndexError):
                    pass

            tflops_fp16 = tflops_fp32 * 2.0  # FP16 is roughly 2x FP32

            return (gpu_name, vram_gb, tflops_fp32, tflops_fp16, "cuda")

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _detect_apple_silicon(self) -> Optional[Tuple[str, float, float, float, str]]:
        """Detect Apple Silicon GPU via sysctl and Metal framework."""
        if platform.system() != "Darwin":
            return None

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            stdout = result.stdout
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            brand = stdout.strip()
            if "Apple" not in brand:
                return None

            # Extract chip name (e.g., "Apple M4 Pro")
            chip_name = brand  # e.g., "Apple M4 Pro"

            # Look up TFLOPS from table
            tflops_fp32 = 0.0
            for key, val in GPU_TFLOPS_TABLE.items():
                if key in chip_name:
                    tflops_fp32 = val
                    break

            # Estimate VRAM as unified memory (use total RAM as proxy)
            try:
                import psutil
                vram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                vram_gb = 0.0

            tflops_fp16 = tflops_fp32 * 2.0

            return (chip_name, vram_gb, tflops_fp32, tflops_fp16, "metal")

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _detect_storage(self) -> float:
        """Detect available disk space via shutil.disk_usage."""
        try:
            usage = shutil.disk_usage("/")
            return usage.free / (1024 ** 3)
        except OSError:
            return 0.0

    def _detect_thermal_class(self) -> ThermalClass:
        """Classify thermal headroom based on battery/power state."""
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery is not None:
                # Has a battery = laptop form factor
                if not battery.power_plugged:
                    return ThermalClass.BURST
                # Plugged in laptop can sustain
                return ThermalClass.SUSTAINED
        except (ImportError, AttributeError):
            pass

        # Desktop or unknown - assume sustained
        return ThermalClass.SUSTAINED

    def _detect_tee(self) -> Tuple[bool, str]:
        """Detect trusted execution environment support."""
        # Check Linux SGX driver
        if os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave"):
            return True, "sgx"
        # Check Apple Secure Enclave
        if platform.system() == "Darwin":
            if os.path.exists("/System/Library/PrivateFrameworks/LocalAuthentication.framework"):
                return True, "secure_enclave"
        return False, ""

    def _estimate_cpu_tflops(self, cores: int, freq_mhz: float) -> float:
        """Rough CPU TFLOPS estimate when no GPU is available.

        Assumes AVX2 (8 FP32 ops per cycle, fused multiply-add = x2).
        """
        if freq_mhz <= 0:
            freq_mhz = 2000.0  # Conservative default

        freq_ghz = freq_mhz / 1000.0
        # 8 FP32 ops/cycle (AVX2 256-bit) * 2 (FMA) * cores * freq
        ops_per_second = 8 * 2 * cores * freq_ghz * 1e9
        tflops = ops_per_second / 1e12
        return tflops
