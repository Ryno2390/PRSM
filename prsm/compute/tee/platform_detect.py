"""
TEE Platform Detection
======================

Detailed detection of TEE capabilities per platform.
Detects Intel SGX/TDX, AMD SEV, Apple Secure Enclave, ARM TrustZone.
"""

import logging
import os
import platform
import subprocess
from typing import Dict, Optional

from prsm.compute.tee.models import TEEType, TEECapability

logger = logging.getLogger(__name__)


def detect_tee_capability() -> TEECapability:
    """Detect the best available TEE on this platform."""
    detectors = [
        _detect_intel_sgx,
        _detect_intel_tdx,
        _detect_amd_sev,
        _detect_apple_secure_enclave,
        _detect_arm_trustzone,
    ]

    for detector in detectors:
        try:
            cap = detector()
            if cap and cap.tee_type != TEEType.NONE:
                logger.info(f"TEE detected: {cap.tee_type.value} (memory={cap.max_enclave_memory_mb}MB)")
                return cap
        except Exception as e:
            logger.debug(f"TEE detection failed for {detector.__name__}: {e}")

    # Fallback: software-only TEE (WASM sandbox)
    return TEECapability(tee_type=TEEType.SOFTWARE, max_enclave_memory_mb=256)


def _detect_intel_sgx() -> Optional[TEECapability]:
    """Detect Intel SGX support."""
    # Check for SGX device driver (Linux)
    if os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave"):
        return TEECapability(
            tee_type=TEEType.SGX,
            max_enclave_memory_mb=128,  # Default EPC size
            max_threads=8,
            attestation_supported=True,
        )
    # Check via cpuid (if available)
    try:
        result = subprocess.run(["grep", "-c", "sgx", "/proc/cpuinfo"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return TEECapability(tee_type=TEEType.SGX, max_enclave_memory_mb=128, attestation_supported=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _detect_intel_tdx() -> Optional[TEECapability]:
    """Detect Intel TDX (Trust Domain Extensions)."""
    if os.path.exists("/dev/tdx-guest") or os.path.exists("/sys/firmware/tdx"):
        return TEECapability(
            tee_type=TEEType.TDX,
            max_enclave_memory_mb=4096,
            max_threads=32,
            attestation_supported=True,
        )
    return None


def _detect_amd_sev() -> Optional[TEECapability]:
    """Detect AMD SEV (Secure Encrypted Virtualization)."""
    if os.path.exists("/dev/sev") or os.path.exists("/dev/sev-guest"):
        return TEECapability(
            tee_type=TEEType.SEV,
            max_enclave_memory_mb=4096,
            max_threads=32,
            attestation_supported=True,
        )
    return None


def _detect_apple_secure_enclave() -> Optional[TEECapability]:
    """Detect Apple Secure Enclave (T2 chip or Apple Silicon)."""
    if platform.system() != "Darwin":
        return None

    # Check for Secure Enclave via LocalAuthentication framework
    if os.path.exists("/System/Library/PrivateFrameworks/LocalAuthentication.framework"):
        # Determine memory based on chip
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5)
            brand = result.stdout.strip()
            # Apple Silicon has larger secure memory
            memory_mb = 512 if "Apple" in brand else 256
        except Exception:
            memory_mb = 256

        return TEECapability(
            tee_type=TEEType.SECURE_ENCLAVE,
            max_enclave_memory_mb=memory_mb,
            max_threads=4,
            attestation_supported=True,
        )
    return None


def _detect_arm_trustzone() -> Optional[TEECapability]:
    """Detect ARM TrustZone."""
    if platform.machine().startswith("aarch64") or platform.machine().startswith("arm"):
        if os.path.exists("/dev/tee0") or os.path.exists("/dev/optee-tz"):
            return TEECapability(
                tee_type=TEEType.TRUSTZONE,
                max_enclave_memory_mb=64,
                max_threads=2,
                attestation_supported=False,
            )
    return None


def get_tee_summary() -> Dict[str, str]:
    """Get a human-readable TEE summary for CLI/dashboard."""
    cap = detect_tee_capability()
    return {
        "type": cap.tee_type.value,
        "hardware_backed": str(cap.is_hardware_backed),
        "memory_mb": str(cap.max_enclave_memory_mb),
        "attestation": str(cap.attestation_supported),
    }
