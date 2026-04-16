# Ring 7 — "The Vault" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PRSM node can detect TEE hardware capabilities, execute computations inside a trusted execution environment (with WASM fallback), and apply differential privacy noise to intermediate activations before they leave the node.

**Architecture:** A `TEERuntime` interface mirrors `WASMRuntime` from Ring 1. A `DPNoiseInjector` applies calibrated Gaussian noise to tensor data. The `HardwareProfiler` is extended to detect TEE availability (Intel SGX, Apple Secure Enclave, ARM TrustZone). TEE capabilities are gossiped via `GOSSIP_TEE_CAPABILITY`. A `ConfidentialExecutor` wraps the TEE runtime + DP injector into a single secure execution pipeline.

**Tech Stack:** Existing PRSM infrastructure + `numpy` (already a dependency) for noise generation. No new external dependencies — actual TEE SDK integration (SGX SDK, CryptoKit) is deferred to platform-specific implementations.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/compute/tee/__init__.py` | Package exports |
| Create | `prsm/compute/tee/models.py` | `TEECapability`, `DPConfig`, `ConfidentialResult` |
| Create | `prsm/compute/tee/runtime.py` | `TEERuntime` ABC + `SoftwareTEERuntime` (WASM-backed fallback) |
| Create | `prsm/compute/tee/dp_noise.py` | `DPNoiseInjector` — calibrated Gaussian noise for activations |
| Create | `prsm/compute/tee/confidential_executor.py` | `ConfidentialExecutor` — TEE + DP pipeline |
| Modify | `prsm/compute/wasm/profiler_models.py` | Add TEE fields to `HardwareProfile` |
| Modify | `prsm/compute/wasm/profiler.py` | Add `_detect_tee()` method |
| Modify | `prsm/node/gossip.py` | Add `GOSSIP_TEE_CAPABILITY` |
| Modify | `prsm/node/node.py` | Wire ConfidentialExecutor into PRSMNode |
| Create | `tests/unit/test_tee_models.py` | TEE data model tests |
| Create | `tests/unit/test_dp_noise.py` | DP noise injection tests |
| Create | `tests/unit/test_confidential_executor.py` | Confidential execution pipeline tests |
| Create | `tests/integration/test_ring7_vault.py` | End-to-end smoke test |

---

### Task 1: TEE Data Models + DP Config

**Files:**
- Create: `prsm/compute/tee/__init__.py`
- Create: `prsm/compute/tee/models.py`
- Test: `tests/unit/test_tee_models.py`

- [ ] **Step 1:** `mkdir -p prsm/compute/tee`

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_tee_models.py`:

```python
"""Tests for TEE and confidential compute data models."""

import pytest
from prsm.compute.tee.models import (
    TEEType,
    TEECapability,
    DPConfig,
    ConfidentialResult,
    PrivacyLevel,
)


class TestTEEType:
    def test_all_types_exist(self):
        assert TEEType.NONE == "none"
        assert TEEType.SGX == "sgx"
        assert TEEType.TDX == "tdx"
        assert TEEType.SEV == "sev"
        assert TEEType.TRUSTZONE == "trustzone"
        assert TEEType.SECURE_ENCLAVE == "secure_enclave"
        assert TEEType.SOFTWARE == "software"


class TestTEECapability:
    def test_capability_creation(self):
        cap = TEECapability(
            tee_type=TEEType.SGX,
            max_enclave_memory_mb=256,
            max_threads=8,
            attestation_supported=True,
        )
        assert cap.tee_type == TEEType.SGX
        assert cap.available is True

    def test_no_tee_capability(self):
        cap = TEECapability(tee_type=TEEType.NONE)
        assert cap.available is False

    def test_software_fallback(self):
        cap = TEECapability(tee_type=TEEType.SOFTWARE)
        assert cap.available is True
        assert cap.is_hardware_backed is False

    def test_hardware_backed(self):
        cap = TEECapability(tee_type=TEEType.SGX)
        assert cap.is_hardware_backed is True

    def test_to_dict_roundtrip(self):
        cap = TEECapability(
            tee_type=TEEType.SECURE_ENCLAVE,
            max_enclave_memory_mb=512,
            attestation_supported=True,
        )
        d = cap.to_dict()
        restored = TEECapability.from_dict(d)
        assert restored.tee_type == TEEType.SECURE_ENCLAVE
        assert restored.max_enclave_memory_mb == 512


class TestDPConfig:
    def test_default_config(self):
        config = DPConfig()
        assert config.epsilon == 8.0
        assert config.delta == 1e-5
        assert config.noise_mechanism == "gaussian"
        assert config.clip_norm == 1.0

    def test_strict_privacy(self):
        config = DPConfig(epsilon=1.0)
        assert config.epsilon == 1.0

    def test_sensitivity_calculation(self):
        config = DPConfig(epsilon=8.0, clip_norm=1.0)
        sigma = config.noise_scale
        assert sigma > 0

    def test_lower_epsilon_means_more_noise(self):
        strict = DPConfig(epsilon=1.0, clip_norm=1.0)
        relaxed = DPConfig(epsilon=8.0, clip_norm=1.0)
        assert strict.noise_scale > relaxed.noise_scale


class TestPrivacyLevel:
    def test_levels_exist(self):
        assert PrivacyLevel.NONE == "none"
        assert PrivacyLevel.STANDARD == "standard"
        assert PrivacyLevel.HIGH == "high"
        assert PrivacyLevel.MAXIMUM == "maximum"

    def test_epsilon_from_level(self):
        assert PrivacyLevel.config_for_level(PrivacyLevel.STANDARD).epsilon == 8.0
        assert PrivacyLevel.config_for_level(PrivacyLevel.HIGH).epsilon == 4.0
        assert PrivacyLevel.config_for_level(PrivacyLevel.MAXIMUM).epsilon == 1.0


class TestConfidentialResult:
    def test_result_creation(self):
        result = ConfidentialResult(
            output=b'{"answer": 42}',
            dp_applied=True,
            epsilon_spent=8.0,
            tee_type=TEEType.SGX,
            attestation_proof="proof-bytes-here",
            execution_time_seconds=1.5,
        )
        assert result.dp_applied is True
        assert result.tee_type == TEEType.SGX

    def test_result_without_tee(self):
        result = ConfidentialResult(
            output=b"data",
            dp_applied=True,
            epsilon_spent=8.0,
            tee_type=TEEType.SOFTWARE,
        )
        assert result.is_hardware_attested is False

    def test_result_with_hardware_attestation(self):
        result = ConfidentialResult(
            output=b"data",
            dp_applied=True,
            epsilon_spent=8.0,
            tee_type=TEEType.SGX,
            attestation_proof="sgx-quote",
        )
        assert result.is_hardware_attested is True
```

- [ ] **Step 3: Implement models**

Create `prsm/compute/tee/__init__.py`:

```python
"""
Trusted Execution Environment (TEE) Runtime
============================================

Confidential compute for PRSM: TEE-backed execution with
differential privacy on intermediate activations.
Ring 7 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.tee.models import (
    TEEType,
    TEECapability,
    DPConfig,
    ConfidentialResult,
    PrivacyLevel,
)

__all__ = [
    "TEEType",
    "TEECapability",
    "DPConfig",
    "ConfidentialResult",
    "PrivacyLevel",
]
```

Create `prsm/compute/tee/models.py`:

```python
"""
TEE and Confidential Compute Data Models
=========================================

Types for trusted execution environments, differential privacy
configuration, and confidential execution results.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TEEType(str, Enum):
    """Supported trusted execution environment types."""
    NONE = "none"
    SGX = "sgx"                    # Intel Software Guard Extensions
    TDX = "tdx"                    # Intel Trust Domain Extensions
    SEV = "sev"                    # AMD Secure Encrypted Virtualization
    TRUSTZONE = "trustzone"        # ARM TrustZone
    SECURE_ENCLAVE = "secure_enclave"  # Apple Secure Enclave
    SOFTWARE = "software"          # WASM sandbox fallback (no hardware TEE)

    HARDWARE_TYPES = frozenset({"sgx", "tdx", "sev", "trustzone", "secure_enclave"})


@dataclass
class TEECapability:
    """TEE hardware capability for a node."""
    tee_type: TEEType = TEEType.NONE
    max_enclave_memory_mb: int = 0
    max_threads: int = 0
    attestation_supported: bool = False

    @property
    def available(self) -> bool:
        return self.tee_type != TEEType.NONE

    @property
    def is_hardware_backed(self) -> bool:
        return self.tee_type.value in TEEType.HARDWARE_TYPES.default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tee_type": self.tee_type.value,
            "max_enclave_memory_mb": self.max_enclave_memory_mb,
            "max_threads": self.max_threads,
            "attestation_supported": self.attestation_supported,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TEECapability":
        return cls(
            tee_type=TEEType(d.get("tee_type", "none")),
            max_enclave_memory_mb=d.get("max_enclave_memory_mb", 0),
            max_threads=d.get("max_threads", 0),
            attestation_supported=d.get("attestation_supported", False),
        )


@dataclass
class DPConfig:
    """Differential privacy configuration for activation noise."""
    epsilon: float = 8.0          # Privacy budget (lower = more private)
    delta: float = 1e-5           # Probability of privacy breach
    clip_norm: float = 1.0        # L2 norm clipping bound for activations
    noise_mechanism: str = "gaussian"

    @property
    def noise_scale(self) -> float:
        """Compute Gaussian noise standard deviation (sigma).

        sigma = clip_norm * sqrt(2 * ln(1.25/delta)) / epsilon
        """
        if self.epsilon <= 0:
            return float("inf")
        return self.clip_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon


class PrivacyLevel(str, Enum):
    """Preset privacy levels for easy configuration."""
    NONE = "none"          # No DP noise (open computation)
    STANDARD = "standard"  # epsilon=8.0 (moderate, <1% accuracy loss)
    HIGH = "high"          # epsilon=4.0 (strong, ~2% accuracy loss)
    MAXIMUM = "maximum"    # epsilon=1.0 (very strong, ~5% accuracy loss)

    @staticmethod
    def config_for_level(level: "PrivacyLevel") -> DPConfig:
        configs = {
            PrivacyLevel.NONE: DPConfig(epsilon=float("inf")),
            PrivacyLevel.STANDARD: DPConfig(epsilon=8.0),
            PrivacyLevel.HIGH: DPConfig(epsilon=4.0),
            PrivacyLevel.MAXIMUM: DPConfig(epsilon=1.0),
        }
        return configs.get(level, DPConfig())


@dataclass
class ConfidentialResult:
    """Result from a confidential execution."""
    output: bytes
    dp_applied: bool = False
    epsilon_spent: float = 0.0
    tee_type: TEEType = TEEType.SOFTWARE
    attestation_proof: Optional[str] = None
    execution_time_seconds: float = 0.0
    memory_used_bytes: int = 0

    @property
    def is_hardware_attested(self) -> bool:
        return (
            self.tee_type.value in TEEType.HARDWARE_TYPES.default
            and self.attestation_proof is not None
            and len(self.attestation_proof) > 0
        )
```

NOTE: The `TEEType.HARDWARE_TYPES` needs to be accessible as a class-level set. Since Enum members can't easily have class attributes, use a different approach — define `HARDWARE_TEE_TYPES` as a module-level frozenset and reference it in the properties:

```python
HARDWARE_TEE_TYPES = frozenset({"sgx", "tdx", "sev", "trustzone", "secure_enclave"})
```

Then in `TEECapability.is_hardware_backed`:
```python
return self.tee_type.value in HARDWARE_TEE_TYPES
```

And in `ConfidentialResult.is_hardware_attested`:
```python
return self.tee_type.value in HARDWARE_TEE_TYPES and ...
```

- [ ] **Step 4: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_tee_models.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/tee/__init__.py prsm/compute/tee/models.py tests/unit/test_tee_models.py
git commit -m "feat(ring7): TEEType, TEECapability, DPConfig, PrivacyLevel, ConfidentialResult models"
```

---

### Task 2: Differential Privacy Noise Injector

**Files:**
- Create: `prsm/compute/tee/dp_noise.py`
- Test: `tests/unit/test_dp_noise.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_dp_noise.py`:

```python
"""Tests for differential privacy noise injection."""

import pytest
import numpy as np

from prsm.compute.tee.dp_noise import DPNoiseInjector
from prsm.compute.tee.models import DPConfig, PrivacyLevel


class TestDPNoiseInjector:
    def test_inject_noise_changes_data(self):
        config = DPConfig(epsilon=8.0, clip_norm=1.0)
        injector = DPNoiseInjector(config)

        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy = injector.inject(original)

        assert noisy.shape == original.shape
        assert not np.array_equal(noisy, original)

    def test_inject_preserves_shape(self):
        injector = DPNoiseInjector(DPConfig())
        tensor = np.random.randn(10, 20, 30)
        noisy = injector.inject(tensor)
        assert noisy.shape == (10, 20, 30)

    def test_lower_epsilon_means_more_noise(self):
        strict = DPNoiseInjector(DPConfig(epsilon=1.0))
        relaxed = DPNoiseInjector(DPConfig(epsilon=8.0))

        original = np.ones(10000)

        strict_noisy = strict.inject(original)
        relaxed_noisy = relaxed.inject(original)

        # More noise = larger deviation from original
        strict_deviation = np.std(strict_noisy - original)
        relaxed_deviation = np.std(relaxed_noisy - original)

        assert strict_deviation > relaxed_deviation

    def test_clipping_enforced(self):
        config = DPConfig(epsilon=8.0, clip_norm=0.5)
        injector = DPNoiseInjector(config)

        # Large tensor that exceeds clip norm
        big_tensor = np.array([100.0, 200.0, 300.0])
        clipped = injector._clip(big_tensor)

        # L2 norm should be <= clip_norm
        assert np.linalg.norm(clipped) <= config.clip_norm + 1e-6

    def test_inject_bytes_roundtrip(self):
        injector = DPNoiseInjector(DPConfig(epsilon=8.0))

        original_bytes = b'{"tensor": [1.0, 2.0, 3.0]}'
        noisy_bytes = injector.inject_bytes(original_bytes)

        assert isinstance(noisy_bytes, bytes)
        assert len(noisy_bytes) > 0

    def test_no_noise_when_epsilon_inf(self):
        config = DPConfig(epsilon=float("inf"))
        injector = DPNoiseInjector(config)

        original = np.array([1.0, 2.0, 3.0])
        noisy = injector.inject(original)

        np.testing.assert_array_almost_equal(noisy, original)

    def test_privacy_budget_tracking(self):
        injector = DPNoiseInjector(DPConfig(epsilon=8.0))
        assert injector.epsilon_spent == 0.0

        injector.inject(np.ones(100))
        assert injector.epsilon_spent == 8.0

        injector.inject(np.ones(100))
        assert injector.epsilon_spent == 16.0

    def test_from_privacy_level(self):
        injector = DPNoiseInjector.from_privacy_level(PrivacyLevel.HIGH)
        assert injector.config.epsilon == 4.0
```

- [ ] **Step 2: Implement DPNoiseInjector**

Create `prsm/compute/tee/dp_noise.py`:

```python
"""
Differential Privacy Noise Injector
====================================

Applies calibrated Gaussian noise to tensor activations before they
leave a node, ensuring differential privacy guarantees on intermediate
computations.
"""

import json
import logging
from typing import Optional

import numpy as np

from prsm.compute.tee.models import DPConfig, PrivacyLevel

logger = logging.getLogger(__name__)


class DPNoiseInjector:
    """Injects calibrated Gaussian noise for differential privacy."""

    def __init__(self, config: Optional[DPConfig] = None):
        self.config = config or DPConfig()
        self._epsilon_spent = 0.0

    @classmethod
    def from_privacy_level(cls, level: PrivacyLevel) -> "DPNoiseInjector":
        return cls(config=PrivacyLevel.config_for_level(level))

    @property
    def epsilon_spent(self) -> float:
        return self._epsilon_spent

    def _clip(self, tensor: np.ndarray) -> np.ndarray:
        """Clip tensor to L2 norm bound."""
        norm = np.linalg.norm(tensor)
        if norm > self.config.clip_norm:
            return tensor * (self.config.clip_norm / norm)
        return tensor

    def inject(self, tensor: np.ndarray) -> np.ndarray:
        """Apply DP noise to a tensor.

        1. Clip to L2 norm bound
        2. Add calibrated Gaussian noise
        3. Track privacy budget spent
        """
        if self.config.epsilon == float("inf"):
            return tensor.copy()

        sigma = self.config.noise_scale

        clipped = self._clip(tensor)
        noise = np.random.normal(0, sigma, size=clipped.shape)
        noisy = clipped + noise

        self._epsilon_spent += self.config.epsilon

        return noisy

    def inject_bytes(self, data: bytes) -> bytes:
        """Apply DP noise to JSON-encoded tensor data.

        Parses the bytes as JSON, extracts numeric arrays,
        applies noise, and re-encodes.
        """
        try:
            parsed = json.loads(data)

            if isinstance(parsed, dict) and "tensor" in parsed:
                tensor = np.array(parsed["tensor"], dtype=np.float64)
                noisy = self.inject(tensor)
                parsed["tensor"] = noisy.tolist()
                parsed["dp_applied"] = True
                parsed["epsilon"] = self.config.epsilon
            elif isinstance(parsed, list):
                tensor = np.array(parsed, dtype=np.float64)
                noisy = self.inject(tensor)
                return json.dumps(noisy.tolist()).encode()
            else:
                return data

            return json.dumps(parsed).encode()

        except (json.JSONDecodeError, ValueError, TypeError):
            return data
```

- [ ] **Step 3: Update `__init__.py` exports**

Add to `prsm/compute/tee/__init__.py`:

```python
from prsm.compute.tee.dp_noise import DPNoiseInjector
```
Add `"DPNoiseInjector"` to `__all__`.

- [ ] **Step 4: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_dp_noise.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/tee/dp_noise.py prsm/compute/tee/__init__.py tests/unit/test_dp_noise.py
git commit -m "feat(ring7): DPNoiseInjector — calibrated Gaussian noise for activation privacy"
```

---

### Task 3: TEE Runtime + Confidential Executor + Hardware Detection

**Files:**
- Create: `prsm/compute/tee/runtime.py`
- Create: `prsm/compute/tee/confidential_executor.py`
- Modify: `prsm/compute/wasm/profiler_models.py`
- Modify: `prsm/compute/wasm/profiler.py`
- Modify: `prsm/node/gossip.py`
- Test: `tests/unit/test_confidential_executor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_confidential_executor.py`:

```python
"""Tests for TEE runtime and confidential execution pipeline."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
from prsm.compute.tee.confidential_executor import ConfidentialExecutor
from prsm.compute.tee.models import TEEType, DPConfig, PrivacyLevel, TEECapability
from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestTEERuntime:
    def test_software_runtime_implements_interface(self):
        runtime = SoftwareTEERuntime()
        assert isinstance(runtime, TEERuntime)

    def test_software_runtime_name(self):
        runtime = SoftwareTEERuntime()
        assert runtime.name == "software"

    def test_software_runtime_tee_type(self):
        runtime = SoftwareTEERuntime()
        assert runtime.tee_type == TEEType.SOFTWARE

    def test_software_runtime_available(self):
        runtime = SoftwareTEERuntime()
        # Available if wasmtime is installed
        assert isinstance(runtime.available, bool)


class TestSoftwareTEERuntimeExecution:
    @pytest.mark.skipif(
        not SoftwareTEERuntime().available,
        reason="wasmtime not installed",
    )
    def test_execute_in_software_tee(self):
        runtime = SoftwareTEERuntime()
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(
            module=module,
            input_data=b"test",
            resource_limits=ResourceLimits(max_memory_bytes=64 * 1024 * 1024),
        )
        assert result.status == ExecutionStatus.SUCCESS


class TestConfidentialExecutor:
    def test_executor_creation(self):
        executor = ConfidentialExecutor(
            privacy_level=PrivacyLevel.STANDARD,
        )
        assert executor is not None
        assert executor.privacy_level == PrivacyLevel.STANDARD

    @pytest.mark.skipif(
        not SoftwareTEERuntime().available,
        reason="wasmtime not installed",
    )
    def test_confidential_execution(self):
        executor = ConfidentialExecutor(
            privacy_level=PrivacyLevel.STANDARD,
        )
        result = executor.execute_confidential(
            wasm_bytes=MINIMAL_WASM,
            input_data=b"test",
            resource_limits=ResourceLimits(max_memory_bytes=64 * 1024 * 1024),
        )
        assert result.tee_type == TEEType.SOFTWARE
        assert result.dp_applied is True
        assert result.epsilon_spent == 8.0
        assert len(result.output) > 0

    def test_no_privacy_skips_dp(self):
        executor = ConfidentialExecutor(
            privacy_level=PrivacyLevel.NONE,
        )
        # Even without execution, the config should reflect no DP
        assert executor.dp_config.epsilon == float("inf")


class TestHardwareProfileTEE:
    def test_profile_has_tee_fields(self):
        from prsm.compute.wasm.profiler_models import HardwareProfile
        profile = HardwareProfile()
        assert hasattr(profile, "tee_available")
        assert hasattr(profile, "tee_type")
        assert profile.tee_available is False
        assert profile.tee_type == ""

    def test_profile_to_dict_includes_tee(self):
        from prsm.compute.wasm.profiler_models import HardwareProfile
        profile = HardwareProfile(tee_available=True, tee_type="sgx")
        d = profile.to_dict()
        assert d["tee_available"] is True
        assert d["tee_type"] == "sgx"


class TestGossipTEECapability:
    def test_constant_exists(self):
        from prsm.node.gossip import GOSSIP_TEE_CAPABILITY
        assert GOSSIP_TEE_CAPABILITY == "tee_capability"

    def test_retention(self):
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS
        assert GOSSIP_RETENTION_SECONDS.get("tee_capability") == 86400
```

- [ ] **Step 2: Implement TEE runtime**

Create `prsm/compute/tee/runtime.py`:

```python
"""
TEE Runtime Interface
=====================

Abstract interface for trusted execution environments.
SoftwareTEERuntime wraps the Ring 1 WASM sandbox as a fallback
when no hardware TEE is available.
"""

import abc
import logging
from typing import Any

from prsm.compute.tee.models import TEEType
from prsm.compute.wasm.models import ExecutionResult, ResourceLimits

logger = logging.getLogger(__name__)


class TEERuntime(abc.ABC):
    """Abstract interface for TEE execution runtimes."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runtime name."""

    @property
    @abc.abstractmethod
    def tee_type(self) -> TEEType:
        """Type of TEE this runtime provides."""

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this runtime is available on the current hardware."""

    @abc.abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Load a WASM module for execution."""

    @abc.abstractmethod
    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        """Execute in the TEE sandbox."""


class SoftwareTEERuntime(TEERuntime):
    """Software-only TEE using Ring 1 WASM sandbox.

    Provides the same interface as hardware TEEs but without
    hardware-level isolation. Used as fallback when no TEE hardware
    is available.
    """

    def __init__(self):
        self._wasm_runtime = None

    def _get_wasm_runtime(self):
        if self._wasm_runtime is None:
            from prsm.compute.wasm.runtime import WasmtimeRuntime
            self._wasm_runtime = WasmtimeRuntime()
        return self._wasm_runtime

    @property
    def name(self) -> str:
        return "software"

    @property
    def tee_type(self) -> TEEType:
        return TEEType.SOFTWARE

    @property
    def available(self) -> bool:
        return self._get_wasm_runtime().available

    def load(self, wasm_bytes: bytes) -> Any:
        return self._get_wasm_runtime().load(wasm_bytes)

    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        return self._get_wasm_runtime().execute(module, input_data, resource_limits)
```

- [ ] **Step 3: Implement ConfidentialExecutor**

Create `prsm/compute/tee/confidential_executor.py`:

```python
"""
Confidential Executor
=====================

Combines TEE runtime + DP noise injection into a single
secure execution pipeline.
"""

import logging
import time
from typing import Optional

from prsm.compute.tee.models import (
    TEEType,
    DPConfig,
    ConfidentialResult,
    PrivacyLevel,
)
from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
from prsm.compute.tee.dp_noise import DPNoiseInjector
from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus

logger = logging.getLogger(__name__)


class ConfidentialExecutor:
    """Executes WASM modules with TEE isolation and DP noise."""

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
        tee_runtime: Optional[TEERuntime] = None,
    ):
        self.privacy_level = privacy_level
        self.dp_config = PrivacyLevel.config_for_level(privacy_level)
        self._dp_injector = DPNoiseInjector(self.dp_config)
        self._runtime = tee_runtime or SoftwareTEERuntime()

    def execute_confidential(
        self,
        wasm_bytes: bytes,
        input_data: bytes,
        resource_limits: Optional[ResourceLimits] = None,
    ) -> ConfidentialResult:
        """Execute WASM in TEE with DP noise on output."""
        limits = resource_limits or ResourceLimits()
        started_at = time.time()

        try:
            module = self._runtime.load(wasm_bytes)
            exec_result = self._runtime.execute(module, input_data, limits)

            if exec_result.status != ExecutionStatus.SUCCESS:
                return ConfidentialResult(
                    output=exec_result.output,
                    dp_applied=False,
                    epsilon_spent=0.0,
                    tee_type=self._runtime.tee_type,
                    execution_time_seconds=time.time() - started_at,
                )

            # Apply DP noise to output
            if self.privacy_level != PrivacyLevel.NONE:
                noisy_output = self._dp_injector.inject_bytes(exec_result.output)
                dp_applied = True
                epsilon_spent = self.dp_config.epsilon
            else:
                noisy_output = exec_result.output
                dp_applied = False
                epsilon_spent = 0.0

            return ConfidentialResult(
                output=noisy_output,
                dp_applied=dp_applied,
                epsilon_spent=epsilon_spent,
                tee_type=self._runtime.tee_type,
                execution_time_seconds=time.time() - started_at,
                memory_used_bytes=exec_result.memory_used_bytes,
            )

        except Exception as e:
            logger.error(f"Confidential execution failed: {e}")
            return ConfidentialResult(
                output=b"",
                dp_applied=False,
                epsilon_spent=0.0,
                tee_type=self._runtime.tee_type,
                execution_time_seconds=time.time() - started_at,
            )
```

- [ ] **Step 4: Add TEE fields to HardwareProfile**

In `prsm/compute/wasm/profiler_models.py`, add to the `HardwareProfile` dataclass:

```python
    # TEE (Ring 7)
    tee_available: bool = False
    tee_type: str = ""  # "sgx", "tdx", "sev", "trustzone", "secure_enclave", ""
```

Update `to_dict()` and `from_dict()` to include these fields.

- [ ] **Step 5: Add TEE detection to HardwareProfiler**

In `prsm/compute/wasm/profiler.py`, add a `_detect_tee()` method:

```python
    def _detect_tee(self) -> tuple:
        """Detect trusted execution environment support.

        Returns (tee_available: bool, tee_type: str).
        """
        # Check Intel SGX
        try:
            import subprocess
            result = subprocess.run(
                ["cpuid", "-1", "-l", "0x12"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "SGX" in result.stdout:
                return True, "sgx"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check Apple Secure Enclave (macOS with T2/Apple Silicon)
        import platform
        if platform.system() == "Darwin":
            import os
            if os.path.exists("/usr/lib/libcoreauthd.dylib"):
                return True, "secure_enclave"

        # Check /dev/sgx_enclave (Linux SGX driver)
        import os
        if os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/sgx/enclave"):
            return True, "sgx"

        return False, ""
```

Call `_detect_tee()` in the `detect()` method and pass results to HardwareProfile.

- [ ] **Step 6: Add GOSSIP_TEE_CAPABILITY**

In `prsm/node/gossip.py`, add:

```python
GOSSIP_TEE_CAPABILITY = "tee_capability"
```

And in `GOSSIP_RETENTION_SECONDS`:
```python
"tee_capability": 86400,  # 24 hours
```

- [ ] **Step 7: Update `__init__.py` exports**

Add to `prsm/compute/tee/__init__.py`:
```python
from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
from prsm.compute.tee.confidential_executor import ConfidentialExecutor
```

- [ ] **Step 8: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_tee_models.py tests/unit/test_dp_noise.py tests/unit/test_confidential_executor.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add prsm/compute/tee/runtime.py prsm/compute/tee/confidential_executor.py prsm/compute/tee/__init__.py prsm/compute/wasm/profiler_models.py prsm/compute/wasm/profiler.py prsm/node/gossip.py tests/unit/test_confidential_executor.py
git commit -m "feat(ring7): TEERuntime, ConfidentialExecutor, TEE hardware detection, GOSSIP_TEE_CAPABILITY"
```

---

### Task 4: Node Integration + Smoke Test + Version Bump + Publish

**Files:**
- Modify: `prsm/node/node.py`
- Create: `tests/integration/test_ring7_vault.py`
- Modify: `prsm/__init__.py`, `pyproject.toml`

- [ ] **Step 1: Wire into node.py**

After the Ring 5 block, add:

```python
        # ── Confidential Compute (Ring 7) ─────────────────────────────
        try:
            from prsm.compute.tee.confidential_executor import ConfidentialExecutor
            from prsm.compute.tee.models import PrivacyLevel

            self.confidential_executor = ConfidentialExecutor(
                privacy_level=PrivacyLevel.STANDARD,
            )
            logger.info("Confidential compute (Ring 7) initialized")
        except ImportError:
            self.confidential_executor = None
            logger.debug("Confidential compute not available")
```

- [ ] **Step 2: Create smoke test**

Create `tests/integration/test_ring7_vault.py` with tests for: TEE capability detection, DP noise injection on real data, confidential execution pipeline, gossip constant, all Ring 1-7 imports.

- [ ] **Step 3: Version bump to 0.32.0, commit, push, publish**

```bash
git add prsm/node/node.py tests/integration/test_ring7_vault.py prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.32.0 for Ring 7 — The Vault (confidential compute)"
git push origin main
python3 -m build && python3 -m twine upload dist/prsm_network-0.32.0*
```
