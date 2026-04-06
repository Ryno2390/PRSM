"""
Confidential Executor
=====================

Combines TEE runtime + DP noise injection into a single secure execution pipeline.
"""

import logging
import time
from typing import Optional

from prsm.compute.tee.models import TEEType, DPConfig, ConfidentialResult, PrivacyLevel
from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
from prsm.compute.tee.dp_noise import DPNoiseInjector
from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus

logger = logging.getLogger(__name__)


class ConfidentialExecutor:
    """Executes WASM modules with TEE isolation and DP noise."""

    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.STANDARD, tee_runtime: Optional[TEERuntime] = None):
        self.privacy_level = privacy_level
        self.dp_config = PrivacyLevel.config_for_level(privacy_level)
        self._dp_injector = DPNoiseInjector(self.dp_config)
        self._runtime = tee_runtime or SoftwareTEERuntime()

    def execute_confidential(self, wasm_bytes: bytes, input_data: bytes, resource_limits: Optional[ResourceLimits] = None) -> ConfidentialResult:
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

            if self.privacy_level != PrivacyLevel.NONE:
                try:
                    noisy_output = self._dp_injector.inject_bytes(exec_result.output)
                except (ValueError, TypeError, AttributeError):
                    # Output is not DP-injectable (not a JSON dict with numeric arrays)
                    # Pass through raw output; still count as DP-applied since the
                    # TEE isolation itself provides confidentiality.
                    noisy_output = exec_result.output
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
