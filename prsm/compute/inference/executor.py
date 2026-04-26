"""
Inference executor — interface for TEE-attested model inference on PRSM.

Phase 3.x.1 Task 1 scaffold: this module provides the abstract interface and
a `MockInferenceExecutor` that returns deterministic stub results, sufficient
for downstream test harnesses while real implementations land in Task 4.

Real ``InferenceExecutor`` will compose:
- ``TensorParallelExecutor`` (Phase 2 — already shipped)
- ``ConfidentialExecutor`` (Phase 2 TEE — already shipped)
- ``content_tier_gate`` (Task 3) for Tier B/C decryption inside TEE
- ``InferenceReceipt`` signing (Task 2)
- ``PaymentEscrow`` integration via ``/compute/inference`` API (Task 5)

Until those land, all higher-level code (MCP tool, HTTP API) can develop
against the abstract interface using the mock executor.
"""

from __future__ import annotations

import abc
import hashlib
import time
import uuid
from decimal import Decimal
from typing import Optional

from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType


class InferenceExecutorError(Exception):
    """Base exception for inference execution failures."""


class InsufficientBudgetError(InferenceExecutorError):
    """Raised when the request budget is below the minimum cost estimate."""


class UnsupportedModelError(InferenceExecutorError):
    """Raised when the requested model_id is not registered."""


class TEERequirementError(InferenceExecutorError):
    """Raised when a privacy_tier requires hardware TEE but only software is available.

    Per design plan §6.5: ``privacy_tier=none`` accepts software-TEE; ``standard``
    / ``high`` / ``maximum`` require hardware TEE.
    """


class InferenceExecutor(abc.ABC):
    """Abstract interface for inference execution.

    Real implementations land in Task 4 (``TensorParallelInferenceExecutor``).
    This base class plus ``MockInferenceExecutor`` lets Tasks 5-13 develop
    against a stable interface without waiting on the real engine.
    """

    @abc.abstractmethod
    async def execute(self, request: InferenceRequest) -> InferenceResult:
        """Run inference for a single request.

        Implementations are responsible for:
        1. Validating ``request.budget_ftns`` against the cost estimate
        2. Validating ``request.privacy_tier`` against available TEE
        3. Loading model shards per ``request.model_id``
        4. Decrypting Tier B/C content inside TEE if applicable
        5. Running tensor-parallel inference
        6. Applying DP noise per privacy_tier
        7. Generating a signed ``InferenceReceipt``
        8. Returning ``InferenceResult`` with output + receipt

        On failure, return ``InferenceResult.failure(request_id, error)``.
        """

    @abc.abstractmethod
    async def estimate_cost(self, request: InferenceRequest) -> Decimal:
        """Estimate the FTNS cost for a request without executing it.

        Used by the ``prsm_quote`` flow and pre-execution budget validation.
        """

    @abc.abstractmethod
    def supported_models(self) -> list[str]:
        """Return the list of currently-registered ``model_id`` values."""


# --------------------------------------------------------------------------
# Mock executor
# --------------------------------------------------------------------------


class MockInferenceExecutor(InferenceExecutor):
    """Deterministic stub executor for use in tests and during scaffold phase.

    Returns a synthetic output derived from the prompt + model_id, with a
    matching deterministic receipt. Does NOT actually run any model;
    cryptographic fields (``tee_attestation``, ``settler_signature``) contain
    zero bytes and should NOT be trusted by real verifiers.

    Once Task 4's real executor lands, ``MockInferenceExecutor`` remains useful
    for unit tests that exercise the surrounding pipeline (MCP tool, API
    endpoint, billing visibility) without spinning up real model shards.
    """

    DEFAULT_MOCK_MODELS = ("mock-llama-3-8b", "mock-mistral-7b", "mock-phi-3")

    def __init__(self, models: Optional[list[str]] = None, fixed_cost: Decimal = Decimal("0.10")) -> None:
        self._models = list(models) if models is not None else list(self.DEFAULT_MOCK_MODELS)
        self._fixed_cost = fixed_cost

    def supported_models(self) -> list[str]:
        return list(self._models)

    async def estimate_cost(self, request: InferenceRequest) -> Decimal:
        # Mock: flat cost regardless of inputs. Real executor scales with PCU
        # consumption per model + token count + privacy tier (TEE attestation
        # has measurable gas overhead).
        if request.model_id not in self._models:
            raise UnsupportedModelError(f"Unknown model_id: {request.model_id}")
        return self._fixed_cost

    async def execute(self, request: InferenceRequest) -> InferenceResult:
        # Validate model
        if request.model_id not in self._models:
            return InferenceResult.failure(
                request.request_id,
                f"Unknown model_id: {request.model_id}",
            )

        # Validate budget
        cost = self._fixed_cost
        if request.budget_ftns < cost:
            return InferenceResult.failure(
                request.request_id,
                f"Insufficient budget: {request.budget_ftns} FTNS < required {cost} FTNS",
            )

        # Synthetic deterministic output
        seed = f"{request.model_id}|{request.prompt}".encode("utf-8")
        digest = hashlib.sha256(seed).hexdigest()
        output = (
            f"[mock {request.model_id}] echo response for prompt-hash={digest[:16]} "
            f"(privacy={request.privacy_tier.value}, content_tier={request.content_tier.value})"
        )

        # Build mock receipt with deterministic but zero-valued cryptographic fields
        output_hash = hashlib.sha256(output.encode("utf-8")).digest()
        job_id = f"mock-job-{uuid.uuid4().hex[:12]}"
        receipt = InferenceReceipt(
            job_id=job_id,
            request_id=request.request_id,
            model_id=request.model_id,
            content_tier=request.content_tier,
            privacy_tier=request.privacy_tier,
            epsilon_spent=self._epsilon_for_level(request.privacy_tier),
            tee_type=TEEType.SOFTWARE,  # Mock executor never has hardware TEE
            tee_attestation=b"\x00" * 64,  # Placeholder — real attestation in Task 2
            output_hash=output_hash,
            duration_seconds=0.001,
            cost_ftns=cost,
            settler_signature=b"\x00" * 64,  # Placeholder — real signature in Task 2
            settler_node_id="mock-settler",
        )

        return InferenceResult(
            request_id=request.request_id,
            success=True,
            output=output,
            receipt=receipt,
        )

    @staticmethod
    def _epsilon_for_level(level: PrivacyLevel) -> float:
        """Map privacy level to epsilon for the mock receipt.

        Real executor pulls this from the ``ConfidentialExecutor.dp_config``
        once inference completes. Mock uses the canonical mapping.
        """
        return {
            PrivacyLevel.NONE: float("inf"),
            PrivacyLevel.STANDARD: 8.0,
            PrivacyLevel.HIGH: 4.0,
            PrivacyLevel.MAXIMUM: 1.0,
        }[level]


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def default_mock_executor() -> MockInferenceExecutor:
    """Convenience factory for the default mock executor.

    Useful in tests that don't need to customize models or pricing.
    """
    return MockInferenceExecutor()
