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
import logging
import os
import time
import uuid
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

# Public marker that brands software-TEE attestation blobs as
# dev-only. Verifiers MUST treat any receipt whose tee_attestation
# starts with this prefix as having NO confidentiality guarantee
# beyond the job_id binding, regardless of what tee_type the receipt
# claims. Hardware-TEE attestations never have this prefix — they
# carry platform-vendor signed enclave measurements.
SOFTWARE_TEE_ATTESTATION_PREFIX = b"DEV-ONLY-SW-TEE:"

# Environment variable that controls whether a software-TEE
# executor accepts non-NONE privacy tiers at construction time.
# Set to "0" / "false" / "no" on production nodes serving Tier B/C
# confidential workloads. Default (unset) preserves the
# constructor-arg default (currently True) so dev/test runs
# don't break.
SOFTWARE_TEE_PRIVACY_ENV = "PRSM_ALLOW_SOFTWARE_TEE_FOR_PRIVACY"

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


# --------------------------------------------------------------------------
# Real executor — Phase 3.x.1 Task 4
# --------------------------------------------------------------------------


class TensorParallelInferenceExecutor(InferenceExecutor):
    """Real inference executor: tensor-parallel forward pass + DP noise + receipt.

    Adapter from ``InferenceRequest`` to the existing
    ``TensorParallelExecutor`` (Phase 2 Ring 8 — already shipped) and
    ``DPNoiseInjector`` (Phase 2 TEE — already shipped).

    Composition:
      1. Validate model + budget + TEE.
      2. Encode prompt → numpy float64 input vector sized to model's
         first-shard input dim.
      3. Build all-local node assignments and call
         ``TensorParallelExecutor.execute_parallel(...)`` for the
         per-shard forward pass + ring all-reduce aggregation.
      4. Inject calibrated Gaussian DP noise on the aggregated output
         per ``request.privacy_tier`` (NONE skips).
      5. Format human-readable output + sha256(output_hash).
      6. Build ``InferenceReceipt`` with TEE attestation +
         duration + cost. ``settler_signature`` is set to zeros here;
         the API layer (Task 5 — ``/compute/inference``) signs the
         receipt under the serving node's identity before returning.

    Single-node by default (all shards local). Distributed dispatch
    via ``TensorParallelExecutor.remote_dispatcher`` is wired through
    when callers supply a ``tensor_executor`` configured with one.

    Model registry: as of Phase 3.x.2 the constructor accepts either a
    ``ModelRegistry`` (production path — typically
    ``FilesystemModelRegistry`` for restart survival, or
    ``InMemoryModelRegistry`` for tests) or a ``Dict[str, ShardedModel]``
    (Phase 3.x.1 back-compat). Dict callers get a scaffold identity
    that signs the auto-built manifests; the scaffold key never
    touches inference receipts (those continue to be signed by the
    node identity at the API layer). Every ``execute()`` re-verifies
    the registry's signature + shard sha256 commitments — tampering
    surfaces as ``InferenceResult.failure(...)``, not silent corruption.
    """

    def __init__(
        self,
        model_registry,  # Dict[str, ShardedModel] OR ModelRegistry
        *,
        tensor_executor: "Optional[TensorParallelExecutor]" = None,  # noqa: F821
        tee_runtime: "Optional[TEERuntime]" = None,  # noqa: F821
        cost_per_shard: Decimal = Decimal("0.05"),
        privacy_overhead: "Optional[Dict[PrivacyLevel, Decimal]]" = None,
        allow_software_tee_for_privacy: Optional[bool] = None,
    ) -> None:
        # Imports done lazily to avoid heavyweight numpy/runtime cost when
        # callers only need the abstract interface or the mock executor.
        from prsm.compute.model_registry import (
            InMemoryModelRegistry,
            ModelRegistry,
        )
        from prsm.compute.model_sharding.executor import TensorParallelExecutor
        from prsm.compute.tee.runtime import SoftwareTEERuntime
        from prsm.node.identity import generate_node_identity

        # Phase 3.x.2 Task 5 — accept either a ModelRegistry (preferred,
        # production path) or a Dict[str, ShardedModel] (Phase 3.x.1
        # back-compat). Dict callers get a scaffold identity that signs
        # the in-memory manifests on their behalf; the scaffold key
        # never touches inference receipts (those are signed by the
        # node identity at the API layer).
        if isinstance(model_registry, ModelRegistry):
            self._registry = model_registry
            self._scaffold_identity = None
        elif isinstance(model_registry, dict):
            scaffold = generate_node_identity(
                display_name="prsm-tpie-scaffold"
            )
            registry = InMemoryModelRegistry()
            for model in model_registry.values():
                registry.register(model, identity=scaffold)
            self._registry = registry
            self._scaffold_identity = scaffold
        else:
            raise TypeError(
                f"model_registry must be a ModelRegistry or "
                f"Dict[str, ShardedModel], got {type(model_registry).__name__}"
            )

        self._tensor = tensor_executor or TensorParallelExecutor()
        self._tee_runtime = tee_runtime or SoftwareTEERuntime()
        self._cost_per_shard = Decimal(cost_per_shard)
        self._privacy_overhead = privacy_overhead or {
            PrivacyLevel.NONE: Decimal("1.00"),
            PrivacyLevel.STANDARD: Decimal("1.10"),
            PrivacyLevel.HIGH: Decimal("1.25"),
            PrivacyLevel.MAXIMUM: Decimal("1.50"),
        }

        # Resolve allow_software_tee_for_privacy in this precedence order:
        #   1. explicit constructor arg (highest)
        #   2. PRSM_ALLOW_SOFTWARE_TEE_FOR_PRIVACY env var
        #   3. default True (Phase 3.x.1 dev-friendly default)
        # When the resolved value is True AND the runtime is software,
        # emit an operator-facing warning so accidental production runs
        # don't ship Tier B/C confidential workloads on a software TEE.
        self._allow_software_tee_for_privacy = self._resolve_swtee_privacy_flag(
            explicit=allow_software_tee_for_privacy
        )
        if (
            self._allow_software_tee_for_privacy
            and self._tee_runtime.tee_type == TEEType.SOFTWARE
        ):
            logger.warning(
                "TensorParallelInferenceExecutor: software TEE accepts non-NONE "
                "privacy tiers (allow_software_tee_for_privacy=True). DP-ε spends "
                "will be recorded on receipts but the underlying confidentiality "
                "guarantee is software-only. Set %s=0 on production nodes serving "
                "Tier B/C confidential workloads.",
                SOFTWARE_TEE_PRIVACY_ENV,
            )

    @staticmethod
    def _resolve_swtee_privacy_flag(*, explicit: Optional[bool]) -> bool:
        """Resolve allow_software_tee_for_privacy from explicit arg / env / default."""
        if explicit is not None:
            return bool(explicit)
        env_value = os.environ.get(SOFTWARE_TEE_PRIVACY_ENV)
        if env_value is not None:
            return env_value.strip().lower() not in {"0", "false", "no", "off", ""}
        return True  # Phase 3.x.1 dev-friendly default

    @property
    def tee_runtime(self):
        """Expose the underlying TEE runtime for callers/tests."""
        return self._tee_runtime

    @property
    def registry(self):
        """The underlying ModelRegistry — exposed for callers/tests."""
        return self._registry

    def supported_models(self) -> list[str]:
        return self._registry.list_models()

    def register_model(
        self,
        model,  # ShardedModel
        *,
        identity=None,  # Optional[NodeIdentity]
    ) -> None:
        """Register a model with the underlying registry.

        Identity precedence:
          1. Explicit ``identity`` kwarg (production path: caller owns
             the publisher key).
          2. Scaffold identity generated for dict-arg constructions.
          3. Fresh scaffold (cached) for callers who passed a real
             ``ModelRegistry`` but didn't supply an identity for this
             call — preserves the Phase 3.x.1 ``register_model(m)``
             call shape.

        Raises ``ModelAlreadyRegisteredError`` on duplicate model_id —
        explicit semantics replace the Phase 3.x.1 silent-overwrite
        behavior. Callers needing replacement must build a fresh
        registry or, in the future, call an explicit ``unregister``
        (not in v1 scope).
        """
        from prsm.node.identity import generate_node_identity

        publisher = identity
        if publisher is None:
            if self._scaffold_identity is None:
                # First call on a real-registry executor without an
                # identity — generate and stash so subsequent calls
                # use the same publisher.
                self._scaffold_identity = generate_node_identity(
                    display_name="prsm-tpie-scaffold"
                )
            publisher = self._scaffold_identity
        self._registry.register(model, identity=publisher)

    async def estimate_cost(self, request: InferenceRequest) -> Decimal:
        # Use get_manifest (metadata only) — cheaper than get() because
        # it skips shard byte verification.
        from prsm.compute.model_registry import ModelNotFoundError
        try:
            manifest = self._registry.get_manifest(request.model_id)
        except ModelNotFoundError as e:
            raise UnsupportedModelError(f"Unknown model_id: {request.model_id}") from e
        base = self._cost_per_shard * Decimal(manifest.total_shards)
        overhead = self._privacy_overhead.get(request.privacy_tier, Decimal("1.0"))
        return base * overhead

    async def execute(self, request: InferenceRequest) -> InferenceResult:
        # 1. Model lookup — registry.get() verifies signature + shard
        # sha256 commitments. Tampering surfaces as ManifestVerificationError
        # which we map to InferenceResult.failure (rather than letting it
        # propagate) so the caller gets a structured response.
        from prsm.compute.model_registry import (
            ManifestVerificationError,
            ModelNotFoundError,
        )
        try:
            sharded = self._registry.get(request.model_id)
        except ModelNotFoundError:
            return InferenceResult.failure(
                request.request_id, f"Unknown model_id: {request.model_id}"
            )
        except ManifestVerificationError as e:
            return InferenceResult.failure(
                request.request_id,
                f"Model registry verification failed: {e}",
            )

        # 2. Budget check
        cost = await self.estimate_cost(request)
        if request.budget_ftns < cost:
            return InferenceResult.failure(
                request.request_id,
                f"Insufficient budget: {request.budget_ftns} FTNS < required {cost} FTNS",
            )

        # 3. TEE check for privacy_tier != NONE
        if request.privacy_tier != PrivacyLevel.NONE:
            if (
                self._tee_runtime.tee_type == TEEType.SOFTWARE
                and not self._allow_software_tee_for_privacy
            ):
                return InferenceResult.failure(
                    request.request_id,
                    f"privacy_tier={request.privacy_tier.value} requires hardware TEE; "
                    f"available={self._tee_runtime.tee_type.value}",
                )

        started_at = time.time()

        # 4. Encode prompt → numpy bytes sized to first shard's input dim
        try:
            input_data = self._encode_prompt(request.prompt, sharded)
        except (ValueError, IndexError) as e:
            return InferenceResult.failure(
                request.request_id, f"Prompt encoding failed: {e}"
            )

        # 5. All-local assignments — one entry per shard
        assignments = [
            {"shard_index": s.shard_index, "node_id": "local"}
            for s in sharded.shards
        ]

        # 6. Tensor-parallel forward pass
        try:
            tp_result = await self._tensor.execute_parallel(
                sharded, input_data, assignments
            )
        except Exception as e:
            return InferenceResult.failure(
                request.request_id, f"Tensor-parallel execution failed: {e}"
            )

        if tp_result.get("status") != "success":
            return InferenceResult.failure(
                request.request_id,
                f"Tensor-parallel execution failed: {tp_result.get('errors')}",
            )

        # 7. Apply DP noise on aggregated output
        import numpy as np
        from prsm.compute.tee.dp_noise import DPNoiseInjector

        aggregated = np.asarray(tp_result["aggregated_output"], dtype=np.float64)
        dp_config = PrivacyLevel.config_for_level(request.privacy_tier)
        if request.privacy_tier != PrivacyLevel.NONE:
            injector = DPNoiseInjector(dp_config)
            aggregated = injector.inject(aggregated)
            epsilon_spent = dp_config.epsilon
        else:
            epsilon_spent = 0.0

        # 8. Format output + receipt
        output = self._format_output(request, aggregated)
        output_hash = hashlib.sha256(output.encode("utf-8")).digest()
        job_id = f"infer-job-{uuid.uuid4().hex[:12]}"
        duration = time.time() - started_at

        receipt = InferenceReceipt(
            job_id=job_id,
            request_id=request.request_id,
            model_id=request.model_id,
            content_tier=request.content_tier,
            privacy_tier=request.privacy_tier,
            epsilon_spent=epsilon_spent,
            tee_type=self._tee_runtime.tee_type,
            tee_attestation=self._build_attestation(job_id),
            output_hash=output_hash,
            duration_seconds=duration,
            cost_ftns=cost,
            settler_signature=b"\x00" * 64,  # signed at API layer (Task 5)
            settler_node_id="",  # populated at API layer
        )

        return InferenceResult(
            request_id=request.request_id,
            success=True,
            output=output,
            receipt=receipt,
        )

    @staticmethod
    def _encode_prompt(prompt: str, model) -> bytes:
        """Encode a prompt string as numpy float64 bytes sized to the
        first shard's input dimension.

        This is a deterministic hash-based encoding — NOT real LLM
        tokenization. Real tokenization (BPE/SentencePiece) lands when
        a real model registry replaces the in-memory one. The numerics
        contract for now: same prompt + same model → identical bytes,
        suitable as input to ``execute_shard_locally``.
        """
        import numpy as np

        if not model.shards:
            raise ValueError(f"model {model.model_id!r} has no shards")
        first = model.shards[0]
        tensor = np.frombuffer(first.tensor_data, dtype=np.float64).reshape(
            first.tensor_shape
        )
        if tensor.ndim == 2:
            input_dim = int(tensor.shape[1])
        else:
            input_dim = int(tensor.shape[0])
        if input_dim <= 0:
            raise ValueError(f"non-positive input dim: {input_dim}")

        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        # Spread digest bytes across input_dim float values in [-1, 1].
        floats = np.array(
            [(digest[i % len(digest)] / 128.0 - 1.0) for i in range(input_dim)],
            dtype=np.float64,
        )
        return floats.tobytes()

    @staticmethod
    def _format_output(request: InferenceRequest, aggregated) -> str:
        """Human-readable summary of the aggregated tensor output.

        Real LLM inference would decode logits → tokens here; until a
        real model registry lands, callers see the tensor stats.
        """
        flat = aggregated.flatten()
        sample = [round(float(x), 6) for x in flat[:8].tolist()]
        return (
            f"[{request.model_id}] tensor-parallel inference complete | "
            f"prompt_len={len(request.prompt)} | "
            f"privacy={request.privacy_tier.value} | "
            f"content_tier={request.content_tier.value} | "
            f"output_dim={int(flat.size)} | "
            f"sample={sample}"
        )

    @staticmethod
    def _build_attestation(job_id: str) -> bytes:
        """Produce a job-bound attestation blob.

        Real hardware TEE attestation includes enclave measurements +
        platform certificate signed by the platform vendor (Intel ASP,
        AMD KDS, Apple SEP, etc.). Software fallback emits a 64-byte
        DEV-ONLY-marked blob: a 16-byte ASCII prefix
        (``SOFTWARE_TEE_ATTESTATION_PREFIX``) followed by 48 bytes of
        sha384("sw-tee:" + job_id) for job binding.

        Verifiers MUST reject any attestation starting with the
        DEV-ONLY prefix as a confidentiality proof, regardless of the
        receipt's claimed tee_type. The prefix exists so a single
        bytestring inspection — no parsing of receipt fields — is
        enough to flag the attestation as non-production.
        """
        digest = hashlib.sha384(f"sw-tee:{job_id}".encode("utf-8")).digest()
        blob = SOFTWARE_TEE_ATTESTATION_PREFIX + digest
        # Belt-and-suspenders length check: 16 + 48 = 64. If the prefix
        # ever drifts, fail loudly instead of silently shrinking receipts.
        assert len(blob) == 64, f"sw-tee attestation must be 64 bytes, got {len(blob)}"
        return blob
