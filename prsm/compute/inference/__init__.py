"""
PRSM Inference module — TEE-attested model inference.

Phase 3.x.1 scaffold. Surfaces the public types + interfaces for the
``prsm_inference`` MCP tool path (Tasks 4-6) and downstream HTTP API
(Task 5). Implementation of the real ``InferenceExecutor`` lands in Task 4;
``InferenceReceipt`` Ed25519 signing in Task 2; Tier B/C content gating in
Task 3.

Until Task 4 lands, ``MockInferenceExecutor`` provides a deterministic stub
sufficient for developing the surrounding pipeline (MCP tool, billing
visibility, streaming, E2E tests).

See `docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md` for
the full design + task list.
"""

from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.executor import (
    InferenceExecutor,
    InferenceExecutorError,
    InsufficientBudgetError,
    MockInferenceExecutor,
    TEERequirementError,
    TensorParallelInferenceExecutor,
    UnsupportedModelError,
    default_mock_executor,
)
from prsm.compute.inference.multi_stage_attestation import (
    MULTI_STAGE_ATTESTATION_VERSION,
    MULTI_STAGE_MAGIC_PREFIX,
    MultiStageAttestationError,
    MultiStageMalformedError,
    StageAttestation,
    StageVerificationResult,
    decode_multi_stage_attestation,
    encode_multi_stage_attestation,
    is_multi_stage_attestation,
    worst_case_tee_type,
)
from prsm.compute.inference.receipt import (
    is_signed,
    sign_receipt,
    verify_receipt,
    verify_stage_attestations,
)
from prsm.compute.inference.content_tier_gate import (
    ContentTierGateError,
    MissingMaterialError,
    TEEContext,
    TEEContextRequiredError,
    TierBMaterial,
    TierCMaterial,
    open_content,
    open_tier_a,
    open_tier_b,
    open_tier_c,
)
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
    ChainExecutor,
    GpuPoolProvider,
    ParallaxScheduledExecutor,
)

__all__ = [
    # Models
    "ContentTier",
    "InferenceReceipt",
    "InferenceRequest",
    "InferenceResult",
    # Executor interface
    "InferenceExecutor",
    "MockInferenceExecutor",
    "TensorParallelInferenceExecutor",
    "default_mock_executor",
    # Phase 3.x.6 — Parallax-scheduled distributed executor
    "ParallaxScheduledExecutor",
    "ChainExecutor",
    "ChainExecutionResult",
    "GpuPoolProvider",
    # Receipt signing (Task 2)
    "sign_receipt",
    "verify_receipt",
    "is_signed",
    # Multi-stage TEE attestation (Phase 3.x.7 Task 5)
    "MULTI_STAGE_ATTESTATION_VERSION",
    "MULTI_STAGE_MAGIC_PREFIX",
    "MultiStageAttestationError",
    "MultiStageMalformedError",
    "StageAttestation",
    "StageVerificationResult",
    "decode_multi_stage_attestation",
    "encode_multi_stage_attestation",
    "is_multi_stage_attestation",
    "verify_stage_attestations",
    "worst_case_tee_type",
    # Content tier gate (Task 3)
    "TEEContext",
    "TierBMaterial",
    "TierCMaterial",
    "open_content",
    "open_tier_a",
    "open_tier_b",
    "open_tier_c",
    # Exceptions
    "InferenceExecutorError",
    "InsufficientBudgetError",
    "TEERequirementError",
    "UnsupportedModelError",
    "ContentTierGateError",
    "TEEContextRequiredError",
    "MissingMaterialError",
]
