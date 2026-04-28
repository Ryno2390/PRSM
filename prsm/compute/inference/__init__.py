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

# Phase 3.x.7 — cross-host ChainExecutor implementation is re-exported
# at the inference package surface so production callers get one-call
# wiring: `from prsm.compute.inference import make_rpc_chain_executor`.
#
# Lazy via PEP 562 ``__getattr__`` because ``prsm.compute.chain_rpc``
# imports from ``prsm.compute.inference.models`` — eager re-export
# would create a circular import. The lazy path triggers the
# chain_rpc import only when one of these names is accessed.
_LAZY_CHAIN_RPC_NAMES = frozenset({
    "ChainRpcError",
    "LayerStageServer",
    "RpcChainExecutor",
    "make_layer_stage_server",
    "make_rpc_chain_executor",
})


def __getattr__(name: str):
    if name in _LAZY_CHAIN_RPC_NAMES:
        from prsm.compute.chain_rpc import (
            ChainExecutionError,
            LayerStageServer,
            RpcChainExecutor,
            make_layer_stage_server,
            make_rpc_chain_executor,
        )
        attrs = {
            "ChainRpcError": ChainExecutionError,
            "LayerStageServer": LayerStageServer,
            "RpcChainExecutor": RpcChainExecutor,
            "make_layer_stage_server": make_layer_stage_server,
            "make_rpc_chain_executor": make_rpc_chain_executor,
        }
        return attrs[name]
    raise AttributeError(
        f"module 'prsm.compute.inference' has no attribute {name!r}"
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
    # Phase 3.x.7 — cross-host ChainExecutor implementation
    "RpcChainExecutor",
    "LayerStageServer",
    "ChainRpcError",
    "make_rpc_chain_executor",
    "make_layer_stage_server",
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
