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
    UnsupportedModelError,
    default_mock_executor,
)
from prsm.compute.inference.receipt import (
    is_signed,
    sign_receipt,
    verify_receipt,
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

__all__ = [
    # Models
    "ContentTier",
    "InferenceReceipt",
    "InferenceRequest",
    "InferenceResult",
    # Executor interface
    "InferenceExecutor",
    "MockInferenceExecutor",
    "default_mock_executor",
    # Receipt signing (Task 2)
    "sign_receipt",
    "verify_receipt",
    "is_signed",
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
