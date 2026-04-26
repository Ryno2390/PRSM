"""
Inference data models.

Defines the request/response/receipt types for TEE-attested model inference
exposed via the `prsm_inference` MCP tool. Phase 3.x.1 Task 1 scaffold —
types and serialization only; actual inference execution lands in Tasks 4-5.

Two layers of privacy per `PRSM_Vision.md` §7:
- **Layer 1 — content tier** (A / B / C): encryption status of the data being
  queried. Tier A is public plaintext; B is encrypted-before-sharding;
  C is B plus Reed-Solomon erasure coding plus Shamir-split decryption keys.
- **Layer 2 — privacy tier** (none / standard / high / maximum): TEE attestation
  plus DP noise on activations. Reuses `prsm.compute.tee.PrivacyLevel`.

Both surface as explicit fields on `InferenceRequest`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
import uuid

from prsm.compute.tee.models import PrivacyLevel, TEEType


class ContentTier(str, Enum):
    """Encryption status of content being queried.

    See `PRSM_Vision.md` §2 "Data layer" for full definitions of each tier.

    - ``A`` — public content; node operators see plaintext shards
    - ``B`` — encrypted-before-sharding; ciphertext-only on operators;
      decryption key released on payment per Phase 7-storage
    - ``C`` — Tier B plus Reed-Solomon erasure coding (K-of-N reconstruction)
      plus Shamir-split decryption keys (M-of-N reconstruction)
    """

    A = "A"
    B = "B"
    C = "C"


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceRequest:
    """A request to run TEE-attested inference on PRSM.

    Constructed by `prsm_inference` MCP handler from caller arguments; passed
    to ``InferenceExecutor.execute()``. Frozen because requests should be
    immutable once accepted into the dispatch pipeline (any per-shard mutation
    happens on copies).
    """

    prompt: str
    model_id: str
    budget_ftns: Decimal
    privacy_tier: PrivacyLevel = PrivacyLevel.STANDARD
    content_tier: ContentTier = ContentTier.A
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    requester_node_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: f"infer-{uuid.uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ to coerce types after init
        if not isinstance(self.budget_ftns, Decimal):
            object.__setattr__(self, "budget_ftns", Decimal(str(self.budget_ftns)))
        if not isinstance(self.privacy_tier, PrivacyLevel):
            object.__setattr__(self, "privacy_tier", PrivacyLevel(self.privacy_tier))
        if not isinstance(self.content_tier, ContentTier):
            object.__setattr__(self, "content_tier", ContentTier(self.content_tier))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "model_id": self.model_id,
            "budget_ftns": str(self.budget_ftns),
            "privacy_tier": self.privacy_tier.value,
            "content_tier": self.content_tier.value,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "requester_node_id": self.requester_node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        d = dict(data)
        if "budget_ftns" in d:
            d["budget_ftns"] = Decimal(str(d["budget_ftns"]))
        if "privacy_tier" in d:
            d["privacy_tier"] = PrivacyLevel(d["privacy_tier"])
        if "content_tier" in d:
            d["content_tier"] = ContentTier(d["content_tier"])
        # Drop unknown keys gracefully so additive schema changes don't break
        # cross-version replays.
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceReceipt:
    """TEE-attested receipt for a single inference execution.

    Per Phase 3.x.1 design plan §3.3, this receipt converts MCP inference
    from "trust the provider" to "verifiable inference":

    1. Caller receives receipt + output
    2. Caller verifies ``settler_signature`` against settling node identity
    3. Caller verifies ``tee_attestation`` against platform vendor's attestation
       service (Intel ASP for SGX/TDX, AMD KDS for SEV-SNP)
    4. Caller confirms ``output_hash`` matches received output

    All four steps run client-side; PRSM does not need to be trusted.
    """

    job_id: str
    request_id: str
    model_id: str
    content_tier: ContentTier
    privacy_tier: PrivacyLevel
    epsilon_spent: float
    tee_type: TEEType
    tee_attestation: bytes
    output_hash: bytes
    duration_seconds: float
    cost_ftns: Decimal
    settler_signature: bytes = b""
    settler_node_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "content_tier": self.content_tier.value,
            "privacy_tier": self.privacy_tier.value,
            "epsilon_spent": self.epsilon_spent,
            "tee_type": self.tee_type.value,
            "tee_attestation": self.tee_attestation.hex(),
            "output_hash": self.output_hash.hex(),
            "duration_seconds": self.duration_seconds,
            "cost_ftns": str(self.cost_ftns),
            "settler_signature": self.settler_signature.hex(),
            "settler_node_id": self.settler_node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceReceipt":
        d = dict(data)
        if "content_tier" in d and not isinstance(d["content_tier"], ContentTier):
            d["content_tier"] = ContentTier(d["content_tier"])
        if "privacy_tier" in d and not isinstance(d["privacy_tier"], PrivacyLevel):
            d["privacy_tier"] = PrivacyLevel(d["privacy_tier"])
        if "tee_type" in d and not isinstance(d["tee_type"], TEEType):
            d["tee_type"] = TEEType(d["tee_type"])
        if "tee_attestation" in d and isinstance(d["tee_attestation"], str):
            d["tee_attestation"] = bytes.fromhex(d["tee_attestation"])
        if "output_hash" in d and isinstance(d["output_hash"], str):
            d["output_hash"] = bytes.fromhex(d["output_hash"])
        if "settler_signature" in d and isinstance(d["settler_signature"], str):
            d["settler_signature"] = bytes.fromhex(d["settler_signature"])
        if "cost_ftns" in d and not isinstance(d["cost_ftns"], Decimal):
            d["cost_ftns"] = Decimal(str(d["cost_ftns"]))
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})

    def signing_payload(self) -> bytes:
        """Canonical bytes used for ``settler_signature`` generation/verification.

        Excludes ``settler_signature`` itself (would be circular). Field order
        is fixed; do not reorder without bumping a receipt-schema version.

        Implemented in this scaffold; consumed by Task 2 (Ed25519 signing).
        """
        parts = [
            self.job_id,
            self.request_id,
            self.model_id,
            self.content_tier.value,
            self.privacy_tier.value,
            f"{self.epsilon_spent:.10f}",
            self.tee_type.value,
            self.tee_attestation.hex(),
            self.output_hash.hex(),
            f"{self.duration_seconds:.6f}",
            str(self.cost_ftns),
            self.settler_node_id,
        ]
        return "\n".join(parts).encode("utf-8")


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceResult:
    """Output of one inference execution, including the verifiable receipt.

    On success: ``output`` is the model's response and ``receipt`` carries the
    TEE attestation + signed settlement evidence. On failure: ``error`` is a
    human-readable description, ``output`` is empty, and ``receipt`` is None
    (no FTNS settled).
    """

    request_id: str
    success: bool
    output: str = ""
    receipt: Optional[InferenceReceipt] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "output": self.output,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        d = dict(data)
        if d.get("receipt"):
            d["receipt"] = InferenceReceipt.from_dict(d["receipt"])
        else:
            d["receipt"] = None
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})

    @classmethod
    def failure(cls, request_id: str, error: str) -> "InferenceResult":
        """Convenience constructor for failed inference."""
        return cls(request_id=request_id, success=False, error=error)
