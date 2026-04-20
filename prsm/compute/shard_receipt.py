"""Phase 2: ShardExecutionReceipt + VerificationStrategy.

Signed receipt for remote shard execution. The provider signs a
canonical payload after executing; the requester verifies the
signature matches the provider's advertised pubkey AND the
output_hash matches the actual bytes returned.

Tier-A verification (receipt-only) is the Phase 2 default. Tiers B
(redundant execution) and C (stake + slash) plug in at Phase 7 via
the same VerificationStrategy protocol without changing the receipt
format or the dispatch protocol.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Protocol

logger = logging.getLogger(__name__)


def build_receipt_signing_payload(
    job_id: str,
    shard_index: int,
    output_hash: str,
    executed_at_unix: int,
) -> bytes:
    """Canonical bytes the provider signs. Requesters rebuild the same
    payload and verify the provider's signature over it.

    Format: "{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    encoded as UTF-8.
    """
    return (
        f"{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    ).encode("utf-8")


@dataclass(frozen=True)
class ShardExecutionReceipt:
    """Signed proof that a provider executed a shard.

    Fields form the serialized `receipt` sub-object in a
    shard_execute_response MSG_DIRECT payload.
    """
    job_id: str
    shard_index: int
    provider_id: str
    provider_pubkey_b64: str
    output_hash: str
    executed_at_unix: int
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardExecutionReceipt":
        return cls(
            job_id=data["job_id"],
            shard_index=data["shard_index"],
            provider_id=data["provider_id"],
            provider_pubkey_b64=data["provider_pubkey_b64"],
            output_hash=data["output_hash"],
            executed_at_unix=data["executed_at_unix"],
            signature=data["signature"],
        )


class VerificationStrategy(Protocol):
    """Pluggable interface for receipt verification. Phase 2 implements
    Tier A (receipt-only). Tiers B (redundant execution) and C (stake +
    slash) implement this protocol at Phase 7."""

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
    ) -> bool: ...


class ReceiptOnlyVerification:
    """Tier A: signature check only. Phase 2 default.

    Two checks:
      1. output_hash == sha256(output_bytes)  — the declared hash
         actually matches the bytes returned.
      2. Ed25519(signature) valid against provider_pubkey_b64 over
         build_receipt_signing_payload(...) — the receipt was signed
         by the claimed provider.

    Returns True iff both checks pass. Never raises on invalid input —
    logs a warning and returns False so dispatchers can uniformly
    refund escrow on verification failure.
    """

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
    ) -> bool:
        declared_hash = receipt.get("output_hash")
        actual_hash = hashlib.sha256(output_bytes).hexdigest()
        if declared_hash != actual_hash:
            logger.warning(
                f"receipt verification failed: output_hash mismatch "
                f"(declared={str(declared_hash)[:16]}…, actual={actual_hash[:16]}…)"
            )
            return False

        try:
            payload = build_receipt_signing_payload(
                job_id=receipt["job_id"],
                shard_index=receipt["shard_index"],
                output_hash=declared_hash,
                executed_at_unix=receipt["executed_at_unix"],
            )
            pubkey_b64 = receipt["provider_pubkey_b64"]
            signature = receipt["signature"]
        except KeyError as exc:
            logger.warning(f"receipt verification failed: missing field {exc}")
            return False

        try:
            from prsm.node.identity import verify_signature
            if not verify_signature(pubkey_b64, payload, signature):
                logger.warning(
                    f"receipt verification failed: signature invalid for "
                    f"provider {str(receipt.get('provider_id', '?'))[:12]}…"
                )
                return False
        except ImportError:
            logger.warning(
                "receipt verification failed: verify_signature unavailable"
            )
            return False
        except Exception as exc:
            logger.warning(f"receipt verification raised: {exc}")
            return False

        return True
