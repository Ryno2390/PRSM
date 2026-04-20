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

import base64
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol

from eth_utils import keccak

logger = logging.getLogger(__name__)


def build_receipt_signing_payload(
    job_id: str,
    shard_index: int,
    output_hash: str,
    executed_at_unix: int,
) -> bytes:
    """Canonical bytes the provider signs. Requesters rebuild the same
    payload and verify the provider's signature over it.

    Format per design §136: keccak256(
        "{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    ). Matches the on-chain-compatible hashing scheme for Phase 7 slashing.
    """
    raw = (
        f"{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    ).encode("utf-8")
    return keccak(raw)


def _derive_node_id_from_pubkey_b64(pubkey_b64: str) -> Optional[str]:
    """Recompute NodeIdentity.node_id from an advertised pubkey.

    Must match generate_node_identity(): hex(sha256(public_key_bytes))[:32].
    Returns None on decode failure so callers can treat it as verification
    failure rather than raise.
    """
    try:
        pub_bytes = base64.b64decode(pubkey_b64)
    except Exception:
        return None
    return hashlib.sha256(pub_bytes).hexdigest()[:32]


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
        expected_provider_id: Optional[str] = None,
    ) -> bool: ...


class ReceiptOnlyVerification:
    """Tier A: signature check only. Phase 2 default.

    Four checks (all must pass):
      1. output_hash == sha256(output_bytes) — the declared hash
         actually matches the bytes returned.
      2. node_id binding: receipt['provider_id'] ==
         hex(sha256(b64decode(provider_pubkey_b64)))[:32]. Closes the
         attack where a receipt claims 'provider A' but carries B's
         pubkey + signature — without this check, verification would
         be self-authenticating against whatever pubkey the receipt
         carries.
      3. expected match (optional): if expected_provider_id is supplied
         by the caller, it must equal receipt['provider_id']. Binds the
         receipt to the node the dispatcher actually sent the request to.
      4. Ed25519(signature) valid against provider_pubkey_b64 over
         build_receipt_signing_payload(...) — the receipt was signed
         by the claimed provider.

    Returns True iff all checks pass. Never raises on invalid input —
    logs a warning and returns False so dispatchers can uniformly
    refund escrow on verification failure.
    """

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
        expected_provider_id: Optional[str] = None,
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
            provider_id = receipt["provider_id"]
            pubkey_b64 = receipt["provider_pubkey_b64"]
            signature = receipt["signature"]
            payload = build_receipt_signing_payload(
                job_id=receipt["job_id"],
                shard_index=receipt["shard_index"],
                output_hash=declared_hash,
                executed_at_unix=receipt["executed_at_unix"],
            )
        except KeyError as exc:
            logger.warning(f"receipt verification failed: missing field {exc}")
            return False

        derived_node_id = _derive_node_id_from_pubkey_b64(pubkey_b64)
        if derived_node_id is None or derived_node_id != provider_id:
            logger.warning(
                f"receipt verification failed: provider_id "
                f"{str(provider_id)[:12]}… does not match pubkey-derived "
                f"node_id {str(derived_node_id)[:12]}…"
            )
            return False

        if expected_provider_id is not None and expected_provider_id != provider_id:
            logger.warning(
                f"receipt verification failed: expected provider "
                f"{expected_provider_id[:12]}… but receipt claims "
                f"{str(provider_id)[:12]}…"
            )
            return False

        try:
            from prsm.node.identity import verify_signature
            if not verify_signature(pubkey_b64, payload, signature):
                logger.warning(
                    f"receipt verification failed: signature invalid for "
                    f"provider {str(provider_id)[:12]}…"
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
