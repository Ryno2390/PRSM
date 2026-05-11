"""Sprint 247 — on-chain content-access royalty dispatcher.

Settlement-side helper that ships the per-shard access fee to
the on-chain RoyaltyDistributor via the existing
``distribute_royalty(content_hash, serving_node, gross)`` call.

Design notes
------------

The on-chain contract resolves the *creator* address itself
(via the ProvenanceRegistry lookup keyed on content_hash) — we
only need to supply:

  - ``content_hash``: 32-byte content hash from ContentRecord
  - ``serving_node``: 0x-prefixed eth address of the node that
    fetched + served the shard (typically the operator's own
    address)
  - ``gross``: FTNS amount in wei to distribute

The dispatcher iterates the supplied ``shards`` list, resolves
each shard's content_hash via the ContentIndex, validates the
hash is a well-formed 64-hex-char string, and dispatches one
``distribute_royalty`` tx per shard. Each tx fail-soft so one
bad shard doesn't crash the batch — operator sees structured
per-shard ``DispatchResult`` records for telemetry / retries.

This module does NOT decide whether to dispatch — that's an
opt-in operator decision wired via env var
``PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED`` at the settlement
caller layer (sprint TBD). Pure helper, no global state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchResult:
    """One per shard. ``status`` is one of:

      - ``sent``: distribute_royalty returned successfully
      - ``skipped_no_record``: ContentIndex.lookup(cid) was None
      - ``skipped_bad_hash``: content_hash missing / wrong
        length / non-hex
      - ``failed``: distribute_royalty raised; ``error`` carries
        the exception text
    """

    cid: str
    status: str
    tx_hash: Optional[str] = None
    error: Optional[str] = None


def _decode_content_hash(hex_str: str) -> Optional[bytes]:
    """Return 32-byte payload or None if malformed."""
    if not isinstance(hex_str, str):
        return None
    if hex_str.startswith("0x") or hex_str.startswith("0X"):
        hex_str = hex_str[2:]
    if len(hex_str) != 64:
        return None
    try:
        return bytes.fromhex(hex_str)
    except ValueError:
        return None


def dispatch_content_access_royalties(
    *,
    shards: List[str],
    content_index: Any,
    royalty_client: Any,
    serving_node_address: str,
    gross_per_shard_wei: int,
) -> List[DispatchResult]:
    """Send one ``distribute_royalty`` tx per shard.

    Raises
    ------
    ValueError
        ``gross_per_shard_wei <= 0``.
    """
    if gross_per_shard_wei <= 0:
        raise ValueError(
            f"gross_per_shard_wei must be > 0; "
            f"got {gross_per_shard_wei}"
        )

    results: List[DispatchResult] = []
    for cid in shards:
        try:
            record = content_index.lookup(cid)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "onchain_content_royalty: lookup raised for %s "
                "(%s); skipping", cid[:12], exc,
            )
            results.append(DispatchResult(
                cid=cid,
                status="skipped_no_record",
            ))
            continue
        if record is None:
            results.append(DispatchResult(
                cid=cid,
                status="skipped_no_record",
            ))
            continue
        hash_bytes = _decode_content_hash(
            getattr(record, "content_hash", ""),
        )
        if hash_bytes is None:
            results.append(DispatchResult(
                cid=cid,
                status="skipped_bad_hash",
            ))
            continue
        try:
            tx_hash, _status = royalty_client.distribute_royalty(
                hash_bytes,
                serving_node_address,
                gross_per_shard_wei,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "distribute_royalty failed for cid=%s: %s",
                cid[:12], exc,
            )
            results.append(DispatchResult(
                cid=cid,
                status="failed",
                error=str(exc),
            ))
            continue
        results.append(DispatchResult(
            cid=cid,
            status="sent",
            tx_hash=tx_hash,
        ))
    return results
