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
from typing import Any, Dict, List, Optional

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


def allocate_royalty_amounts(
    *,
    shards: List[str],
    content_index: Any,
    total_pool_wei: int,
    mode: str = "uniform",
) -> Dict[str, int]:
    """Sprint 256 — allocate ``total_pool_wei`` across shards.

    Modes
    -----
    ``uniform``
        Each shard receives an equal share. Remainder wei (from
        integer division) is absorbed by the last shard so the
        sum equals ``total_pool_wei`` exactly.

    ``rate_weighted``
        Each shard's share is proportional to its
        ``ContentRecord.royalty_rate`` (looked up via
        ``content_index``). Missing records / lookup exceptions
        contribute zero weight. If ALL weights are zero, falls
        back to uniform allocation. Remainder absorbed by the
        last positive-weight shard.

    Returns
    -------
    ``{cid: wei_int}`` mapping. Sum equals ``total_pool_wei``
    unless ``shards`` is empty.

    Raises
    ------
    ValueError
        ``total_pool_wei <= 0`` or unknown ``mode``.
    """
    if total_pool_wei <= 0:
        raise ValueError(
            f"total_pool_wei must be > 0; got {total_pool_wei}"
        )
    if mode not in ("uniform", "rate_weighted"):
        raise ValueError(
            f"mode must be 'uniform' or 'rate_weighted'; "
            f"got {mode!r}"
        )
    if not shards:
        return {}

    def _uniform() -> Dict[str, int]:
        n = len(shards)
        per = total_pool_wei // n
        out: Dict[str, int] = {c: per for c in shards}
        # Last absorbs remainder.
        out[shards[-1]] += total_pool_wei - per * n
        return out

    if mode == "uniform":
        return _uniform()

    # rate_weighted
    weights: Dict[str, float] = {}
    for cid in shards:
        try:
            record = content_index.lookup(cid)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "allocate_royalty_amounts: lookup raised for "
                "%s (%s); weight=0", cid[:12], exc,
            )
            weights[cid] = 0.0
            continue
        if record is None:
            weights[cid] = 0.0
            continue
        rate = float(getattr(record, "royalty_rate", 0.0) or 0.0)
        weights[cid] = max(rate, 0.0)

    total_w = sum(weights.values())
    if total_w <= 0:
        # Degenerate — fall back to uniform.
        return _uniform()

    out = {
        c: int(total_pool_wei * (weights[c] / total_w))
        for c in shards
    }
    # Remainder: assign to the shard with the highest weight so
    # the sum equals total_pool_wei exactly.
    delta = total_pool_wei - sum(out.values())
    if delta != 0:
        # Pick the highest-weight cid as the absorber. Stable
        # tie-break: first occurrence wins.
        absorber = max(shards, key=lambda c: weights[c])
        out[absorber] += delta
    return out


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
