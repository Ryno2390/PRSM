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
from typing import Any, Callable, Dict, List, Optional

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)

logger = logging.getLogger(__name__)


def royalty_dispatch_key(settlement_key: str, cid: str) -> str:
    """sp911 — the deterministic idempotency key for one shard's royalty
    dispatch within a settlement event. Callers atomically claim this key
    (e.g. ledger.record_nonce) BEFORE dispatching so a retry of the same
    settlement does not re-pay on chain. `settlement_key` must be STABLE
    across logical retries (a content/query hash + epoch), NOT a random
    per-request job id."""
    return f"royalty_dispatch:{settlement_key}:{cid}"


@dataclass(frozen=True)
class DispatchResult:
    """One per shard. ``status`` is one of:

      - ``sent``: distribute_royalty returned successfully
      - ``pending``: broadcast succeeded but the receipt is unconfirmed
        (OnChainPendingError). ``tx_hash`` is set. The caller MUST NOT
        re-dispatch — the tx may still settle; reconcile via tx_hash.
      - ``reverted``: the tx confirmed and reverted on chain
        (OnChainRevertedError) — safe to retry / fall back (no state change)
      - ``skipped_already_dispatched``: this (settlement_key, cid) was
        already claimed — idempotent retry short-circuit (no double-pay)
      - ``skipped_no_record``: ContentIndex.lookup(cid) was None
      - ``skipped_bad_hash``: content_hash missing / wrong
        length / non-hex
      - ``skipped_zero_amount``: per-shard amount was 0
      - ``failed``: broadcast never reached the network
        (BroadcastFailedError) or an unexpected error; ``error`` carries
        the text. Safe to retry (the chain saw nothing).
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
    gross_per_shard_wei: Optional[int] = None,
    gross_amounts_wei: Optional[Dict[str, int]] = None,
    settlement_key: Optional[str] = None,
    claim_fn: Optional[Callable[[str], bool]] = None,
) -> List[DispatchResult]:
    """Send one ``distribute_royalty`` tx per shard.

    Caller MUST supply exactly one of:
      - ``gross_per_shard_wei``: uniform amount applied to every
        shard (legacy sprint-248 signature, still supported).
      - ``gross_amounts_wei``: per-shard ``{cid: wei}`` dict
        produced by :func:`allocate_royalty_amounts`. Allows
        rate-weighted or other non-uniform policies.

    Idempotency (sp911)
    -------------------
    When BOTH ``settlement_key`` (a stable per-settlement-event id — NOT a
    random job id) and ``claim_fn`` are supplied, each shard is gated on
    ``claim_fn(royalty_dispatch_key(settlement_key, cid))``. ``claim_fn``
    must ATOMICALLY claim the key and return True iff THIS call won it
    (the sp898 ledger.record_nonce primitive); a shard whose key was
    already claimed is short-circuited as ``skipped_already_dispatched`` —
    so a retry of the same settlement does NOT re-pay on chain. The async
    caller pre-claims via ``await ledger.record_nonce(...)`` and passes a
    sync lookup of the per-shard result. When omitted, no dedup is applied
    (back-compat).

    Raises
    ------
    ValueError
        Neither / both amount kwargs supplied, or any amount <= 0.
    """
    if (
        (gross_per_shard_wei is None) ==
        (gross_amounts_wei is None)
    ):
        raise ValueError(
            "must supply exactly one of gross_per_shard_wei OR "
            "gross_amounts_wei"
        )
    if gross_per_shard_wei is not None and gross_per_shard_wei <= 0:
        raise ValueError(
            f"gross_per_shard_wei must be > 0; "
            f"got {gross_per_shard_wei}"
        )
    if gross_amounts_wei is not None:
        for cid, amt in gross_amounts_wei.items():
            if amt < 0:
                raise ValueError(
                    f"gross_amounts_wei[{cid!r}] must be >= 0; "
                    f"got {amt}"
                )

    results: List[DispatchResult] = []
    for cid in shards:
        # Resolve the per-shard amount for the tx + audit ring.
        if gross_per_shard_wei is not None:
            amount_for_cid = gross_per_shard_wei
        else:
            amount_for_cid = gross_amounts_wei.get(cid, 0)
        # Zero amount short-circuits — don't waste an on-chain tx
        # paying nothing. Surface as a distinct status so audit
        # ring can show why.
        if amount_for_cid == 0:
            results.append(DispatchResult(
                cid=cid,
                status="skipped_zero_amount",
            ))
            continue
        # sp911 — idempotency: atomically claim this (settlement, shard)
        # BEFORE the on-chain tx. A retry of the same settlement finds the
        # key already claimed and short-circuits, so no double-pay.
        if settlement_key is not None and claim_fn is not None:
            _key = royalty_dispatch_key(settlement_key, cid)
            try:
                won = claim_fn(_key)
            except Exception as exc:  # noqa: BLE001
                # Fail-CLOSED on a claim error: do NOT dispatch if we can't
                # confirm we won the claim (avoids a double-pay on a flaky
                # claim store). Surface for operator retry.
                logger.warning(
                    "royalty dispatch claim raised for cid=%s: %s",
                    cid[:12], exc,
                )
                results.append(DispatchResult(
                    cid=cid, status="failed",
                    error=f"claim error: {exc}",
                ))
                continue
            if not won:
                results.append(DispatchResult(
                    cid=cid,
                    status="skipped_already_dispatched",
                ))
                continue
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
                amount_for_cid,
            )
        except OnChainPendingError as exc:
            # sp911 — broadcast SUCCEEDED but the receipt is unconfirmed.
            # The tx may still settle, so do NOT present this as a plain
            # 'failed' that invites the operator to re-dispatch (→ double
            # pay). Surface a distinct 'pending' status carrying the
            # tx_hash so reconciliation polls the receipt instead.
            logger.warning(
                "distribute_royalty PENDING for cid=%s (tx=%s); do not "
                "re-dispatch, reconcile via tx_hash",
                cid[:12], getattr(exc, "tx_hash", None),
            )
            results.append(DispatchResult(
                cid=cid,
                status="pending",
                tx_hash=getattr(exc, "tx_hash", None),
                error=str(exc),
            ))
            continue
        except OnChainRevertedError as exc:
            # Confirmed + reverted: the chain rolled it back atomically, so
            # no payment occurred — safe to retry / fall back.
            logger.warning(
                "distribute_royalty REVERTED for cid=%s: %s", cid[:12], exc,
            )
            results.append(DispatchResult(
                cid=cid, status="reverted", error=str(exc),
            ))
            continue
        except (BroadcastFailedError, Exception) as exc:  # noqa: BLE001
            # BroadcastFailedError: never reached the network — safe retry.
            # Any other unexpected error is treated the same (chain saw
            # nothing it could settle from this path).
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
