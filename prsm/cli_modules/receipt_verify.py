"""Sprint 636 — verify-receipts core, factored out of cli.py.

The Click command in `prsm node verify-receipts` is a thin shell
around `verify_receipt_record` + `verify_receipts_file`. Splitting
them means:

  1. Unit tests can exercise the verification logic directly without
     subprocess + Click runner gymnastics.
  2. The cryptographic-chain integrity properties (tamper detection,
     anti-substitution invariant, anchor-resolution failure modes)
     are pinned at the function level — refactors that subtly break
     verification get caught by CI, not by an audit-time surprise.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional


def verify_receipt_record(
    rec: Dict[str, Any],
    *,
    anchor: Any,
    pubkey_cache: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Verify one receipt record (one JSON line from --save-receipts).

    Returns a result dict with these fields always present:
      - status: one of OK / SIGNATURE_INVALID / PUBKEY_NOT_REGISTERED
        / ANCHOR_LOOKUP_FAILED / RECONSTRUCT_FAILED / UNVERIFIABLE
      - stage_node_id: from receipt (may be None on early-rejected lines)
      - request_id: from receipt (may be None on early-rejected lines)
      - next_token_text: from receipt (may be None)
      - reason: human-readable detail (present for non-OK statuses)

    ``anchor`` MUST expose a ``lookup(node_id) -> str | None`` method
    matching PublisherKeyAnchorClient. Tests pass a duck-typed mock.

    ``pubkey_cache`` is an optional caller-owned dict for memoizing
    pubkey lookups across multiple records (avoid repeated on-chain
    reads for the same stage_node_id). Caller is responsible for
    lifecycle; pass {} for single-record use.
    """
    from prsm.compute.chain_rpc.protocol import (
        RunLayerSliceResponse,
    )
    from prsm.compute.tee.models import TEEType

    base_result: Dict[str, Any] = {
        "stage_node_id": rec.get("stage_node_id"),
        "request_id": rec.get("request_id"),
        "next_token_text": rec.get("next_token_text"),
    }

    if "activation_blob_b64" not in rec:
        return {
            **base_result,
            "status": "UNVERIFIABLE",
            "reason": (
                "receipt lacks activation_blob_b64 (pre-sprint-635 "
                "format); cannot reconstruct signing payload"
            ),
        }

    # Reconstruct RunLayerSliceResponse from receipt fields. Any
    # missing/malformed field surfaces as RECONSTRUCT_FAILED rather
    # than a confusing downstream verification error.
    try:
        activation_bytes = base64.b64decode(rec["activation_blob_b64"])
        tee_attest = base64.b64decode(
            rec.get("tee_attestation_b64", ""),
        )
        resp = RunLayerSliceResponse(
            request_id=rec["request_id"],
            activation_blob=activation_bytes,
            activation_shape=tuple(rec["activation_shape"]),
            activation_dtype=rec["activation_dtype"],
            duration_seconds=float(rec["duration_seconds"]),
            tee_attestation=tee_attest,
            tee_type=TEEType(rec["tee_type"]),
            epsilon_spent=float(rec["epsilon_spent"]),
            stage_signature_b64=rec["stage_signature_b64"],
            stage_node_id=rec["stage_node_id"],
            protocol_version=int(rec.get("protocol_version", 2)),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            **base_result,
            "status": "RECONSTRUCT_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    cache = pubkey_cache if pubkey_cache is not None else {}
    if resp.stage_node_id not in cache:
        try:
            cache[resp.stage_node_id] = anchor.lookup(resp.stage_node_id)
        except Exception as exc:  # noqa: BLE001
            return {
                **base_result,
                "status": "ANCHOR_LOOKUP_FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    if not cache[resp.stage_node_id]:
        return {
            **base_result,
            "status": "PUBKEY_NOT_REGISTERED",
            "reason": (
                f"stage_node_id {resp.stage_node_id} has no pubkey "
                f"in the live PublisherKeyAnchor"
            ),
        }
    ok = resp.verify_with_anchor(
        anchor, expected_stage_node_id=resp.stage_node_id,
    )
    return {
        **base_result,
        "status": "OK" if ok else "SIGNATURE_INVALID",
        **({} if ok else {
            "reason": "Ed25519 verification failed against anchor pubkey",
        }),
    }


class _CachingAnchor:
    """Proxy that memoizes pubkey lookups across receipts in a single
    verify_receipts_file run. Wraps the real anchor so every code path
    that does ``anchor.lookup(node_id)`` — including the inner
    ``RunLayerSliceResponse.verify_with_anchor`` — benefits from the
    cache, not just the outer reconstruct branch.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: Dict[str, Optional[str]] = {}

    def lookup(self, node_id: str) -> Optional[str]:
        if node_id not in self._cache:
            self._cache[node_id] = self._inner.lookup(node_id)
        return self._cache[node_id]


def verify_receipts_file(
    receipts_path: str,
    *,
    anchor: Any,
) -> List[Dict[str, Any]]:
    """Run `verify_receipt_record` over every line in `receipts_path`.

    Returns the per-line results list. Empty/blank lines are
    skipped (not counted in totals). Malformed JSON lines surface
    as MALFORMED_JSON status records.

    Anchor lookups are memoized for the lifetime of this call so a
    10-token receipts file from a single stage makes ONE on-chain
    read, not 20 (one per outer reconstruct + one per inner
    verify_with_anchor).
    """
    from pathlib import Path as _Path

    cached_anchor = _CachingAnchor(anchor)
    results: List[Dict[str, Any]] = []
    for line_idx, raw in enumerate(_Path(receipts_path).read_text().splitlines()):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            results.append({
                "line": line_idx,
                "status": "MALFORMED_JSON",
                "reason": f"{exc}",
                "stage_node_id": None,
                "request_id": None,
                "next_token_text": None,
            })
            continue
        result = verify_receipt_record(rec, anchor=cached_anchor)
        result["line"] = line_idx
        results.append(result)
    return results
