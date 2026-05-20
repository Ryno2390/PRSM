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


def verify_chain_invariants(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sprint 637 — chain-of-custody invariants over a receipts list.

    Each receipt's stage signature (verified by `verify_receipt_record`)
    proves the stage SIGNED OVER specific bytes, but says nothing about
    whether the sequence of receipts forms a coherent generation. An
    auditor could:
      - reorder receipts
      - insert duplicates (replay)
      - claim a receipt for a different model came from this one
      - swap next_token_id without invalidating the stage signature
        (since next_token_id is a CLIENT-COMPUTED derivation, not
        part of the signed payload for non-tail-stage responses)

    Sprint 637 checks the cross-receipt invariants offline. Returns
    a list of finding dicts (empty if every invariant holds), each
    with: ``kind`` (constant identifying the failure class),
    ``message`` (human-readable detail), and ``line_indices``
    (which receipts contributed to the finding).

    Invariants:
      C1: ``settler_node_id`` consistent across the run
      C2: ``model_id`` consistent across the run
      C3: All ``request_id``s unique (no replay)
      C4: ``wall_unix`` monotonic non-decreasing
      C5: ``next_token_id`` matches argmax of activation_blob (sprint
          635+ format only; pre-635 receipts skipped — same as
          UNVERIFIABLE in signature path)

    Caller decides what to do with the findings. The CLI surface
    treats non-empty findings as exit-code-1.
    """
    findings: List[Dict[str, Any]] = []
    if not records:
        return findings

    # C1: settler consistency
    settlers = {
        (rec.get("settler_node_id"), idx)
        for idx, rec in enumerate(records)
        if rec.get("settler_node_id")
    }
    unique_settlers = {s for s, _ in settlers}
    if len(unique_settlers) > 1:
        findings.append({
            "kind": "INCONSISTENT_SETTLER",
            "message": (
                f"receipts span {len(unique_settlers)} different "
                f"settler_node_id values: "
                f"{sorted(unique_settlers)}; an audit run must come "
                f"from a single requester"
            ),
            "line_indices": [idx for _, idx in settlers],
        })

    # C2: model consistency
    models = {
        (rec.get("model_id"), idx)
        for idx, rec in enumerate(records)
        if rec.get("model_id")
    }
    unique_models = {m for m, _ in models}
    if len(unique_models) > 1:
        findings.append({
            "kind": "INCONSISTENT_MODEL",
            "message": (
                f"receipts span {len(unique_models)} different "
                f"model_id values: {sorted(unique_models)}; "
                f"a generation run must stay on one model"
            ),
            "line_indices": [idx for _, idx in models],
        })

    # C3: request_id uniqueness (anti-replay)
    seen_request_ids: Dict[str, int] = {}
    duplicates: List[int] = []
    for idx, rec in enumerate(records):
        rid = rec.get("request_id")
        if not rid:
            continue
        if rid in seen_request_ids:
            duplicates.append(idx)
        else:
            seen_request_ids[rid] = idx
    if duplicates:
        findings.append({
            "kind": "DUPLICATE_REQUEST_ID",
            "message": (
                f"{len(duplicates)} receipt(s) reuse a request_id "
                f"already seen earlier in the file — replay or "
                f"audit-trail tampering"
            ),
            "line_indices": duplicates,
        })

    # C4: wall_unix monotonic
    prev_wall: Optional[float] = None
    out_of_order: List[int] = []
    for idx, rec in enumerate(records):
        wall = rec.get("wall_unix")
        if wall is None:
            continue
        if prev_wall is not None and wall < prev_wall:
            out_of_order.append(idx)
        prev_wall = wall
    if out_of_order:
        findings.append({
            "kind": "NON_MONOTONIC_WALL_UNIX",
            "message": (
                f"{len(out_of_order)} receipt(s) have wall_unix "
                f"smaller than a previous receipt's; receipts were "
                f"likely reordered post-generation"
            ),
            "line_indices": out_of_order,
        })

    # ── helpers for C5 ────────────────────────────────────────
    def _parse_sampling_mode(mode_str: str) -> Optional[Dict[str, float]]:
        """Sprint 640 — parse the canonical sampling_mode string
        ("temperature:0.700,top_k:40,seed:42") back into the params
        we need to re-derive a sample. Returns None for greedy /
        unparseable / seed-missing (each of those takes a different
        C5 path).
        """
        if not mode_str or mode_str == "greedy":
            return None
        parts = {}
        for token in mode_str.split(","):
            if ":" not in token:
                continue
            k, v = token.split(":", 1)
            try:
                parts[k.strip()] = float(v.strip())
            except ValueError:
                return None  # malformed mode string
        if "seed" not in parts or "temperature" not in parts:
            return None
        return parts

    def _replay_sample(
        logits_last: Any,
        params: Dict[str, float],
        step_idx: int,
    ) -> Optional[int]:
        """Sprint 641 — uses the shared sampling helper so replay is
        byte-identical with the CLI sampling code. Drift between
        the two would silently break audits; this delegation
        prevents that by construction.
        """
        try:
            from prsm.cli_modules.sampling import (
                sample_token_from_logits,
            )
        except ImportError:
            return None
        temp = float(params["temperature"])
        seed = int(params["seed"])
        top_k = int(params.get("top_k", 0)) if params.get("top_k") else None
        try:
            return sample_token_from_logits(
                logits_last,
                temperature=temp,
                top_k=top_k,
                seed=seed,
                step=step_idx,
            )
        except Exception:  # noqa: BLE001
            return None

    # C5: next_token_id matches argmax of activation_blob.
    # Receipt's "next_token_id" is the operator-recorded sample;
    # we recompute argmax from the activation bytes that the stage
    # signed over. Mismatch = operator-side tampering after sampling.
    #
    # Sprint 639: C5 only applies in greedy sampling mode. For
    # temperature/top-k runs the operator recorded a SAMPLE drawn
    # from the distribution, not the argmax — so a mismatch is
    # expected behavior, not tampering. Receipts that omit
    # `sampling_mode` (sprint 633-638 format) are treated as greedy
    # for backwards compatibility (the only sampling sprint 633-638
    # supported was greedy).
    try:
        import numpy as _np  # lazy — keeps tests lightweight
    except ImportError:
        return findings  # can't run C5 without numpy; skip silently
    mismatches: List[int] = []
    for idx, rec in enumerate(records):
        blob_b64 = rec.get("activation_blob_b64")
        if not blob_b64:
            continue  # pre-635 format; can't run C5
        claimed_id = rec.get("next_token_id")
        if claimed_id is None:
            continue
        sampling_mode = rec.get("sampling_mode", "greedy")
        # Decode the signed activation bytes (shared by both
        # greedy + seed-replay paths).
        try:
            blob = base64.b64decode(blob_b64)
            shape = tuple(rec["activation_shape"])
            dtype = rec["activation_dtype"]
            logits = _np.frombuffer(blob, dtype=dtype).reshape(shape)
            last_logits = logits[0, -1, :]
        except Exception:  # noqa: BLE001
            continue  # malformed numerics; C5 skipped for this row

        if sampling_mode == "greedy":
            recomputed_id = int(last_logits.argmax())
        else:
            # Sprint 640: try seed-replay. Receipts with a recorded
            # seed in sampling_mode can be deterministically
            # re-sampled to verify next_token_id matches.
            sampling_params = _parse_sampling_mode(sampling_mode)
            if sampling_params is None:
                # No seed (or unparseable) → can't replay; skip C5
                # for this row. Operator who wants strong audit
                # should always pass --seed for non-greedy runs.
                continue
            step_idx = int(rec.get("step", idx))
            replayed = _replay_sample(
                last_logits, sampling_params, step_idx,
            )
            if replayed is None:
                continue  # numpy unavailable etc.
            recomputed_id = replayed
        if recomputed_id != int(claimed_id):
            mismatches.append(idx)
    if mismatches:
        findings.append({
            "kind": "TOKEN_ID_ARGMAX_MISMATCH",
            "message": (
                f"{len(mismatches)} receipt(s) declare a "
                f"next_token_id that doesn't match argmax of the "
                f"signed activation bytes — operator tampered with "
                f"the sampled token after the stage signature was "
                f"observed"
            ),
            "line_indices": mismatches,
        })

    return findings


def verify_receipts_file(
    receipts_path: str,
    *,
    anchor: Any,
    check_chain: bool = False,
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
    raw_records: List[Dict[str, Any]] = []
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
        raw_records.append(rec)
        result = verify_receipt_record(rec, anchor=cached_anchor)
        result["line"] = line_idx
        results.append(result)

    if check_chain:
        # Sprint 637 — attach chain-level findings to the last result
        # entry under a `chain_findings` key. Caller renders them
        # specially. Empty list = chain OK.
        findings = verify_chain_invariants(raw_records)
        if results:
            results[-1]["chain_findings"] = findings
        else:
            # No receipts to attach to; surface a synthetic record so
            # callers can still consume the findings.
            results.append({
                "line": -1,
                "status": "CHAIN_ONLY",
                "stage_node_id": None,
                "request_id": None,
                "next_token_text": None,
                "chain_findings": findings,
            })
    return results
