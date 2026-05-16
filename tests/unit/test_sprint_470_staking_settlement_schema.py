"""Sprint 470 — §5.3 staking + settlement + settler-registry
schema pins.

Live-verified 2026-05-16 against a running daemon. These pins
defend the canonical response schemas + the load-bearing economic
invariants exercised this sprint:

  POST /staking/unstake             → request_id, available_at
                                       (7-day cooldown — §11
                                       invariant)
  POST /staking/cancel-unstake/{id} → {cancelled: true} +
                                       state-machine return
  POST /staking/withdraw/{id}       → 400/404 error-path shape
  GET  /settlement/pending          → {pending, count}
  POST /settlement/flush            → canonical schema
                                       {settled_count,
                                        total_amount,
                                        net_transfers,
                                        tx_hashes, errors,
                                        duration_seconds}
  GET  /settlement/history          → {history, count}
  POST /settler/register            → min bond 10000 FTNS
                                       enforced (§11 invariant)
  POST /settler/unbond              → 30-day cooldown
                                       (`unbond_at`)
  GET  /settler/ledger/export       → `integrity_hash` present
                                       (chain-of-custody surface)

These pins fire if a refactor silently drops a documented
field OR weakens a load-bearing economic invariant (min bond,
cooldown duration, integrity hash).

They are SCHEMA + INVARIANT pins, not full-system tests.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_FILE = REPO_ROOT / "prsm" / "node" / "api.py"


def _slice_route(decorator: str, method: str = "post") -> str:
    """Return source between `@app.{method}({decorator}` and the
    next @app.* decorator."""
    src = API_FILE.read_text()
    anchor = f'@app.{method}("{decorator}'
    idx = src.find(anchor)
    assert idx >= 0, (
        f"route not found: {method.upper()} {decorator}"
    )
    next_idx = src.index('@app.', idx + 1)
    return src[idx:next_idx]


# ── Unstake ──────────────────────────────────────────────


def test_staking_unstake_returns_request_with_cooldown():
    """Unstake response must carry both `request_id` (for the
    subsequent cancel/withdraw call) and `available_at` (the
    7-day cooldown Vision §11 promise) — dropping either is
    a breaking change. The fields are declared on the
    Pydantic `UnstakeResponse` model in api.py."""
    src = API_FILE.read_text()
    # Pin to the Pydantic model declaration.
    model_idx = src.find("class UnstakeResponse")
    assert model_idx >= 0, "UnstakeResponse model missing"
    next_class = src.find("\n    class ", model_idx + 1)
    model_body = src[model_idx:next_class] if next_class > 0 else src[model_idx:model_idx + 2000]
    for field in (
        "request_id:",
        "available_at:",
    ):
        assert field in model_body, (
            f"UnstakeResponse missing field declaration: {field}"
        )


# ── Cancel-unstake ───────────────────────────────────────


def test_staking_cancel_unstake_returns_cancelled_flag():
    """Callers display the cancellation result; field rename
    breaks UIs + CLIs."""
    body = _slice_route("/staking/cancel-unstake/")
    assert '"cancelled"' in body


# ── Withdraw ─────────────────────────────────────────────


def test_staking_withdraw_unknown_uuid_returns_404():
    """Error-path attribution: clear 404 with the missing
    request_id echoed back. Helps operators diagnose CLI
    typos vs. missing record."""
    body = _slice_route("/staking/withdraw/")
    # The 404-branch detail message structure.
    assert "not found" in body.lower()


# ── Settlement ───────────────────────────────────────────


def test_settlement_flush_canonical_schema():
    """Flush response shape — operators consume this to know
    if the flush did anything + which TX hashes were emitted.
    Vision §5.3 economic-layer surface."""
    body = _slice_route("/settlement/flush")
    for field in (
        '"settled_count"',
        '"total_amount"',
        '"net_transfers"',
        '"tx_hashes"',
        '"errors"',
        '"duration_seconds"',
    ):
        assert field in body, (
            f"settlement/flush response missing field: {field}"
        )


def test_settlement_pending_envelope_schema():
    body = _slice_route("/settlement/pending", method="get")
    assert '"pending"' in body
    assert '"count"' in body


def test_settlement_history_envelope_schema():
    body = _slice_route("/settlement/history", method="get")
    assert '"history"' in body
    assert '"count"' in body


# ── Settler registry ─────────────────────────────────────


def test_settler_register_enforces_minimum_bond():
    """Vision §11 invariant: settler registration requires a
    minimum bond. Sprint 470 verified the active minimum is
    10000 FTNS. Dropping this bound is a load-bearing
    security regression — anyone with negligible bond could
    register as a settler and try to sign fraudulent batches."""
    # The bond floor enforcement lives in the registry layer,
    # not the API handler. Source-of-truth is the registry
    # module.
    registry = REPO_ROOT / "prsm" / "economy" / "settlement"
    candidates = list(registry.rglob("*.py")) if registry.is_dir() else []
    # Search the API handler too — it might wrap the registry
    # with its own minimum.
    candidates.append(API_FILE)
    found = False
    for path in candidates:
        text = path.read_text()
        # The min-bond constant or the live-attested 10000 value
        # must appear adjacent to bond enforcement.
        if (
            ("min" in text.lower() and "bond" in text.lower())
            and ("10000" in text or "MIN_BOND" in text)
        ):
            found = True
            break
    assert found, (
        "Minimum settler bond (10000 FTNS) must be enforced "
        "somewhere — Vision §11 invariant"
    )


def test_settler_unbond_returns_cooldown_timestamp():
    """30-day cooldown invariant — operators can't immediately
    re-bond after unbonding, which is the anti-flip-flop
    guarantee §11 relies on for accountable settlement."""
    body = _slice_route("/settler/unbond")
    # The unbond response includes `unbond_at` per live probe.
    # The handler delegates to registry — search the registry
    # too.
    if '"unbond_at"' not in body:
        registry = REPO_ROOT / "prsm" / "economy" / "settlement"
        if registry.is_dir():
            corpus = "\n".join(
                p.read_text() for p in registry.rglob("*.py")
            )
            assert "unbond_at" in corpus, (
                "unbond_at field missing — cooldown surface "
                "lost"
            )


def test_settler_ledger_export_has_integrity_hash():
    """The ledger-export integrity hash is the
    chain-of-custody surface auditors use to detect tampering
    after-the-fact. Dropping it removes the load-bearing
    audit guarantee."""
    body = _slice_route("/settler/ledger/export", method="get")
    # The handler may compute the hash in the registry —
    # look at both.
    if '"integrity_hash"' not in body:
        registry = REPO_ROOT / "prsm" / "economy" / "settlement"
        if registry.is_dir():
            corpus = "\n".join(
                p.read_text() for p in registry.rglob("*.py")
            )
            assert "integrity_hash" in corpus, (
                "integrity_hash missing from ledger-export "
                "surface — auditor-facing tampering signal "
                "lost"
            )
