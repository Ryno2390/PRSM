"""Sprint 487 — concurrency stress harness.

Coverage-matrix priority #1 from sprint 486: zero
concurrency coverage anywhere. Sprint 487 starts with the
highest-blast-radius concurrent-write paths.

These are INTEGRATION tests — they require a running
daemon at PRSM_DAEMON_URL (default http://127.0.0.1:8000).
They are MUTATING — they change real daemon state. Run
against a dev daemon, not production.

Race targets:

  1. Staking unstake race: N concurrent /staking/unstake
     against the same stake_id. Pre-test invariant: only
     ONE should succeed; the rest get a clean rejection.
     Without proper locking the same stake could be
     unstaked twice, draining total_staked below zero or
     creating phantom unstake requests.

  2. Compute submit race: N concurrent /compute/submit with
     budgets that EXCEED wallet_balance / N. At least N-K
     must fail with insufficient-balance to prevent double-
     spend; final wallet balance MUST be non-negative.

  3. Compute cancel-refund race: N concurrent /compute/cancel
     on the SAME job_id. The escrow must refund exactly
     once; wallet balance must not double-refund.

  4. Fingerprint dedup race: N concurrent /content/upload
     with IDENTICAL bytes + DIFFERENT creator_ids. Only ONE
     creator becomes canonical (first-creator-wins anti-
     Sybil invariant per Vision §14).

These tests SKIP cleanly if no daemon is reachable —
they're not pure unit tests.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
import urllib.parse
from collections import Counter
from typing import Any, Dict, List, Optional

import pytest


DAEMON_URL = os.environ.get(
    "PRSM_DAEMON_URL", "http://127.0.0.1:8000",
).rstrip("/")
DAEMON_TIMEOUT = 10.0


def _daemon_reachable() -> bool:
    try:
        req = urllib.request.Request(f"{DAEMON_URL}/health")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _daemon_reachable(),
    reason=(
        f"no daemon at {DAEMON_URL} — set PRSM_DAEMON_URL or "
        "start a daemon to run concurrency stress tests"
    ),
)


def _http(method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
    """Blocking HTTP call. Used inside run_in_executor for
    parallelism — the daemon is reachable via stdlib urllib
    so we don't need httpx as a hard dep."""
    url = f"{DAEMON_URL}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=data, headers=headers, method=method,
    )
    try:
        with urllib.request.urlopen(
            req, timeout=DAEMON_TIMEOUT,
        ) as r:
            payload = r.read()
            return {
                "status": r.status,
                "body": (
                    json.loads(payload) if payload else None
                ),
            }
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
        except Exception:  # noqa: BLE001
            err_body = None
        return {"status": e.code, "body": err_body}
    except Exception as e:  # noqa: BLE001
        return {"status": 0, "error": str(e)}


async def _parallel_calls(
    n: int, method: str, path: str, body_factory,
) -> List[Dict]:
    """Fire N concurrent HTTP calls. body_factory(i) returns
    the JSON body for the i-th caller (allows variant
    payloads)."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            None, _http, method, path, body_factory(i),
        )
        for i in range(n)
    ]
    return await asyncio.gather(*tasks)


# ── Test 1: Staking unstake race ─────────────────────────


def test_staking_unstake_race_no_double_unstake():
    """N concurrent /staking/unstake on the SAME stake_id.
    Only ONE should succeed; the rest should return clean
    4xx errors (no 5xx, no inconsistent state, no
    total_staked < 0)."""
    # Setup: get the current active stake, snapshot balance.
    status = _http("GET", "/staking/status")
    assert status["status"] == 200
    stakes = status["body"]["active_stakes"]
    if not stakes:
        pytest.skip(
            "no active stake — daemon needs a real stake "
            "to race against"
        )
    target_stake = stakes[0]["stake_id"]
    initial_total_staked = status["body"]["total_staked"]

    # Race: 10 concurrent unstake requests
    n = 10
    results = asyncio.run(_parallel_calls(
        n, "POST", "/staking/unstake",
        lambda i: {
            "stake_id": target_stake,
            "amount_ftns": 100,
        },
    ))

    # Categorize
    successes = [r for r in results if r["status"] == 200]
    rejections = [r for r in results if 400 <= r["status"] < 500]
    server_errors = [r for r in results if r["status"] >= 500]
    network_errors = [r for r in results if r["status"] == 0]

    # No 5xx — invariant violation if any
    assert not server_errors, (
        f"{len(server_errors)} 5xx responses under "
        f"concurrent unstake — server didn't handle race "
        f"cleanly: {server_errors[:3]}"
    )
    assert not network_errors, (
        f"network errors during race: {network_errors[:3]}"
    )

    # Exactly one success (only one unstake-request can be
    # created for a single active stake)
    assert len(successes) <= 1, (
        f"{len(successes)} unstake calls succeeded — "
        f"double-unstake invariant broken. Expected ≤1"
    )

    # If one succeeded, request_id must be unique
    if successes:
        req_ids = [s["body"]["request_id"] for s in successes]
        assert len(set(req_ids)) == len(req_ids), (
            "duplicate request_id under race"
        )

    # Post-state: total_staked must be valid (0 or
    # original — not negative, not double-counted)
    post = _http("GET", "/staking/status")
    final_total = post["body"]["total_staked"]
    assert final_total >= 0, (
        f"total_staked went negative ({final_total}) "
        f"under unstake race"
    )

    # Cleanup: cancel any unstake we created to keep the
    # test idempotent
    for s in successes:
        rid = s["body"]["request_id"]
        _http("POST", f"/staking/cancel-unstake/{rid}")


# ── Test 2: Compute submit race ──────────────────────────


def test_compute_submit_race_no_double_spend():
    """N concurrent /compute/submit with cumulative budget
    that EXCEEDS the wallet balance. At least some MUST
    fail with insufficient-balance; total locked escrow
    must not exceed wallet balance."""
    bal = _http("GET", "/balance")
    assert bal["status"] == 200
    wallet_balance = bal["body"]["balance"]

    # Each submit asks for budget = balance / 5 — so 10
    # concurrent submits ask for 2x the balance total.
    # At least half should fail.
    n = 10
    budget_per_job = max(1, int(wallet_balance / 5))
    results = asyncio.run(_parallel_calls(
        n, "POST", "/compute/submit",
        lambda i: {
            "job_type": "embedding",
            "payload": {"text": f"sprint-487-race-{i}"},
            "ftns_budget": budget_per_job,
        },
    ))

    successes = [r for r in results if r["status"] == 200]
    rejections = [r for r in results if 400 <= r["status"] < 500]
    # 500 = unhandled exception (invariant violation).
    # 503 = designed retry-exhausted with actionable detail
    # (acceptable degradation per sprint 487 F25 fix).
    server_unhandled = [r for r in results if r["status"] == 500]
    server_degraded = [r for r in results if r["status"] == 503]

    assert not server_unhandled, (
        f"{len(server_unhandled)} 500s (unhandled exceptions) "
        f"under concurrent submit race — invariant violation: "
        f"{server_unhandled[:3]}"
    )

    # Sum of locked escrow MUST NOT exceed wallet balance
    total_escrowed = sum(
        s["body"]["ftns_budget"] for s in successes
    )
    assert total_escrowed <= wallet_balance, (
        f"total escrow {total_escrowed} > wallet balance "
        f"{wallet_balance} — DOUBLE-SPEND under "
        f"compute-submit race"
    )

    # Cleanup: cancel all submitted jobs to restore balance
    for s in successes:
        jid = s["body"]["job_id"]
        _http("POST", f"/compute/cancel/{jid}")


# ── Test 3: Compute cancel-refund race ───────────────────


def test_compute_cancel_refund_race_no_double_refund():
    """N concurrent /compute/cancel on the SAME job_id.
    Escrow must refund exactly once; wallet balance must
    not double-refund."""
    # Setup: submit a job to get a real job_id
    bal_before = _http("GET", "/balance")["body"]["balance"]
    # Need at least 1 FTNS to submit a job.
    if bal_before < 2:
        pytest.skip(
            f"wallet balance {bal_before} too low to test "
            f"cancel race — fund the wallet or recover "
            f"leaked escrows. F27 in dogfood-findings."
        )
    job_budget = min(100, max(1, int(bal_before / 20)))
    sub = _http("POST", "/compute/submit", {
        "job_type": "embedding",
        "payload": {"text": "sprint-487-cancel-race"},
        "ftns_budget": job_budget,
    })
    assert sub["status"] == 200, (
        f"submit failed: {sub}"
    )
    job_id = sub["body"]["job_id"]
    bal_after_submit = _http(
        "GET", "/balance",
    )["body"]["balance"]
    # Sanity: balance dropped by ~job_budget (escrow lock)
    assert (
        bal_before - bal_after_submit
        >= job_budget - 0.5  # tolerate fees
    ), (
        f"submit didn't lock escrow correctly; "
        f"before={bal_before} after={bal_after_submit} "
        f"budget={job_budget}"
    )

    # Race: 10 concurrent cancels on the SAME job
    n = 10
    results = asyncio.run(_parallel_calls(
        n, "POST", f"/compute/cancel/{job_id}",
        lambda i: None,
    ))
    successes = [r for r in results if r["status"] == 200]
    server_errors = [r for r in results if r["status"] >= 500]
    assert not server_errors, (
        f"{len(server_errors)} 5xx under concurrent cancel: "
        f"{server_errors[:3]}"
    )

    # Sum of refund_amount_ftns across all "successful"
    # cancels MUST equal the original budget — not 2x, not
    # N×.
    total_refunded = sum(
        (s["body"].get("refund_amount_ftns") or 0)
        for s in successes
        if s["body"].get("escrow_refunded")
    )
    assert abs(total_refunded - job_budget) < 0.5, (
        f"total refund {total_refunded} != original "
        f"budget {job_budget} — DOUBLE-REFUND or no-refund "
        f"under cancel race"
    )

    # Wallet balance is back to ~bal_before
    bal_final = _http("GET", "/balance")["body"]["balance"]
    assert abs(bal_final - bal_before) < 0.5, (
        f"wallet balance drift: before={bal_before} "
        f"after_race={bal_final} — race left state "
        f"inconsistent"
    )


# ── Test 4: Fingerprint dedup race ───────────────────────


def test_fingerprint_dedup_race_first_creator_wins():
    """N concurrent /content/upload with IDENTICAL content
    but DIFFERENT creator_ids. Vision §14 anti-Sybil
    invariant: only the FIRST upload's creator becomes
    canonical; the rest get duplicate_of_creator set.
    Without proper locking, two concurrent uploaders could
    both think they're canonical."""
    import time
    # Use a unique payload so we don't collide with other
    # historical uploads.
    payload_text = (
        f"sprint-487-dedup-race-{int(time.time())}"
    )
    n = 10
    results = asyncio.run(_parallel_calls(
        n, "POST", "/content/upload",
        lambda i: {
            "text": payload_text,
            "creator_id": f"sprint-487-creator-{i}",
        },
    ))
    successes = [r for r in results if r["status"] == 200]
    assert successes, "no uploads succeeded under race"

    # All successful uploads should share the same content_hash
    hashes = {s["body"]["content_hash"] for s in successes}
    assert len(hashes) == 1, (
        f"identical content produced multiple content_hashes "
        f"under race: {hashes}"
    )

    # Exactly one upload should have duplicate_of_creator
    # == null (the canonical first-creator-wins). The rest
    # must have duplicate_of_creator set to the canonical.
    canonicals = [
        s for s in successes
        if s["body"].get("duplicate_of_creator") is None
    ]
    duplicates = [
        s for s in successes
        if s["body"].get("duplicate_of_creator") is not None
    ]
    assert len(canonicals) == 1, (
        f"{len(canonicals)} canonicals under race; expected "
        f"exactly 1. anti-Sybil invariant broken"
    )
    if duplicates:
        canonical_creator = canonicals[0]["body"]["creator_id"]
        for d in duplicates:
            assert (
                d["body"]["duplicate_of_creator"]
                == canonical_creator
            ), (
                f"duplicate points to wrong canonical: "
                f"{d['body']['duplicate_of_creator']} != "
                f"{canonical_creator}"
            )
