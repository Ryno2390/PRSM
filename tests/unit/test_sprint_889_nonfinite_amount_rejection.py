"""Sprint 889 — reject non-finite (NaN/Infinity) fiat amounts.

sp887 finding: FiatComplianceRing.record guarded `usd_amount < 0`,
but `NaN < 0` and `inf < 0` are both False in Python — so a
non-finite amount slipped through. Worse, Python's json.loads
accepts the non-standard `NaN`/`Infinity`/`-Infinity` tokens, and
the fiat endpoints' `if usd_amount <= 0` check ALSO passes NaN
(`NaN <= 0` is False). A single NaN amount poisons the rolling
tier total: total_usd_for_user sums to NaN, and the sp884 check
`(NaN + requested) > limit` is always False — permanently
disabling tier-limit enforcement for that user (a compliance
bypass / DoS-of-enforcement).

Sp889 rejects non-finite amounts at two layers (defense in depth):
  - FiatComplianceRing.record — the rolling-total chokepoint
  - the onramp-quote / onramp-execute / offramp-quote endpoints
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.fiat_compliance_ring import FiatComplianceRing
from prsm.node.api import create_api_app


# ── Ring chokepoint ──────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), float("-inf"),
])
def test_ring_rejects_non_finite_usd(bad):
    r = FiatComplianceRing()
    with pytest.raises(ValueError) as exc:
        r.record(
            kind="onramp_execute", user_id="alice",
            usd_amount=bad, ftns_amount=0.0, status="CONFIRMED",
        )
    assert "finite" in str(exc.value).lower()


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), float("-inf"),
])
def test_ring_rejects_non_finite_ftns(bad):
    r = FiatComplianceRing()
    with pytest.raises(ValueError) as exc:
        r.record(
            kind="onramp_execute", user_id="alice",
            usd_amount=10.0, ftns_amount=bad, status="CONFIRMED",
        )
    assert "finite" in str(exc.value).lower()


def test_ring_still_rejects_negative():
    """The existing negative guard must remain."""
    r = FiatComplianceRing()
    with pytest.raises(ValueError):
        r.record(
            kind="onramp_execute", user_id="alice",
            usd_amount=-1.0, ftns_amount=0.0, status="CONFIRMED",
        )


def test_ring_accepts_normal_amount():
    r = FiatComplianceRing()
    e = r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=100.0, ftns_amount=50.0, status="CONFIRMED",
    )
    assert e.usd_amount == 100.0


def test_nan_does_not_poison_rolling_total():
    """The load-bearing regression: even if a caller TRIES to
    record NaN, it's rejected, so total_usd_for_user stays a
    finite, enforceable number."""
    r = FiatComplianceRing()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=500.0, ftns_amount=0.0, status="CONFIRMED",
    )
    with pytest.raises(ValueError):
        r.record(
            kind="onramp_execute", user_id="alice",
            usd_amount=float("nan"), ftns_amount=0.0,
            status="CONFIRMED",
        )
    total = r.total_usd_for_user("alice")
    assert math.isfinite(total)
    assert total == 500.0  # NaN never entered the sum


# ── Endpoint layer ───────────────────────────────────────────

def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._coinbase_waas_client = None
    node._kyc_client = None
    node._fiat_compliance_ring = None
    node._aerodrome_client = None
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_onramp_quote_rejects_non_finite(token):
    # Send a raw JSON body with the non-standard non-finite token
    # (Python's json + Starlette accept these).
    resp = _client().post(
        "/wallet/onramp/quote",
        content=(
            '{"usd_amount": ' + token
            + ', "destination_address": "0x'
            + "11" * 20 + '"}'
        ),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_onramp_execute_rejects_non_finite(token):
    resp = _client().post(
        "/wallet/onramp/execute",
        content=(
            '{"usd_amount": ' + token
            + ', "destination_address": "0x'
            + "11" * 20 + '"}'
        ),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_offramp_quote_rejects_non_finite(token):
    resp = _client().post(
        "/wallet/offramp/quote",
        content='{"usd_amount": ' + token + '}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in (400, 422)


def test_onramp_quote_still_accepts_normal_amount():
    """Regression: a normal positive amount still works (the
    non-finite guard must not reject finite values)."""
    resp = _client().post(
        "/wallet/onramp/quote",
        json={
            "usd_amount": 100.0,
            "destination_address": "0x" + "11" * 20,
        },
    )
    assert resp.status_code == 200
