"""Sprint 858 — KYC→WaaS auto-provision orchestrator pin tests.

Defends the conditional WaaS-provision trigger that fires when a
Persona webhook flips a KYC record to VERIFIED. Closes the manual
"call provision after KYC" UX seam.

Pin tests:
  - VERIFIED triggers provision_wallet
  - REJECTED / EXPIRED / PENDING do NOT trigger
  - Idempotent: same VERIFIED→VERIFIED transition skips (operator
    re-fired webhook, no need to re-provision)
  - No-op when waas_client is None (operator hasn't wired WaaS yet)
  - No-op when email is missing (defensive — provision requires email)
  - Provision exceptions caught + logged (webhook still returns 200)
  - Returns the provisioned record when successful
  - Returns None when no provision attempted
"""
from __future__ import annotations

from prsm.economy.web3.kyc_to_waas_orchestrator import (
    maybe_auto_provision_waas,
)


class _FakeWaas:
    """In-memory WaaS double for testing the orchestration trigger."""
    def __init__(self, *, raise_on_provision: bool = False):
        self.calls: list = []
        self._raise = raise_on_provision

    def provision_wallet(self, user_id: str, email: str):
        self.calls.append((user_id, email))
        if self._raise:
            raise RuntimeError("simulated CDP failure")

        class _Rec:
            status = "PROVISIONED"
            address = "0x" + "11" * 20
        return _Rec()


# ── Status filtering ─────────────────────────────────────────

def test_verified_triggers_provision():
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="VERIFIED", old_status="INITIATED",
    )
    assert rec is not None
    assert rec.status == "PROVISIONED"
    assert waas.calls == [("alice", "a@x.io")]


def test_rejected_does_not_trigger():
    """A rejected KYC user has no business getting a wallet
    provisioned — keeps the user-bind surface narrow."""
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="REJECTED", old_status="INITIATED",
    )
    assert rec is None
    assert waas.calls == []


def test_expired_does_not_trigger():
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="EXPIRED", old_status="INITIATED",
    )
    assert rec is None
    assert waas.calls == []


def test_pending_does_not_trigger():
    """PENDING means user submitted but vendor hasn't decided yet
    — definitely don't provision on pending."""
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="PENDING", old_status="INITIATED",
    )
    assert rec is None


def test_initiated_does_not_trigger():
    """INITIATED is the freshly-created state — not a vendor
    decision."""
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="INITIATED", old_status=None,
    )
    assert rec is None


# ── Idempotency ──────────────────────────────────────────────

def test_verified_to_verified_transition_skips():
    """Persona retries webhooks — if same VERIFIED event re-fires,
    don't double-provision (WaaS is idempotent anyway but the
    skip saves the round-trip)."""
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="VERIFIED", old_status="VERIFIED",
    )
    assert rec is None
    assert waas.calls == []


def test_initiated_to_verified_triggers():
    """The canonical happy path — user submits, vendor approves."""
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="VERIFIED", old_status="INITIATED",
    )
    assert rec is not None
    assert waas.calls == [("alice", "a@x.io")]


def test_pending_to_verified_triggers():
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="VERIFIED", old_status="PENDING",
    )
    assert rec is not None


# ── Fail-soft (webhook must stay 200) ────────────────────────

def test_no_waas_client_skips():
    """Operator hasn't wired WaaS yet — skip silently."""
    rec = maybe_auto_provision_waas(
        waas_client=None, user_id="alice", email="a@x.io",
        new_status="VERIFIED", old_status="INITIATED",
    )
    assert rec is None


def test_missing_email_skips():
    """Defensive: WaaS.provision_wallet requires email; never
    call with empty."""
    waas = _FakeWaas()
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="",
        new_status="VERIFIED", old_status="INITIATED",
    )
    assert rec is None
    assert waas.calls == []


def test_provision_raises_caught_and_logged():
    """The whole point of fail-soft: webhook handler must keep
    returning 200 to Persona even if provision blows up."""
    waas = _FakeWaas(raise_on_provision=True)
    rec = maybe_auto_provision_waas(
        waas_client=waas, user_id="alice", email="a@x.io",
        new_status="VERIFIED", old_status="INITIATED",
    )
    # Returns None on failure — webhook returns 200 with the
    # updated KYC record (VERIFIED stands).
    assert rec is None
    # Provision WAS attempted — just failed cleanly
    assert waas.calls == [("alice", "a@x.io")]
