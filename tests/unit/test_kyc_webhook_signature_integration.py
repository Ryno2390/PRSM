"""Sprint 283 — webhook endpoint signature enforcement.

Tests the POST /wallet/kyc/webhook/{vendor} endpoint's
integration with the sprint-283 signature verifier. Confirms:
  - Valid signature → 200
  - Invalid signature → 401
  - Missing signature when secret configured → 401
  - Secret unset → bypass (sprint-280 behavior preserved
    when operator hasn't wired secrets yet)
  - PRSM_KYC_WEBHOOK_VERIFY_DISABLED=1 → bypass (dev/staging
    escape hatch)
"""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.economy.web3.kyc_client import KYCClient
from prsm.node.api import create_api_app


class FakeKYCBackend:
    def initiate_session(self, user_id, email, level):
        return {
            "vendor_ref": f"persona-{user_id}",
            "session_url": f"https://persona.example/v/{user_id}",
            "status": "INITIATED",
        }


def _client(kyc=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._kyc_client = kyc
    node._fiat_compliance_ring = None
    node._coinbase_waas_client = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _commissioned_kyc(vendor="persona"):
    return KYCClient(
        vendor=vendor, api_key="k", backend=FakeKYCBackend(),
    )


def _seed_alice(kyc):
    kyc.initiate(user_id="alice", email="a@x.io", level="basic")


def _persona_header(body_bytes, secret, ts=None):
    # Sprint 284 — default to current unix ts so freshness
    # window check passes. Callers exercising the stale-ts
    # case must supply ts explicitly.
    if ts is None:
        import time as _time
        ts = str(int(_time.time()))
    payload = f"{ts}.".encode("utf-8") + body_bytes
    sig = hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256,
    ).hexdigest()
    return f"t={ts},v1={sig}"


def _onfido_header(body_bytes, token):
    return hmac.new(
        token.encode("utf-8"), body_bytes, hashlib.sha256,
    ).hexdigest()


# ── No secret configured: bypass (preserves sprint-280) ──


def test_webhook_no_secret_bypasses(monkeypatch):
    """When PERSONA_WEBHOOK_SECRET is unset, sprint-280's
    behavior is preserved — the webhook updates state without
    signature check. This is a deliberate v1 default to keep
    existing operators unblocked while they wire secrets."""
    monkeypatch.delenv("PERSONA_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("ONFIDO_WEBHOOK_TOKEN", raising=False)
    monkeypatch.delenv(
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED", raising=False,
    )
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "VERIFIED"},
    )
    assert resp.status_code == 200


# ── Persona secret configured: enforcement on ────────────


def test_webhook_persona_valid_signature_200(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    header = _persona_header(body, "wh_secret")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "VERIFIED"


def test_webhook_persona_missing_signature_401(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "VERIFIED"},
    )
    assert resp.status_code == 401


def test_webhook_persona_bad_signature_401(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    bad_header = _persona_header(body, "wrong_secret")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": bad_header,
        },
    )
    assert resp.status_code == 401


def test_webhook_persona_tampered_body_401(monkeypatch):
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    signed_body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    header = _persona_header(signed_body, "wh_secret")
    tampered = json.dumps({
        "user_id": "alice", "status": "REJECTED",
    }).encode("utf-8")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=tampered,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": header,
        },
    )
    assert resp.status_code == 401


def test_webhook_persona_bad_signature_does_not_update_state(
    monkeypatch,
):
    """Verify the security property: failed signature MUST NOT
    update KYC state. This is the actual attack we're
    defending against."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    bad_header = _persona_header(body, "wrong_secret")
    _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        content=body,
        headers={
            "Content-Type": "application/json",
            "Persona-Signature": bad_header,
        },
    )
    rec = kyc.get_status("alice")
    assert rec.status == "INITIATED"  # not VERIFIED


# ── Onfido secret configured: enforcement on ─────────────


def test_webhook_onfido_valid_signature_200(monkeypatch):
    monkeypatch.setenv("ONFIDO_WEBHOOK_TOKEN", "wh_tok")
    kyc = _commissioned_kyc(vendor="onfido")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    header = _onfido_header(body, "wh_tok")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/onfido",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-SHA2-Signature": header,
        },
    )
    assert resp.status_code == 200


def test_webhook_onfido_bad_signature_401(monkeypatch):
    monkeypatch.setenv("ONFIDO_WEBHOOK_TOKEN", "wh_tok")
    kyc = _commissioned_kyc(vendor="onfido")
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    bad_header = _onfido_header(body, "wrong_token")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/onfido",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-SHA2-Signature": bad_header,
        },
    )
    assert resp.status_code == 401


# ── Disable flag escape hatch ────────────────────────────


def test_verify_disabled_flag_bypasses(monkeypatch):
    """Dev/staging escape hatch: PRSM_KYC_WEBHOOK_VERIFY_DISABLED
    forces bypass even when secret is set."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    monkeypatch.setenv(
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED", "1",
    )
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "VERIFIED"},
    )
    assert resp.status_code == 200


def test_verify_disabled_flag_must_be_explicit_true(monkeypatch):
    """The disable flag must be exactly '1' / 'true' / etc —
    not just 'set to anything'."""
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    monkeypatch.setenv("PRSM_KYC_WEBHOOK_VERIFY_DISABLED", "0")
    kyc = _commissioned_kyc(vendor="persona")
    _seed_alice(kyc)
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/persona",
        json={"user_id": "alice", "status": "VERIFIED"},
    )
    # '0' is NOT a disable signal — enforcement still on
    assert resp.status_code == 401


# ── Path-vendor / configured-vendor disagreement ─────────


def test_webhook_path_vendor_mismatches_configured(monkeypatch):
    """Operator runs `KYC_VENDOR=persona` but receives a hit
    on /wallet/kyc/webhook/onfido — current behavior: use the
    path vendor for signature verification (vendor-specific
    secrets are env-var-keyed by vendor name). This matches
    the realistic case where one operator runs multi-vendor
    receive-paths."""
    monkeypatch.setenv("ONFIDO_WEBHOOK_TOKEN", "wh_tok")
    kyc = _commissioned_kyc(vendor="persona")  # configured
    _seed_alice(kyc)
    body = json.dumps({
        "user_id": "alice", "status": "VERIFIED",
    }).encode("utf-8")
    header = _onfido_header(body, "wh_tok")
    resp = _client(kyc).post(
        "/wallet/kyc/webhook/onfido",  # path-vendor: onfido
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-SHA2-Signature": header,
        },
    )
    # Path-vendor's secret validates → 200
    assert resp.status_code == 200
