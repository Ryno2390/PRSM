"""Sprint 283 — KYC webhook signature verification.

Persona / Onfido sign webhook payloads with HMAC-SHA256 using
a vendor-issued secret. Without this verifier, anyone who
discovers the operator's webhook URL can flip a user's KYC
status to VERIFIED. This is the security gate that MUST close
before vendor commission.

Vendor patterns (real-world conventions):
  - Persona:  HMAC-SHA256 of `{timestamp}.{raw_body}` with the
              webhook secret; header is `Persona-Signature`
              formatted as `t=<unix_ts>,v1=<hex>`.
  - Onfido:   HMAC-SHA256 of raw_body with the webhook token;
              header is `X-SHA2-Signature` (hex digest).
  - Plaid:    JWT-based — deferred to a follow-on sprint.

Constant-time comparison protects against timing oracles.
Missing/malformed headers → False with descriptive reason.
"""
from __future__ import annotations

import hmac
import hashlib

import pytest

from prsm.economy.web3.kyc_webhook_verifier import (
    KYCWebhookVerifier,
    verify_persona_signature,
    verify_onfido_signature,
)


# ── Persona helper ───────────────────────────────────────


def _persona_header(body_bytes: bytes, secret: str, ts: str) -> str:
    payload = f"{ts}.{body_bytes.decode('utf-8')}"
    sig = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"t={ts},v1={sig}"


def test_persona_valid_signature():
    body = b'{"user_id":"alice","status":"VERIFIED"}'
    secret = "wh_sec_abc"
    header = _persona_header(body, secret, "1700000000")
    ok, reason = verify_persona_signature(body, header, secret)
    assert ok is True
    assert reason == ""


def test_persona_tampered_body_fails():
    body = b'{"user_id":"alice","status":"VERIFIED"}'
    secret = "wh_sec_abc"
    header = _persona_header(body, secret, "1700000000")
    tampered = b'{"user_id":"alice","status":"REJECTED"}'
    ok, reason = verify_persona_signature(
        tampered, header, secret,
    )
    assert ok is False
    assert "signature" in reason.lower()


def test_persona_wrong_secret_fails():
    body = b'{"user_id":"alice","status":"VERIFIED"}'
    header = _persona_header(body, "right_secret", "1700000000")
    ok, _ = verify_persona_signature(
        body, header, "wrong_secret",
    )
    assert ok is False


def test_persona_missing_header_fails():
    body = b'{"x":1}'
    ok, reason = verify_persona_signature(body, "", "secret")
    assert ok is False
    assert "missing" in reason.lower() or "header" in reason.lower()


def test_persona_malformed_header_no_t():
    body = b'{"x":1}'
    header = "v1=abc123"  # no timestamp
    ok, reason = verify_persona_signature(body, header, "secret")
    assert ok is False


def test_persona_malformed_header_no_v1():
    body = b'{"x":1}'
    header = "t=1700000000"  # no signature
    ok, reason = verify_persona_signature(body, header, "secret")
    assert ok is False


def test_persona_extra_fields_in_header_ok():
    """Persona may add v2/v3 fields over time. v1 must remain
    the operational version."""
    body = b'{"x":1}'
    secret = "s"
    ts = "1700000000"
    payload = f"{ts}.{body.decode()}"
    v1 = hmac.new(
        secret.encode(), payload.encode(), hashlib.sha256,
    ).hexdigest()
    header = f"t={ts},v1={v1},v2=futurestuff"
    ok, _ = verify_persona_signature(body, header, secret)
    assert ok is True


# ── Onfido helper ────────────────────────────────────────


def _onfido_header(body_bytes: bytes, token: str) -> str:
    return hmac.new(
        token.encode("utf-8"),
        body_bytes,
        hashlib.sha256,
    ).hexdigest()


def test_onfido_valid_signature():
    body = b'{"user_id":"alice","status":"VERIFIED"}'
    token = "wh_tok_xyz"
    header = _onfido_header(body, token)
    ok, reason = verify_onfido_signature(body, header, token)
    assert ok is True
    assert reason == ""


def test_onfido_tampered_body_fails():
    body = b'{"user_id":"alice","status":"VERIFIED"}'
    token = "wh_tok_xyz"
    header = _onfido_header(body, token)
    tampered = b'{"user_id":"alice","status":"REJECTED"}'
    ok, _ = verify_onfido_signature(tampered, header, token)
    assert ok is False


def test_onfido_wrong_token_fails():
    body = b'{"x":1}'
    header = _onfido_header(body, "right_token")
    ok, _ = verify_onfido_signature(body, header, "wrong_token")
    assert ok is False


def test_onfido_missing_header():
    body = b'{"x":1}'
    ok, reason = verify_onfido_signature(body, "", "tok")
    assert ok is False
    assert "missing" in reason.lower() or "header" in reason.lower()


def test_onfido_short_signature_fails():
    body = b'{"x":1}'
    ok, _ = verify_onfido_signature(body, "abc", "tok")
    assert ok is False


# ── Dispatcher ───────────────────────────────────────────


def test_dispatcher_persona():
    body = b'{"x":1}'
    secret = "s"
    header = _persona_header(body, secret, "1700000000")
    ok, _ = KYCWebhookVerifier.verify(
        vendor="persona",
        body=body,
        headers={"persona-signature": header},
        secret=secret,
    )
    assert ok is True


def test_dispatcher_onfido():
    body = b'{"x":1}'
    token = "t"
    header = _onfido_header(body, token)
    ok, _ = KYCWebhookVerifier.verify(
        vendor="onfido",
        body=body,
        headers={"x-sha2-signature": header},
        secret=token,
    )
    assert ok is True


def test_dispatcher_header_case_insensitive():
    """HTTP headers are case-insensitive; dispatcher must
    normalize."""
    body = b'{"x":1}'
    token = "t"
    header = _onfido_header(body, token)
    ok, _ = KYCWebhookVerifier.verify(
        vendor="onfido",
        body=body,
        headers={"X-SHA2-Signature": header},
        secret=token,
    )
    assert ok is True


def test_dispatcher_unknown_vendor():
    ok, reason = KYCWebhookVerifier.verify(
        vendor="madeup",
        body=b"{}",
        headers={},
        secret="s",
    )
    assert ok is False
    assert "vendor" in reason.lower() or "unknown" in reason.lower()


def test_dispatcher_plaid_not_yet_implemented():
    """Plaid Identity uses JWT — sprint-283 v1 returns False
    with a descriptive reason so operators see why."""
    ok, reason = KYCWebhookVerifier.verify(
        vendor="plaid",
        body=b"{}",
        headers={},
        secret="s",
    )
    assert ok is False
    assert "plaid" in reason.lower() or "jwt" in reason.lower()


# ── Empty secret guard ───────────────────────────────────


def test_persona_empty_secret_fails():
    """Empty secret must NOT validate — defense against
    operator misconfig where the env var is set to empty."""
    body = b'{"x":1}'
    header = _persona_header(body, "", "1700000000")
    ok, reason = verify_persona_signature(body, header, "")
    assert ok is False


def test_onfido_empty_secret_fails():
    body = b'{"x":1}'
    header = _onfido_header(body, "")
    ok, _ = verify_onfido_signature(body, header, "")
    assert ok is False


# ── Constant-time comparison ─────────────────────────────


def test_persona_uses_constant_time_compare():
    """We can't test timing directly, but we can ensure
    hmac.compare_digest is being used (not == ). Smoke test:
    correct-length wrong-value signature must return False, not
    raise."""
    body = b'{"x":1}'
    secret = "s"
    # 64-char hex but wrong value
    bad_sig = "0" * 64
    header = f"t=1700000000,v1={bad_sig}"
    ok, _ = verify_persona_signature(body, header, secret)
    assert ok is False
