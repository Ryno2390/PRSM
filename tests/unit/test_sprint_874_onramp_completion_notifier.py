"""Sprint 874 — onramp completion outbound webhook notifier tests."""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass

import httpx
import pytest


_real_Client = httpx.Client


@pytest.fixture(autouse=True)
def _restore_real_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", httpx.MockTransport)
    yield


from prsm.economy.web3.onramp_completion_notifier import (
    DeliveryRecord,
    OnrampCompletionNotifier,
    _signature_header,
    from_env,
)


@dataclass
class _FakeIntent:
    intent_id: str = "onramp_abc"
    user_id: str = "alice"
    destination_address: str = "0x" + "11" * 20
    expected_usd: float = 5.0
    usdc_received: float = 4.92
    confirmed_at: float = 1_700_000_100.0
    swap_envelope: dict = None


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── Unconfigured: no-op ──────────────────────────────────────

def test_unconfigured_returns_none(tmp_path):
    n = OnrampCompletionNotifier(log_dir=tmp_path)
    result = n.notify(intent=_FakeIntent())
    assert result is None


def test_is_configured_reflects_url_presence(tmp_path):
    n_no = OnrampCompletionNotifier(log_dir=tmp_path)
    assert n_no.is_configured() is False
    n_yes = OnrampCompletionNotifier(
        url="https://example.com/wh", log_dir=tmp_path,
    )
    assert n_yes.is_configured() is True


# ── Configured: successful POST ──────────────────────────────

def test_successful_post_recorded(tmp_path):
    captured = []

    def handler(request):
        captured.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": json.loads(request.content),
        })
        return httpx.Response(200, json={"ok": True})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    rec = n.notify(intent=_FakeIntent())
    assert rec is not None
    assert rec.success is True
    assert rec.status_code == 200
    assert rec.error is None
    assert len(captured) == 1
    assert captured[0]["body"]["intent_id"] == "onramp_abc"
    assert captured[0]["body"]["usdc_received"] == 4.92
    assert captured[0]["body"]["event"] == "onramp.completion"


def test_post_persisted_to_disk(tmp_path):
    def handler(request):
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    n.notify(intent=_FakeIntent())
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


# ── HMAC signature ───────────────────────────────────────────

def test_signature_header_format():
    """The signature header matches Persona's sp283 pattern:
    `t=<unix>,v1=<hex hmac>` so receivers reuse the same
    verification code path PRSM already uses for KYC webhooks."""
    secret = "wbhsec_test"
    body = b'{"hello":"world"}'
    name, val = _signature_header(secret, body)
    assert name == "X-PRSM-Signature"
    assert val.startswith("t=")
    assert ",v1=" in val
    parts = dict(p.split("=", 1) for p in val.split(","))
    assert "t" in parts and "v1" in parts
    # Verify the HMAC
    expected = hmac.new(
        secret.encode(),
        f"{parts['t']}.".encode() + body,
        hashlib.sha256,
    ).hexdigest()
    assert parts["v1"] == expected


def test_signature_attached_when_secret_set(tmp_path):
    captured = []

    def handler(request):
        captured.append(dict(request.headers))
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        secret="wbhsec_test",
        log_dir=tmp_path, client=_mock(handler),
    )
    rec = n.notify(intent=_FakeIntent())
    assert rec.signature_attached is True
    assert "x-prsm-signature" in captured[0]


def test_no_signature_when_secret_unset(tmp_path):
    captured = []

    def handler(request):
        captured.append(dict(request.headers))
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        secret=None,
        log_dir=tmp_path, client=_mock(handler),
    )
    rec = n.notify(intent=_FakeIntent())
    assert rec.signature_attached is False
    assert "x-prsm-signature" not in captured[0]


# ── Fail-soft on transport / 5xx ─────────────────────────────

def test_5xx_response_recorded_as_failure(tmp_path):
    def handler(request):
        return httpx.Response(500, text="internal")

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    rec = n.notify(intent=_FakeIntent())
    assert rec.success is False
    assert rec.status_code == 500
    assert "internal" in (rec.error or "")


def test_4xx_response_recorded_as_failure(tmp_path):
    def handler(request):
        return httpx.Response(401, text="bad sig")

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    rec = n.notify(intent=_FakeIntent())
    assert rec.success is False
    assert rec.status_code == 401


def test_transport_error_recorded_as_failure(tmp_path):
    """Connection refused / timeout — record with status_code=0
    and error string. CONFIRMED transition (sp871) must still
    hold; the webhook just records the dispatch failure for
    operator investigation."""

    def handler(request):
        raise httpx.ConnectError("refused")

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    rec = n.notify(intent=_FakeIntent())
    assert rec.success is False
    assert rec.status_code == 0
    assert "ConnectError" in rec.error or "refused" in rec.error


# ── list_deliveries ──────────────────────────────────────────

def test_list_deliveries_newest_first(tmp_path):
    """Multiple deliveries persist + list in newest-first order."""
    def handler(request):
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    n.notify(intent=_FakeIntent(intent_id="i1"))
    import time
    time.sleep(0.01)
    n.notify(intent=_FakeIntent(intent_id="i2"))
    records = n.list_deliveries()
    assert len(records) == 2
    # Newest first — i2 was written second
    assert records[0]["intent_id"] == "i2"
    assert records[1]["intent_id"] == "i1"


def test_list_deliveries_respects_limit(tmp_path):
    def handler(request):
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    for i in range(5):
        import time as _t
        _t.sleep(0.001)
        n.notify(intent=_FakeIntent(intent_id=f"i{i}"))
    records = n.list_deliveries(limit=3)
    assert len(records) == 3


def test_list_deliveries_empty_when_log_dir_memory(monkeypatch):
    """`:memory:` opt-out: no log dir, list returns []."""
    monkeypatch.setenv(
        "PRSM_ONRAMP_COMPLETION_WEBHOOK_LOG_DIR", ":memory:",
    )
    n = OnrampCompletionNotifier(url="https://example.com/wh")
    assert n.list_deliveries() == []


# ── from_env ─────────────────────────────────────────────────

def test_from_env_no_url_unconfigured(monkeypatch):
    monkeypatch.delenv(
        "PRSM_ONRAMP_COMPLETION_WEBHOOK_URL", raising=False,
    )
    monkeypatch.delenv(
        "PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET", raising=False,
    )
    n = from_env()
    assert n.is_configured() is False


def test_from_env_with_url_and_secret(monkeypatch):
    monkeypatch.setenv(
        "PRSM_ONRAMP_COMPLETION_WEBHOOK_URL",
        "https://example.com/wh",
    )
    monkeypatch.setenv(
        "PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET", "wbhsec_x",
    )
    n = from_env()
    assert n.is_configured() is True
    assert n._url == "https://example.com/wh"
    assert n._secret == "wbhsec_x"


# ── Payload shape ────────────────────────────────────────────

def test_payload_includes_canonical_fields(tmp_path):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    intent = _FakeIntent(
        swap_envelope={"status": "READY_FOR_SUBMISSION"},
    )
    n.notify(intent=intent)
    body = captured[0]
    # All canonical fields present
    for field in (
        "event", "intent_id", "user_id",
        "destination_address", "expected_usd",
        "usdc_received", "confirmed_at", "swap_envelope",
    ):
        assert field in body
    assert body["swap_envelope"]["status"] == (
        "READY_FOR_SUBMISSION"
    )


def test_payload_swap_envelope_null_when_intent_has_none(tmp_path):
    """When pool ceremony pending, swap_envelope is None; payload
    serializes it as JSON null + downstream receivers see
    'completion happened but envelope not yet ready'."""
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={})

    n = OnrampCompletionNotifier(
        url="https://example.com/wh",
        log_dir=tmp_path, client=_mock(handler),
    )
    intent = _FakeIntent()
    intent.swap_envelope = None
    n.notify(intent=intent)
    assert captured[0]["swap_envelope"] is None
