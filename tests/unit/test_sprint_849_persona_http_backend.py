"""Sprint 849 — Persona HTTP backend + KYCClient auto-wire.

Closes the (commissioned=True, adapter_wired=False) state that sp848
surfaced for the KYC vendor track. Wires a real httpx-based Persona
backend that:
  1. POSTs to /inquiries with reference-id + email
  2. POSTs to /inquiries/{id}/generate-one-time-link
  3. Returns vendor_ref + session_url for KYCRecord persistence

KYCClient.from_env() now auto-attaches PersonaHttpBackend when
vendor=persona + KYC_VENDOR_API_KEY + PERSONA_TEMPLATE_ID are set.

Pin tests:
  - PersonaHttpBackend.initiate_session hits both Persona endpoints
    in order with correct headers + payload (httpx.MockTransport)
  - Returns vendor_ref from inquiry id + session_url from one-time link
  - Falls back to hosted-flow URL when one-time link is missing
  - Raises clean error when inquiry create returns no id
  - HTTP 4xx surfaces as httpx error (caller catches → REJECTED)
  - from_env() returns None when api_key or template_id missing
  - from_env() constructs backend when both env vars present
  - KYCClient.from_env() auto-attaches the backend when env set
  - KYCClient.from_env() respects explicit backend= override
  - KYCClient.from_env() falls back gracefully when persona import fails
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

# Capture real httpx classes before the conftest autouse fixture
# patches them — mirrors the pattern in
# tests/unit/test_http_aggregate_transport.py. Without this, the
# global `mock_http_requests` fixture replaces httpx.Client with a
# MagicMock that returns canned responses, breaking MockTransport.
_real_Client = httpx.Client
_real_MockTransport = httpx.MockTransport
_real_Response = httpx.Response
_real_HTTPStatusError = httpx.HTTPStatusError


@pytest.fixture(autouse=True)
def _restore_real_httpx_classes(monkeypatch):
    """Override the global conftest httpx mocking for this file."""
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", _real_MockTransport)
    monkeypatch.setattr(httpx, "Response", _real_Response)
    monkeypatch.setattr(httpx, "HTTPStatusError", _real_HTTPStatusError)
    yield


from prsm.economy.web3.kyc_persona_backend import (
    PersonaHttpBackend, from_env as persona_from_env,
)
from prsm.economy.web3.kyc_client import KYCClient


# ── HTTP MockTransport handler helpers ───────────────────────

def _mock_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _handler_two_step_happy(
    inquiry_id: str = "inq_test_123",
    link: str = "https://withpersona.com/verify/onetime/abc",
):
    """Returns a handler that responds to both Persona calls."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": (
                json.loads(request.content) if request.content else None
            ),
        })
        if request.url.path.endswith("/inquiries"):
            return httpx.Response(
                201,
                json={
                    "data": {
                        "id": inquiry_id,
                        "type": "inquiry",
                        "attributes": {"status": "created"},
                    }
                },
            )
        if "generate-one-time-link" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "attributes": {"link": link},
                    }
                },
            )
        return httpx.Response(404, json={"error": "unmocked path"})

    return handler, calls


# ── PersonaHttpBackend.initiate_session ──────────────────────

def test_initiate_session_two_step_happy_path():
    handler, calls = _handler_two_step_happy()
    client = _mock_transport(handler)
    backend = PersonaHttpBackend(
        api_key="persona_sandbox_xxx",
        template_id="itmpl_yyy",
        client=client,
    )
    result = backend.initiate_session(
        user_id="alice", email="a@x.io", level="basic",
    )
    assert result["vendor_ref"] == "inq_test_123"
    assert result["session_url"] == (
        "https://withpersona.com/verify/onetime/abc"
    )
    assert result["status"] == "INITIATED"

    # 2 calls in order
    assert len(calls) == 2
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("/inquiries")
    assert calls[1]["method"] == "POST"
    assert "generate-one-time-link" in calls[1]["url"]


def test_initiate_sends_bearer_token_and_api_version():
    handler, calls = _handler_two_step_happy()
    client = _mock_transport(handler)
    backend = PersonaHttpBackend(
        api_key="persona_sandbox_xxx",
        template_id="itmpl_yyy",
        client=client,
    )
    backend.initiate_session("alice", "a@x.io", "basic")
    h0 = calls[0]["headers"]
    assert h0["authorization"] == "Bearer persona_sandbox_xxx"
    assert h0["persona-version"] == "2023-01-05"
    assert h0["content-type"] == "application/json"


def test_initiate_payload_jsonapi_shape():
    """Persona's inquiries API expects JSON:API shape with
    inquiry-template-id + reference-id + fields.email-address."""
    handler, calls = _handler_two_step_happy()
    client = _mock_transport(handler)
    backend = PersonaHttpBackend(
        api_key="k", template_id="itmpl_abc", client=client,
    )
    backend.initiate_session("user_42", "u@x.io", "basic")
    body = calls[0]["body"]
    attrs = body["data"]["attributes"]
    assert attrs["inquiry-template-id"] == "itmpl_abc"
    assert attrs["reference-id"] == "user_42"
    assert attrs["fields"]["email-address"] == "u@x.io"


def test_initiate_falls_back_to_hosted_flow_when_no_link():
    """If Persona's one-time-link endpoint returns no link
    attribute (some configs omit it), fall back to the hosted
    flow URL pattern rather than raising — the user still has
    a usable verification URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/inquiries"):
            return httpx.Response(
                201,
                json={"data": {"id": "inq_no_link", "type": "inquiry"}},
            )
        # link endpoint returns 200 but no link attribute
        return httpx.Response(200, json={"data": {"attributes": {}}})

    client = _mock_transport(handler)
    backend = PersonaHttpBackend(
        api_key="k", template_id="itmpl_y", client=client,
    )
    result = backend.initiate_session("a", "a@x.io", "basic")
    assert result["session_url"] == (
        "https://withpersona.com/verify?inquiry-id=inq_no_link"
    )


def test_initiate_raises_when_inquiry_create_returns_no_id():
    """If Persona returns a malformed inquiry response (no id),
    raise RuntimeError. KYCClient.initiate catches generic
    exceptions and persists a REJECTED record — better signal
    than silently treating bad responses as success."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})  # no id

    client = _mock_transport(handler)
    backend = PersonaHttpBackend(
        api_key="k", template_id="itmpl_y", client=client,
    )
    with pytest.raises(RuntimeError) as exc:
        backend.initiate_session("a", "a@x.io", "basic")
    assert "no id" in str(exc.value).lower()


def test_initiate_raises_on_http_401():
    """Persona returns 401 for bad API key. httpx.raise_for_status
    surfaces the error; KYCClient.initiate catches + persists
    REJECTED record."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = _mock_transport(handler)
    backend = PersonaHttpBackend(
        api_key="bad_key", template_id="itmpl_y", client=client,
    )
    with pytest.raises(httpx.HTTPStatusError):
        backend.initiate_session("a", "a@x.io", "basic")


def test_constructor_validates_required_args():
    with pytest.raises(ValueError):
        PersonaHttpBackend(api_key="", template_id="itmpl_x")
    with pytest.raises(ValueError):
        PersonaHttpBackend(api_key="k", template_id="")


# ── kyc_persona_backend.from_env ─────────────────────────────

def test_persona_from_env_returns_none_when_api_key_missing(
    monkeypatch,
):
    monkeypatch.delenv("KYC_VENDOR_API_KEY", raising=False)
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_y")
    assert persona_from_env() is None


def test_persona_from_env_returns_none_when_template_missing(
    monkeypatch,
):
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.delenv("PERSONA_TEMPLATE_ID", raising=False)
    assert persona_from_env() is None


def test_persona_from_env_constructs_when_both_present(monkeypatch):
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_y")
    backend = persona_from_env()
    assert backend is not None
    assert isinstance(backend, PersonaHttpBackend)


# ── KYCClient.from_env auto-wire ─────────────────────────────

def test_kyc_from_env_auto_attaches_persona_when_env_set(monkeypatch):
    """The load-bearing sp849 wire: env→client→backend
    auto-attaches. Closes (commissioned=T, adapter_wired=F)."""
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_y")
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)
    c = KYCClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is True


def test_kyc_from_env_no_auto_attach_without_template(monkeypatch):
    """Half-configured env — vendor + api_key but no template id.
    Persona backend should NOT auto-attach (would crash on
    initiate). adapter_wired stays False; operator sees the gap."""
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.delenv("PERSONA_TEMPLATE_ID", raising=False)
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)
    c = KYCClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False


def test_kyc_from_env_respects_explicit_backend(monkeypatch):
    """Test seam preservation: explicit backend= override beats
    auto-wire so existing tests can still inject fakes."""
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_y")
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)

    class _Fake:
        def initiate_session(self, *a, **k):
            return {"vendor_ref": "f", "session_url": "u"}

    fake = _Fake()
    c = KYCClient.from_env(backend=fake)
    assert c._backend is fake
    assert c.adapter_wired() is True


def test_kyc_from_env_graceful_fallback_on_persona_import_failure(
    monkeypatch,
):
    """If kyc_persona_backend.from_env raises (e.g., httpx missing
    in some weird env), KYCClient.from_env logs + returns an
    un-backed client rather than crashing the whole node boot."""
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_y")
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)
    with patch(
        "prsm.economy.web3.kyc_persona_backend.from_env",
        side_effect=RuntimeError("boom"),
    ):
        c = KYCClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False


def test_kyc_from_env_no_auto_attach_when_vendor_not_persona(
    monkeypatch,
):
    """When KYC_VENDOR=onfido (or anything not persona), don't
    try to wire a persona backend even if api_key present."""
    monkeypatch.setenv("KYC_VENDOR", "onfido")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_y")
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)
    c = KYCClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False
