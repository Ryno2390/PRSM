"""Sprint 883 — KYC enhanced-tier template selection.

sp849's PersonaHttpBackend accepted a `level` arg on
initiate_session but ignored it — every inquiry used the single
PERSONA_TEMPLATE_ID. Enhanced KYC (Tier 2/3: proof-of-address +
source-of-funds, required for >$1k transaction limits) needs a
SEPARATE Persona inquiry template that collects those documents.

Sp883 wires level → template selection:
  level="basic"    → base template (PERSONA_TEMPLATE_ID)
  level="enhanced" → enhanced template (PERSONA_ENHANCED_TEMPLATE_ID)
                     falling back to base + a warning if the
                     enhanced template id is unset (so an operator
                     who hasn't created the enhanced template yet
                     degrades gracefully instead of 500ing).

The level already flows API→KYCClient→backend (verified sp881), so
this change is isolated to PersonaHttpBackend + from_env.
"""
from __future__ import annotations

import json

import httpx
import pytest


# Capture real httpx before conftest mocks it (sp849 pattern).
_real_Client = httpx.Client
_real_MockTransport = httpx.MockTransport
_real_Response = httpx.Response


@pytest.fixture(autouse=True)
def _restore_real_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", _real_MockTransport)
    monkeypatch.setattr(httpx, "Response", _real_Response)
    yield


from prsm.economy.web3.kyc_persona_backend import (  # noqa: E402
    PersonaHttpBackend,
    from_env as persona_from_env,
)

_BASE_TMPL = "itmpl_basic_xxx"
_ENH_TMPL = "itmpl_enhanced_yyy"


def _capture_handler():
    """Returns (handler, calls). Records the create-inquiry body +
    serves both Persona endpoints."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/inquiries"):
            calls.append(json.loads(request.content))
            return httpx.Response(
                201, json={"data": {"id": "inq_test"}},
            )
        # one-time-link
        return httpx.Response(
            200,
            json={"data": {"attributes": {"link": "https://x"}}},
        )

    return handler, calls


def _mock_client(handler):
    return _real_Client(transport=_real_MockTransport(handler))


# ── Template selection by level ──────────────────────────────

def test_basic_level_uses_base_template():
    handler, calls = _capture_handler()
    backend = PersonaHttpBackend(
        api_key="k", template_id=_BASE_TMPL,
        enhanced_template_id=_ENH_TMPL,
        client=_mock_client(handler),
    )
    backend.initiate_session("alice", "a@x.io", "basic")
    attrs = calls[0]["data"]["attributes"]
    assert attrs["inquiry-template-id"] == _BASE_TMPL


def test_enhanced_level_uses_enhanced_template():
    handler, calls = _capture_handler()
    backend = PersonaHttpBackend(
        api_key="k", template_id=_BASE_TMPL,
        enhanced_template_id=_ENH_TMPL,
        client=_mock_client(handler),
    )
    backend.initiate_session("alice", "a@x.io", "enhanced")
    attrs = calls[0]["data"]["attributes"]
    assert attrs["inquiry-template-id"] == _ENH_TMPL


def test_enhanced_level_falls_back_to_base_when_enhanced_unset():
    """Operator hasn't created the enhanced template yet — degrade
    to base + warn, NOT 500. (Better to over-collect-less than to
    hard-fail the user's upgrade attempt.)"""
    handler, calls = _capture_handler()
    backend = PersonaHttpBackend(
        api_key="k", template_id=_BASE_TMPL,
        enhanced_template_id=None,
        client=_mock_client(handler),
    )
    backend.initiate_session("alice", "a@x.io", "enhanced")
    attrs = calls[0]["data"]["attributes"]
    assert attrs["inquiry-template-id"] == _BASE_TMPL


def test_unknown_level_uses_base_template():
    """Defensive: a level string neither basic nor enhanced uses
    the base template (KYCClient validates levels upstream, but the
    backend must not KeyError on an unexpected value)."""
    handler, calls = _capture_handler()
    backend = PersonaHttpBackend(
        api_key="k", template_id=_BASE_TMPL,
        enhanced_template_id=_ENH_TMPL,
        client=_mock_client(handler),
    )
    backend.initiate_session("alice", "a@x.io", "something-else")
    attrs = calls[0]["data"]["attributes"]
    assert attrs["inquiry-template-id"] == _BASE_TMPL


def test_enhanced_returns_vendor_ref_and_session_url():
    """Enhanced flow still returns the canonical contract shape."""
    handler, _ = _capture_handler()
    backend = PersonaHttpBackend(
        api_key="k", template_id=_BASE_TMPL,
        enhanced_template_id=_ENH_TMPL,
        client=_mock_client(handler),
    )
    result = backend.initiate_session("alice", "a@x.io", "enhanced")
    assert result["vendor_ref"] == "inq_test"
    assert result["session_url"] == "https://x"
    assert result["status"] == "INITIATED"


# ── Backward-compat: enhanced_template_id is optional ────────

def test_constructor_without_enhanced_template_still_works():
    """sp849 callers that pass only template_id must keep working."""
    handler, calls = _capture_handler()
    backend = PersonaHttpBackend(
        api_key="k", template_id=_BASE_TMPL,
        client=_mock_client(handler),
    )
    # basic works
    backend.initiate_session("alice", "a@x.io", "basic")
    assert calls[0]["data"]["attributes"]["inquiry-template-id"] == (
        _BASE_TMPL
    )


# ── from_env reads PERSONA_ENHANCED_TEMPLATE_ID ──────────────

def test_from_env_reads_enhanced_template(monkeypatch):
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", _BASE_TMPL)
    monkeypatch.setenv("PERSONA_ENHANCED_TEMPLATE_ID", _ENH_TMPL)
    backend = persona_from_env()
    assert backend is not None
    assert backend._enhanced_template_id == _ENH_TMPL
    assert backend._template_id == _BASE_TMPL


def test_from_env_enhanced_template_optional(monkeypatch):
    """No PERSONA_ENHANCED_TEMPLATE_ID → backend still constructs;
    enhanced inquiries fall back to base."""
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", _BASE_TMPL)
    monkeypatch.delenv("PERSONA_ENHANCED_TEMPLATE_ID", raising=False)
    backend = persona_from_env()
    assert backend is not None
    assert backend._enhanced_template_id is None


def test_from_env_still_requires_base_template(monkeypatch):
    """Base template remains required — enhanced is additive."""
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.delenv("PERSONA_TEMPLATE_ID", raising=False)
    monkeypatch.setenv("PERSONA_ENHANCED_TEMPLATE_ID", _ENH_TMPL)
    backend = persona_from_env()
    assert backend is None  # no base template → None
