"""Sprint 852 — Vendor webhook payload normalizer pin tests.

Defends the translation from Persona's JSON:API event envelope
into PRSM's canonical {user_id, status, vendor_ref, event_name}
shape that the sp283 webhook receiver consumes.

Pin tests:
  - Persona inquiry.approved → VERIFIED
  - Persona inquiry.declined → REJECTED
  - Persona inquiry.expired → EXPIRED
  - Persona inquiry.completed → PENDING (not VERIFIED — Persona's
    "completed" means the user submitted, not that they passed)
  - Persona inquiry.marked-* admin overrides map correctly
  - Unknown Persona event → PENDING (defensive — never drop)
  - Reference-id maps to user_id; inquiry id maps to vendor_ref
  - Non-Persona payload → None (handler falls through to legacy)
  - Persona event missing payload → ValueError (handler 422s)
  - Persona event missing reference-id → ValueError
  - Generic vendor dispatch routes to correct normalizer
  - Onfido + Plaid normalizers return None (deferred)
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.kyc_webhook_normalizer import (
    normalize_persona_webhook,
    normalize_webhook_payload,
)


def _persona_event(event_name: str, **kwargs) -> dict:
    """Build a realistic Persona JSON:API event envelope."""
    inquiry_id = kwargs.pop("inquiry_id", "inq_TEST123")
    reference_id = kwargs.pop("reference_id", "user_42")
    inner_status = kwargs.pop("inner_status", "approved")
    return {
        "data": {
            "type": "event",
            "id": "evt_xyz",
            "attributes": {
                "name": event_name,
                "payload": {
                    "data": {
                        "id": inquiry_id,
                        "type": "inquiry",
                        "attributes": {
                            "reference-id": reference_id,
                            "status": inner_status,
                        },
                    }
                },
            },
        }
    }


# ── Persona event → PRSM status mapping ──────────────────────

def test_persona_approved_maps_to_verified():
    r = normalize_persona_webhook(_persona_event("inquiry.approved"))
    assert r["status"] == "VERIFIED"


def test_persona_marked_approved_maps_to_verified():
    """Admin-override path (operator manually approves in Persona
    dashboard) must also flip the PRSM record."""
    r = normalize_persona_webhook(
        _persona_event("inquiry.marked-approved"),
    )
    assert r["status"] == "VERIFIED"


def test_persona_declined_maps_to_rejected():
    r = normalize_persona_webhook(_persona_event("inquiry.declined"))
    assert r["status"] == "REJECTED"


def test_persona_marked_declined_maps_to_rejected():
    r = normalize_persona_webhook(
        _persona_event("inquiry.marked-declined"),
    )
    assert r["status"] == "REJECTED"


def test_persona_expired_maps_to_expired():
    r = normalize_persona_webhook(_persona_event("inquiry.expired"))
    assert r["status"] == "EXPIRED"


def test_persona_failed_maps_to_rejected():
    """failed != declined in Persona's vocabulary (failed = system
    error, declined = manual review reject) but both should
    PREVENT the user from acting on KYC-gated flows."""
    r = normalize_persona_webhook(_persona_event("inquiry.failed"))
    assert r["status"] == "REJECTED"


def test_persona_completed_maps_to_pending():
    """The semantic subtlety: 'completed' in Persona = user
    submitted; the actual approve/decline comes in a later event.
    We DON'T treat completed as VERIFIED — would let users transact
    before vendor review finished."""
    r = normalize_persona_webhook(_persona_event("inquiry.completed"))
    assert r["status"] == "PENDING"


def test_persona_unknown_event_falls_back_to_pending():
    """Defensive: a Persona event we don't recognize still records
    SOMETHING so the operator sees activity in the KYC store
    rather than the event being silently dropped."""
    r = normalize_persona_webhook(_persona_event("inquiry.future-event"))
    assert r["status"] == "PENDING"


# ── Field mapping ────────────────────────────────────────────

def test_reference_id_maps_to_user_id():
    r = normalize_persona_webhook(
        _persona_event("inquiry.approved", reference_id="alice_99"),
    )
    assert r["user_id"] == "alice_99"


def test_inquiry_id_maps_to_vendor_ref():
    r = normalize_persona_webhook(
        _persona_event("inquiry.approved", inquiry_id="inq_xyz999"),
    )
    assert r["vendor_ref"] == "inq_xyz999"


def test_event_name_preserved_in_output():
    """Operator-side observability: event_name is preserved so
    log lines + audit trails can record the exact vendor event,
    not just the canonical PRSM status."""
    r = normalize_persona_webhook(_persona_event("inquiry.approved"))
    assert r["event_name"] == "inquiry.approved"


# ── Fall-through semantics ───────────────────────────────────

def test_non_persona_envelope_returns_none():
    """A body that's just a dict with no JSON:API event structure
    is NOT a Persona event — return None so the handler can fall
    through to legacy flat-shape parsing (preserves sp280
    pass-through path for external proxies)."""
    assert normalize_persona_webhook({}) is None
    assert normalize_persona_webhook(
        {"user_id": "alice", "status": "VERIFIED"},
    ) is None


def test_non_event_data_type_returns_none():
    """data.type != 'event' isn't ours — fall through."""
    body = {"data": {"type": "inquiry", "attributes": {}}}
    assert normalize_persona_webhook(body) is None


def test_data_attributes_missing_returns_none():
    """If the envelope shape doesn't include data.attributes.name,
    we can't extract event metadata — fall through."""
    body = {"data": {"type": "event"}}
    assert normalize_persona_webhook(body) is None


def test_non_dict_input_returns_none():
    """Defensive: caller passes a list or None."""
    assert normalize_persona_webhook(None) is None
    assert normalize_persona_webhook([]) is None
    assert normalize_persona_webhook("string") is None


# ── Hard error paths ─────────────────────────────────────────

def test_persona_event_missing_payload_raises():
    """A confirmed Persona event MUST have attributes.payload —
    if it doesn't, that's a vendor-protocol violation, surface
    as 422 not silent fallthrough."""
    body = {
        "data": {
            "type": "event",
            "attributes": {"name": "inquiry.approved"},
        }
    }
    with pytest.raises(ValueError) as exc:
        normalize_persona_webhook(body)
    assert "payload" in str(exc.value)


def test_persona_event_missing_reference_id_raises():
    """Without reference-id we have no way to know which user the
    event refers to — 422."""
    bad = _persona_event("inquiry.approved")
    bad["data"]["attributes"]["payload"]["data"]["attributes"].pop(
        "reference-id",
    )
    with pytest.raises(ValueError) as exc:
        normalize_persona_webhook(bad)
    assert "reference-id" in str(exc.value)


# ── Dispatcher ───────────────────────────────────────────────

def test_dispatcher_routes_persona():
    body = _persona_event("inquiry.approved")
    r = normalize_webhook_payload("persona", body)
    assert r is not None
    assert r["status"] == "VERIFIED"


def test_dispatcher_persona_case_insensitive():
    r = normalize_webhook_payload(
        "PERSONA", _persona_event("inquiry.approved"),
    )
    assert r is not None


def test_dispatcher_onfido_returns_none():
    """Onfido normalizer is deferred; dispatcher returns None so
    the handler falls through to legacy parsing."""
    assert normalize_webhook_payload("onfido", {"any": "shape"}) is None


def test_dispatcher_plaid_returns_none():
    assert normalize_webhook_payload("plaid", {"any": "shape"}) is None


def test_dispatcher_unknown_vendor_returns_none():
    assert normalize_webhook_payload(
        "fake-vendor", _persona_event("inquiry.approved"),
    ) is None
