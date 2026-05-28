"""Sprint 852 — Vendor webhook payload normalizer.

The sp283 webhook receiver verifies HMAC signatures, enforces
timestamp freshness, and dedupes replays — but assumes a flat
``{user_id, status, vendor_ref}`` body shape. Real vendor payloads
are JSON:API-style envelopes; this module translates each vendor's
native shape into PRSM's canonical shape so the existing
``kyc.update_status`` call works unchanged.

Persona example (real webhook, sandbox):

  {
    "data": {
      "type": "event",
      "id": "evt_...",
      "attributes": {
        "name": "inquiry.approved",
        "payload": {
          "data": {
            "id": "inq_A4RYNX...",
            "attributes": {
              "reference-id": "alice",
              "status": "approved",
              ...
            }
          }
        }
      }
    }
  }

Normalized:
  {"user_id": "alice", "status": "VERIFIED",
   "vendor_ref": "inq_A4RYNX...",
   "event_name": "inquiry.approved"}

Onfido + Plaid stubs return None (deferred). When a normalizer
returns None, the webhook handler falls through to the legacy
flat-shape parser — preserving sprint-280 behavior for any
operator already shimming externally.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# Persona event name → PRSM KYC status (canonical).
# See prsm/economy/web3/kyc_client.py for status constants.
_PERSONA_EVENT_TO_STATUS: Dict[str, str] = {
    "inquiry.created": "INITIATED",
    "inquiry.started": "INITIATED",
    "inquiry.transitioned": "PENDING",
    "inquiry.completed": "PENDING",
    "inquiry.approved": "VERIFIED",
    "inquiry.marked-approved": "VERIFIED",
    "inquiry.declined": "REJECTED",
    "inquiry.marked-declined": "REJECTED",
    "inquiry.expired": "EXPIRED",
    "inquiry.failed": "REJECTED",
}


def normalize_persona_webhook(
    body: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Translate a Persona webhook payload to PRSM canonical shape.

    Returns None when the payload doesn't look like a Persona
    event (so the handler can fall through to legacy parsing).
    Raises ValueError when the payload IS a Persona event but
    missing required fields — the handler surfaces these as 422.
    """
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, dict):
        return None
    if data.get("type") != "event":
        return None
    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        return None
    event_name = attrs.get("name")
    if not isinstance(event_name, str):
        return None

    # At this point the payload IS a Persona event — any missing
    # downstream fields are a hard error (not silent fallthrough).
    inner = attrs.get("payload")
    if not isinstance(inner, dict):
        raise ValueError(
            "persona event missing attributes.payload",
        )
    inner_data = inner.get("data")
    if not isinstance(inner_data, dict):
        raise ValueError(
            "persona event missing attributes.payload.data",
        )
    inquiry_id = inner_data.get("id")
    inner_attrs = inner_data.get("attributes")
    if not isinstance(inner_attrs, dict):
        raise ValueError(
            "persona event missing payload.data.attributes",
        )
    reference_id = inner_attrs.get("reference-id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError(
            "persona event missing reference-id",
        )

    # Map event name → PRSM canonical status. Unknown events get
    # PENDING (defensive: rather than dropping the update, we log
    # the event happened + leave the operator a record to look at).
    canonical_status = _PERSONA_EVENT_TO_STATUS.get(
        event_name, "PENDING",
    )

    return {
        "user_id": reference_id,
        "status": canonical_status,
        "vendor_ref": inquiry_id,
        "event_name": event_name,
    }


def normalize_webhook_payload(
    vendor: str, body: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Dispatch normalizer by vendor name. Returns None when no
    vendor-specific normalizer is registered (handler falls
    through to legacy flat parsing)."""
    vendor_lower = (vendor or "").strip().lower()
    if vendor_lower == "persona":
        return normalize_persona_webhook(body)
    # Onfido + Plaid normalizers are deferred follow-ons.
    return None
