"""Sprint 849 — Persona KYC vendor HTTP backend.

Implements the ``_KYCBackend`` Protocol used by ``KYCClient`` against
Persona's Inquiries API
(https://docs.withpersona.com/reference/create-an-inquiry).

Two-call flow per ``initiate_session``:

  1. POST ``/inquiries`` with reference-id + email → returns inquiry id
  2. POST ``/inquiries/{id}/generate-one-time-link`` → user-facing URL

The one-time link is the session URL handed back to KYCClient.initiate()
and stored on the ``KYCRecord``. The user visits it, Persona runs ID
verification + selfie liveness, then fires our
``/wallet/kyc/webhook/persona`` endpoint with the inquiry-completed
event (signed with ``PERSONA_WEBHOOK_SECRET``).

Constructor accepts an injected ``httpx.Client`` so tests can wire
``httpx.MockTransport`` (matching the convention in
``tests/unit/test_http_aggregate_transport.py``). Production code
omits the kwarg and gets a fresh httpx.Client with a 30s timeout.

``from_env()`` returns ``None`` when required env vars are missing —
``KYCClient.from_env()`` uses that signal to gracefully fall back to
the un-backed PENDING_COMMISSION pattern rather than raising.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_PERSONA_BASE_URL = "https://api.withpersona.com/api/v1"
_PERSONA_API_VERSION = "2023-01-05"
_PERSONA_TIMEOUT_SECONDS = 30.0


class PersonaHttpBackend:
    """Persona Inquiries API backend implementing _KYCBackend Protocol."""

    def __init__(
        self,
        api_key: str,
        template_id: str,
        *,
        enhanced_template_id: Optional[str] = None,
        base_url: str = _PERSONA_BASE_URL,
        api_version: str = _PERSONA_API_VERSION,
        client: Any = None,  # injected httpx.Client for tests
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not template_id:
            raise ValueError("template_id is required")
        self._api_key = api_key
        self._template_id = template_id
        # Sp883 — separate Persona template for enhanced KYC (Tier
        # 2/3: proof-of-address + source-of-funds). Optional: when
        # unset, enhanced inquiries fall back to the base template
        # + a warning rather than 500ing the user's upgrade attempt.
        self._enhanced_template_id = enhanced_template_id or None
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        if client is None:
            import httpx
            self._client = httpx.Client(
                timeout=_PERSONA_TIMEOUT_SECONDS,
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Persona-Version": self._api_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def initiate_session(
        self, user_id: str, email: str, level: str,
    ) -> Dict[str, Any]:
        """Create a Persona inquiry + return a session URL.

        Returns a dict shaped for ``KYCClient.initiate`` consumption:
          - ``vendor_ref``: Persona inquiry id (``inq_...``)
          - ``session_url``: one-time link the user visits
          - ``status``: KYC_STATUS_INITIATED (Persona returns the
            inquiry in "created" state; the webhook flips it later)
        """
        # Sp883 — select the inquiry template by KYC level.
        # enhanced → enhanced template (proof-of-address +
        # source-of-funds); basic / anything else → base template.
        # If enhanced is requested but no enhanced template is
        # configured, fall back to base + warn (graceful degrade).
        template_id = self._template_id
        if level == "enhanced":
            if self._enhanced_template_id:
                template_id = self._enhanced_template_id
            else:
                logger.warning(
                    "persona: enhanced KYC requested for user_id=%s "
                    "but PERSONA_ENHANCED_TEMPLATE_ID is unset — "
                    "falling back to the base template (proof-of-"
                    "address / source-of-funds will NOT be "
                    "collected). Create an enhanced inquiry template "
                    "in Persona + set the env var to enable Tier 2/3.",
                    user_id,
                )

        # Step 1: create inquiry
        create_body = {
            "data": {
                "attributes": {
                    "inquiry-template-id": template_id,
                    "reference-id": user_id,
                    "fields": {"email-address": email},
                }
            }
        }
        resp = self._client.post(
            f"{self._base_url}/inquiries",
            headers=self._headers(),
            json=create_body,
        )
        resp.raise_for_status()
        payload = resp.json()
        inquiry_id = payload.get("data", {}).get("id")
        if not inquiry_id:
            raise RuntimeError(
                f"persona inquiry create returned no id: "
                f"{payload!r}"
            )

        # Step 2: generate one-time link (the user-facing URL)
        link_resp = self._client.post(
            f"{self._base_url}/inquiries/"
            f"{inquiry_id}/generate-one-time-link",
            headers=self._headers(),
        )
        link_resp.raise_for_status()
        link_payload = link_resp.json()
        session_url = (
            link_payload.get("data", {})
            .get("attributes", {})
            .get("link")
        )
        if not session_url:
            # Some Persona configs don't issue one-time links;
            # fall back to the hosted-flow URL pattern.
            session_url = (
                f"https://withpersona.com/verify?"
                f"inquiry-id={inquiry_id}"
            )
            logger.warning(
                "persona one-time link missing for inquiry %s; "
                "falling back to hosted-flow URL",
                inquiry_id,
            )

        return {
            "vendor_ref": inquiry_id,
            "session_url": session_url,
            "status": "INITIATED",
        }


def from_env(
    *,
    api_key: Optional[str] = None,
    template_id: Optional[str] = None,
    enhanced_template_id: Optional[str] = None,
    client: Any = None,
) -> Optional["PersonaHttpBackend"]:
    """Construct a PersonaHttpBackend from env, or None when missing.

    Returns None (rather than raising) when ``KYC_VENDOR_API_KEY`` or
    ``PERSONA_TEMPLATE_ID`` are absent, so ``KYCClient.from_env()`` can
    gracefully fall back to the un-backed PENDING_COMMISSION pattern.

    Sp883 — ``PERSONA_ENHANCED_TEMPLATE_ID`` is OPTIONAL: when set it
    routes ``level="enhanced"`` inquiries (Tier 2/3) to a separate
    proof-of-address / source-of-funds template. When unset, enhanced
    inquiries fall back to the base template with a warning. The base
    template remains required — enhanced is purely additive.
    """
    api_key = api_key or os.environ.get("KYC_VENDOR_API_KEY")
    template_id = (
        template_id or os.environ.get("PERSONA_TEMPLATE_ID")
    )
    enhanced_template_id = (
        enhanced_template_id
        or os.environ.get("PERSONA_ENHANCED_TEMPLATE_ID")
    )
    if not api_key or not template_id:
        return None
    return PersonaHttpBackend(
        api_key=api_key,
        template_id=template_id,
        enhanced_template_id=enhanced_template_id,
        client=client,
    )
