"""Sprint 881 — Phase 5 fiat onboarding loop end-to-end integration.

Exercises the ENTIRE Phase 5 fiat-onboarding chain in-process with
injected fakes (no real network calls), proving the stages wire
together as the Vision §8/§11/§14 onboarding flow claims:

  1. KYCClient.initiate(...) with a fake Persona backend → INITIATED
  2. Persona JSON:API inquiry.approved event envelope
     → normalize_webhook_payload("persona", body)
     → canonical {user_id, status=VERIFIED, vendor_ref}
  3. KYCClient.update_status(user_id, VERIFIED) → VERIFIED record
  4. maybe_auto_provision_waas(...) → WaasWalletRecord PROVISIONED
     with a Base address
  5. OnrampFunnel.record_intent(..., destination=<provisioned addr>)
     → INTENT_RECORDED
  6. OnrampFunnel.sweep(balance_reader=<fake, usdc >= expected*0.95>,
     on_confirmed=make_on_confirmed_callback(funnel,
       aerodrome_client=<configured fake>, ftns_address,
       completion_notifier=<httpx-mock-backed notifier>))
     → CONFIRMED + swap_envelope attached + outbound webhook fired
  7. Cross-chain assertions on every artifact.

httpx conftest interference:
  tests/conftest.py has an autouse `mock_http_requests` fixture that
  does `patch('httpx.Client')`. We capture the REAL classes at module
  import (before any fixture runs) and an autouse fixture restores
  httpx.Client / httpx.MockTransport / httpx.Response per-test — the
  same proven pattern as test_sprint_874.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import pytest


# ── Capture real httpx before conftest autouse fixtures mock it ──
_real_Client = httpx.Client
_real_MockTransport = httpx.MockTransport
_real_Response = httpx.Response


@pytest.fixture(autouse=True)
def _restore_real_httpx(monkeypatch):
    """Undo tests/conftest.py's `patch('httpx.Client')` for the body
    of each test so the MockTransport-backed clients actually fire."""
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", _real_MockTransport)
    monkeypatch.setattr(httpx, "Response", _real_Response)
    yield


from prsm.economy.web3.kyc_client import (  # noqa: E402
    KYC_LEVEL_BASIC,
    KYC_STATUS_INITIATED,
    KYC_STATUS_REJECTED,
    KYC_STATUS_VERIFIED,
    KYCClient,
)
from prsm.economy.web3.kyc_webhook_normalizer import (  # noqa: E402
    normalize_webhook_payload,
)
from prsm.economy.web3.kyc_to_waas_orchestrator import (  # noqa: E402
    maybe_auto_provision_waas,
)
from prsm.economy.web3.coinbase_waas_client import (  # noqa: E402
    CoinbaseWaaSClient,
)
from prsm.economy.web3.onramp_funnel import (  # noqa: E402
    STATUS_CONFIRMED,
    STATUS_INTENT_RECORDED,
    OnrampFunnel,
)
from prsm.economy.web3.onramp_to_swap_orchestrator import (  # noqa: E402
    make_on_confirmed_callback,
)
from prsm.economy.web3.onramp_completion_notifier import (  # noqa: E402
    OnrampCompletionNotifier,
)


# ── Test constants ───────────────────────────────────────────────
_USER_ID = "alice"
_EMAIL = "alice@example.io"
_INQUIRY_ID = "inq_A4RYNX123"
_SESSION_URL = "https://withpersona.com/verify/onetime/abc123"
_PROVISIONED_ADDRESS = "0x" + "11" * 20  # 0x1111...11 (40 hex)
_WALLET_ID = "wallet_test_xyz"
_NETWORK = "base-mainnet"
_FTNS_ADDRESS = "0x" + "ab" * 20
_EXPECTED_USD = 100.0
# Coinbase takes a fee; the user actually RECEIVES slightly less, but
# still >= expected * 0.95 (the funnel CONFIRM threshold).
_USDC_RECEIVED = 98.5
_USDC_RECEIVED_UNITS = int(_USDC_RECEIVED * 10 ** 6)
# Mock Aerodrome quote: 98.5 USDC → ~4925 FTNS (18 decimals).
_QUOTE_AMOUNT_OUT_UNITS = 4925 * 10 ** 18
_QUOTE_PRICE_IMPACT_BPS = 12
_QUOTE_FEE_BPS = 30
_WEBHOOK_URL = "https://operator.example.com/onramp-completion"


# ── Module-level fakes ───────────────────────────────────────────


class _FakePersonaBackend:
    """Drop-in _KYCBackend: initiate_session returns the
    {vendor_ref, session_url, status} contract WITHOUT httpx."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def initiate_session(
        self, user_id: str, email: str, level: str,
    ) -> Dict[str, Any]:
        self.calls.append((user_id, email, level))
        return {
            "vendor_ref": _INQUIRY_ID,
            "session_url": _SESSION_URL,
            "status": "INITIATED",
        }


class _FakeWaasBackend:
    """Drop-in _WaasBackend: create_wallet returns the
    {wallet_id, address, network} contract."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create_wallet(
        self, user_id: str, email: str,
    ) -> Dict[str, Any]:
        self.calls.append((user_id, email))
        return {
            "wallet_id": _WALLET_ID,
            "address": _PROVISIONED_ADDRESS,
            "network": _NETWORK,
        }


@dataclass
class _FakeQuote:
    """Duck-typed AerodromeQuote: the envelope builder only reads
    .amount_out, .price_impact_bps, .fee_bps."""

    amount_out: int = _QUOTE_AMOUNT_OUT_UNITS
    price_impact_bps: int = _QUOTE_PRICE_IMPACT_BPS
    fee_bps: int = _QUOTE_FEE_BPS


class _FakeAerodromeClient:
    """Configured Aerodrome client returning a usable quote."""

    def __init__(self, *, configured: bool = True) -> None:
        self._configured = configured
        self.quote_calls: list[dict] = []

    def is_configured(self) -> bool:
        return self._configured

    def quote_swap(
        self,
        amount_in: int,
        token_in: str,
        pool_address: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[_FakeQuote]:
        if not self._configured:
            return None
        self.quote_calls.append(
            {"amount_in": amount_in, "token_in": token_in},
        )
        return _FakeQuote()


@dataclass
class _FakeBalance:
    """Duck-typed WalletBalances slice the sweep reads: .usdc +
    .usdc_units."""

    usdc: float
    usdc_units: int


class _FakeBalanceReader:
    """Returns a configured USDC balance ONLY for the address we
    provisioned; everything else reads zero (so a stray intent for a
    different address wouldn't spuriously confirm)."""

    def __init__(self, *, address: str, usdc: float) -> None:
        self._address = address
        self._usdc = usdc
        self.read_addresses: list[str] = []

    def get_balances(self, address: str) -> _FakeBalance:
        self.read_addresses.append(address)
        if address == self._address:
            return _FakeBalance(
                usdc=self._usdc,
                usdc_units=int(self._usdc * 10 ** 6),
            )
        return _FakeBalance(usdc=0.0, usdc_units=0)


def _mock_httpx_client(handler):
    """httpx.Client wired to a MockTransport handler. Uses the
    module-captured real classes so it survives the conftest patch."""
    return _real_Client(transport=_real_MockTransport(handler))


def _persona_approved_envelope(
    *, user_id: str = _USER_ID, inquiry_id: str = _INQUIRY_ID,
) -> Dict[str, Any]:
    """A Persona JSON:API `inquiry.approved` event envelope."""
    return {
        "data": {
            "type": "event",
            "id": "evt_xyz",
            "attributes": {
                "name": "inquiry.approved",
                "payload": {
                    "data": {
                        "id": inquiry_id,
                        "type": "inquiry",
                        "attributes": {
                            "reference-id": user_id,
                            "status": "approved",
                        },
                    },
                },
            },
        },
    }


def _persona_declined_envelope(
    *, user_id: str = _USER_ID, inquiry_id: str = _INQUIRY_ID,
) -> Dict[str, Any]:
    """A Persona `inquiry.declined` event → REJECTED."""
    env = _persona_approved_envelope(
        user_id=user_id, inquiry_id=inquiry_id,
    )
    env["data"]["attributes"]["name"] = "inquiry.declined"
    env["data"]["attributes"]["payload"]["data"]["attributes"][
        "status"
    ] = "declined"
    return env


# ── Builders for the wired stages ───────────────────────────────


def _build_kyc_client(tmp_path, *, backend) -> KYCClient:
    """Commissioned + backend-wired KYCClient over a tmp store."""
    return KYCClient(
        vendor="persona",
        api_key="kyc_test_key",
        persist_dir=tmp_path / "kyc",
        backend=backend,
    )


def _build_waas_client(tmp_path, *, backend) -> CoinbaseWaaSClient:
    """Commissioned (both keys) + backend-wired WaaS client."""
    return CoinbaseWaaSClient(
        api_key_name="cdp_key_name",
        api_key_private="cdp_key_private",
        network=_NETWORK,
        persist_dir=tmp_path / "waas",
        backend=backend,
    )


# ── (a) Happy-path full loop ─────────────────────────────────────


def test_full_loop_kyc_to_confirmed_with_swap_envelope(tmp_path):
    """The complete Phase 5 fiat onboarding loop, end-to-end."""
    # Stage 1 — KYC initiate via fake Persona backend → INITIATED.
    persona_backend = _FakePersonaBackend()
    kyc = _build_kyc_client(tmp_path, backend=persona_backend)
    assert kyc.is_commissioned() is True
    assert kyc.adapter_wired() is True

    rec = kyc.initiate(_USER_ID, _EMAIL, KYC_LEVEL_BASIC)
    assert rec.status == KYC_STATUS_INITIATED
    assert rec.vendor_ref == _INQUIRY_ID
    assert rec.session_url == _SESSION_URL
    assert persona_backend.calls == [
        (_USER_ID, _EMAIL, KYC_LEVEL_BASIC),
    ]
    old_status = rec.status

    # Stage 2 — Persona approved webhook → canonical normalized shape.
    normalized = normalize_webhook_payload(
        "persona", _persona_approved_envelope(),
    )
    assert normalized is not None
    assert normalized["user_id"] == _USER_ID
    assert normalized["status"] == KYC_STATUS_VERIFIED
    assert normalized["vendor_ref"] == _INQUIRY_ID
    assert normalized["event_name"] == "inquiry.approved"

    # Stage 3 — apply the transition → VERIFIED.
    updated = kyc.update_status(
        normalized["user_id"],
        normalized["status"],
        vendor_ref_update=normalized["vendor_ref"],
    )
    assert updated is not None
    assert updated.status == KYC_STATUS_VERIFIED
    assert updated.verified_at > 0
    assert kyc.is_verified(_USER_ID) is True

    # Stage 4 — KYC VERIFIED auto-provisions a WaaS wallet.
    waas_backend = _FakeWaasBackend()
    waas = _build_waas_client(tmp_path, backend=waas_backend)
    assert waas.is_commissioned() is True
    assert waas.adapter_wired() is True

    wallet = maybe_auto_provision_waas(
        waas_client=waas,
        user_id=updated.user_id,
        email=updated.email,
        new_status=updated.status,
        old_status=old_status,
    )
    assert wallet is not None
    assert wallet.status == "PROVISIONED"
    assert wallet.address == _PROVISIONED_ADDRESS
    assert wallet.wallet_id == _WALLET_ID
    assert wallet.network == _NETWORK
    assert waas_backend.calls == [(_USER_ID, _EMAIL)]

    # Stage 5 — record the onramp intent against the provisioned addr.
    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    intent = funnel.record_intent(
        user_id=_USER_ID,
        destination_address=wallet.address,
        expected_usd=_EXPECTED_USD,
        session_token="cb_session_token_xyz",
    )
    assert intent.status == STATUS_INTENT_RECORDED
    assert intent.destination_address == _PROVISIONED_ADDRESS

    # Stage 6 — sweep: USDC arrived above threshold → CONFIRMED,
    # swap envelope built, completion webhook fired.
    captured_posts: list[dict] = []

    def _webhook_handler(request):
        captured_posts.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": json.loads(request.content),
        })
        return httpx.Response(200, json={"ok": True})

    notifier = OnrampCompletionNotifier(
        url=_WEBHOOK_URL,
        log_dir=tmp_path / "deliveries",
        client=_mock_httpx_client(_webhook_handler),
    )
    aerodrome = _FakeAerodromeClient(configured=True)
    reader = _FakeBalanceReader(
        address=wallet.address, usdc=_USDC_RECEIVED,
    )
    on_confirmed = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=aerodrome,
        ftns_address=_FTNS_ADDRESS,
        completion_notifier=notifier,
    )

    summary = funnel.sweep(
        balance_reader=reader, on_confirmed=on_confirmed,
    )

    # Stage 7 — cross-chain assertions.
    # Sweep summary: exactly one new confirmation.
    assert summary["checked"] == 1
    assert summary["confirmed_new"] == 1
    assert summary["expired_new"] == 0

    # Funnel intent CONFIRMED with received amount recorded.
    confirmed = funnel.get_intent(intent.intent_id)
    assert confirmed.status == STATUS_CONFIRMED
    assert confirmed.usdc_received == _USDC_RECEIVED
    assert confirmed.usdc_received_units == _USDC_RECEIVED_UNITS
    assert confirmed.confirmed_at > 0

    # Aerodrome was quoted with amount_in derived from usdc_received
    # (NOT expected_usd) — the critical sp871 invariant.
    assert len(aerodrome.quote_calls) == 1
    assert aerodrome.quote_calls[0]["amount_in"] == (
        _USDC_RECEIVED_UNITS
    )

    # Swap envelope present + READY_FOR_SUBMISSION with correct
    # amount_in.
    env = confirmed.swap_envelope
    assert env is not None
    assert env["status"] == "READY_FOR_SUBMISSION"
    assert env["args"]["amountIn"] == _USDC_RECEIVED_UNITS
    assert env["quote"]["amount_in_usdc"] == _USDC_RECEIVED
    assert env["quote"]["amount_in_units"] == _USDC_RECEIVED_UNITS
    assert env["quote"]["amount_out_units"] == (
        _QUOTE_AMOUNT_OUT_UNITS
    )
    # amountOutMin = amount_out * (10000 - 100) / 10000 (1% slippage).
    expected_min = _QUOTE_AMOUNT_OUT_UNITS * (10_000 - 100) // 10_000
    assert env["args"]["amountOutMin"] == expected_min
    assert env["quote"]["price_impact_bps"] == _QUOTE_PRICE_IMPACT_BPS
    assert env["quote"]["fee_bps"] == _QUOTE_FEE_BPS
    assert env["destination_address"] == _PROVISIONED_ADDRESS
    assert env["intent_id"] == intent.intent_id

    # Completion webhook delivered exactly once with the right body.
    assert len(captured_posts) == 1
    body = captured_posts[0]["body"]
    assert captured_posts[0]["url"] == _WEBHOOK_URL
    assert body["event"] == "onramp.completion"
    assert body["intent_id"] == intent.intent_id
    assert body["user_id"] == _USER_ID
    assert body["destination_address"] == _PROVISIONED_ADDRESS
    assert body["usdc_received"] == _USDC_RECEIVED
    assert body["swap_envelope"] is not None
    assert body["swap_envelope"]["status"] == "READY_FOR_SUBMISSION"

    # Delivery persisted to the notifier's log dir.
    deliveries = notifier.list_deliveries()
    assert len(deliveries) == 1
    assert deliveries[0]["intent_id"] == intent.intent_id
    assert deliveries[0]["success"] is True
    assert deliveries[0]["status_code"] == 200


# ── (b) Loop halts on KYC rejection ──────────────────────────────


def test_loop_halts_when_kyc_rejected(tmp_path):
    """REJECTED KYC → no WaaS provision → no funnel intent."""
    persona_backend = _FakePersonaBackend()
    kyc = _build_kyc_client(tmp_path, backend=persona_backend)
    rec = kyc.initiate(_USER_ID, _EMAIL, KYC_LEVEL_BASIC)
    old_status = rec.status
    assert rec.status == KYC_STATUS_INITIATED

    # Persona declines the inquiry → REJECTED canonical status.
    normalized = normalize_webhook_payload(
        "persona", _persona_declined_envelope(),
    )
    assert normalized is not None
    assert normalized["status"] == KYC_STATUS_REJECTED

    updated = kyc.update_status(
        normalized["user_id"], normalized["status"],
        vendor_ref_update=normalized["vendor_ref"],
    )
    assert updated.status == KYC_STATUS_REJECTED
    assert kyc.is_verified(_USER_ID) is False

    # Auto-provision must NOT fire for a non-VERIFIED status.
    waas_backend = _FakeWaasBackend()
    waas = _build_waas_client(tmp_path, backend=waas_backend)
    wallet = maybe_auto_provision_waas(
        waas_client=waas,
        user_id=updated.user_id,
        email=updated.email,
        new_status=updated.status,
        old_status=old_status,
    )
    assert wallet is None
    assert waas_backend.calls == []  # backend never touched
    assert waas.get_wallet(_USER_ID) is None

    # No wallet ⇒ no destination address ⇒ no intent recorded.
    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    assert funnel.list_intents() == []
    assert funnel.summary()["total_intents"] == 0


# ── (c) Envelope deferred when pool unconfigured ─────────────────


def test_loop_envelope_deferred_when_pool_unconfigured(tmp_path):
    """Aerodrome pool ceremony pending → intent CONFIRMS, swap
    envelope stays None, completion webhook STILL fires (with a
    null swap_envelope) so downstream learns the USDC arrived."""
    # Fast-path KYC→VERIFIED→provision (already covered in (a)).
    persona_backend = _FakePersonaBackend()
    kyc = _build_kyc_client(tmp_path, backend=persona_backend)
    kyc.initiate(_USER_ID, _EMAIL, KYC_LEVEL_BASIC)
    kyc.update_status(_USER_ID, KYC_STATUS_VERIFIED)

    waas = _build_waas_client(tmp_path, backend=_FakeWaasBackend())
    wallet = maybe_auto_provision_waas(
        waas_client=waas,
        user_id=_USER_ID,
        email=_EMAIL,
        new_status=KYC_STATUS_VERIFIED,
        old_status=KYC_STATUS_INITIATED,
    )
    assert wallet.status == "PROVISIONED"

    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    intent = funnel.record_intent(
        user_id=_USER_ID,
        destination_address=wallet.address,
        expected_usd=_EXPECTED_USD,
        session_token="cb_session_token_xyz",
    )

    captured_posts: list[dict] = []

    def _webhook_handler(request):
        captured_posts.append(json.loads(request.content))
        return httpx.Response(200, json={})

    notifier = OnrampCompletionNotifier(
        url=_WEBHOOK_URL,
        log_dir=tmp_path / "deliveries",
        client=_mock_httpx_client(_webhook_handler),
    )
    # Pool ceremony pending → is_configured() == False.
    aerodrome = _FakeAerodromeClient(configured=False)
    reader = _FakeBalanceReader(
        address=wallet.address, usdc=_USDC_RECEIVED,
    )
    on_confirmed = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=aerodrome,
        ftns_address=_FTNS_ADDRESS,
        completion_notifier=notifier,
    )

    summary = funnel.sweep(
        balance_reader=reader, on_confirmed=on_confirmed,
    )
    assert summary["confirmed_new"] == 1

    confirmed = funnel.get_intent(intent.intent_id)
    # Intent still CONFIRMS — the on-chain USDC arrival is the source
    # of truth; pool config is orthogonal.
    assert confirmed.status == STATUS_CONFIRMED
    assert confirmed.usdc_received == _USDC_RECEIVED
    # Envelope deferred (unconfigured pool short-circuits the builder).
    assert confirmed.swap_envelope is None
    # quote_swap never called when unconfigured.
    assert aerodrome.quote_calls == []

    # Webhook STILL fires — carries a null swap_envelope.
    assert len(captured_posts) == 1
    assert captured_posts[0]["event"] == "onramp.completion"
    assert captured_posts[0]["intent_id"] == intent.intent_id
    assert captured_posts[0]["swap_envelope"] is None


# ── (d) HMAC signature when secret set ───────────────────────────


def test_completion_webhook_carries_signature_when_secret_set(
    tmp_path,
):
    """When PRSM_ONRAMP_COMPLETION_WEBHOOK_SECRET is configured, the
    completion POST carries an X-PRSM-Signature header that verifies
    against the raw body bytes."""
    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    intent = funnel.record_intent(
        user_id=_USER_ID,
        destination_address=_PROVISIONED_ADDRESS,
        expected_usd=_EXPECTED_USD,
        session_token="cb_session_token_xyz",
    )

    captured: list[dict] = []

    def _webhook_handler(request):
        captured.append({
            "headers": dict(request.headers),
            "content": request.content,  # raw bytes
        })
        return httpx.Response(200, json={})

    secret = "wbhsec_phase5_test"
    notifier = OnrampCompletionNotifier(
        url=_WEBHOOK_URL,
        secret=secret,
        log_dir=tmp_path / "deliveries",
        client=_mock_httpx_client(_webhook_handler),
    )
    aerodrome = _FakeAerodromeClient(configured=True)
    reader = _FakeBalanceReader(
        address=_PROVISIONED_ADDRESS, usdc=_USDC_RECEIVED,
    )
    on_confirmed = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=aerodrome,
        ftns_address=_FTNS_ADDRESS,
        completion_notifier=notifier,
    )

    funnel.sweep(balance_reader=reader, on_confirmed=on_confirmed)

    assert len(captured) == 1
    headers = captured[0]["headers"]
    # httpx lowercases header keys.
    assert "x-prsm-signature" in headers
    sig_value = headers["x-prsm-signature"]
    assert sig_value.startswith("t=")
    assert ",v1=" in sig_value

    # Recompute the HMAC over `<timestamp>.<body_bytes>` and verify.
    parts = dict(p.split("=", 1) for p in sig_value.split(","))
    assert "t" in parts and "v1" in parts
    body_bytes = captured[0]["content"]
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{parts['t']}.".encode("utf-8") + body_bytes,
        hashlib.sha256,
    ).hexdigest()
    assert parts["v1"] == expected

    # The delivery record marks the signature as attached.
    deliveries = notifier.list_deliveries()
    assert len(deliveries) == 1
    assert deliveries[0]["signature_attached"] is True
    assert deliveries[0]["success"] is True
