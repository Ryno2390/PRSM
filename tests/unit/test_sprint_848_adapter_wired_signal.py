"""Sprint 848 — adapter_wired signal across Phase 5 fiat surface clients.

Closes a UX gap surfaced during sp848 dogfood walkthrough: the three
Phase 5 commissioned-vendor clients (KYCClient, PaymasterClient,
CoinbaseWaasClient) all expose ``is_commissioned()`` that returns
``True`` iff vendor env vars are present, but their actual ``initiate``
/``provision_wallet`` / ``sponsor_user_op`` methods short-circuit to
PENDING_COMMISSION when ``self._backend is None``.

The result during dogfood:
  curl /wallet/kyc/status        → {"commissioned": true, ...}
  curl -X POST /wallet/kyc/initiate → {"status": "PENDING_COMMISSION"}

Operators reasonably read "commissioned: true" as "ready to execute";
the truth is "credentials wired, but the SDK adapter hasn't been
plugged in yet". Sp848 surfaces the gap as a second boolean.

Contract added (this sprint):

  client.adapter_wired() -> bool
    True iff ``self._backend is not None``. Distinguished from
    is_commissioned() so operators can see both signals.

  Status endpoints expose ``adapter_wired`` alongside ``commissioned``:
    /wallet/kyc/status        → {commissioned, adapter_wired, ...}
    /wallet/paymaster/status  → {commissioned, adapter_wired, ...}
    /wallet/waas/status       → {commissioned, adapter_wired, ...}

Pin tests defend:
  - The method exists on all 3 clients
  - Returns False when constructed with backend=None
  - Returns True when constructed with a non-None backend
  - is_commissioned() behavior unchanged (env-var-only check)
  - The two signals are orthogonal — env-wired-no-backend AND
    backend-wired-no-env are both valid states
"""
from __future__ import annotations

from prsm.economy.web3.kyc_client import KYCClient
from prsm.economy.web3.paymaster_client import PaymasterClient
from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient


class _FakeKYCBackend:
    def initiate_session(self, user_id, email, level):
        return {"vendor_ref": "fake_ref", "session_url": "https://x"}


class _FakePaymasterBackend:
    def estimate_gas(self, user_op):
        return {"gas_estimate_wei": 0}

    def submit_sponsored(self, user_op):
        return {"tx_hash": "0xfa", "sponsor_amount_wei": 0}


class _FakeWaasBackend:
    def create_wallet(self, user_id, email):
        return {"wallet_id": "w_fake", "address": "0xfa"}


# ── KYCClient ────────────────────────────────────────────────

def test_kyc_adapter_wired_false_without_backend():
    c = KYCClient(vendor="persona", api_key="key", backend=None)
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False


def test_kyc_adapter_wired_true_with_backend():
    c = KYCClient(
        vendor="persona", api_key="key", backend=_FakeKYCBackend(),
    )
    assert c.is_commissioned() is True
    assert c.adapter_wired() is True


def test_kyc_signals_orthogonal_backend_without_env():
    """Backend wired but no API key. is_commissioned False;
    adapter_wired True. Operator sees an honest 'half configured'
    state instead of a silent fail."""
    c = KYCClient(vendor=None, api_key=None, backend=_FakeKYCBackend())
    assert c.is_commissioned() is False
    assert c.adapter_wired() is True


# ── PaymasterClient ──────────────────────────────────────────

def test_paymaster_adapter_wired_false_without_backend():
    c = PaymasterClient(
        endpoint="https://x", api_key="k", backend=None,
    )
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False


def test_paymaster_adapter_wired_true_with_backend():
    c = PaymasterClient(
        endpoint="https://x", api_key="k",
        backend=_FakePaymasterBackend(),
    )
    assert c.is_commissioned() is True
    assert c.adapter_wired() is True


def test_paymaster_spend_summary_exposes_adapter_wired():
    """spend_summary() is what /wallet/paymaster/status returns.
    Must surface adapter_wired so operators see both signals."""
    c = PaymasterClient(
        endpoint="https://x", api_key="k", backend=None,
    )
    summary = c.spend_summary()
    assert "adapter_wired" in summary
    assert summary["adapter_wired"] is False
    assert summary["commissioned"] is True

    c2 = PaymasterClient(
        endpoint="https://x", api_key="k",
        backend=_FakePaymasterBackend(),
    )
    summary2 = c2.spend_summary()
    assert summary2["adapter_wired"] is True
    assert summary2["commissioned"] is True


# ── CoinbaseWaasClient ───────────────────────────────────────

def test_waas_adapter_wired_false_without_backend():
    c = CoinbaseWaaSClient(
        api_key_name="n", api_key_private="p", backend=None,
    )
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False


def test_waas_adapter_wired_true_with_backend():
    c = CoinbaseWaaSClient(
        api_key_name="n", api_key_private="p",
        backend=_FakeWaasBackend(),
    )
    assert c.is_commissioned() is True
    assert c.adapter_wired() is True


# ── Cross-client invariant ───────────────────────────────────

def test_all_three_clients_expose_adapter_wired_method():
    """Defensive: if a future sprint splits the WaaS or KYC client
    further, every commissioned-vendor adapter MUST keep this
    contract so the /status endpoints stay honest."""
    kyc = KYCClient(vendor=None, api_key=None, backend=None)
    pay = PaymasterClient(endpoint=None, api_key=None, backend=None)
    waas = CoinbaseWaaSClient(
        api_key_name=None, api_key_private=None, backend=None,
    )
    for client in (kyc, pay, waas):
        assert callable(getattr(client, "adapter_wired", None)), (
            f"{type(client).__name__} missing adapter_wired() — "
            "Phase 5 status endpoints will lie about readiness."
        )
        assert client.adapter_wired() is False
