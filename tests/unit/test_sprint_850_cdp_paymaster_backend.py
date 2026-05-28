"""Sprint 850 — CDP Paymaster HTTP backend + PaymasterClient auto-wire.

Mirrors sp849's Persona work on the Paymaster track: closes the
(commissioned=True, adapter_wired=False) state by shipping a real
JSON-RPC backend against Coinbase Developer Platform's Paymaster
endpoint + auto-wiring it via ``PaymasterClient.from_env()``.

Pin tests:
  - estimate_gas issues pm_sponsorUserOperation + extracts wei
  - submit_sponsored issues both pm_sponsorUserOperation AND
    eth_sendUserOperation in order, merging paymaster fields
  - sponsor_amount_wei computed correctly from gas units * maxFeePerGas
  - _wei_from_hex handles "0x..." decimal-string + int + None
  - _CdpRpcError raised when JSON-RPC returns `error` field
  - HTTP 4xx surfaces as httpx.HTTPStatusError
  - from_env() returns None when endpoint missing
  - from_env() constructs backend when endpoint present (api_key
    optional, since CDP v2 URL has token baked in)
  - PaymasterClient.from_env() auto-attaches CDP backend when env set
  - PaymasterClient.from_env() respects explicit backend= override
  - Graceful fallback when CDP backend construction raises
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest


# Capture real httpx classes before conftest autouse mocks them
# (same pattern as sp849 / test_http_aggregate_transport.py).
_real_Client = httpx.Client
_real_MockTransport = httpx.MockTransport
_real_Response = httpx.Response
_real_HTTPStatusError = httpx.HTTPStatusError


@pytest.fixture(autouse=True)
def _restore_real_httpx_classes(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", _real_MockTransport)
    monkeypatch.setattr(httpx, "Response", _real_Response)
    monkeypatch.setattr(httpx, "HTTPStatusError", _real_HTTPStatusError)
    yield


from prsm.economy.web3.paymaster_cdp_backend import (
    CdpPaymasterBackend,
    _CdpRpcError,
    _wei_from_hex,
    from_env as cdp_from_env,
)
from prsm.economy.web3.paymaster_client import PaymasterClient


# ── Helpers ──────────────────────────────────────────────────

_TEST_ENDPOINT = (
    "https://api.developer.coinbase.com/rpc/v1/base/TESTTOKEN"
)
_TEST_USER_OP = {
    "sender": "0x" + "11" * 20,
    "nonce": "0x0",
    "callData": "0x",
    "callGasLimit": "0x" + format(100_000, "x"),
    "verificationGasLimit": "0x" + format(150_000, "x"),
    "preVerificationGas": "0x" + format(50_000, "x"),
    "maxFeePerGas": "0x" + format(2_000_000_000, "x"),  # 2 gwei
    "maxPriorityFeePerGas": "0x" + format(1_000_000_000, "x"),
    "signature": "0x" + "00" * 65,
}


def _mock_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── _wei_from_hex helper ─────────────────────────────────────

def test_wei_from_hex_handles_hex_string():
    assert _wei_from_hex("0x5208") == 21000


def test_wei_from_hex_handles_int():
    assert _wei_from_hex(42) == 42


def test_wei_from_hex_handles_none_with_default():
    assert _wei_from_hex(None) == 0
    assert _wei_from_hex(None, default=99) == 99


def test_wei_from_hex_handles_decimal_string():
    assert _wei_from_hex("12345") == 12345


def test_wei_from_hex_handles_invalid_string():
    """Bad input → default, not crash."""
    assert _wei_from_hex("not-a-number") == 0


# ── CdpPaymasterBackend.estimate_gas ─────────────────────────

def test_estimate_gas_issues_pm_sponsor_user_operation():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert body["method"] == "pm_sponsorUserOperation"
        return httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "callGasLimit": "0x186a0",  # 100000
                "verificationGasLimit": "0x249f0",  # 150000
                "preVerificationGas": "0xc350",  # 50000
                "paymasterVerificationGasLimit": "0x186a0",  # 100000
                "paymasterPostOpGasLimit": "0x186a0",  # 100000
                "paymaster": "0x" + "ab" * 20,
                "paymasterData": "0x",
            },
        })

    backend = CdpPaymasterBackend(
        endpoint=_TEST_ENDPOINT, client=_mock_transport(handler),
    )
    result = backend.estimate_gas(_TEST_USER_OP)
    # Total gas units = 100k + 150k + 50k + 100k + 100k = 500k
    # maxFeePerGas = 2 gwei = 2e9 wei
    # Total = 500k * 2e9 = 1e15 wei
    assert result["gas_estimate_wei"] == 500_000 * 2_000_000_000
    assert len(calls) == 1


def test_estimate_gas_jsonrpc_envelope_correct():
    """JSON-RPC body must include method, params, id, jsonrpc."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.update(body)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"], "result": {},
        })

    backend = CdpPaymasterBackend(
        endpoint=_TEST_ENDPOINT, client=_mock_transport(handler),
    )
    backend.estimate_gas(_TEST_USER_OP)
    assert captured["jsonrpc"] == "2.0"
    assert captured["method"] == "pm_sponsorUserOperation"
    assert isinstance(captured["params"], list)
    assert captured["params"][0] == _TEST_USER_OP
    assert captured["params"][1].startswith("0x")  # entry point


# ── CdpPaymasterBackend.submit_sponsored ─────────────────────

def test_submit_sponsored_two_step_happy_path():
    """pm_sponsorUserOperation THEN eth_sendUserOperation, with
    paymaster fields merged into the user_op between calls."""
    method_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method_calls.append(body["method"])
        if body["method"] == "pm_sponsorUserOperation":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {
                    "callGasLimit": "0x186a0",
                    "verificationGasLimit": "0x249f0",
                    "preVerificationGas": "0xc350",
                    "paymasterVerificationGasLimit": "0x186a0",
                    "paymasterPostOpGasLimit": "0x186a0",
                    "paymaster": "0x" + "ab" * 20,
                    "paymasterData": "0xdeadbeef",
                },
            })
        if body["method"] == "eth_sendUserOperation":
            # Verify merged user op has paymaster fields
            merged = body["params"][0]
            assert merged["paymaster"] == "0x" + "ab" * 20
            assert merged["paymasterData"] == "0xdeadbeef"
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": "0xfeedfeedfeed",
            })
        return httpx.Response(400, json={"error": "unmocked"})

    backend = CdpPaymasterBackend(
        endpoint=_TEST_ENDPOINT, client=_mock_transport(handler),
    )
    result = backend.submit_sponsored(_TEST_USER_OP)
    assert method_calls == [
        "pm_sponsorUserOperation",
        "eth_sendUserOperation",
    ]
    assert result["user_op_hash"] == "0xfeedfeedfeed"
    assert result["tx_hash"] is None  # async via receipt poll
    # sponsor_amount = 500k * 2e9 = 1e15
    assert result["sponsor_amount_wei"] == 500_000 * 2_000_000_000


def test_submit_sponsored_raises_on_jsonrpc_error():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"],
            "error": {"code": -32000, "message": "policy reject"},
        })

    backend = CdpPaymasterBackend(
        endpoint=_TEST_ENDPOINT, client=_mock_transport(handler),
    )
    with pytest.raises(_CdpRpcError) as exc:
        backend.submit_sponsored(_TEST_USER_OP)
    assert "policy reject" in str(exc.value)


def test_estimate_raises_on_http_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad token"})

    backend = CdpPaymasterBackend(
        endpoint=_TEST_ENDPOINT, client=_mock_transport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        backend.estimate_gas(_TEST_USER_OP)


def test_constructor_rejects_empty_endpoint():
    with pytest.raises(ValueError):
        CdpPaymasterBackend(endpoint="")


# ── cdp_from_env ─────────────────────────────────────────────

def test_cdp_from_env_returns_none_when_endpoint_missing(
    monkeypatch,
):
    monkeypatch.delenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", raising=False,
    )
    assert cdp_from_env() is None


def test_cdp_from_env_constructs_when_endpoint_present(monkeypatch):
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", _TEST_ENDPOINT,
    )
    backend = cdp_from_env()
    assert backend is not None
    assert isinstance(backend, CdpPaymasterBackend)


def test_cdp_from_env_uses_custom_entry_point_env(monkeypatch):
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", _TEST_ENDPOINT,
    )
    monkeypatch.setenv(
        "PRSM_ERC4337_ENTRY_POINT", "0x" + "cd" * 20,
    )
    backend = cdp_from_env()
    assert backend._entry_point == "0x" + "cd" * 20


# ── PaymasterClient.from_env auto-wire ───────────────────────

def test_paymaster_from_env_auto_attaches_cdp_when_endpoint_set(
    monkeypatch,
):
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", _TEST_ENDPOINT,
    )
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_API_KEY", "TESTTOKEN",
    )
    c = PaymasterClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is True


def test_paymaster_from_env_respects_explicit_backend(monkeypatch):
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", _TEST_ENDPOINT,
    )
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_API_KEY", "TESTTOKEN",
    )

    class _Fake:
        def estimate_gas(self, u):
            return {"gas_estimate_wei": 0}

        def submit_sponsored(self, u):
            return {
                "tx_hash": None, "user_op_hash": None,
                "sponsor_amount_wei": 0,
            }

    fake = _Fake()
    c = PaymasterClient.from_env(backend=fake)
    assert c._backend is fake


def test_paymaster_from_env_no_auto_attach_without_endpoint(
    monkeypatch,
):
    monkeypatch.delenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", raising=False,
    )
    monkeypatch.delenv(
        "COINBASE_CDP_PAYMASTER_API_KEY", raising=False,
    )
    c = PaymasterClient.from_env()
    assert c.is_commissioned() is False
    assert c.adapter_wired() is False


def test_paymaster_from_env_graceful_fallback_on_cdp_import_failure(
    monkeypatch,
):
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_ENDPOINT", _TEST_ENDPOINT,
    )
    monkeypatch.setenv(
        "COINBASE_CDP_PAYMASTER_API_KEY", "TESTTOKEN",
    )
    with patch(
        "prsm.economy.web3.paymaster_cdp_backend.from_env",
        side_effect=RuntimeError("boom"),
    ):
        c = PaymasterClient.from_env()
    assert c.is_commissioned() is True
    assert c.adapter_wired() is False
