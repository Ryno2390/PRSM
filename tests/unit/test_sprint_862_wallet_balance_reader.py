"""Sprint 862 — Wallet balance reader pin tests.

Defends the JSON-RPC ERC-20 ``balanceOf(address)`` encoding +
decimal normalization for USDC (6) / FTNS (18) / native ETH (18).

Pin tests:
  - _addr_to_call_data emits selector + padded address correctly
  - _addr_to_call_data rejects non-20-byte addresses
  - _decode_uint256 handles 0x / empty / valid hex / bad input
  - get_balances issues 4 RPC calls (USDC + FTNS + native + block)
  - Decimal normalization: USDC 6, FTNS 18, ETH 18
  - WalletBalances dataclass roundtrips cleanly through to_dict()
  - from_env falls back to public Base RPC when BASE_RPC_URL unset
  - from_env honors BASE_RPC_URL when set
  - RPC error payloads raise RuntimeError with context
  - bad address rejected at boundary
"""
from __future__ import annotations

import httpx
import pytest

_real_Client = httpx.Client


@pytest.fixture(autouse=True)
def _restore_real_httpx(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _real_Client)
    monkeypatch.setattr(httpx, "MockTransport", httpx.MockTransport)
    monkeypatch.setattr(httpx, "Response", httpx.Response)
    yield


from prsm.economy.web3.wallet_balance_reader import (
    WalletBalanceReader,
    WalletBalances,
    _addr_to_call_data,
    _decode_uint256,
    from_env as wbr_from_env,
)


_TEST_ADDR = "0x01D1c152Ef261b1d74983EDC36C47D9cE3ba2fA5"
_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── _addr_to_call_data ───────────────────────────────────────

def test_call_data_selector_prefix():
    data = _addr_to_call_data(_TEST_ADDR)
    # balanceOf(address) selector
    assert data.startswith("0x70a08231")


def test_call_data_padded_to_32_bytes():
    data = _addr_to_call_data(_TEST_ADDR)
    # selector (4 bytes = 10 chars w/ 0x) + 32-byte address (64 chars)
    assert len(data) == 10 + 64


def test_call_data_strips_0x_prefix():
    data = _addr_to_call_data("0x" + "ab" * 20)
    assert data == (
        "0x70a08231"
        + "0" * 24
        + "ab" * 20
    )


def test_call_data_lowercases():
    """ERC-20 balanceOf is checksum-agnostic; we lowercase to
    ensure consistent canonical encoding."""
    data_upper = _addr_to_call_data("0x" + "AB" * 20)
    data_lower = _addr_to_call_data("0x" + "ab" * 20)
    assert data_upper == data_lower


def test_call_data_rejects_short_address():
    with pytest.raises(ValueError) as exc:
        _addr_to_call_data("0xabc")
    assert "20 bytes" in str(exc.value)


def test_call_data_rejects_long_address():
    with pytest.raises(ValueError):
        _addr_to_call_data("0x" + "ab" * 21)


# ── _decode_uint256 ──────────────────────────────────────────

def test_decode_uint256_valid_hex():
    assert _decode_uint256("0x" + "ff" * 2) == 65535


def test_decode_uint256_empty_returns_0():
    assert _decode_uint256("0x") == 0


def test_decode_uint256_none_returns_0():
    assert _decode_uint256(None) == 0


def test_decode_uint256_bad_string_returns_0():
    """Defensive — don't crash the balance reader on malformed
    RPC responses; surface as 0 balance."""
    assert _decode_uint256("not-hex") == 0


def test_decode_uint256_zero_value():
    assert _decode_uint256("0x0") == 0


# ── get_balances integration ─────────────────────────────────

def test_get_balances_issues_4_rpc_calls():
    """USDC eth_call + FTNS eth_call + eth_getBalance +
    eth_blockNumber — exactly 4 round-trips."""
    calls = []

    def handler(request):
        body = request.read()
        import json as _j
        body_obj = _j.loads(body)
        calls.append(body_obj["method"])
        # Echo back a result for each method
        method = body_obj["method"]
        if method == "eth_call":
            return httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": body_obj["id"],
                "result": "0x" + "00" * 31 + "2a",  # 42 base units
            })
        if method == "eth_getBalance":
            return httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": body_obj["id"],
                "result": "0xde0b6b3a7640000",  # 1 ETH (1e18 wei)
            })
        if method == "eth_blockNumber":
            return httpx.Response(200, json={
                "jsonrpc": "2.0",
                "id": body_obj["id"],
                "result": "0x123456",
            })
        return httpx.Response(404, json={
            "jsonrpc": "2.0", "id": body_obj["id"],
            "error": {"message": "unknown method"},
        })

    reader = WalletBalanceReader(
        ftns_address="0x" + "ff" * 20,
        rpc_url="https://mock",
        client=_mock(handler),
    )
    bal = reader.get_balances(_TEST_ADDR)
    assert calls == [
        "eth_call",       # USDC
        "eth_call",       # FTNS
        "eth_getBalance", # native ETH
        "eth_blockNumber",
    ]
    assert bal.block_number == 0x123456


def test_get_balances_usdc_6_decimals():
    """1 USDC = 1_000_000 base units; reader returns 1.0."""
    def handler(request):
        import json as _j
        body = _j.loads(request.read())
        method = body["method"]
        if method == "eth_call":
            # Only respond to USDC call with non-zero (first eth_call)
            data = body["params"][0]["data"]
            if body["params"][0]["to"] == _USDC:
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": body["id"],
                    "result": hex(1_000_000),  # 1 USDC
                })
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": "0x0",
            })
        if method == "eth_getBalance":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"], "result": "0x0",
            })
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"], "result": "0x1",
        })

    reader = WalletBalanceReader(
        ftns_address="0x" + "ff" * 20,
        rpc_url="https://mock",
        client=_mock(handler),
    )
    bal = reader.get_balances(_TEST_ADDR)
    assert bal.usdc == 1.0
    assert bal.usdc_units == 1_000_000


def test_get_balances_ftns_18_decimals():
    """100 FTNS = 100 * 1e18 = 1e20 base units → 100.0 whole."""
    ftns_addr = "0xffffffffffffffffffffffffffffffffffffffff"

    def handler(request):
        import json as _j
        body = _j.loads(request.read())
        method = body["method"]
        if method == "eth_call":
            to_addr = body["params"][0]["to"].lower()
            if to_addr == ftns_addr.lower():
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": body["id"],
                    "result": hex(100 * 10**18),
                })
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": "0x0",
            })
        if method == "eth_getBalance":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"], "result": "0x0",
            })
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"], "result": "0x1",
        })

    reader = WalletBalanceReader(
        ftns_address=ftns_addr, rpc_url="https://mock",
        client=_mock(handler),
    )
    bal = reader.get_balances(_TEST_ADDR)
    assert bal.ftns == 100.0
    assert bal.ftns_units == 100 * 10**18


def test_get_balances_native_eth_18_decimals():
    """0.5 ETH = 5e17 wei → 0.5 whole."""
    def handler(request):
        import json as _j
        body = _j.loads(request.read())
        if body["method"] == "eth_getBalance":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": hex(5 * 10**17),
            })
        if body["method"] == "eth_call":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": "0x0",
            })
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"], "result": "0x1",
        })

    reader = WalletBalanceReader(
        ftns_address="0x" + "ff" * 20, rpc_url="https://mock",
        client=_mock(handler),
    )
    bal = reader.get_balances(_TEST_ADDR)
    assert bal.native_eth == 0.5
    assert bal.native_eth_wei == 5 * 10**17


# ── Error handling ───────────────────────────────────────────

def test_rpc_error_payload_raises():
    def handler(request):
        import json as _j
        body = _j.loads(request.read())
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"],
            "error": {"code": -32000, "message": "rate limited"},
        })

    reader = WalletBalanceReader(
        ftns_address="0x" + "ff" * 20, rpc_url="https://mock",
        client=_mock(handler),
    )
    with pytest.raises(RuntimeError) as exc:
        reader.get_balances(_TEST_ADDR)
    assert "rate limited" in str(exc.value)


def test_bad_address_rejected_at_boundary():
    reader = WalletBalanceReader(
        ftns_address="0x" + "ff" * 20, rpc_url="https://mock",
        client=_mock(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(ValueError) as exc:
        reader.get_balances("not-an-address")
    assert "0x EVM" in str(exc.value)


def test_http_500_raises():
    def handler(request):
        return httpx.Response(500, text="upstream error")

    reader = WalletBalanceReader(
        ftns_address="0x" + "ff" * 20, rpc_url="https://mock",
        client=_mock(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        reader.get_balances(_TEST_ADDR)


# ── from_env ─────────────────────────────────────────────────

def test_from_env_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("BASE_RPC_URL", raising=False)
    reader = wbr_from_env()
    assert reader._rpc_url == "https://mainnet.base.org"


def test_from_env_honors_base_rpc_url(monkeypatch):
    monkeypatch.setenv("BASE_RPC_URL", "https://custom-rpc.example")
    reader = wbr_from_env()
    assert reader._rpc_url == "https://custom-rpc.example"


def test_from_env_explicit_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("BASE_RPC_URL", "https://env-rpc")
    reader = wbr_from_env(rpc_url="https://kwarg-rpc")
    assert reader._rpc_url == "https://kwarg-rpc"


# ── WalletBalances dataclass ─────────────────────────────────

def test_wallet_balances_to_dict_canonical_fields():
    bal = WalletBalances(
        address=_TEST_ADDR, usdc=1.5, usdc_units=1_500_000,
        ftns=2.5, ftns_units=int(2.5 * 10**18),
        native_eth=0.1, native_eth_wei=int(0.1 * 10**18),
        block_number=12345, rpc_url="https://mock",
    )
    d = bal.to_dict()
    assert d["address"] == _TEST_ADDR
    assert d["usdc"] == 1.5
    assert d["ftns"] == 2.5
    assert d["block_number"] == 12345
    # Canonical schema invariant — these field names are
    # consumed by /wallet/balance/* endpoints + CLI surfaces.
    expected_keys = {
        "address", "usdc", "usdc_units", "ftns", "ftns_units",
        "native_eth", "native_eth_wei", "block_number", "rpc_url",
    }
    assert set(d.keys()) == expected_keys
