"""Sprint 902 — real Web3 backend for AerodromeClient pool reads.

sp901's go-live harness (and the live node's onramp→swap quoting,
node.py:2634) need live pool state, but AerodromeClient.from_env wired
NO backend — get_pool_state returned None, so neither the harness nor
the production swap quoter could actually read the seeded pool. This
ships AerodromeRpcBackend: eth_call against the Aerodrome Pool contract
(token0/token1/getReserves/stable/totalSupply) + eth_blockNumber,
mirroring sp862's WalletBalanceReader JSON-RPC pattern. from_env now
wires it whenever BASE_RPC_URL is set, so the whole Aerodrome read path
goes live with no code change at seed time.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.aerodrome_client import (
    AerodromeClient,
    AerodromeRpcBackend,
)
from prsm.economy.web3.aerodrome_pool_ceremony import (
    MAINNET_CONFIG, USDC_BASE_MAINNET, FTNS_BASE_MAINNET,
)

_R0 = 500_000 * 10 ** 6       # 500k USDC (token0)
_R1 = 2_000_000 * 10 ** 18    # 2M FTNS (token1)


def _word_addr(addr: str) -> str:
    return addr[2:].lower().rjust(64, "0")


def _word_uint(n: int) -> str:
    return format(n, "064x")


class _FakeResp:
    def __init__(self, result):
        self._result = result

    def raise_for_status(self):
        pass

    def json(self):
        return {"jsonrpc": "2.0", "id": 1, "result": self._result}


class _FakeRpcClient:
    """Routes eth_call by selector + answers eth_blockNumber."""

    def __init__(self, *, stable=False):
        self._stable = stable
        self.closed = False

    def post(self, url, json=None, headers=None):
        method = json["method"]
        if method == "eth_blockNumber":
            return _FakeResp("0x" + format(12345, "x"))
        # eth_call — route by 4-byte selector.
        sel = json["params"][0]["data"][:10]
        table = {
            "0x0dfe1681": "0x" + _word_addr(USDC_BASE_MAINNET),  # token0
            "0xd21220a7": "0x" + _word_addr(FTNS_BASE_MAINNET),  # token1
            "0x0902f1ac": (  # getReserves → (r0, r1, ts)
                "0x" + _word_uint(_R0) + _word_uint(_R1)
                + _word_uint(1717000000)
            ),
            "0x22be3de1": "0x" + _word_uint(1 if self._stable else 0),
            "0x18160ddd": "0x" + _word_uint(10 ** 24),  # totalSupply
        }
        return _FakeResp(table[sel])

    def close(self):
        self.closed = True


# ── Backend decodes pool state from eth_call ─────────────────

def test_backend_decodes_pool_state():
    backend = AerodromeRpcBackend(
        rpc_url="https://rpc.test", client=_FakeRpcClient(),
    )
    state = backend.get_pool_state("0xPOOL")
    assert state["token0"].lower() == USDC_BASE_MAINNET.lower()
    assert state["token1"].lower() == FTNS_BASE_MAINNET.lower()
    assert state["reserve0"] == _R0
    assert state["reserve1"] == _R1
    assert state["stable"] is False
    assert state["total_supply"] == 10 ** 24
    assert state["block_number"] == 12345


def test_client_with_backend_returns_pool_state():
    client = AerodromeClient(
        rpc_url="https://rpc.test", pool_address="0xPOOL",
        backend=AerodromeRpcBackend(
            rpc_url="https://rpc.test", client=_FakeRpcClient(),
        ),
    )
    state = client.get_pool_state()
    assert state is not None
    assert state.reserve0 == _R0
    assert state.reserve1 == _R1
    assert state.stable is False


# ── End-to-end: the go-live harness passes with the real backend ─

def test_go_live_harness_passes_with_rpc_backend():
    from prsm.economy.web3.go_live_verification import (
        run_go_live_verification,
    )
    client = AerodromeClient(
        rpc_url="https://rpc.test", pool_address="0xPOOL",
        backend=AerodromeRpcBackend(
            rpc_url="https://rpc.test", client=_FakeRpcClient(),
        ),
    )
    report = run_go_live_verification(
        client, MAINNET_CONFIG, probe_usdc_units=1_000_000,
    )
    assert report.go is True, report.to_dict()
    assert report.prepared_envelope is not None


def test_stable_pool_decoded_true():
    backend = AerodromeRpcBackend(
        rpc_url="https://rpc.test", client=_FakeRpcClient(stable=True),
    )
    assert backend.get_pool_state("0xPOOL")["stable"] is True


# ── from_env wires the backend when BASE_RPC_URL is set ──────

def test_from_env_wires_rpc_backend(monkeypatch):
    monkeypatch.setenv("BASE_RPC_URL", "https://rpc.test")
    monkeypatch.setenv(
        "AERODROME_USDC_FTNS_POOL_ADDRESS", "0xPOOL",
    )
    client = AerodromeClient.from_env()
    assert client.is_configured()
    # A real backend is now wired (was None pre-sp902).
    assert client._backend is not None
    assert isinstance(client._backend, AerodromeRpcBackend)


def test_from_env_no_rpc_no_backend(monkeypatch):
    monkeypatch.delenv("BASE_RPC_URL", raising=False)
    monkeypatch.delenv(
        "AERODROME_USDC_FTNS_POOL_ADDRESS", raising=False,
    )
    client = AerodromeClient.from_env()
    assert client._backend is None


# ── Fail-soft: RPC error → client returns None (not a crash) ──

def test_rpc_error_fails_soft():
    class _BoomClient:
        def post(self, *a, **k):
            raise RuntimeError("rpc down")
        def close(self):
            pass

    client = AerodromeClient(
        rpc_url="https://rpc.test", pool_address="0xPOOL",
        backend=AerodromeRpcBackend(
            rpc_url="https://rpc.test", client=_BoomClient(),
        ),
    )
    # AerodromeClient.get_pool_state swallows backend exceptions.
    assert client.get_pool_state() is None
