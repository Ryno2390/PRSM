"""Sprint 279 — AerodromeClient (read-only pool quoter).

Real production code — no commission gate, no on-chain writes.
Reads pool state via eth_call against the Aerodrome USDC-FTNS
pool once it's seeded (Vision gantt 2026-06-15). Pre-seed, the
adapter surfaces NOT_CONFIGURED so operators can verify the
plumbing is wired before the seeding ceremony.

Volatile-pool quote math: standard Uniswap-V2 constant-product
formula with fee_bps deducted from amount_in. Stable-pool curve
(Aerodrome / Velodrome stableswap invariant) is out of scope
for v1 — adapter returns STABLE_POOL_NOT_SUPPORTED on those.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.aerodrome_client import (
    AerodromeClient, AerodromePoolState, AerodromeQuote,
    AerodromeQuoteError,
)


class FakeAerodromeBackend:
    """Returns canned pool state. Production = real Web3 contract."""

    def __init__(
        self, reserves=(1_000_000, 1_000_000),
        token0="0xusdc", token1="0xftns",
        stable=False, fee_bps=30, total_supply=1_000_000,
        block_number=42,
    ):
        self.reserves = reserves
        self.token0 = token0
        self.token1 = token1
        self.stable = stable
        self.fee_bps = fee_bps
        self.total_supply = total_supply
        self.block_number = block_number
        self.calls = []

    def get_pool_state(self, pool_address):
        self.calls.append(pool_address)
        return {
            "pool_address": pool_address,
            "token0": self.token0,
            "token1": self.token1,
            "reserve0": self.reserves[0],
            "reserve1": self.reserves[1],
            "stable": self.stable,
            "fee_bps": self.fee_bps,
            "total_supply": self.total_supply,
            "block_number": self.block_number,
        }


# ── NOT_CONFIGURED paths ─────────────────────────────────


def test_not_configured_when_pool_address_missing():
    c = AerodromeClient(rpc_url="x")  # no pool address
    assert c.is_configured() is False
    assert c.get_pool_state() is None


def test_not_configured_when_rpc_url_missing():
    c = AerodromeClient(pool_address="0xpool")  # no rpc
    assert c.is_configured() is False


def test_configured_when_both_present():
    c = AerodromeClient(rpc_url="x", pool_address="0xpool")
    assert c.is_configured() is True


def test_from_env_reads_vars(monkeypatch):
    monkeypatch.setenv("BASE_RPC_URL", "https://rpc.example")
    monkeypatch.setenv(
        "AERODROME_USDC_FTNS_POOL_ADDRESS", "0xpool",
    )
    c = AerodromeClient.from_env()
    assert c.is_configured() is True


def test_from_env_missing_pool_is_unconfigured(monkeypatch):
    monkeypatch.setenv("BASE_RPC_URL", "https://rpc.example")
    monkeypatch.delenv(
        "AERODROME_USDC_FTNS_POOL_ADDRESS", raising=False,
    )
    c = AerodromeClient.from_env()
    assert c.is_configured() is False


# ── get_pool_state ───────────────────────────────────────


def test_get_pool_state_returns_record():
    fake = FakeAerodromeBackend(reserves=(1_000, 2_000))
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    state = c.get_pool_state()
    assert isinstance(state, AerodromePoolState)
    assert state.pool_address == "0xpool"
    assert state.reserve0 == 1_000
    assert state.reserve1 == 2_000
    assert state.token0 == "0xusdc"
    assert state.token1 == "0xftns"
    assert state.stable is False
    assert state.fee_bps == 30
    assert state.block_number == 42


def test_get_pool_state_fail_soft_on_backend_exception():
    class BoomBackend:
        def get_pool_state(self, addr):
            raise RuntimeError("rpc down")
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=BoomBackend(),
    )
    assert c.get_pool_state() is None


# ── quote_swap volatile ──────────────────────────────────


def test_quote_swap_constant_product_math():
    """Uniswap-V2 volatile pool: out = (in * (10000-fee) * R_out) /
    (10000 * R_in + in * (10000-fee))."""
    fake = FakeAerodromeBackend(
        reserves=(1_000_000, 1_000_000),  # equal reserves
        fee_bps=30,  # 0.3% fee
    )
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    q = c.quote_swap(amount_in=1000, token_in="0xusdc")
    assert isinstance(q, AerodromeQuote)
    assert q.amount_in == 1000
    assert q.token_in == "0xusdc"
    assert q.token_out == "0xftns"
    # Expected: (1000 * 9970 * 1_000_000) / (10000*1_000_000 + 1000*9970)
    #         = 9_970_000_000_000 / 10_009_970_000 = 996.0089...
    assert 995 <= q.amount_out <= 997


def test_quote_swap_reverse_direction():
    fake = FakeAerodromeBackend(
        reserves=(1_000_000, 1_000_000), fee_bps=30,
    )
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    q = c.quote_swap(amount_in=1000, token_in="0xftns")
    assert q.token_in == "0xftns"
    assert q.token_out == "0xusdc"
    assert 995 <= q.amount_out <= 997


def test_quote_swap_unknown_token_raises():
    fake = FakeAerodromeBackend()
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    with pytest.raises(AerodromeQuoteError):
        c.quote_swap(amount_in=1000, token_in="0xeth")


def test_quote_swap_zero_amount_raises():
    fake = FakeAerodromeBackend()
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    with pytest.raises(ValueError):
        c.quote_swap(amount_in=0, token_in="0xusdc")


def test_quote_swap_negative_amount_raises():
    fake = FakeAerodromeBackend()
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    with pytest.raises(ValueError):
        c.quote_swap(amount_in=-10, token_in="0xusdc")


def test_quote_swap_when_unconfigured_returns_none():
    c = AerodromeClient()  # nothing configured
    q = c.quote_swap(amount_in=1000, token_in="0xusdc")
    assert q is None


def test_quote_swap_price_impact_small_trade():
    fake = FakeAerodromeBackend(
        reserves=(1_000_000_000, 1_000_000_000),  # deep liquidity
        fee_bps=30,
    )
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    q = c.quote_swap(amount_in=100, token_in="0xusdc")
    # Tiny relative size → minimal price impact
    assert q.price_impact_bps < 10


def test_quote_swap_price_impact_large_trade():
    fake = FakeAerodromeBackend(
        reserves=(1_000_000, 1_000_000),
        fee_bps=30,
    )
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    q = c.quote_swap(amount_in=100_000, token_in="0xusdc")
    # 10% of reserves → meaningful impact
    assert q.price_impact_bps > 500


# ── stable pool unsupported in v1 ────────────────────────


def test_quote_swap_stable_pool_returns_not_supported():
    fake = FakeAerodromeBackend(stable=True)
    c = AerodromeClient(
        rpc_url="x", pool_address="0xpool", backend=fake,
    )
    with pytest.raises(AerodromeQuoteError) as exc:
        c.quote_swap(amount_in=1000, token_in="0xusdc")
    assert "stable" in str(exc.value).lower()


# ── Dataclass round-trip ─────────────────────────────────


def test_pool_state_to_dict():
    s = AerodromePoolState(
        pool_address="0xpool",
        token0="0xa", token1="0xb",
        reserve0=100, reserve1=200,
        stable=False, fee_bps=30,
        total_supply=500, block_number=42,
    )
    d = s.to_dict()
    assert d["reserve0"] == 100
    assert d["pool_address"] == "0xpool"


def test_quote_to_dict():
    q = AerodromeQuote(
        amount_in=1000, token_in="0xa",
        token_out="0xb", amount_out=996,
        price_impact_bps=10, route="aerodrome",
        fee_bps=30,
    )
    d = q.to_dict()
    assert d["amount_in"] == 1000
    assert d["price_impact_bps"] == 10
