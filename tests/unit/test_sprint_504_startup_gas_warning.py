"""Sprint 504 — daemon startup logs gas-status warning.

Sprint 502 shipped the endpoint, sprint 503 wired it into
/health/detailed. Both surfaces are pull-based — operators have
to actively query. Sprint 504 adds a push signal: at daemon
startup (end of OnChainFTNSLedger.initialize()), log a clear
warning when balance is low or critical.

Boundary: tests call `_emit_startup_gas_log` directly with a
mocked w3, capturing log records via caplog. The real
initialize() path was already covered by sprints 502 + 503's
live-verify runs.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from prsm.economy.ftns_onchain import OnChainFTNSLedger


def _build_ledger(eth_wei=None, has_w3=True):
    led = OnChainFTNSLedger(
        node_id="t",
        wallet_private_key=None,
    )
    led._connected_address = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    if has_w3:
        led.w3 = MagicMock()
        led.w3.eth.get_balance.return_value = eth_wei
    else:
        led.w3 = None
    return led


def test_emit_warning_log_when_low(caplog):
    """0.0003 ETH (low band) → WARNING-level log."""
    led = _build_ledger(eth_wei=3 * 10**14)
    with caplog.at_level(logging.WARNING):
        led._emit_startup_gas_log()
    records = [
        r for r in caplog.records
        if "gas" in r.getMessage().lower()
        and r.levelno == logging.WARNING
    ]
    assert records, (
        f"expected WARNING gas log, got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    assert "low" in records[0].getMessage().lower()


def test_emit_error_log_when_critical(caplog):
    """0.00005 ETH (critical band) → ERROR-level log."""
    led = _build_ledger(eth_wei=5 * 10**13)
    with caplog.at_level(logging.ERROR):
        led._emit_startup_gas_log()
    records = [
        r for r in caplog.records
        if "gas" in r.getMessage().lower()
        and r.levelno == logging.ERROR
    ]
    assert records
    assert "critical" in records[0].getMessage().lower()


def test_emit_info_log_when_ok(caplog):
    """Healthy balance still logs at INFO so operators see
    the address + balance in startup output for confirmation."""
    led = _build_ledger(eth_wei=10**15)  # 0.001 ETH
    with caplog.at_level(logging.INFO):
        led._emit_startup_gas_log()
    records = [
        r for r in caplog.records
        if "gas" in r.getMessage().lower()
    ]
    assert records


def test_emit_silent_when_no_w3(caplog):
    """If w3 is missing, no log noise. The user already
    sees web3 init failure logs upstream."""
    led = _build_ledger(has_w3=False)
    with caplog.at_level(logging.DEBUG):
        led._emit_startup_gas_log()
    gas_records = [
        r for r in caplog.records
        if "gas" in r.getMessage().lower()
    ]
    assert not gas_records


def test_emit_silent_on_rpc_exception(caplog):
    """get_balance throwing must NOT crash startup. Log a
    debug message but no warning/error — the user can pull
    real status via /wallet/gas-status."""
    led = _build_ledger(eth_wei=0)
    led.w3.eth.get_balance.side_effect = RuntimeError("rpc down")
    with caplog.at_level(logging.DEBUG):
        led._emit_startup_gas_log()
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "gas" in r.getMessage().lower()
    ]
    assert not warn_records, (
        f"startup must not warn on RPC fail: "
        f"{[(r.levelname, r.getMessage()) for r in warn_records]}"
    )


@pytest.mark.asyncio
async def test_initialize_calls_emit_startup_gas_log(monkeypatch):
    """The public initialize() must call
    _emit_startup_gas_log at the end so operators get the
    push signal."""
    led = OnChainFTNSLedger(
        node_id="t",
        wallet_private_key=None,
    )
    called = {"yes": False}

    def fake_emit():
        called["yes"] = True

    monkeypatch.setattr(
        led, "_emit_startup_gas_log", fake_emit,
    )

    # Force the no-web3 short-circuit so we don't try to hit
    # a real RPC. We still want the emit call to happen — so
    # patch HAS_WEB3 to True and stub the heavy connect path.
    import prsm.economy.ftns_onchain as mod
    monkeypatch.setattr(mod, "HAS_WEB3", False)
    # With HAS_WEB3=False, initialize() returns False early.
    # That's fine — _emit_startup_gas_log should still be
    # called in the success path. Let's verify the call site
    # by patching HAS_WEB3=True and stubbing Web3.
    monkeypatch.setattr(mod, "HAS_WEB3", True)

    class _FakeEth:
        chain_id = 8453
        def contract(self, address=None, abi=None):
            c = MagicMock()
            c.functions.decimals().call.return_value = 18
            c.functions.name().call.return_value = "FTNS"
            c.functions.symbol().call.return_value = "FTNS"
            return c

    class _FakeW3:
        def __init__(self, *a, **kw):
            self.eth = _FakeEth()
        def is_connected(self):
            return True
        @staticmethod
        def to_checksum_address(a):
            return a
        @staticmethod
        def HTTPProvider(*a, **kw):
            return None

    monkeypatch.setattr(mod, "Web3", _FakeW3)
    # Also bypass _init_persistence since aiosqlite isn't
    # exercised here.
    async def _noop():
        return None
    monkeypatch.setattr(led, "_init_persistence", _noop)

    await led.initialize()
    assert called["yes"], (
        "initialize() must call _emit_startup_gas_log"
    )
