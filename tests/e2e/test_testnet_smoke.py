"""T8 — E2E smoke test against the deployed Base Sepolia bundle.

Locks in the post-T1/T6 wiring: with `PRSM_NETWORK=testnet` plus a
`FTNS_WALLET_PRIVATE_KEY`, every on-chain client reads from Base
Sepolia (chain 84532) and the FTNS proxy at the address pinned in
`prsm.config.networks.TESTNET`. Without this regression guard, a
future re-introduction of hardcoded mainnet defaults (the bug the
2026-05-07 T6 wiring closed) would silently retarget testnet
operators at production contracts.

The test is gated by the `PRSM_TESTNET_E2E=1` env var because it
makes a live RPC call. CI runs it on a schedule rather than on every
push so a flaky public Sepolia RPC doesn't block ordinary pushes.
"""
from __future__ import annotations

import asyncio
import os
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("PRSM_TESTNET_E2E", "").lower() not in ("1", "true", "yes"),
    reason="PRSM_TESTNET_E2E not set — live Base Sepolia RPC required",
)


# Pinned testnet expectations — keep in sync with TESTNET in
# prsm/config/networks.py. If these values diverge, either someone
# moved the testnet deploy (re-pin) or the resolver is broken (fix).
TESTNET_CHAIN_ID = 84532
TESTNET_FTNS_TOKEN = "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
TESTNET_FTNS_TOTAL_SUPPLY_MIN = 99_000_000  # initial mint = 100M; allow slop


@pytest.fixture(autouse=True)
def _testnet_env(monkeypatch):
    """Prime PRSM_NETWORK=testnet for every test in this module."""
    monkeypatch.setenv("PRSM_NETWORK", "testnet")
    yield


def test_resolve_endpoints_returns_testnet_addresses():
    from prsm.config.networks import resolve_endpoints

    endpoints = resolve_endpoints()
    assert endpoints.network_name == "testnet"
    assert endpoints.chain_id == TESTNET_CHAIN_ID
    assert endpoints.ftns_token == TESTNET_FTNS_TOKEN
    # Audit-bundle + Phase 8 + Phase 7-storage must all be set:
    for f in (
        "settlement_registry",
        "escrow_pool",
        "stake_bond",
        "emission_controller",
        "compensation_distributor",
        "storage_slashing",
        "key_distribution",
    ):
        addr = getattr(endpoints, f)
        assert addr is not None and addr.startswith("0x"), f"{f} unset"


def test_onchain_ftns_ledger_connects_to_sepolia():
    pk = os.getenv("PRIVATE_KEY") or os.getenv("FTNS_WALLET_PRIVATE_KEY")
    if not pk:
        pytest.skip("PRIVATE_KEY not set")
    os.environ.setdefault("FTNS_WALLET_PRIVATE_KEY", pk)

    # Module-level constants are resolved at import-time. Re-import
    # forces the testnet env vars to flow through.
    import importlib
    from prsm.economy import ftns_onchain
    importlib.reload(ftns_onchain)

    ledger = ftns_onchain.OnChainFTNSLedger(node_id="t8-smoke")
    assert ledger.contract_address.lower() == TESTNET_FTNS_TOKEN.lower()
    assert ledger.chain_id == TESTNET_CHAIN_ID
    assert ledger._connected_address is not None  # derived sync from PK

    asyncio.run(ledger.initialize())
    assert ledger._is_initialized is True
    assert ledger._decimals == 18

    # Read the FTNS balance of the connected address. On the live
    # testnet deploy the deployer minted 100M FTNS — anyone running
    # this test from a different deployer key just verifies the call
    # path; balance can be 0.
    bal = asyncio.run(ledger.get_balance(ledger._connected_address))
    assert bal >= 0


def test_total_supply_matches_initial_mint():
    """Pin the post-deploy invariant: 100M FTNS minted at construction."""
    pytest.importorskip("web3")
    from web3 import Web3

    rpc = (
        os.getenv("BASE_SEPOLIA_RPC_URL")
        or os.getenv("PRSM_BASE_RPC_URL")
        or "https://sepolia.base.org"
    )
    w3 = Web3(Web3.HTTPProvider(rpc))
    abi = [
        {
            "constant": True,
            "inputs": [],
            "name": "totalSupply",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
        }
    ]
    ftns = w3.eth.contract(address=TESTNET_FTNS_TOKEN, abi=abi)
    supply_wei = ftns.functions.totalSupply().call()
    supply_ftns = supply_wei / 10**18
    assert supply_ftns >= TESTNET_FTNS_TOTAL_SUPPLY_MIN, (
        f"unexpected total supply {supply_ftns} FTNS — testnet bundle "
        f"may have been redeployed; refresh networks.py + this constant."
    )
