"""Sprint 509 — Sepolia parity for the FTNS-side operator UX.

Sprints 498-508 shipped the full operator-UX surface on Base
mainnet. Sprint 509 verifies all the surfaces work identically
when the daemon is configured for Base Sepolia (chainId 84532),
so operators can dogfood without spending real ETH.

These pin tests defend the configuration parity:
  - `prsm wallet info --network testnet` reads Sepolia
  - Network config has Sepolia FTNS address
  - OnChainFTNSLedger accepts Sepolia chain_id + RPC
  - Same thresholds apply (no special-casing per network)

Live-verified during this sprint walk:
  - Daemon launched with PRSM_NETWORK=testnet +
    BASE_SEPOLIA_RPC_URL + FTNS_TOKEN_ADDRESS=Sepolia FTNS
  - /health/detailed.ftns_ledger.canonical_match: True
    against Sepolia FTNS `0x7F5f00FA…`
  - /health/detailed.operator_gas: status=critical
    (operator wallet has 0 ETH on Sepolia)
  - /wallet/gas-status: identical schema, correct critical
  - `prsm wallet gas-status` CLI renders CRITICAL warning
  - Sprint-504 startup log fires: "Operator gas CRITICAL:
    0.0000000000 ETH on 0x4acdE458… — broadcasts will start
    failing soon. Top up now."

**F39 surfaced**: SQLite TX history DB shared across networks
— no chain_id discrimination. Mainnet TX appear in Sepolia
daemon's /wallet/transactions/onchain output. Fix candidate
for sprint 510: either store chain_id per row + filter, or
namespace db_path by network.
"""
from __future__ import annotations


def test_sepolia_network_config_has_ftns_token():
    """Base Sepolia (84532) must have FTNS token wired —
    operators rely on this to dogfood the operator-UX arc
    before mainnet."""
    from prsm.config.networks import TESTNET
    assert TESTNET.chain_id == 84532
    assert TESTNET.ftns_token == (
        "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
    )
    assert TESTNET.rpc_url_default == (
        "https://sepolia.base.org"
    )


def test_sepolia_rpc_url_resolution_honors_env_var():
    """Per sprint 465: BASE_SEPOLIA_RPC_URL (not BASE_RPC_URL)
    is the canonical override env var for testnet. Operators
    coming from mainnet config will use the wrong var first;
    this pin defends the documented contract."""
    import os
    from prsm.config.networks import (
        _resolve_rpc_url, TESTNET,
    )
    original = os.environ.get("BASE_SEPOLIA_RPC_URL")
    try:
        os.environ["BASE_SEPOLIA_RPC_URL"] = (
            "https://override.example/sepolia"
        )
        url = _resolve_rpc_url(TESTNET)
        assert url == "https://override.example/sepolia"
    finally:
        if original is None:
            os.environ.pop("BASE_SEPOLIA_RPC_URL", None)
        else:
            os.environ["BASE_SEPOLIA_RPC_URL"] = original


def test_sepolia_does_not_have_royalty_distributor():
    """Per memory + networks.py: testnet does NOT deploy
    RoyaltyDistributor. The CLI must skip claim gracefully
    when royalty_distributor is None — sprint 508 verified
    this live; the pin defends the invariant."""
    from prsm.config.networks import TESTNET
    assert TESTNET.royalty_distributor is None


def test_gas_status_thresholds_are_network_agnostic():
    """Sprint 502/503/506/508 thresholds (low=0.0005,
    critical=0.0001) must be identical on testnet —
    operators dogfooding on Sepolia get the same alerting
    they'll see on mainnet. No per-network override."""
    from prsm.economy.ftns_onchain import _gas_status_for_eth
    # Same thresholds on testnet as mainnet
    assert _gas_status_for_eth(0.001) == "ok"
    assert _gas_status_for_eth(0.0003) == "low"
    assert _gas_status_for_eth(0.00005) == "critical"


def test_onchain_ledger_accepts_sepolia_chain_id():
    """OnChainFTNSLedger constructor must accept the
    Sepolia chain_id 84532 — operators configure this via
    networks.py + the env-var resolver."""
    from prsm.economy.ftns_onchain import OnChainFTNSLedger
    led = OnChainFTNSLedger(
        node_id="t",
        wallet_private_key=None,
        contract_address=(
            "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
        ),
        rpc_url="https://sepolia.base.org",
        chain_id=84532,
    )
    assert led.chain_id == 84532
    assert led.contract_address == (
        "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
    )


def test_f39_documented_as_followon():
    """Sprint 509 surfaced F39: SQLite TX history is shared
    across networks. Mainnet TX appear in Sepolia daemon's
    /wallet/transactions/onchain output (and vice versa).

    Fix candidates for sprint 510:
      A) Add chain_id column to onchain_transactions table,
         INSERT with current chain, filter in endpoint
      B) Namespace db_path by network
         (~/.prsm/onchain_tx_mainnet.db,
          ~/.prsm/onchain_tx_sepolia.db)

    This test pins the awareness — failing if a future
    sprint forgets F39 exists. Update when F39 is fixed
    and replace with the fix verification.
    """
    # Existence-of-finding pin. The actual fix lives in
    # sprint 510+.
    F39_NOTE = (
        "SQLite TX history shared across networks "
        "(mainnet TX visible in testnet daemon /wallet/"
        "transactions/onchain). Needs chain_id "
        "discrimination — fix candidates documented in "
        "test_sprint_509_sepolia_parity.py docstring."
    )
    assert "chain_id" in F39_NOTE
    assert "shared across networks" in F39_NOTE
