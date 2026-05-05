"""Per-network contract addresses + RPC endpoints.

Single source of truth for which contracts a node should talk to on each
chain. Loaded by `prsm.node.node.PRSMNode`, the bootstrap protocol, and
the CLI's `prsm join-testnet` command.

To add a new network: append an entry to NETWORK_CONFIGS keyed by the
network name. Network names are stable user-facing strings (e.g.,
"mainnet", "testnet"); chain IDs are the EIP-155 numeric identifier.

Mainnet contract addresses are pinned to Base mainnet deployments from
Phase 1.3 Task 8 (2026-05-04). Testnet contract addresses are filled in
post-T1 deploy (see docs/2026-05-05-public-testnet-deploy-plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NetworkConfig:
    """Immutable per-network configuration."""

    name: str
    chain_id: int
    rpc_url_default: str
    explorer_url: str

    # Phase 1.3 contracts (deployed)
    ftns_token: Optional[str] = None
    provenance_registry: Optional[str] = None
    royalty_distributor: Optional[str] = None
    foundation_safe: Optional[str] = None

    # Audit-bundle (deployed at L4 audit clear on mainnet; deployed at T1 on testnet)
    escrow_pool: Optional[str] = None
    stake_bond: Optional[str] = None
    settlement_registry: Optional[str] = None
    signature_verifier: Optional[str] = None

    # Phase 8 emission stack
    emission_controller: Optional[str] = None
    compensation_distributor: Optional[str] = None

    # Phase 7-storage
    storage_slashing: Optional[str] = None
    key_distribution: Optional[str] = None

    # Phase 3.x.3 publisher key anchor
    publisher_key_anchor: Optional[str] = None

    # Network-specific operational notes for users
    notes: tuple[str, ...] = field(default_factory=tuple)

    def is_complete(self) -> bool:
        """True if all expected contracts are deployed (no None placeholders)."""
        required = [
            self.ftns_token,
            self.provenance_registry,
            self.royalty_distributor,
            self.foundation_safe,
            self.escrow_pool,
            self.stake_bond,
            self.settlement_registry,
            self.signature_verifier,
            self.emission_controller,
            self.compensation_distributor,
            self.storage_slashing,
            self.key_distribution,
        ]
        return all(addr is not None for addr in required)


# ────────────────────────────────────────────────────────────────────────
# Mainnet (Base, chainId 8453)
# ────────────────────────────────────────────────────────────────────────
MAINNET = NetworkConfig(
    name="mainnet",
    chain_id=8453,
    rpc_url_default="https://mainnet.base.org",
    explorer_url="https://basescan.org",
    # Phase 1.3 Task 8 deploys (2026-05-04):
    ftns_token="0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
    provenance_registry="0xdF470BFa9eF310B196801D5105468515d0069915",
    royalty_distributor=None,  # TODO: pin v1 address from contracts/deployments/provenance-base-1777917793612.json or fill in v2 post-redeploy
    foundation_safe="0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791",
    # Audit-bundle (gated on L4 firm pair-review; not yet deployed):
    escrow_pool=None,
    stake_bond=None,
    settlement_registry=None,
    signature_verifier=None,
    emission_controller=None,
    compensation_distributor=None,
    storage_slashing=None,
    key_distribution=None,
    publisher_key_anchor=None,  # Phase 3.x.3 — not yet on mainnet
    notes=(
        "Mainnet uses the real 2-of-3 Foundation Safe (Ledger + Trezor + OneKey) at 0x91b0...5791.",
        "Audit-bundle contracts (EscrowPool, StakeBond, etc.) deploy after L4 firm pair-review clears.",
        "Until audit-bundle deploys, mainnet supports content registration + royalty payouts only.",
    ),
)

# ────────────────────────────────────────────────────────────────────────
# Testnet (Base Sepolia, chainId 84532)
# Addresses filled in post-T1 deploy.
# ────────────────────────────────────────────────────────────────────────
TESTNET = NetworkConfig(
    name="testnet",
    chain_id=84532,
    rpc_url_default="https://sepolia.base.org",
    explorer_url="https://sepolia.basescan.org",
    # T1 deploy fills these in (see docs/2026-05-05-public-testnet-deploy-plan.md):
    ftns_token=None,                  # TODO post-T1: from contracts/deployments/mock-ftns-base-sepolia-*.json
    provenance_registry=None,         # TODO post-T1: redeploy with current source
    royalty_distributor=None,         # TODO post-T1: redeploy with v2 (HIGH-1 burn fix + Pausable + D-04 pull-payment)
    foundation_safe=None,             # TODO post-T1: deployer EOA address (per §9 ratified decision)
    escrow_pool=None,                 # TODO post-T1
    stake_bond=None,                  # TODO post-T1
    settlement_registry=None,         # TODO post-T1
    signature_verifier=None,          # TODO post-T1
    emission_controller=None,         # TODO post-T1 (uses mainnet's 4-year halving; accelerated halving is task T10)
    compensation_distributor=None,    # TODO post-T1
    storage_slashing=None,            # TODO post-T1
    key_distribution=None,            # TODO post-T1
    publisher_key_anchor=None,        # TODO post-T1 (or reuse Phase 3.x.3 Sepolia deploy if applicable)
    notes=(
        "TESTNET — testnet-FTNS has zero monetary value.",
        "Foundation 'Safe' on testnet is a single deployer EOA, NOT the real 2-of-3 mainnet multisig.",
        "Halving curve uses mainnet's 4-year cadence (constant in EmissionController.sol). "
        "Accelerated-halving variant is a planned follow-up task (T10).",
        "Faucet: ask in #testnet-faucet on Discord; founder airdrops within 24h.",
    ),
)

# ────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────
NETWORK_CONFIGS: dict[str, NetworkConfig] = {
    "mainnet": MAINNET,
    "testnet": TESTNET,
}

DEFAULT_NETWORK = "mainnet"


def get_network_config(name: Optional[str] = None) -> NetworkConfig:
    """Look up a network by name. Falls back to DEFAULT_NETWORK if name is None.

    Raises KeyError if the name doesn't match any known network.
    """
    key = (name or DEFAULT_NETWORK).lower()
    if key not in NETWORK_CONFIGS:
        known = ", ".join(sorted(NETWORK_CONFIGS.keys()))
        raise KeyError(
            f"unknown network {name!r}; known networks: {known}"
        )
    return NETWORK_CONFIGS[key]
