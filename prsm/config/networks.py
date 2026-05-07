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
    royalty_distributor="0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2",
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
    # T1 deploy 2026-05-07 (post-HIGH-1/2 + MED-cluster + A-06 fixes; deployer
    # `PRSM Testnet Deployer` = 0xCCAc7b21695De068979b1ca47B0cfBD328654220):
    ftns_token="0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a",
    # Provenance + Royalty are Phase 1.3 surfaces deployed independently;
    # not in scope of this T1 rehearsal. Testnet versions can be deployed
    # later if/when content-registration flow needs validating on testnet.
    provenance_registry=None,
    royalty_distributor=None,
    foundation_safe="0xCCAc7b21695De068979b1ca47B0cfBD328654220",  # deployer EOA per §9 ratified decision
    escrow_pool="0xaa28b5818242608e04C1773c3e34bF7bFfb96248",
    stake_bond="0xF93aCa6551F408fFfe24292288d5488864D5264c",
    settlement_registry="0xF8BEEb4362222b50109b6034767322B31aA92449",
    signature_verifier="0x208dc98545Fe062d0B13Ac07b073633E6a62b5A9",  # production Ed25519Verifier
    emission_controller="0x30b6810F5653B99464AE6c2c2Ef37963bdbb0d99",
    compensation_distributor="0x18c875743DD722fBDd7a694A1644b502BC433DBB",
    storage_slashing="0x2ba1B361d2AD49f15F1131762fA3512d7824EB06",
    key_distribution="0xdB41A471AAC86285cD855bEdC27D7FC810dc3318",
    publisher_key_anchor=None,  # not yet deployed on Base Sepolia; Phase 3.x.3 Sepolia deploy was on Ethereum Sepolia
    notes=(
        "TESTNET — testnet-FTNS has zero monetary value.",
        "Foundation 'Safe' on testnet is a single deployer EOA, NOT the real 2-of-3 mainnet multisig.",
        "Foundation reserve wallet on StakeBond is set to the FTNS token address itself "
        "(passes MED-4 code.length>0 gate; foundation-share slashes accumulate passively at "
        "the FTNS contract — no economic-recovery path on testnet, but slashing flow IS exercisable).",
        "Halving curve uses mainnet's 4-year cadence (constant in EmissionController.sol). "
        "Accelerated-halving variant is a planned follow-up task (T10).",
        "Provenance + Royalty contracts are NOT deployed on testnet — content-registration "
        "flow remains mainnet-only for now.",
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
