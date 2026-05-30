"""Sprint 875 — Aerodrome USDC↔FTNS pool ceremony helper.

The Foundation Safe USDC↔FTNS pool seeding ceremony (Vision gantt
2026-06-15) is the single most economically significant action in
PRSM's deployment: the seed ratio sets the opening market price for
FTNS, and the deposit is irreversible (LP tokens to Safe but funds
subject to market dynamics thereafter). Sp875 de-risks the ceremony
by:

  1. Generating the exact transaction batch the multi-sig will
     execute (USDC.approve + FTNS.approve + Router.addLiquidity)
     as Safe-Transaction-Builder-compatible JSON
  2. Producing a markdown runbook the co-signers can review
     line-by-line before signing
  3. Supporting both Sepolia rehearsal (testnet — zero real-money
     risk) and mainnet execution from the same code path

Output artifacts are PURE PAYLOAD — this module never signs or
submits. Operators upload the JSON to Safe Transaction Builder
(via wallet.safe.global UI) where hardware-wallet co-signers sign
+ execute.

The Sepolia rehearsal is the load-bearing risk-reduction step:
catches operator-procedure bugs (wrong address pasted, wrong
approval amount, wrong slippage) where they cost ~$0 instead of
~$50k.

Cost projection:
  Gas on Base mainnet ~$1-2 per tx × 3 txs = ~$3-6 total gas.
  Seed amounts are at Foundation Safe operator discretion + set
  the opening price. (Sp875 doesn't presume to choose seed sizes.)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── Canonical Base mainnet addresses ─────────────────────────

# Match sp855's pins exactly — drift here = ceremony hits wrong
# contracts.
AERODROME_ROUTER_V2_MAINNET = (
    "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43"
)
AERODROME_POOL_FACTORY_MAINNET = (
    "0x420DD381b31aEf6683db6B902084cB0FFECe40Da"
)
USDC_BASE_MAINNET = (
    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
)
FTNS_BASE_MAINNET = (
    "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
)

# Base Sepolia (testnet) — rehearsal addresses. Testnet USDC is
# Circle's Sepolia deployment.
USDC_BASE_SEPOLIA = (
    "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
)
FTNS_BASE_SEPOLIA = (
    "0x7F5f00FAA2421c4C585cc66c87420b1659c98e6a"
)
# Aerodrome's Sepolia deployment — operator must verify these
# addresses are still current before rehearsal (Aerodrome's
# testnet deployments are sometimes redeployed without notice).
AERODROME_ROUTER_V2_SEPOLIA = (
    "0x0000000000000000000000000000000000000000"  # placeholder
)
AERODROME_POOL_FACTORY_SEPOLIA = (
    "0x0000000000000000000000000000000000000000"  # placeholder
)


# ── Selectors ────────────────────────────────────────────────

SELECTOR_ERC20_APPROVE = bytes.fromhex("095ea7b3")
# Aerodrome Router.addLiquidity(tokenA, tokenB, stable, amountADesired,
# amountBDesired, amountAMin, amountBMin, to, deadline)
SELECTOR_AERODROME_ADD_LIQUIDITY = bytes.fromhex("e8e33700")


# ── Network bundle ───────────────────────────────────────────

@dataclass
class CeremonyNetworkConfig:
    name: str  # "mainnet" or "sepolia"
    chain_id: int
    usdc_address: str
    ftns_address: str
    router_address: str
    factory_address: str


MAINNET_CONFIG = CeremonyNetworkConfig(
    name="mainnet", chain_id=8453,
    usdc_address=USDC_BASE_MAINNET,
    ftns_address=FTNS_BASE_MAINNET,
    router_address=AERODROME_ROUTER_V2_MAINNET,
    factory_address=AERODROME_POOL_FACTORY_MAINNET,
)

SEPOLIA_CONFIG = CeremonyNetworkConfig(
    name="sepolia", chain_id=84532,
    usdc_address=USDC_BASE_SEPOLIA,
    ftns_address=FTNS_BASE_SEPOLIA,
    router_address=AERODROME_ROUTER_V2_SEPOLIA,
    factory_address=AERODROME_POOL_FACTORY_SEPOLIA,
)


# Sprint 903 — Option-A guard. Per PRSM_Tokenomics.md §9.2 / §10
# invariant #9, the PRSM Foundation does NOT seed AMM pools, provide
# liquidity, or sell FTNS — the pool is seeded by the operating entity
# (Prismatica) or an independent third party. These are the known
# Foundation Safe / treasury addresses that must never be the ceremony
# seeder. Compared case-insensitively.
_KNOWN_FOUNDATION_SAFES = frozenset({
    "0x91b0e6f85a371d82de94ed13a3812d9f5a4e5791",  # mainnet Foundation Safe
    "0xccac7b21695de068979b1ca47b0cfbd328654220",  # sepolia/testnet foundation addr
})


def _assert_seeder_not_foundation(seeder_safe: str) -> None:
    """Refuse to build a seed ceremony whose executor is a known
    Foundation Safe. Option A (this session): the Foundation must not
    provide pool liquidity — the seeding entity is Prismatica or an
    independent third party."""
    if (seeder_safe or "").strip().lower() in _KNOWN_FOUNDATION_SAFES:
        raise ValueError(
            f"seeder_safe {seeder_safe!r} is a known PRSM Foundation "
            "Safe. Per Option A (PRSM_Tokenomics.md §9.2 / §10 #9) the "
            "Foundation does not seed AMM pools or provide liquidity — "
            "seed from the operating entity (Prismatica) or an "
            "independent third party. Transfer the FTNS to that "
            "entity's Safe first, then run the ceremony from it."
        )


# ── ABI encoding helpers ─────────────────────────────────────

def _padded_address(addr: str) -> bytes:
    return bytes(12) + bytes.fromhex(addr.lower().removeprefix("0x"))


def _u256_to_bytes(n: int) -> bytes:
    return n.to_bytes(32, "big")


def _bool_to_bytes(b: bool) -> bytes:
    return _u256_to_bytes(1 if b else 0)


def encode_approve_calldata(spender: str, amount: int) -> bytes:
    """ERC-20 approve(spender, amount)"""
    return (
        SELECTOR_ERC20_APPROVE
        + _padded_address(spender)
        + _u256_to_bytes(amount)
    )


def encode_add_liquidity_calldata(
    *,
    token_a: str,
    token_b: str,
    stable: bool,
    amount_a_desired: int,
    amount_b_desired: int,
    amount_a_min: int,
    amount_b_min: int,
    to: str,
    deadline: int,
) -> bytes:
    """Aerodrome Router.addLiquidity(token_a, token_b, stable, ...)"""
    return (
        SELECTOR_AERODROME_ADD_LIQUIDITY
        + _padded_address(token_a)
        + _padded_address(token_b)
        + _bool_to_bytes(stable)
        + _u256_to_bytes(amount_a_desired)
        + _u256_to_bytes(amount_b_desired)
        + _u256_to_bytes(amount_a_min)
        + _u256_to_bytes(amount_b_min)
        + _padded_address(to)
        + _u256_to_bytes(deadline)
    )


# ── Safe Transaction Builder JSON ────────────────────────────

def build_ceremony_batch(
    *,
    network: CeremonyNetworkConfig,
    seeder_safe: str,
    seed_usdc_units: int,    # base units, 6 decimals
    seed_ftns_units: int,    # base units, 18 decimals
    slippage_bps: int = 100,  # 1% default
    deadline_seconds: int = 3600,  # 1h default
    stable: bool = False,
) -> Dict[str, Any]:
    """Build the canonical Safe-Transaction-Builder JSON for the
    3-tx ceremony batch.

    ``seeder_safe`` is the Safe that EXECUTES the seed and receives the
    LP position. Under Option A (PRSM_Tokenomics.md §9.2 / §10 #9) this
    is the operating entity (Prismatica) or an independent third party
    — NOT the PRSM Foundation Safe, which must not provide pool
    liquidity. Building with a known Foundation Safe raises ValueError.

    Returns a dict suitable for `json.dumps` then upload via Safe
    UI → Transaction Builder → "Load from JSON".
    """
    _assert_seeder_not_foundation(seeder_safe)
    if seed_usdc_units <= 0 or seed_ftns_units <= 0:
        raise ValueError("seed amounts must be > 0")
    if not (0 <= slippage_bps <= 10_000):
        raise ValueError("slippage_bps must be in [0, 10000]")

    # Pool ordering: token_a + token_b must be sorted by address.
    # Aerodrome enforces this; if reversed, the call reverts.
    usdc = network.usdc_address
    ftns = network.ftns_address
    if int(usdc, 16) < int(ftns, 16):
        token_a, token_b = usdc, ftns
        amount_a, amount_b = seed_usdc_units, seed_ftns_units
    else:
        token_a, token_b = ftns, usdc
        amount_a, amount_b = seed_ftns_units, seed_usdc_units

    amount_a_min = (
        amount_a * (10_000 - slippage_bps) // 10_000
    )
    amount_b_min = (
        amount_b * (10_000 - slippage_bps) // 10_000
    )

    # Deadline: now + N. Operators should regenerate close to
    # ceremony time to avoid stale deadlines.
    deadline = int(time.time()) + deadline_seconds

    # Tx 1: USDC.approve(Router, seed_usdc_units)
    tx1_data = encode_approve_calldata(
        network.router_address, seed_usdc_units,
    )
    # Tx 2: FTNS.approve(Router, seed_ftns_units)
    tx2_data = encode_approve_calldata(
        network.router_address, seed_ftns_units,
    )
    # Tx 3: Router.addLiquidity(token_a, token_b, stable=False,
    #                           desired, min, to=Safe, deadline)
    tx3_data = encode_add_liquidity_calldata(
        token_a=token_a, token_b=token_b,
        stable=stable,
        amount_a_desired=amount_a, amount_b_desired=amount_b,
        amount_a_min=amount_a_min,
        amount_b_min=amount_b_min,
        to=seeder_safe, deadline=deadline,
    )

    return {
        "version": "1.0",
        "chainId": str(network.chain_id),
        "createdAt": int(time.time() * 1000),
        "meta": {
            "name": f"PRSM Aerodrome USDC↔FTNS Seed ({network.name})",
            "description": (
                f"Sp875/903 ceremony — seeding entity (Option A: NOT "
                f"the Foundation Safe) seeds Aerodrome USDC↔FTNS "
                f"volatile pool with "
                f"{seed_usdc_units / 10**6:.6f} USDC + "
                f"{seed_ftns_units / 10**18:.6f} FTNS. "
                f"slippage_bps={slippage_bps}, "
                f"deadline=+{deadline_seconds}s."
            ),
            "txBuilderVersion": "1.16.5",
            "createdFromSafeAddress": seeder_safe,
        },
        "transactions": [
            {
                "to": network.usdc_address,
                "value": "0",
                "data": "0x" + tx1_data.hex(),
            },
            {
                "to": network.ftns_address,
                "value": "0",
                "data": "0x" + tx2_data.hex(),
            },
            {
                "to": network.router_address,
                "value": "0",
                "data": "0x" + tx3_data.hex(),
            },
        ],
    }


def build_runbook_markdown(
    *,
    network: CeremonyNetworkConfig,
    seeder_safe: str,
    seed_usdc_units: int,
    seed_ftns_units: int,
    slippage_bps: int = 100,
) -> str:
    """Generate the co-signer runbook as markdown.

    ``seeder_safe`` is the operating-entity (Prismatica) or
    independent-third-party Safe that seeds the pool — NOT the PRSM
    Foundation Safe (Option A; see build_ceremony_batch). Operator
    distributes to hardware-wallet co-signers BEFORE the ceremony so
    they can verify each transaction byte-for-byte independently.
    """
    _assert_seeder_not_foundation(seeder_safe)
    usdc_whole = seed_usdc_units / 10**6
    ftns_whole = seed_ftns_units / 10**18
    opening_price = (
        usdc_whole / ftns_whole if ftns_whole > 0 else 0
    )
    lines = [
        f"# PRSM Aerodrome USDC↔FTNS Pool Seeding Ceremony — "
        f"{network.name.upper()}",
        "",
        "## Ceremony Summary",
        "",
        f"- **Network**: {network.name} (chain_id {network.chain_id})",
        f"- **Seeding Safe (operating entity / Prismatica — NOT the "
        f"Foundation)**: `{seeder_safe}`",
        "- **Option A**: the PRSM Foundation does NOT seed this pool, "
        "provide liquidity, or sell FTNS (PRSM_Tokenomics.md §9.2 / "
        "§10 #9). The FTNS must be transferred from the Foundation "
        "treasury to the seeding entity's Safe BEFORE this ceremony.",
        f"- **USDC seed**: {usdc_whole:.6f} USDC "
        f"({seed_usdc_units} base units, 6 decimals)",
        f"- **FTNS seed**: {ftns_whole:.6f} FTNS "
        f"({seed_ftns_units} base units, 18 decimals)",
        f"- **Implied opening price**: "
        f"${opening_price:.6f} per FTNS",
        f"- **Slippage tolerance**: {slippage_bps} bps "
        f"({slippage_bps / 100}%)",
        f"- **Pool type**: Volatile (Aerodrome supports stable +"
        f" volatile; USDC↔FTNS goes volatile because FTNS price "
        f"will fluctuate independently of USDC's $1 peg)",
        "",
        "## Pre-Flight Checks (Co-Signers MUST verify before signing)",
        "",
        f"1. Seeding Safe address matches `{seeder_safe}` "
        "on the Safe UI (this is the operating entity's Safe, NOT the "
        "Foundation Safe)",
        f"2. Seeding Safe currently holds AT LEAST "
        f"{usdc_whole:.6f} USDC + {ftns_whole:.6f} FTNS "
        "(FTNS transferred from the Foundation treasury beforehand)",
        f"3. USDC contract address is `{network.usdc_address}` — "
        "verify on basescan.org against canonical Circle deployment",
        f"4. FTNS contract address is `{network.ftns_address}` — "
        "verify on basescan.org matches PRSM's published address",
        f"5. Aerodrome Router address is `{network.router_address}` "
        "— verify on basescan.org against canonical Aerodrome v2 "
        "deployment",
        f"6. Aerodrome Pool Factory address is "
        f"`{network.factory_address}` — verify on basescan.org",
        "",
        "## Transaction Batch (3 calls in 1 Safe transaction)",
        "",
        f"**Tx 1**: `USDC.approve({network.router_address}, "
        f"{seed_usdc_units})`",
        f"   - Allows Router to pull {usdc_whole:.6f} USDC "
        "from the Safe",
        "",
        f"**Tx 2**: `FTNS.approve({network.router_address}, "
        f"{seed_ftns_units})`",
        f"   - Allows Router to pull {ftns_whole:.6f} FTNS "
        "from the Safe",
        "",
        f"**Tx 3**: `Router.addLiquidity(...)` — seeds the pool",
        f"   - Mints LP tokens to the Safe representing the "
        "seeding entity's share of the pool",
        f"   - Deadline: +1h from ceremony start (regenerate "
        "JSON close to ceremony time)",
        "",
        "## Execution Steps",
        "",
        "1. Open https://wallet.safe.global → connect to the Seeding "
        "Safe (operating entity, NOT the Foundation Safe)",
        "2. Apps tab → Transaction Builder",
        "3. **Load from JSON** → upload `ceremony-batch.json` "
        "(generated by this module)",
        "4. **Review each call** byte-for-byte against this runbook",
        "5. **2-of-3 (or whatever your threshold) co-signers sign** "
        "via hardware wallets",
        "6. Execute → wait for inclusion (~1-2 blocks on Base)",
        "7. **Post-ceremony**: verify Safe holds LP tokens via "
        "etherscan-style explorer at the pool address (factory "
        "auto-creates pool at deterministic address on first "
        "addLiquidity call)",
        "",
        "## Post-Ceremony PRSM Wiring",
        "",
        "Once the pool address is confirmed on-chain, set the "
        "operator env var:",
        "",
        "```",
        "AERODROME_USDC_FTNS_POOL_ADDRESS=0x<pool address>",
        "BASE_RPC_URL=<paid or default mainnet.base.org>",
        "```",
        "",
        "Restart all PRSM operator daemons. The sp859 readiness "
        "aggregator should flip `aerodrome.live_exec: True` and "
        "the sp871 onramp→swap orchestrator will start building "
        "envelopes for already-CONFIRMED intents on the next "
        "sweep cycle.",
        "",
        "## Rollback / Abort",
        "",
        "If anything looks wrong during Safe UI review BEFORE "
        "the 2nd signature lands:",
        "- Any co-signer can reject + the tx never executes",
        "- After 1st signature: 1st signer can rotate keys / "
        "withdraw signature (Safe UI supports this)",
        "- After execution: irreversible — Safe holds LP tokens "
        "which can be withdrawn but the swap impact is permanent",
        "",
        "## Sepolia Rehearsal (STRONGLY RECOMMENDED)",
        "",
        "Run this same module against Sepolia config FIRST with "
        "throwaway amounts (e.g., 1 USDC + 1 FTNS):",
        "",
        "```python",
        "from prsm.economy.web3.aerodrome_pool_ceremony import (",
        "    SEPOLIA_CONFIG, build_ceremony_batch,",
        "    build_runbook_markdown,",
        ")",
        "batch = build_ceremony_batch(",
        "    network=SEPOLIA_CONFIG,",
        "    seeder_safe='<your sepolia test safe — NOT the Foundation>',",
        "    seed_usdc_units=1_000_000,  # 1 USDC",
        "    seed_ftns_units=1 * 10**18,  # 1 FTNS",
        ")",
        "```",
        "",
        "Catches operator-procedure bugs (wrong address pasted, "
        "wrong approval amount, wrong slippage) where they cost "
        "~$0 instead of ~$50k.",
    ]
    return "\n".join(lines) + "\n"
