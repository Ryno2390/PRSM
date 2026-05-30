"""Sprint 903 — pool-seed ceremony enforces Option A (Foundation never seeds).

Decision (this session): under the utility-token posture in
PRSM_Tokenomics.md §9.2 / §10 invariant #9 — "the foundation does not
provide liquidity, seed AMM pools, or operate swap venues" and "FTNS is
never sold by the foundation" — the USDC↔FTNS pool must be seeded by the
operating entity (Prismatica) or an independent third party, NOT the
PRSM Foundation Safe.

The sp875/876 ceremony tooling originally modeled the executor as the
"foundation_safe". This re-points it to a neutral "seeder_safe" and adds
a hard guard that REFUSES to build a seed ceremony whose executor is a
known Foundation Safe — encoding the Option-A securities decision in
code so it can't be undone by accident.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.aerodrome_pool_ceremony import (
    MAINNET_CONFIG,
    SEPOLIA_CONFIG,
    build_ceremony_batch,
    build_runbook_markdown,
)

# Canonical Foundation Safe (mainnet) — must NOT be the seeder.
_FOUNDATION_SAFE = "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
# A legitimate non-Foundation seeding Safe (e.g. Prismatica's).
_SEEDER = "0x" + "11" * 20


def _batch(**kw):
    base = dict(
        network=MAINNET_CONFIG,
        seeder_safe=_SEEDER,
        seed_usdc_units=50_000 * 10 ** 6,
        seed_ftns_units=200_000 * 10 ** 18,
    )
    base.update(kw)
    return build_ceremony_batch(**base)


# ── seeder_safe replaces foundation_safe ─────────────────────

def test_seeder_safe_param_builds_batch():
    batch = _batch()
    assert batch["meta"]["createdFromSafeAddress"] == _SEEDER
    # The addLiquidity `to` (LP recipient) is the seeder, not the
    # Foundation — the LP position lands with the seeding entity.
    assert _SEEDER[2:].lower() in batch["transactions"][2]["data"].lower()


def test_runbook_uses_seeder_safe():
    rb = build_runbook_markdown(
        network=MAINNET_CONFIG, seeder_safe=_SEEDER,
        seed_usdc_units=50_000 * 10 ** 6,
        seed_ftns_units=200_000 * 10 ** 18,
    )
    assert _SEEDER in rb


# ── Option-A guard: refuse a Foundation-Safe seeder ──────────

def test_ceremony_refuses_foundation_safe_seeder():
    with pytest.raises(ValueError, match="(?i)foundation"):
        _batch(seeder_safe=_FOUNDATION_SAFE)


def test_refuses_foundation_safe_case_insensitive():
    with pytest.raises(ValueError, match="(?i)foundation"):
        _batch(seeder_safe=_FOUNDATION_SAFE.lower())


def test_refuses_sepolia_foundation_address():
    # The sepolia/testnet foundation address must also be refused.
    with pytest.raises(ValueError, match="(?i)foundation"):
        build_ceremony_batch(
            network=SEPOLIA_CONFIG,
            seeder_safe="0xCCAc7b21695De068979b1ca47B0cfBD328654220",
            seed_usdc_units=1 * 10 ** 6,
            seed_ftns_units=4 * 10 ** 18,
        )


def test_runbook_refuses_foundation_safe_seeder():
    with pytest.raises(ValueError, match="(?i)foundation"):
        build_runbook_markdown(
            network=MAINNET_CONFIG, seeder_safe=_FOUNDATION_SAFE,
            seed_usdc_units=50_000 * 10 ** 6,
            seed_ftns_units=200_000 * 10 ** 18,
        )


# ── Runbook states the Option-A separation explicitly ────────

def test_runbook_documents_option_a_separation():
    rb = build_runbook_markdown(
        network=MAINNET_CONFIG, seeder_safe=_SEEDER,
        seed_usdc_units=50_000 * 10 ** 6,
        seed_ftns_units=200_000 * 10 ** 18,
    )
    low = rb.lower()
    # Must make clear the Foundation does NOT provide the liquidity.
    assert "foundation" in low and (
        "does not" in low or "not the foundation" in low
        or "not provide" in low
    )
