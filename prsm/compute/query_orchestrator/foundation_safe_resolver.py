"""Foundation Safe address resolution — per-network override.

Production deployments need the Foundation Safe address that's
load-bearing for the A6 beacon binding (per the threat model
§"Open governance questions" item 4). The mainnet Safe is at
`0x91b0000000000000000000000000000000005791` per memory entry
`project_phase1_3_task8_deploy_complete_2026_05_04.md`.

Testnet operators + alternative-network operators need a way to
override that default without forking node.py.

This module supplies the override:

  - Default: Base mainnet Safe address
  - Override: `PRSM_FOUNDATION_SAFE_ADDRESS` env var

The override is intentionally an explicit env var rather than a
chain-id-detection table because:

  1. Chain-id detection is footgun-y — getting the chain-id read
     wrong defaults to mainnet, which silently misroutes beacons.

  2. Multi-network deployments (e.g., a node serving both Base
     mainnet + Optimism via separate Foundation entities) need
     per-instance override anyway.

  3. The A6 beacon binding is forensic, not transactional — there's
     no contract call here to get wrong; the only risk is selecting
     the wrong beacon series, which the operator can always fix by
     restarting with the corrected env var.

Closes the Foundation-Safe-per-network-override placeholder follow-on
flagged in node.py B7 wiring.
"""
from __future__ import annotations

import os
from typing import Optional

# Base mainnet Foundation Safe — 2-of-3 hardware multisig deployed
# per project_phase1_3_task8_deploy_complete_2026_05_04.md memory.
# This is the default when PRSM_FOUNDATION_SAFE_ADDRESS is unset.
# Source-of-truth pin: prsm/config/networks.py MAINNET.foundation_safe.
# A regression test (test_default_matches_canonical_networks_py_address)
# enforces the equivalence so a divergence breaks CI immediately.
DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS: str = (
    "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
)


# Env var operators set to override the default (e.g. for testnet
# deployments or alternative networks).
FOUNDATION_SAFE_ADDRESS_ENV: str = "PRSM_FOUNDATION_SAFE_ADDRESS"


def resolve_foundation_safe_address(
    *, env_value: Optional[str] = None,
) -> str:
    """Resolve the Foundation Safe address for this deployment.

    Resolution order:
      1. ``env_value`` arg (test-friendly injection)
      2. ``PRSM_FOUNDATION_SAFE_ADDRESS`` env var
      3. ``DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS``

    Raises ``ValueError`` if the resolved value is empty or
    obviously malformed (not 0x-prefixed hex of 42 chars). The
    caller (typically `node.py:1714`-area construction) propagates
    the error up to the wiring-failure log; the orchestrator falls
    back to ``agent_forge=None`` per the existing B7 try/except
    guard.
    """
    if env_value is None:
        env_value = os.environ.get(FOUNDATION_SAFE_ADDRESS_ENV, "").strip()
    candidate = env_value or DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS

    if not candidate:
        raise ValueError(
            "Foundation Safe address resolved to empty string — "
            "either set PRSM_FOUNDATION_SAFE_ADDRESS or accept the "
            "Base mainnet default"
        )
    if not candidate.startswith("0x"):
        raise ValueError(
            f"Foundation Safe address must be 0x-prefixed hex, got "
            f"{candidate!r}"
        )
    # Ethereum addresses are 20 bytes = 40 hex chars + "0x" prefix.
    if len(candidate) != 42:
        raise ValueError(
            f"Foundation Safe address must be 42 chars (0x + 40 hex), "
            f"got {len(candidate)} chars: {candidate!r}"
        )
    return candidate
