"""Sprint 582 — /health/detailed trust-stack subsystem.

Sprint 579 added `prsm node trust-stack` CLI surfacing the 4
Phase-1 env-driven kinds. Sprint 582 mirrors that at the REST
layer so monitoring dashboards + MCP tools + downstream consumers
see the same view without spawning a CLI subprocess.

This test asserts the trust_stack subsystem is wired into
``/health/detailed`` via source-grep (full integration testing
of /health/detailed is impractical with MagicMock — too many
inner subsystems). Live attestation against a running daemon
covers behavioral verification.
"""
from __future__ import annotations


def test_health_detailed_source_includes_trust_stack_subsystem():
    """The /health/detailed handler in prsm/node/api.py must
    construct a 'trust_stack' subsystems entry with the four
    Phase-1 env-driven components.
    """
    from pathlib import Path
    src = (
        Path(__file__).parent.parent.parent
        / "prsm" / "node" / "api.py"
    ).read_text(encoding="utf-8")
    assert 'subsystems["trust_stack"]' in src, (
        "Sprint 582: trust_stack subsystem entry not wired into "
        "/health/detailed"
    )
    # All four Phase-1 env var names must appear in the section
    for envname in (
        "PRSM_PARALLAX_TRUST_STACK_KIND",
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND",
        "PRSM_PARALLAX_CONSENSUS_SUBMITTER_KIND",
        "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
    ):
        assert envname in src, (
            f"Sprint 582: env var {envname} missing from "
            f"/health/detailed trust_stack section"
        )


def test_trust_stack_subsystem_marked_as_optional():
    """The trust_stack entry must NOT be in the `core` list of
    /health/detailed (would let an operator env-typo flip the
    daemon to 'unhealthy'). Confirms by source grep — trust_stack
    is not present in the core subsystem list literal.
    """
    from pathlib import Path
    src = (
        Path(__file__).parent.parent.parent
        / "prsm" / "node" / "api.py"
    ).read_text(encoding="utf-8")
    # The core list literal in /health/detailed is
    #   core = ["ftns_ledger", "payment_escrow"]
    # Confirm trust_stack is not present in any "core = " line
    for line in src.splitlines():
        if line.strip().startswith("core = ["):
            assert "trust_stack" not in line, (
                "Sprint 582: trust_stack must NOT be in core[] — "
                "it's informational and operator env typos must "
                "not flip top-level health to unhealthy"
            )
