"""Sprint 587 — prsm_section7_readiness MCP tool wrapper.

Sprint 585 shipped `prsm node section7-readiness` CLI for shell-
accessible operators. Sprint 587 wraps the same probe set in an
MCP tool so AI triage agents (claude-in-chrome, etc.) can verify
§7 production readiness on a remote daemon without shell access.

Invariants:
- Tool registered in TOOLS list with name "prsm_section7_readiness"
- Handler returns a multiline string summary
- Handler tolerates env-unset (most common dev state) without
  raising
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def test_tool_registered_in_tools_list():
    """The TOOLS list must include prsm_section7_readiness."""
    from prsm.mcp_server import TOOLS
    names = {t.name for t in TOOLS}
    assert "prsm_section7_readiness" in names, (
        "Sprint 587: MCP tool prsm_section7_readiness not registered"
    )


def test_tool_handler_registered():
    """Handler dispatch table must include prsm_section7_readiness."""
    import prsm.mcp_server as mcp
    # The dispatch is `tools` dict at module bottom
    src = open(mcp.__file__).read()
    assert '"prsm_section7_readiness":' in src, (
        "Sprint 587: dispatch entry for prsm_section7_readiness missing"
    )


@pytest.mark.asyncio
async def test_handler_returns_summary_when_all_unset():
    """All envs unset → handler returns 'not_ready' summary."""
    keys = [
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
        "PRSM_STAKE_BOND_ADDRESS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        from prsm.mcp_server import handle_prsm_section7_readiness
        result = await handle_prsm_section7_readiness({})
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert isinstance(result, str)
    assert "not_ready" in result.lower() or "anchor" in result.lower()


@pytest.mark.asyncio
async def test_handler_includes_all_three_components_in_output():
    keys = ["PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", "PRSM_STAKE_BOND_ADDRESS"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        from prsm.mcp_server import handle_prsm_section7_readiness
        result = await handle_prsm_section7_readiness({})
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    text = result.lower()
    assert "anchor" in text
    assert "stake" in text
    assert "rpc" in text
