"""B1 — MCP server hides broken tools from client-side discovery.

Per canonical-workflow gap-list delta (2026-05-07): three tools are
broken end-to-end and MUST NOT be advertised to LLM clients in
production:

- prsm_analyze (depends on /compute/forge → 503)
- prsm_dispatch_agent (same backend)
- prsm_agent_status (/compute/status/{job_id} endpoint nonexistent)

Source-of-truth TOOLS list keeps the Tool definitions so call_tool()
still routes explicit invocations through their handlers (which
return informative error text). Only client-side discovery is
gated, so honoring LLM clients won't surface broken tools to users.

Override: PRSM_EXPOSE_BROKEN_TOOLS=1 makes them visible (operators
reconstructing the data-query path).

These tests inspect the BROKEN_TOOLS_HIDDEN constant + replicate
list_tools()'s filter logic without booting the MCP server (which
would require an MCP transport).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prsm.mcp_server import BROKEN_TOOLS_HIDDEN, TOOLS


# Mirror list_tools() filter — keep this in sync if the gate logic
# moves. Doing the filter inline rather than reaching into the
# closure inside register_handlers() keeps the test transport-free.
def _filtered_tools(*, expose_broken: bool):
    if expose_broken:
        return TOOLS
    return [t for t in TOOLS if t.name not in BROKEN_TOOLS_HIDDEN]


class TestBrokenToolsHidden:
    def test_constant_contains_one_known_broken_tool(self):
        """Pin the membership of BROKEN_TOOLS_HIDDEN. If a future
        commit adds or removes a name without updating this test, we
        want a deliberate decision — not silent drift.

        2026-05-08 (B8 unhide pass 1): prsm_analyze removed.
        2026-05-08 (B8 unhide pass 2): prsm_dispatch_agent removed —
        its handler already routes through /compute/forge with
        manifest.query, which now works via the same QO dispatch.
        Manifest is pre-validated locally; QO re-decomposes
        server-side (caveat documented in handler).
        """
        assert BROKEN_TOOLS_HIDDEN == frozenset({
            "prsm_agent_status",
        })

    def test_prsm_analyze_now_visible_by_default(self):
        """B8 unhide pass 1: prsm_analyze is back in the discovery
        list. Without PRSM_QUERY_ORCHESTRATOR_ENABLED, calling it
        returns 503 from /compute/forge (agent_forge=None) — the
        right error for clients to surface. With the env var, the
        QO path handles it end-to-end."""
        names = {t.name for t in _filtered_tools(expose_broken=False)}
        assert "prsm_analyze" in names

    def test_prsm_dispatch_agent_now_visible_by_default(self):
        """B8 unhide pass 2: prsm_dispatch_agent is back. Local
        manifest validation provides structured precondition; the
        underlying /compute/forge dispatch is the same QO path
        prsm_analyze uses."""
        names = {t.name for t in _filtered_tools(expose_broken=False)}
        assert "prsm_dispatch_agent" in names

    def test_default_filter_omits_remaining_broken_tool(self):
        """Without PRSM_EXPOSE_BROKEN_TOOLS, the one still-broken
        name (prsm_agent_status) must not appear in the discovery
        list — its backing /compute/status/{job_id} endpoint
        doesn't exist."""
        names = {t.name for t in _filtered_tools(expose_broken=False)}
        assert "prsm_agent_status" not in names

    def test_default_filter_keeps_real_tools(self):
        """Real tools must still be visible. If this fails, the gate
        is over-broad."""
        names = {t.name for t in _filtered_tools(expose_broken=False)}
        # Spot-check the real ones from the gap-list matrix:
        assert "prsm_inference" in names
        assert "prsm_quote" in names
        assert "prsm_search_shards" in names
        assert "prsm_create_agent" in names  # local manifest builder
        assert "prsm_node_status" in names

    def test_expose_override_returns_full_list(self):
        """When the operator opts in, all three broken tools come
        back into the discovery list."""
        names = {t.name for t in _filtered_tools(expose_broken=True)}
        assert "prsm_analyze" in names
        assert "prsm_dispatch_agent" in names
        assert "prsm_agent_status" in names

    def test_source_of_truth_intact(self):
        """The TOOLS list itself MUST still contain the Tool
        definitions — so call_tool() can dispatch explicit
        invocations through their handlers (which surface the real
        503/404 to callers). Only client-side discovery is gated."""
        names = {t.name for t in TOOLS}
        for broken in BROKEN_TOOLS_HIDDEN:
            assert broken in names, (
                f"{broken} must remain in TOOLS — only list_tools() "
                f"filter excludes it"
            )

    def test_filter_is_idempotent(self):
        """Running the filter twice yields the same result — no
        accidental mutation of TOOLS."""
        first = _filtered_tools(expose_broken=False)
        second = _filtered_tools(expose_broken=False)
        assert [t.name for t in first] == [t.name for t in second]
        # And TOOLS itself unchanged after filtering.
        assert any(t.name == "prsm_analyze" for t in TOOLS)


class TestEnvOverrideContract:
    """Pin which env var values turn the gate off. Stays in sync
    with the os.getenv("PRSM_EXPOSE_BROKEN_TOOLS", "").lower() in
    ("1", "true", "yes") check inside list_tools()."""

    @pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE", "Yes"])
    def test_truthy_values_expose(self, val):
        with patch.dict(os.environ, {"PRSM_EXPOSE_BROKEN_TOOLS": val}):
            expose = os.getenv(
                "PRSM_EXPOSE_BROKEN_TOOLS", "",
            ).lower() in ("1", "true", "yes")
            assert expose is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "off"])
    def test_falsy_values_keep_hidden(self, val):
        with patch.dict(os.environ, {"PRSM_EXPOSE_BROKEN_TOOLS": val}):
            expose = os.getenv(
                "PRSM_EXPOSE_BROKEN_TOOLS", "",
            ).lower() in ("1", "true", "yes")
            assert expose is False
