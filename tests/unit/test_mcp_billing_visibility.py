"""Tests for per-tool billing visibility (Phase 3.x.1 Task 7).

Verifies that all 4 FTNS-consuming tools surface a cost-reconciliation
footer with the canonical fields and that prsm_billing_status correctly
queries node escrow state.
"""

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOLS,
    TOOL_HANDLERS,
    _format_cost_footer,
    handle_prsm_analyze,
    handle_prsm_billing_status,
    handle_prsm_create_agent,
    handle_prsm_dispatch_agent,
    handle_prsm_inference,
)


# ── Footer helper ──────────────────────────────────────────────────────────


class TestFormatCostFooter:
    def test_minimal_footer(self):
        out = _format_cost_footer(job_id="forge-abc")
        assert "forge-abc" in out
        assert "prsm_billing_status" in out
        # Two horizontal rules
        assert out.count("—" * 60) == 2

    def test_cost_takes_precedence_over_budget(self):
        out = _format_cost_footer(job_id="x", cost_ftns="1.5", budget_ftns=10.0)
        assert "Cost:" in out
        assert "1.5" in out
        assert "Budget reserved" not in out

    def test_budget_used_when_cost_missing(self):
        out = _format_cost_footer(job_id="x", cost_ftns=None, budget_ftns=5.0)
        assert "Budget reserved" in out
        assert "5.0" in out
        assert "Cost:" not in out

    def test_extra_fields_rendered(self):
        out = _format_cost_footer(
            job_id="x",
            cost_ftns="0.5",
            extra_fields={"Route": "swarm", "Privacy level": "standard"},
        )
        assert "Route:" in out
        assert "swarm" in out
        assert "Privacy level:" in out

    def test_note_appended_below_rule(self):
        out = _format_cost_footer(job_id="x", note="Manifest only — no FTNS consumed.")
        # Note appears AFTER the second rule
        rule = "—" * 60
        last_rule_idx = out.rfind(rule)
        note_idx = out.rfind("Manifest only")
        assert note_idx > last_rule_idx

    def test_reconcile_via_uses_actual_job_id(self):
        out = _format_cost_footer(job_id="custom-job-99")
        assert 'prsm_billing_status(job_id="custom-job-99")' in out


# ── Tool definitions ──────────────────────────────────────────────────────


class TestPrsmBillingStatusToolDefinition:
    def _tool(self):
        return next(t for t in TOOLS if t.name == "prsm_billing_status")

    def test_tool_exists(self):
        names = [t.name for t in TOOLS]
        assert "prsm_billing_status" in names

    def test_tool_count_now_eighteen(self):
        # 17 prior + prsm_billing_status = 18
        assert len(TOOLS) == 18

    def test_required_fields(self):
        schema = self._tool().inputSchema
        assert schema["required"] == ["job_id"]

    def test_handler_registered(self):
        assert TOOL_HANDLERS["prsm_billing_status"] is handle_prsm_billing_status


# ── prsm_billing_status handler ────────────────────────────────────────────


class TestPrsmBillingStatusHandler:
    @pytest.mark.asyncio
    async def test_missing_job_id_returns_error(self):
        result = await handle_prsm_billing_status({})
        assert "Missing required 'job_id'" in result

    @pytest.mark.asyncio
    async def test_empty_job_id_returns_error(self):
        result = await handle_prsm_billing_status({"job_id": "   "})
        assert "Missing required 'job_id'" in result

    @pytest.mark.asyncio
    async def test_node_unreachable_returns_helpful_error(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.side_effect = Exception("Connection refused")
            result = await handle_prsm_billing_status({"job_id": "infer-123"})
        assert "Failed to query billing" in result
        assert "infer-123" in result
        assert "prsm node start" in result

    @pytest.mark.asyncio
    async def test_404_response_pass_through(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"detail": "No escrow found for job_id=infer-zzz"}
            result = await handle_prsm_billing_status({"job_id": "infer-zzz"})
        assert "No escrow found" in result
        assert "infer-zzz" in result

    @pytest.mark.asyncio
    async def test_successful_response_formatted(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "job_id": "infer-abc-123",
                "escrow_id": "escrow-xyz",
                "amount_ftns": 0.5,
                "status": "released",
                "requester_id": "node-r-1",
                "provider_winner": "node-p-7",
                "tx_lock": "0xabc1",
                "tx_release": "0xdef2",
                "created_at": 1745625600.0,
                "completed_at": 1745625610.5,
            }
            result = await handle_prsm_billing_status({"job_id": "infer-abc-123"})
        assert "PRSM Billing Status" in result
        assert "infer-abc-123" in result
        assert "released" in result
        assert "0.5" in result
        assert "node-p-7" in result
        assert "0xabc1" in result
        assert "0xdef2" in result

    @pytest.mark.asyncio
    async def test_pending_response_omits_release_fields(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "job_id": "infer-pending",
                "escrow_id": "esc-1",
                "amount_ftns": 1.0,
                "status": "pending",
                "requester_id": "node-r",
                "provider_winner": None,
                "tx_lock": None,
                "tx_release": None,
                "created_at": 1745625600.0,
                "completed_at": None,
            }
            result = await handle_prsm_billing_status({"job_id": "infer-pending"})
        assert "pending" in result
        # None fields should NOT appear as labels
        assert "Provider:" not in result
        assert "Lock tx:" not in result
        assert "Release tx:" not in result
        assert "Completed at:" not in result

    @pytest.mark.asyncio
    async def test_handler_calls_correct_endpoint(self):
        captured: dict = {}

        async def capture(method, path, data=None):
            captured["method"] = method
            captured["path"] = path
            return {"job_id": "x", "escrow_id": "e", "status": "released",
                    "amount_ftns": 1.0, "requester_id": "r"}

        with patch("prsm.mcp_server._call_node_api", side_effect=capture):
            await handle_prsm_billing_status({"job_id": "infer-test-99"})

        assert captured["method"] == "GET"
        assert captured["path"] == "/billing/infer-test-99"


# ── Footer presence in FTNS-consuming tools ───────────────────────────────


class TestCostFooterPresence:
    """Each of the 4 FTNS-consuming tools surfaces the cost footer."""

    @pytest.mark.asyncio
    async def test_prsm_inference_includes_footer(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "success": True,
                "output": "answer",
                "receipt": {
                    "job_id": "infer-i1",
                    "model_id": "mock-llama-3-8b",
                    "privacy_tier": "standard",
                    "content_tier": "A",
                    "tee_type": "software",
                    "epsilon_spent": 8.0,
                    "cost_ftns": "0.1",
                    "duration_seconds": 0.5,
                    "settler_node_id": "node-x",
                    "settler_signature": "deadbeef",
                },
            }
            result = await handle_prsm_inference({
                "prompt": "x", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0,
            })
        assert 'prsm_billing_status(job_id="infer-i1")' in result
        assert "Cost:" in result and "0.1" in result
        assert result.count("—" * 60) == 2

    @pytest.mark.asyncio
    async def test_prsm_analyze_includes_footer(self):
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "response": "answer", "route": "swarm", "job_id": "forge-a1",
            }
            result = await handle_prsm_analyze({
                "query": "EV trends", "budget_ftns": 5.0,
            })
        assert 'prsm_billing_status(job_id="forge-a1")' in result
        assert "Route:" in result and "swarm" in result
        assert "Budget reserved:" in result and "5.0" in result

    @pytest.mark.asyncio
    async def test_prsm_create_agent_includes_footer(self):
        result = await handle_prsm_create_agent({
            "query": "test agent",
            "instructions": [{"op": "filter", "field": "x", "value": "y"}],
            "budget_ftns": 5.0,
        })
        # No real job_id yet — manifest-only
        assert "(none — assigned at dispatch)" in result
        assert "Budget reserved:" in result
        assert "5.0" in result
        # Note explains no FTNS consumed (case-insensitive — output lowercased)
        assert "no ftns consumed yet" in result.lower()

    @pytest.mark.asyncio
    async def test_prsm_dispatch_agent_includes_footer(self):
        from prsm.compute.agents.instruction_set import (
            AgentOp, AgentInstruction, InstructionManifest,
        )
        manifest = InstructionManifest(
            query="test",
            instructions=[AgentInstruction(op=AgentOp("filter"), field="f", value="v")],
        )
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "response": "agent result", "route": "swarm", "job_id": "forge-d1",
            }
            result = await handle_prsm_dispatch_agent({
                "instructions_json": manifest.to_json(),
                "budget_ftns": 5.0,
            })
        assert 'prsm_billing_status(job_id="forge-d1")' in result
        assert "Route:" in result and "swarm" in result
        assert "Operations:" in result


# ── Acceptance criteria for Phase 3.x.1 Task 7 ────────────────────────────


class TestTask7Acceptance:
    """Validates the explicit acceptance criteria from Phase 3.x.1 Task 7.

    Acceptance: All 4 FTNS-consuming tools (prsm_analyze, prsm_inference,
    prsm_create_agent, prsm_dispatch_agent) show cost footer.
    """

    @pytest.mark.asyncio
    async def test_all_four_ftns_tools_show_footer(self):
        # Each tool's response must include the canonical reconcile-via line
        from prsm.compute.agents.instruction_set import (
            AgentOp, AgentInstruction, InstructionManifest,
        )

        outputs = []

        # prsm_inference
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {
                "success": True, "output": "o",
                "receipt": {"job_id": "infer-test",
                            "model_id": "m", "privacy_tier": "none",
                            "content_tier": "A", "tee_type": "software",
                            "epsilon_spent": 0, "cost_ftns": "0",
                            "duration_seconds": 0, "settler_node_id": "s",
                            "settler_signature": ""},
            }
            outputs.append(("prsm_inference", await handle_prsm_inference(
                {"prompt": "x", "budget_ftns": 1.0}
            )))

        # prsm_analyze
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"response": "x", "route": "r", "job_id": "j"}
            outputs.append(("prsm_analyze", await handle_prsm_analyze(
                {"query": "q", "budget_ftns": 1.0}
            )))

        # prsm_create_agent (no API call needed)
        outputs.append(("prsm_create_agent", await handle_prsm_create_agent({
            "query": "x",
            "instructions": [{"op": "count"}],
            "budget_ftns": 1.0,
        })))

        # prsm_dispatch_agent
        manifest = InstructionManifest(
            query="x",
            instructions=[AgentInstruction(op=AgentOp("count"))],
        )
        with patch("prsm.mcp_server._call_node_api") as mock_call:
            mock_call.return_value = {"response": "x", "route": "r", "job_id": "d"}
            outputs.append(("prsm_dispatch_agent", await handle_prsm_dispatch_agent(
                {"instructions_json": manifest.to_json(), "budget_ftns": 1.0}
            )))

        # Validate: each output has the canonical footer marker
        for tool_name, output in outputs:
            assert "prsm_billing_status" in output, \
                f"{tool_name} missing prsm_billing_status reference"
            assert "—" * 60 in output, \
                f"{tool_name} missing horizontal rule"

    @pytest.mark.asyncio
    async def test_billing_status_queries_escrow_correctly(self):
        captured: dict = {}

        async def capture(method, path, data=None):
            captured["method"] = method
            captured["path"] = path
            return {"job_id": "x", "escrow_id": "e", "status": "released",
                    "amount_ftns": 1.0, "requester_id": "r"}

        with patch("prsm.mcp_server._call_node_api", side_effect=capture):
            await handle_prsm_billing_status({"job_id": "infer-acceptance"})

        # Acceptance: prsm_billing_status queries escrow correctly
        assert captured["method"] == "GET"
        assert captured["path"] == "/billing/infer-acceptance"
