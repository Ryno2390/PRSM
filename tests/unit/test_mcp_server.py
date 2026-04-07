"""Tests for the PRSM MCP Server."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from prsm.mcp_server import (
    create_server,
    TOOLS,
    TOOL_HANDLERS,
    handle_prsm_quote,
    handle_prsm_hardware_benchmark,
    handle_prsm_analyze,
    handle_prsm_node_status,
    handle_prsm_create_agent,
    handle_prsm_dispatch_agent,
)


class TestMCPToolDefinitions:
    def test_seven_tools_defined(self):
        assert len(TOOLS) == 7

    def test_tool_names(self):
        names = [t.name for t in TOOLS]
        assert "prsm_analyze" in names
        assert "prsm_quote" in names
        assert "prsm_list_datasets" in names
        assert "prsm_node_status" in names
        assert "prsm_hardware_benchmark" in names
        assert "prsm_create_agent" in names
        assert "prsm_dispatch_agent" in names

    def test_all_tools_have_descriptions(self):
        for tool in TOOLS:
            assert len(tool.description) > 20, f"Tool {tool.name} has short description"

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"

    def test_analyze_requires_query(self):
        analyze = next(t for t in TOOLS if t.name == "prsm_analyze")
        assert "query" in analyze.inputSchema.get("required", [])

    def test_all_handlers_registered(self):
        for tool in TOOLS:
            assert tool.name in TOOL_HANDLERS, f"No handler for {tool.name}"


class TestMCPServer:
    def test_create_server(self):
        server = create_server()
        assert server is not None

    def test_server_name(self):
        server = create_server()
        assert server.name == "prsm"


class TestToolHandlers:
    @pytest.mark.asyncio
    async def test_quote_handler(self):
        result = await handle_prsm_quote({
            "query": "EV trends",
            "shard_count": 3,
            "hardware_tier": "t2",
        })
        assert "Cost Estimate" in result
        assert "FTNS" in result
        assert "Total" in result

    @pytest.mark.asyncio
    async def test_quote_handler_defaults(self):
        result = await handle_prsm_quote({"query": "test"})
        assert "Cost Estimate" in result

    @pytest.mark.asyncio
    async def test_benchmark_handler(self):
        result = await handle_prsm_hardware_benchmark({})
        assert "Benchmark" in result
        assert "CPU" in result
        assert "TFLOPS" in result
        assert "Tier" in result

    @pytest.mark.asyncio
    async def test_analyze_handler_no_node(self):
        """Analyze should gracefully handle no running node."""
        result = await handle_prsm_analyze({"query": "test", "budget_ftns": 1.0})
        # Should either return a result or a helpful error
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_analyze_rejects_zero_budget(self):
        """Analyze must reject zero budget with helpful message."""
        result = await handle_prsm_analyze({"query": "test", "budget_ftns": 0})
        assert "FTNS" in result
        assert "budget" in result.lower() or "prsm_quote" in result

    @pytest.mark.asyncio
    async def test_analyze_rejects_negative_budget(self):
        """Analyze must reject negative budget."""
        result = await handle_prsm_analyze({"query": "test", "budget_ftns": -5.0})
        assert "FTNS" in result

    @pytest.mark.asyncio
    async def test_node_status_no_node(self):
        """Status should gracefully handle no running node."""
        result = await handle_prsm_node_status({})
        assert len(result) > 0


class TestCLICommand:
    def test_mcp_server_command_exists(self):
        from click.testing import CliRunner
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["mcp-server", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output or "mcp" in result.output

    def test_compute_run_rejects_zero_budget_query(self):
        """CLI should reject --query with --budget 0."""
        from click.testing import CliRunner
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["compute", "run", "--query", "test", "--budget", "0"])
        assert result.exit_code != 0
        assert "FTNS" in result.output or "budget" in result.output.lower()


class TestAgentCreationTools:
    @pytest.mark.asyncio
    async def test_create_agent_with_instructions(self):
        result = await handle_prsm_create_agent({
            "query": "Count EV registrations in NC",
            "instructions": [
                {"op": "filter", "field": "state", "value": "NC"},
                {"op": "filter", "field": "vehicle_type", "value": "EV"},
                {"op": "count"},
            ],
            "budget_ftns": 5.0,
        })
        assert "Agent Manifest Created" in result
        assert "3" in result  # 3 operations
        assert "filter" in result
        assert "count" in result
        assert "Manifest JSON" in result

    @pytest.mark.asyncio
    async def test_create_agent_rejects_unknown_op(self):
        result = await handle_prsm_create_agent({
            "query": "test",
            "instructions": [{"op": "explode_database"}],
            "budget_ftns": 1.0,
        })
        assert "Unknown operation" in result

    @pytest.mark.asyncio
    async def test_create_agent_rejects_zero_budget(self):
        result = await handle_prsm_create_agent({
            "query": "test",
            "instructions": [{"op": "count"}],
            "budget_ftns": 0,
        })
        assert "FTNS" in result

    @pytest.mark.asyncio
    async def test_create_agent_requires_instructions(self):
        result = await handle_prsm_create_agent({
            "query": "test",
            "instructions": [],
            "budget_ftns": 1.0,
        })
        assert "At least one instruction" in result

    @pytest.mark.asyncio
    async def test_dispatch_agent_rejects_empty_manifest(self):
        result = await handle_prsm_dispatch_agent({
            "instructions_json": "",
            "budget_ftns": 1.0,
        })
        assert "Missing" in result or "prsm_create_agent" in result

    @pytest.mark.asyncio
    async def test_dispatch_agent_validates_manifest(self):
        from prsm.compute.agents.instruction_set import InstructionManifest, AgentInstruction, AgentOp

        manifest = InstructionManifest(
            query="Count records",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        result = await handle_prsm_dispatch_agent({
            "instructions_json": manifest.to_json(),
            "budget_ftns": 1.0,
        })
        # Will fail to connect to node but should validate the manifest
        assert "1 operations" in result or "dispatch failed" in result
