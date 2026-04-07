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
)


class TestMCPToolDefinitions:
    def test_five_tools_defined(self):
        assert len(TOOLS) == 5

    def test_tool_names(self):
        names = [t.name for t in TOOLS]
        assert "prsm_analyze" in names
        assert "prsm_quote" in names
        assert "prsm_list_datasets" in names
        assert "prsm_node_status" in names
        assert "prsm_hardware_benchmark" in names

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
        result = await handle_prsm_analyze({"query": "test"})
        # Should either return a result or a helpful error
        assert len(result) > 0

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
