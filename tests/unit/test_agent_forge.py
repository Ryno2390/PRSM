"""
Tests for Agent Forge pipeline (Ring 5 — Task 2).
"""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.compute.nwtn.agent_forge.forge import AgentForge
from prsm.compute.nwtn.agent_forge.models import (
    ExecutionRoute,
    TaskDecomposition,
    TaskPlan,
    ThermalRequirement,
)


# ---------------------------------------------------------------------------
# Fixtures / mocks
# ---------------------------------------------------------------------------

LLM_JSON = (
    '{"required_datasets": ["ds1"], "operations": ["search"], '
    '"parallelizable": false, "min_hardware_tier": "t2", "estimated_complexity": 0.6}'
)

SIMPLE_LLM_JSON = (
    '{"required_datasets": [], "operations": ["generate"], '
    '"parallelizable": false, "min_hardware_tier": "t1", "estimated_complexity": 0.2}'
)


def _mock_backend(content: str = LLM_JSON):
    """Return a mock registry whose execute_with_fallback returns *content*."""
    backend = MagicMock()
    result = SimpleNamespace(content=content)
    backend.execute_with_fallback = AsyncMock(return_value=result)
    return backend


def _mock_pricing():
    """Return a mock pricing engine."""
    quote = SimpleNamespace(
        compute_cost=Decimal("0.5"),
        data_cost=Decimal("0"),
        network_fee=Decimal("0.025"),
        shard_breakdown=[],
        confidence=0.95,
        total=Decimal("0.525"),
        to_dict=lambda: {"compute_cost": "0.5", "total": "0.525"},
    )
    engine = MagicMock()
    engine.quote_swarm_job.return_value = quote
    return engine, quote


def _mock_swarm():
    """Return a mock swarm coordinator."""
    swarm_result = SimpleNamespace(
        job_id="swarm-abc123",
        shards_completed=2,
        total_pcu=3.0,
        aggregated_output={"summary": "done"},
    )
    job = SimpleNamespace(job_id="swarm-abc123")
    coord = MagicMock()
    coord.create_swarm_job.return_value = job
    coord.execute = AsyncMock(return_value=swarm_result)
    return coord, swarm_result


def _mock_dispatcher():
    """Return a mock agent dispatcher."""
    agent = SimpleNamespace(agent_id="agent-001")
    disp = MagicMock()
    disp.create_agent.return_value = agent
    disp.dispatch = AsyncMock(return_value={"answer": "42"})
    return disp, agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecompose:
    @pytest.mark.asyncio
    async def test_decompose_returns_task_decomposition(self):
        forge = AgentForge(backend_registry=_mock_backend())
        td = await forge.decompose("search ds1 for X")
        assert isinstance(td, TaskDecomposition)
        assert td.required_datasets == ["ds1"]
        assert td.min_hardware_tier == "t2"

    @pytest.mark.asyncio
    async def test_decompose_simple_routes_direct_llm(self):
        forge = AgentForge(backend_registry=_mock_backend(SIMPLE_LLM_JSON))
        td = await forge.decompose("what is 2+2?")
        assert td.recommended_route == ExecutionRoute.DIRECT_LLM

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_error(self):
        backend = MagicMock()
        backend.execute_with_fallback = AsyncMock(side_effect=RuntimeError("boom"))
        forge = AgentForge(backend_registry=backend)
        td = await forge.decompose("broken query")
        # Should fall back to defaults (no datasets -> DIRECT_LLM)
        assert isinstance(td, TaskDecomposition)
        assert td.recommended_route == ExecutionRoute.DIRECT_LLM


class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_creates_cost_quote(self):
        engine, quote = _mock_pricing()
        forge = AgentForge(pricing_engine=engine)
        td = TaskDecomposition(
            query="q",
            required_datasets=["ds1"],
            operations=["search"],
            estimated_complexity=0.6,
        )
        plan, cq = await forge.plan(td, shard_cids=["cid1", "cid2"])
        assert isinstance(plan, TaskPlan)
        assert cq is quote
        engine.quote_swarm_job.assert_called_once()


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_swarm_route(self):
        coord, swarm_result = _mock_swarm()
        forge = AgentForge(swarm_coordinator=coord)
        td = TaskDecomposition(
            query="agg",
            required_datasets=["d1", "d2"],
            parallelizable=True,
        )
        plan = TaskPlan(
            decomposition=td,
            route=ExecutionRoute.SWARM,
            target_shard_cids=["cid1", "cid2"],
        )
        result = await forge.execute(plan)
        assert result["route"] == "swarm"
        assert result["job_id"] == "swarm-abc123"
        coord.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_single_agent_route(self):
        disp, agent = _mock_dispatcher()
        forge = AgentForge(agent_dispatcher=disp)
        td = TaskDecomposition(query="q", required_datasets=["ds1"])
        plan = TaskPlan(decomposition=td, route=ExecutionRoute.SINGLE_AGENT)
        result = await forge.execute(plan)
        assert result["route"] == "single_agent"
        assert result["agent_id"] == "agent-001"
        disp.dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_direct_llm_route(self):
        backend = _mock_backend()
        # Override content for direct call
        backend.execute_with_fallback = AsyncMock(
            return_value=SimpleNamespace(content="The answer is 4")
        )
        forge = AgentForge(backend_registry=backend)
        td = TaskDecomposition(query="what is 2+2?")
        plan = TaskPlan(decomposition=td, route=ExecutionRoute.DIRECT_LLM)
        result = await forge.execute(plan)
        assert result["route"] == "direct_llm"
        assert result["answer"] == "The answer is 4"


class TestTraces:
    def test_traces_start_empty(self):
        forge = AgentForge()
        assert forge.traces == []


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_run_collects_trace(self):
        backend = _mock_backend(SIMPLE_LLM_JSON)
        # The direct_llm execute call will also use the backend
        forge = AgentForge(backend_registry=backend)
        output = await forge.run("hello world")
        assert output["query"] == "hello world"
        assert "decomposition" in output
        assert "plan" in output
        assert "result" in output
        assert len(forge.traces) == 1
        assert forge.traces[0].query == "hello world"
