"""
Tests for Agent Forge data models (Ring 5 — Task 1).
"""

import time

import pytest

from prsm.compute.nwtn.agent_forge.models import (
    AgentTrace,
    ExecutionRoute,
    TaskDecomposition,
    TaskPlan,
    ThermalRequirement,
)


# ---------------------------------------------------------------------------
# ExecutionRoute
# ---------------------------------------------------------------------------


class TestExecutionRoute:
    def test_values(self):
        assert ExecutionRoute.DIRECT_LLM.value == "direct_llm"
        assert ExecutionRoute.LOCAL.value == "local"
        assert ExecutionRoute.SINGLE_AGENT.value == "single_agent"
        assert ExecutionRoute.SWARM.value == "swarm"


# ---------------------------------------------------------------------------
# TaskDecomposition
# ---------------------------------------------------------------------------


class TestTaskDecomposition:
    def test_creation(self):
        td = TaskDecomposition(
            query="Summarise this paper",
            required_datasets=["ds1"],
            operations=["summarise"],
            parallelizable=False,
            min_hardware_tier="t2",
            estimated_complexity=0.7,
        )
        assert td.query == "Summarise this paper"
        assert td.required_datasets == ["ds1"]
        assert td.estimated_complexity == 0.7

    def test_roundtrip(self):
        td = TaskDecomposition(
            query="compare datasets",
            required_datasets=["ds1", "ds2"],
            operations=["compare"],
            parallelizable=True,
            min_hardware_tier="t3",
            estimated_complexity=0.9,
        )
        d = td.to_dict()
        restored = TaskDecomposition.from_dict(d)
        assert restored.query == td.query
        assert restored.required_datasets == td.required_datasets
        assert restored.parallelizable is True
        assert restored.min_hardware_tier == "t3"
        assert restored.estimated_complexity == 0.9

    def test_route_direct_llm(self):
        td = TaskDecomposition(query="What is 2+2?")
        assert td.recommended_route == ExecutionRoute.DIRECT_LLM

    def test_route_single_agent(self):
        td = TaskDecomposition(
            query="search dataset",
            required_datasets=["ds1"],
            parallelizable=False,
        )
        assert td.recommended_route == ExecutionRoute.SINGLE_AGENT

    def test_route_swarm(self):
        td = TaskDecomposition(
            query="aggregate",
            required_datasets=["ds1", "ds2"],
            parallelizable=True,
        )
        assert td.recommended_route == ExecutionRoute.SWARM


# ---------------------------------------------------------------------------
# TaskPlan
# ---------------------------------------------------------------------------


class TestTaskPlan:
    def test_creation(self):
        td = TaskDecomposition(query="hello")
        plan = TaskPlan(
            decomposition=td,
            route=ExecutionRoute.DIRECT_LLM,
            target_shard_cids=["cid1"],
            estimated_pcu=1.5,
            thermal_requirement=ThermalRequirement.BURST,
        )
        assert plan.route == ExecutionRoute.DIRECT_LLM
        assert plan.estimated_pcu == 1.5
        assert plan.thermal_requirement == ThermalRequirement.BURST

    def test_to_dict(self):
        td = TaskDecomposition(query="hello")
        plan = TaskPlan(
            decomposition=td,
            route=ExecutionRoute.SWARM,
            target_shard_cids=["cid1", "cid2"],
            estimated_pcu=3.0,
        )
        d = plan.to_dict()
        assert d["route"] == "swarm"
        assert d["target_shard_cids"] == ["cid1", "cid2"]
        assert d["decomposition"]["query"] == "hello"


# ---------------------------------------------------------------------------
# AgentTrace
# ---------------------------------------------------------------------------


class TestAgentTrace:
    def test_creation(self):
        before = time.time()
        trace = AgentTrace(
            query="test query",
            decomposition={"key": "val"},
            plan={"route": "direct_llm"},
            execution_result={"answer": "42"},
            execution_metrics={"latency": 0.5},
            hardware_tier="t2",
        )
        assert trace.query == "test query"
        assert trace.hardware_tier == "t2"
        assert trace.user_satisfaction is None
        assert trace.timestamp >= before

    def test_to_dict(self):
        trace = AgentTrace(
            query="q",
            decomposition={"a": 1},
            plan={"b": 2},
            execution_result={"c": 3},
            execution_metrics={"d": 4},
            hardware_tier="t1",
            user_satisfaction=0.9,
        )
        d = trace.to_dict()
        assert d["query"] == "q"
        assert d["user_satisfaction"] == 0.9
        assert "timestamp" in d
