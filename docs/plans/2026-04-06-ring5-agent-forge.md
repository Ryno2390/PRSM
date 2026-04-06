# Ring 5 — "The Brain" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NWTN evolves into an agent architect that turns natural language queries into execution plans, selects target shards, quotes costs, and orchestrates the full Rings 2-4 pipeline. LLM backends generate task decompositions; the forge routes to the appropriate execution path (local, single-agent, or swarm).

**Architecture:** An `AgentForge` sits between the NWTN orchestrator and the Ring 2-4 infrastructure. It decomposes queries via the existing LLM backend, produces `TaskPlan` objects, quotes costs via the Ring 4 pricing engine, and dispatches via Ring 2 (single) or Ring 3 (swarm). An `AgentTrace` captures every execution for future fine-tuning. MCP tools expose the pipeline to external LLMs.

**Tech Stack:** Existing PRSM infrastructure (NWTN backends, agent dispatcher, swarm coordinator, pricing engine). No new external dependencies.

**Scope note:** This ring uses frontier models (via OpenRouter/Anthropic/local) for task decomposition. The specialized NWTN fine-tune is deferred per the spec — we collect training data here via AgentTrace.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/compute/nwtn/agent_forge/__init__.py` | Package exports |
| Create | `prsm/compute/nwtn/agent_forge/models.py` | `TaskDecomposition`, `TaskPlan`, `AgentTrace`, `ExecutionRoute` |
| Create | `prsm/compute/nwtn/agent_forge/forge.py` | `AgentForge` — decompose, plan, quote, dispatch |
| Create | `prsm/compute/nwtn/agent_forge/mcp_tools.py` | MCP tool definitions for external LLM access |
| Modify | `prsm/node/node.py` | Wire AgentForge into PRSMNode |
| Create | `tests/unit/test_agent_forge_models.py` | Forge data model tests |
| Create | `tests/unit/test_agent_forge.py` | Forge pipeline tests |
| Create | `tests/integration/test_ring5_forge.py` | End-to-end forge smoke test |

---

### Task 1: Agent Forge Data Models

**Files:**
- Create: `prsm/compute/nwtn/agent_forge/__init__.py`
- Create: `prsm/compute/nwtn/agent_forge/models.py`
- Test: `tests/unit/test_agent_forge_models.py`

- [ ] **Step 1: Create directory**

```bash
mkdir -p prsm/compute/nwtn/agent_forge
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_agent_forge_models.py`:

```python
"""Tests for Agent Forge data models."""

import pytest
import time
from prsm.compute.nwtn.agent_forge.models import (
    TaskDecomposition,
    TaskPlan,
    ExecutionRoute,
    AgentTrace,
    ThermalRequirement,
)


class TestExecutionRoute:
    def test_all_routes_exist(self):
        assert ExecutionRoute.LOCAL == "local"
        assert ExecutionRoute.SINGLE_AGENT == "single_agent"
        assert ExecutionRoute.SWARM == "swarm"
        assert ExecutionRoute.DIRECT_LLM == "direct_llm"


class TestTaskDecomposition:
    def test_decomposition_creation(self):
        decomp = TaskDecomposition(
            query="EV adoption trends in NC",
            required_datasets=["nada-nc-2025"],
            operations=["filter", "aggregate", "time_series"],
            parallelizable=True,
            min_hardware_tier="t2",
            estimated_complexity=0.7,
        )
        assert decomp.parallelizable is True
        assert decomp.estimated_complexity == 0.7
        assert len(decomp.operations) == 3

    def test_decomposition_to_dict_roundtrip(self):
        decomp = TaskDecomposition(
            query="test query",
            required_datasets=["ds-1"],
            operations=["filter"],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        d = decomp.to_dict()
        restored = TaskDecomposition.from_dict(d)
        assert restored.query == "test query"
        assert restored.required_datasets == ["ds-1"]
        assert restored.parallelizable is False

    def test_route_selection_direct_llm(self):
        """Simple query with no data → DIRECT_LLM."""
        decomp = TaskDecomposition(
            query="What is the capital of France?",
            required_datasets=[],
            operations=[],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        assert decomp.recommended_route == ExecutionRoute.DIRECT_LLM

    def test_route_selection_single_agent(self):
        """One dataset, not parallelizable → SINGLE_AGENT."""
        decomp = TaskDecomposition(
            query="Filter NC data",
            required_datasets=["nada-nc"],
            operations=["filter"],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        assert decomp.recommended_route == ExecutionRoute.SINGLE_AGENT

    def test_route_selection_swarm(self):
        """Multiple datasets or parallelizable → SWARM."""
        decomp = TaskDecomposition(
            query="National EV trends",
            required_datasets=["nada-nc", "nada-ca", "nada-tx"],
            operations=["filter", "aggregate"],
            parallelizable=True,
            min_hardware_tier="t2",
        )
        assert decomp.recommended_route == ExecutionRoute.SWARM


class TestTaskPlan:
    def test_plan_creation(self):
        decomp = TaskDecomposition(
            query="test",
            required_datasets=["ds-1"],
            operations=["filter"],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        plan = TaskPlan(
            decomposition=decomp,
            route=ExecutionRoute.SINGLE_AGENT,
            target_shard_cids=["QmShard001"],
            estimated_pcu=50.0,
            thermal_requirement=ThermalRequirement.SUSTAINED,
        )
        assert plan.route == ExecutionRoute.SINGLE_AGENT
        assert len(plan.target_shard_cids) == 1

    def test_plan_to_dict(self):
        decomp = TaskDecomposition(
            query="q",
            required_datasets=[],
            operations=[],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        plan = TaskPlan(
            decomposition=decomp,
            route=ExecutionRoute.DIRECT_LLM,
            target_shard_cids=[],
            estimated_pcu=0.0,
        )
        d = plan.to_dict()
        assert d["route"] == "direct_llm"
        assert "decomposition" in d


class TestAgentTrace:
    def test_trace_creation(self):
        trace = AgentTrace(
            query="EV trends",
            decomposition={"operations": ["filter"]},
            plan={"route": "swarm", "shards": 5},
            execution_result={"status": "success"},
            execution_metrics={"pcu": 2.5, "time_seconds": 10.0},
            hardware_tier="t2",
        )
        assert trace.query == "EV trends"
        assert trace.hardware_tier == "t2"

    def test_trace_to_dict(self):
        trace = AgentTrace(
            query="test",
            decomposition={},
            plan={},
            execution_result={"status": "success"},
            execution_metrics={"pcu": 1.0},
            hardware_tier="t1",
        )
        d = trace.to_dict()
        assert "query" in d
        assert "timestamp" in d
        assert "execution_metrics" in d
```

- [ ] **Step 3: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_agent_forge_models.py::TestExecutionRoute -v`
Expected: FAIL

- [ ] **Step 4: Implement models**

Create `prsm/compute/nwtn/agent_forge/__init__.py`:

```python
"""
Agent Forge
===========

LLM-powered task decomposition and execution routing.
Ring 5 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.nwtn.agent_forge.models import (
    TaskDecomposition,
    TaskPlan,
    ExecutionRoute,
    AgentTrace,
    ThermalRequirement,
)

__all__ = [
    "TaskDecomposition",
    "TaskPlan",
    "ExecutionRoute",
    "AgentTrace",
    "ThermalRequirement",
]
```

Create `prsm/compute/nwtn/agent_forge/models.py`:

```python
"""
Agent Forge Data Models
=======================

Task decomposition, execution planning, and training data collection.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionRoute(str, Enum):
    """How a query should be executed."""
    DIRECT_LLM = "direct_llm"       # Simple query → existing NWTN pipeline
    LOCAL = "local"                   # Local data → Ring 1 sandbox
    SINGLE_AGENT = "single_agent"    # Remote data, one shard → Ring 2
    SWARM = "swarm"                  # Multi-shard → Ring 3


class ThermalRequirement(str, Enum):
    """Thermal class needed for the job."""
    BURST = "burst"        # Short job, any device
    SUSTAINED = "sustained"  # Long job, needs desktop/server
    ANY = "any"


@dataclass
class TaskDecomposition:
    """LLM-produced analysis of what a query needs."""
    query: str
    required_datasets: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    parallelizable: bool = False
    min_hardware_tier: str = "t1"
    estimated_complexity: float = 0.5  # 0.0 (trivial) to 1.0 (extreme)

    @property
    def recommended_route(self) -> ExecutionRoute:
        if not self.required_datasets:
            return ExecutionRoute.DIRECT_LLM
        if self.parallelizable or len(self.required_datasets) > 1:
            return ExecutionRoute.SWARM
        return ExecutionRoute.SINGLE_AGENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "required_datasets": self.required_datasets,
            "operations": self.operations,
            "parallelizable": self.parallelizable,
            "min_hardware_tier": self.min_hardware_tier,
            "estimated_complexity": self.estimated_complexity,
            "recommended_route": self.recommended_route.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskDecomposition":
        return cls(
            query=d.get("query", ""),
            required_datasets=d.get("required_datasets", []),
            operations=d.get("operations", []),
            parallelizable=d.get("parallelizable", False),
            min_hardware_tier=d.get("min_hardware_tier", "t1"),
            estimated_complexity=d.get("estimated_complexity", 0.5),
        )


@dataclass
class TaskPlan:
    """Execution plan derived from a decomposition + network state."""
    decomposition: TaskDecomposition
    route: ExecutionRoute
    target_shard_cids: List[str] = field(default_factory=list)
    estimated_pcu: float = 0.0
    thermal_requirement: ThermalRequirement = ThermalRequirement.ANY
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decomposition": self.decomposition.to_dict(),
            "route": self.route.value,
            "target_shard_cids": self.target_shard_cids,
            "estimated_pcu": self.estimated_pcu,
            "thermal_requirement": self.thermal_requirement.value,
        }


@dataclass
class AgentTrace:
    """Training data collected from each execution for future fine-tuning."""
    query: str
    decomposition: Dict[str, Any]
    plan: Dict[str, Any]
    execution_result: Dict[str, Any]
    execution_metrics: Dict[str, Any]
    hardware_tier: str = "t1"
    user_satisfaction: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "decomposition": self.decomposition,
            "plan": self.plan,
            "execution_result": self.execution_result,
            "execution_metrics": self.execution_metrics,
            "hardware_tier": self.hardware_tier,
            "user_satisfaction": self.user_satisfaction,
            "timestamp": self.timestamp,
        }
```

- [ ] **Step 5: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_agent_forge_models.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/compute/nwtn/agent_forge/__init__.py prsm/compute/nwtn/agent_forge/models.py tests/unit/test_agent_forge_models.py
git commit -m "feat(ring5): TaskDecomposition, TaskPlan, AgentTrace, ExecutionRoute models"
```

---

### Task 2: Agent Forge Pipeline

**Files:**
- Create: `prsm/compute/nwtn/agent_forge/forge.py`
- Modify: `prsm/compute/nwtn/agent_forge/__init__.py`
- Test: `tests/unit/test_agent_forge.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_agent_forge.py`:

```python
"""Tests for AgentForge — decompose, plan, quote, dispatch pipeline."""

import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from prsm.compute.nwtn.agent_forge.forge import AgentForge
from prsm.compute.nwtn.agent_forge.models import (
    TaskDecomposition,
    ExecutionRoute,
    AgentTrace,
)
from prsm.economy.pricing.models import CostQuote


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


@pytest.fixture
def mock_backend():
    backend = AsyncMock()
    backend.execute_with_fallback = AsyncMock(return_value=MagicMock(
        content='{"required_datasets": ["nada-nc"], "operations": ["filter", "aggregate"], "parallelizable": true, "min_hardware_tier": "t2", "estimated_complexity": 0.6}',
        model_id="openrouter/meta-llama/llama-3.1-8b",
    ))
    return backend


@pytest.fixture
def mock_pricing():
    pricing = MagicMock()
    pricing.estimate_pcu = MagicMock(return_value=50.0)
    pricing.quote_swarm_job = MagicMock(return_value=CostQuote(
        compute_cost=Decimal("2.50"),
        data_cost=Decimal("5.00"),
        network_fee=Decimal("0.375"),
    ))
    return pricing


@pytest.fixture
def mock_swarm():
    swarm = AsyncMock()
    swarm.create_swarm_job = MagicMock(return_value=MagicMock(job_id="swarm-test"))
    swarm.execute = AsyncMock(return_value=MagicMock(
        job_id="swarm-test",
        shards_completed=3,
        total_pcu=1.5,
        total_ftns_spent=5.0,
        aggregated_output={"shard_count": 3, "shard_outputs": [{"data": {"count": 42}}]},
    ))
    return swarm


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock()
    agent = MagicMock()
    agent.agent_id = "agent-test"
    dispatcher.create_agent = MagicMock(return_value=agent)
    dispatcher.dispatch = AsyncMock(return_value=MagicMock(status="completed"))
    dispatcher.wait_for_result = AsyncMock(return_value={
        "status": "success",
        "output_b64": base64.b64encode(b'{"result": 42}').decode(),
        "pcu": 0.5,
    })
    dispatcher.select_and_transfer = AsyncMock(return_value=True)
    return dispatcher


@pytest.fixture
def forge(mock_backend, mock_pricing, mock_swarm, mock_dispatcher):
    return AgentForge(
        backend_registry=mock_backend,
        pricing_engine=mock_pricing,
        swarm_coordinator=mock_swarm,
        agent_dispatcher=mock_dispatcher,
        template_wasm=MINIMAL_WASM,
    )


class TestAgentForge:
    @pytest.mark.asyncio
    async def test_decompose_query(self, forge):
        decomp = await forge.decompose("EV adoption trends in NC")
        assert isinstance(decomp, TaskDecomposition)
        assert decomp.query == "EV adoption trends in NC"
        assert len(decomp.required_datasets) > 0

    @pytest.mark.asyncio
    async def test_decompose_simple_query_routes_to_llm(self, forge, mock_backend):
        mock_backend.execute_with_fallback = AsyncMock(return_value=MagicMock(
            content='{"required_datasets": [], "operations": [], "parallelizable": false, "min_hardware_tier": "t1", "estimated_complexity": 0.1}',
        ))
        decomp = await forge.decompose("What is 2+2?")
        assert decomp.recommended_route == ExecutionRoute.DIRECT_LLM

    @pytest.mark.asyncio
    async def test_plan_creates_cost_quote(self, forge, mock_pricing):
        decomp = TaskDecomposition(
            query="test",
            required_datasets=["ds-1"],
            operations=["filter"],
            parallelizable=True,
            min_hardware_tier="t2",
        )
        plan, quote = await forge.plan(decomp, shard_cids=["QmA", "QmB", "QmC"])
        assert quote is not None
        assert quote.total > 0
        mock_pricing.quote_swarm_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_swarm_route(self, forge, mock_swarm):
        decomp = TaskDecomposition(
            query="parallel test",
            required_datasets=["ds-1"],
            operations=["aggregate"],
            parallelizable=True,
            min_hardware_tier="t2",
        )
        plan, _ = await forge.plan(decomp, shard_cids=["QmA", "QmB"])

        result = await forge.execute(plan, budget_ftns=10.0)
        assert result is not None
        mock_swarm.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_single_agent_route(self, forge, mock_dispatcher):
        decomp = TaskDecomposition(
            query="single shard test",
            required_datasets=["ds-1"],
            operations=["filter"],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        plan, _ = await forge.plan(decomp, shard_cids=["QmA"])

        result = await forge.execute(plan, budget_ftns=5.0)
        assert result is not None
        mock_dispatcher.create_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_direct_llm_route(self, forge, mock_backend):
        decomp = TaskDecomposition(
            query="What is the capital of France?",
            required_datasets=[],
            operations=[],
            parallelizable=False,
            min_hardware_tier="t1",
        )
        plan, _ = await forge.plan(decomp, shard_cids=[])

        mock_backend.execute_with_fallback = AsyncMock(return_value=MagicMock(
            content="Paris is the capital of France.",
        ))

        result = await forge.execute(plan, budget_ftns=0.1)
        assert result is not None

    def test_traces_collected(self, forge):
        assert isinstance(forge.traces, list)
        assert len(forge.traces) == 0

    @pytest.mark.asyncio
    async def test_full_pipeline_collects_trace(self, forge):
        result = await forge.run("EV trends in NC", budget_ftns=10.0)
        assert len(forge.traces) == 1
        trace = forge.traces[0]
        assert trace.query == "EV trends in NC"
        assert "execution_result" in trace.to_dict()
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_agent_forge.py::TestAgentForge::test_decompose_query -v`
Expected: FAIL

- [ ] **Step 3: Implement the forge**

Create `prsm/compute/nwtn/agent_forge/forge.py`:

```python
"""
Agent Forge
===========

LLM-powered task decomposition and execution routing.
Connects NWTN orchestrator to Rings 2-4 infrastructure.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from prsm.compute.nwtn.agent_forge.models import (
    TaskDecomposition,
    TaskPlan,
    ExecutionRoute,
    AgentTrace,
    ThermalRequirement,
)

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM_PROMPT = """You are a task decomposition engine for the PRSM distributed compute network.
Given a user query, analyze what data and operations are needed.

Respond with ONLY a JSON object (no markdown, no explanation):
{
    "required_datasets": ["dataset-name"],
    "operations": ["filter", "aggregate", "time_series", "embedding_similarity"],
    "parallelizable": true,
    "min_hardware_tier": "t1",
    "estimated_complexity": 0.5
}

Rules:
- required_datasets: empty list if the query can be answered from LLM knowledge alone
- operations: what data operations are needed (filter, aggregate, join, stats, time_series, embedding_similarity)
- parallelizable: true if the work can be split across multiple data shards
- min_hardware_tier: "t1" (simple), "t2" (moderate), "t3" (GPU-intensive), "t4" (heavy training)
- estimated_complexity: 0.0 (trivial) to 1.0 (extreme)"""


class AgentForge:
    """Decomposes queries, plans execution, and dispatches agents."""

    def __init__(
        self,
        backend_registry=None,
        pricing_engine=None,
        swarm_coordinator=None,
        agent_dispatcher=None,
        template_wasm: Optional[bytes] = None,
    ):
        self._backend = backend_registry
        self._pricing = pricing_engine
        self._swarm = swarm_coordinator
        self._dispatcher = agent_dispatcher
        self._template_wasm = template_wasm or b"\x00asm\x01\x00\x00\x00"
        self.traces: List[AgentTrace] = []

    async def decompose(self, query: str) -> TaskDecomposition:
        """Use LLM to analyze what the query needs."""
        if not self._backend:
            return TaskDecomposition(query=query)

        try:
            result = await self._backend.execute_with_fallback(
                prompt=query,
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                max_tokens=500,
                temperature=0.1,
            )

            parsed = json.loads(result.content)
            return TaskDecomposition(
                query=query,
                required_datasets=parsed.get("required_datasets", []),
                operations=parsed.get("operations", []),
                parallelizable=parsed.get("parallelizable", False),
                min_hardware_tier=parsed.get("min_hardware_tier", "t1"),
                estimated_complexity=parsed.get("estimated_complexity", 0.5),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Decomposition failed, using defaults: {e}")
            return TaskDecomposition(query=query)

    async def plan(
        self,
        decomposition: TaskDecomposition,
        shard_cids: Optional[List[str]] = None,
    ) -> Tuple[TaskPlan, Optional[Any]]:
        """Create an execution plan and cost quote."""
        route = decomposition.recommended_route
        cids = shard_cids or []

        # Estimate PCU
        estimated_pcu = 0.0
        if self._pricing and cids:
            from prsm.compute.agents.models import AgentManifest
            manifest = AgentManifest(
                required_cids=cids[:1],
                min_hardware_tier=decomposition.min_hardware_tier,
            )
            estimated_pcu = self._pricing.estimate_pcu(manifest) * len(cids)

        # Thermal requirement based on complexity
        thermal = ThermalRequirement.ANY
        if decomposition.estimated_complexity > 0.7:
            thermal = ThermalRequirement.SUSTAINED
        elif decomposition.estimated_complexity > 0.3:
            thermal = ThermalRequirement.BURST

        plan = TaskPlan(
            decomposition=decomposition,
            route=route,
            target_shard_cids=cids,
            estimated_pcu=estimated_pcu,
            thermal_requirement=thermal,
        )

        # Get cost quote
        quote = None
        if self._pricing and cids:
            quote = self._pricing.quote_swarm_job(
                shard_count=len(cids),
                hardware_tier=decomposition.min_hardware_tier,
                estimated_pcu_per_shard=estimated_pcu / len(cids) if cids else 0,
            )

        return plan, quote

    async def execute(
        self,
        plan: TaskPlan,
        budget_ftns: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Execute a task plan via the appropriate route."""
        route = plan.route

        if route == ExecutionRoute.DIRECT_LLM:
            return await self._execute_direct_llm(plan)
        elif route == ExecutionRoute.SINGLE_AGENT:
            return await self._execute_single_agent(plan, budget_ftns)
        elif route == ExecutionRoute.SWARM:
            return await self._execute_swarm(plan, budget_ftns)
        elif route == ExecutionRoute.LOCAL:
            return await self._execute_local(plan)
        else:
            logger.error(f"Unknown execution route: {route}")
            return None

    async def _execute_direct_llm(self, plan: TaskPlan) -> Dict[str, Any]:
        """Route simple queries directly to LLM backend."""
        if not self._backend:
            return {"status": "error", "error": "No backend available"}

        result = await self._backend.execute_with_fallback(
            prompt=plan.decomposition.query,
            max_tokens=2000,
        )
        return {
            "status": "success",
            "route": "direct_llm",
            "response": result.content,
            "pcu": 0.0,
        }

    async def _execute_single_agent(
        self,
        plan: TaskPlan,
        budget_ftns: float,
    ) -> Dict[str, Any]:
        """Dispatch a single Ring 2 agent."""
        if not self._dispatcher:
            return {"status": "error", "error": "No dispatcher available"}

        from prsm.compute.agents.models import AgentManifest
        manifest = AgentManifest(
            required_cids=plan.target_shard_cids[:1],
            min_hardware_tier=plan.decomposition.min_hardware_tier,
        )

        agent = self._dispatcher.create_agent(
            wasm_binary=self._template_wasm,
            manifest=manifest,
            ftns_budget=budget_ftns,
        )
        await self._dispatcher.dispatch(agent)
        await self._dispatcher.select_and_transfer(agent.agent_id)
        result = await self._dispatcher.wait_for_result(agent.agent_id)

        return {
            "status": "success" if result else "failed",
            "route": "single_agent",
            "agent_id": agent.agent_id,
            "result": result,
            "pcu": result.get("pcu", 0) if result else 0,
        }

    async def _execute_swarm(
        self,
        plan: TaskPlan,
        budget_ftns: float,
    ) -> Dict[str, Any]:
        """Dispatch a Ring 3 swarm job."""
        if not self._swarm:
            return {"status": "error", "error": "No swarm coordinator available"}

        from prsm.compute.agents.models import AgentManifest
        manifest = AgentManifest(
            required_cids=[],
            min_hardware_tier=plan.decomposition.min_hardware_tier,
        )

        job = self._swarm.create_swarm_job(
            query=plan.decomposition.query,
            shard_cids=plan.target_shard_cids,
            wasm_binary=self._template_wasm,
            manifest=manifest,
            budget_ftns=budget_ftns,
        )
        result = await self._swarm.execute(job)

        return {
            "status": "success" if result.shards_completed > 0 else "failed",
            "route": "swarm",
            "job_id": result.job_id,
            "shards_completed": result.shards_completed,
            "total_pcu": result.total_pcu,
            "aggregated_output": result.aggregated_output,
        }

    async def _execute_local(self, plan: TaskPlan) -> Dict[str, Any]:
        """Execute locally in Ring 1 sandbox."""
        return {
            "status": "success",
            "route": "local",
            "response": "Local execution placeholder",
            "pcu": 0.0,
        }

    async def run(
        self,
        query: str,
        budget_ftns: float = 10.0,
        shard_cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: decompose → plan → quote → execute → trace."""
        started_at = time.time()

        # Decompose
        decomposition = await self.decompose(query)

        # Plan
        plan, quote = await self.plan(decomposition, shard_cids=shard_cids)

        # Execute
        result = await self.execute(plan, budget_ftns=budget_ftns)

        # Collect trace
        trace = AgentTrace(
            query=query,
            decomposition=decomposition.to_dict(),
            plan=plan.to_dict(),
            execution_result=result or {},
            execution_metrics={
                "total_time_seconds": time.time() - started_at,
                "route": plan.route.value,
                "pcu": result.get("pcu", 0) if result else 0,
                "budget_ftns": budget_ftns,
            },
            hardware_tier=decomposition.min_hardware_tier,
        )
        self.traces.append(trace)

        logger.info(
            f"Forge pipeline complete: {query[:40]}... → "
            f"{plan.route.value}, {time.time() - started_at:.2f}s"
        )

        return result
```

- [ ] **Step 4: Update `__init__.py`**

Add to `prsm/compute/nwtn/agent_forge/__init__.py`:

```python
from prsm.compute.nwtn.agent_forge.forge import AgentForge
```
Add `"AgentForge"` to `__all__`.

- [ ] **Step 5: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_agent_forge.py -v`
Expected: All 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/compute/nwtn/agent_forge/forge.py prsm/compute/nwtn/agent_forge/__init__.py tests/unit/test_agent_forge.py
git commit -m "feat(ring5): AgentForge — decompose, plan, quote, dispatch pipeline"
```

---

### Task 3: MCP Tool Definitions + Node Integration + Smoke Test

**Files:**
- Create: `prsm/compute/nwtn/agent_forge/mcp_tools.py`
- Modify: `prsm/node/node.py`
- Create: `tests/integration/test_ring5_forge.py`

- [ ] **Step 1: Create MCP tool definitions**

Create `prsm/compute/nwtn/agent_forge/mcp_tools.py`:

```python
"""
MCP Tool Definitions for Agent Forge
=====================================

Exposes the forge pipeline as MCP tools that any LLM can call.
"""

from typing import Any, Dict, List


# MCP tool definitions following the MCP protocol schema
FORGE_MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "prsm_analyze",
        "description": (
            "Submit a natural language query to the PRSM network for distributed analysis. "
            "Automatically decomposes the query, finds relevant data shards, dispatches "
            "agents to process them, and returns aggregated results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The analysis query in natural language",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "Maximum FTNS tokens to spend on this query",
                    "default": 10.0,
                },
                "dataset_id": {
                    "type": "string",
                    "description": "Optional: specific dataset to query against",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "prsm_quote",
        "description": (
            "Get a cost estimate for a query before committing. Returns compute cost, "
            "data access cost, network fee, and total in FTNS tokens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to estimate costs for",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "prsm_list_datasets",
        "description": "Browse available datasets on the PRSM network with pricing information.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by dataset category",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum base access fee in FTNS",
                },
            },
        },
    },
    {
        "name": "prsm_dispatch_agent",
        "description": (
            "Low-level: dispatch a pre-built WASM agent to a specific data shard. "
            "For advanced users who want direct control over agent execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wasm_url": {
                    "type": "string",
                    "description": "IPFS URL of the WASM agent binary",
                },
                "required_data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CIDs of data shards the agent needs",
                },
                "min_hardware_tier": {
                    "type": "string",
                    "enum": ["t1", "t2", "t3", "t4"],
                    "default": "t1",
                },
                "budget_ftns": {
                    "type": "number",
                    "default": 5.0,
                },
            },
            "required": ["wasm_url", "required_data"],
        },
    },
    {
        "name": "prsm_swarm_status",
        "description": "Check the status of a running swarm job.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The swarm job ID to check",
                },
            },
            "required": ["job_id"],
        },
    },
]


def get_forge_tools() -> List[Dict[str, Any]]:
    """Return MCP tool definitions for the agent forge."""
    return FORGE_MCP_TOOLS
```

- [ ] **Step 2: Wire AgentForge into node.py**

Find the Ring 4 initialization block and add after it:

```python
        # ── Agent Forge (Ring 5) ──────────────────────────────────────
        try:
            from prsm.compute.nwtn.agent_forge.forge import AgentForge

            self.agent_forge = AgentForge(
                backend_registry=getattr(self, '_backend_registry', None),
                pricing_engine=self.pricing_engine,
                swarm_coordinator=self.swarm_coordinator,
                agent_dispatcher=self.agent_dispatcher,
            )
            logger.info("Agent forge (Ring 5) initialized")
        except (ImportError, AttributeError):
            self.agent_forge = None
            logger.debug("Agent forge not available")
```

- [ ] **Step 3: Create integration smoke test**

Create `tests/integration/test_ring5_forge.py`:

```python
"""
Ring 5 Smoke Test
=================

End-to-end: decompose query → plan → quote → execute → collect trace.
"""

import pytest
import base64
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal

from prsm.compute.nwtn.agent_forge import AgentForge, ExecutionRoute
from prsm.economy.pricing.models import CostQuote
from prsm.compute.nwtn.agent_forge.mcp_tools import get_forge_tools


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing5Smoke:
    @pytest.mark.asyncio
    async def test_full_forge_pipeline(self):
        """Decompose → plan → quote → execute (swarm) → trace collected."""
        backend = AsyncMock()
        backend.execute_with_fallback = AsyncMock(return_value=MagicMock(
            content='{"required_datasets": ["nada-nc"], "operations": ["filter", "aggregate"], "parallelizable": true, "min_hardware_tier": "t2", "estimated_complexity": 0.6}',
        ))

        pricing = MagicMock()
        pricing.estimate_pcu = MagicMock(return_value=50.0)
        pricing.quote_swarm_job = MagicMock(return_value=CostQuote(
            compute_cost=Decimal("2.50"),
            data_cost=Decimal("5.00"),
            network_fee=Decimal("0.375"),
        ))

        swarm = AsyncMock()
        swarm.create_swarm_job = MagicMock(return_value=MagicMock(job_id="swarm-1"))
        swarm.execute = AsyncMock(return_value=MagicMock(
            job_id="swarm-1",
            shards_completed=3,
            total_pcu=1.5,
            aggregated_output={"shard_count": 3, "shard_outputs": [{"data": {"ev_count": 142}}]},
        ))

        forge = AgentForge(
            backend_registry=backend,
            pricing_engine=pricing,
            swarm_coordinator=swarm,
            template_wasm=MINIMAL_WASM,
        )

        result = await forge.run(
            "EV adoption trends in NC 2025",
            budget_ftns=10.0,
            shard_cids=["QmA", "QmB", "QmC"],
        )

        assert result is not None
        assert result["status"] == "success"
        assert result["route"] == "swarm"
        assert len(forge.traces) == 1
        assert forge.traces[0].query == "EV adoption trends in NC 2025"

    @pytest.mark.asyncio
    async def test_simple_query_routes_to_llm(self):
        """Simple query bypasses agents entirely."""
        backend = AsyncMock()
        backend.execute_with_fallback = AsyncMock(return_value=MagicMock(
            content='{"required_datasets": [], "operations": [], "parallelizable": false, "min_hardware_tier": "t1", "estimated_complexity": 0.1}',
        ))

        forge = AgentForge(backend_registry=backend, template_wasm=MINIMAL_WASM)

        # Override for the direct LLM execution
        async def direct_answer(*args, **kwargs):
            return MagicMock(content="Paris is the capital of France.")
        backend.execute_with_fallback = AsyncMock(side_effect=[
            MagicMock(content='{"required_datasets": [], "operations": [], "parallelizable": false, "min_hardware_tier": "t1", "estimated_complexity": 0.1}'),
            MagicMock(content="Paris is the capital of France."),
        ])

        result = await forge.run("What is the capital of France?")

        assert result is not None
        assert result["route"] == "direct_llm"
        assert "Paris" in result.get("response", "")

    def test_mcp_tools_defined(self):
        """Verify MCP tool definitions are complete."""
        tools = get_forge_tools()
        assert len(tools) == 5
        names = [t["name"] for t in tools]
        assert "prsm_analyze" in names
        assert "prsm_quote" in names
        assert "prsm_list_datasets" in names
        assert "prsm_dispatch_agent" in names
        assert "prsm_swarm_status" in names

    def test_mcp_tools_have_required_fields(self):
        """Each MCP tool has name, description, parameters."""
        for tool in get_forge_tools():
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert len(tool["description"]) > 20
```

- [ ] **Step 4: Run all Ring 5 tests**

Run: `python -m pytest tests/unit/test_agent_forge_models.py tests/unit/test_agent_forge.py tests/integration/test_ring5_forge.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/nwtn/agent_forge/mcp_tools.py prsm/node/node.py tests/integration/test_ring5_forge.py
git commit -m "feat(ring5): MCP tool definitions + PRSMNode wiring + integration smoke test"
```

---

### Task 4: Version Bump + Push + PyPI

- [ ] **Step 1:** Bump `__version__` in `prsm/__init__.py` to `"0.30.0"` and `version` in `pyproject.toml` to `"0.30.0"`

- [ ] **Step 2:** Final test run

- [ ] **Step 3:** Commit, push, build, publish

```bash
git add prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.30.0 for Ring 5 — The Brain"
git push origin main
rm -rf build/ dist/ prsm_network.egg-info/
python3 -m build
python3 -m twine upload dist/prsm_network-0.30.0*
```
