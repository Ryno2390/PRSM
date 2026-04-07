"""
Agent Forge Pipeline
====================

Full decompose -> plan -> quote -> dispatch pipeline that turns a natural-
language query into routed execution across the PRSM compute fabric.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from prsm.compute.nwtn.agent_forge.models import (
    AgentTrace,
    ExecutionRoute,
    TaskDecomposition,
    TaskPlan,
    ThermalRequirement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt used for LLM-based decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a task decomposition engine for the PRSM distributed compute network.\n"
    "Given a user query, analyze what data and operations are needed.\n"
    "Respond with ONLY a JSON object:\n"
    '{"required_datasets": [], "operations": [], "parallelizable": false, '
    '"min_hardware_tier": "t1", "estimated_complexity": 0.5}'
)

# ---------------------------------------------------------------------------
# Default decomposition (used when the LLM call fails)
# ---------------------------------------------------------------------------

_DEFAULT_DECOMPOSITION_FIELDS: Dict[str, Any] = {
    "required_datasets": [],
    "operations": ["generate"],
    "parallelizable": False,
    "min_hardware_tier": "t1",
    "estimated_complexity": 0.3,
}


class AgentForge:
    """Orchestrates the full Agent Forge pipeline.

    Parameters
    ----------
    backend_registry : optional
        A ``BackendRegistry`` (or compatible) used for LLM calls.
    pricing_engine : optional
        A ``PricingEngine`` used for cost quoting.
    swarm_coordinator : optional
        A ``SwarmCoordinator`` used for swarm-route execution.
    agent_dispatcher : optional
        An agent dispatcher used for single-agent-route execution.
    template_wasm : optional
        Default WASM binary used when spawning agents.
    """

    def __init__(
        self,
        backend_registry=None,
        pricing_engine=None,
        swarm_coordinator=None,
        agent_dispatcher=None,
        template_wasm=None,
    ):
        self.backend_registry = backend_registry
        self.pricing_engine = pricing_engine
        self.swarm_coordinator = swarm_coordinator
        self.agent_dispatcher = agent_dispatcher
        self.template_wasm = template_wasm or b""
        self.traces: List[AgentTrace] = []

    # ------------------------------------------------------------------
    # LLM call helper (supports both BackendRegistry and individual backends)
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str, system_prompt: str = None, **kwargs):
        """Call the LLM backend, handling both registry and single backend."""
        backend = self.backend_registry
        if backend is None:
            raise RuntimeError("No backend available")

        # BackendRegistry has execute_with_fallback
        if hasattr(backend, "execute_with_fallback"):
            return await backend.execute_with_fallback(
                prompt=prompt, system_prompt=system_prompt, **kwargs,
            )
        # Individual backend (e.g., OpenRouterBackend) has generate
        elif hasattr(backend, "generate"):
            return await backend.generate(
                prompt=prompt, system_prompt=system_prompt, **kwargs,
            )
        else:
            raise RuntimeError(f"Backend {type(backend).__name__} has no generate method")

    # ------------------------------------------------------------------
    # decompose
    # ------------------------------------------------------------------

    async def decompose(self, query: str) -> TaskDecomposition:
        """Use the LLM backend to decompose *query* into structured fields."""
        if self.backend_registry is None:
            logger.warning("No backend_registry — returning default decomposition")
            return TaskDecomposition(query=query, **_DEFAULT_DECOMPOSITION_FIELDS)

        try:
            result = await self._call_llm(
                prompt=query,
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            )
            raw = result.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[: -3]
                raw = raw.strip()
            parsed = json.loads(raw)
            return TaskDecomposition(
                query=query,
                required_datasets=parsed.get("required_datasets", []),
                operations=parsed.get("operations", []),
                parallelizable=parsed.get("parallelizable", False),
                min_hardware_tier=parsed.get("min_hardware_tier", "t1"),
                estimated_complexity=parsed.get("estimated_complexity", 0.5),
            )
        except Exception as exc:
            logger.warning("Decompose LLM call failed (%s) — using defaults", exc)
            return TaskDecomposition(query=query, **_DEFAULT_DECOMPOSITION_FIELDS)

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    async def plan(
        self,
        decomposition: TaskDecomposition,
        shard_cids: Optional[List[str]] = None,
    ) -> Tuple[TaskPlan, Optional[Any]]:
        """Build a TaskPlan and optionally a CostQuote."""
        route = decomposition.recommended_route
        cids = shard_cids or []

        thermal = (
            ThermalRequirement.BURST
            if decomposition.estimated_complexity < 0.5
            else ThermalRequirement.SUSTAINED
        )

        task_plan = TaskPlan(
            decomposition=decomposition,
            route=route,
            target_shard_cids=cids,
            estimated_pcu=decomposition.estimated_complexity * 10.0,
            thermal_requirement=thermal,
        )

        cost_quote = None
        if cids and self.pricing_engine is not None:
            cost_quote = self.pricing_engine.quote_swarm_job(
                shard_count=len(cids),
                hardware_tier=decomposition.min_hardware_tier,
                estimated_pcu_per_shard=task_plan.estimated_pcu / max(len(cids), 1),
            )

        return task_plan, cost_quote

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        plan: TaskPlan,
        budget_ftns: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Execute a TaskPlan using the appropriate route."""
        route = plan.route

        # -- DIRECT_LLM -----------------------------------------------------
        if route == ExecutionRoute.DIRECT_LLM:
            if self.backend_registry is None:
                return {"response": "(no backend available)", "route": "direct_llm", "status": "error"}
            result = await self._call_llm(prompt=plan.decomposition.query)
            return {"response": result.content, "route": "direct_llm", "status": "success"}

        # -- SINGLE_AGENT ---------------------------------------------------
        if route == ExecutionRoute.SINGLE_AGENT:
            if self.agent_dispatcher is None:
                return {"error": "no dispatcher available", "route": "single_agent"}
            agent = self.agent_dispatcher.create_agent(
                wasm_binary=self.template_wasm,
                manifest=None,
                ftns_budget=budget_ftns,
            )
            result = await self.agent_dispatcher.dispatch(agent)
            return {
                "agent_id": getattr(agent, "agent_id", str(agent)),
                "result": result,
                "route": "single_agent",
            }

        # -- SWARM -----------------------------------------------------------
        if route == ExecutionRoute.SWARM:
            if self.swarm_coordinator is None:
                return {"error": "no swarm coordinator available", "route": "swarm"}
            job = self.swarm_coordinator.create_swarm_job(
                query=plan.decomposition.query,
                shard_cids=plan.target_shard_cids,
                wasm_binary=self.template_wasm,
                manifest=None,
                budget_ftns=budget_ftns,
            )
            swarm_result = await self.swarm_coordinator.execute(job)
            return {
                "job_id": swarm_result.job_id,
                "shards_completed": swarm_result.shards_completed,
                "total_pcu": swarm_result.total_pcu,
                "aggregated_output": swarm_result.aggregated_output,
                "route": "swarm",
            }

        return None

    # ------------------------------------------------------------------
    # run  (full pipeline)
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        budget_ftns: float = 10.0,
        shard_cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """End-to-end: decompose -> plan -> execute -> trace."""
        start = time.time()

        decomposition = await self.decompose(query)
        task_plan, cost_quote = await self.plan(decomposition, shard_cids)
        try:
            result = await self.execute(task_plan, budget_ftns) or {}
        except Exception as exc:
            logger.warning("Forge execute failed (%s) — recording partial trace", exc)
            result = {"status": "error", "error": str(exc), "route": task_plan.route.value}

        # Ensure route is always present in result
        if "route" not in result:
            result["route"] = task_plan.route.value
        if "status" not in result:
            result["status"] = "unknown"

        elapsed = time.time() - start

        trace = AgentTrace(
            query=query,
            decomposition=decomposition.to_dict(),
            plan=task_plan.to_dict(),
            execution_result=result,
            execution_metrics={
                "elapsed_seconds": elapsed,
                "route": task_plan.route.value,
            },
            hardware_tier=decomposition.min_hardware_tier,
        )
        self.traces.append(trace)

        return {
            "query": query,
            "decomposition": decomposition.to_dict(),
            "plan": task_plan.to_dict(),
            "cost_quote": cost_quote.to_dict() if cost_quote and hasattr(cost_quote, "to_dict") else None,
            "result": result,
            "trace_id": len(self.traces) - 1,
        }
