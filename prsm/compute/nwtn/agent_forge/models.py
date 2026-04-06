"""
Agent Forge Data Models
=======================

Task decomposition, execution routing, and trace structures for the
LLM-powered Agent Forge pipeline (Ring 5).
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExecutionRoute(str, Enum):
    """Where / how to execute a decomposed task."""

    DIRECT_LLM = "direct_llm"
    LOCAL = "local"
    SINGLE_AGENT = "single_agent"
    SWARM = "swarm"


class ThermalRequirement(str, Enum):
    """Expected compute-duration profile."""

    BURST = "burst"
    SUSTAINED = "sustained"
    ANY = "any"


# ---------------------------------------------------------------------------
# TaskDecomposition
# ---------------------------------------------------------------------------


@dataclass
class TaskDecomposition:
    """Result of analysing a user query into data / compute requirements."""

    query: str
    required_datasets: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    parallelizable: bool = False
    min_hardware_tier: str = "t1"
    estimated_complexity: float = 0.5

    # -- routing heuristic ---------------------------------------------------

    @property
    def recommended_route(self) -> ExecutionRoute:
        """Heuristic route selection based on decomposition attributes."""
        if not self.required_datasets:
            return ExecutionRoute.DIRECT_LLM
        if self.parallelizable or len(self.required_datasets) > 1:
            return ExecutionRoute.SWARM
        return ExecutionRoute.SINGLE_AGENT

    # -- serialisation -------------------------------------------------------

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
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDecomposition":
        return cls(
            query=data["query"],
            required_datasets=data.get("required_datasets", []),
            operations=data.get("operations", []),
            parallelizable=data.get("parallelizable", False),
            min_hardware_tier=data.get("min_hardware_tier", "t1"),
            estimated_complexity=data.get("estimated_complexity", 0.5),
        )


# ---------------------------------------------------------------------------
# TaskPlan
# ---------------------------------------------------------------------------


@dataclass
class TaskPlan:
    """Concrete execution plan produced from a TaskDecomposition."""

    decomposition: TaskDecomposition
    route: ExecutionRoute
    target_shard_cids: List[str] = field(default_factory=list)
    estimated_pcu: float = 0.0
    thermal_requirement: ThermalRequirement = ThermalRequirement.ANY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decomposition": self.decomposition.to_dict(),
            "route": self.route.value,
            "target_shard_cids": self.target_shard_cids,
            "estimated_pcu": self.estimated_pcu,
            "thermal_requirement": self.thermal_requirement.value,
        }


# ---------------------------------------------------------------------------
# AgentTrace
# ---------------------------------------------------------------------------


@dataclass
class AgentTrace:
    """Full execution trace for auditing / feedback loops."""

    query: str
    decomposition: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
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
