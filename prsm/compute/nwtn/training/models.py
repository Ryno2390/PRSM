"""NWTN Training Data Models."""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DeploymentStatus(str, Enum):
    REGISTERED = "registered"
    SHARDING = "sharding"
    DEPLOYED = "deployed"
    SERVING = "serving"
    RETIRED = "retired"


@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.1-8B"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 4096
    lora_rank: int = 16
    min_corpus_size: int = 100
    export_format: str = "jsonl"


@dataclass
class TrainingCorpus:
    corpus_id: str = field(default_factory=lambda: f"corpus-{uuid.uuid4().hex[:12]}")
    traces: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def total_traces(self) -> int:
        return len(self.traces)

    def add_trace(self, trace_dict: Dict[str, Any]) -> None:
        self.traces.append(trace_dict)

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.traces:
            errors.append("Corpus is empty")
            return False, errors

        required_fields = {"query", "decomposition", "plan", "execution_result"}
        for i, trace in enumerate(self.traces):
            missing = required_fields - set(trace.keys())
            if missing:
                errors.append(f"Trace {i}: missing fields {missing}")

        return len(errors) == 0, errors

    def export_jsonl(self) -> str:
        lines = []
        for trace in self.traces:
            lines.append(json.dumps(trace, sort_keys=True))
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        routes = {}
        complexities = []
        for trace in self.traces:
            route = trace.get("plan", {}).get("route", "unknown")
            routes[route] = routes.get(route, 0) + 1
            comp = trace.get("decomposition", {}).get("estimated_complexity", 0)
            complexities.append(comp)

        return {
            "total_traces": self.total_traces,
            "route_distribution": routes,
            "avg_complexity": sum(complexities) / len(complexities) if complexities else 0,
            "corpus_id": self.corpus_id,
        }


@dataclass
class ModelCard:
    model_id: str
    model_name: str
    base_model: str
    version: str = "0.1.0"
    training_config: Optional[TrainingConfig] = None
    corpus_stats: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: DeploymentStatus = DeploymentStatus.REGISTERED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "base_model": self.base_model,
            "version": self.version,
            "corpus_stats": self.corpus_stats,
            "metrics": self.metrics,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelCard":
        return cls(
            model_id=d["model_id"],
            model_name=d["model_name"],
            base_model=d.get("base_model", ""),
            version=d.get("version", "0.1.0"),
            corpus_stats=d.get("corpus_stats", {}),
            metrics=d.get("metrics", {}),
            status=DeploymentStatus(d.get("status", "registered")),
            created_at=d.get("created_at", time.time()),
        )
