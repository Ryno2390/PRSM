"""
Agent Instruction Set
=====================

Defines the operations a WASM mobile agent can perform on data shards.
Instead of generating raw WASM code, the agent forge creates an
instruction manifest that a pre-built WASM executor interprets.
"""

import json
import logging
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class AgentOp(str, Enum):
    """Operations available to mobile agents."""
    FILTER = "filter"              # Filter records by field value
    AGGREGATE = "aggregate"        # Sum, count, avg, min, max
    GROUP_BY = "group_by"          # Group records by field
    SORT = "sort"                  # Sort by field
    LIMIT = "limit"               # Take first N records
    COUNT = "count"               # Count records
    SUM = "sum"                   # Sum a numeric field
    AVERAGE = "average"           # Average a numeric field
    SELECT = "select"             # Select specific fields
    COMPARE = "compare"           # Compare values across groups
    TIME_SERIES = "time_series"   # Time-based analysis


@dataclass
class AgentInstruction:
    """A single operation in an agent's execution plan."""
    op: AgentOp
    field: str = ""
    value: Any = None
    params: Dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op.value,
            "field": self.field,
            "value": self.value,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentInstruction":
        return cls(
            op=AgentOp(d["op"]),
            field=d.get("field", ""),
            value=d.get("value"),
            params=d.get("params", {}),
        )


@dataclass
class InstructionManifest:
    """Complete instruction set for a mobile agent.

    This is what the agent forge generates from a user query.
    The WASM executor interprets these instructions on the target node.
    """
    query: str
    instructions: List[AgentInstruction] = dataclass_field(default_factory=list)
    output_format: str = "json"  # json, csv, summary
    max_output_records: int = 1000

    def to_json(self) -> str:
        return json.dumps({
            "query": self.query,
            "instructions": [i.to_dict() for i in self.instructions],
            "output_format": self.output_format,
            "max_output_records": self.max_output_records,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "InstructionManifest":
        d = json.loads(data)
        return cls(
            query=d.get("query", ""),
            instructions=[AgentInstruction.from_dict(i) for i in d.get("instructions", [])],
            output_format=d.get("output_format", "json"),
            max_output_records=d.get("max_output_records", 1000),
        )

    def to_wasm_input(self) -> bytes:
        """Serialize as bytes for WASM stdin input."""
        return self.to_json().encode()


def instructions_from_decomposition(decomposition_dict: Dict[str, Any]) -> InstructionManifest:
    """Convert a forge TaskDecomposition into agent instructions.

    Maps decomposition operations (from LLM) to concrete AgentOps.
    """
    query = decomposition_dict.get("query", "")
    operations = decomposition_dict.get("operations", [])

    instructions = []
    op_mapping = {
        "filter": AgentOp.FILTER,
        "aggregate": AgentOp.AGGREGATE,
        "group_by": AgentOp.GROUP_BY,
        "group by": AgentOp.GROUP_BY,
        "sort": AgentOp.SORT,
        "count": AgentOp.COUNT,
        "sum": AgentOp.SUM,
        "average": AgentOp.AVERAGE,
        "avg": AgentOp.AVERAGE,
        "select": AgentOp.SELECT,
        "compare": AgentOp.COMPARE,
        "time_series": AgentOp.TIME_SERIES,
        "time series": AgentOp.TIME_SERIES,
    }

    for op_str in operations:
        op_lower = op_str.lower().strip()
        # Try exact match first
        if op_lower in op_mapping:
            instructions.append(AgentInstruction(op=op_mapping[op_lower]))
        else:
            # Try partial match
            for key, op_enum in op_mapping.items():
                if key in op_lower:
                    instructions.append(AgentInstruction(
                        op=op_enum,
                        params={"description": op_str},
                    ))
                    break

    if not instructions:
        # Default: just count records
        instructions.append(AgentInstruction(op=AgentOp.COUNT))

    return InstructionManifest(query=query, instructions=instructions)
