"""
PRSM WASM Mobile Agent Runtime
===============================

Lightweight, stateless WASM mobile agents dispatched by third-party LLMs via
MCP tools. Agents execute in a sandbox on provider nodes, query node data,
and return results to the requester.

Core modules:
- dispatcher: Ring 2 Courier — requester-side dispatch lifecycle
- executor: Provider-side WASM execution in WasmtimeRuntime sandbox
- instruction_set: WASM instruction definitions (filter, aggregate, etc.)
- models: Data models (AgentManifest, MobileAgent, DispatchRecord, etc.)
- data_processor: Instruction execution against tabular data
"""

from prsm.compute.agents import dispatcher, executor, instruction_set, models, data_processor

# Re-export key classes for backward compatibility with Ring 9 tests
from prsm.compute.agents.dispatcher import AgentDispatcher
from prsm.compute.agents.executor import AgentExecutor
from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)
from prsm.compute.agents.instruction_set import (
    AgentOp,
    AgentInstruction,
    InstructionManifest,
    instructions_from_decomposition,
)

__all__ = [
    "dispatcher",
    "executor",
    "instruction_set",
    "models",
    "data_processor",
    # Class-level exports for backward compatibility
    "AgentDispatcher",
    "AgentExecutor",
    "AgentManifest",
    "MobileAgent",
    "DispatchStatus",
    "DispatchRecord",
    "AgentOp",
    "AgentInstruction",
    "InstructionManifest",
    "instructions_from_decomposition",
]
