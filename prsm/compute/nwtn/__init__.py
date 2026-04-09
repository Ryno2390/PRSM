"""
PRSM Future NWTN Model Infrastructure (Ring 9 — "The Mind")
============================================================

Ring 9 of the Sovereign-Edge AI architecture: training pipeline and model
service for the future fine-tuned NWTN model that will create WASM agents.

Currently PRSM users rely on third-party LLMs (local or via OAuth/API);
once trained, the NWTN model will be added as an LLM option that
dispatches WASM mobile agents to find data, run computations, and report
back to staked aggregator nodes.

Core modules:
- training: AgentTrace collection, validation, JSONL export for fine-tuning

The legacy NWTN AGI framework (orchestrator, meta_reasoning_engine,
voicebox, agent_forge, etc.) was removed in v1.6.0. Do not reintroduce
those concepts — reasoning belongs in third-party LLMs, not in PRSM.
"""

from prsm.compute.nwtn import training

__all__ = ["training"]
