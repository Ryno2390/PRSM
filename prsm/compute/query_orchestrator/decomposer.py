"""QueryOrchestrator — natural-language query decomposer.

Turns a user's free-text query into a structured `InstructionManifest`
the WASM swarm can execute.

The DSL itself already exists at
`prsm/compute/agents/instruction_set.py` (`AgentOp` — 11 canonical
ops + `instructions_from_decomposition` helper). This module just
produces the input dict that helper consumes.

Two correctness boundaries are load-bearing:

  - **Untrusted output validation.** An LLM's decomposition output is
    untrusted text. Unknown ops surface as `DecomposerOutputError`
    rather than being silently coerced — a swarm runner that later
    sees an unknown op MUST reject it, so we want the failure to land
    at decomposition time, not in a remote agent's WASM sandbox.

  - **Output bound enforcement.** `max_output_records` is the only
    knob that prevents a decomposed query from accidentally requesting
    unbounded output. Capped here at construction time.

Two implementations ship:

  - `RuleBasedDecomposer` — keyword-matching, deterministic, used as
    the default when no LLM is wired. Sufficient for bootstrap +
    test surfaces; production deployments inject an LLMDecomposer.

  - `LLMDecomposer` Protocol — any object with `.decompose(query)`
    returning a dict. The orchestrator's startup wiring constructs one
    backed by the local-node model registry (Phase 3.x.2).

Threat-model note: NO content of this module is in the
aggregator-selector A1–A10 catalog. The decomposer's outputs are
consumed by the orchestrator BEFORE shard dispatch — there's no
per-query identity binding here. If an LLM-side prompt-injection
becomes a concern (e.g., a malicious shard's content reaches the
decomposer's LLM context), that's a separate threat model.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from prsm.compute.agents.instruction_set import (
    AgentOp,
    InstructionManifest,
    instructions_from_decomposition,
)


# Hard cap — even if an LLM (or an upstream caller) requests more,
# never let `max_output_records` exceed this. Keeps a single rogue
# query from accidentally fanning out unbounded WASM output.
#
# Ratification target: Foundation council parameter alongside the six
# aggregator-selector governance questions; not currently load-bearing.
MAX_OUTPUT_RECORDS_HARD_CAP = 100_000


class DecomposerOutputError(ValueError):
    """The decomposer received output that does not conform to the
    expected schema. Distinct from ValueError so the orchestrator's
    error router can distinguish "bad LLM output" from "bad query
    input"."""


@runtime_checkable
class QueryDecomposer(Protocol):
    """Pluggable decomposer interface. Production deployments wire an
    LLM-backed implementation; tests + bootstrap use
    `RuleBasedDecomposer`.

    The contract: `decompose(query)` returns a dict with at minimum
    an `operations` key (list of strings, each a known AgentOp name).
    The dict may additionally carry `query` (echoed back), and other
    keys are ignored. Unknown op strings cause the orchestrator to
    raise `DecomposerOutputError`.
    """

    def decompose(self, query: str) -> dict[str, Any]: ...


# ──────────────────────────────────────────────────────────────────────
# Default rule-based decomposer
# ──────────────────────────────────────────────────────────────────────


_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    # Order matters — first match wins. Specific phrases before
    # general ones so "group by" is recognized before "by".
    ("group by", "group_by"),
    ("group_by", "group_by"),
    ("time series", "time_series"),
    ("time_series", "time_series"),
    ("average", "average"),
    ("avg ", "average"),
    ("compare", "compare"),
    ("filter", "filter"),
    ("aggregate", "aggregate"),
    ("sort", "sort"),
    ("limit", "limit"),
    ("count", "count"),
    ("sum", "sum"),
    ("select", "select"),
)


class RuleBasedDecomposer:
    """Deterministic keyword-rule decomposer. The default when no
    LLM is wired.

    Walks `_KEYWORD_RULES` in order; collects every matching op into
    the operations list. If no rule matches, falls through to a
    single `count` op (matches the existing
    `instructions_from_decomposition` empty-input default — keep the
    contract aligned)."""

    def decompose(self, query: str) -> dict[str, Any]:
        q_lower = query.lower()
        ops: list[str] = []
        for keyword, op_name in _KEYWORD_RULES:
            if keyword in q_lower and op_name not in ops:
                ops.append(op_name)
        if not ops:
            # Fallthrough: see `instructions_from_decomposition`'s own
            # empty-instructions default.
            ops.append("count")
        return {"query": query, "operations": ops}


# ──────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────


def _validate_decomposition(d: Any) -> dict[str, Any]:
    """Tighten the decomposer-output contract.

    Raises `DecomposerOutputError` if:
      - `d` is not a dict
      - `d` lacks an `operations` key
      - `operations` is not a list
      - any operation is not a string
      - any operation string is not a valid `AgentOp` name (after
        normalization to lower / underscore)
    """
    if not isinstance(d, dict):
        raise DecomposerOutputError(
            f"decomposer output must be a dict, got {type(d).__name__}"
        )
    if "operations" not in d:
        raise DecomposerOutputError(
            f"decomposer output missing required 'operations' key "
            f"(keys: {list(d.keys())})"
        )
    ops = d["operations"]
    if not isinstance(ops, list):
        raise DecomposerOutputError(
            f"'operations' must be a list, got {type(ops).__name__}"
        )

    valid_op_names = {op.value for op in AgentOp}
    valid_op_names.update({"avg", "group by", "time series"})  # aliases
    for op in ops:
        if not isinstance(op, str):
            raise DecomposerOutputError(
                f"each operation must be a string, got {type(op).__name__}: {op!r}"
            )
        normalized = op.lower().strip()
        if normalized not in valid_op_names:
            raise DecomposerOutputError(
                f"unknown operation: {op!r} "
                f"(valid: {sorted(valid_op_names)})"
            )
    return d


def decompose_query(
    query: str,
    *,
    llm: QueryDecomposer | None = None,
    max_output_records: int = 1000,
) -> InstructionManifest:
    """Decompose a natural-language query into an `InstructionManifest`.

    `llm` plugs in any object satisfying `QueryDecomposer`. When None,
    the deterministic `RuleBasedDecomposer` runs — sufficient for
    bootstrap + tests; production deployments wire the local-node LLM.

    `max_output_records` is hard-capped at `MAX_OUTPUT_RECORDS_HARD_CAP`
    regardless of the request — prevents accidental unbounded fan-out.

    Raises:
        ValueError: query is empty / whitespace-only.
        DecomposerOutputError: the decomposer's output does not
            conform to the schema (unknown op, bad type, missing key).
    """
    if not query or not query.strip():
        raise ValueError("query is empty")

    decomposer = llm if llm is not None else RuleBasedDecomposer()
    raw = decomposer.decompose(query)
    validated = _validate_decomposition(raw)

    manifest = instructions_from_decomposition(validated)
    # Preserve the user's exact query string. The decomposer may have
    # stuffed something else in there.
    manifest.query = query
    manifest.max_output_records = min(
        int(max_output_records),
        MAX_OUTPUT_RECORDS_HARD_CAP,
    )
    return manifest
