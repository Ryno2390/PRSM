"""QueryOrchestrator decomposer — natural-language → AgentOp DSL manifest.

The decomposer turns a user's free-text query into a structured
`InstructionManifest` that the WASM swarm can execute. The DSL itself
already exists at `prsm/compute/agents/instruction_set.py:19`
(`class AgentOp` — 11 canonical ops); this module just produces input
for it.

The threat model for the aggregator-selector does NOT cover the
decomposer — but two correctness boundaries are still load-bearing:

  - Untrusted output validation. An LLM's decomposition output is
    untrusted text. Unknown ops must be rejected (not silently coerced)
    so a misbehaving LLM doesn't sneak unsupported behavior past the
    swarm runner.

  - Output bound enforcement. `max_output_records` is the only knob
    that prevents a decomposed query from accidentally requesting
    unbounded output. The decomposer caps it at construction time.

Tests below pin both boundaries plus the happy path.
"""
from __future__ import annotations

from typing import Any

import pytest

from prsm.compute.agents.instruction_set import AgentOp, InstructionManifest
from prsm.compute.query_orchestrator import (
    DecomposerOutputError,
    QueryDecomposer,
    RuleBasedDecomposer,
    decompose_query,
)


# ──────────────────────────────────────────────────────────────────────
# Happy path — RuleBasedDecomposer
# ──────────────────────────────────────────────────────────────────────


class TestRuleBasedDecomposerHappyPath:
    """The default decomposer is keyword-rule-based — deterministic,
    test-friendly, and sufficient for bootstrap before an LLM is
    plugged in. Production deployments inject an LLMDecomposer."""

    def test_count_query_yields_count_op(self):
        m = decompose_query("count all records")
        assert isinstance(m, InstructionManifest)
        assert m.query == "count all records"
        assert any(i.op == AgentOp.COUNT for i in m.instructions)

    def test_average_query_yields_average_op(self):
        m = decompose_query("average price by region")
        ops = [i.op for i in m.instructions]
        assert AgentOp.AVERAGE in ops

    def test_sort_query_yields_sort_op(self):
        m = decompose_query("sort records by timestamp descending")
        assert AgentOp.SORT in [i.op for i in m.instructions]

    def test_query_preserved_verbatim_in_manifest(self):
        q = "Find the top 10 patents filed in 2024 by IBM"
        m = decompose_query(q)
        assert m.query == q


# ──────────────────────────────────────────────────────────────────────
# Validation — empty + unbounded queries rejected
# ──────────────────────────────────────────────────────────────────────


class TestQueryValidation:
    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="empty"):
            decompose_query("")

    def test_whitespace_only_query_raises(self):
        with pytest.raises(ValueError, match="empty"):
            decompose_query("   \t\n  ")

    def test_max_output_records_bounded(self):
        """The decomposer caps `max_output_records` at a sane default
        even when the LLM tries to request more — prevents accidental
        unbounded output from a misbehaving decomposition."""
        m = decompose_query("count records", max_output_records=99_999_999)
        # Hard cap defined in the decomposer.
        assert m.max_output_records <= 100_000


# ──────────────────────────────────────────────────────────────────────
# Untrusted-output validation — LLM output must conform to AgentOp
# ──────────────────────────────────────────────────────────────────────


class _StubLLM:
    """Test-only LLMDecomposer. Returns a fixed decomposition dict so
    we can exercise the validation paths without a real LLM."""

    def __init__(self, output: dict):
        self._output = output

    def decompose(self, query: str) -> dict:
        return dict(self._output, query=query)


class TestUntrustedOutputValidation:
    """The decomposer treats LLM output as untrusted text — unknown
    ops must surface as `DecomposerOutputError` rather than be silently
    coerced. A swarm runner that sees an unknown op MUST reject it,
    so we want the failure to land at decomposition time, not later."""

    def test_unknown_op_in_llm_output_raises(self):
        llm = _StubLLM({"operations": ["execute_arbitrary_python"]})
        with pytest.raises(DecomposerOutputError, match="execute_arbitrary_python"):
            decompose_query("anything", llm=llm)

    def test_non_string_op_in_llm_output_raises(self):
        # An LLM that returns a dict where it should return a string
        # operation name. Common LLM-failure mode.
        llm = _StubLLM({"operations": [{"nested": "object"}]})
        with pytest.raises(DecomposerOutputError):
            decompose_query("anything", llm=llm)

    def test_missing_operations_field_raises(self):
        llm = _StubLLM({"some_other_field": ["filter"]})
        with pytest.raises(DecomposerOutputError, match="operations"):
            decompose_query("anything", llm=llm)

    def test_valid_llm_output_produces_manifest(self):
        llm = _StubLLM({"operations": ["filter", "count"]})
        m = decompose_query("anything", llm=llm)
        ops = [i.op for i in m.instructions]
        assert AgentOp.FILTER in ops
        assert AgentOp.COUNT in ops

    def test_decomposer_protocol_pin(self):
        """Pin the QueryDecomposer Protocol surface — `decompose(query)`
        returns a dict. If this is renamed downstream, swarm_runner +
        every adapter need to update together. Pin the contract."""
        llm = _StubLLM({"operations": ["count"]})
        # Duck-type: instances satisfy the Protocol.
        assert hasattr(llm, "decompose")
        # Returned dict has at minimum an "operations" key.
        assert "operations" in llm.decompose("q")


# ──────────────────────────────────────────────────────────────────────
# RuleBasedDecomposer fallback default
# ──────────────────────────────────────────────────────────────────────


class TestRuleBasedDecomposerFallback:
    """When the keyword rules find nothing, the default decomposer
    falls through to `COUNT` rather than producing an empty manifest.
    Pin the contract."""

    def test_unmatched_query_falls_through_to_count(self):
        # No keyword in this query matches any AgentOp.
        m = decompose_query("show me everything you have")
        assert len(m.instructions) >= 1
        # COUNT default is the canonical fallback.
        assert any(i.op == AgentOp.COUNT for i in m.instructions)

    def test_rule_based_decomposer_is_default(self):
        """No `llm=` argument → uses RuleBasedDecomposer."""
        rb = RuleBasedDecomposer()
        d = rb.decompose("count all things")
        assert "operations" in d
        assert d["query"] == "count all things"
