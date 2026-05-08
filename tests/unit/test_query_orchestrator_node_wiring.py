"""B7 — node-side QueryOrchestrator construction factory.

The QueryOrchestrator class accepts 5 required dependencies +
1 optional. `build_query_orchestrator_for_node(...)` is the
factory `node.py:1277` calls — passing the node's existing
primitives + the 4 supporting adapters (B7-prep work).

This module's job is wiring composition + the env-driven enable
gate. When disabled (default), returns None — agent_forge stays
None, MCP gating stays on, behavior unchanged. When enabled,
returns a fully-wired QueryOrchestrator ready for dispatch.

Per `docs/2026-05-08-query-orchestrator-wiring-readiness.md` B7.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from prsm.compute.query_orchestrator import (
    AggregationCommit,
    QueryOrchestrator,
)
from prsm.compute.query_orchestrator.node_wiring import (
    build_query_orchestrator_for_node,
    is_query_orchestrator_enabled,
)


# ──────────────────────────────────────────────────────────────────────
# Stubs satisfying the 5 dependency Protocols
# ──────────────────────────────────────────────────────────────────────


class _StubSemanticIndex:
    def find_top_k(self, query, k):
        return []


class _StubDispatcher:
    async def fan_out(self, manifest, shards):
        return []


class _StubAggregatorClient:
    async def aggregate(self, aggregator, manifest, partials, query_id):
        digest = hashlib.sha256(b"x").digest()
        commit = AggregationCommit(
            query_id=query_id,
            aggregator_pubkey_hash=aggregator.pubkey_hash,
            result_digest=digest,
        )
        return b"x", commit


def _stub_pool_provider():
    return ()


def _stub_beacon_provider():
    return b"\x00" * 32


# ──────────────────────────────────────────────────────────────────────
# Enable gate — env-driven
# ──────────────────────────────────────────────────────────────────────


class TestEnableGate:
    """Default-disabled. The orchestrator only constructs when the
    operator explicitly opts in via PRSM_QUERY_ORCHESTRATOR_ENABLED."""

    def test_default_disabled(self, monkeypatch):
        monkeypatch.delenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", raising=False)
        assert is_query_orchestrator_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE"])
    def test_truthy_values_enable(self, monkeypatch, val):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", val)
        assert is_query_orchestrator_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "off"])
    def test_falsy_values_keep_disabled(self, monkeypatch, val):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", val)
        assert is_query_orchestrator_enabled() is False


# ──────────────────────────────────────────────────────────────────────
# Disabled path — returns None
# ──────────────────────────────────────────────────────────────────────


class TestDisabledPath:
    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", raising=False)
        result = build_query_orchestrator_for_node(
            semantic_index=_StubSemanticIndex(),
            dispatcher=_StubDispatcher(),
            aggregator_client=_StubAggregatorClient(),
            candidate_pool_provider=_stub_pool_provider,
            beacon_provider=_stub_beacon_provider,
        )
        assert result is None

    def test_returns_none_when_explicitly_disabled(self, monkeypatch):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "0")
        result = build_query_orchestrator_for_node(
            semantic_index=_StubSemanticIndex(),
            dispatcher=_StubDispatcher(),
            aggregator_client=_StubAggregatorClient(),
            candidate_pool_provider=_stub_pool_provider,
            beacon_provider=_stub_beacon_provider,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# Enabled path — returns wired QueryOrchestrator
# ──────────────────────────────────────────────────────────────────────


class TestEnabledPath:
    def test_returns_query_orchestrator_when_enabled(self, monkeypatch):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "1")
        result = build_query_orchestrator_for_node(
            semantic_index=_StubSemanticIndex(),
            dispatcher=_StubDispatcher(),
            aggregator_client=_StubAggregatorClient(),
            candidate_pool_provider=_stub_pool_provider,
            beacon_provider=_stub_beacon_provider,
        )
        assert isinstance(result, QueryOrchestrator)

    def test_dependencies_threaded_through(self, monkeypatch):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "1")
        idx = _StubSemanticIndex()
        disp = _StubDispatcher()
        agg = _StubAggregatorClient()
        result = build_query_orchestrator_for_node(
            semantic_index=idx,
            dispatcher=disp,
            aggregator_client=agg,
            candidate_pool_provider=_stub_pool_provider,
            beacon_provider=_stub_beacon_provider,
        )
        assert result.semantic_index is idx
        assert result.dispatcher is disp
        assert result.aggregator_client is agg
        assert result.candidate_pool_provider is _stub_pool_provider
        assert result.beacon_provider is _stub_beacon_provider

    def test_optional_decomposer_threaded(self, monkeypatch):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "1")
        custom_decomposer = MagicMock()
        result = build_query_orchestrator_for_node(
            semantic_index=_StubSemanticIndex(),
            dispatcher=_StubDispatcher(),
            aggregator_client=_StubAggregatorClient(),
            candidate_pool_provider=_stub_pool_provider,
            beacon_provider=_stub_beacon_provider,
            decomposer=custom_decomposer,
        )
        assert result.decomposer is custom_decomposer

    def test_decomposer_defaults_to_none_for_rule_based(self, monkeypatch):
        # When no decomposer passed, QueryOrchestrator's decomposer
        # field is None — decompose_query falls through to
        # RuleBasedDecomposer at call time.
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "1")
        result = build_query_orchestrator_for_node(
            semantic_index=_StubSemanticIndex(),
            dispatcher=_StubDispatcher(),
            aggregator_client=_StubAggregatorClient(),
            candidate_pool_provider=_stub_pool_provider,
            beacon_provider=_stub_beacon_provider,
        )
        assert result.decomposer is None


# ──────────────────────────────────────────────────────────────────────
# Required-arg validation
# ──────────────────────────────────────────────────────────────────────


class TestRequiredArgValidation:
    """build_query_orchestrator_for_node has 5 required kwargs.
    Pin the contract."""

    def test_missing_semantic_index_raises(self, monkeypatch):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "1")
        with pytest.raises(TypeError):
            build_query_orchestrator_for_node(  # type: ignore
                dispatcher=_StubDispatcher(),
                aggregator_client=_StubAggregatorClient(),
                candidate_pool_provider=_stub_pool_provider,
                beacon_provider=_stub_beacon_provider,
            )

    def test_missing_beacon_provider_raises(self, monkeypatch):
        monkeypatch.setenv("PRSM_QUERY_ORCHESTRATOR_ENABLED", "1")
        with pytest.raises(TypeError):
            build_query_orchestrator_for_node(  # type: ignore
                semantic_index=_StubSemanticIndex(),
                dispatcher=_StubDispatcher(),
                aggregator_client=_StubAggregatorClient(),
                candidate_pool_provider=_stub_pool_provider,
            )
