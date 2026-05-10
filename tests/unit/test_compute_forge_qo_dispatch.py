"""B8 — /compute/forge QueryOrchestrator dispatch path.

The /compute/forge endpoint (``prsm/node/api.py:678``) duck-type-
dispatches on whether ``node.agent_forge`` exposes ``dispatch_query``
(QueryOrchestrator surface) vs the legacy ``run`` method
(AgentForge surface). This module pins the QO path's response shape
so MCP clients calling `prsm_analyze` see a stable contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


# ──────────────────────────────────────────────────────────────────────
# Test fixtures — minimal node stub with a stub QueryOrchestrator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _StubParticipant:
    """Mirrors prsm.compute.query_orchestrator.ParticipantAttribution
    (post §4 step 6 settlement extension)."""
    shard_cid: str
    source_agent_pubkey: bytes
    creator_id: str


@dataclass
class _StubAggregatedResult:
    """Mirrors prsm.compute.query_orchestrator.AggregatedResult."""
    query_id: bytes
    payload: bytes
    aggregator_node_id: str
    contributing_shards: tuple
    participants: tuple = ()


class _StubQueryOrchestrator:
    """Stub satisfying the duck-type check (`hasattr dispatch_query`).
    Records the kwargs it was called with so tests can assert routing
    correctness."""

    def __init__(
        self, *, payload: bytes = b'{"count": 7}', shards=(),
        participants: tuple = (),
    ):
        self.calls = []
        self._payload = payload
        self._shards = shards
        self._participants = participants

    async def dispatch_query(
        self,
        *,
        query: str,
        prompter_node_id: str,
        query_id: bytes,
        requires_tee: bool = False,
        governance_denylist=frozenset(),
    ):
        self.calls.append({
            "query": query,
            "prompter_node_id": prompter_node_id,
            "query_id": query_id,
            "requires_tee": requires_tee,
            "governance_denylist": governance_denylist,
        })
        return _StubAggregatedResult(
            query_id=query_id,
            payload=self._payload,
            aggregator_node_id="agg-node-7",
            contributing_shards=tuple(self._shards),
            participants=tuple(self._participants),
        )


class _LegacyAgentForge:
    """Stub satisfying the legacy `run()` surface — no dispatch_query."""

    def __init__(self):
        self.traces = [object(), object()]

    async def run(self, *, query: str, budget_ftns: float, shard_cids):
        return {
            "status": "success",
            "route": "direct_llm",
            "response": f"legacy-answer-for-{query}",
        }


def _make_node(agent_forge):
    node = MagicMock()
    node.agent_forge = agent_forge
    node.identity.node_id = "test-prompter-node"
    # Disable escrow path for these tests — focus is the dispatch surface.
    node._payment_escrow = None
    # Disable privacy_budget recording.
    node.privacy_budget = None
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# QueryOrchestrator dispatch path
# ──────────────────────────────────────────────────────────────────────


class TestQueryOrchestratorPath:
    def test_dispatch_query_called_with_expected_kwargs(self):
        forge = _StubQueryOrchestrator()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "count records",
            "budget_ftns": 1.0,
        })
        assert resp.status_code == 200
        # dispatch_query was called exactly once with our query +
        # the node's identity + a 32-byte query_id.
        assert len(forge.calls) == 1
        call = forge.calls[0]
        assert call["query"] == "count records"
        assert call["prompter_node_id"] == "test-prompter-node"
        assert isinstance(call["query_id"], bytes)
        assert len(call["query_id"]) == 32

    def test_response_route_is_qo_swarm(self):
        forge = _StubQueryOrchestrator()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        assert resp.json()["route"] == "qo_swarm"

    def test_response_text_is_decoded_payload(self):
        forge = _StubQueryOrchestrator(payload=b'{"count": 42}')
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        assert resp.json()["response"] == '{"count": 42}'

    def test_non_utf8_payload_falls_back_to_hex(self):
        # COUNT op produces UTF-8 JSON, but other ops may produce
        # opaque bytes. Endpoint must not crash on non-UTF-8.
        forge = _StubQueryOrchestrator(payload=b"\xff\xfe\xfd\xfc")
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        assert resp.status_code == 200
        assert resp.json()["response"] == "fffefdfc"

    def test_aggregator_node_id_surfaced(self):
        forge = _StubQueryOrchestrator()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        body = resp.json()
        assert body["result"]["aggregator_node_id"] == "agg-node-7"

    def test_contributing_shards_surfaced(self):
        forge = _StubQueryOrchestrator(shards=("cid-a", "cid-b"))
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        assert resp.json()["result"]["contributing_shards"] == ["cid-a", "cid-b"]

    def test_traces_collected_defaults_to_zero_for_orchestrator(self):
        # QueryOrchestrator has no .traces; legacy AgentForge does.
        # Endpoint must not AttributeError.
        forge = _StubQueryOrchestrator()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        assert resp.json()["traces_collected"] == 0


# ──────────────────────────────────────────────────────────────────────
# Legacy AgentForge dispatch path (backwards-compat)
# ──────────────────────────────────────────────────────────────────────


class TestLegacyAgentForgePath:
    def test_legacy_run_path_still_works(self):
        forge = _LegacyAgentForge()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "legacy-q",
            "budget_ftns": 1.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "direct_llm"
        assert body["response"] == "legacy-answer-for-legacy-q"

    def test_legacy_traces_collected_reflects_real_count(self):
        forge = _LegacyAgentForge()  # 2 traces
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q",
            "budget_ftns": 1.0,
        })
        assert resp.json()["traces_collected"] == 2


# ──────────────────────────────────────────────────────────────────────
# Validation — pre-dispatch checks unchanged
# ──────────────────────────────────────────────────────────────────────


class TestRequestValidation:
    def test_missing_query_400(self):
        forge = _StubQueryOrchestrator()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={"budget_ftns": 1.0})
        assert resp.status_code == 400

    def test_zero_budget_422(self):
        """Sprint 153 — zero budget is a validation error (well-formed
        body, semantically rejected). HTTP 422 is the correct status
        per RFC 4918 §11.2; pre-153 returned 400."""
        forge = _StubQueryOrchestrator()
        client = _client(_make_node(forge))
        resp = client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 0.0,
        })
        assert resp.status_code == 422

    def test_no_agent_forge_503(self):
        node = MagicMock()
        node.agent_forge = None
        client = _client(node)
        resp = client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 1.0,
        })
        assert resp.status_code == 503
