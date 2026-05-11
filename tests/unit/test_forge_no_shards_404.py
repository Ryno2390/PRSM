"""Sprint 175 — /compute/forge returns 404 (not 500) when the
QueryOrchestrator finds no shards above similarity threshold.

Pre-fix the orchestrator raised ValueError with message
"no shards above similarity threshold for this query — nothing
to aggregate" when the local semantic index had no matching
content. /compute/forge caught the generic Exception and mapped
it to a 500. But this is NOT a server error — it's a
query/content mismatch the operator can fix (upload content or
refine query).

Surface during sprint 174 dogfood:
  PRSM_QUERY_ORCHESTRATOR_ENABLED=1 fresh node, empty content
  curl /compute/forge?query=hello → 500 "Forge pipeline error..."

Post-fix: 404 with actionable detail.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_with_forge_raising_no_shards():
    """Node with a stub agent_forge whose run() raises the
    no-shards ValueError that the real QueryOrchestrator raises."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._operator_address = None

    # Stub forge that raises the canonical no-shards error.
    forge = MagicMock()
    forge.run = AsyncMock(
        side_effect=ValueError(
            "no shards above similarity threshold for this query — "
            "nothing to aggregate"
        ),
    )
    # Also support the QO duck-type dispatch surface — if the handler
    # checks for dispatch_query before run, it'll see this attribute.
    forge.dispatch_query = AsyncMock(
        side_effect=ValueError(
            "no shards above similarity threshold for this query — "
            "nothing to aggregate"
        ),
    )
    node.agent_forge = forge
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_no_shards_returns_404():
    """Sprint 175 — no-shards is 404, not 500."""
    resp = _client(_node_with_forge_raising_no_shards()).post(
        "/compute/forge",
        json={"query": "hello", "budget_ftns": 1.0},
    )
    assert resp.status_code == 404
    assert "shard" in resp.json()["detail"].lower()


def test_other_forge_errors_still_500():
    """Sprint 175 invariant — only the no-shards class flips
    to 404. Other unexpected pipeline errors stay 500."""
    node = _node_with_forge_raising_no_shards()
    node.agent_forge.run = AsyncMock(
        side_effect=RuntimeError("unexpected pipeline crash"),
    )
    node.agent_forge.dispatch_query = AsyncMock(
        side_effect=RuntimeError("unexpected pipeline crash"),
    )
    resp = _client(node).post(
        "/compute/forge",
        json={"query": "hello", "budget_ftns": 1.0},
    )
    assert resp.status_code == 500
