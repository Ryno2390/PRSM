"""Sprint 302 — formal-invariant HTTP + MCP surface.

GET  /admin/formal-verification/invariants     — PUBLIC list
GET  /admin/formal-verification/check          — run all
GET  /admin/formal-verification/check/{id}     — run one

prsm_formal_verification MCP tool: list | check | check_one.

The /invariants endpoint is PUBLIC by design (Vision §14
transparency promise — the formal spec is published before
any incident, in the same posture as the §14 item 5
playbook).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.formal_invariants import (
    INVARIANT_REGISTRY, Invariant, InvariantChecker,
    InvariantKind, InvariantSeverity,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_formal_verification,
)
from prsm.node.api import create_api_app


class _FakeChecker:
    """Stub for InvariantChecker — captures calls + returns
    canned results."""

    def __init__(self, results):
        self._results = results
        self.last_call = None

    def check_contract(
        self, contract_name, *, contract_address,
    ):
        self.last_call = (contract_name, contract_address)
        return list(self._results)

    def check_one(self, inv, contract_address):
        self.last_call = (inv.id, contract_address)
        return self._results[0]


def _client(checker=None, addresses=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._formal_invariant_checker = checker
    node._formal_invariant_addresses = addresses or {}
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── /invariants (PUBLIC list) ────────────────────────


def test_list_invariants_returns_registry():
    body = _client().get(
        "/admin/formal-verification/invariants",
    ).json()
    assert "invariants" in body
    rd_invs = [
        i for i in body["invariants"]
        if i["contract_name"] == "royalty_distributor"
    ]
    assert len(rd_invs) >= 5
    # No callable fields leak into JSON
    assert all("check_fn" not in i for i in rd_invs)


def test_list_invariants_filter_by_contract():
    body = _client().get(
        "/admin/formal-verification/invariants"
        "?contract=royalty_distributor",
    ).json()
    contracts = {i["contract_name"] for i in body["invariants"]}
    assert contracts == {"royalty_distributor"}


def test_list_invariants_unknown_contract_empty():
    body = _client().get(
        "/admin/formal-verification/invariants"
        "?contract=nope",
    ).json()
    assert body["invariants"] == []


# ── /check (run all) ─────────────────────────────────


from prsm.economy.web3.formal_invariants import (
    InvariantResult, InvariantStatus,
)


def test_check_503_when_unwired():
    resp = _client(checker=None).get(
        "/admin/formal-verification/check"
        "?contract=royalty_distributor",
    )
    assert resp.status_code == 503


def test_check_503_when_address_unset():
    checker = _FakeChecker([])
    resp = _client(
        checker=checker, addresses={},
    ).get(
        "/admin/formal-verification/check"
        "?contract=royalty_distributor",
    )
    assert resp.status_code == 503


def test_check_happy_path():
    results = [
        InvariantResult(
            invariant_id="INV-RD-1",
            status=InvariantStatus.PASS,
            value=200, expected=200,
        ),
        InvariantResult(
            invariant_id="INV-RD-4",
            status=InvariantStatus.FAIL,
            value=1000, expected=2000,
            diagnostic="balance=1000, totalClaimable=2000",
        ),
    ]
    checker = _FakeChecker(results)
    body = _client(
        checker=checker,
        addresses={
            "royalty_distributor": "0x" + "ab" * 20,
        },
    ).get(
        "/admin/formal-verification/check"
        "?contract=royalty_distributor",
    ).json()
    assert body["contract"] == "royalty_distributor"
    assert body["address"] == "0x" + "ab" * 20
    assert body["summary"]["pass"] == 1
    assert body["summary"]["fail"] == 1
    assert body["summary"]["skipped"] == 0
    assert len(body["results"]) == 2


def test_check_422_unknown_contract():
    checker = _FakeChecker([])
    resp = _client(
        checker=checker,
        addresses={
            "royalty_distributor": "0x" + "ab" * 20,
        },
    ).get(
        "/admin/formal-verification/check?contract=nope",
    )
    assert resp.status_code == 422


# ── /check/{invariant_id} (run one) ──────────────────


def test_check_one_503_when_unwired():
    resp = _client(checker=None).get(
        "/admin/formal-verification/check/INV-RD-1",
    )
    assert resp.status_code == 503


def test_check_one_404_unknown_id():
    checker = _FakeChecker([])
    resp = _client(
        checker=checker,
        addresses={
            "royalty_distributor": "0x" + "ab" * 20,
        },
    ).get(
        "/admin/formal-verification/check/INV-DOES-NOT-EXIST",
    )
    assert resp.status_code == 404


def test_check_one_happy_path():
    result = InvariantResult(
        invariant_id="INV-RD-1",
        status=InvariantStatus.PASS,
        value=200, expected=200,
    )
    checker = _FakeChecker([result])
    body = _client(
        checker=checker,
        addresses={
            "royalty_distributor": "0x" + "ab" * 20,
        },
    ).get(
        "/admin/formal-verification/check/INV-RD-1",
    ).json()
    assert body["invariant_id"] == "INV-RD-1"
    assert body["status"] == "pass"


# ── MCP ──────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_formal_verification" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_formal_verification({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_formal_verification(
        {"action": "boom"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list_renders_table():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "invariants": [
                {
                    "id": "INV-RD-1",
                    "contract_name": "royalty_distributor",
                    "title": "Network fee is 200 bps",
                    "description": "x",
                    "severity": "critical",
                    "spec_text": "NETWORK_FEE_BPS() == 200",
                    "kind": "uint256_eq",
                    "selector": "0xdead",
                    "expected": 200,
                    "params": {},
                },
            ],
        }),
    ):
        r = await handle_prsm_formal_verification({
            "action": "list",
        })
    assert "INV-RD-1" in r
    assert "critical" in r.lower()
    assert "200" in r


@pytest.mark.asyncio
async def test_mcp_check_renders_summary():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "contract": "royalty_distributor",
            "address": "0x" + "ab" * 20,
            "summary": {"pass": 4, "fail": 1, "skipped": 0},
            "results": [
                {
                    "invariant_id": "INV-RD-1",
                    "status": "pass",
                    "value": 200, "expected": 200,
                    "diagnostic": "",
                    "error": None,
                },
                {
                    "invariant_id": "INV-RD-4",
                    "status": "fail",
                    "value": 1000, "expected": 2000,
                    "diagnostic": (
                        "balance=1000, totalClaimable=2000"
                    ),
                    "error": None,
                },
            ],
        }),
    ) as mock_call:
        r = await handle_prsm_formal_verification({
            "action": "check",
            "contract": "royalty_distributor",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/formal-verification/check"
        "?contract=royalty_distributor"
    )
    assert "INV-RD-4" in r
    assert "fail" in r.lower()
    assert "1/5" in r or "4 pass" in r.lower()


@pytest.mark.asyncio
async def test_mcp_check_one():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "invariant_id": "INV-RD-1",
            "status": "pass",
            "value": 200, "expected": 200,
            "diagnostic": "",
            "error": None,
        }),
    ) as mock_call:
        r = await handle_prsm_formal_verification({
            "action": "check_one",
            "invariant_id": "INV-RD-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/formal-verification/check/INV-RD-1"
    )
    assert "INV-RD-1" in r
    assert "pass" in r.lower()


@pytest.mark.asyncio
async def test_mcp_check_requires_contract():
    r = await handle_prsm_formal_verification({
        "action": "check",
    })
    assert "contract" in r.lower()


@pytest.mark.asyncio
async def test_mcp_check_one_requires_id():
    r = await handle_prsm_formal_verification({
        "action": "check_one",
    })
    assert "invariant_id" in r.lower()
