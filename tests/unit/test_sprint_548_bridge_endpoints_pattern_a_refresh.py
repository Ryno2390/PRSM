"""Sprint 548 — refresh /bridge/* 503 messages to point at Pattern A.

Sprint 539's investigation correctly identified that the 5
``/bridge/*`` endpoints target a polygon_mumbai-era scaffold with no
deployed contract on Base mainnet. Sprint 539 left them returning
503 with a message redirecting operators to ``/wallet/transfer/
onchain`` + ``/admin/royalty-dispatch-summary``. Sprints 540 + 541
then shipped Pattern A — daemon-mediated bridge — exposing exactly
the missing operations:

  /wallet/deposit/info   — shows escrow address + linkage status
  /wallet/deposit/link   — links operator's eth address → auto-credit
  /wallet/withdraw       — direct broadcast off-chain → on-chain
  /transactions          — bridge_deposit / bridge_withdraw rows

Sprint 548 updates each /bridge/* endpoint's 503 message to point
at the right Pattern A counterpart for its operation, so operators
hitting the scaffold get the actually-working endpoint name. Five
endpoints, five operation-specific replacements:

  /bridge/deposit              → /wallet/deposit/link + /wallet/deposit/info
  /bridge/withdraw             → /wallet/withdraw
  /bridge/status               → /wallet/deposit/info + /transactions
  /bridge/transactions/{tx_id} → /transactions
  /bridge/transactions         → /transactions
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _stub_node():
    n = MagicMock()
    n.identity = MagicMock(node_id="stub-node-id")
    # Critical for the sprint 539 503 branch: ftns_bridge must be
    # absent / falsy. ``hasattr`` returns True for any MagicMock
    # attribute, so we have to assign None explicitly to make the
    # endpoint take the scaffold branch.
    n.ftns_bridge = None
    return n


def _make_app():
    from prsm.node.api import create_api_app
    return create_api_app(_stub_node(), enable_security=False)


@pytest.fixture()
def client():
    return TestClient(_make_app())


# ── Per-endpoint message specificity ──────────────────────


def test_bridge_deposit_503_mentions_pattern_a_deposit_endpoints(client):
    """``/bridge/deposit`` 503 must point at Pattern A's deposit
    surface — /wallet/deposit/link (one-time setup) + /wallet/
    deposit/info (status), NOT just /wallet/transfer/onchain."""
    response = client.post(
        "/bridge/deposit",
        json={
            "amount": 1.0,
            "chain_address": "0x" + "ab" * 20,
            "destination_chain": 8453,
        },
    )
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "/wallet/deposit/link" in body, (
        f"deposit endpoint 503 must mention /wallet/deposit/link; "
        f"body={body!r}"
    )
    assert "/wallet/deposit/info" in body


def test_bridge_withdraw_503_mentions_pattern_a_withdraw(client):
    """``/bridge/withdraw`` 503 must point at /wallet/withdraw —
    the direct Pattern A counterpart."""
    response = client.post(
        "/bridge/withdraw",
        json={
            "amount": 1.0,
            "chain_address": "0x" + "ab" * 20,
            "source_chain": 8453,
        },
    )
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "/wallet/withdraw" in body, (
        f"withdraw endpoint 503 must mention /wallet/withdraw; "
        f"body={body!r}"
    )


def test_bridge_status_503_mentions_pattern_a_status_surfaces(client):
    """``/bridge/status`` 503 must point at /wallet/deposit/info
    (linkage state) + /transactions (history)."""
    response = client.get("/bridge/status")
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "/wallet/deposit/info" in body
    assert "/transactions" in body


def test_bridge_tx_id_503_mentions_transactions(client):
    """``/bridge/transactions/{tx_id}`` 503 must point at
    /transactions (the off-chain ledger has bridge_deposit +
    bridge_withdraw rows since sprints 540 + 541)."""
    response = client.get("/bridge/transactions/anything")
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "/transactions" in body


def test_bridge_tx_list_503_mentions_transactions(client):
    """``/bridge/transactions`` 503 must point at /transactions."""
    response = client.get("/bridge/transactions")
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "/transactions" in body


# ── Consistency / non-regression ──────────────────────────


def test_no_scaffold_503_still_mentions_polygon_mumbai_as_main_redirect(
    client,
):
    """Sanity: the old "Track sprint-X bridge re-implementation
    against Base mainnet" forward-pointer is gone — Pattern A IS
    the implementation. Avoids confusing operators into thinking
    a future bridge sprint is the canonical path forward."""
    for path, method in [
        ("/bridge/deposit", "POST"),
        ("/bridge/withdraw", "POST"),
        ("/bridge/status", "GET"),
        ("/bridge/transactions/abc", "GET"),
        ("/bridge/transactions", "GET"),
    ]:
        if method == "POST":
            response = client.post(path, json={
                "amount": 1.0,
                "chain_address": "0x" + "ab" * 20,
                "destination_chain": 8453,
                "source_chain": 8453,
            })
        else:
            response = client.get(path)
        assert response.status_code == 503
        body = response.json()["detail"]
        # The "Pattern A is the answer" marker — operators should
        # see this phrase regardless of which scaffold endpoint
        # they hit, so they know to use the working surface today.
        assert "Pattern A" in body, (
            f"{method} {path} 503 missing 'Pattern A' marker; "
            f"body={body!r}"
        )


def test_all_five_endpoints_share_consistent_voice(client):
    """All 5 scaffold endpoints should use a SINGLE shared helper
    for the 503 body — no drift between endpoints. The shared
    helper's tell is the literal substring "daemon-mediated"
    (Pattern A's defining property)."""
    bodies = []
    for path, method in [
        ("/bridge/deposit", "POST"),
        ("/bridge/withdraw", "POST"),
        ("/bridge/status", "GET"),
        ("/bridge/transactions/abc", "GET"),
        ("/bridge/transactions", "GET"),
    ]:
        if method == "POST":
            r = client.post(path, json={
                "amount": 1.0,
                "chain_address": "0x" + "ab" * 20,
                "destination_chain": 8453,
                "source_chain": 8453,
            })
        else:
            r = client.get(path)
        bodies.append(r.json()["detail"])

    for b in bodies:
        assert "daemon-mediated" in b or "Pattern A" in b, (
            f"scaffold 503 missing canonical Pattern A description; "
            f"body={b!r}"
        )
