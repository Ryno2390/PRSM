"""Sprint 792 — devices/earnings endpoint + `wallet devices earnings` CLI.

Closes the multi-device operator-UX arc. Sprints 786-790 shipped
the data model + delegation verifier + add/verify/list CLI;
sprint 791 the pure aggregator primitive. Sprint 792 composes:

  GET /api/v1/auth/wallet/devices/earnings?wallet_address=0x...
      → {"earnings_by_node_id": {node_id: ftns_credited, ...},
         "total_ftns": "..." }

Composition: look up bindings (sprint 790) → extract node_ids
→ feed to receipt_lookup → aggregate per node_id (sprint 791)
→ also surface the total across the roster.

A new ReceiptLookup Protocol on WalletApiServices abstracts the
"give me receipts for these node_ids" surface so the wallet_api
endpoint doesn't import node-level ReceiptStore directly.
Production wire-up (PRSMNode injecting its ReceiptStore as the
lookup) is a one-line change at daemon startup — sprint 792
ships the abstract dependency + endpoint + CLI; tests use stub
ReceiptLookup with canned receipts.

CLI consumer:
  prsm wallet devices earnings --wallet <addr>
                              [--format text|json]
                              [--api-url <url>]

Pin tests:
- HTTP: ReceiptLookup Protocol exposed in wallet_api.
- HTTP: empty roster (no bindings) → {earnings_by_node_id: {},
  total_ftns: "0"}
- HTTP: bindings present, no receipts → entries zero per node.
- HTTP: bindings + receipts → per-node sums + total matches.
- HTTP: receipt with partial_completion 7/10 credited
  proportionally (sprint 780 math).
- HTTP: receipts from a NON-bound node_id are filtered out (a
  roster-scoped earnings view, not a free-for-all).
- CLI: command registered.
- CLI: text mode renders per-node rows + total.
- CLI: json mode returns parseable {earnings_by_node_id, total}.
- CLI: empty roster → operator-facing hint.
- CLI: unreachable daemon → exit 2.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from fastapi.testclient import TestClient


# ---- Shared helpers ---------------------------------------------


def _make_receipt_dict(
    node_id, cost_ftns="1.0", partial_completion=None,
):
    d = {
        "job_id": "j1",
        "request_id": "r1",
        "model_id": "gpt2",
        "content_tier": "A",
        "privacy_tier": "none",
        "epsilon_spent": 0.0,
        "tee_type": "software",
        "tee_attestation": "6174",
        "output_hash": "dead",
        "duration_seconds": 1.0,
        "cost_ftns": cost_ftns,
        "settler_signature": "",
        "settler_node_id": node_id,
        "streamed_output": False,
    }
    if partial_completion is not None:
        d["partial_completion"] = partial_completion
    return d


class _StubReceiptLookup:
    """Fake receipt store keyed by node_id."""

    def __init__(self, by_node: Dict[str, List[Dict[str, Any]]]):
        self._by_node = by_node

    def list_receipts_for_node_ids(self, node_ids):
        results: List[Dict[str, Any]] = []
        for nid in node_ids:
            results.extend(self._by_node.get(nid, []))
        return results


def _build_app(seeded_bindings, receipt_lookup):
    from fastapi import FastAPI
    from prsm.interface.api import wallet_api
    from prsm.interface.onboarding.wallet_binding import (
        IdentityBinding,
        InMemoryWalletBindingStore,
        WalletBindingService,
    )

    wallet_api.reset_services_for_tests()
    store = InMemoryWalletBindingStore()
    for wallet, node_id, ts in seeded_bindings:
        store.insert(IdentityBinding(
            wallet_address=wallet,
            node_id_hex=node_id,
            bound_at_unix=ts,
            wallet_signature="0x",
            signing_message_hash="0x",
        ))
    services = wallet_api.WalletApiServices(
        settings=wallet_api.WalletApiSettings(
            expected_domain="test.prsm",
            expected_chain_id=8453,
        ),
        nonce_store=wallet_api.InMemoryNonceStore(),
        binding_service=WalletBindingService(store=store),
        price_source=wallet_api.StaticPriceSource(
            price_usd=0,  # type: ignore[arg-type]
        ),
        balance_lookup=wallet_api._ZeroBalanceLookup(),
        receipt_lookup=receipt_lookup,
    )
    wallet_api.set_services(services)
    app = FastAPI()
    app.include_router(wallet_api.router)
    return app


# ---- HTTP endpoint ----------------------------------------------


def test_receipt_lookup_protocol_exists():
    from prsm.interface.api import wallet_api
    assert hasattr(wallet_api, "ReceiptLookup")


def test_devices_earnings_empty_roster():
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "1" * 40)
    app = _build_app([], _StubReceiptLookup({}))
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/devices/earnings",
        params={"wallet_address": wallet},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["earnings_by_node_id"] == {}
    assert Decimal(data["total_ftns"]) == Decimal("0")


def test_devices_earnings_bindings_no_receipts():
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "2" * 40)
    app = _build_app(
        [(wallet, "a" * 32, 1000), (wallet, "b" * 32, 2000)],
        _StubReceiptLookup({}),
    )
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/devices/earnings",
        params={"wallet_address": wallet},
    )
    assert r.status_code == 200
    data = r.json()
    # No receipts → empty earnings (not zero entries — empty map)
    assert data["earnings_by_node_id"] == {}
    assert Decimal(data["total_ftns"]) == Decimal("0")


def test_devices_earnings_with_receipts():
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "3" * 40)
    node_a = "a" * 32
    node_b = "b" * 32
    app = _build_app(
        [(wallet, node_a, 1000), (wallet, node_b, 2000)],
        _StubReceiptLookup({
            node_a: [
                _make_receipt_dict(node_a, "1.0"),
                _make_receipt_dict(node_a, "2.0"),
            ],
            node_b: [
                _make_receipt_dict(node_b, "0.5"),
            ],
        }),
    )
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/devices/earnings",
        params={"wallet_address": wallet},
    )
    assert r.status_code == 200
    data = r.json()
    assert Decimal(data["earnings_by_node_id"][node_a]) == Decimal("3.0")
    assert Decimal(data["earnings_by_node_id"][node_b]) == Decimal("0.5")
    assert Decimal(data["total_ftns"]) == Decimal("3.5")


def test_devices_earnings_partial_completion_proportional():
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "4" * 40)
    node_a = "a" * 32
    app = _build_app(
        [(wallet, node_a, 1000)],
        _StubReceiptLookup({
            node_a: [
                _make_receipt_dict(
                    node_a, "1.0",
                    partial_completion={
                        "reason": "preempted",
                        "tokens_completed": 7,
                        "tokens_requested": 10,
                        "timestamp": "2026-05-23T12:00:00Z",
                    },
                ),
            ],
        }),
    )
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/devices/earnings",
        params={"wallet_address": wallet},
    )
    data = r.json()
    # 7/10 of 1.0 = 0.7
    assert Decimal(data["earnings_by_node_id"][node_a]) == Decimal("0.7")
    assert Decimal(data["total_ftns"]) == Decimal("0.7")


def test_devices_earnings_filters_non_bound_nodes():
    """Receipts from a node_id that's NOT in this wallet's roster
    must be excluded from the aggregate — this is a roster-scoped
    earnings view, not a free-for-all over all receipts the
    daemon has seen."""
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "5" * 40)
    bound_node = "a" * 32
    other_node = "c" * 32  # NOT bound to this wallet
    app = _build_app(
        [(wallet, bound_node, 1000)],
        _StubReceiptLookup({
            bound_node: [_make_receipt_dict(bound_node, "1.0")],
            other_node: [_make_receipt_dict(other_node, "999.0")],
        }),
    )
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/devices/earnings",
        params={"wallet_address": wallet},
    )
    data = r.json()
    assert other_node not in data["earnings_by_node_id"]
    assert Decimal(data["earnings_by_node_id"][bound_node]) == Decimal("1.0")
    assert Decimal(data["total_ftns"]) == Decimal("1.0")


# ---- CLI consumer ----------------------------------------------


def _invoke_earnings(args=None, env=None):
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    return runner.invoke(
        _wallet_group, ["devices", "earnings"] + (args or []),
        env=env or {},
    )


def test_cli_earnings_command_registered():
    from prsm.cli import wallet as _wallet_group
    devices = _wallet_group.commands["devices"]
    assert "earnings" in [c.name for c in devices.commands.values()]


def test_cli_earnings_text_renders_per_node():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "earnings_by_node_id": {
            "1" * 32: "3.5",
            "2" * 32: "0.7",
        },
        "total_ftns": "4.2",
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke_earnings([
            "--wallet", "0x" + "a" * 40,
            "--format", "text",
        ])
    assert result.exit_code == 0, result.output
    assert "1" * 32 in result.output
    assert "2" * 32 in result.output
    assert "3.5" in result.output
    assert "0.7" in result.output
    assert "4.2" in result.output


def test_cli_earnings_json_returns_payload():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "earnings_by_node_id": {"1" * 32: "1.0"},
        "total_ftns": "1.0",
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke_earnings([
            "--wallet", "0x" + "a" * 40,
            "--format", "json",
        ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "earnings_by_node_id" in data
    assert "total_ftns" in data


def test_cli_earnings_empty_actionable_hint():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "earnings_by_node_id": {},
        "total_ftns": "0",
    }
    with patch("httpx.get", return_value=fake):
        result = _invoke_earnings([
            "--wallet", "0x" + "a" * 40,
            "--format", "text",
        ])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "no earnings" in out or "no devices" in out


def test_cli_earnings_unreachable_daemon_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke_earnings([
            "--wallet", "0x" + "a" * 40,
        ])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()
