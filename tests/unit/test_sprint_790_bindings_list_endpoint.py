"""Sprint 790 — daemon binding-list endpoint + `prsm wallet devices list`.

Closes the round-trip for the multi-device arc. Sprint 786
shipped the 1:N data model; sprint 788 the verifier; sprint 789
the operator's `add/verify` CLI. Sprint 790 lets an operator
QUERY their roster — "what devices are bound to my wallet?".

Two pieces:

  HTTP: GET /api/v1/auth/wallet/bindings?wallet_address=0x...
    Returns a JSON list of WalletBindResponse (one per bound
    device). Empty list when wallet has no bindings.

  CLI:  prsm wallet devices list --wallet <addr>
    Hits the endpoint above; renders one row per binding
    (text mode) or full JSON (json mode). Exit 0 on success,
    2 on daemon-unreachable.

Pin tests:
- HTTP: empty wallet → []
- HTTP: wallet with two bindings → 2 items, oldest-first
- HTTP: bindings have correct shape (4 fields per item)
- CLI: list command registered
- CLI: happy text path renders both bound node_ids
- CLI: happy json path returns parseable list
- CLI: unreachable daemon → exit 2
- CLI: empty list → "no devices bound" actionable hint
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from fastapi.testclient import TestClient


def _build_app_with_seeded_bindings(bindings_to_seed):
    """Build a FastAPI app + register wallet_api router + seed
    the in-memory binding store with the given (wallet, node_id,
    bound_at_unix) tuples."""
    from fastapi import FastAPI
    from prsm.interface.api import wallet_api
    from prsm.interface.onboarding.wallet_binding import (
        IdentityBinding,
        InMemoryWalletBindingStore,
        WalletBindingService,
    )

    wallet_api.reset_services_for_tests()
    store = InMemoryWalletBindingStore()
    for wallet, node_id, bound_at in bindings_to_seed:
        store.insert(IdentityBinding(
            wallet_address=wallet,
            node_id_hex=node_id,
            bound_at_unix=bound_at,
            wallet_signature="0xsig",
            signing_message_hash="0xhash",
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
    )
    wallet_api.set_services(services)

    app = FastAPI()
    app.include_router(wallet_api.router)
    return app


# ---- HTTP endpoint ---------------------------------------------


def test_bindings_endpoint_empty_wallet_returns_empty_list():
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "1" * 40)
    app = _build_app_with_seeded_bindings([])
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/bindings",
        params={"wallet_address": wallet},
    )
    assert r.status_code == 200
    assert r.json() == []


def test_bindings_endpoint_returns_all_for_wallet_oldest_first():
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "2" * 40)
    # Two devices bound to same wallet, inserted out of order
    app = _build_app_with_seeded_bindings([
        (wallet, "b" * 32, 2000),
        (wallet, "a" * 32, 1000),
    ])
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/bindings",
        params={"wallet_address": wallet},
    )
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 2
    # Oldest-first ordering
    assert items[0]["node_id_hex"] == "a" * 32
    assert items[1]["node_id_hex"] == "b" * 32


def test_bindings_endpoint_response_shape():
    """Each item has wallet_address + node_id_hex + bound_at_unix
    + signing_message_hash (matches WalletBindResponse shape)."""
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "3" * 40)
    app = _build_app_with_seeded_bindings([
        (wallet, "a" * 32, 1000),
    ])
    client = TestClient(app)
    r = client.get(
        "/api/v1/auth/wallet/bindings",
        params={"wallet_address": wallet},
    )
    item = r.json()[0]
    assert item["wallet_address"] == wallet
    assert item["node_id_hex"] == "a" * 32
    assert item["bound_at_unix"] == 1000
    assert "signing_message_hash" in item


# ---- CLI consumer ---------------------------------------------


def _invoke_devices_list(args=None, env=None):
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    return runner.invoke(
        _wallet_group, ["devices", "list"] + (args or []),
        env=env or {},
    )


def test_cli_list_command_registered():
    from prsm.cli import wallet as _wallet_group
    devices = _wallet_group.commands["devices"]
    assert "list" in [c.name for c in devices.commands.values()]


def test_cli_list_renders_bound_node_ids_text():
    """Happy path: daemon returns 2 bindings → both node_ids in output."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = [
        {
            "wallet_address": "0x" + "a" * 40,
            "node_id_hex": "1" * 32,
            "bound_at_unix": 1000,
            "signing_message_hash": "0xh1",
        },
        {
            "wallet_address": "0x" + "a" * 40,
            "node_id_hex": "2" * 32,
            "bound_at_unix": 2000,
            "signing_message_hash": "0xh2",
        },
    ]
    with patch("httpx.get", return_value=fake):
        result = _invoke_devices_list([
            "--wallet", "0x" + "a" * 40,
            "--format", "text",
        ])
    assert result.exit_code == 0, result.output
    assert "1" * 32 in result.output
    assert "2" * 32 in result.output


def test_cli_list_returns_json():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = [
        {
            "wallet_address": "0x" + "a" * 40,
            "node_id_hex": "1" * 32,
            "bound_at_unix": 1000,
            "signing_message_hash": "0xh",
        },
    ]
    with patch("httpx.get", return_value=fake):
        result = _invoke_devices_list([
            "--wallet", "0x" + "a" * 40,
            "--format", "json",
        ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["node_id_hex"] == "1" * 32


def test_cli_list_empty_actionable_hint():
    """0 bindings → operator-facing 'no devices' hint, not silent."""
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = []
    with patch("httpx.get", return_value=fake):
        result = _invoke_devices_list([
            "--wallet", "0x" + "a" * 40,
            "--format", "text",
        ])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "no devices" in out or "no bindings" in out


def test_cli_list_unreachable_daemon_exits_2():
    with patch(
        "httpx.get",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke_devices_list([
            "--wallet", "0x" + "a" * 40,
            "--format", "text",
        ])
    assert result.exit_code == 2
    assert "unreachable" in result.output.lower()
