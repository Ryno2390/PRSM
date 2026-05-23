"""Sprint 794 — SqliteWalletBindingStore in production wire-up.

Sprint 793 shipped the production wire-up for WalletApiServices
but used `InMemoryWalletBindingStore` — bindings VANISH on
daemon restart. For real operators with persistent device
rosters, that's a hard regression: a routine systemd restart
means re-running `wallet devices add` for every device.

Sprint 794 swaps in `SqliteWalletBindingStore` for production
wiring + adds:

  _resolve_wallet_bindings_db_path() -> Optional[Path]
    Reads PRSM_WALLET_BINDINGS_DB env (operator-set absolute
    path) OR falls back to ~/.prsm/wallet_bindings.db. Returns
    Path object; caller is responsible for ensuring the parent
    directory is writable.

  wire_wallet_api_services(node) — updated to try Sqlite at
    the resolved path; on ANY failure (permission denied, IO
    error, schema incompatibility) falls back to InMemory +
    logs warning. Daemon does NOT crash on this peripheral
    surface.

The persistence test is the load-bearing pin: bind via service
A, throw away the services slot, re-wire (simulated daemon
restart), and verify the binding is still there.

Pin tests:
- _resolve helper exists.
- Env override wins over default.
- Default falls back to ~/.prsm/wallet_bindings.db when env unset.
- Wire helper uses SqliteWalletBindingStore when env override
  points at a writable tmp path.
- Wire helper falls back to InMemory on Sqlite init failure
  (e.g. unwritable path).
- **Persistence**: bindings survive a "restart" cycle.
- Idempotency: wire twice doesn't double-init schema or raise.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock


# ---- Resolver helper -------------------------------------------


def test_resolve_helper_exists():
    from prsm.node.wallet_api_wiring import (
        _resolve_wallet_bindings_db_path,
    )
    assert callable(_resolve_wallet_bindings_db_path)


def test_resolve_env_override_wins(monkeypatch, tmp_path: Path):
    from prsm.node.wallet_api_wiring import (
        _resolve_wallet_bindings_db_path,
    )
    monkeypatch.setenv(
        "PRSM_WALLET_BINDINGS_DB",
        str(tmp_path / "custom.db"),
    )
    assert _resolve_wallet_bindings_db_path() == (
        tmp_path / "custom.db"
    )


def test_resolve_default_is_under_dot_prsm(monkeypatch):
    from prsm.node.wallet_api_wiring import (
        _resolve_wallet_bindings_db_path,
    )
    monkeypatch.delenv("PRSM_WALLET_BINDINGS_DB", raising=False)
    result = _resolve_wallet_bindings_db_path()
    assert result is not None
    assert result.name == "wallet_bindings.db"
    # Lives under ~/.prsm/
    assert ".prsm" in str(result)


# ---- Wire helper uses Sqlite ----------------------------------


def _build_node():
    from prsm.node.receipt_store import ReceiptStore
    node = MagicMock()
    node._receipt_store = ReceiptStore(persist_dir=None)
    return node


def test_wire_uses_sqlite_when_path_valid(monkeypatch, tmp_path: Path):
    """When PRSM_WALLET_BINDINGS_DB points at a writable path,
    the wired services' binding_service is backed by Sqlite."""
    from prsm.interface.api import wallet_api
    from prsm.interface.onboarding.wallet_binding import (
        SqliteWalletBindingStore,
    )
    from prsm.node.wallet_api_wiring import wire_wallet_api_services

    db_path = tmp_path / "wallet_bindings.db"
    monkeypatch.setenv("PRSM_WALLET_BINDINGS_DB", str(db_path))

    wallet_api.reset_services_for_tests()
    node = _build_node()
    wire_wallet_api_services(node)

    services = wallet_api.get_services()
    # The binding_service wraps a store; verify it's Sqlite.
    store = services.binding_service._store
    assert isinstance(store, SqliteWalletBindingStore)
    # File created
    assert db_path.exists()

    wallet_api.reset_services_for_tests()


def test_wire_falls_back_to_inmemory_on_sqlite_failure(
    monkeypatch, tmp_path: Path,
):
    """Unwritable path → fail-soft fallback to InMemory."""
    from prsm.interface.api import wallet_api
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
    )
    from prsm.node.wallet_api_wiring import wire_wallet_api_services

    # Point env at an unwritable path. On most systems, a path
    # under /proc is unwritable for normal users.
    bad_path = tmp_path / "nonexistent_subdir_that_we_will_make_a_file" / "wallet_bindings.db"
    # Create a FILE where the parent directory should be — makes
    # mkdir(parents=True) fail.
    (tmp_path / "nonexistent_subdir_that_we_will_make_a_file").write_text("not a dir")
    monkeypatch.setenv("PRSM_WALLET_BINDINGS_DB", str(bad_path))

    wallet_api.reset_services_for_tests()
    node = _build_node()
    wire_wallet_api_services(node)

    services = wallet_api.get_services()
    store = services.binding_service._store
    assert isinstance(store, InMemoryWalletBindingStore)

    wallet_api.reset_services_for_tests()


# ---- Load-bearing persistence test -----------------------------


def test_bindings_persist_across_restart(monkeypatch, tmp_path: Path):
    """The whole sprint reason-for-existing. Bind via service A,
    drop the services slot (simulated daemon restart), re-wire
    via service B, and verify the binding is still there."""
    from datetime import datetime, timezone
    from eth_account import Account
    from eth_account.messages import encode_defunct
    from prsm.interface.api import wallet_api
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    from prsm.node.wallet_api_wiring import wire_wallet_api_services

    db_path = tmp_path / "wallet_bindings.db"
    monkeypatch.setenv("PRSM_WALLET_BINDINGS_DB", str(db_path))

    # ----- "Boot 1" --------------------------------------------
    wallet_api.reset_services_for_tests()
    node = _build_node()
    wire_wallet_api_services(node)
    svc1 = wallet_api.get_services().binding_service

    acct = Account.create()
    node_id = "a" * 32
    issued_at_iso = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    msg = build_binding_message(acct.address, node_id, issued_at_iso)
    sig = acct.sign_message(
        encode_defunct(text=msg),
    ).signature.to_0x_hex()
    binding = svc1.bind(
        wallet_address=acct.address,
        node_id_hex=node_id,
        signature=sig,
        issued_at_iso=issued_at_iso,
    )
    assert binding.node_id_hex == node_id

    # ----- "Boot 2" — simulated daemon restart ----------------
    # Drop the services slot. Re-wire fresh from the same DB path.
    wallet_api.reset_services_for_tests()
    node2 = _build_node()
    wire_wallet_api_services(node2)
    svc2 = wallet_api.get_services().binding_service

    # The binding from Boot 1 should still be retrievable.
    bindings = svc2.get_all_by_wallet(acct.address)
    assert len(bindings) == 1
    assert bindings[0].node_id_hex == node_id

    wallet_api.reset_services_for_tests()


def test_wire_idempotent_with_sqlite(monkeypatch, tmp_path: Path):
    """Double-wire to the same Sqlite path doesn't raise (CREATE
    TABLE IF NOT EXISTS handles re-init)."""
    from prsm.interface.api import wallet_api
    from prsm.node.wallet_api_wiring import wire_wallet_api_services

    db_path = tmp_path / "wallet_bindings.db"
    monkeypatch.setenv("PRSM_WALLET_BINDINGS_DB", str(db_path))

    wallet_api.reset_services_for_tests()
    node = _build_node()

    wire_wallet_api_services(node)
    wire_wallet_api_services(node)  # second call must succeed

    assert wallet_api.get_services() is not None
    wallet_api.reset_services_for_tests()
