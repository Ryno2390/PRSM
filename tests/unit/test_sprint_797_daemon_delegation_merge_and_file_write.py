"""Sprint 797 — daemon merges PRSM_OPERATOR_DELEGATION into its
own hardware_profile + CLI gains --write-delegation-file.

Two halves of one operator-flow gap:

  PRODUCER (daemon side): pre-797 the daemon never actually
    merged PRSM_OPERATOR_DELEGATION into its own
    hardware_profile. Sprint 788 wired the VERIFIER (peers
    check incoming delegations) but the daemon's own profile
    didn't carry one — so the peer-verify side had nothing to
    check. New `_merge_operator_delegation` parallels sprint
    690's `_merge_operator_address`, reading
    `PRSM_OPERATOR_DELEGATION` (raw JSON in env) OR
    `PRSM_OPERATOR_DELEGATION_FILE` (path to JSON) and
    merging the blob into `data["operator_delegation"]`.

  CONSUMER (CLI side): `wallet devices add --write` writes
    the minted delegation to `~/.prsm/operator_delegation.json`
    by default (or to `--write` path if explicit). chmod 600
    since it's a signing artifact. Closes the copy-paste
    friction.

End-to-end operator flow post-797:
  1. `prsm wallet devices add --node-id <hex> --write --register`
     → file at ~/.prsm/operator_delegation.json + binding
     recorded in daemon
  2. Restart daemon (or `prsm node start` for first boot)
     → reads PRSM_OPERATOR_DELEGATION_FILE (default
     ~/.prsm/operator_delegation.json), merges into
     hardware_profile, announces to network
  3. Peers verify via sprint 788's pool-provider gate

Pin tests (producer):
- _merge_operator_delegation function exists.
- PRSM_OPERATOR_DELEGATION env (JSON string) → merged into data.
- PRSM_OPERATOR_DELEGATION_FILE env (path) → read + merged.
- Env wins over file when both set.
- Default path ~/.prsm/operator_delegation.json picked up if
  no env vars set + file exists.
- Malformed JSON in env → skip + warn (no crash).
- Missing file path → skip (no crash).
- load_local_hardware_profile calls _merge_operator_delegation.

Pin tests (consumer):
- --write flag on `wallet devices add` exists.
- --write default → writes ~/.prsm/operator_delegation.json.
- --write <path> → writes that path.
- Written file contains valid delegation JSON.
- chmod 600 on written file (security hygiene).
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner


# ---- Helpers ----------------------------------------------------


def _make_blob():
    """Build a real EIP-191 delegation blob using a fresh key."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    acct = Account.create()
    node_id = "a" * 32
    issued_at_iso = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    msg = build_binding_message(acct.address, node_id, issued_at_iso)
    signed = acct.sign_message(encode_defunct(text=msg))
    return {
        "wallet_address": acct.address,
        "node_id_hex": node_id,
        "issued_at_iso": issued_at_iso,
        "signature": signed.signature.to_0x_hex(),
    }


# ============================================================
# PRODUCER side — daemon hardware_profile_loader merge
# ============================================================


def test_merge_function_exists():
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    assert callable(_merge_operator_delegation)


def test_env_json_merged_into_data(monkeypatch):
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    blob = _make_blob()
    monkeypatch.setenv(
        "PRSM_OPERATOR_DELEGATION", json.dumps(blob),
    )
    monkeypatch.delenv(
        "PRSM_OPERATOR_DELEGATION_FILE", raising=False,
    )
    data: dict = {}
    _merge_operator_delegation(data)
    assert "operator_delegation" in data
    assert data["operator_delegation"]["signature"] == blob["signature"]


def test_file_env_merged_into_data(
    monkeypatch, tmp_path: Path,
):
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    blob = _make_blob()
    path = tmp_path / "delegation.json"
    path.write_text(json.dumps(blob))

    monkeypatch.delenv("PRSM_OPERATOR_DELEGATION", raising=False)
    monkeypatch.setenv(
        "PRSM_OPERATOR_DELEGATION_FILE", str(path),
    )
    # Suppress default path lookup
    monkeypatch.setattr(
        "pathlib.Path.home", lambda: tmp_path / "no-home",
    )
    data: dict = {}
    _merge_operator_delegation(data)
    assert "operator_delegation" in data
    assert data["operator_delegation"]["signature"] == blob["signature"]


def test_env_wins_over_file(monkeypatch, tmp_path: Path):
    """Both env vars set → env wins (it's the explicit override)."""
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    env_blob = _make_blob()
    file_blob = _make_blob()
    path = tmp_path / "delegation.json"
    path.write_text(json.dumps(file_blob))

    monkeypatch.setenv(
        "PRSM_OPERATOR_DELEGATION", json.dumps(env_blob),
    )
    monkeypatch.setenv(
        "PRSM_OPERATOR_DELEGATION_FILE", str(path),
    )
    data: dict = {}
    _merge_operator_delegation(data)
    assert (
        data["operator_delegation"]["signature"]
        == env_blob["signature"]
    )


def test_default_path_picked_up_when_no_env(
    monkeypatch, tmp_path: Path,
):
    """No env vars but ~/.prsm/operator_delegation.json exists
    → merged. This is the smooth post-797 operator path."""
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    blob = _make_blob()
    # Point Path.home() at our tmp dir + create the default file
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    prsm_dir = tmp_path / ".prsm"
    prsm_dir.mkdir()
    (prsm_dir / "operator_delegation.json").write_text(
        json.dumps(blob),
    )
    monkeypatch.delenv("PRSM_OPERATOR_DELEGATION", raising=False)
    monkeypatch.delenv(
        "PRSM_OPERATOR_DELEGATION_FILE", raising=False,
    )

    data: dict = {}
    _merge_operator_delegation(data)
    assert "operator_delegation" in data
    assert data["operator_delegation"]["signature"] == blob["signature"]


def test_malformed_env_json_skipped(monkeypatch):
    """Garbage env value → skip without crashing."""
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    monkeypatch.setenv("PRSM_OPERATOR_DELEGATION", "{ not json")
    monkeypatch.delenv(
        "PRSM_OPERATOR_DELEGATION_FILE", raising=False,
    )
    data: dict = {}
    _merge_operator_delegation(data)  # must not raise
    assert "operator_delegation" not in data


def test_missing_file_path_skipped(monkeypatch, tmp_path: Path):
    from prsm.node.hardware_profile_loader import (
        _merge_operator_delegation,
    )
    monkeypatch.delenv("PRSM_OPERATOR_DELEGATION", raising=False)
    monkeypatch.setenv(
        "PRSM_OPERATOR_DELEGATION_FILE",
        str(tmp_path / "nonexistent.json"),
    )
    monkeypatch.setattr(
        "pathlib.Path.home", lambda: tmp_path / "no-home",
    )
    data: dict = {}
    _merge_operator_delegation(data)
    assert "operator_delegation" not in data


def test_load_local_hardware_profile_calls_merge(monkeypatch, tmp_path):
    """Source-shape: load_local_hardware_profile invokes the new
    merge helper alongside the existing operator_address one."""
    import inspect
    from prsm.node import hardware_profile_loader as _hpl

    src = inspect.getsource(_hpl)
    # Both helpers are referenced in the module
    assert "_merge_operator_delegation" in src
    # Count occurrences of the call vs the def: there must be at
    # least one call site in addition to the def.
    call_count = src.count("_merge_operator_delegation(")
    # 1 def + ≥1 call
    assert call_count >= 2


# ============================================================
# CONSUMER side — `wallet devices add --write`
# ============================================================


def _invoke_add(args, env=None):
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    return runner.invoke(
        _wallet_group, ["devices", "add"] + list(args),
        env=env or {},
    )


def test_write_flag_exists():
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    devices = _wallet_group.commands["devices"]
    add_cmd = devices.commands["add"]
    result = runner.invoke(add_cmd, ["--help"])
    assert result.exit_code == 0
    assert "--write" in result.output


def test_write_explicit_path(tmp_path: Path):
    from eth_account import Account
    acct = Account.create()
    target = tmp_path / "delegation.json"

    result = _invoke_add(
        [
            "--node-id", "a" * 32,
            "--write-path", str(target),
            "--format", "json",
        ],
        env={"PRIVATE_KEY": acct.key.to_0x_hex()},
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    blob = json.loads(target.read_text())
    assert blob["wallet_address"] == acct.address
    assert blob["node_id_hex"] == "a" * 32
    assert blob["signature"].startswith("0x")


def test_write_chmods_600(tmp_path: Path):
    """File contains a signing artifact — chmod 600 (defense)."""
    from eth_account import Account
    acct = Account.create()
    target = tmp_path / "delegation.json"

    result = _invoke_add(
        [
            "--node-id", "b" * 32,
            "--write-path", str(target),
        ],
        env={"PRIVATE_KEY": acct.key.to_0x_hex()},
    )
    assert result.exit_code == 0
    mode = target.stat().st_mode & 0o777
    assert mode == 0o600, (
        f"expected chmod 600 on delegation file, got {oct(mode)}"
    )


def test_write_default_path_resolves_to_dot_prsm(
    tmp_path: Path, monkeypatch,
):
    """--write with no value uses the default
    ~/.prsm/operator_delegation.json path. We monkey-patch
    Path.home() to redirect to tmp_path so the test doesn't
    pollute the real user dir."""
    from eth_account import Account
    acct = Account.create()

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    result = _invoke_add(
        [
            "--node-id", "c" * 32,
            "--write",  # bare flag → default path
        ],
        env={"PRIVATE_KEY": acct.key.to_0x_hex()},
    )
    assert result.exit_code == 0, result.output
    default_path = tmp_path / ".prsm" / "operator_delegation.json"
    assert default_path.exists()
