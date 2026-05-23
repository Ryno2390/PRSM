"""Sprint 789 — `prsm wallet devices add/verify` CLI.

Closes the operator-UX gap for the multi-device arc. Sprint 788
defines the on-the-wire `operator_delegation` blob shape; sprint
789 ships the operator's tooling to MINT one (using their ETH
private key) and VERIFY one (without a key — anyone can verify).

Two commands shipped:

  prsm wallet devices add --node-id <hex> [--format text|json]
      Reads PRIVATE_KEY env, builds the canonical sprint-786
      binding message for (wallet_address derived from key,
      node_id, now-ISO), signs it with EIP-191, and outputs the
      delegation JSON. Operator copies the JSON into the new
      device's PRSM_OPERATOR_DELEGATION env / file.

  prsm wallet devices verify --node-id <hex> --operator <addr>
      [--delegation-file <path>] [--format text|json]
      Reads a delegation JSON (from --delegation-file or stdin),
      runs sprint-788 verify_operator_delegation_blob, prints
      PASS/FAIL + reason. Exit code 0 on PASS, 1 on FAIL.

`list` is intentionally out of scope until the daemon ships a
binding-list HTTP endpoint (sprint 790+).

Pin tests:
- Subgroup + 2 commands registered.
- add: PRIVATE_KEY unset → error exit (no silent default).
- add: with PRIVATE_KEY, emits a JSON-parseable delegation blob
  with all 4 required fields.
- add: emitted delegation passes sprint-788 verify against the
  same key's address + the same node_id.
- verify: valid delegation file → exit 0 + "PASS" in output.
- verify: invalid delegation (wrong signer) → exit 1 + "FAIL".
- verify: malformed JSON → exit 1.
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


def _invoke(args, env=None):
    """Invoke prsm wallet devices ... with optional env."""
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    return runner.invoke(
        _wallet_group, ["devices"] + list(args), env=env or {},
    )


# ---- Subgroup + command registration ---------------------------


def test_devices_subgroup_registered():
    """`prsm wallet devices` exists."""
    from prsm.cli import wallet as _wallet_group
    cmd_names = [c.name for c in _wallet_group.commands.values()]
    assert "devices" in cmd_names


def test_devices_add_command_registered():
    from prsm.cli import wallet as _wallet_group
    devices = _wallet_group.commands["devices"]
    assert "add" in [c.name for c in devices.commands.values()]


def test_devices_verify_command_registered():
    from prsm.cli import wallet as _wallet_group
    devices = _wallet_group.commands["devices"]
    assert "verify" in [c.name for c in devices.commands.values()]


# ---- add: needs PRIVATE_KEY -------------------------------------


def test_add_without_private_key_errors():
    """No PRIVATE_KEY → exit non-zero with actionable message.
    Don't silently fall back to anything."""
    result = _invoke(
        ["add", "--node-id", "a" * 32],
        env={"PRIVATE_KEY": ""},
    )
    assert result.exit_code != 0
    assert "PRIVATE_KEY" in result.output


# ---- add: round-trip with sprint-788 verify ---------------------


def test_add_emits_verifiable_delegation():
    """add outputs a JSON delegation that sprint-788 verify
    accepts for the same node_id + the key's derived address."""
    from eth_account import Account
    acct = Account.create()
    pk = acct.key.to_0x_hex()
    node_id = "a" * 32

    result = _invoke(
        ["add", "--node-id", node_id, "--format", "json"],
        env={"PRIVATE_KEY": pk},
    )
    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)

    # Shape check
    assert blob["wallet_address"] == acct.address
    assert blob["node_id_hex"] == node_id
    assert "issued_at_iso" in blob
    assert blob["signature"].startswith("0x")

    # Round-trip with sprint-788 verifier
    from prsm.node.operator_delegation import (
        verify_operator_delegation_blob,
    )
    assert verify_operator_delegation_blob(
        node_id=node_id,
        operator_address=acct.address,
        delegation=blob,
    ) is True


def test_add_text_mode_human_readable():
    """text mode shows the JSON + an explanatory hint about
    where to drop it."""
    from eth_account import Account
    acct = Account.create()
    node_id = "b" * 32

    result = _invoke(
        ["add", "--node-id", node_id, "--format", "text"],
        env={"PRIVATE_KEY": acct.key.to_0x_hex()},
    )
    assert result.exit_code == 0
    # node_id appears in the output
    assert node_id in result.output
    # Operator gets a hint about deploying to the new device
    out = result.output.lower()
    assert (
        "operator_delegation" in out
        or "deploy" in out
        or "new device" in out
    )


# ---- verify: pass + fail paths ----------------------------------


def test_verify_passes_on_honest_delegation(tmp_path: Path):
    from eth_account import Account
    from datetime import datetime, timezone
    from eth_account.messages import encode_defunct
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    acct = Account.create()
    node_id = "c" * 32
    issued_at_iso = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    msg = build_binding_message(acct.address, node_id, issued_at_iso)
    signed = acct.sign_message(encode_defunct(text=msg))
    blob = {
        "wallet_address": acct.address,
        "node_id_hex": node_id,
        "issued_at_iso": issued_at_iso,
        "signature": signed.signature.to_0x_hex(),
    }
    path = tmp_path / "deleg.json"
    path.write_text(json.dumps(blob))

    result = _invoke([
        "verify",
        "--node-id", node_id,
        "--operator", acct.address,
        "--delegation-file", str(path),
    ])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_verify_fails_on_wrong_signer(tmp_path: Path):
    from eth_account import Account
    from datetime import datetime, timezone
    from eth_account.messages import encode_defunct
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    acct_a = Account.create()
    acct_b = Account.create()
    node_id = "d" * 32
    # A signs; we'll verify against B → FAIL
    issued_at_iso = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    msg = build_binding_message(acct_a.address, node_id, issued_at_iso)
    signed = acct_a.sign_message(encode_defunct(text=msg))
    blob = {
        "wallet_address": acct_a.address,
        "node_id_hex": node_id,
        "issued_at_iso": issued_at_iso,
        "signature": signed.signature.to_0x_hex(),
    }
    path = tmp_path / "deleg.json"
    path.write_text(json.dumps(blob))

    result = _invoke([
        "verify",
        "--node-id", node_id,
        "--operator", acct_b.address,
        "--delegation-file", str(path),
    ])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_verify_fails_on_malformed_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{ not valid json")
    result = _invoke([
        "verify",
        "--node-id", "e" * 32,
        "--operator", "0x" + "1" * 40,
        "--delegation-file", str(path),
    ])
    assert result.exit_code == 1
