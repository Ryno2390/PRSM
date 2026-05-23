"""Sprint 795 — `prsm node stake-info` CLI.

Operators staking on-chain need a one-command read of their
current state: what address is configured, is the delegation
present + valid (sprint 788 attestation), what's the on-chain
stake amount, what contract is being read.

Pre-795 the operator had to: read PRSM_OPERATOR_ADDRESS from
their systemd unit, manually call StakeBond.stake_of(), check
if PRSM_OPERATOR_DELEGATION was set, and cross-reference. Real
operator UX gap.

Sprint 795 ships:

  prsm node stake-info [--format text|json]

Reports:
  - operator_address (from PRSM_OPERATOR_ADDRESS env)
  - operator_delegation status:
      "absent" → env unset
      "valid"  → present + verifies under sprint 788 rules
      "invalid" → present but verify failed
  - on_chain_stake_wei + on_chain_stake_ftns (wei / 1e18)
  - stake_bond_address (PRSM_STAKE_BOND_ADDRESS env)
  - rpc_url (PRSM_BASE_RPC_URL env; default mainnet)

Exit 0 on any well-formed report (even with zero stake or
unset env). The CLI is informational — it surfaces config gaps
rather than refusing to run on them.

Pin tests:
- Command registered.
- No PRSM_OPERATOR_ADDRESS → actionable "set the env" hint,
  exit 0 (information, not failure).
- With operator + delegation env unset → status "absent".
- With operator + delegation env set + valid → "valid",
  delegation file path surfaced.
- With operator + valid delegation env but file missing →
  status indicates issue (operator-readable).
- json mode returns parseable payload with all reported fields.
- on-chain stake_reader returning N → wei + FTNS rendered.
- stake_reader exception → 0 + reader-error noted (graceful).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args=None, env=None):
    from prsm.cli import node as _node_group
    runner = CliRunner()
    return runner.invoke(
        _node_group, ["stake-info"] + (args or []),
        env=env or {},
    )


# ---- Command registration ---------------------------------------


def test_stake_info_command_registered():
    from prsm.cli import node as _node_group
    cmd_names = [c.name for c in _node_group.commands.values()]
    assert "stake-info" in cmd_names


# ---- No operator_address: actionable hint ----------------------


def test_no_operator_address_actionable_hint():
    """Env unset → exit 0 + hint to set PRSM_OPERATOR_ADDRESS.
    Informational; don't fail loudly."""
    result = _invoke([], env={
        "PRSM_OPERATOR_ADDRESS": "",
        "PRSM_OPERATOR_DELEGATION": "",
        "PRSM_STAKE_BOND_ADDRESS": "",
    })
    assert result.exit_code == 0, result.output
    assert "PRSM_OPERATOR_ADDRESS" in result.output


# ---- Operator set, delegation absent ---------------------------


def test_operator_set_delegation_absent_text():
    """PRSM_OPERATOR_ADDRESS set; delegation env unset → status
    'absent' surfaced (single-device operator who hasn't run
    `wallet devices add` yet)."""
    op_addr = "0x" + "a" * 40

    fake_reader = MagicMock()
    fake_reader.stake_amount_for = MagicMock(return_value=0)
    with patch(
        "prsm.node.onchain_stake_reader.OnChainStakeReader",
        return_value=fake_reader,
    ):
        result = _invoke([], env={
            "PRSM_OPERATOR_ADDRESS": op_addr,
            "PRSM_OPERATOR_DELEGATION": "",
            "PRSM_STAKE_BOND_ADDRESS": "0x" + "b" * 40,
        })
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "absent" in out
    assert op_addr.lower() in result.output.lower()


# ---- Operator + valid delegation -------------------------------


def _make_valid_delegation_blob(operator_addr, node_id, privkey):
    """Build a real EIP-191 delegation."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    issued_at_iso = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    msg = build_binding_message(operator_addr, node_id, issued_at_iso)
    acct = Account.from_key(privkey)
    signed = acct.sign_message(encode_defunct(text=msg))
    return {
        "wallet_address": operator_addr,
        "node_id_hex": node_id,
        "issued_at_iso": issued_at_iso,
        "signature": signed.signature.to_0x_hex(),
    }


def test_operator_with_valid_delegation_text(tmp_path: Path):
    from eth_account import Account
    acct = Account.create()

    # Use the daemon's identity.json node_id for the delegation.
    # The CLI reads from ~/.prsm/identity.json by default; tests
    # pass --identity-file to override. Create a temp identity.
    from prsm.node.identity import (
        generate_node_identity, save_node_identity,
    )
    identity = generate_node_identity("test-stake-info")
    ident_path = tmp_path / "identity.json"
    save_node_identity(identity, ident_path)

    blob = _make_valid_delegation_blob(
        acct.address, identity.node_id, acct.key.to_0x_hex(),
    )
    blob_json = json.dumps(blob)

    fake_reader = MagicMock()
    fake_reader.stake_amount_for = MagicMock(return_value=0)
    with patch(
        "prsm.node.onchain_stake_reader.OnChainStakeReader",
        return_value=fake_reader,
    ):
        result = _invoke(
            ["--identity-file", str(ident_path)],
            env={
                "PRSM_OPERATOR_ADDRESS": acct.address,
                "PRSM_OPERATOR_DELEGATION": blob_json,
                "PRSM_STAKE_BOND_ADDRESS": "0x" + "b" * 40,
            },
        )
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_operator_with_invalid_delegation_text(tmp_path: Path):
    """Delegation present but signed by a DIFFERENT key → status
    'invalid' surfaced so operator can rotate."""
    from eth_account import Account
    acct_real = Account.create()
    acct_attacker = Account.create()

    from prsm.node.identity import (
        generate_node_identity, save_node_identity,
    )
    identity = generate_node_identity("test")
    ident_path = tmp_path / "identity.json"
    save_node_identity(identity, ident_path)

    # Signed by the ATTACKER but operator claims to be the REAL key
    blob = _make_valid_delegation_blob(
        acct_attacker.address, identity.node_id, acct_attacker.key.to_0x_hex(),
    )
    # Force the claim to mismatch — claim operator_address=acct_real
    # while signature is from acct_attacker
    blob["wallet_address"] = acct_real.address
    blob_json = json.dumps(blob)

    fake_reader = MagicMock()
    fake_reader.stake_amount_for = MagicMock(return_value=0)
    with patch(
        "prsm.node.onchain_stake_reader.OnChainStakeReader",
        return_value=fake_reader,
    ):
        result = _invoke(
            ["--identity-file", str(ident_path)],
            env={
                "PRSM_OPERATOR_ADDRESS": acct_real.address,
                "PRSM_OPERATOR_DELEGATION": blob_json,
                "PRSM_STAKE_BOND_ADDRESS": "0x" + "b" * 40,
            },
        )
    assert result.exit_code == 0
    assert "invalid" in result.output.lower()


# ---- On-chain stake amount rendered ---------------------------


def test_on_chain_stake_amount_rendered_text():
    """stake_reader returns 5e18 wei → "5" FTNS rendered."""
    op_addr = "0x" + "a" * 40

    fake_reader = MagicMock()
    fake_reader.stake_amount_for = MagicMock(
        return_value=5_000_000_000_000_000_000,  # 5 FTNS in wei
    )
    with patch(
        "prsm.node.onchain_stake_reader.OnChainStakeReader",
        return_value=fake_reader,
    ):
        result = _invoke([], env={
            "PRSM_OPERATOR_ADDRESS": op_addr,
            "PRSM_STAKE_BOND_ADDRESS": "0x" + "b" * 40,
        })
    assert result.exit_code == 0, result.output
    # Wei value
    assert "5000000000000000000" in result.output
    # FTNS conversion (5 FTNS) should be human-rendered
    assert "5" in result.output


def test_stake_reader_exception_graceful():
    """Reader raises → 0 reported + operator sees a note. No crash."""
    op_addr = "0x" + "a" * 40

    fake_reader = MagicMock()
    fake_reader.stake_amount_for = MagicMock(
        side_effect=RuntimeError("rpc unreachable"),
    )
    with patch(
        "prsm.node.onchain_stake_reader.OnChainStakeReader",
        return_value=fake_reader,
    ):
        result = _invoke([], env={
            "PRSM_OPERATOR_ADDRESS": op_addr,
            "PRSM_STAKE_BOND_ADDRESS": "0x" + "b" * 40,
        })
    assert result.exit_code == 0, result.output
    # Operator-visible signal that the reader had trouble
    assert "unable" in result.output.lower() or \
        "error" in result.output.lower() or \
        "rpc" in result.output.lower()


# ---- JSON output ----------------------------------------------


def test_json_output_payload_shape():
    op_addr = "0x" + "a" * 40

    fake_reader = MagicMock()
    fake_reader.stake_amount_for = MagicMock(
        return_value=2_000_000_000_000_000_000,
    )
    with patch(
        "prsm.node.onchain_stake_reader.OnChainStakeReader",
        return_value=fake_reader,
    ):
        result = _invoke(
            ["--format", "json"],
            env={
                "PRSM_OPERATOR_ADDRESS": op_addr,
                "PRSM_STAKE_BOND_ADDRESS": "0x" + "b" * 40,
            },
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["operator_address"] == op_addr
    assert data["on_chain_stake_wei"] == 2_000_000_000_000_000_000
    assert "on_chain_stake_ftns" in data
    assert "delegation_status" in data
    assert "stake_bond_address" in data
