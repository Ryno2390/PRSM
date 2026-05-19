"""Sprint 583 — `prsm node stake-bond-probe` CLI.

Mirror of sprint-581's anchor-probe for the sibling §7 production
env var ``PRSM_STAKE_BOND_ADDRESS`` (sprint 561). Same diagnostic
pattern: probe StakeManagerClient construction in-process + report
outcome before operator flips PRSM_PARALLAX_TRUST_STACK_KIND=production.
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


def _run(args, env=None):
    from prsm.cli import node
    runner = CliRunner()
    if env:
        with patch.dict(os.environ, env, clear=False):
            return runner.invoke(node, args)
    return runner.invoke(node, args)


def test_stake_bond_probe_reports_unset():
    saved = os.environ.pop("PRSM_STAKE_BOND_ADDRESS", None)
    try:
        result = _run(["stake-bond-probe"])
    finally:
        if saved is not None:
            os.environ["PRSM_STAKE_BOND_ADDRESS"] = saved
    assert result.exit_code != 0
    assert "unset" in result.output.lower() or "<unset>" in result.output


def test_stake_bond_probe_construction_failure(caplog):
    fake_module = MagicMock()
    fake_module.StakeManagerClient = MagicMock(
        side_effect=RuntimeError("rpc unreachable"),
    )
    with patch.dict(
        os.environ,
        {"PRSM_STAKE_BOND_ADDRESS": "0xdead"},
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.economy.web3.stake_manager": fake_module},
        ):
            result = _run(["stake-bond-probe"])
    assert result.exit_code != 0
    out = result.output.lower()
    assert "rpc unreachable" in out or "failed" in out


def test_stake_bond_probe_success():
    fake_module = MagicMock()
    fake_client = MagicMock()
    fake_module.StakeManagerClient = MagicMock(return_value=fake_client)
    with patch.dict(
        os.environ,
        {"PRSM_STAKE_BOND_ADDRESS": "0xdead"},
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.economy.web3.stake_manager": fake_module},
        ):
            result = _run(["stake-bond-probe"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output.lower() or "✓" in result.output


def test_stake_bond_probe_json_format():
    import json
    saved = os.environ.pop("PRSM_STAKE_BOND_ADDRESS", None)
    try:
        result = _run(["stake-bond-probe", "--format", "json"])
    finally:
        if saved is not None:
            os.environ["PRSM_STAKE_BOND_ADDRESS"] = saved
    data = json.loads(result.output)
    assert "PRSM_STAKE_BOND_ADDRESS" in data
    assert "outcome" in data
    assert data["outcome"] in ("ok", "unset", "construction_failed")
