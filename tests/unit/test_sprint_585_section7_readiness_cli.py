"""Sprint 585 — `prsm node section7-readiness` aggregate CLI.

Sprints 581/583/584 shipped per-component preflights. Sprint 585
aggregates them into a single operator-facing command that
reports overall §7 production-readiness.

Output schema:
  - per-component: anchor / stake_bond / rpc with their outcome
    (ok/unset/construction_failed/unreachable/error)
  - overall:
      "ready"        — all three ok
      "not_ready"    — at least one ok-blocker (unset / failed /
                        unreachable / error)

Exit 0 only when overall=ready. Operators can chain in CI:
  prsm node section7-readiness && systemctl restart prsm-operator
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


def _run(args, env=None):
    from prsm.cli import node
    runner = CliRunner()
    if env:
        with patch.dict(os.environ, env, clear=False):
            return runner.invoke(node, args)
    return runner.invoke(node, args)


def test_readiness_reports_not_ready_when_all_unset():
    keys = [
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
        "PRSM_STAKE_BOND_ADDRESS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        result = _run(["section7-readiness", "--format", "json"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert result.exit_code != 0
    import json
    data = json.loads(result.output)
    assert data["overall"] == "not_ready"
    assert data["components"]["anchor"]["outcome"] == "unset"
    assert data["components"]["stake_bond"]["outcome"] == "unset"


def test_readiness_reports_ready_when_all_ok():
    """All three probes return ok → overall=ready, exit 0."""
    fake_anchor_mod = MagicMock()
    fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(return_value=MagicMock())
    fake_stake_mod = MagicMock()
    fake_stake_mod.StakeManagerClient = MagicMock(return_value=MagicMock())
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"result": "0x2105"}

    with patch.dict(
        os.environ,
        {
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xa1",
            "PRSM_STAKE_BOND_ADDRESS": "0xa2",
        },
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {
                "prsm.security.publisher_key_anchor.client": fake_anchor_mod,
                "prsm.economy.web3.stake_manager": fake_stake_mod,
            },
        ):
            with patch("httpx.post", return_value=fake_resp):
                result = _run(["section7-readiness", "--format", "json"])
    assert result.exit_code == 0, result.output
    import json
    data = json.loads(result.output)
    assert data["overall"] == "ready"
    assert data["components"]["anchor"]["outcome"] == "ok"
    assert data["components"]["stake_bond"]["outcome"] == "ok"
    assert data["components"]["rpc"]["outcome"] == "ok"


def test_readiness_text_format_renders_summary():
    """Text format renders human-readable summary."""
    keys = [
        "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
        "PRSM_STAKE_BOND_ADDRESS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        result = _run(["section7-readiness"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert "anchor" in result.output.lower()
    assert "stake" in result.output.lower()
    assert "rpc" in result.output.lower()
