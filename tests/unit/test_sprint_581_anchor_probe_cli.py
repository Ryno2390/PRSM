"""Sprint 581 — `prsm node anchor-probe` CLI.

After sprint 580 extracted `_build_anchor_or_none()`, operators
have a clean way to construct the anchor. But there's no surface
that EXERCISES the helper without booting the full daemon — the
existing pathway is "set env, restart daemon, grep startup logs
for the warning". That's a slow feedback loop for operators
trying to flip PRSM_PARALLAX_TRUST_STACK_KIND=production for the
first time.

Sprint 581 adds `prsm node anchor-probe [--format text|json]`:
calls _build_anchor_or_none(), reports:
  - PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS env value (or <unset>)
  - PRSM_BASE_RPC_URL env value (or default)
  - Outcome: ok / unset / construction_failed
  - Helpful guidance based on outcome

Same diagnostic value pattern as sprint 568's `prsm bootstrap-server
status` command — operator-side preflight before flipping a
production toggle.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner


def _run(args, env=None):
    from prsm.cli import node
    runner = CliRunner()
    if env:
        with patch.dict(os.environ, env, clear=False):
            return runner.invoke(node, args)
    return runner.invoke(node, args)


def test_anchor_probe_reports_unset_when_env_missing():
    saved = os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
    try:
        result = _run(["anchor-probe"])
    finally:
        if saved is not None:
            os.environ["PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS"] = saved
    assert result.exit_code != 0
    assert (
        "unset" in result.output.lower()
        or "not set" in result.output.lower()
        or "<unset>" in result.output
    )


def test_anchor_probe_reports_construction_failure_with_actionable_hint():
    """Env set but RPC unreachable → reports the failure clearly."""
    from unittest.mock import MagicMock
    fake_module = MagicMock()
    fake_module.PublisherKeyAnchorClient = MagicMock(
        side_effect=RuntimeError("rpc unreachable"),
    )
    with patch.dict(
        os.environ,
        {"PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead"},
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_module},
        ):
            result = _run(["anchor-probe"])
    assert result.exit_code != 0
    out = result.output.lower()
    # surfaces failure mode
    assert "rpc unreachable" in out or "construction_failed" in out or "failed" in out


def test_anchor_probe_success_reports_ok():
    """Env set + construction succeeds → exit 0 + 'ok'."""
    from unittest.mock import MagicMock
    fake_client = MagicMock()
    fake_module = MagicMock()
    fake_module.PublisherKeyAnchorClient = MagicMock(return_value=fake_client)
    with patch.dict(
        os.environ,
        {"PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead"},
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_module},
        ):
            result = _run(["anchor-probe"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output.lower() or "✓" in result.output


def test_anchor_probe_json_format():
    import json
    saved = os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
    try:
        result = _run(["anchor-probe", "--format", "json"])
    finally:
        if saved is not None:
            os.environ["PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS"] = saved
    # Exit code 1 (unset) but JSON parseable
    data = json.loads(result.output)
    assert "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS" in data
    assert "outcome" in data
    assert data["outcome"] in ("ok", "unset", "construction_failed")
