"""Sprint 584 — `prsm node rpc-probe` CLI.

Completes the §7 production preflight trifecta:
  - sprint 581  anchor-probe          (PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS)
  - sprint 583  stake-bond-probe      (PRSM_STAKE_BOND_ADDRESS)
  - sprint 584  rpc-probe             (PRSM_BASE_RPC_URL — this sprint)

Both 581/583 fail with construction_failed when the RPC is the
real problem (e.g., infura key revoked, network outage). The rpc-
probe isolates: tests RPC reachability via eth_chainId JSON-RPC
call, so operators can distinguish "wrong contract address" from
"unreachable RPC".

Outcome: ok / unreachable / error
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


def test_rpc_probe_unreachable_when_post_raises():
    """httpx.HTTPError on POST → outcome=unreachable, exit nonzero."""
    import httpx as _httpx
    with patch(
        "httpx.post",
        side_effect=_httpx.ConnectError("dns failure"),
    ):
        result = _run(["rpc-probe", "--format", "json"])
    assert result.exit_code != 0
    import json
    data = json.loads(result.output)
    assert data["outcome"] == "unreachable"


def test_rpc_probe_ok_when_post_returns_chainId():
    """Mock POST returns valid eth_chainId → outcome=ok, exit 0."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": "0x2105"}
    with patch("httpx.post", return_value=fake_resp):
        result = _run(["rpc-probe", "--format", "json"])
    assert result.exit_code == 0, result.output
    import json
    data = json.loads(result.output)
    assert data["outcome"] == "ok"
    assert data.get("chain_id_hex") == "0x2105"


def test_rpc_probe_error_when_non_200_response():
    """Non-200 → outcome=error, exit nonzero."""
    fake_resp = MagicMock()
    fake_resp.status_code = 500
    fake_resp.text = "internal server error"
    fake_resp.json.side_effect = ValueError("not json")
    with patch("httpx.post", return_value=fake_resp):
        result = _run(["rpc-probe", "--format", "json"])
    assert result.exit_code != 0
    import json
    data = json.loads(result.output)
    assert data["outcome"] in ("error", "unreachable")


def test_rpc_probe_uses_default_when_env_unset():
    saved = os.environ.pop("PRSM_BASE_RPC_URL", None)
    try:
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"jsonrpc": "2.0", "result": "0x2105"}
        with patch("httpx.post", return_value=fake_resp) as p:
            result = _run(["rpc-probe", "--format", "json"])
        url_called = p.call_args.args[0] if p.call_args.args else p.call_args.kwargs.get("url", "")
        assert "mainnet.base.org" in url_called
    finally:
        if saved is not None:
            os.environ["PRSM_BASE_RPC_URL"] = saved
    assert result.exit_code == 0, result.output
