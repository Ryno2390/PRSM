"""Preflight diagnostic for FTNS_WALLET_PRIVATE_KEY (sprint 126).

New check surfaces wallet config status pre-startup so operators
can verify the right key is wired before the node begins on-chain
operations.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.cli import _node_preflight_diagnostics


# Test private key (well-known test value, never used on mainnet)
_TEST_PK = (
    "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
)


def _config():
    cfg = MagicMock()
    cfg.config_path = MagicMock()
    cfg.config_path.exists = MagicMock(return_value=False)
    cfg.api_port = 0  # arbitrary; bind probe uses ephemeral
    cfg.bootstrap_nodes = []
    cfg.p2p_port = 9001
    return cfg


def _wallet_check(checks):
    return next(
        (c for c in checks if c.name == "Wallet config (optional)"),
        None,
    )


class TestWalletPreflight:
    def test_no_pk_warns(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FTNS_WALLET_PRIVATE_KEY", None)
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "WARN"
        assert "not set" in wc.details

    def test_valid_pk_shows_address(self):
        with patch.dict(
            os.environ, {"FTNS_WALLET_PRIVATE_KEY": _TEST_PK},
            clear=False,
        ):
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "PASS"
        # Truncated address format
        assert "Operator address:" in wc.details
        assert "..." in wc.details

    def test_malformed_pk_warns(self):
        with patch.dict(
            os.environ,
            {"FTNS_WALLET_PRIVATE_KEY": "definitely-not-a-key"},
            clear=False,
        ):
            checks = _node_preflight_diagnostics(_config())
        wc = _wallet_check(checks)
        assert wc is not None
        assert wc.status == "WARN"
