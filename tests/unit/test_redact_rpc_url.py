"""Sprint 171 — _redact_rpc_url() helper drops URL path/query so
log lines can't carry Alchemy / Infura / Quicknode API keys.

Pre-fix the 6 client-wiring log lines in prsm/node/node.py emitted:
  "X wired: 0x... via https://base-mainnet.g.alchemy.com/v2/<KEY>"

Any structured-logging pipeline (Datadog / Splunk / journald)
collecting these logs ended up with the secret API key in
searchable storage. Sprint 171 redacts to host before logging.

Also covers a similar redaction for the dogfood script's echo
line that surfaced the same URL to operator scrollback.
"""
from __future__ import annotations

import pytest

from prsm.node.node import _redact_rpc_url


def test_redact_alchemy_strips_path_and_key():
    """Alchemy URL: secret key in path → host-only after redaction."""
    full = "https://base-mainnet.g.alchemy.com/v2/SECRET_KEY_HERE"
    out = _redact_rpc_url(full)
    assert "SECRET_KEY_HERE" not in out
    assert out == "https://base-mainnet.g.alchemy.com"


def test_redact_infura_strips_key():
    full = "https://base.infura.io/v3/abcdef1234567890"
    out = _redact_rpc_url(full)
    assert "abcdef1234567890" not in out
    assert out == "https://base.infura.io"


def test_redact_quicknode_strips_token():
    """Quicknode embeds subdomain-prefix token; safe to keep host
    since the token IS the hostname — operators who care about
    that level of secrecy use a private RPC proxy."""
    full = "https://soft-name-xyz.base-mainnet.quiknode.pro/TOKEN/"
    out = _redact_rpc_url(full)
    assert "TOKEN" not in out
    assert "quiknode.pro" in out  # host preserved


def test_redact_public_base_org_unchanged():
    """Free public RPC: no secret in URL — redaction returns the
    same scheme+host."""
    full = "https://mainnet.base.org"
    out = _redact_rpc_url(full)
    assert out == "https://mainnet.base.org"


def test_redact_preserves_non_default_port():
    """Custom port (e.g., local hardhat node) preserved in output."""
    full = "http://localhost:8545"
    out = _redact_rpc_url(full)
    assert out == "http://localhost:8545"


def test_redact_empty_string():
    """Defensive — empty input returns placeholder, never raises."""
    assert _redact_rpc_url("") == "<rpc>"


def test_redact_garbage_input():
    """Unparseable input returns placeholder."""
    assert _redact_rpc_url("not a url at all") == "<rpc>"


def test_redact_wss_url():
    """Bootstrap servers use wss:// scheme — preserved."""
    full = "wss://bootstrap1.prsm-network.com:8765"
    out = _redact_rpc_url(full)
    assert out == "wss://bootstrap1.prsm-network.com:8765"
