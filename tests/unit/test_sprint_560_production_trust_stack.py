"""Sprint 560 — `production` trust_stack kind: real anchor wiring.

Sprint 558 shipped the wiring contract; sprint 559 hardened the
catalog schema. Sprint 560 progresses the trust-stack from "mock
only" to a partial-production kind:

  PRSM_PARALLAX_TRUST_STACK_KIND=production

Today (sprint 560) "production" means:
  - anchor          : REAL PublisherKeyAnchorClient (Phase 3.x.3)
  - stake_lookup    : PLACEHOLDER (zero stake — passes-through;
                                   sprint 561 wires real)
  - profile_source  : PLACEHOLDER (empty InMemoryProfileSource;
                                   sprint 562 wires real)
  - consensus_hook  : PLACEHOLDER (no-op submitter;
                                   sprint 562 wires arbitration
                                   queue submitter)

The kind name is "production" because that's the target — the
log message explicitly enumerates which sub-components are still
placeholders so operators know what's actually verified.

Anchor requires PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS env var (and
optionally PRSM_BASE_RPC_URL). Missing → return None with the
actionable hint pointing at the env var.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest


def _stub_node():
    from prsm.node.identity import generate_node_identity
    n = MagicMock()
    n.identity = generate_node_identity("test-settler")
    return n


def _v1_catalog(tmp_path):
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps({
        "schema_version": "v1",
        "models": {
            "test-model": {
                "model_name": "test-model",
                "mlx_model_name": "test-model",
                "head_size": 64, "hidden_dim": 128,
                "intermediate_dim": 256,
                "num_attention_heads": 4, "num_kv_heads": 4,
                "vocab_size": 1000, "num_layers": 4,
            },
        },
    }))
    return p


def _set_env_production(monkeypatch, tmp_path, anchor_addr=None):
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        str(_v1_catalog(tmp_path)),
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "production",
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    if anchor_addr is None:
        monkeypatch.delenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", raising=False,
        )
    else:
        monkeypatch.setenv(
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", anchor_addr,
        )


# ── production kind recognized ────────────────────────────


def test_production_kind_constructs_when_anchor_address_set(
    monkeypatch, tmp_path,
):
    """With PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS set, the builder
    constructs a real PublisherKeyAnchorClient + returns a working
    ParallaxScheduledExecutor. The web3 layer is mocked so this
    test doesn't require an RPC endpoint."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    _set_env_production(
        monkeypatch, tmp_path,
        anchor_addr="0x" + "ab" * 20,
    )
    # Mock PublisherKeyAnchorClient construction so we don't talk
    # to RPC during unit tests.
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        autospec=True,
    ) as mock_cls:
        mock_cls.return_value.lookup = MagicMock(return_value=None)
        result = build_parallax_executor_or_none(_stub_node())
    assert isinstance(result, ParallaxScheduledExecutor)
    assert "test-model" in result.supported_models()


def test_production_kind_returns_none_when_no_anchor_address(
    monkeypatch, tmp_path, caplog,
):
    """No PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS → can't build a real
    anchor → return None with actionable hint."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    _set_env_production(monkeypatch, tmp_path, anchor_addr=None)
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS" in log_text


def test_production_kind_logs_placeholder_components(
    monkeypatch, tmp_path, caplog,
):
    """When production kind constructs, the log explicitly names
    which sub-components are still PLACEHOLDER so operators don't
    mistakenly trust them. Today: stake_lookup, profile_source,
    consensus_hook."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    _set_env_production(
        monkeypatch, tmp_path, anchor_addr="0x" + "cd" * 20,
    )
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        autospec=True,
    ) as mock_cls:
        mock_cls.return_value.lookup = MagicMock(return_value=None)
        with caplog.at_level(logging.INFO):
            result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    log_text = " ".join(r.message for r in caplog.records)
    # The log MUST name the placeholders so operators know.
    assert "stake_lookup" in log_text
    assert "profile_source" in log_text
    assert "consensus_hook" in log_text
    # And it MUST also name the part that's actually production.
    assert "anchor" in log_text.lower()


def test_production_kind_anchor_propagates_to_trust_stack(
    monkeypatch, tmp_path,
):
    """The constructed PublisherKeyAnchorClient is wired through to
    the executor's trust_stack.anchor_verify so anchor verification
    actually goes through the real client (test asserts via mock
    invocation when AnchorVerifyAdapter calls lookup)."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    _set_env_production(
        monkeypatch, tmp_path, anchor_addr="0x" + "ef" * 20,
    )
    fake_anchor = MagicMock()
    fake_anchor.lookup = MagicMock(return_value="addr-from-anchor")
    with patch(
        "prsm.security.publisher_key_anchor.client."
        "PublisherKeyAnchorClient",
        return_value=fake_anchor,
    ):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    # Trust stack's anchor_verify is the real adapter wrapping the
    # mocked client. The adapter delegates to client.lookup.
    trust = result._trust
    assert trust.anchor_verify.anchor is fake_anchor


def test_unknown_kind_still_rejected(monkeypatch, tmp_path, caplog):
    """Sanity: sprint 558's reject-unknown-kind behavior is
    preserved when the kind isn't `mock` or `production`."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    _set_env_production(
        monkeypatch, tmp_path, anchor_addr="0x" + "11" * 20,
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "totally-bogus",
    )
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "totally-bogus" in log_text


def test_mock_kind_still_works(monkeypatch, tmp_path):
    """Sprint-558's mock kind path is unchanged — adding 'production'
    didn't accidentally regress 'mock'."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE",
        str(_v1_catalog(tmp_path)),
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )
    result = build_parallax_executor_or_none(_stub_node())
    assert isinstance(result, ParallaxScheduledExecutor)
