"""Sprint 598 (Phase 2D step 4) — wire _build_chain_executor rpc kind.

Tests:
  - Default (kind=stub) still returns _StubChainExecutor.
  - rpc kind with missing anchor → falls back to stub + warns.
  - rpc kind with missing node._loop → falls back to stub + warns.
  - rpc kind with missing node.transport → falls back to stub + warns.
  - rpc kind with missing node.identity → falls back to stub + warns.
  - rpc kind with ALL deps present → returns real RpcChainExecutor.
  - make_rpc_chain_executor raising → falls back to stub + warns.

Sprint 597's response handler still needs to be wired into transport
dispatch (sprint 599). Sprint 598 just makes the CONSTRUCTION work;
runtime dispatch will time out until 599 lands.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


def _stub_node(loop=None, identity=True, transport=True):
    n = MagicMock()
    n.identity = MagicMock(node_id="self") if identity else None
    n.transport = MagicMock() if transport else None
    n._loop = loop
    n._chain_executor_pending = {}
    return n


def test_stub_kind_still_returns_stub():
    from prsm.node.inference_wiring import (
        _build_chain_executor, _StubChainExecutor,
    )
    os.environ.pop("PRSM_PARALLAX_CHAIN_EXECUTOR_KIND", None)
    exe = _build_chain_executor(_stub_node())
    assert isinstance(exe, _StubChainExecutor)


def test_rpc_falls_back_when_anchor_unset(caplog):
    """No anchor resolvable → stub fallback + warn.

    Sprint 629 added the networks.py fallback for Base mainnet,
    so to exercise the "anchor unresolvable" branch this test pins
    PRSM_NETWORK=sepolia (no default published yet).
    """
    from prsm.node.inference_wiring import (
        _build_chain_executor, _StubChainExecutor,
    )
    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
            "PRSM_NETWORK": "sepolia",
        },
        clear=False,
    ):
        os.environ.pop("PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS", None)
        with caplog.at_level("WARNING"):
            exe = _build_chain_executor(_stub_node(loop=MagicMock()))
    assert isinstance(exe, _StubChainExecutor)
    # Warning surfaces the missing piece
    assert any(
        "anchor" in r.getMessage().lower()
        or "PUBLISHER_KEY_ANCHOR" in r.getMessage()
        for r in caplog.records
    )


def test_rpc_falls_back_when_loop_unset(caplog):
    from prsm.node.inference_wiring import (
        _build_chain_executor, _StubChainExecutor,
    )
    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead",
        },
        clear=False,
    ):
        # Mock anchor construction so anchor IS present
        fake_anchor_mod = MagicMock()
        fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(
            return_value=MagicMock(),
        )
        with patch.dict(
            "sys.modules",
            {"prsm.security.publisher_key_anchor.client": fake_anchor_mod},
        ):
            with caplog.at_level("WARNING"):
                # loop=None forces the missing dependency
                exe = _build_chain_executor(_stub_node(loop=None))
    assert isinstance(exe, _StubChainExecutor)
    assert any("_loop" in r.getMessage() for r in caplog.records)


def test_rpc_constructs_real_executor_when_all_deps_present():
    """Happy path: all deps available → real RpcChainExecutor."""
    from prsm.node.inference_wiring import (
        _build_chain_executor, _StubChainExecutor,
    )

    fake_anchor_mod = MagicMock()
    fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(
        return_value=MagicMock(),
    )
    fake_exe = MagicMock(name="RpcChainExecutor")
    fake_factory_mod = MagicMock()
    fake_factory_mod.make_rpc_chain_executor = MagicMock(
        return_value=fake_exe,
    )

    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead",
        },
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {
                "prsm.security.publisher_key_anchor.client": fake_anchor_mod,
                "prsm.compute.chain_rpc.factories": fake_factory_mod,
            },
        ):
            exe = _build_chain_executor(_stub_node(loop=MagicMock()))

    assert exe is fake_exe
    assert not isinstance(exe, _StubChainExecutor)
    # Factory was called with the right kwargs
    fake_factory_mod.make_rpc_chain_executor.assert_called_once()
    kwargs = fake_factory_mod.make_rpc_chain_executor.call_args.kwargs
    assert "settler_identity" in kwargs
    assert "send_message" in kwargs
    assert "anchor" in kwargs
    assert "address_resolver" in kwargs


def test_rpc_falls_back_when_factory_raises(caplog):
    from prsm.node.inference_wiring import (
        _build_chain_executor, _StubChainExecutor,
    )

    fake_anchor_mod = MagicMock()
    fake_anchor_mod.PublisherKeyAnchorClient = MagicMock(
        return_value=MagicMock(),
    )
    fake_factory_mod = MagicMock()
    fake_factory_mod.make_rpc_chain_executor = MagicMock(
        side_effect=RuntimeError("factory kaboom"),
    )

    with patch.dict(
        os.environ,
        {
            "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND": "rpc",
            "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS": "0xdead",
        },
        clear=False,
    ):
        with patch.dict(
            "sys.modules",
            {
                "prsm.security.publisher_key_anchor.client": fake_anchor_mod,
                "prsm.compute.chain_rpc.factories": fake_factory_mod,
            },
        ):
            with caplog.at_level("WARNING"):
                exe = _build_chain_executor(_stub_node(loop=MagicMock()))
    assert isinstance(exe, _StubChainExecutor)
    assert any("factory kaboom" in r.getMessage() for r in caplog.records)
