"""Sprint 689 F39 — chain executor decorators pass through streaming.

Live-attest of /compute/inference/stream after sprint 688 closed
the unary path returned:
  "chain executor does not support streaming: 'ActivationDPAware
   ChainExecutor' object has no attribute 'execute_chain_streaming'"

Root cause: TopologyAwareChainExecutor + ActivationDPAwareChain
Executor wrapped the unary execute_chain method but did not
forward execute_chain_streaming. The streaming method exists on
RpcChainExecutor; the wrappers shadowed it by not exposing a
matching name.

Sprint 689 adds:
  - TopologyAwareChainExecutor.execute_chain_streaming — pure
    passthrough (no topology recording on per-token frames).
  - ActivationDPAwareChainExecutor.execute_chain_streaming —
    passthrough for tier NONE; raises clear error for tier
    STANDARD+ because DP-aware streaming requires a streaming-
    aware injector (deferred sprint).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_topology_decorator_forwards_streaming():
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    inner = MagicMock()
    inner.execute_chain = MagicMock()  # required by ctor
    inner.execute_chain_streaming = MagicMock(return_value=iter([]))
    dec = TopologyAwareChainExecutor(inner=inner)
    chain = MagicMock()
    chain.stages = ["a", "b"]
    request = MagicMock()
    dec.execute_chain_streaming(request=request, chain=chain)
    inner.execute_chain_streaming.assert_called_once_with(
        request=request, chain=chain,
    )


def test_topology_decorator_raises_when_inner_no_streaming():
    """Inner that doesn't support streaming → clear AttributeError."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    # MagicMock with explicit spec excluding execute_chain_streaming
    inner = MagicMock(spec=["execute_chain"])
    dec = TopologyAwareChainExecutor(inner=inner)
    chain = MagicMock()
    chain.stages = []
    request = MagicMock()
    with pytest.raises(AttributeError, match="streaming"):
        dec.execute_chain_streaming(request=request, chain=chain)


def test_dp_decorator_passes_through_streaming_when_tier_none():
    """privacy_tier=NONE → DP wouldn't fire even unary; streaming
    passes through cleanly."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel
    inner = MagicMock()
    inner.execute_chain = MagicMock()
    inner.execute_chain_streaming = MagicMock(return_value=iter([]))
    dec = ActivationDPAwareChainExecutor(inner=inner)
    chain = MagicMock()
    chain.stages = ["a"]
    request = MagicMock()
    request.privacy_tier = PrivacyLevel.NONE
    dec.execute_chain_streaming(request=request, chain=chain)
    inner.execute_chain_streaming.assert_called_once_with(
        request=request, chain=chain,
    )


def test_dp_decorator_raises_streaming_for_tier_standard():
    """privacy_tier=STANDARD → DP+streaming not yet wired; raise."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel
    inner = MagicMock()
    inner.execute_chain = MagicMock()
    inner.execute_chain_streaming = MagicMock()
    dec = ActivationDPAwareChainExecutor(inner=inner)
    chain = MagicMock()
    chain.stages = ["a"]
    request = MagicMock()
    request.privacy_tier = PrivacyLevel.STANDARD
    with pytest.raises(RuntimeError, match="DP-aware streaming"):
        dec.execute_chain_streaming(request=request, chain=chain)
    # Inner streaming MUST NOT be called when DP would be required
    inner.execute_chain_streaming.assert_not_called()
