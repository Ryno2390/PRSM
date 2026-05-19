"""Sprint 417 — default factory wraps with TopologyAware.

Pre-sprint-417 `make_rpc_chain_executor` returned a bare
`RpcChainExecutor`. Sprints 413-415 shipped the
TopologyAwareChainExecutor decorator + end-to-end signed-
receipt verification, but operators had to explicitly
wrap the factory's output to get topology in their
receipts.

Sprint 417 flips the factory default ON: production
receipts immediately carry verifiable
topology_assignment without operator opt-in. Operators
who want the raw RpcChainExecutor pass
``wrap_topology_aware=False``.

This is sound: the sprint-414 decorator is a strict-
superset wrapper (preserves all inner result fields,
propagates errors unchanged, only ADDS topology when
inner doesn't already set it). No backwards-compat risk
for callers that consume the ChainExecutor Protocol —
both classes implement it identically.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _identity():
    from prsm.node.identity import generate_node_identity
    return generate_node_identity("test-settler")


def _factory_kwargs():
    """Minimal kwargs to satisfy make_rpc_chain_executor.

    Sprint 546 added an outer ActivationDPAwareChainExecutor wrap by
    default. These sprint-417 tests scope to the topology-default
    surface only, so they explicitly opt out of the DP wrap.
    Sprint 546's own test file owns asserting that the DP layer
    composes correctly on top.
    """
    return dict(
        settler_identity=_identity(),
        send_message=MagicMock(return_value=b""),
        anchor=MagicMock(),
        wrap_activation_dp_aware=False,
    )


# ── Default behavior ─────────────────────────────────────


def test_default_wraps_with_topology_aware():
    """Sprint 417 — bare factory call returns a Topology-
    AwareChainExecutor, not a raw RpcChainExecutor."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(**_factory_kwargs())
    assert isinstance(executor, TopologyAwareChainExecutor)


def test_default_inner_is_rpc_chain_executor():
    """The wrapping is shallow: inner is the canonical
    RpcChainExecutor that pre-sprint-417 callers would
    have received directly."""
    from prsm.compute.chain_rpc.client import RpcChainExecutor
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    executor = make_rpc_chain_executor(**_factory_kwargs())
    assert isinstance(executor._inner, RpcChainExecutor)


# ── Opt-out path ─────────────────────────────────────────


def test_wrap_false_returns_raw_rpc_chain_executor():
    """Operators who want the pre-sprint-417 behavior pass
    wrap_topology_aware=False."""
    from prsm.compute.chain_rpc.client import RpcChainExecutor
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    executor = make_rpc_chain_executor(
        wrap_topology_aware=False, **_factory_kwargs(),
    )
    assert isinstance(executor, RpcChainExecutor)


def test_wrap_false_has_no_topology_aware_layer():
    """No accidental wrap when opt-out is set."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(
        wrap_topology_aware=False, **_factory_kwargs(),
    )
    assert not isinstance(executor, TopologyAwareChainExecutor)


# ── Wrap-true explicit ───────────────────────────────────


def test_wrap_true_explicit_wraps():
    """wrap_topology_aware=True is the same as the default
    (idempotent — explicit and default behave the same).
    """
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(
        wrap_topology_aware=True, **_factory_kwargs(),
    )
    assert isinstance(executor, TopologyAwareChainExecutor)


# ── Protocol surface preserved ───────────────────────────


def test_returned_executor_exposes_execute_chain():
    """Both wrapped and unwrapped returns implement the
    ChainExecutor Protocol — consumers don't need to know
    which form they got."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    wrapped = make_rpc_chain_executor(**_factory_kwargs())
    raw = make_rpc_chain_executor(
        wrap_topology_aware=False, **_factory_kwargs(),
    )
    assert hasattr(wrapped, "execute_chain")
    assert hasattr(raw, "execute_chain")
    assert callable(wrapped.execute_chain)
    assert callable(raw.execute_chain)
