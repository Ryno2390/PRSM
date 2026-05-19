"""Sprint 546 — default factory wraps with ActivationDPAware too.

Sprint 417 wired ``TopologyAwareChainExecutor`` into
``make_rpc_chain_executor`` so production receipts immediately
carry verifiable ``topology_assignment``. Sprint 419 shipped
``ActivationDPAwareChainExecutor`` — the sibling decorator that
populates ``activation_noise_trace`` — but never wired it into
the factory. Real ``/compute/inference`` requests run topology-
aware but DP-unaware.

Sprint 546 flips that bit ON by default: the factory wraps with
ActivationDPAware (OUTERMOST, per the docstring composition order
``base → topology → dp``). Opt-out via ``wrap_activation_dp_aware
=False`` for callers that want a hook-free executor (test fixtures
asserting on the bare receipt shape, callers composing their own
DP path).

Soundness: ActivationDPAwareChainExecutor for Tier NONE returns
the inner result unchanged (no trace, no hook). Lower tiers
install the injector hook + populate the trace. Composition with
the topology decorator commutes via the **kwargs passthrough that
sprint 418 ensured.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _identity():
    from prsm.node.identity import generate_node_identity
    return generate_node_identity("test-settler")


def _factory_kwargs():
    return dict(
        settler_identity=_identity(),
        send_message=MagicMock(return_value=b""),
        anchor=MagicMock(),
    )


# ── Default behavior ──────────────────────────────────────


def test_default_wraps_with_activation_dp_aware():
    """Bare factory call returns an ActivationDPAware-
    ChainExecutor at the OUTERMOST layer."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(**_factory_kwargs())
    assert isinstance(executor, ActivationDPAwareChainExecutor)


def test_default_dp_inner_is_topology_aware():
    """Composition order: dp(topology(rpc)). The DP layer's
    inner is the TopologyAware decorator."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(**_factory_kwargs())
    assert isinstance(executor._inner, TopologyAwareChainExecutor)


def test_default_innermost_is_rpc_chain_executor():
    """Deepest layer is still the canonical RpcChainExecutor."""
    from prsm.compute.chain_rpc.client import RpcChainExecutor
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    executor = make_rpc_chain_executor(**_factory_kwargs())
    # dp → topology → rpc
    assert isinstance(executor._inner._inner, RpcChainExecutor)


# ── Opt-out path ──────────────────────────────────────────


def test_wrap_dp_false_returns_topology_aware_only():
    """wrap_activation_dp_aware=False keeps sprint-417 default
    (topology-aware, no DP wrap)."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(
        wrap_activation_dp_aware=False, **_factory_kwargs(),
    )
    assert not isinstance(executor, ActivationDPAwareChainExecutor)
    assert isinstance(executor, TopologyAwareChainExecutor)


def test_both_wraps_false_returns_raw_rpc():
    """Both opt-outs combined → pre-sprint-417 bare RpcChainExecutor."""
    from prsm.compute.chain_rpc.client import RpcChainExecutor
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    executor = make_rpc_chain_executor(
        wrap_topology_aware=False,
        wrap_activation_dp_aware=False,
        **_factory_kwargs(),
    )
    assert isinstance(executor, RpcChainExecutor)


def test_topology_off_but_dp_on_still_wraps_dp():
    """Operators can disable topology while keeping DP.
    The DP decorator's inner becomes the bare
    RpcChainExecutor — composition still sound because
    the DP decorator only needs ``execute_chain`` on its
    inner."""
    from prsm.compute.chain_rpc.client import RpcChainExecutor
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(
        wrap_topology_aware=False,
        wrap_activation_dp_aware=True,
        **_factory_kwargs(),
    )
    assert isinstance(executor, ActivationDPAwareChainExecutor)
    # Inner is the raw RPC since topology was opted out.
    assert isinstance(executor._inner, RpcChainExecutor)


# ── Wrap-true explicit ────────────────────────────────────


def test_wrap_dp_true_explicit_wraps():
    """Explicit True is the same as default."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    executor = make_rpc_chain_executor(
        wrap_activation_dp_aware=True, **_factory_kwargs(),
    )
    assert isinstance(executor, ActivationDPAwareChainExecutor)


# ── Protocol surface preserved ────────────────────────────


def test_returned_executor_exposes_execute_chain():
    """All four wrap permutations expose .execute_chain."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    perms = [
        dict(),  # default
        dict(wrap_topology_aware=False),
        dict(wrap_activation_dp_aware=False),
        dict(
            wrap_topology_aware=False,
            wrap_activation_dp_aware=False,
        ),
    ]
    for p in perms:
        exec_ = make_rpc_chain_executor(**p, **_factory_kwargs())
        assert hasattr(exec_, "execute_chain")
        assert callable(exec_.execute_chain)


# ── DP-decorator parameters threaded through ──────────────


def test_dp_clip_norm_and_delta_threaded_through_factory():
    """Factory exposes the DP decorator's clip_norm + delta
    knobs so operators can tune sensitivity bound + δ without
    hand-wrapping."""
    from prsm.compute.chain_rpc.factories import (
        make_rpc_chain_executor,
    )
    executor = make_rpc_chain_executor(
        dp_clip_norm=2.5,
        dp_delta=1e-6,
        **_factory_kwargs(),
    )
    assert executor._clip_norm == 2.5
    assert executor._delta == 1e-6
