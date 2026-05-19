"""Sprint 576 — profile_source env-driven plumbing (Phase 1).

Sprint 562 left profile_source as a hardcoded placeholder:

  profile_source=StakeWeightedTrustAdapter(
      inner=InMemoryProfileSource(snapshots={}),  # hardcoded empty
      stake_lookup=stake_lookup,
  ),

The deferral note said: "ProfileDHT requires multi-host send_message
+ peers, out of scope for single-node". Multi-host is now alive
(sprints 569-573), but ProfileDHT integration is a multi-sprint
job (wire ProfileMessageType handlers + protocol envelopes into
the WS transport, etc.).

Sprint 576 is Phase 1: refactor the hardcoded inner ProfileSource
construction into a helper `_build_inner_profile_source()` that
switches on `PRSM_PARALLAX_PROFILE_SOURCE_KIND`. Default
`in_memory` = behavior unchanged for every existing operator.
Future `dht` returns a real ProfileDHT (Phase 2 sprint, deferred).
Unknown kind = warn + fall back to in_memory (no daemon-startup
break on misconfigured env).

Invariants:
- Default operator (env unset) sees no behavior change.
- Helper exported so callers + tests can use it without invoking
  full trust-stack construction.
- Unknown kind logs warning, returns in_memory fallback.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def test_default_kind_in_memory_returns_empty_in_memory_source():
    """Unset env → InMemoryProfileSource(snapshots={})."""
    from prsm.node.inference_wiring import _build_inner_profile_source
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRSM_PARALLAX_PROFILE_SOURCE_KIND", None)
        inner = _build_inner_profile_source()
    assert isinstance(inner, InMemoryProfileSource)
    # Default starts empty
    assert getattr(inner, "snapshots", {}) == {}


def test_explicit_in_memory_kind_is_explicit_alias():
    from prsm.node.inference_wiring import _build_inner_profile_source
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_PROFILE_SOURCE_KIND": "in_memory"},
        clear=False,
    ):
        inner = _build_inner_profile_source()
    assert isinstance(inner, InMemoryProfileSource)


def test_unknown_kind_falls_back_to_in_memory_with_warning(caplog):
    """Misconfigured env must not crash daemon startup. Logs a
    structured warning + returns a usable in_memory source.
    """
    from prsm.node.inference_wiring import _build_inner_profile_source
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_PROFILE_SOURCE_KIND": "bogus_kind_xyz"},
        clear=False,
    ):
        with caplog.at_level("WARNING"):
            inner = _build_inner_profile_source()
    assert isinstance(inner, InMemoryProfileSource)
    # Some warning about the unknown kind
    assert any(
        "bogus_kind_xyz" in r.getMessage() or "unknown" in r.getMessage().lower()
        for r in caplog.records
    ), "Unknown kind must log a warning"


def test_dht_kind_returns_not_yet_implemented_via_in_memory_fallback():
    """Phase 2 will implement DHT; Phase 1 must NOT crash if an
    operator sets `dht` early — fall back to in_memory + warn so
    they can keep their daemon up while waiting for Phase 2.
    """
    from prsm.node.inference_wiring import _build_inner_profile_source
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )
    with patch.dict(
        os.environ,
        {"PRSM_PARALLAX_PROFILE_SOURCE_KIND": "dht"},
        clear=False,
    ):
        inner = _build_inner_profile_source()
    # Phase 1: dht falls back to in_memory (Phase 2 will wire ProfileDHT)
    assert isinstance(inner, InMemoryProfileSource)


def test_production_trust_stack_uses_helper():
    """The production trust-stack constructor must call the helper
    (not hardcode InMemoryProfileSource directly) so future
    profile_source kinds activate without churning that constructor.
    """
    import inspect
    from prsm.node.inference_wiring import (
        _build_production_trust_stack_or_none,
    )
    src = inspect.getsource(_build_production_trust_stack_or_none)
    assert "_build_inner_profile_source" in src, (
        "_build_production_trust_stack must call the helper so "
        "PRSM_PARALLAX_PROFILE_SOURCE_KIND env is honored"
    )
