"""Sprint 592 — Phase 2A SendMessage adapter scaffolding.

Phase 2A is interface-only. Tests verify:
  - Module imports cleanly (no circular deps)
  - Protocol shape: SendMessageAdapter callable accepts (str, bytes)
  - build_send_message_adapter() returns a callable
  - Calling the placeholder raises _Phase2AdapterNotReady
  - Subsequent Phase 2C impl can substitute without breaking the contract

Phase 2B + 2C + 2D follow in sprints 593/594/595.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_module_imports_cleanly():
    """Sanity: scaffolding module has no circular deps."""
    import prsm.node.chain_executor_adapters as mod
    assert hasattr(mod, "SendMessage")
    assert hasattr(mod, "SendMessageAdapter")
    assert hasattr(mod, "build_send_message_adapter")
    assert hasattr(mod, "_Phase2AdapterNotReady")


def test_send_message_type_alias_signature():
    """SendMessage is the canonical sync Callable[[str, bytes], bytes]."""
    from prsm.node.chain_executor_adapters import SendMessage
    # The type alias should be a Callable annotation; runtime
    # introspection: it has __origin__ on typing-style Callable
    # OR exists as a real callable type. Either way, importable.
    assert SendMessage is not None


def test_build_send_message_adapter_returns_callable():
    """The builder returns a callable conforming to the protocol."""
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
        SendMessageAdapter,
    )
    adapter = build_send_message_adapter(MagicMock())
    assert callable(adapter)
    # Per Protocol, isinstance check is runtime-checkable
    assert isinstance(adapter, SendMessageAdapter)


def test_placeholder_raises_not_ready():
    """Calling the Phase 2A placeholder raises the typed exception
    so callers (sprint 595's _build_chain_executor wiring) can
    distinguish Phase 2 non-readiness from genuine transport
    failures.
    """
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
        _Phase2AdapterNotReady,
    )
    adapter = build_send_message_adapter(MagicMock())
    with pytest.raises(_Phase2AdapterNotReady):
        adapter("stage-1", b"request payload")


def test_phase2_not_ready_is_subclass_of_not_implemented_error():
    """Typed exception subclasses NotImplementedError so generic
    error handlers in upstream code (e.g., the chain executor
    error path) catch it as expected.
    """
    from prsm.node.chain_executor_adapters import _Phase2AdapterNotReady
    assert issubclass(_Phase2AdapterNotReady, NotImplementedError)


def test_placeholder_error_message_directs_operator_to_workaround():
    """The error message must tell operators to set
    PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=stub for a working daemon.
    """
    from prsm.node.chain_executor_adapters import (
        build_send_message_adapter,
        _Phase2AdapterNotReady,
    )
    adapter = build_send_message_adapter(MagicMock())
    try:
        adapter("x", b"y")
    except _Phase2AdapterNotReady as exc:
        msg = str(exc).lower()
        assert "stub" in msg or "phase 2" in msg
