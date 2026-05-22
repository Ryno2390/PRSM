"""Sprint 725 F58 — unary request_bytes size limit (F55 sibling).

Sprint 721 closed F55 on the streaming server-side handler. Sprint
725 closes the identical fix-class on the UNARY server-side
handler (`handle_chain_executor_request`).

Pre-725, the unary handler decoded CHAIN_PAYLOAD_KEY base64 with
NO size check. A malicious peer sending a 100MB+ base64 string
would force ~75MB allocation on b64decode → executor.execute()
gets a giant request_bytes → memory exhaustion on 2GB-RAM
operator droplets.

Fix mirrors F55:
- `_resolve_unary_request_max_bytes()` reads
  `PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES` (default 16 MiB; <=0 =
  unbounded; non-int safely defaults to 16 MiB).
- Pre-decode estimated_decoded = len(b64) * 3/4 check rejects
  before allocating.
- Post-decode `len(payload_bytes) > max` defense-in-depth.
- Both paths set `decode_error` which flows into the existing
  CHAIN_ERROR_KEY response — requester's future resolves with
  the actionable error message.
"""
from __future__ import annotations

import os


def test_resolve_unary_request_max_bytes_default():
    """Unset env → 16 MiB."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_request_max_bytes,
    )
    os.environ.pop("PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES", None)
    assert _resolve_unary_request_max_bytes() == 16 * 1024 * 1024


def test_resolve_unary_request_max_bytes_explicit_override():
    """Valid int env → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"] = "4194304"
    try:
        assert _resolve_unary_request_max_bytes() == 4194304
    finally:
        del os.environ["PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"]


def test_resolve_unary_request_max_bytes_zero_unbounded():
    """0 → unbounded (pre-725 behavior)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"] = "0"
    try:
        assert _resolve_unary_request_max_bytes() == 0
    finally:
        del os.environ["PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"]


def test_resolve_unary_request_max_bytes_typo_safely_defaults():
    """Non-int → safe-default 16 MiB."""
    from prsm.node.chain_executor_adapters import (
        _resolve_unary_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"] = "huge"
    try:
        assert _resolve_unary_request_max_bytes() == 16 * 1024 * 1024
    finally:
        del os.environ["PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES"]


def _unary_handler_source() -> str:
    """Sprint 726 refactored the body of `handle_chain_executor_request`
    into `_handle_chain_executor_request_body` for per-peer-cap
    wrapping. Source-inspection tests must look at BOTH so the
    invariant survives that refactor."""
    import inspect
    from prsm.node import chain_executor_adapters as _mod
    return inspect.getsource(_mod.handle_chain_executor_request) + (
        inspect.getsource(_mod._handle_chain_executor_request_body)
    )


def test_unary_handler_source_uses_resolved_max_bytes():
    """Pin: unary handler calls `_resolve_unary_request_max_bytes()`
    and gates decode on it."""
    src = _unary_handler_source()
    assert "_resolve_unary_request_max_bytes" in src, (
        "F58 fix requires the unary size gate to be wired into "
        "the unary handler"
    )


def test_unary_handler_emits_error_referencing_env_var():
    """When payload exceeds limit, error message references the
    env var so operators reading logs know what to tune."""
    src = _unary_handler_source()
    assert "PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES" in src, (
        "size-exceeded error must reference the env var so "
        "operators know what to tune"
    )
    assert "exceeds max bytes" in src
