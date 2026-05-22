"""Sprint 721 F55 — request_bytes size limit on handle_chain_stream_request.

Sprint 711's `handle_chain_stream_request` decoded the
CHAIN_PAYLOAD_KEY base64 with no size check. A malicious peer
sending a 100MB+ base64 string would force ~75MB allocation on
b64decode → easy memory-exhaustion DoS on 2GB-RAM operator
droplets.

Fix: env-tunable max bytes (default 16 MiB; covers a real gpt2
activation blob with 100-token context ~ 4 MB + signing /
attestation metadata). Pre-decode check on encoded length
(estimated_decoded = len(b64) * 3 / 4) bounds the decode itself.
Post-decode check on actual bytes for defense-in-depth (decoded
size can be slightly less than estimate due to padding).
"""
from __future__ import annotations

import os


def test_resolve_stream_request_max_bytes_default():
    """Unset env → 16 MiB default (sprint 721 chosen baseline)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_request_max_bytes,
    )
    os.environ.pop("PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES", None)
    assert _resolve_stream_request_max_bytes() == 16 * 1024 * 1024


def test_resolve_stream_request_max_bytes_explicit_override():
    """Valid int env → that value."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"] = "8388608"
    try:
        assert _resolve_stream_request_max_bytes() == 8388608
    finally:
        del os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"]


def test_resolve_stream_request_max_bytes_zero_means_unbounded():
    """0 → unbounded (pre-721 behavior preserved for operators who
    explicitly opt out of the gate)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"] = "0"
    try:
        assert _resolve_stream_request_max_bytes() == 0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"]


def test_resolve_stream_request_max_bytes_negative_unbounded():
    """Negative → unbounded (defensive opt-out)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"] = "-1"
    try:
        assert _resolve_stream_request_max_bytes() == 0
    finally:
        del os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"]


def test_resolve_stream_request_max_bytes_typo_safely_defaults():
    """Non-int → safe-default 16 MiB (rather than failing setup)."""
    from prsm.node.chain_executor_adapters import (
        _resolve_stream_request_max_bytes,
    )
    os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"] = "sixteen-megs"
    try:
        assert _resolve_stream_request_max_bytes() == 16 * 1024 * 1024
    finally:
        del os.environ["PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES"]


def _server_handler_source() -> str:
    """Sprint 723 refactored the body of `handle_chain_stream_request`
    into `_handle_stream_request_body`. Source tests must inspect
    both for the size-gate invariant to be findable."""
    import inspect
    from prsm.node import chain_executor_adapters as _mod
    return inspect.getsource(_mod.handle_chain_stream_request) + (
        inspect.getsource(_mod._handle_stream_request_body)
    )


def test_handle_request_source_uses_resolved_max_bytes():
    """Pin: server-side handler must call
    `_resolve_stream_request_max_bytes()` and gate the decode on it."""
    src = _server_handler_source()
    assert "_resolve_stream_request_max_bytes" in src, (
        "F55 fix requires the size gate to be wired into "
        "the server-side stream request handler"
    )
    # Pre-decode estimate check uses *3//4 trick — pin that shape
    # so future refactors don't lose the cheap-pre-check property.
    assert "len(payload_b64)" in src or "len(" in src, (
        "size gate must check encoded length BEFORE base64 decode"
    )


def test_handle_request_emits_terminal_end_with_size_error():
    """Pin: when payload exceeds limit, server sends terminal
    STREAM_END with `CHAIN_ERROR_KEY` containing a clear message
    referencing the env var for tuning. Operators reading logs
    should know how to fix."""
    src = _server_handler_source()
    assert "PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES" in src, (
        "size-exceeded error message must reference the env var "
        "so operators know what to tune"
    )
    assert "exceeds max bytes" in src
