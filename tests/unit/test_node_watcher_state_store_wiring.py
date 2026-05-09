"""node.py wiring for the watcher last_processed_block state store.

Closes the deferred follow-on from
`watcher-last-processed-block-persistence-merge-ready-20260509`:
yesterday's persistence sprint shipped the surface; today's
sprint wires it through env-driven Node startup.

Activation pattern (dual-gate, matching the daemon-builder
precedent established in commits 321de20c + 56df2460):
  - PRSM_WATCHER_STATE_PERSISTENCE_ENABLED=1   (required to enable)
  - PRSM_WATCHER_STATE_DIR=<path>              (optional override
    of default ~/.prsm/watchers/)

When enabled: a single FilesystemLastProcessedBlockStore is
constructed at Node.initialize and threaded into all 3 watcher
builders. Each watcher uses its own WATCHER_KEY namespace
within the shared store.

When disabled (default): all 3 watcher builders construct
watchers with state_store=None, preserving today's chain-tip-
baseline-only legacy behavior.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.node import (
    _build_compensation_distributor_watcher_or_none,
    _build_key_distribution_watcher_or_none,
    _build_storage_slashing_watcher_or_none,
    _build_watcher_state_store_or_none,
)


# ──────────────────────────────────────────────────────────────────────
# State store builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildWatcherStateStore:
    def test_returns_none_when_persistence_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_WATCHER_STATE_PERSISTENCE_ENABLED", None)
            assert _build_watcher_state_store_or_none() is None

    def test_returns_none_when_persistence_unset_to_zero(self):
        with patch.dict(os.environ, {
            "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED": "0",
        }):
            assert _build_watcher_state_store_or_none() is None

    def test_returns_store_when_enabled(self):
        from prsm.economy.web3.last_processed_block_store import (
            FilesystemLastProcessedBlockStore,
        )
        with patch.dict(os.environ, {
            "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED": "1",
        }):
            store = _build_watcher_state_store_or_none()
            assert isinstance(store, FilesystemLastProcessedBlockStore)

    def test_returns_store_with_default_base_dir_when_no_override(self):
        with patch.dict(os.environ, {
            "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED": "1",
        }, clear=False):
            os.environ.pop("PRSM_WATCHER_STATE_DIR", None)
            store = _build_watcher_state_store_or_none()
            assert store is not None
            # Default base dir under ~/.prsm/watchers
            assert str(store.base_dir).endswith(".prsm/watchers") or \
                   str(store.base_dir).endswith(".prsm\\watchers")

    def test_returns_store_with_override_base_dir(self, tmp_path):
        with patch.dict(os.environ, {
            "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED": "1",
            "PRSM_WATCHER_STATE_DIR": str(tmp_path),
        }):
            store = _build_watcher_state_store_or_none()
            assert store is not None
            assert store.base_dir == Path(str(tmp_path))


# ──────────────────────────────────────────────────────────────────────
# Watcher builders thread the state_store through
# ──────────────────────────────────────────────────────────────────────


class TestWatcherBuildersThreadStateStore:
    """Each of the 3 watcher builders must accept a state_store
    kwarg + pass it to the watcher constructor when both client and
    enable env are set."""

    def test_key_distribution_builder_passes_state_store(self):
        client = MagicMock()
        store = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
        }):
            watcher = _build_key_distribution_watcher_or_none(
                client=client, state_store=store,
            )
        assert watcher is not None
        assert watcher._state_store is store

    def test_storage_slashing_builder_passes_state_store(self):
        client = MagicMock()
        store = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
        }):
            watcher = _build_storage_slashing_watcher_or_none(
                client=client, state_store=store,
            )
        assert watcher is not None
        assert watcher._state_store is store

    def test_compensation_distributor_builder_passes_state_store(self):
        client = MagicMock()
        store = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
        }):
            watcher = _build_compensation_distributor_watcher_or_none(
                client=client, state_store=store,
            )
        assert watcher is not None
        assert watcher._state_store is store

    def test_state_store_kwarg_is_optional(self):
        """Backwards-compat: omitting state_store kwarg = no
        persistence (legacy chain-tip baseline behavior preserved)."""
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
        }):
            # No state_store kwarg supplied.
            watcher = _build_key_distribution_watcher_or_none(client=client)
        assert watcher is not None
        assert watcher._state_store is None

    def test_state_store_none_explicitly_passes_through(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
        }):
            watcher = _build_storage_slashing_watcher_or_none(
                client=client, state_store=None,
            )
        assert watcher is not None
        assert watcher._state_store is None


# ──────────────────────────────────────────────────────────────────────
# Static wiring integrity (lifecycle)
# ──────────────────────────────────────────────────────────────────────


class TestNodeInitializeWiresStateStore:
    """Verify Node.initialize references the state-store builder +
    pre-allocates self._watcher_state_store. Mirrors the pattern
    from test_node_phase78_lifecycle.py."""

    def test_initialize_calls_state_store_builder(self):
        import inspect
        from prsm.node.node import PRSMNode as Node
        init_src = inspect.getsource(Node.initialize)
        assert "_build_watcher_state_store_or_none" in init_src, (
            "Node.initialize must call _build_watcher_state_store_or_none "
            "to construct the shared state store. Without this call, the "
            "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED env var has no effect."
        )

    def test_initialize_threads_state_store_into_3_watcher_builders(self):
        import inspect
        from prsm.node.node import PRSMNode as Node
        init_src = inspect.getsource(Node.initialize)
        # Each watcher builder must receive state_store= argument
        # (not just unconditionally None — the wiring is the value).
        for builder in (
            "_build_key_distribution_watcher_or_none",
            "_build_storage_slashing_watcher_or_none",
            "_build_compensation_distributor_watcher_or_none",
        ):
            # Find the builder call site and verify state_store=
            # appears within a few lines.
            idx = init_src.find(builder)
            assert idx >= 0, f"{builder} call missing from Node.initialize"
            # Look in the next 250 chars for state_store=
            snippet = init_src[idx:idx + 250]
            assert "state_store=" in snippet, (
                f"{builder} call site does not pass state_store=. "
                f"Without this, the env-driven persistence path doesn't "
                f"reach the watchers."
            )
