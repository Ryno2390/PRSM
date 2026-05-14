"""Sprint 383 — PRSM_PEER_DB_PATH env-var override.

Real bug surfaced during sprint-382 AWS Tokyo deployment:
``prsm/bootstrap/config.py:123`` hardcodes
``peer_db_path = "/app/data/bootstrap_peers.db"`` with no
env-var override. Bare-metal installs (not running in
Docker) hit ``Failed to save peer database: [Errno 30]
Read-only file system: '/app'`` on every peer state change.

Additionally, ``__post_init__`` calls ``_validate_paths()``
BEFORE ``_load_from_environment()`` — so even if the env-var
were added naively, the path validation (which tries to
mkdir the directory) runs against the DEFAULT path first,
producing a spurious warning even when the operator
override is in place. Reorder fixes that.

Sprint 383 fixes both:
  1. Adds ``PRSM_PEER_DB_PATH`` env-var override in
     ``_load_from_environment``.
  2. Adds ``PRSM_PERSIST_PEERS`` env-var (operators who
     don't want peer-DB persistence at all).
  3. Reorders ``__post_init__`` so env-loading happens
     BEFORE path validation — the validated path is the
     resolved-from-env path, not the default.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.bootstrap.config import BootstrapConfig


class TestPeerDbPathEnvVar:
    """Sprint 383 — PRSM_PEER_DB_PATH override."""

    def test_default_peer_db_path_unchanged_when_env_unset(
        self,
    ):
        """Backwards-compat: with no env override, the
        Docker-conventional default still applies."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_PEER_DB_PATH", None)
            cfg = BootstrapConfig()
            assert cfg.peer_db_path == "/app/data/bootstrap_peers.db"

    def test_peer_db_path_overridden_by_env(self, tmp_path):
        """PRSM_PEER_DB_PATH overrides the default."""
        custom = str(tmp_path / "subdir" / "peers.db")
        with patch.dict(
            os.environ,
            {"PRSM_PEER_DB_PATH": custom},
            clear=False,
        ):
            cfg = BootstrapConfig()
            assert cfg.peer_db_path == custom

    def test_env_path_directory_actually_created(
        self, tmp_path,
    ):
        """The fix REORDERS __post_init__ so path validation
        runs AFTER env-loading — meaning the directory that
        gets mkdir'd is the env-supplied path, not the
        default. Verify the env-supplied directory is now
        present after BootstrapConfig instantiation."""
        custom_dir = tmp_path / "prsm-peers"
        custom_path = str(custom_dir / "bootstrap_peers.db")
        assert not custom_dir.exists()  # precondition

        with patch.dict(
            os.environ,
            {"PRSM_PEER_DB_PATH": custom_path},
            clear=False,
        ):
            cfg = BootstrapConfig()

        # The validate_paths step should have created the
        # custom directory, NOT the default /app/data
        assert custom_dir.exists()
        assert cfg.peer_db_path == custom_path

    def test_env_path_handles_already_existing_directory(
        self, tmp_path,
    ):
        """Pre-existing directory is fine — mkdir(exist_ok=
        True) is idempotent."""
        custom_dir = tmp_path / "already-here"
        custom_dir.mkdir()
        custom_path = str(custom_dir / "peers.db")

        with patch.dict(
            os.environ,
            {"PRSM_PEER_DB_PATH": custom_path},
            clear=False,
        ):
            cfg = BootstrapConfig()
        assert cfg.peer_db_path == custom_path

    def test_empty_env_value_falls_back_to_default(self):
        """Empty string env value should NOT override (matches
        the pattern of other env-var loaders in this file)."""
        with patch.dict(
            os.environ,
            {"PRSM_PEER_DB_PATH": ""},
            clear=False,
        ):
            cfg = BootstrapConfig()
            assert cfg.peer_db_path == "/app/data/bootstrap_peers.db"


class TestPersistPeersEnvVar:
    """Sprint 383 — PRSM_PERSIST_PEERS override."""

    def test_persist_peers_defaults_to_true(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_PERSIST_PEERS", None)
            cfg = BootstrapConfig()
            assert cfg.persist_peers is True

    def test_persist_peers_can_be_disabled(self):
        for val in ("false", "0", "no", "FALSE"):
            with patch.dict(
                os.environ,
                {"PRSM_PERSIST_PEERS": val},
                clear=False,
            ):
                cfg = BootstrapConfig()
                assert cfg.persist_peers is False, (
                    f"PRSM_PERSIST_PEERS={val!r} should "
                    f"disable persistence"
                )

    def test_persist_peers_truthy_values_keep_enabled(self):
        for val in ("true", "1", "yes", "TRUE"):
            with patch.dict(
                os.environ,
                {"PRSM_PERSIST_PEERS": val},
                clear=False,
            ):
                cfg = BootstrapConfig()
                assert cfg.persist_peers is True


class TestPostInitOrdering:
    """The reordering fix — env loading runs BEFORE path
    validation so validated paths reflect env overrides."""

    def test_env_path_no_spurious_default_dir_creation(
        self, tmp_path, caplog,
    ):
        """When PRSM_PEER_DB_PATH is set, NO mkdir attempt
        on the default /app/data path. This is the
        regression-class fix: pre-sprint-383, the validator
        would try /app/data first (failing on bare metal)
        AND THEN load the env override, leaving the
        spurious warning in logs."""
        custom_path = str(tmp_path / "ok" / "peers.db")
        with patch.dict(
            os.environ,
            {"PRSM_PEER_DB_PATH": custom_path},
            clear=False,
        ):
            with caplog.at_level("WARNING"):
                BootstrapConfig()
        # No /app/data warning when env override is set
        all_msgs = " ".join(r.message for r in caplog.records)
        assert "/app/data" not in all_msgs, (
            "Pre-fix would warn about /app/data even when "
            "operator overrode the path"
        )
