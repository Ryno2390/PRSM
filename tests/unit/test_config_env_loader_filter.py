"""EnvironmentConfigLoader filters non-schema env vars (sprint 119).

Pre-fix: load() walked all PRSM_-prefixed env vars and tried to
nest them as config keys. PRSM_STORAGE_SLASHING_ADDRESS got
parsed as `{storage: {slashing: {address: ...}}}` which the
PRSMConfig schema rejected with extra_forbidden — producing
9-error validation cascades on every node startup.

Post-fix: filter to only env vars whose top-level segment
matches a known PRSMConfig field. Operational env vars
(storage, slashing, webhook, network, daemon, slash, heartbeat,
distribution, compensation, base) are silently ignored at the
config layer.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prsm.core.config.loaders import EnvironmentConfigLoader


class TestSchemaFieldFilter:
    def test_known_field_loaded(self):
        # `app_name` is a real PRSMConfig field
        with patch.dict(os.environ, {"PRSM_APP_NAME": "test"}, clear=True):
            result = EnvironmentConfigLoader().load(prefix="PRSM_")
        assert "app_name" in result
        assert result["app_name"] == "test"

    def test_unknown_top_level_filtered(self):
        # These all caused the pre-fix validation cascade
        env = {
            "PRSM_STORAGE_SLASHING_ADDRESS": "0xABC",
            "PRSM_WEBHOOK_URL": "https://hook.example.com",
            "PRSM_NETWORK": "testnet",
            "PRSM_DAEMON_WATCHDOG_INTERVAL_SEC": "10",
            "PRSM_SLASH_EVENT_LOG_DIR": "/tmp/slash",
            "PRSM_HEARTBEAT_LOG_DIR": "/tmp/hb",
            "PRSM_DISTRIBUTION_LOG_DIR": "/tmp/dist",
            "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS": "0xDEF",
            "PRSM_BASE_RPC_URL": "https://sepolia.base.org",
        }
        with patch.dict(os.environ, env, clear=True):
            result = EnvironmentConfigLoader().load(prefix="PRSM_")
        # None of these top-level keys should be in result
        for forbidden in (
            "storage", "webhook", "network", "daemon", "slash",
            "heartbeat", "distribution", "compensation", "base",
        ):
            assert forbidden not in result, (
                f"{forbidden} should be filtered (operational env, "
                f"not schema field)"
            )

    def test_known_and_unknown_mixed(self):
        env = {
            "PRSM_APP_NAME": "test",  # known schema field
            "PRSM_STORAGE_SLASHING_ADDRESS": "0xABC",  # unknown
        }
        with patch.dict(os.environ, env, clear=True):
            result = EnvironmentConfigLoader().load(prefix="PRSM_")
        assert "app_name" in result
        assert "storage" not in result

    def test_no_validation_errors_under_full_dogfood_env(self):
        """End-to-end: under the dogfood env vars, PRSMConfig
        validates cleanly (no extra_forbidden cascade)."""
        env = {
            # The full dogfood set from sprint 115
            "PRSM_NETWORK": "testnet",
            "PRSM_BASE_RPC_URL": "https://sepolia.base.org",
            "PRSM_STORAGE_SLASHING_ADDRESS": "0x" + "ab" * 20,
            "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS": "0x" + "cd" * 20,
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
            "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
            "PRSM_WEBHOOK_URL": "https://hook.example.com",
            "PRSM_DAEMON_WATCHDOG_INTERVAL_SEC": "10",
            "PRSM_SLASH_EVENT_LOG_DIR": "/tmp/slash",
            "PRSM_HEARTBEAT_LOG_DIR": "/tmp/hb",
            "PRSM_DISTRIBUTION_LOG_DIR": "/tmp/dist",
        }
        with patch.dict(os.environ, env, clear=True):
            result = EnvironmentConfigLoader().load(prefix="PRSM_")
            # Try constructing PRSMConfig — should NOT raise
            from prsm.core.config.schemas import PRSMConfig
            PRSMConfig(**result)  # validates cleanly
