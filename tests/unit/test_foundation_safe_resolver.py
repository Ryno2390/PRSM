"""Foundation Safe address resolver — env-overridable per-network.

Closes the Foundation-Safe-per-network-override placeholder follow-on
flagged in node.py B7 wiring.
"""
from __future__ import annotations

import pytest

from prsm.compute.query_orchestrator.foundation_safe_resolver import (
    DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS,
    FOUNDATION_SAFE_ADDRESS_ENV,
    resolve_foundation_safe_address,
)


# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────


class TestDefaults:
    def test_default_is_mainnet_safe(self, monkeypatch):
        monkeypatch.delenv(FOUNDATION_SAFE_ADDRESS_ENV, raising=False)
        assert resolve_foundation_safe_address() == DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS

    def test_default_is_42_char_hex_address(self):
        assert DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS.startswith("0x")
        assert len(DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS) == 42


# ──────────────────────────────────────────────────────────────────────
# Env override
# ──────────────────────────────────────────────────────────────────────


class TestEnvOverride:
    def test_env_var_overrides_default(self, monkeypatch):
        custom = "0xff00000000000000000000000000000000000000"
        monkeypatch.setenv(FOUNDATION_SAFE_ADDRESS_ENV, custom)
        assert resolve_foundation_safe_address() == custom

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(FOUNDATION_SAFE_ADDRESS_ENV, "")
        assert resolve_foundation_safe_address() == DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS

    def test_whitespace_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(FOUNDATION_SAFE_ADDRESS_ENV, "   ")
        assert resolve_foundation_safe_address() == DEFAULT_MAINNET_FOUNDATION_SAFE_ADDRESS

    def test_explicit_env_value_arg_takes_precedence_over_env(self, monkeypatch):
        # Test-friendly injection path bypasses the env var.
        monkeypatch.setenv(FOUNDATION_SAFE_ADDRESS_ENV, "0x" + "aa" * 20)
        result = resolve_foundation_safe_address(
            env_value="0x" + "bb" * 20,
        )
        assert result == "0x" + "bb" * 20


# ──────────────────────────────────────────────────────────────────────
# Validation — malformed addresses rejected
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_no_0x_prefix_raises(self):
        with pytest.raises(ValueError, match="0x-prefixed"):
            resolve_foundation_safe_address(env_value="ff" * 20)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="42 chars"):
            resolve_foundation_safe_address(env_value="0xff")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="42 chars"):
            resolve_foundation_safe_address(env_value="0x" + "ff" * 30)


# ──────────────────────────────────────────────────────────────────────
# Env name pin
# ──────────────────────────────────────────────────────────────────────


class TestEnvNamePin:
    """The env name `PRSM_FOUNDATION_SAFE_ADDRESS` is operator-facing
    config — operators set it in deploy scripts, docker-compose, etc.
    Renaming it without coordination breaks deployments. Pin it."""

    def test_env_var_name(self):
        assert FOUNDATION_SAFE_ADDRESS_ENV == "PRSM_FOUNDATION_SAFE_ADDRESS"
