"""Sprint 386 — PRSM_SECRETS_DIR env-var override for FileBackend.

Codebase audit (sprint 386) surfaced another sprint-383-
class bug: prsm/security/secrets.py:727 factory hardcoded
the Docker-conventional `/run/secrets` path for the
FileBackend without an env-var fallback. Sibling backends
(Vault, AWS, GCP) all use env-var fallbacks (VAULT_ADDR,
AWS_REGION, GCP_PROJECT). The file backend was the
odd-one-out — bare-metal operators using
`SECRETS_BACKEND=file` would silently read None for every
secret because `/run/secrets` doesn't exist on non-Docker
hosts (path-exists check at line 226 returns False →
silent None).

This sprint adds `PRSM_SECRETS_DIR` env-var fallback at
the factory site, mirroring the Vault/AWS/GCP pattern.
Default stays `/run/secrets` for backwards-compat with
existing Docker deployments.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from prsm.security.secrets import (
    FileBackend,
    SecretBackend,
    SecretsManager,
)


class TestFileBackendDefault:
    """Constructor default unchanged — backwards compat."""

    def test_filebackend_default_dir_unchanged(self):
        """Direct construction still uses Docker-
        conventional default."""
        backend = FileBackend()
        assert backend.secrets_dir == "/run/secrets"

    def test_filebackend_explicit_dir_honored(self, tmp_path):
        """Explicit secrets_dir kwarg wins."""
        backend = FileBackend(secrets_dir=str(tmp_path))
        assert backend.secrets_dir == str(tmp_path)


class TestSecretsManagerFactoryEnvOverride:
    """Sprint 386 — factory's _create_backend respects
    PRSM_SECRETS_DIR env var for the FILE backend."""

    @pytest.mark.asyncio
    async def test_factory_uses_env_var_when_set(
        self, tmp_path,
    ):
        custom = str(tmp_path / "custom-secrets")
        with patch.dict(
            os.environ,
            {"PRSM_SECRETS_DIR": custom},
            clear=False,
        ):
            mgr = SecretsManager(backend=SecretBackend.FILE)
            backend = await mgr._create_backend()
        assert backend.secrets_dir == custom

    @pytest.mark.asyncio
    async def test_factory_falls_back_to_docker_default(self):
        """When env unset, the factory uses /run/secrets
        (backwards-compat with existing Docker deployments)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_SECRETS_DIR", None)
            mgr = SecretsManager(backend=SecretBackend.FILE)
            backend = await mgr._create_backend()
        assert backend.secrets_dir == "/run/secrets"

    @pytest.mark.asyncio
    async def test_explicit_kwarg_beats_env_var(
        self, tmp_path,
    ):
        """If operator passes secrets_dir kwarg explicitly,
        it wins over the env-var fallback. Same precedence
        as the Vault/AWS/GCP factories."""
        explicit = str(tmp_path / "explicit")
        with patch.dict(
            os.environ,
            {"PRSM_SECRETS_DIR": str(tmp_path / "env-default")},
            clear=False,
        ):
            mgr = SecretsManager(
                backend=SecretBackend.FILE,
                secrets_dir=explicit,
            )
            backend = await mgr._create_backend()
        assert backend.secrets_dir == explicit


class TestSiblingBackendPattern:
    """Verify the file backend now mirrors the sibling-
    backend env-var pattern (Vault uses VAULT_ADDR; AWS
    uses AWS_REGION; GCP uses GCP_PROJECT). Pre-sprint-386,
    file was the only backend without an env-var fallback
    — that asymmetry was the bug."""

    def test_factory_source_uses_env_fallback_for_filebackend(
        self,
    ):
        """Source-level check that the file-backend factory
        uses os.getenv. Defends against the regression of
        someone re-hardcoding the path."""
        from pathlib import Path
        text = (
            Path(__file__).resolve().parents[2]
            / "prsm" / "security" / "secrets.py"
        ).read_text()
        # The constant appears somewhere in the file
        assert "PRSM_SECRETS_DIR" in text
        # Find the FileBackend factory block (now multi-line
        # post-sprint-386)
        marker_idx = text.find(
            "self.backend == SecretBackend.FILE"
        )
        assert marker_idx > 0
        # Within 1500 chars after that (comment block +
        # multi-line return is ~700-800 chars), expect both
        # os.getenv and PRSM_SECRETS_DIR — the canonical
        # env-var pattern this sprint enforces.
        nearby = text[marker_idx:marker_idx + 1500]
        assert "os.getenv" in nearby
        assert "PRSM_SECRETS_DIR" in nearby
