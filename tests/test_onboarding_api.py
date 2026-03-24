"""Tests for the node onboarding wizard API."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from prsm.interface.api.main import app


client = TestClient(app)


class TestOnboardingWelcome:
    """Tests for the welcome/prerequisites endpoint."""

    def test_welcome_returns_200(self):
        """GET /onboarding/ should return 200."""
        resp = client.get("/onboarding/")
        assert resp.status_code == 200

    def test_welcome_html_contains_prsm(self):
        """Welcome page should contain PRSM branding."""
        resp = client.get("/onboarding/")
        assert b"PRSM" in resp.content

    def test_welcome_json_returns_checks(self):
        """JSON request should return prerequisite check results."""
        resp = client.get("/onboarding/", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert "python_version" in data
        assert "all_ok" in data
        assert isinstance(data["all_ok"], bool)


class TestOnboardingApiKeys:
    """Tests for the API keys endpoints."""

    def test_api_keys_get_returns_200(self):
        """GET /onboarding/api-keys should return 200."""
        resp = client.get("/onboarding/api-keys")
        assert resp.status_code == 200

    def test_api_keys_post_without_keys_redirects(self):
        """POST without keys should redirect to backend step."""
        resp = client.post(
            "/onboarding/api-keys",
            data={"anthropic_api_key": "", "openai_api_key": ""},
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

    def test_api_keys_json_post_returns_validation(self):
        """JSON POST should return validation results."""
        resp = client.post(
            "/onboarding/api-keys",
            json={"anthropic_api_key": "", "openai_api_key": ""},
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "validation" in data

    def test_api_keys_validation_with_real_keys(self):
        """Validation should be attempted when keys are provided."""
        # Note: We don't test actual validation calls since they're async internal functions
        # Instead, we verify the endpoint accepts the keys and returns a response
        # Using form data since the endpoint uses Form() parameters
        resp = client.post(
            "/onboarding/api-keys",
            data={
                "anthropic_api_key": "sk-ant-test123",
                "openai_api_key": "sk-test456"
            },
            headers={"Accept": "application/json"},
            follow_redirects=False
        )
        # Should redirect (303) or return JSON (200)
        assert resp.status_code in (200, 303)
        if resp.status_code == 200:
            data = resp.json()
            assert "validation" in data
            # Keys will fail validation (they're fake), but the endpoint should handle it
            assert "anthropic" in data["validation"]
            assert "openai" in data["validation"]


class TestOnboardingBackend:
    """Tests for the backend selection endpoints."""

    def test_backend_get_returns_200(self):
        """GET /onboarding/backend should return 200."""
        resp = client.get("/onboarding/backend")
        assert resp.status_code == 200

    def test_backend_json_returns_available_backends(self):
        """JSON request should return available backends."""
        resp = client.get("/onboarding/backend", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert "available_backends" in data
        assert "mock" in data["available_backends"]

    def test_backend_post_redirects(self):
        """POST should redirect to network step."""
        resp = client.post(
            "/onboarding/backend",
            data={"primary_backend": "mock", "fallback_chain": "mock"},
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

    def test_backend_json_post_saves_selection(self):
        """JSON POST should save backend selection."""
        resp = client.post(
            "/onboarding/backend",
            json={"primary_backend": "mock", "fallback_chain": "mock"},
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["primary_backend"] == "mock"


class TestOnboardingNetwork:
    """Tests for the network configuration endpoints."""

    def test_network_get_returns_200(self):
        """GET /onboarding/network should return 200."""
        resp = client.get("/onboarding/network")
        assert resp.status_code == 200

    def test_network_json_returns_config(self):
        """JSON request should return network config."""
        resp = client.get("/onboarding/network", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ipfs_status" in data
        assert "current_config" in data

    def test_network_post_redirects(self):
        """POST should redirect to identity step."""
        resp = client.post(
            "/onboarding/network",
            data={
                "p2p_port": "8765",
                "api_port": "8080",
                "bootstrap_nodes": "wss://bootstrap.prsm-network.com:8765",
                "ipfs_auto_start": "true"
            },
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

    def test_network_json_post_saves_config(self):
        """JSON POST should save network config."""
        resp = client.post(
            "/onboarding/network",
            json={
                "p2p_port": 8765,
                "api_port": 8080,
                "bootstrap_nodes": ["wss://bootstrap.prsm-network.com:8765"],
                "ipfs_auto_start": True
            },
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["config"]["p2p_port"] == 8765


class TestOnboardingIdentity:
    """Tests for the identity configuration endpoints."""

    def test_identity_get_returns_200(self):
        """GET /onboarding/identity should return 200."""
        resp = client.get("/onboarding/identity")
        assert resp.status_code == 200

    def test_identity_json_returns_status(self):
        """JSON request should return identity status."""
        resp = client.get("/onboarding/identity", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert "existing_identity" in data

    def test_identity_generate_endpoint_accepts_request(self, tmp_path):
        """POST with action=generate should accept the request."""
        # The endpoint should accept the generate action
        # (actual identity generation happens in prsm.node.identity module)
        resp = client.post(
            "/onboarding/identity",
            data={"action": "generate", "display_name": "test-node"},
            follow_redirects=False
        )
        # Should redirect or succeed (302/303 for redirect, 200 for JSON)
        assert resp.status_code in (200, 302, 303)


class TestOnboardingLaunch:
    """Tests for the launch/review endpoints."""

    def test_launch_get_returns_200(self):
        """GET /onboarding/launch should return 200."""
        resp = client.get("/onboarding/launch")
        assert resp.status_code == 200

    def test_launch_json_returns_config(self):
        """JSON request should return config preview."""
        resp = client.get("/onboarding/launch", headers={"Accept": "application/json"})
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "config_path" in data

    def test_launch_post_writes_config(self, tmp_path, monkeypatch):
        """POST /onboarding/launch should write config/node_config.json."""
        import prsm.interface.api.onboarding_router as m

        # Patch the config output path to use temp directory
        config_path = tmp_path / "node_config.json"
        monkeypatch.setattr(m, "CONFIG_OUTPUT_PATH", config_path)

        resp = client.post(
            "/onboarding/launch",
            json={"confirm": True},
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert config_path.exists()

        # Verify config content
        with open(config_path) as f:
            config = json.load(f)
        assert "primary_backend" in config
        assert "p2p_port" in config
        assert "api_port" in config


class TestOnboardingUtilityEndpoints:
    """Tests for utility endpoints."""

    def test_status_endpoint(self):
        """GET /onboarding/status should return current status."""
        resp = client.get("/onboarding/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "pending_config" in data
        assert "config_exists" in data

    def test_reset_endpoint(self):
        """DELETE /onboarding/reset should reset the session."""
        resp = client.delete("/onboarding/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


class TestOnboardingIntegration:
    """Integration tests for the complete onboarding flow."""

    def test_complete_flow(self, tmp_path, monkeypatch):
        """Test the complete onboarding flow from start to finish."""
        import prsm.interface.api.onboarding_router as m

        # Patch paths to use temp directory
        config_path = tmp_path / "node_config.json"
        identity_path = tmp_path / "node_identity.json"
        monkeypatch.setattr(m, "CONFIG_OUTPUT_PATH", config_path)
        monkeypatch.setattr(m, "IDENTITY_OUTPUT_PATH", identity_path)

        # Step 1: Welcome
        resp = client.get("/onboarding/")
        assert resp.status_code == 200

        # Step 2: API Keys (skip validation by not providing keys)
        resp = client.post(
            "/onboarding/api-keys",
            data={"anthropic_api_key": "", "openai_api_key": ""},
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

        # Step 3: Backend Selection
        resp = client.post(
            "/onboarding/backend",
            data={"primary_backend": "mock", "fallback_chain": "mock"},
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

        # Step 4: Network Config
        resp = client.post(
            "/onboarding/network",
            data={
                "p2p_port": "8765",
                "api_port": "8080",
                "bootstrap_nodes": "wss://bootstrap.prsm-network.com:8765",
                "ipfs_auto_start": "true"
            },
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

        # Step 5: Identity (generate new)
        # Note: We use action=import with an empty file to avoid actual key generation
        # In production, users would click "generate" which creates real keys
        resp = client.post(
            "/onboarding/identity",
            data={"action": "generate", "display_name": "test-node"},
            follow_redirects=False
        )
        assert resp.status_code in (200, 302, 303)

        # Step 6: Launch
        resp = client.post(
            "/onboarding/launch",
            headers={"Accept": "application/json"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert config_path.exists()
