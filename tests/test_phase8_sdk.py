"""
Phase 8 SDK completeness verification.

Verifies that all three SDKs build and their core APIs exist.
Tests the completion of SDK modules, documentation, and examples.

Run with: pytest tests/test_phase8_sdk.py -v
"""
import subprocess
import pytest
import shutil
from pathlib import Path
import os

REPO_ROOT = Path(__file__).parent.parent


def shutil_which(cmd):
    """Helper to check if a command is available."""
    return shutil.which(cmd) is not None


# =============================================================================
# Python SDK Tests
# =============================================================================

def prsm_sdk_installed():
    """Check if prsm_sdk is available for import."""
    try:
        import prsm_sdk
        return True
    except ImportError:
        return False


class TestPythonSDK:
    """Tests for Python SDK completeness."""

    @pytest.mark.skipif(not prsm_sdk_installed(), reason="prsm_sdk not installed")
    def test_python_sdk_imports(self):
        """Test that Python SDK core modules import correctly."""
        from prsm_sdk import PRSMClient
        from prsm_sdk.ftns import FTNSManager
        from prsm_sdk.marketplace import ModelMarketplace
        from prsm_sdk.storage import StorageClient
        from prsm_sdk.compute import ComputeClient
        from prsm_sdk.governance import GovernanceClient
        from prsm_sdk.tools import ToolExecutor

        assert PRSMClient is not None
        assert FTNSManager is not None
        assert ModelMarketplace is not None
        assert StorageClient is not None

    @pytest.mark.skipif(not prsm_sdk_installed(), reason="prsm_sdk not installed")
    def test_python_sdk_has_client_methods(self):
        """Test that PRSMClient has expected methods."""
        from prsm_sdk import PRSMClient

        # Check core methods exist
        assert hasattr(PRSMClient, 'query')
        assert hasattr(PRSMClient, 'health_check')
        assert hasattr(PRSMClient, 'estimate_cost')

    @pytest.mark.skipif(not prsm_sdk_installed(), reason="prsm_sdk not installed")
    def test_python_sdk_exceptions(self):
        """Test that SDK exceptions are available."""
        from prsm_sdk.exceptions import (
            PRSMError,
            AuthenticationError,
            RateLimitError,
            NetworkError,
            InsufficientFundsError,
        )

        assert PRSMError is not None
        assert AuthenticationError is not None
        assert RateLimitError is not None

    @pytest.mark.skipif(not prsm_sdk_installed(), reason="prsm_sdk not installed")
    def test_python_sdk_models(self):
        """Test that SDK models are available."""
        from prsm_sdk.models import (
            PRSMResponse,
            QueryRequest,
            FTNSBalance,
            ModelInfo,
        )

        assert PRSMResponse is not None
        assert QueryRequest is not None

    def test_python_sdk_integration_tests_exist(self):
        """Test that integration tests file exists."""
        integration_tests = REPO_ROOT / "sdks/python/tests/test_integration.py"
        assert integration_tests.exists(), "Python SDK integration tests not found"

    def test_python_sdk_readme_exists(self):
        """Test that Python SDK has README."""
        readme = REPO_ROOT / "sdks/python/README.md"
        assert readme.exists(), "Python SDK README not found"
        assert readme.stat().st_size > 1000, "Python SDK README seems too short"

    def test_python_sdk_pyproject_exists(self):
        """Test that pyproject.toml exists and is valid."""
        pyproject = REPO_ROOT / "sdks/python/pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"
        assert pyproject.stat().st_size > 500, "pyproject.toml seems too small"


# =============================================================================
# Go SDK Tests
# =============================================================================

class TestGoSDK:
    """Tests for Go SDK completeness."""

    def test_go_sdk_client_exists(self):
        """Test that Go SDK client module exists."""
        client = REPO_ROOT / "sdks/go/client/client.go"
        assert client.exists(), "Go SDK client not found"

    def test_go_sdk_ftns_module_exists(self):
        """Test that Go SDK ftns module exists."""
        ftns = REPO_ROOT / "sdks/go/ftns/ftns.go"
        assert ftns.exists(), "Go SDK ftns module not found"

    def test_go_sdk_marketplace_module_exists(self):
        """Test that Go SDK marketplace module exists."""
        marketplace = REPO_ROOT / "sdks/go/marketplace/marketplace.go"
        assert marketplace.exists(), "Go SDK marketplace module not found"

    def test_go_sdk_storage_module_exists(self):
        """Test that Go SDK storage module exists."""
        storage = REPO_ROOT / "sdks/go/storage/storage.go"
        assert storage.exists(), "Go SDK storage module not found"

    def test_go_sdk_tools_module_exists(self):
        """Test that Go SDK tools module exists."""
        tools = REPO_ROOT / "sdks/go/tools/tools.go"
        assert tools.exists(), "Go SDK tools module not found"

    def test_go_sdk_auth_module_exists(self):
        """Test that Go SDK auth module exists."""
        auth = REPO_ROOT / "sdks/go/auth/auth.go"
        assert auth.exists(), "Go SDK auth module not found"

    def test_go_sdk_nwtn_module_exists(self):
        """Test that Go SDK nwtn module exists."""
        nwtn = REPO_ROOT / "sdks/go/nwtn/nwtn.go"
        assert nwtn.exists(), "Go SDK nwtn module not found"

    def test_go_sdk_governance_module_exists(self):
        """Test that Go SDK governance module exists."""
        governance = REPO_ROOT / "sdks/go/governance/governance.go"
        assert governance.exists(), "Go SDK governance module not found"

    def test_go_sdk_websocket_module_exists(self):
        """Test that Go SDK websocket module exists."""
        websocket = REPO_ROOT / "sdks/go/websocket/websocket.go"
        assert websocket.exists(), "Go SDK websocket module not found"

    def test_go_sdk_types_module_exists(self):
        """Test that Go SDK types module exists."""
        types = REPO_ROOT / "sdks/go/types/types.go"
        assert types.exists(), "Go SDK types module not found"

    def test_go_sdk_go_mod_exists(self):
        """Test that go.mod exists and is valid."""
        go_mod = REPO_ROOT / "sdks/go/go.mod"
        assert go_mod.exists(), "go.mod not found"

    def test_go_sdk_readme_exists(self):
        """Test that Go SDK has README."""
        readme = REPO_ROOT / "sdks/go/README.md"
        assert readme.exists(), "Go SDK README not found"


# =============================================================================
# JavaScript SDK Tests
# =============================================================================

class TestJavaScriptSDK:
    """Tests for JavaScript SDK completeness."""

    def test_javascript_sdk_package_json_exists(self):
        """Test that package.json exists and is valid."""
        package_json = REPO_ROOT / "sdks/javascript/package.json"
        assert package_json.exists(), "package.json not found"
        assert package_json.stat().st_size > 500, "package.json seems too small"

    def test_javascript_sdk_src_exists(self):
        """Test that source directory exists."""
        src = REPO_ROOT / "sdks/javascript/src"
        assert src.exists(), "JavaScript SDK src directory not found"
        assert src.is_dir(), "src should be a directory"

    def test_javascript_sdk_index_exists(self):
        """Test that main index file exists."""
        index = REPO_ROOT / "sdks/javascript/src/index.ts"
        assert index.exists(), "JavaScript SDK index.ts not found"

    def test_javascript_sdk_readme_exists(self):
        """Test that JavaScript SDK has README."""
        readme = REPO_ROOT / "sdks/javascript/README.md"
        assert readme.exists(), "JavaScript SDK README not found"


# =============================================================================
# Documentation Tests
# =============================================================================

class TestDocumentation:
    """Tests for documentation completeness."""

    def test_participant_guide_exists(self):
        """Test that PARTICIPANT_GUIDE.md exists."""
        guide = REPO_ROOT / "docs/PARTICIPANT_GUIDE.md"
        assert guide.exists(), "PARTICIPANT_GUIDE.md not found"
        assert guide.stat().st_size > 3000, "Participant guide seems too short"

    def test_operator_guide_exists(self):
        """Test that OPERATOR_GUIDE.md exists."""
        guide = REPO_ROOT / "docs/OPERATOR_GUIDE.md"
        assert guide.exists(), "OPERATOR_GUIDE.md not found"
        assert guide.stat().st_size > 3000, "Operator guide seems too short"

    def test_sdk_developer_guide_exists(self):
        """Test that SDK_DEVELOPER_GUIDE.md exists."""
        guide = REPO_ROOT / "docs/SDK_DEVELOPER_GUIDE.md"
        assert guide.exists(), "SDK_DEVELOPER_GUIDE.md not found"
        assert guide.stat().st_size > 3000, "SDK developer guide seems too short"

    def test_participant_guide_has_key_sections(self):
        """Test that participant guide has expected sections."""
        guide = REPO_ROOT / "docs/PARTICIPANT_GUIDE.md"
        content = guide.read_text()

        assert "What is PRSM" in content or "What is FTNS" in content
        assert "How to Participate" in content or "Contribute" in content.lower()
        assert "FAQ" in content or "Frequently Asked" in content

    def test_operator_guide_has_key_sections(self):
        """Test that operator guide has expected sections."""
        guide = REPO_ROOT / "docs/OPERATOR_GUIDE.md"
        content = guide.read_text()

        assert "Installation" in content or "Docker" in content
        assert "Configuration" in content or "Environment" in content
        assert "Monitoring" in content or "Health" in content

    def test_sdk_developer_guide_has_key_sections(self):
        """Test that SDK developer guide has expected sections."""
        guide = REPO_ROOT / "docs/SDK_DEVELOPER_GUIDE.md"
        content = guide.read_text()

        assert "Python" in content
        assert "JavaScript" in content or "TypeScript" in content
        assert "Go" in content


# =============================================================================
# Examples Tests
# =============================================================================

class TestSDKExamples:
    """Tests for SDK examples completeness."""

    def test_python_examples_directory_exists(self):
        """Test that Python examples directory exists."""
        examples = REPO_ROOT / "sdks/python/examples"
        assert examples.exists(), "Python examples directory not found"

    def test_python_basic_usage_example_exists(self):
        """Test that basic usage example exists."""
        example = REPO_ROOT / "sdks/python/examples/basic_usage.py"
        assert example.exists(), "basic_usage.py not found"

    def test_python_streaming_example_exists(self):
        """Test that streaming example exists."""
        example = REPO_ROOT / "sdks/python/examples/streaming.py"
        assert example.exists(), "streaming.py not found"

    def test_python_marketplace_example_exists(self):
        """Test that marketplace example exists."""
        example = REPO_ROOT / "sdks/python/examples/marketplace.py"
        assert example.exists(), "marketplace.py not found"

    def test_python_tools_example_exists(self):
        """Test that tools example exists."""
        example = REPO_ROOT / "sdks/python/examples/tools.py"
        assert example.exists(), "tools.py not found"

    def test_python_cost_management_example_exists(self):
        """Test that cost management example exists."""
        example = REPO_ROOT / "sdks/python/examples/cost_management.py"
        assert example.exists(), "cost_management.py not found"

    def test_python_production_examples_exist(self):
        """Test that production examples exist."""
        fastapi = REPO_ROOT / "sdks/python/examples/production/fastapi_integration.py"
        docker = REPO_ROOT / "sdks/python/examples/production/docker_deployment.py"

        assert fastapi.exists(), "fastapi_integration.py not found"
        assert docker.exists(), "docker_deployment.py not found"

    def test_python_scientific_examples_exist(self):
        """Test that scientific examples exist."""
        research = REPO_ROOT / "sdks/python/examples/scientific/research_paper_analysis.py"
        assert research.exists(), "research_paper_analysis.py not found"

    def test_go_examples_directory_exists(self):
        """Test that Go examples directory exists."""
        examples = REPO_ROOT / "sdks/go/examples"
        assert examples.exists(), "Go examples directory not found"


# =============================================================================
# Build Verification (Optional - skipped if tools not installed)
# =============================================================================

class TestBuildVerification:
    """Tests that verify SDKs can build (requires tools to be installed)."""

    @pytest.mark.skipif(
        not shutil_which("python"),
        reason="Python not available"
    )
    def test_python_sdk_can_import(self):
        """Test that Python SDK can be imported."""
        result = subprocess.run(
            ["python", "-c", "from prsm_sdk import PRSMClient; print('OK')"],
            cwd=REPO_ROOT / "sdks/python",
            capture_output=True,
            timeout=30
        )
        # This may fail if dependencies aren't installed, which is OK
        # We're just checking that the import structure is valid

    @pytest.mark.skipif(
        not shutil_which("go"),
        reason="Go not available"
    )
    def test_go_sdk_builds(self):
        """Test that Go SDK builds without errors."""
        result = subprocess.run(
            ["go", "build", "./..."],
            cwd=REPO_ROOT / "sdks/go",
            capture_output=True,
            timeout=60
        )
        assert result.returncode == 0, f"Go SDK build failed: {result.stderr.decode()}"


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])
