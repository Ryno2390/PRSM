"""
Phase 4 deployment verification tests.

These tests verify the external deployment configuration is correct.
They do NOT hit live infrastructure — they validate config values,
file structure, and contract address formats.

For live integration testing, set PRSM_LIVE_TESTS=1 in the environment.
"""
import os
import json
import re
import socket
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Bootstrap config tests
# ---------------------------------------------------------------------------

def test_bootstrap_default_is_live_domain():
    """Default BOOTSTRAP_PRIMARY env var default matches live server domain."""
    from prsm.node.config import DEFAULT_BOOTSTRAP_NODES
    assert len(DEFAULT_BOOTSTRAP_NODES) >= 1
    primary = DEFAULT_BOOTSTRAP_NODES[0]
    assert "prsm-network.com" in primary, (
        f"Default bootstrap should use prsm-network.com, got: {primary}"
    )
    assert "8765" in primary, (
        f"Default bootstrap should use port 8765, got: {primary}"
    )


def test_fallback_bootstrap_nodes_defined():
    """Fallback bootstrap nodes reference EU and APAC regions."""
    from prsm.node.config import FALLBACK_BOOTSTRAP_NODES
    assert len(FALLBACK_BOOTSTRAP_NODES) >= 2
    combined = " ".join(FALLBACK_BOOTSTRAP_NODES)
    assert "eu" in combined.lower() or "ams" in combined.lower(), (
        "Expected an EU fallback bootstrap node"
    )
    assert "apac" in combined.lower() or "sgp" in combined.lower() or "asia" in combined.lower(), (
        "Expected an APAC fallback bootstrap node"
    )


def test_node_config_template_has_correct_bootstrap():
    """config/node_config.json.template uses the live bootstrap domain."""
    template_path = REPO_ROOT / "config" / "node_config.json.template"
    assert template_path.exists(), "config/node_config.json.template must exist"
    content = template_path.read_text()
    data = json.loads(content)
    bootstrap = data.get("bootstrap_nodes", [])
    assert len(bootstrap) >= 1
    assert any("prsm-network.com" in b for b in bootstrap), (
        f"Template bootstrap_nodes should reference prsm-network.com, got: {bootstrap}"
    )


def test_node_config_template_has_fallback_nodes():
    """config/node_config.json.template includes fallback bootstrap nodes."""
    template_path = REPO_ROOT / "config" / "node_config.json.template"
    content = template_path.read_text()
    data = json.loads(content)
    fallback = data.get("bootstrap_fallback_nodes", [])
    assert len(fallback) >= 2, (
        f"Template should have at least 2 fallback nodes, got: {len(fallback)}"
    )
    combined = " ".join(fallback)
    assert "eu" in combined.lower(), "Expected EU fallback in template"
    assert "apac" in combined.lower(), "Expected APAC fallback in template"


# ---------------------------------------------------------------------------
# Mainnet contract address tests
# ---------------------------------------------------------------------------

ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def test_sepolia_contract_address_format():
    """Known Sepolia FTNS contract address is a valid Ethereum address."""
    address = "0xd979c096BE297F4C3a85175774Bc38C22b95E6a4"
    assert ETH_ADDRESS_RE.match(address), f"Invalid address format: {address}"


def test_mainnet_contract_address_when_set():
    """If FTNS_TOKEN_ADDRESS_MAINNET env var is set, it must be a valid address."""
    mainnet_addr = os.getenv("FTNS_TOKEN_ADDRESS_MAINNET", "")
    if not mainnet_addr or mainnet_addr.startswith("<"):
        pytest.skip("Mainnet contract not yet deployed")
    assert ETH_ADDRESS_RE.match(mainnet_addr), (
        f"FTNS_TOKEN_ADDRESS_MAINNET is not a valid Ethereum address: {mainnet_addr}"
    )


def test_contracts_mainnet_json_when_exists():
    """If config/contracts_mainnet.json exists, it has required fields."""
    contracts_path = REPO_ROOT / "config" / "contracts_mainnet.json"
    if not contracts_path.exists():
        pytest.skip("contracts_mainnet.json not yet created")
    data = json.loads(contracts_path.read_text())
    assert "ftns_token" in data, "contracts_mainnet.json must have ftns_token field"
    assert "etherscan_url" in data, "contracts_mainnet.json must have etherscan_url"
    assert ETH_ADDRESS_RE.match(data["ftns_token"]), (
        f"ftns_token is not a valid Ethereum address: {data['ftns_token']}"
    )


# ---------------------------------------------------------------------------
# secure.env.template coverage tests
# ---------------------------------------------------------------------------

def test_secure_env_template_documents_bootstrap_vars():
    """secure.env.template documents bootstrap environment variables."""
    template_path = REPO_ROOT / "config" / "secure.env.template"
    content = template_path.read_text()
    assert "BOOTSTRAP_PRIMARY" in content, "Template must document BOOTSTRAP_PRIMARY"
    assert "BOOTSTRAP_FALLBACK_EU" in content, "Template must document BOOTSTRAP_FALLBACK_EU"
    assert "BOOTSTRAP_FALLBACK_APAC" in content, "Template must document BOOTSTRAP_FALLBACK_APAC"


def test_secure_env_template_documents_mainnet_vars():
    """secure.env.template documents mainnet deployment variables."""
    template_path = REPO_ROOT / "config" / "secure.env.template"
    content = template_path.read_text()
    assert "MAINNET_RPC_URL" in content, "Template must document MAINNET_RPC_URL"
    assert "ETHERSCAN_API_KEY" in content, "Template must document ETHERSCAN_API_KEY"
    assert "WEB3_NETWORK" in content, "Template must document WEB3_NETWORK"


def test_secure_env_template_documents_contract_addresses():
    """secure.env.template documents contract address placeholders."""
    template_path = REPO_ROOT / "config" / "secure.env.template"
    content = template_path.read_text()
    assert "FTNS_TOKEN_ADDRESS_SEPOLIA" in content, "Template must document FTNS_TOKEN_ADDRESS_SEPOLIA"
    assert "FTNS_TOKEN_ADDRESS_MAINNET" in content, "Template must document FTNS_TOKEN_ADDRESS_MAINNET"


# ---------------------------------------------------------------------------
# Live connectivity tests (require PRSM_LIVE_TESTS=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap1_resolves():
    """bootstrap1.prsm-network.com resolves to a valid IP."""
    ip = socket.gethostbyname("bootstrap1.prsm-network.com")
    assert ip and ip != "0.0.0.0", f"Expected valid IP, got: {ip}"


@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap1_port_open():
    """bootstrap1.prsm-network.com port 8765 is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        result = s.connect_ex(("bootstrap1.prsm-network.com", 8765))
    assert result == 0, f"Port 8765 is not open on bootstrap1 (connect_ex returned {result})"


@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap_eu_resolves():
    """bootstrap-eu.prsm-network.com resolves once deployed."""
    try:
        ip = socket.gethostbyname("bootstrap-eu.prsm-network.com")
        assert ip and ip != "0.0.0.0"
    except socket.gaierror:
        pytest.fail("bootstrap-eu.prsm-network.com does not resolve — deploy EU node first")


@pytest.mark.skipif(
    not os.getenv("PRSM_LIVE_TESTS"),
    reason="Set PRSM_LIVE_TESTS=1 to run live connectivity tests"
)
def test_bootstrap_apac_resolves():
    """bootstrap-apac.prsm-network.com resolves once deployed."""
    try:
        ip = socket.gethostbyname("bootstrap-apac.prsm-network.com")
        assert ip and ip != "0.0.0.0"
    except socket.gaierror:
        pytest.fail("bootstrap-apac.prsm-network.com does not resolve — deploy APAC node first")
