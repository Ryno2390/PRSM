"""End-to-end integration test for Phase 1 on-chain provenance.

Boots a local Hardhat node, deploys MockERC20 + ProvenanceRegistry +
RoyaltyDistributor, exercises the full Python client flow, and asserts
on-chain state matches expectations.

Skipped automatically if `npx hardhat` is not on PATH or hardhat
node_modules are not installed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

# IMPORTANT: tests/conftest.py applies a session-scoped autouse fixture that
# mocks subprocess.Popen and subprocess.run for the entire test session. We
# *do* need real processes here (real Hardhat node + real deploy scripts),
# so capture the real callables at module-import time, before the patch is
# applied to the subprocess module attributes.
from subprocess import Popen as _REAL_POPEN, run as _REAL_RUN  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"


def _hardhat_available() -> bool:
    return (
        shutil.which("npx") is not None
        and (CONTRACTS_DIR / "node_modules" / ".bin" / "hardhat").exists()
    )


pytestmark = pytest.mark.skipif(
    not _hardhat_available(), reason="Hardhat / npx not available"
)


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.3)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


@pytest.fixture(scope="module")
def hardhat_node():
    """Start a local hardhat node for the duration of the module."""
    # Capture stderr so we can surface a useful failure if hardhat dies.
    log_path = Path("/tmp/prsm_e2e_hardhat.log")
    log_fh = open(log_path, "wb")
    # shell=True to inherit the user's full PATH; some pytest setups strip
    # non-essential PATH entries from the subprocess environment, which
    # makes `npx hardhat` resolve to a stub that exits immediately.
    proc = _REAL_POPEN(
        "exec npx hardhat node --port 8545",
        cwd=str(CONTRACTS_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        shell=True,
        start_new_session=True,
    )
    try:
        # Poll for either port-open or process exit, whichever comes first.
        deadline = time.time() + 120.0
        port_open = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log_fh.close()
                tail = log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(
                    f"hardhat node exited early (rc={proc.returncode})\n{tail}"
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                try:
                    s.connect(("127.0.0.1", 8545))
                    port_open = True
                    break
                except OSError:
                    time.sleep(0.5)
        if not port_open:
            log_fh.close()
            tail = log_path.read_text(errors="replace")[-2000:]
            raise TimeoutError(
                f"hardhat node did not open port 8545 within 120s\n{tail}"
            )
        time.sleep(1.0)
        yield "http://127.0.0.1:8545"
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            try:
                log_fh.close()
            except Exception:
                pass


@pytest.fixture(scope="module")
def deployed(hardhat_node):
    """Compile + deploy the three contracts using a small inline JS deploy script."""
    deploy_js = """
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const [deployer, treasury] = await hre.ethers.getSigners();

  const Token = await hre.ethers.getContractFactory("MockERC20");
  const token = await Token.deploy();
  await token.waitForDeployment();

  const Registry = await hre.ethers.getContractFactory("ProvenanceRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();

  const Distributor = await hre.ethers.getContractFactory("RoyaltyDistributor");
  const distributor = await Distributor.deploy(
    await token.getAddress(),
    await registry.getAddress(),
    treasury.address
  );
  await distributor.waitForDeployment();

  // Mint some tokens to deployer for the test
  await token.mint(deployer.address, hre.ethers.parseUnits("10000", 18));

  const out = {
    token: await token.getAddress(),
    registry: await registry.getAddress(),
    distributor: await distributor.getAddress(),
    treasury: treasury.address,
    deployer: deployer.address,
    // Hardhat default account #0 private key (well-known, public testnet only):
    deployerKey: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
  };
  fs.writeFileSync("/tmp/prsm_e2e_deploy.json", JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""
    script = CONTRACTS_DIR / "scripts" / "_e2e_deploy.js"
    script.write_text(deploy_js)
    def _run(argv):
        """Run argv to completion using the unmocked Popen.
        Note: subprocess.run internally calls subprocess.Popen which is
        mocked in this session, so we use _REAL_POPEN + wait directly."""
        p = _REAL_POPEN(argv, cwd=str(CONTRACTS_DIR))
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"command failed (rc={rc}): {' '.join(argv)}")

    try:
        _run(["npx", "hardhat", "compile"])
        _run(
            [
                "npx",
                "hardhat",
                "run",
                "scripts/_e2e_deploy.js",
                "--network",
                "localhost",
            ]
        )
        with open("/tmp/prsm_e2e_deploy.json") as f:
            return json.load(f)
    finally:
        if script.exists():
            script.unlink()


def _content_hash(s: str) -> bytes:
    return hashlib.sha3_256(s.encode()).digest()


def test_register_then_distribute_e2e(hardhat_node, deployed):
    from prsm.economy.web3.provenance_registry import ProvenanceRegistryClient
    from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient

    registry_client = ProvenanceRegistryClient(
        rpc_url=hardhat_node,
        contract_address=deployed["registry"],
        private_key=deployed["deployerKey"],
    )

    distributor_client = RoyaltyDistributorClient(
        rpc_url=hardhat_node,
        distributor_address=deployed["distributor"],
        ftns_token_address=deployed["token"],
        private_key=deployed["deployerKey"],
    )

    content_hash = _content_hash("e2e-test-content")

    # 1. Register
    assert not registry_client.is_registered(content_hash)
    tx_register = registry_client.register_content(
        content_hash, royalty_rate_bps=800, metadata_uri="ipfs://e2e"
    )
    assert tx_register.startswith("0x")
    assert registry_client.is_registered(content_hash)

    record = registry_client.get_content(content_hash)
    assert record is not None
    assert record.royalty_rate_bps == 800
    assert record.metadata_uri == "ipfs://e2e"
    assert record.creator.lower() == deployed["deployer"].lower()

    # 2. Preview
    gross_wei = 100 * 10**18  # 100 FTNS
    preview = distributor_client.preview_split(content_hash, gross_wei)
    assert preview.creator_amount == 8 * 10**18
    assert preview.network_amount == 2 * 10**18
    assert preview.serving_node_amount == 90 * 10**18

    # 3. Distribute. The deployer is also payer; serving node = a fresh address.
    serving_node = "0x000000000000000000000000000000000000dEaD"
    tx_dist = distributor_client.distribute_royalty(
        content_hash, serving_node, gross_wei
    )
    assert tx_dist.startswith("0x")

    # 4. Verify balances on-chain via direct token contract reads
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(hardhat_node))
    erc20_abi = [
        {
            "inputs": [{"name": "owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    token_contract = w3.eth.contract(
        address=Web3.to_checksum_address(deployed["token"]), abi=erc20_abi
    )
    treasury_balance = token_contract.functions.balanceOf(
        Web3.to_checksum_address(deployed["treasury"])
    ).call()
    serving_balance = token_contract.functions.balanceOf(
        Web3.to_checksum_address(serving_node)
    ).call()
    creator_balance = token_contract.functions.balanceOf(
        Web3.to_checksum_address(deployed["deployer"])
    ).call()

    assert treasury_balance == 2 * 10**18
    assert serving_balance == 90 * 10**18
    # Creator started with 10_000 FTNS, paid 100 (gross), received 8 back.
    # Net: 10_000 - 100 + 8 = 9908
    assert creator_balance == (10_000 - 100 + 8) * 10**18
