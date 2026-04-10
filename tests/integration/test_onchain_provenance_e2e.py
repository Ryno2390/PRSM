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


# Phase 1.1: per-test skip instead of module-level skip so the local
# fallback regression test (which doesn't need Hardhat) still runs.
requires_hardhat = pytest.mark.skipif(
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


def _content_hash(creator_address: str, raw: bytes) -> bytes:
    """Phase 1.1: canonical creator-bound content hash."""
    from prsm.economy.web3.provenance_registry import compute_content_hash

    return compute_content_hash(creator_address, raw)


@requires_hardhat
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

    content_hash = _content_hash(deployed["deployer"], b"e2e-test-content")

    # 1. Register
    assert not registry_client.is_registered(content_hash)
    register_result = registry_client.register_content(
        content_hash, royalty_rate_bps=800, metadata_uri="ipfs://e2e"
    )
    # Phase 1.1 Task 4: returns (tx_hash_hex, TransferStatus)
    tx_register, register_status = register_result
    assert tx_register.startswith("0x")
    assert register_status.value == "confirmed"
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
    distribute_result = distributor_client.distribute_royalty(
        content_hash, serving_node, gross_wei
    )
    # Phase 1.1 Task 4: returns (tx_hash_hex, TransferStatus)
    tx_dist, distribute_status = distribute_result
    assert tx_dist.startswith("0x")
    assert distribute_status.value == "confirmed"

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


# ── Phase 1.1 Task 10: local fallback economics regression ───────────────
# This test does NOT need Hardhat (it constructs ContentEconomy with mocked
# dependencies), but it lives in the integration suite because it's the
# cross-cut regression for the Phase 1.1 P1 #6 fallback economics fix.


def test_local_fallback_pays_serving_node_share():
    """Without on-chain (no provenance_hash in metadata), the local Phase4
    path must produce the 8/2/90 split with the 90% remainder going to the
    serving node — not the direct creator. Phase 1.1 P1 #6 regression."""
    import asyncio
    from decimal import Decimal
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_economy import (
        ContentEconomy,
        ContentAccessPayment,
        PaymentStatus,
        RoyaltyModel,
    )

    identity = MagicMock()
    identity.node_id = "node-serving-xyz"
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    ledger.debit = AsyncMock()
    gossip = MagicMock()
    content_index = MagicMock()

    economy = ContentEconomy(
        identity=identity,
        ledger=ledger,
        gossip=gossip,
        content_index=content_index,
        ftns_ledger=None,  # no on-chain ledger → on-chain branch skipped
        royalty_model=RoyaltyModel.PHASE4,
    )
    economy._credit_royalty = AsyncMock()

    # Mock the provenance chain so we can assert deterministic shares.
    chain = MagicMock()
    chain.original_creator = "creator-original"
    chain.original_content_id = "cid-original"
    chain.derivative_creators = []
    economy._resolve_provenance_chain = AsyncMock(return_value=chain)

    payment = ContentAccessPayment(
        payment_id="p1",
        content_id="cid-test",
        accessor_id="accessor-x",
        creator_id="creator-direct",
        amount=Decimal("100"),
        royalty_model=RoyaltyModel.PHASE4,
        status=PaymentStatus.PENDING,
    )

    distributions = asyncio.run(
        economy._distribute_royalties(
            payment=payment,
            creator_id="creator-direct",
            parent_content_ids=[],
            content_metadata={},  # no provenance_hash → on-chain skipped
        )
    )

    by_type = {d["type"]: d for d in distributions}

    # 8% to original creator
    assert "original_creator" in by_type
    assert by_type["original_creator"]["amount"] == 8.0

    # 2% network fee
    assert "network_fee" in by_type
    assert by_type["network_fee"]["amount"] == 2.0

    # 90% to serving node — the regression assertion. Pre-Phase-1.1 this
    # was paid to "direct_creator" instead, so the serving node got 0.
    assert "serving_node" in by_type, (
        "local fallback should pay serving node — Phase 1.1 P1 #6 regression"
    )
    assert by_type["serving_node"]["recipient_id"] == "node-serving-xyz"
    assert by_type["serving_node"]["amount"] == 90.0

    # And explicitly: the old direct_creator type should be GONE.
    assert "direct_creator" not in by_type


# ── Phase 1.2 Task 2: malformed provenance_hash must not fail payment ────


def test_malformed_provenance_hash_falls_back_safely():
    """A malformed provenance_hash in metadata must NOT mark the payment
    FAILED. The on-chain branch should skip it (return None) and the
    local fallback should handle distribution. Phase 1.2 P2 regression."""
    import asyncio
    from decimal import Decimal
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_economy import (
        ContentEconomy,
        ContentAccessPayment,
        PaymentStatus,
        RoyaltyModel,
    )

    identity = MagicMock()
    identity.node_id = "node-serving-xyz"
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    ledger.debit = AsyncMock()
    gossip = MagicMock()
    content_index = MagicMock()

    # Pretend the on-chain stack is wired so _try_onchain_distribute is taken.
    fake_distributor = MagicMock()
    fake_ftns_ledger = MagicMock()
    fake_ftns_ledger._connected_address = (
        "0xdEAd000000000000000000000000000000000001"
    )

    economy = ContentEconomy(
        identity=identity,
        ledger=ledger,
        gossip=gossip,
        content_index=content_index,
        ftns_ledger=fake_ftns_ledger,
        royalty_model=RoyaltyModel.PHASE4,
    )
    economy._credit_royalty = AsyncMock()
    chain = MagicMock()
    chain.original_creator = "creator-original"
    chain.original_content_id = "cid-original"
    chain.derivative_creators = []
    economy._resolve_provenance_chain = AsyncMock(return_value=chain)
    economy._get_royalty_distributor = MagicMock(return_value=fake_distributor)

    payment = ContentAccessPayment(
        payment_id="p-bad-hash",
        content_id="cid-test",
        accessor_id="accessor-x",
        creator_id="creator-direct",
        amount=Decimal("100"),
        royalty_model=RoyaltyModel.PHASE4,
        status=PaymentStatus.PENDING,
    )

    distributions = asyncio.run(
        economy._distribute_royalties(
            payment=payment,
            creator_id="creator-direct",
            parent_content_ids=[],
            content_metadata={"provenance_hash": "GARBAGE_NOT_HEX"},
        )
    )

    # The on-chain branch must have skipped, and the local fallback must
    # have produced a valid distribution list (not raised).
    assert distributions
    assert any(d["type"] == "serving_node" for d in distributions)


# ── Phase 1.2 Task 3: broadcast_pending surfaces as PENDING_ONCHAIN ──────


def test_broadcast_pending_distribution_surfaces_as_pending_onchain_status():
    """When _try_onchain_distribute returns a 'broadcast_pending' stub
    (RPC lost receipt after broadcast), process_content_access must set
    payment.status to PENDING_ONCHAIN, NOT COMPLETED. The API uses this
    status to tell clients the payment needs reconciliation. Phase 1.2
    P2 regression."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_economy import (
        ContentEconomy,
        PaymentStatus,
        RoyaltyModel,
    )

    identity = MagicMock()
    identity.node_id = "node-serving-xyz"
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    ledger.debit = AsyncMock()
    gossip = MagicMock()
    content_index = MagicMock()

    economy = ContentEconomy(
        identity=identity,
        ledger=ledger,
        gossip=gossip,
        content_index=content_index,
        ftns_ledger=None,
        royalty_model=RoyaltyModel.PHASE4,
    )

    # Force _try_onchain_distribute to return a broadcast_pending stub.
    async def fake_try_onchain(payment, content_metadata):
        return [
            {
                "recipient_id": "onchain:in_flight",
                "amount": float(payment.amount),
                "type": "broadcast_pending",
                "tx_hash": "0x" + "ab" * 32,
            }
        ]

    economy._try_onchain_distribute = fake_try_onchain

    payment = asyncio.run(
        economy.process_content_access(
            content_id="cid-test",
            accessor_id="node-serving-xyz",
            content_metadata={"royalty_rate": 100, "creator_id": "c"},
        )
    )

    assert payment.status == PaymentStatus.PENDING_ONCHAIN, (
        f"expected PENDING_ONCHAIN, got {payment.status}"
    )
    assert any(
        d["type"] == "broadcast_pending" for d in payment.royalty_distributions
    )


# ── Phase 1.2 Task 4: ContentUploader computes/persists provenance_hash ─


def test_uploader_with_creator_address_computes_provenance_hash():
    """ContentUploader constructed with a creator_address must compute
    and persist the canonical provenance_hash. Phase 1.2 P1 #1 fix."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_uploader import ContentUploader

    identity = MagicMock()
    identity.node_id = "node-uploader-xyz"
    identity.public_key_b64 = "fakekey"
    identity.sign = MagicMock(return_value="sig")
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    ledger = MagicMock()

    creator_address = "0x1111111111111111111111111111111111111111"

    uploader = ContentUploader(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        creator_address=creator_address,
    )
    # Bypass actual IPFS + DB persistence.
    uploader._ipfs_add = AsyncMock(return_value="QmFakeCID123")
    uploader._persist_provenance = AsyncMock()

    file_bytes = b"hello world phase 1.2"
    uploaded = asyncio.run(
        uploader.upload(content=file_bytes, filename="test.txt")
    )

    assert uploaded is not None
    assert uploaded.provenance_hash is not None
    assert uploaded.provenance_hash.startswith("0x")
    assert len(uploaded.provenance_hash) == 66  # 0x + 64 hex chars

    # Matches what the CLI helper would produce.
    from prsm.economy.web3.provenance_registry import compute_content_hash
    expected = "0x" + compute_content_hash(creator_address, file_bytes).hex()
    assert uploaded.provenance_hash == expected

    # Verify the gossip CONTENT_ADVERTISE payload carries the hash so
    # other nodes can route their payments on-chain too.
    from prsm.node.gossip import GOSSIP_CONTENT_ADVERTISE
    ad_calls = [
        call for call in gossip.publish.call_args_list
        if call.args and call.args[0] == GOSSIP_CONTENT_ADVERTISE
    ]
    assert ad_calls, "expected a GOSSIP_CONTENT_ADVERTISE publish"
    advertised = ad_calls[0].args[1]
    assert advertised.get("provenance_hash") == expected


def test_uploader_without_creator_address_skips_provenance_hash():
    """ContentUploader without creator_address gets None — backward
    compatible. The on-chain branch then skips and local handles."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_uploader import ContentUploader

    identity = MagicMock()
    identity.node_id = "node-uploader-xyz"
    identity.public_key_b64 = "fakekey"
    identity.sign = MagicMock(return_value="sig")
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    ledger = MagicMock()

    uploader = ContentUploader(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        creator_address=None,
    )
    uploader._ipfs_add = AsyncMock(return_value="QmFakeCID456")
    uploader._persist_provenance = AsyncMock()

    uploaded = asyncio.run(
        uploader.upload(content=b"some bytes", filename="x.txt")
    )

    assert uploaded is not None
    assert uploaded.provenance_hash is None


# ── Phase 1.2 Task 5: ContentProvider forwards provenance_hash ──────────


def test_provider_forwards_provenance_hash_to_content_economy():
    """ContentProvider.register_local_content with a provenance_hash in
    metadata must forward it to ContentEconomy.process_content_access at
    serve time. Phase 1.2 P1 #1 fix (provider half).

    Uses a REAL ContentEconomy instance (not a MagicMock for
    process_content_access) so that any signature mismatch between the
    provider call site and the real method raises TypeError. This test
    exists because the original Task 5 test missed a pre-existing
    cid=/content_id= kwarg bug — a free-function mock swallowed it.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_provider import ContentProvider
    from prsm.node.content_economy import ContentEconomy, RoyaltyModel

    identity = MagicMock()
    identity.node_id = "node-provider-xyz"
    transport = MagicMock()
    transport.send_message = AsyncMock()
    gossip = MagicMock()
    ledger = MagicMock()
    ledger.debit = AsyncMock()
    content_index = MagicMock()

    # Real ContentEconomy instance — its process_content_access has the
    # real signature, so a bad kwarg (e.g. cid= instead of content_id=)
    # or a missing metadata key will blow up loudly.
    economy = ContentEconomy(
        identity=identity,
        ledger=ledger,
        gossip=gossip,
        content_index=content_index,
        ftns_ledger=None,
        royalty_model=RoyaltyModel.PHASE4,
    )

    # Stub the royalty distribution so we can capture the metadata
    # the provider forwarded without exercising the full royalty
    # pipeline. This is the one point we intercept; process_content_access
    # itself runs for real.
    captured: dict = {}

    async def capture_distribute(
        payment, creator_id, parent_content_ids, content_metadata
    ):
        captured["content_id"] = payment.content_id
        captured["creator_id"] = creator_id
        captured["parent_content_ids"] = parent_content_ids
        captured["content_metadata"] = content_metadata
        return []

    economy._distribute_royalties = capture_distribute

    provider = ContentProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        content_economy=economy,
    )

    expected_hash = "0x" + "ab" * 32
    provider.register_local_content(
        cid="cid-with-hash",
        size_bytes=100,
        content_hash="sha256-of-bytes",
        filename="x.txt",
        metadata={
            "creator_id": "creator-xyz",
            "royalty_rate": 0.01,
            "provenance_hash": expected_hash,
        },
    )

    asyncio.run(
        provider._fire_payment_for_test(
            cid="cid-with-hash",
            accessor_id="some-peer",
        )
    )

    assert captured["content_id"] == "cid-with-hash"
    assert captured["content_metadata"].get("provenance_hash") == expected_hash


# ── Phase 1.2 Task 9: gossip discovery preserves provenance_hash ────────


def test_content_announcement_roundtrips_provenance_hash():
    """ContentAnnouncement.to_gossip_data / from_gossip_data must round-trip
    provenance_hash. Phase 1.2 Task 9 — otherwise peers discovering
    content via gossip lose the hash and the on-chain branch in
    content_economy silently falls back to local."""
    from prsm.node.content_provider import ContentAnnouncement

    expected_hash = "0x" + "cd" * 32
    ann = ContentAnnouncement(
        cid="QmABC",
        size=1024,
        content_type="text/plain",
        content_hash="sha256-abc",
        provider_id="provider-1",
        filename="x.txt",
        metadata={"creator_id": "creator-z", "royalty_rate": 0.01},
        provenance_hash=expected_hash,
    )

    data = ann.to_gossip_data()
    assert data.get("provenance_hash") == expected_hash

    ann2 = ContentAnnouncement.from_gossip_data(data, origin="peer-1")
    assert ann2.provenance_hash == expected_hash


def test_content_index_preserves_provenance_hash_from_advertisement():
    """ContentIndex._on_content_advertise must copy provenance_hash from
    the gossip payload into the ContentRecord so downstream consumers
    (replication, serve-time lookup on another node) can find it.
    Phase 1.2 Task 9 regression."""
    import asyncio
    from unittest.mock import MagicMock

    from prsm.node.content_index import ContentIndex

    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    index = ContentIndex(gossip=gossip)

    expected_hash = "0x" + "ef" * 32
    payload = {
        "cid": "QmDiscovered",
        "filename": "d.txt",
        "size_bytes": 42,
        "content_hash": "sha256-d",
        "creator_id": "creator-q",
        "provider_id": "peer-42",
        "created_at": 1_700_000_000.0,
        "metadata": {},
        "royalty_rate": 0.01,
        "parent_cids": [],
        "provenance_hash": expected_hash,
    }
    asyncio.run(
        index._on_content_advertise(
            subtype="content_advertise", data=payload, origin="peer-42"
        )
    )

    record = index._records.get("QmDiscovered")
    assert record is not None
    assert record.provenance_hash == expected_hash

    # A second advertisement from a different peer missing the hash must
    # not clobber the populated value.
    payload2 = dict(payload)
    payload2["provider_id"] = "peer-99"
    payload2.pop("provenance_hash")
    asyncio.run(
        index._on_content_advertise(
            subtype="content_advertise", data=payload2, origin="peer-99"
        )
    )
    record = index._records["QmDiscovered"]
    assert record.provenance_hash == expected_hash
    assert "peer-99" in record.providers


def test_content_index_backfills_provenance_hash_on_later_advertisement():
    """If the first advertisement lacks provenance_hash but a later one
    carries it, the index should backfill the populated value. Phase 1.2
    Task 9 defense-in-depth."""
    import asyncio
    from unittest.mock import MagicMock

    from prsm.node.content_index import ContentIndex

    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    index = ContentIndex(gossip=gossip)

    first = {
        "cid": "QmLate",
        "filename": "late.txt",
        "size_bytes": 10,
        "content_hash": "sha256-late",
        "creator_id": "creator-x",
        "provider_id": "peer-1",
        "created_at": 1_700_000_000.0,
        "metadata": {},
        # no provenance_hash
    }
    asyncio.run(
        index._on_content_advertise(
            subtype="content_advertise", data=first, origin="peer-1"
        )
    )
    assert index._records["QmLate"].provenance_hash is None

    expected_hash = "0x" + "12" * 32
    second = dict(first)
    second["provider_id"] = "peer-2"
    second["provenance_hash"] = expected_hash
    asyncio.run(
        index._on_content_advertise(
            subtype="content_advertise", data=second, origin="peer-2"
        )
    )
    assert index._records["QmLate"].provenance_hash == expected_hash


# ── Phase 1.2 Task 10: retrieval flow handles PENDING_ONCHAIN ───────────


def test_request_content_retrieval_accepts_pending_onchain_payment(monkeypatch):
    """request_content_retrieval() must NOT treat PaymentStatus.PENDING_ONCHAIN
    as a hard failure. After Task 3 added the new status, the retrieval
    path still checked `status != COMPLETED` and aborted — discarding
    content even when the on-chain broadcast was in flight. Phase 1.2
    Task 10 regression from the second codex re-review."""
    import asyncio
    from decimal import Decimal
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node import content_economy as ce_mod
    from prsm.node.content_economy import (
        ContentEconomy,
        ContentAccessPayment,
        PaymentStatus,
        ProviderBid,
        RoyaltyModel,
    )

    identity = MagicMock()
    identity.node_id = "node-retriever"
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    ledger.debit = AsyncMock()
    gossip = MagicMock()
    gossip.publish = AsyncMock()
    content_index = MagicMock()

    economy = ContentEconomy(
        identity=identity,
        ledger=ledger,
        gossip=gossip,
        content_index=content_index,
        ftns_ledger=None,
        royalty_model=RoyaltyModel.PHASE4,
    )

    # Patch asyncio.sleep inside the content_economy module so the bid
    # collection wait returns instantly *and* injects a bid into the
    # most recent RetrievalRequest during the first sleep.
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        if economy._retrieval_requests:
            latest_req = next(reversed(economy._retrieval_requests.values()))
            if not latest_req.bids:
                latest_req.bids.append(
                    ProviderBid(
                        provider_id="peer-42",
                        content_id=latest_req.content_id,
                        price_ftns=Decimal("0.5"),
                        estimated_latency_ms=100.0,
                        available_bandwidth_mbps=100.0,
                        reputation_score=1.0,
                    )
                )
        await real_sleep(0)

    monkeypatch.setattr(ce_mod.asyncio, "sleep", fake_sleep)

    # Stub process_content_access to return PENDING_ONCHAIN so we
    # exercise the guard clause under test.
    async def fake_process(content_id, accessor_id, content_metadata):
        return ContentAccessPayment(
            payment_id="p-pending",
            content_id=content_id,
            accessor_id=accessor_id,
            creator_id=content_metadata.get("creator_id", ""),
            amount=Decimal("1"),
            royalty_model=RoyaltyModel.PHASE4,
            status=PaymentStatus.PENDING_ONCHAIN,
            royalty_distributions=[
                {
                    "recipient_id": "onchain:in_flight",
                    "amount": 1.0,
                    "type": "broadcast_pending",
                    "tx_hash": "0x" + "ab" * 32,
                }
            ],
        )

    economy.process_content_access = fake_process

    # The real request_content_retrieval returns None even for the
    # success path (content delivery is a separate step), so we can't
    # use the return value to distinguish in-flight-OK from payment-fail.
    # Instead we capture whether the guard logged an error.
    error_logs: list = []
    monkeypatch.setattr(
        ce_mod.logger,
        "error",
        lambda *args, **kwargs: error_logs.append((args, kwargs)),
    )

    asyncio.run(
        economy.request_content_retrieval(
            content_id="cid-needed",
            max_price_ftns=Decimal("1"),
            timeout=0.01,
        )
    )

    payment_failed_logs = [
        e for e in error_logs if "Payment failed for retrieval" in str(e[0])
    ]
    assert not payment_failed_logs, (
        "PENDING_ONCHAIN must not be treated as a hard payment failure; "
        f"got error logs: {payment_failed_logs}"
    )


def test_request_content_retrieval_forwards_provenance_hash_from_index(monkeypatch):
    """request_content_retrieval() must look up provenance_hash from the
    content_index and pass it into process_content_access, otherwise the
    marketplace retrieval path can never route through the on-chain
    RoyaltyDistributor. Phase 1.2 Task 10 second-pass regression."""
    import asyncio
    from decimal import Decimal
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node import content_economy as ce_mod
    from prsm.node.content_economy import (
        ContentEconomy,
        ContentAccessPayment,
        PaymentStatus,
        ProviderBid,
        RoyaltyModel,
    )

    identity = MagicMock()
    identity.node_id = "node-retriever"
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    ledger.debit = AsyncMock()
    gossip = MagicMock()
    gossip.publish = AsyncMock()

    expected_hash = "0x" + "99" * 32

    # content_index returns a record with provenance_hash set.
    content_index = MagicMock()
    fake_record = MagicMock()
    fake_record.provenance_hash = expected_hash
    fake_record.creator_id = "creator-from-index"
    fake_record.parent_cids = ["parent-1"]
    content_index.lookup = MagicMock(return_value=fake_record)

    economy = ContentEconomy(
        identity=identity,
        ledger=ledger,
        gossip=gossip,
        content_index=content_index,
        ftns_ledger=None,
        royalty_model=RoyaltyModel.PHASE4,
    )

    # Inject a bid during the first sleep so request_content_retrieval
    # progresses past bid collection.
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        if economy._retrieval_requests:
            latest_req = next(reversed(economy._retrieval_requests.values()))
            if not latest_req.bids:
                latest_req.bids.append(
                    ProviderBid(
                        provider_id="peer-42",
                        content_id=latest_req.content_id,
                        price_ftns=Decimal("0.5"),
                        estimated_latency_ms=100.0,
                        available_bandwidth_mbps=100.0,
                        reputation_score=1.0,
                    )
                )
        await real_sleep(0)

    monkeypatch.setattr(ce_mod.asyncio, "sleep", fake_sleep)

    # Capture the content_metadata that process_content_access receives.
    captured: dict = {}

    async def fake_process(content_id, accessor_id, content_metadata):
        captured["content_metadata"] = content_metadata
        return ContentAccessPayment(
            payment_id="p-ok",
            content_id=content_id,
            accessor_id=accessor_id,
            creator_id=content_metadata.get("creator_id", ""),
            amount=Decimal("1"),
            royalty_model=RoyaltyModel.PHASE4,
            status=PaymentStatus.COMPLETED,
        )

    economy.process_content_access = fake_process

    asyncio.run(
        economy.request_content_retrieval(
            content_id="cid-known-to-index",
            max_price_ftns=Decimal("1"),
            timeout=0.01,
        )
    )

    assert captured.get("content_metadata", {}).get("provenance_hash") == expected_hash
    assert captured["content_metadata"]["creator_id"] == "creator-from-index"
    assert captured["content_metadata"]["parent_content_ids"] == ["parent-1"]


def test_announce_content_helper_forwards_provenance_hash():
    """ContentProvider.announce_content() must forward a caller-supplied
    provenance_hash into the gossip payload. Phase 1.2 Task 10 second-pass
    regression — the helper previously dropped it."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from prsm.node.content_provider import ContentProvider

    identity = MagicMock()
    identity.node_id = "node-announcer"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.publish = AsyncMock()

    provider = ContentProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        content_economy=None,
    )

    expected_hash = "0x" + "77" * 32
    asyncio.run(
        provider.announce_content(
            cid="QmXYZ",
            size=1024,
            content_type="text/plain",
            content_hash="sha256-xyz",
            filename="file.txt",
            provenance_hash=expected_hash,
        )
    )

    assert gossip.publish.await_count == 1
    _topic, payload = gossip.publish.await_args.args
    assert payload.get("provenance_hash") == expected_hash
