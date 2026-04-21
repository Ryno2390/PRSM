"""End-to-end integration test for Phase 7.1 Task 7 — k-of-n consensus.

Boots a local Hardhat node, deploys the full Phase 7 stack, then drives
a complete k-of-n redundant-execution scenario end-to-end:

  1. Three providers each bond at the critical tier (100% slash rate).
  2. Provider A commits their single-receipt batch with output hash H1.
  3. Provider B commits their single-receipt batch with the same H1.
  4. Provider C commits their single-receipt batch with a DIFFERENT
     output hash H2 — same job, same shard, different result → cheater.
  5. Python-side ShardConsensus runs over the three receipts, partitions
     {A, B} as majority / {C} as minority.
  6. Requester challenges C's batch via CONSENSUS_MISMATCH, passing A's
     batch-id + proof + leaf as the conflicting-batch evidence.
  7. Slash fires against C (batch.provider): C's stake drops to 0;
     requester gets 70% bounty payable; Foundation gets 30%.
  8. Requester claims bounty via StakeManagerClient; FTNS moves.
  9. ReputationTracker.record_slash consumes the Slashed event; C's
     score is crushed; has_been_slashed flips True.

This is the companion to tests/integration/test_phase7_stake_slash_e2e.py
— same deploy harness, reuses its patterns. Skips automatically if
`npx hardhat` is not on PATH or hardhat node_modules aren't installed.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

# The session-level conftest mocks subprocess.Popen; capture real
# callable at import so the hardhat node + deploy subprocesses run.
from subprocess import Popen as _REAL_POPEN  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"


def _hardhat_available() -> bool:
    return (
        shutil.which("npx") is not None
        and (CONTRACTS_DIR / "node_modules" / ".bin" / "hardhat").exists()
    )


requires_hardhat = pytest.mark.skipif(
    not _hardhat_available(), reason="Hardhat / npx not available"
)


# ── Fixtures: hardhat node + deploy stack ────────────────────────────


@pytest.fixture(scope="module")
def hardhat_node():
    """Boot a local hardhat node for the module.

    Port 8545 matches hardhat.config.js's default `localhost` network.
    Module-scoped so compile+deploy run once per session. Do NOT run
    pytest in parallel against this and other Phase 7 E2E modules —
    they share :8545.
    """
    log_path = Path("/tmp/prsm_phase7_1_e2e_hardhat.log")
    log_fh = open(log_path, "wb")
    proc = _REAL_POPEN(
        "exec npx hardhat node --port 8545",
        cwd=str(CONTRACTS_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        shell=True,
        start_new_session=True,
    )
    try:
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
    """Deploy Phase 7 stack + mint FTNS to three providers.

    Same structure as tests/integration/test_phase7_stake_slash_e2e.py
    but funds three providers (A/B/C) instead of one — they're the
    k-of-n participants.
    """
    deploy_js = """
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const [deployer, provA, provB, provC, requester] =
    await hre.ethers.getSigners();

  const Token = await hre.ethers.getContractFactory("MockERC20");
  const token = await Token.deploy();
  await token.waitForDeployment();

  const Pool = await hre.ethers.getContractFactory("EscrowPool");
  const pool = await Pool.deploy(
    deployer.address, await token.getAddress(), hre.ethers.ZeroAddress,
  );
  await pool.waitForDeployment();

  const Registry = await hre.ethers.getContractFactory(
    "BatchSettlementRegistry",
  );
  const registry = await Registry.deploy(deployer.address, 3600);
  await registry.waitForDeployment();
  await pool.connect(deployer).setSettlementRegistry(await registry.getAddress());
  await registry.connect(deployer).setEscrowPool(await pool.getAddress());

  const Verifier = await hre.ethers.getContractFactory(
    "MockSignatureVerifier",
  );
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();
  await registry.connect(deployer).setSignatureVerifier(
    await verifier.getAddress(),
  );

  const Bond = await hre.ethers.getContractFactory("StakeBond");
  const bond = await Bond.deploy(
    deployer.address, await token.getAddress(), 86400,
  );
  await bond.waitForDeployment();
  await bond.connect(deployer).setSlasher(await registry.getAddress());
  await bond.connect(deployer).setFoundationReserveWallet(deployer.address);
  await registry.connect(deployer).setStakeBond(await bond.getAddress());

  // Mint 100K FTNS to each provider so they can bond 50K.
  const ONE_FTNS = 10n ** 18n;
  for (const p of [provA, provB, provC]) {
    await token.mint(p.address, 100000n * ONE_FTNS);
  }

  const out = {
    token: await token.getAddress(),
    pool: await pool.getAddress(),
    registry: await registry.getAddress(),
    verifier: await verifier.getAddress(),
    bond: await bond.getAddress(),
    deployer: deployer.address,
    provA: provA.address,
    provB: provB.address,
    provC: provC.address,
    requester: requester.address,
    // Hardhat default account private keys (well-known, testnet-only):
    deployerKey:
      "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    provAKey:
      "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    provBKey:
      "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    provCKey:
      "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    requesterKey:
      "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
  };
  fs.writeFileSync("/tmp/prsm_phase7_1_e2e_deploy.json", JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""
    script = CONTRACTS_DIR / "scripts" / "_phase7_1_e2e_deploy.js"
    script.write_text(deploy_js)

    def _run(argv):
        p = _REAL_POPEN(argv, cwd=str(CONTRACTS_DIR))
        if p.wait() != 0:
            raise RuntimeError(f"command failed: {' '.join(argv)}")

    try:
        _run(["npx", "hardhat", "compile"])
        _run([
            "npx", "hardhat", "run", "scripts/_phase7_1_e2e_deploy.js",
            "--network", "localhost",
        ])
        with open("/tmp/prsm_phase7_1_e2e_deploy.json") as f:
            return json.load(f)
    finally:
        if script.exists():
            script.unlink()


# ── Helpers ──────────────────────────────────────────────────────────


def _leaf_hash(leaf: dict) -> bytes:
    """Python mirror of BatchSettlementRegistry._hashLeaf — must match
    the contract's abi.encode(ReceiptLeaf) + keccak256 byte-for-byte."""
    from eth_abi import encode
    from eth_utils import keccak

    encoded = encode(
        [
            "(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)",
        ],
        [(
            leaf["jobIdHash"], leaf["shardIndex"], leaf["providerIdHash"],
            leaf["providerPubkeyHash"], leaf["outputHash"],
            leaf["executedAtUnix"], leaf["valueFtns"], leaf["signatureHash"],
        )],
    )
    return keccak(encoded)


def _encode_consensus_aux(
    conflicting_batch_id: bytes, majority_proof: list, majority_leaf: dict,
) -> bytes:
    """Mirror _handleConsensusMismatch's auxData layout:
       abi.encode(bytes32, bytes32[], ReceiptLeaf)
    """
    from eth_abi import encode
    return encode(
        [
            "bytes32",
            "bytes32[]",
            "(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)",
        ],
        [
            conflicting_batch_id,
            majority_proof,
            (
                majority_leaf["jobIdHash"], majority_leaf["shardIndex"],
                majority_leaf["providerIdHash"],
                majority_leaf["providerPubkeyHash"],
                majority_leaf["outputHash"],
                majority_leaf["executedAtUnix"],
                majority_leaf["valueFtns"], majority_leaf["signatureHash"],
            ),
        ],
    )


# ── The E2E ──────────────────────────────────────────────────────────


@requires_hardhat
def test_phase7_1_consensus_mismatch_e2e(hardhat_node, deployed):
    """Full k-of-n scenario across contracts + Python wrappers."""
    from eth_utils import keccak, to_checksum_address
    from web3 import Web3

    from prsm.compute.shard_consensus import ShardConsensus
    from prsm.compute.shard_receipt import ShardExecutionReceipt
    from prsm.economy.web3.stake_manager import (
        SLASH_RATE_CRITICAL_BPS,
        StakeManagerClient,
        TIER_CRITICAL_MIN_WEI,
        ReasonCode,
    )
    from prsm.marketplace.reputation import ReputationTracker
    from prsm.node.result_consensus import ConsensusMode

    w3 = Web3(Web3.HTTPProvider(hardhat_node))
    erc20_abi = [
        {
            "inputs": [
                {"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ],
            "name": "approve",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
    ]
    token = w3.eth.contract(
        address=to_checksum_address(deployed["token"]), abi=erc20_abi,
    )

    # ── 1. Three providers bond at critical tier ───────────────────────

    provider_accounts = {
        "A": (deployed["provA"], deployed["provAKey"]),
        "B": (deployed["provB"], deployed["provBKey"]),
        "C": (deployed["provC"], deployed["provCKey"]),
    }
    for label, (addr, key) in provider_accounts.items():
        acct = w3.eth.account.from_key(key)
        # approve StakeBond for the bond amount
        tx = token.functions.approve(
            to_checksum_address(deployed["bond"]), TIER_CRITICAL_MIN_WEI,
        ).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
        })
        signed = w3.eth.account.sign_transaction(tx, acct.key)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        rcpt = w3.eth.wait_for_transaction_receipt(
            w3.eth.send_raw_transaction(raw), timeout=60,
        )
        assert rcpt.status == 1, f"provider {label} approve failed"

        # bond
        sm = StakeManagerClient(
            rpc_url=hardhat_node,
            contract_address=deployed["bond"],
            private_key=key,
        )
        _, status = sm.bond(
            amount_wei=TIER_CRITICAL_MIN_WEI,
            tier_slash_rate_bps=SLASH_RATE_CRITICAL_BPS,
        )
        assert status.value == "confirmed"

    # ── 2. Build the three receipts (A/B match, C differs) ─────────────

    job_id = "job-phase7.1-e2e"
    shard_index = 0
    executed_at = int(time.time())
    agreed_hash_bytes = keccak(b"phase7.1-agreed-output")
    cheater_hash_bytes = keccak(b"phase7.1-cheater-output")

    # Python-side receipts: the wrappers use hex-string output_hash for
    # ShardExecutionReceipt.output_hash (matches Phase 2 format). The
    # contract-side leaf's outputHash is bytes32 — separate encoding.
    python_receipts = []
    contract_leaves = {}  # label → contract leaf dict
    for label, (addr, _key) in provider_accounts.items():
        out_hash_bytes = (
            agreed_hash_bytes if label in ("A", "B") else cheater_hash_bytes
        )
        python_receipts.append(ShardExecutionReceipt(
            job_id=job_id,
            shard_index=shard_index,
            provider_id=addr,
            provider_pubkey_b64=f"pubkey-{label}",
            output_hash=out_hash_bytes.hex(),
            executed_at_unix=executed_at,
            signature=f"sig-{label}",
        ))
        contract_leaves[label] = {
            "jobIdHash": keccak(job_id.encode()),
            "shardIndex": shard_index,
            "providerIdHash": keccak(addr.encode()),
            "providerPubkeyHash": keccak(f"pubkey-{label}".encode()),
            "outputHash": out_hash_bytes,
            "executedAtUnix": executed_at,
            "valueFtns": 10 * 10**18,
            "signatureHash": keccak(f"sig-{label}".encode()),
        }

    # ── 3. Python-side consensus resolves before on-chain work ─────────

    consensus = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    outcome = consensus.resolve(python_receipts)
    assert outcome.agreed is True
    assert outcome.agreed_output_hash == agreed_hash_bytes.hex()
    majority_providers = {r.provider_id for r in outcome.majority}
    minority_providers = {r.provider_id for r in outcome.minority}
    assert majority_providers == {deployed["provA"], deployed["provB"]}
    assert minority_providers == {deployed["provC"]}

    # ── 4. Each provider commits their own single-receipt batch ────────

    registry_abi_subset = [
        {
            "inputs": [
                {"name": "requester", "type": "address"},
                {"name": "merkleRoot", "type": "bytes32"},
                {"name": "receiptCount", "type": "uint256"},
                {"name": "totalValueFTNS", "type": "uint256"},
                {"name": "tierSlashRateBps", "type": "uint16"},
                {"name": "consensusGroupId", "type": "bytes32"},
                {"name": "metadataURI", "type": "string"},
            ],
            "name": "commitBatch",
            "outputs": [{"name": "batchId", "type": "bytes32"}],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "batchId", "type": "bytes32"},
                {
                    "components": [
                        {"name": "jobIdHash", "type": "bytes32"},
                        {"name": "shardIndex", "type": "uint32"},
                        {"name": "providerIdHash", "type": "bytes32"},
                        {"name": "providerPubkeyHash", "type": "bytes32"},
                        {"name": "outputHash", "type": "bytes32"},
                        {"name": "executedAtUnix", "type": "uint64"},
                        {"name": "valueFtns", "type": "uint128"},
                        {"name": "signatureHash", "type": "bytes32"},
                    ],
                    "name": "leaf",
                    "type": "tuple",
                },
                {"name": "merkleProof", "type": "bytes32[]"},
                {"name": "reason", "type": "uint8"},
                {"name": "auxData", "type": "bytes"},
            ],
            "name": "challengeReceipt",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "batchId", "type": "bytes32"},
                {"indexed": True, "name": "provider", "type": "address"},
                {"indexed": False, "name": "merkleRoot", "type": "bytes32"},
                {"indexed": False, "name": "receiptCount", "type": "uint256"},
                {"indexed": False, "name": "totalValueFTNS", "type": "uint256"},
                {"indexed": False, "name": "commitTimestamp", "type": "uint64"},
                {"indexed": False, "name": "metadataURI", "type": "string"},
            ],
            "name": "BatchCommitted",
            "type": "event",
        },
    ]
    registry = w3.eth.contract(
        address=to_checksum_address(deployed["registry"]),
        abi=registry_abi_subset,
    )

    # Phase 7.1x §8.7: all k providers in a consensus dispatch share
    # ONE group_id. The contract rejects CONSENSUS_MISMATCH challenges
    # unless both batches carry the same non-zero group_id.
    consensus_group_id = keccak(f"consensus-group-{job_id}".encode())

    batch_ids = {}  # label → batch_id
    for label, (addr, key) in provider_accounts.items():
        acct = w3.eth.account.from_key(key)
        leaf = contract_leaves[label]
        root = _leaf_hash(leaf)  # 1-leaf tree: root == leaf hash
        tx = registry.functions.commitBatch(
            to_checksum_address(deployed["requester"]),
            root,
            1,
            leaf["valueFtns"],
            SLASH_RATE_CRITICAL_BPS,
            consensus_group_id,
            f"ipfs://phase7.1-{label}",
        ).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
        })
        signed = w3.eth.account.sign_transaction(tx, acct.key)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        rcpt = w3.eth.wait_for_transaction_receipt(
            w3.eth.send_raw_transaction(raw), timeout=60,
        )
        assert rcpt.status == 1, f"provider {label} commit failed"
        batch_id = None
        for log in rcpt.logs:
            try:
                decoded = registry.events.BatchCommitted().process_log(log)
                batch_id = decoded["args"]["batchId"]
                break
            except Exception:
                continue
        assert batch_id is not None, f"no BatchCommitted event from {label}"
        batch_ids[label] = batch_id

    # ── 5. Requester submits CONSENSUS_MISMATCH via ConsensusChallengeSubmitter ─

    # Phase 7.1x: the submitter service is the production handoff point
    # for minority-receipt → on-chain challenge. The E2E exercises it
    # here to prove the auxData encoding + tx shape match the contract
    # byte-for-byte (unit tests use mocked web3; this proves the bytes
    # themselves are right).
    from prsm.marketplace.consensus_submitter import (
        ChallengeAttempt, ConsensusChallengeSubmitter, ReceiptLeafFields,
    )

    submitter = ConsensusChallengeSubmitter(
        rpc_url=hardhat_node,
        registry_address=deployed["registry"],
        private_key=deployed["requesterKey"],
    )

    # Convert the on-chain leaf dicts to ReceiptLeafFields for the
    # submitter API (normally the caller derives these from Python
    # ShardExecutionReceipt objects via from_python_receipt; here we
    # build directly because the E2E has the contract-leaf form at hand).
    def _fields(leaf: dict) -> ReceiptLeafFields:
        return ReceiptLeafFields(
            job_id_hash=leaf["jobIdHash"],
            shard_index=leaf["shardIndex"],
            provider_id_hash=leaf["providerIdHash"],
            provider_pubkey_hash=leaf["providerPubkeyHash"],
            output_hash=leaf["outputHash"],
            executed_at_unix=leaf["executedAtUnix"],
            value_ftns_wei=leaf["valueFtns"],
            signature_hash=leaf["signatureHash"],
        )

    attempt = ChallengeAttempt(
        minority_batch_id=batch_ids["C"],
        minority_leaf=_fields(contract_leaves["C"]),
        minority_proof=[],   # 1-leaf batch
        majority_batch_id=batch_ids["A"],
        majority_leaf=_fields(contract_leaves["A"]),
        majority_proof=[],   # 1-leaf batch
    )
    result = submitter.submit_one(attempt)
    assert result.success, (
        f"submitter failed: {result.error_type}: {result.error_message}"
    )
    assert result.tx_hash_hex.startswith("0x")

    # Look up the submitted tx's receipt so we can decode Slashed event below.
    challenge_rcpt = w3.eth.wait_for_transaction_receipt(
        bytes.fromhex(result.tx_hash_hex[2:]), timeout=60,
    )
    assert challenge_rcpt.status == 1, "submitter tx reverted"

    # ── 6. Assert slash landed on C, not on A or B ─────────────────────

    sm_requester = StakeManagerClient(
        rpc_url=hardhat_node, contract_address=deployed["bond"],
        private_key=deployed["requesterKey"],
    )
    stake_A = sm_requester.stake_of(deployed["provA"])
    stake_B = sm_requester.stake_of(deployed["provB"])
    stake_C = sm_requester.stake_of(deployed["provC"])
    assert stake_A.amount_wei == TIER_CRITICAL_MIN_WEI, "A should be untouched"
    assert stake_B.amount_wei == TIER_CRITICAL_MIN_WEI, "B should be untouched"
    assert stake_C.amount_wei == 0, "C should be fully slashed at critical tier"

    expected_bounty = TIER_CRITICAL_MIN_WEI * 7000 // 10000  # 70%
    expected_foundation = TIER_CRITICAL_MIN_WEI - expected_bounty  # 30%
    assert sm_requester.slashed_bounty_payable(deployed["requester"]) == expected_bounty
    assert sm_requester.foundation_reserve_balance() == expected_foundation

    # ── 7. Requester claims bounty; FTNS actually moves ────────────────

    pre_balance = token.functions.balanceOf(
        to_checksum_address(deployed["requester"])
    ).call()
    _, status = sm_requester.claim_bounty()
    assert status.value == "confirmed"
    post_balance = token.functions.balanceOf(
        to_checksum_address(deployed["requester"])
    ).call()
    assert post_balance - pre_balance == expected_bounty
    # payable is zeroed on claim
    assert sm_requester.slashed_bounty_payable(deployed["requester"]) == 0

    # ── 8. Reputation consumes the Slashed event ───────────────────────

    slashed_event_abi = {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "provider", "type": "address"},
            {"indexed": True, "name": "challenger", "type": "address"},
            {"indexed": True, "name": "reasonId", "type": "bytes32"},
            {"indexed": False, "name": "slashAmount", "type": "uint256"},
            {"indexed": False, "name": "challengerBounty", "type": "uint256"},
            {"indexed": False, "name": "foundationShare", "type": "uint256"},
        ],
        "name": "Slashed",
        "type": "event",
    }
    bond_contract = w3.eth.contract(
        address=to_checksum_address(deployed["bond"]),
        abi=[slashed_event_abi],
    )
    slash_event = None
    for log in challenge_rcpt.logs:
        if log.address.lower() != deployed["bond"].lower():
            continue
        try:
            slash_event = bond_contract.events.Slashed().process_log(log)
            break
        except Exception:
            continue
    assert slash_event is not None, "Slashed event not emitted"
    assert slash_event["args"]["provider"].lower() == deployed["provC"].lower()
    assert slash_event["args"]["slashAmount"] == TIER_CRITICAL_MIN_WEI

    # Feed into reputation tracker (the shape a production challenge-
    # submitter would implement).
    tracker = ReputationTracker()
    tracker.record_slash(
        provider_id=deployed["provC"],
        batch_id="0x" + batch_ids["C"].hex(),
        slash_amount_wei=int(slash_event["args"]["slashAmount"]),
        reason=ReasonCode.CONSENSUS_MISMATCH.name,
        tx_hash=challenge_rcpt.transactionHash.hex(),
    )
    assert tracker.has_been_slashed(deployed["provC"]) is True
    assert tracker.slashed_count(deployed["provC"]) == 1
    events = tracker.get_slash_events(deployed["provC"])
    assert events[0].reason == "CONSENSUS_MISMATCH"
    assert events[0].slash_amount_wei == TIER_CRITICAL_MIN_WEI
    # Fresh cheater: score crushed to 0.0 (no cold-start shield).
    assert tracker.score_for(deployed["provC"]) == 0.0
    # Majority providers have no slash history.
    assert tracker.has_been_slashed(deployed["provA"]) is False
    assert tracker.has_been_slashed(deployed["provB"]) is False
