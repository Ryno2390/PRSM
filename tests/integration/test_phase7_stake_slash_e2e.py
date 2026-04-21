"""End-to-end integration test for Phase 7 stake-and-slash.

Boots a local Hardhat node, deploys the full Phase 7 stack
(MockERC20 + EscrowPool + MockSignatureVerifier + BatchSettlementRegistry
+ StakeBond, wired together), and drives a complete scenario end-to-end:

  1. Provider bonds FTNS at the critical tier (50K FTNS, 100% slash rate).
  2. StakeManagerClient reports the on-chain effectiveTier as "critical".
  3. Provider commits a batch against the registry (single-leaf tree,
     tierSlashRateBps snapshotted in).
  4. Challenger submits an INVALID_SIGNATURE challenge — mock verifier
     returns false → the challenge is proven → the registry fires
     stakeBond.slash(provider, challenger, batchId).
  5. Stake drops to 0 (100% slash at critical tier), slashedBountyPayable
     credits 70% to the challenger, foundationReserveBalance absorbs 30%.
  6. Challenger calls claimBounty via StakeManagerClient; their FTNS
     balance increases by the bounty amount.
  7. Python-side ReputationTracker.record_slash is fed from the Slashed
     event decoded off the challenge receipt; has_been_slashed flips.

Skipped automatically if `npx hardhat` is not on PATH or hardhat
node_modules are not installed.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

# conftest patches subprocess.Popen / subprocess.run / time.sleep for the
# session. We need the real versions for the hardhat lifecycle — capture
# them at import time before the autouse patches land.
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


@pytest.fixture(scope="module")
def hardhat_node():
    """Start a local hardhat node for the duration of the module.

    Port 8545 matches the hardhat config's default `localhost` network.
    Module-scoped so compile+deploy run once per test run. Do NOT run
    pytest with xdist against this module in parallel with other E2E
    modules — they all share :8545."""
    log_path = Path("/tmp/prsm_phase7_e2e_hardhat.log")
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
    """Compile + deploy the full Phase 7 stack via an inline JS script.

    Deployment order matters — Registry needs to know about StakeBond
    (via setStakeBond), and StakeBond needs to know about Registry (via
    setSlasher). We deploy both, then cross-wire.
    """
    deploy_js = """
const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const [deployer, provider, challenger, requester] =
    await hre.ethers.getSigners();

  // 1. Token.
  const Token = await hre.ethers.getContractFactory("MockERC20");
  const token = await Token.deploy();
  await token.waitForDeployment();

  // 2. EscrowPool (satellite — needed so Registry.setEscrowPool can be
  //    invoked; we don't actually use it in this test since the
  //    challenge path invalidates pre-finalization, but the Registry
  //    enforces the wire on finalize).
  const Pool = await hre.ethers.getContractFactory("EscrowPool");
  const pool = await Pool.deploy(
    deployer.address, await token.getAddress(), hre.ethers.ZeroAddress
  );
  await pool.waitForDeployment();

  // 3. Registry with a 1-hour challenge window (min allowed).
  const Registry = await hre.ethers.getContractFactory(
    "BatchSettlementRegistry"
  );
  const registry = await Registry.deploy(deployer.address, 3600);
  await registry.waitForDeployment();

  await pool.connect(deployer).setSettlementRegistry(await registry.getAddress());
  await registry.connect(deployer).setEscrowPool(await pool.getAddress());

  // 4. Mock signature verifier — defaults to returning false, which is
  //    what INVALID_SIGNATURE proof needs.
  const Verifier = await hre.ethers.getContractFactory(
    "MockSignatureVerifier"
  );
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();
  await registry.connect(deployer).setSignatureVerifier(
    await verifier.getAddress()
  );

  // 5. StakeBond with a 1-day unbond delay (min allowed) and the
  //    deployer as the Foundation reserve wallet.
  const Bond = await hre.ethers.getContractFactory("StakeBond");
  const bond = await Bond.deploy(
    deployer.address, await token.getAddress(), 86400
  );
  await bond.waitForDeployment();

  // Cross-wire.
  await bond.connect(deployer).setSlasher(await registry.getAddress());
  await bond.connect(deployer).setFoundationReserveWallet(deployer.address);
  await registry.connect(deployer).setStakeBond(await bond.getAddress());

  // 6. Mint FTNS to the provider so they can bond 50K.
  const ONE_FTNS = 10n ** 18n;
  await token.mint(provider.address, 100000n * ONE_FTNS);

  const out = {
    token: await token.getAddress(),
    pool: await pool.getAddress(),
    registry: await registry.getAddress(),
    verifier: await verifier.getAddress(),
    bond: await bond.getAddress(),
    deployer: deployer.address,
    provider: provider.address,
    challenger: challenger.address,
    requester: requester.address,
    // Hardhat default account private keys (well-known, testnet-only).
    deployerKey:
      "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    providerKey:
      "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    challengerKey:
      "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    requesterKey:
      "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
  };
  fs.writeFileSync("/tmp/prsm_phase7_e2e_deploy.json", JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""
    script = CONTRACTS_DIR / "scripts" / "_phase7_e2e_deploy.js"
    script.write_text(deploy_js)

    def _run(argv):
        p = _REAL_POPEN(argv, cwd=str(CONTRACTS_DIR))
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"command failed (rc={rc}): {' '.join(argv)}")

    try:
        _run(["npx", "hardhat", "compile"])
        _run([
            "npx", "hardhat", "run", "scripts/_phase7_e2e_deploy.js",
            "--network", "localhost",
        ])
        with open("/tmp/prsm_phase7_e2e_deploy.json") as f:
            return json.load(f)
    finally:
        if script.exists():
            script.unlink()


def _leaf_hash(leaf: dict) -> bytes:
    """Python mirror of BatchSettlementRegistry._hashLeaf.

    abi.encode the ReceiptLeaf tuple, keccak it. Must match the contract
    exactly, field order and types, or the Merkle proof verification
    will reject.
    """
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


def _encode_invalid_sig_aux(
    signing_msg: bytes, pubkey: bytes, signature: bytes,
) -> bytes:
    """Mirror _handleInvalidSignature's auxData layout:
       abi.encode(bytes, bytes, bytes) = (signingMessage, publicKey, signature)
    """
    from eth_abi import encode
    return encode(["bytes", "bytes", "bytes"], [signing_msg, pubkey, signature])


@requires_hardhat
def test_phase7_bond_slash_claim_rep_e2e(hardhat_node, deployed):
    """Full Phase 7 scenario across contracts + Python wrappers."""
    from eth_utils import keccak, to_checksum_address
    from web3 import Web3

    from prsm.economy.web3.stake_manager import (
        SLASH_RATE_CRITICAL_BPS,
        StakeManagerClient,
        StakeStatus,
        TIER_CRITICAL_MIN_WEI,
    )
    from prsm.marketplace.reputation import ReputationTracker

    w3 = Web3(Web3.HTTPProvider(hardhat_node))

    # ── 1. Provider approves + bonds at critical tier ──────────────────

    # First: approve StakeBond to pull the bond amount from provider FTNS.
    provider_acct = w3.eth.account.from_key(deployed["providerKey"])
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
    tx = token.functions.approve(
        to_checksum_address(deployed["bond"]), TIER_CRITICAL_MIN_WEI,
    ).build_transaction({
        "from": provider_acct.address,
        "nonce": w3.eth.get_transaction_count(provider_acct.address, "pending"),
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    })
    signed = w3.eth.account.sign_transaction(tx, provider_acct.key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    rcpt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    assert rcpt.status == 1, "approve tx reverted"

    # Now bond via the Python client.
    provider_sm = StakeManagerClient(
        rpc_url=hardhat_node,
        contract_address=deployed["bond"],
        private_key=deployed["providerKey"],
    )
    bond_tx, bond_status = provider_sm.bond(
        amount_wei=TIER_CRITICAL_MIN_WEI,
        tier_slash_rate_bps=SLASH_RATE_CRITICAL_BPS,
    )
    assert bond_status.value == "confirmed"
    assert bond_tx.startswith("0x")

    # ── 2. effective_tier reflects critical on-chain ───────────────────
    tier = provider_sm.effective_tier(deployed["provider"])
    assert tier == "critical", f"expected 'critical', got {tier!r}"

    stake_rec = provider_sm.stake_of(deployed["provider"])
    assert stake_rec.status == StakeStatus.BONDED
    assert stake_rec.amount_wei == TIER_CRITICAL_MIN_WEI
    assert stake_rec.tier_slash_rate_bps == SLASH_RATE_CRITICAL_BPS

    # ── 3. Provider commits a single-leaf batch ────────────────────────

    # Build the leaf. We'll use real-looking pubkey/signature bytes
    # because the INVALID_SIGNATURE path binds pubkey + signature hashes
    # from auxData back to the leaf-committed hashes.
    pubkey = b"phase7-e2e-pubkey"
    signature = b"phase7-e2e-signature"
    signing_msg = b"phase7-e2e-signing-message"

    leaf = {
        "jobIdHash": keccak(b"job-phase7-e2e"),
        "shardIndex": 0,
        "providerIdHash": keccak(b"provider-phase7-e2e"),
        "providerPubkeyHash": keccak(pubkey),
        "outputHash": keccak(b"output-phase7-e2e"),
        "executedAtUnix": int(time.time()),
        "valueFtns": 10 * 10**18,  # 10 FTNS per receipt
        "signatureHash": keccak(signature),
    }
    leaf_root = _leaf_hash(leaf)  # 1-leaf tree: root == leafHash

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

    commit_tx = registry.functions.commitBatch(
        to_checksum_address(deployed["requester"]),
        leaf_root,
        1,
        leaf["valueFtns"],
        SLASH_RATE_CRITICAL_BPS,
        b"\x00" * 32,   # consensus_group_id unused for this single-provider DOUBLE_SPEND test
        "ipfs://phase7-e2e",
    ).build_transaction({
        "from": provider_acct.address,
        "nonce": w3.eth.get_transaction_count(provider_acct.address, "pending"),
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    })
    signed = w3.eth.account.sign_transaction(commit_tx, provider_acct.key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    commit_rcpt = w3.eth.wait_for_transaction_receipt(
        w3.eth.send_raw_transaction(raw), timeout=60,
    )
    assert commit_rcpt.status == 1, "commitBatch reverted"

    batch_id = None
    for log in commit_rcpt.logs:
        try:
            decoded = registry.events.BatchCommitted().process_log(log)
            batch_id = decoded["args"]["batchId"]
            break
        except Exception:
            continue
    assert batch_id is not None, "BatchCommitted event not found"

    # ── 4. Challenger fires INVALID_SIGNATURE ──────────────────────────

    challenger_acct = w3.eth.account.from_key(deployed["challengerKey"])
    aux = _encode_invalid_sig_aux(signing_msg, pubkey, signature)
    # ReasonCode.INVALID_SIGNATURE == 1
    challenge_tx = registry.functions.challengeReceipt(
        batch_id, leaf, [], 1, aux,
    ).build_transaction({
        "from": challenger_acct.address,
        "nonce": w3.eth.get_transaction_count(
            challenger_acct.address, "pending"
        ),
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
        # Force a large gas budget. Default estimateGas only budgets for
        # the outer call; because challengeReceipt wraps stakeBond.slash
        # in try/catch, the estimator returns a value that lets the
        # slash() revert silently via OOG. Provide a budget large enough
        # for the full slash path (Merkle verify + sig check + stake
        # mutation + event emission).
        "gas": 1_000_000,
    })
    signed = w3.eth.account.sign_transaction(challenge_tx, challenger_acct.key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    challenge_rcpt = w3.eth.wait_for_transaction_receipt(
        w3.eth.send_raw_transaction(raw), timeout=60,
    )
    assert challenge_rcpt.status == 1, "challengeReceipt reverted"

    # ── 5. Stake drained, bounty + reserve accrued ─────────────────────

    post_stake = provider_sm.stake_of(deployed["provider"])
    assert post_stake.amount_wei == 0, (
        f"expected full 100% slash (critical tier), got amount={post_stake.amount_wei}"
    )

    total_slashed = TIER_CRITICAL_MIN_WEI
    expected_bounty = total_slashed * 7000 // 10000   # 70%
    expected_foundation = total_slashed - expected_bounty  # 30%

    bounty_payable = provider_sm.slashed_bounty_payable(deployed["challenger"])
    assert bounty_payable == expected_bounty, (
        f"bounty payable mismatch: expected {expected_bounty}, got {bounty_payable}"
    )
    reserve = provider_sm.foundation_reserve_balance()
    assert reserve == expected_foundation, (
        f"foundation reserve mismatch: expected {expected_foundation}, got {reserve}"
    )

    # ── 6. Challenger claims bounty; FTNS actually moves ───────────────

    challenger_sm = StakeManagerClient(
        rpc_url=hardhat_node,
        contract_address=deployed["bond"],
        private_key=deployed["challengerKey"],
    )
    pre_claim_balance = token.functions.balanceOf(
        to_checksum_address(deployed["challenger"])
    ).call()

    claim_tx, claim_status = challenger_sm.claim_bounty()
    assert claim_status.value == "confirmed"
    assert claim_tx.startswith("0x")

    post_claim_balance = token.functions.balanceOf(
        to_checksum_address(deployed["challenger"])
    ).call()
    assert post_claim_balance - pre_claim_balance == expected_bounty, (
        "challenger FTNS balance should have risen by the bounty amount"
    )
    # Bounty payable is zeroed after claim.
    assert provider_sm.slashed_bounty_payable(deployed["challenger"]) == 0

    # ── 7. ReputationTracker records the slash from the event ──────────

    # Decode the Slashed event from the challenge receipt's logs. We
    # model the event ABI inline here rather than loading from the
    # wrapper's STAKE_BOND_ABI to keep the test boundary explicit —
    # event decoding is the bridge between on-chain state and the
    # reputation layer, and we want to exercise it in isolation.
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
    assert slash_event["args"]["slashAmount"] == total_slashed
    assert slash_event["args"]["challengerBounty"] == expected_bounty
    assert slash_event["args"]["foundationShare"] == expected_foundation

    tracker = ReputationTracker()
    tracker.record_slash(
        provider_id=deployed["provider"],
        batch_id="0x" + batch_id.hex(),
        slash_amount_wei=int(slash_event["args"]["slashAmount"]),
        reason="INVALID_SIGNATURE",
        tx_hash=challenge_rcpt.transactionHash.hex(),
    )
    assert tracker.has_been_slashed(deployed["provider"]) is True
    assert tracker.slashed_count(deployed["provider"]) == 1

    events = tracker.get_slash_events(deployed["provider"])
    assert len(events) == 1
    assert events[0].reason == "INVALID_SIGNATURE"
    assert events[0].slash_amount_wei == total_slashed

    # Score check: a fresh provider with exactly one slash must drop
    # well below the neutral 0.5 cold-start shield. With SLASH_WEIGHT=100
    # and no successes, this is exactly 0.0.
    assert tracker.score_for(deployed["provider"]) == 0.0
