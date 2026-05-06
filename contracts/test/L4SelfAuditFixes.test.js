const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// L4 self-audit (2026-05-06) regression tests.
// Covers HIGH-1 (A-01 ≡ D-01) and HIGH-2 (B-01) fixes.
//
// HIGH-1: `StakeBond.requestUnbond` reads the LIVE challengeWindow from
//   BSR but per-batch snapshots pin the actual challenge period. A
//   subsequent setChallengeWindowSeconds reduction reopens the
//   slash-evasion race. Fix: BSR.lastPendingBatchExpiry tracker that
//   StakeBond reads via ISlasherWithProviderExpiry.
//
// HIGH-2: Owner pause() during a batch's challenge window irrecoverably
//   consumes that window because the wall-clock elapsed >= snapshot.
//   Fix: pause-time accumulator subtracted from effective elapsed.

describe("L4 Self-Audit Fixes (2026-05-06)", function () {
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60;          // 3 days
  const MAX_WINDOW = 30 * 24 * 60 * 60;             // 30 days
  const MIN_WINDOW = 60 * 60;                       // 1 hour
  const MIN_UNBOND_DELAY = 24 * 60 * 60;            // 1 day
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000;

  describe("HIGH-1 (A-01 ≡ D-01) — `lastPendingBatchExpiry` tracker", function () {
    let registry, bond, token;
    let owner, provider;

    beforeEach(async function () {
      [owner, provider] = await ethers.getSigners();

      const Token = await ethers.getContractFactory("MockERC20");
      token = await Token.deploy();
      await token.waitForDeployment();

      // Deploy BSR first, then StakeBond pointing at it as slasher.
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      registry = await Registry.deploy(owner.address, MAX_WINDOW);
      await registry.waitForDeployment();

      const Bond = await ethers.getContractFactory("StakeBond");
      bond = await Bond.deploy(
        owner.address,
        await token.getAddress(),
        MIN_UNBOND_DELAY,
        await registry.getAddress()
      );
      await bond.waitForDeployment();

      // Fund + approve provider for stake.
      await token.mint(provider.address, PREMIUM_STAKE);
      await token.connect(provider).approve(await bond.getAddress(), PREMIUM_STAKE);
    });

    it("commitBatch updates lastPendingBatchExpiry to commit + window", async function () {
      // Pre-commit: tracker is zero.
      expect(await registry.lastPendingBatchExpiry(provider.address)).to.equal(0);

      const tx = await registry.connect(provider).commitBatch(
        owner.address,                                            // requester
        ethers.keccak256(ethers.toUtf8Bytes("merkle-root-1")),   // merkleRoot
        1,                                                        // receiptCount
        ONE_FTNS,                                                 // totalValueFTNS
        PREMIUM_SLASH_BPS,                                        // tierSlashRateBps
        ethers.ZeroHash,                                          // consensusGroupId
        "ipfs://test"                                             // metadataURI
      );
      const receipt = await tx.wait();
      const block = await ethers.provider.getBlock(receipt.blockNumber);

      // Tracker now equals commitTimestamp + window (= MAX_WINDOW).
      const expected = BigInt(block.timestamp) + BigInt(MAX_WINDOW);
      expect(await registry.lastPendingBatchExpiry(provider.address)).to.equal(expected);
    });

    it("monotonic update: second commit with later expiry overwrites; earlier expiry does NOT", async function () {
      // Commit at T0 with MAX_WINDOW = 30 days. Tracker = T0 + 30d.
      await registry.connect(provider).commitBatch(
        owner.address,
        ethers.keccak256(ethers.toUtf8Bytes("merkle-root-1")),
        1, ONE_FTNS, PREMIUM_SLASH_BPS, ethers.ZeroHash, "ipfs://1"
      );
      const after1 = await registry.lastPendingBatchExpiry(provider.address);

      // Owner reduces window to MIN_WINDOW = 1 hour.
      await registry.connect(owner).setChallengeWindowSeconds(MIN_WINDOW);

      // Commit again. New batch's expiry = (T0+ε) + 1h, which is
      // EARLIER than after1 (T0 + 30d). Tracker must NOT decrease.
      await registry.connect(provider).commitBatch(
        owner.address,
        ethers.keccak256(ethers.toUtf8Bytes("merkle-root-2")),
        1, ONE_FTNS, PREMIUM_SLASH_BPS, ethers.ZeroHash, "ipfs://2"
      );
      const after2 = await registry.lastPendingBatchExpiry(provider.address);
      expect(after2).to.equal(after1); // unchanged — original max wins
    });

    it("HIGH-1 vector: requestUnbond after window-reduction respects per-batch snapshot", async function () {
      // 1. Commit batch under MAX_WINDOW (30 days). Snapshot = 30d.
      const commitTx = await registry.connect(provider).commitBatch(
        owner.address,
        ethers.keccak256(ethers.toUtf8Bytes("merkle-root-1")),
        1, ONE_FTNS, PREMIUM_SLASH_BPS, ethers.ZeroHash, "ipfs://test"
      );
      const commitReceipt = await commitTx.wait();
      const commitBlock = await ethers.provider.getBlock(commitReceipt.blockNumber);
      const expectedFloor = BigInt(commitBlock.timestamp) + BigInt(MAX_WINDOW);

      // 2. Provider bonds.
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);

      // 3. Owner reduces live window to MIN_WINDOW (1 hour). The
      //    per-batch snapshot for the existing PENDING batch is still 30d.
      await registry.connect(owner).setChallengeWindowSeconds(MIN_WINDOW);
      expect(await registry.challengeWindowSeconds()).to.equal(MIN_WINDOW);

      // 4. Provider requests unbond. The HIGH-1 fix should clamp
      //    unbond_eligible_at to AT LEAST the per-batch snapshot
      //    expiry (commit + 30 days), not the new live window.
      await bond.connect(provider).requestUnbond();
      const stake = await bond.stakeOf(provider.address);

      // Pre-fix (LIVE-window-only): unbond_eligible_at would be
      //   max(now + 1d, now + 1h) ≈ now + 1d, which is FAR less than
      //   the original commit + 30d.
      // Post-fix: unbond_eligible_at >= commit + 30d.
      expect(BigInt(stake.unbond_eligible_at)).to.be.gte(expectedFloor);
    });

    it("stale tracker (past timestamp) does not penalise honest providers", async function () {
      // Provider commits, then we fast-forward past the window. The
      // tracker is now stale (in the past). A subsequent requestUnbond
      // should fall through to the local floor — not be artificially
      // delayed by the stale value.
      await registry.connect(provider).commitBatch(
        owner.address,
        ethers.keccak256(ethers.toUtf8Bytes("merkle-root-1")),
        1, ONE_FTNS, PREMIUM_SLASH_BPS, ethers.ZeroHash, "ipfs://test"
      );

      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);

      // Fast-forward past MAX_WINDOW + buffer so the tracker is stale.
      await time.increase(MAX_WINDOW + 60);

      // Reduce live window to MIN so liveWindow is also small.
      await registry.connect(owner).setChallengeWindowSeconds(MIN_WINDOW);

      await bond.connect(provider).requestUnbond();
      const stake = await bond.stakeOf(provider.address);
      const now = BigInt((await ethers.provider.getBlock("latest")).timestamp);

      // unbond_eligible_at should be ~now + max(unbondDelay, MIN_WINDOW)
      // = now + 1 day. NOT inflated by the stale tracker value.
      // Allow ±60 sec tolerance for block timing.
      const expectedAt = now + BigInt(MIN_UNBOND_DELAY);
      const stakeAt = BigInt(stake.unbond_eligible_at);
      expect(stakeAt).to.be.gte(expectedAt - 60n);
      expect(stakeAt).to.be.lte(expectedAt + 60n);
    });
  });

  describe("HIGH-2 (B-01) — pause-time accumulator", function () {
    let registry;
    let owner, provider, requester;

    beforeEach(async function () {
      [owner, provider, requester] = await ethers.getSigners();

      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
      await registry.waitForDeployment();
    });

    it("totalPausedSeconds increments only on _unpause and reflects duration", async function () {
      expect(await registry.totalPausedSeconds()).to.equal(0);

      await registry.connect(owner).pause();
      // While paused, accumulator is unchanged (only updates on unpause).
      expect(await registry.totalPausedSeconds()).to.equal(0);

      await time.increase(1000);
      await registry.connect(owner).unpause();

      // After unpause, totalPausedSeconds reflects ~1000 seconds.
      // Allow ±2 sec tolerance for block-mining edge effects.
      const total = await registry.totalPausedSeconds();
      expect(total).to.be.gte(998n);
      expect(total).to.be.lte(1003n);
    });

    it("HIGH-2 vector: sustained pause does NOT consume challenge window", async function () {
      // 1. Commit batch at T=0. Window = 3 days.
      await registry.connect(provider).commitBatch(
        requester.address,
        ethers.keccak256(ethers.toUtf8Bytes("merkle-root-1")),
        1, ONE_FTNS, PREMIUM_SLASH_BPS, ethers.ZeroHash, "ipfs://test"
      );

      // 2. Advance T+1h. isFinalizable should be false (still 3 days - 1h to go).
      await time.increase(60 * 60);
      expect(await registry.isFinalizable.staticCallResult).to.exist; // sanity

      // 3. Pause at T+1h.
      await registry.connect(owner).pause();

      // 4. Advance through the entire original window: T+3d+1m total.
      //    At this wall-clock elapsed >= challengeWindow, but we're paused.
      await time.increase(3 * 24 * 60 * 60); // +3 days while paused

      // 5. Unpause. Pause duration was ~3 days.
      await registry.connect(owner).unpause();

      // 6. Wall-clock elapsed since commit is now ~3d+1h. Naively this
      //    would mean window-elapsed = true. With the HIGH-2 fix, the
      //    pause duration is subtracted: effective elapsed = 1h.
      //    Window = 3d. So isFinalizable should still be FALSE.
      const batchId = await getProviderBatchId(registry, provider.address, 0);
      expect(await registry.isFinalizable(batchId)).to.equal(false);

      // 7. secondsUntilFinalizable should be approx (3 days - 1 hour).
      const remaining = await registry.secondsUntilFinalizable(batchId);
      expect(remaining).to.be.gte(BigInt(3 * 24 * 60 * 60 - 60 * 60 - 60));
      expect(remaining).to.be.lte(BigInt(3 * 24 * 60 * 60 - 60 * 60 + 60));
    });

    it("batches committed AFTER a pause do NOT see pre-commit pause time", async function () {
      // Pause + unpause to accumulate some paused time.
      await registry.connect(owner).pause();
      await time.increase(1000);
      await registry.connect(owner).unpause();
      const totalAfter = await registry.totalPausedSeconds();
      expect(totalAfter).to.be.gte(998n);

      // New batch committed AFTER the pause. Its
      // totalPausedAtBatchOrigin snapshots the post-pause accumulator.
      await registry.connect(provider).commitBatch(
        requester.address,
        ethers.keccak256(ethers.toUtf8Bytes("post-pause-root")),
        1, ONE_FTNS, PREMIUM_SLASH_BPS, ethers.ZeroHash, "ipfs://post"
      );
      const batchId = await getProviderBatchId(registry, provider.address, 0);
      const batch = await registry.batches(batchId);

      // Snapshot equals the current (post-unpause) accumulator value.
      expect(batch.totalPausedAtBatchOrigin).to.equal(totalAfter);

      // Verify isFinalizable behaves normally for this batch — no
      // pre-commit pause "credit" applied. Advance ~3 days, no pauses.
      await time.increase(3 * 24 * 60 * 60 + 1);
      expect(await registry.isFinalizable(batchId)).to.equal(true);
    });
  });

  // Helper: derive batchId from the provider's sequence-N commit.
  // Replicates the contract's keccak256(abi.encode(...)) layout.
  async function getProviderBatchId(registry, provider, sequence) {
    // Easier path: read the BatchCommitted event from the latest block.
    const filter = registry.filters.BatchCommitted(null, provider);
    const events = await registry.queryFilter(filter);
    return events[sequence].args.batchId;
  }
});
