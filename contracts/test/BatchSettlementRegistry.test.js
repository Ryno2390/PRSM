const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("BatchSettlementRegistry", function () {
  let registry;
  let owner, provider, other, watchdog;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60; // 3 days

  // Helper: a non-zero Merkle root (doesn't need to be a real tree for Task 1).
  const DUMMY_ROOT = ethers.keccak256(ethers.toUtf8Bytes("test-merkle-root-1"));
  const DUMMY_ROOT_2 = ethers.keccak256(ethers.toUtf8Bytes("test-merkle-root-2"));

  beforeEach(async function () {
    [owner, provider, other, watchdog] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();
  });

  describe("constructor", function () {
    it("sets the challenge window and owner", async function () {
      expect(await registry.challengeWindowSeconds()).to.equal(DEFAULT_WINDOW);
      expect(await registry.owner()).to.equal(owner.address);
    });

    it("reverts when initial window is below minimum", async function () {
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      const tooSmall = 1; // 1 second, below MIN_CHALLENGE_WINDOW_SECONDS = 1 hour
      await expect(
        Registry.deploy(owner.address, tooSmall)
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });

    it("reverts when initial window exceeds maximum", async function () {
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      const tooLarge = 365 * 24 * 60 * 60; // 1 year, above MAX = 30 days
      await expect(
        Registry.deploy(owner.address, tooLarge)
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });
  });

  describe("commitBatch", function () {
    it("records a new batch and emits BatchCommitted", async function () {
      const receiptCount = 1000;
      const totalValue = ONE_FTNS * 500n;
      const metadataURI = "ipfs://QmFakeHashHere";

      const tx = await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, receiptCount, totalValue, metadataURI);
      const receipt = await tx.wait();

      // Extract batchId from the event.
      const event = receipt.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      );
      expect(event).to.not.be.undefined;
      const batchId = event.args[0];

      const batch = await registry.getBatch(batchId);
      expect(batch.provider).to.equal(provider.address);
      expect(batch.merkleRoot).to.equal(DUMMY_ROOT);
      expect(batch.receiptCount).to.equal(receiptCount);
      expect(batch.totalValueFTNS).to.equal(totalValue);
      expect(batch.invalidatedValueFTNS).to.equal(0);
      expect(batch.status).to.equal(1); // PENDING
      expect(batch.metadataURI).to.equal(metadataURI);
    });

    it("increments the provider's batch sequence per commit", async function () {
      expect(await registry.providerBatchSequence(provider.address)).to.equal(0);

      await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, 10, ONE_FTNS, "");
      expect(await registry.providerBatchSequence(provider.address)).to.equal(1);

      await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT_2, 10, ONE_FTNS, "");
      expect(await registry.providerBatchSequence(provider.address)).to.equal(2);
    });

    it("produces distinct batchIds for back-to-back commits with same root", async function () {
      // Tests that the per-provider sequence counter avoids collision
      // when a provider accidentally commits an identical Merkle root twice.
      const tx1 = await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, 100, ONE_FTNS, "");
      const r1 = await tx1.wait();
      const batchId1 = r1.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const tx2 = await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, 100, ONE_FTNS, "");
      const r2 = await tx2.wait();
      const batchId2 = r2.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      expect(batchId1).to.not.equal(batchId2);
    });

    it("reverts on zero Merkle root", async function () {
      await expect(
        registry
          .connect(provider)
          .commitBatch(ethers.ZeroHash, 10, ONE_FTNS, "")
      ).to.be.revertedWithCustomError(registry, "EmptyMerkleRoot");
    });

    it("reverts on zero receipt count", async function () {
      await expect(
        registry
          .connect(provider)
          .commitBatch(DUMMY_ROOT, 0, ONE_FTNS, "")
      ).to.be.revertedWithCustomError(registry, "ZeroReceiptCount");
    });

    it("accepts zero totalValueFTNS (pathological but not strictly invalid)", async function () {
      // A batch with 0 total value is weird but not inconsistent — e.g., a
      // batch where every receipt is free (some Tier-A test scenarios).
      // Reject would be a policy choice we don't make at Task 1.
      await expect(
        registry.connect(provider).commitBatch(DUMMY_ROOT, 1, 0, "")
      ).to.not.be.reverted;
    });
  });

  describe("finalizeBatch", function () {
    let batchId;
    beforeEach(async function () {
      const tx = await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, 100, ONE_FTNS * 50n, "ipfs://batch-meta");
      const receipt = await tx.wait();
      batchId = receipt.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
    });

    it("reverts when called before the challenge window elapses", async function () {
      // Default window = 3 days; no time advanced.
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(registry, "ChallengeWindowNotElapsed");
    });

    it("reverts when called shortly before window elapses", async function () {
      // Hardhat auto-advances block.timestamp by 1s per tx, so we use a
      // conservative margin (10s) to reliably land inside the window.
      await time.increase(DEFAULT_WINDOW - 10);
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(registry, "ChallengeWindowNotElapsed");
    });

    it("succeeds at exactly the window boundary", async function () {
      await time.increase(DEFAULT_WINDOW);
      await expect(registry.finalizeBatch(batchId))
        .to.emit(registry, "BatchFinalized")
        .withArgs(
          batchId,
          provider.address,
          ONE_FTNS * 50n,  // finalValue == totalValue since no invalidations
          0,               // invalidatedCount (Task 3 adds real values here)
          anyUint()        // finalizeTimestamp — checked via event shape, not exact
        );

      const batch = await registry.getBatch(batchId);
      expect(batch.status).to.equal(2); // FINALIZED
    });

    it("is callable by any address, not just the provider", async function () {
      await time.increase(DEFAULT_WINDOW);
      // Watchdog — a third party — finalizes on the provider's behalf.
      await registry.connect(watchdog).finalizeBatch(batchId);
      const batch = await registry.getBatch(batchId);
      expect(batch.status).to.equal(2); // FINALIZED
    });

    it("reverts on double-finalize", async function () {
      await time.increase(DEFAULT_WINDOW);
      await registry.finalizeBatch(batchId);
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(registry, "BatchNotPending");
    });

    it("reverts on unknown batchId", async function () {
      const fakeBatchId = ethers.keccak256(ethers.toUtf8Bytes("nonexistent"));
      await expect(
        registry.finalizeBatch(fakeBatchId)
      ).to.be.revertedWithCustomError(registry, "BatchNotFound");
    });
  });

  describe("governance: setChallengeWindowSeconds", function () {
    it("owner can update the window within bounds", async function () {
      const newWindow = 7 * 24 * 60 * 60; // 7 days
      await expect(
        registry.connect(owner).setChallengeWindowSeconds(newWindow)
      )
        .to.emit(registry, "ChallengeWindowUpdated")
        .withArgs(DEFAULT_WINDOW, newWindow);
      expect(await registry.challengeWindowSeconds()).to.equal(newWindow);
    });

    it("non-owner cannot update the window", async function () {
      await expect(
        registry.connect(other).setChallengeWindowSeconds(1 * 24 * 60 * 60)
      ).to.be.reverted; // Ownable reverts with OwnableUnauthorizedAccount
    });

    it("reverts when new window is below minimum", async function () {
      await expect(
        registry.connect(owner).setChallengeWindowSeconds(60) // 1 minute
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });

    it("reverts when new window exceeds maximum", async function () {
      await expect(
        registry.connect(owner).setChallengeWindowSeconds(60 * 24 * 60 * 60) // 60 days
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });

    it("shorter post-commit window lets older pending batches finalize sooner", async function () {
      // Commit a batch under the 3-day window.
      const tx = await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, 10, ONE_FTNS, "");
      const r = await tx.wait();
      const id = r.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      // Advance 1 day (well under 3-day window).
      await time.increase(1 * 24 * 60 * 60);

      // Owner shortens the window to 1 hour. The already-pending batch
      // is now finalize-eligible because elapsed (1 day) >= new window
      // (1 hour).
      await registry.connect(owner).setChallengeWindowSeconds(60 * 60);

      await expect(registry.finalizeBatch(id))
        .to.emit(registry, "BatchFinalized");
    });
  });

  describe("views", function () {
    let batchId;
    beforeEach(async function () {
      const tx = await registry
        .connect(provider)
        .commitBatch(DUMMY_ROOT, 100, ONE_FTNS * 10n, "");
      const r = await tx.wait();
      batchId = r.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
    });

    it("isFinalizable is false inside the window", async function () {
      expect(await registry.isFinalizable(batchId)).to.equal(false);
    });

    it("isFinalizable is true after the window elapses", async function () {
      await time.increase(DEFAULT_WINDOW);
      expect(await registry.isFinalizable(batchId)).to.equal(true);
    });

    it("isFinalizable is false after finalization", async function () {
      await time.increase(DEFAULT_WINDOW);
      await registry.finalizeBatch(batchId);
      expect(await registry.isFinalizable(batchId)).to.equal(false);
    });

    it("secondsUntilFinalizable decreases with time", async function () {
      const t0 = await registry.secondsUntilFinalizable(batchId);
      await time.increase(24 * 60 * 60); // 1 day
      const t1 = await registry.secondsUntilFinalizable(batchId);
      expect(t1).to.be.lessThan(t0);
    });

    it("secondsUntilFinalizable returns 0 after window elapses", async function () {
      await time.increase(DEFAULT_WINDOW);
      expect(await registry.secondsUntilFinalizable(batchId)).to.equal(0);
    });

    it("getBatch returns NONEXISTENT status for unknown id", async function () {
      const fakeId = ethers.keccak256(ethers.toUtf8Bytes("unknown"));
      const batch = await registry.getBatch(fakeId);
      expect(batch.status).to.equal(0); // NONEXISTENT
      expect(batch.provider).to.equal(ethers.ZeroAddress);
    });
  });
});

// Helper for event arg matching when we don't want to pin the exact value
function anyUint() {
  return (value) => typeof value === "bigint" || typeof value === "number";
}
