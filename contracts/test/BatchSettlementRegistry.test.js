const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 3.1 Task 1 + Task 2. Integration-style tests: BatchSettlementRegistry
// is exercised alongside a real EscrowPool + MockERC20 so finalizeBatch
// actually moves FTNS.

describe("BatchSettlementRegistry", function () {
  let registry, escrowPool, token;
  let owner, provider, requester, other, watchdog;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60; // 3 days
  const INITIAL_ESCROW = ONE_FTNS * 1000n; // requester deposits 1000 FTNS

  const DUMMY_ROOT = ethers.keccak256(ethers.toUtf8Bytes("test-merkle-root-1"));
  const DUMMY_ROOT_2 = ethers.keccak256(ethers.toUtf8Bytes("test-merkle-root-2"));

  beforeEach(async function () {
    [owner, provider, requester, other, watchdog] = await ethers.getSigners();

    // Deploy MockERC20 as stand-in for FTNS.
    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    // Post-HIGH-6 deploy ordering: Registry FIRST (because EscrowPool's
    // settlementRegistry is now immutable), then EscrowPool with the
    // registry address baked in. Registry's reciprocal pointer is set
    // after via setEscrowPool — that direction remains mutable.
    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(
      owner.address,
      await token.getAddress(),
      await registry.getAddress()
    );
    await escrowPool.waitForDeployment();

    // Wire the registry's pointer back at the pool.
    await registry
      .connect(owner)
      .setEscrowPool(await escrowPool.getAddress());

    // Fund requester with FTNS + deposit into escrow pool so settlements
    // can pull from it.
    await token.mint(requester.address, INITIAL_ESCROW);
    await token
      .connect(requester)
      .approve(await escrowPool.getAddress(), INITIAL_ESCROW);
    await escrowPool.connect(requester).deposit(INITIAL_ESCROW);
  });

  describe("constructor", function () {
    it("sets the challenge window and owner", async function () {
      expect(await registry.challengeWindowSeconds()).to.equal(DEFAULT_WINDOW);
      expect(await registry.owner()).to.equal(owner.address);
    });

    it("reverts when initial window is below minimum", async function () {
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      await expect(
        Registry.deploy(owner.address, 1)
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });

    it("reverts when initial window exceeds maximum", async function () {
      const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
      const tooLarge = 365 * 24 * 60 * 60;
      await expect(
        Registry.deploy(owner.address, tooLarge)
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });
  });

  describe("commitBatch", function () {
    it("records a new batch and emits BatchCommitted", async function () {
      const receiptCount = 1000;
      const totalValue = ONE_FTNS * 50n;
      const metadataURI = "ipfs://QmFakeHashHere";

      const tx = await registry
        .connect(provider)
        .commitBatch(
          requester.address, DUMMY_ROOT, receiptCount, totalValue, 0, ethers.ZeroHash, metadataURI
        );
      const r = await tx.wait();
      const batchId = r.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      const batch = await registry.getBatch(batchId);
      expect(batch.provider).to.equal(provider.address);
      expect(batch.requester).to.equal(requester.address);
      expect(batch.merkleRoot).to.equal(DUMMY_ROOT);
      expect(batch.receiptCount).to.equal(receiptCount);
      expect(batch.totalValueFTNS).to.equal(totalValue);
      expect(batch.invalidatedValueFTNS).to.equal(0);
      expect(batch.status).to.equal(1); // PENDING
      expect(batch.metadataURI).to.equal(metadataURI);
    });

    it("reverts on zero requester address", async function () {
      await expect(
        registry
          .connect(provider)
          .commitBatch(ethers.ZeroAddress, DUMMY_ROOT, 10, ONE_FTNS, 0, ethers.ZeroHash, "")
      ).to.be.revertedWithCustomError(registry, "ZeroRequester");
    });

    it("increments the provider's batch sequence per commit", async function () {
      expect(await registry.providerBatchSequence(provider.address)).to.equal(0);
      await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 10, ONE_FTNS,  0, ethers.ZeroHash, "");
      expect(await registry.providerBatchSequence(provider.address)).to.equal(1);
      await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT_2, 10, ONE_FTNS,  0, ethers.ZeroHash, "");
      expect(await registry.providerBatchSequence(provider.address)).to.equal(2);
    });

    it("produces distinct batchIds for back-to-back identical commits", async function () {
      const tx1 = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 100, ONE_FTNS,  0, ethers.ZeroHash, "");
      const r1 = await tx1.wait();
      const id1 = r1.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];

      const tx2 = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 100, ONE_FTNS,  0, ethers.ZeroHash, "");
      const r2 = await tx2.wait();
      const id2 = r2.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];

      expect(id1).to.not.equal(id2);
    });

    it("reverts on zero Merkle root", async function () {
      await expect(
        registry
          .connect(provider)
          .commitBatch(requester.address, ethers.ZeroHash, 10, ONE_FTNS, 0, ethers.ZeroHash, "")
      ).to.be.revertedWithCustomError(registry, "EmptyMerkleRoot");
    });

    it("reverts on zero receipt count", async function () {
      await expect(
        registry
          .connect(provider)
          .commitBatch(requester.address, DUMMY_ROOT, 0, ONE_FTNS, 0, ethers.ZeroHash, "")
      ).to.be.revertedWithCustomError(registry, "ZeroReceiptCount");
    });

    it("accepts zero totalValueFTNS (pathological but not invalid)", async function () {
      await expect(
        registry
          .connect(provider)
          .commitBatch(requester.address, DUMMY_ROOT, 1, 0, 0, ethers.ZeroHash, "")
      ).to.not.be.reverted;
    });
  });

  describe("finalizeBatch (with EscrowPool wiring)", function () {
    let batchId;
    const batchValue = ONE_FTNS * 50n;

    beforeEach(async function () {
      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 100, batchValue,  0, ethers.ZeroHash, "ipfs://m");
      const r = await tx.wait();
      batchId = r.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];
    });

    it("reverts when called before the challenge window elapses", async function () {
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(registry, "ChallengeWindowNotElapsed");
    });

    it("reverts when called shortly before window elapses", async function () {
      await time.increase(DEFAULT_WINDOW - 10);
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(registry, "ChallengeWindowNotElapsed");
    });

    it("transfers FTNS from requester escrow to provider on finalize", async function () {
      await time.increase(DEFAULT_WINDOW);

      const poolBalBefore = await token.balanceOf(await escrowPool.getAddress());
      const providerBalBefore = await token.balanceOf(provider.address);
      const escrowBalBefore = await escrowPool.balanceOf(requester.address);

      await expect(registry.finalizeBatch(batchId))
        .to.emit(registry, "BatchFinalized")
        .to.emit(escrowPool, "Settled")
        .withArgs(
          requester.address,
          provider.address,
          batchValue,
          await registry.getAddress()
        );

      const poolBalAfter = await token.balanceOf(await escrowPool.getAddress());
      const providerBalAfter = await token.balanceOf(provider.address);
      const escrowBalAfter = await escrowPool.balanceOf(requester.address);

      expect(poolBalBefore - poolBalAfter).to.equal(batchValue);
      expect(providerBalAfter - providerBalBefore).to.equal(batchValue);
      expect(escrowBalBefore - escrowBalAfter).to.equal(batchValue);

      const batch = await registry.getBatch(batchId);
      expect(batch.status).to.equal(2); // FINALIZED
    });

    it("is callable by any address (watchdog)", async function () {
      await time.increase(DEFAULT_WINDOW);
      await registry.connect(watchdog).finalizeBatch(batchId);
      const batch = await registry.getBatch(batchId);
      expect(batch.status).to.equal(2);
    });

    it("reverts on double-finalize", async function () {
      await time.increase(DEFAULT_WINDOW);
      await registry.finalizeBatch(batchId);
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(registry, "BatchNotPending");
    });

    it("reverts on unknown batchId", async function () {
      const fakeId = ethers.keccak256(ethers.toUtf8Bytes("nonexistent"));
      await expect(
        registry.finalizeBatch(fakeId)
      ).to.be.revertedWithCustomError(registry, "BatchNotFound");
    });

    it("zero-value batch finalizes without invoking escrow pool", async function () {
      // A batch with totalValueFTNS=0 should finalize cleanly even if
      // escrowPool were unset. Verifies the `if (finalValue > 0)` guard.
      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT_2, 1, 0,  0, ethers.ZeroHash, "");
      const r = await tx.wait();
      const zeroBatchId = r.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted"
      ).args[0];

      await time.increase(DEFAULT_WINDOW);

      const escrowBalBefore = await escrowPool.balanceOf(requester.address);
      const providerBalBefore = await token.balanceOf(provider.address);

      await registry.finalizeBatch(zeroBatchId);

      expect(await escrowPool.balanceOf(requester.address)).to.equal(escrowBalBefore);
      expect(await token.balanceOf(provider.address)).to.equal(providerBalBefore);
    });

    it("REGRESSION (MEDIUM D-03): mid-flight setEscrowPool(0) does NOT brick in-flight finalize — per-batch snapshot is load-bearing", async function () {
      // Pre-D-03: this test asserted finalize reverts with
      // EscrowPoolNotConfigured because the LIVE pointer was zero at
      // finalize time. Post-D-03: finalize reads the per-batch
      // escrowPoolAtCommit snapshot, which is unaffected by mid-flight
      // setEscrowPool calls. The in-flight batch settles cleanly
      // against its committed-time pool.
      await registry.connect(owner).setEscrowPool(ethers.ZeroAddress);
      await time.increase(DEFAULT_WINDOW);
      await expect(registry.finalizeBatch(batchId)).to.emit(registry, "BatchFinalized");
    });

    it("REGRESSION (MEDIUM D-03): finalize reverts EscrowPoolNotConfigured if AT-COMMIT pool was zero", async function () {
      // Re-validate the EscrowPoolNotConfigured revert path with the
      // new per-batch semantics: commit a NEW batch while pool is
      // unset, and that batch's snapshot is zero — finalize reverts.
      await registry.connect(owner).setEscrowPool(ethers.ZeroAddress);
      const newRoot = ethers.keccak256(ethers.toUtf8Bytes("after-zero"));
      const tx = await registry.connect(provider).commitBatch(
        requester.address, newRoot, 1, ONE_FTNS, 0, ethers.ZeroHash, ""
      );
      const r = await tx.wait();
      const newId = r.logs.find(
        (l) => l.fragment && l.fragment.name === "BatchCommitted",
      ).args[0];
      await time.increase(DEFAULT_WINDOW);
      await expect(
        registry.finalizeBatch(newId),
      ).to.be.revertedWithCustomError(registry, "EscrowPoolNotConfigured");
    });

    it("reverts finalize when requester escrow has insufficient balance", async function () {
      // Requester withdraws most of escrow, leaving less than batchValue.
      const remainingAfterDrain = batchValue - 1n;
      const drain = INITIAL_ESCROW - remainingAfterDrain;
      await escrowPool.connect(requester).withdraw(drain);

      await time.increase(DEFAULT_WINDOW);
      await expect(
        registry.finalizeBatch(batchId)
      ).to.be.revertedWithCustomError(escrowPool, "InsufficientBalance");
    });
  });

  describe("governance: setChallengeWindowSeconds", function () {
    it("owner can update the window within bounds", async function () {
      const newWindow = 7 * 24 * 60 * 60;
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
      ).to.be.reverted;
    });

    it("reverts when new window is below minimum", async function () {
      await expect(
        registry.connect(owner).setChallengeWindowSeconds(60)
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });

    it("reverts when new window exceeds maximum", async function () {
      await expect(
        registry.connect(owner).setChallengeWindowSeconds(60 * 24 * 60 * 60)
      ).to.be.revertedWithCustomError(registry, "InvalidChallengeWindow");
    });

    it("REGRESSION (MEDIUM D-05): shortening the global window does NOT retroactively shorten an already-PENDING batch's window — per-batch snapshot is load-bearing", async function () {
      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 10, ONE_FTNS,  0, ethers.ZeroHash, "");
      const r = await tx.wait();
      const id = r.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];

      // Snapshot at commit was 3 days (DEFAULT_WINDOW). Owner shortens
      // global to 1 hour mid-flight; finalize must STILL refuse until
      // the original 3-day window elapses.
      await time.increase(1 * 24 * 60 * 60);
      await registry.connect(owner).setChallengeWindowSeconds(60 * 60);

      // Pre-D-05: finalize would succeed because elapsed (1d) >= live
      // window (1h). Post-D-05: finalize reverts because elapsed (1d)
      // < per-batch snapshot (3d).
      await expect(registry.finalizeBatch(id))
        .to.be.revertedWithCustomError(registry, "ChallengeWindowNotElapsed");

      // After original 3-day snapshot elapses, finalize succeeds.
      await time.increase(2 * 24 * 60 * 60 + 60);
      await expect(registry.finalizeBatch(id)).to.emit(registry, "BatchFinalized");
    });

    it("REGRESSION (MEDIUM D-05): lengthening the global window does NOT retroactively lengthen an already-PENDING batch's window either", async function () {
      // Symmetric test — challenges submitted within the original window
      // succeed even if global was extended afterwards.
      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 10, ONE_FTNS, 0, ethers.ZeroHash, "");
      const r = await tx.wait();
      const id = r.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];

      // Snapshot at commit was 3 days. Owner extends global to 7 days.
      await registry.connect(owner).setChallengeWindowSeconds(7 * 24 * 60 * 60);

      // After original 3-day snapshot elapses, finalize succeeds even
      // though the LIVE global is now 7 days.
      await time.increase(3 * 24 * 60 * 60 + 60);
      await expect(registry.finalizeBatch(id)).to.emit(registry, "BatchFinalized");
    });
  });

  describe("governance: setEscrowPool", function () {
    it("owner can update the pool address (must be a contract)", async function () {
      // L4 self-audit MED-7: setter rejects EOA targets. Use a
      // freshly-deployed contract as the new pool stand-in.
      const Stub = await ethers.getContractFactory("MockERC20");
      const stub = await Stub.deploy();
      await stub.waitForDeployment();
      const newPool = await stub.getAddress();
      await expect(registry.connect(owner).setEscrowPool(newPool))
        .to.emit(registry, "EscrowPoolUpdated")
        .withArgs(await escrowPool.getAddress(), newPool);
      expect(await registry.escrowPool()).to.equal(newPool);
    });

    it("non-owner cannot update the pool address", async function () {
      await expect(
        registry.connect(other).setEscrowPool(other.address)
      ).to.be.reverted;
    });

    it("REGRESSION (L4 self-audit MED-7 / D-03): rejects EOA target — setter requires contract", async function () {
      await expect(
        registry.connect(owner).setEscrowPool(other.address)
      ).to.be.revertedWithCustomError(registry, "SetterTargetNotContract");
    });

    it("permits address(0) — documented disable mode (per-batch D-03 snapshot protects in-flight)", async function () {
      await expect(registry.connect(owner).setEscrowPool(ethers.ZeroAddress))
        .to.emit(registry, "EscrowPoolUpdated")
        .withArgs(await escrowPool.getAddress(), ethers.ZeroAddress);
    });
  });

  describe("views", function () {
    let batchId;
    beforeEach(async function () {
      const tx = await registry
        .connect(provider)
        .commitBatch(requester.address, DUMMY_ROOT, 100, ONE_FTNS * 10n,  0, ethers.ZeroHash, "");
      const r = await tx.wait();
      batchId = r.logs.find((l) => l.fragment && l.fragment.name === "BatchCommitted").args[0];
    });

    it("isFinalizable flips at window boundary", async function () {
      expect(await registry.isFinalizable(batchId)).to.equal(false);
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
      await time.increase(24 * 60 * 60);
      const t1 = await registry.secondsUntilFinalizable(batchId);
      expect(t1).to.be.lessThan(t0);
    });

    it("secondsUntilFinalizable returns 0 after window elapses", async function () {
      await time.increase(DEFAULT_WINDOW);
      expect(await registry.secondsUntilFinalizable(batchId)).to.equal(0);
    });

    it("getBatch returns NONEXISTENT for unknown id", async function () {
      const fakeId = ethers.keccak256(ethers.toUtf8Bytes("unknown"));
      const batch = await registry.getBatch(fakeId);
      expect(batch.status).to.equal(0);
    });
  });
});
