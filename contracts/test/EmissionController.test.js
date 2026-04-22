const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 8 Task 1 — EmissionController core mint + rate.
// Per docs/2026-04-22-phase8-design-plan.md §6 Task 1.

describe("EmissionController — core mint + rate", function () {
  let controller, mockToken;
  let owner, distributor, other;

  const EPOCH = 4 * 365 * 24 * 60 * 60; // 4 years
  const ONE_FTNS = ethers.parseUnits("1", 18);

  // Baseline chosen so rate × one-epoch ≤ mintCap comfortably.
  // 1 FTNS/sec × 4 years ≈ 126M FTNS.
  const BASELINE_RATE = ONE_FTNS;

  // Test cap — 900M in production; smaller here to keep cap-overflow
  // tests tractable without pushing time forward for decades.
  const TEST_MINT_CAP = 1_000_000_000n * ONE_FTNS;

  async function deploy(epochZeroOffsetFromNow = 0, mintCap = TEST_MINT_CAP) {
    [owner, distributor, other] = await ethers.getSigners();

    const Mock = await ethers.getContractFactory("MockFTNSMinter");
    mockToken = await Mock.deploy();

    const now = await time.latest();
    const epochZero = BigInt(now + epochZeroOffsetFromNow);

    const Controller = await ethers.getContractFactory("EmissionController");
    controller = await Controller.deploy(
      await mockToken.getAddress(),
      epochZero,
      BASELINE_RATE,
      mintCap,
      owner.address
    );

    // Default wiring for most tests — owner sets distributor, test overrides
    // when needed.
    await controller.connect(owner).setAuthorizedDistributor(distributor.address);

    return { controller, mockToken, epochZero };
  }

  // -------------------------------------------------------------------------
  // currentEpoch
  // -------------------------------------------------------------------------

  describe("currentEpoch", function () {
    it("returns 0 before epoch zero start", async function () {
      await deploy(3600); // epoch zero 1h in the future
      expect(await controller.currentEpoch()).to.equal(0);
    });

    it("returns 0 during epoch 0", async function () {
      await deploy(0);
      await time.increase(EPOCH - 100);
      expect(await controller.currentEpoch()).to.equal(0);
    });

    it("advances to epoch 1 at boundary", async function () {
      await deploy(0);
      await time.increase(EPOCH);
      expect(await controller.currentEpoch()).to.equal(1);
    });

    it("advances to epoch 2 at second boundary", async function () {
      await deploy(0);
      await time.increase(2 * EPOCH);
      expect(await controller.currentEpoch()).to.equal(2);
    });

    it("reaches epoch 10 after 10 × epoch duration", async function () {
      await deploy(0);
      await time.increase(10 * EPOCH);
      expect(await controller.currentEpoch()).to.equal(10);
    });
  });

  // -------------------------------------------------------------------------
  // currentEpochRate
  // -------------------------------------------------------------------------

  describe("currentEpochRate", function () {
    it("equals baseline at epoch 0", async function () {
      await deploy(0);
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE);
    });

    it("halves at epoch 1", async function () {
      await deploy(0);
      await time.increase(EPOCH);
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE / 2n);
    });

    it("halves again at epoch 2", async function () {
      await deploy(0);
      await time.increase(2 * EPOCH);
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE / 4n);
    });

    it("is monotone non-increasing across adjacent epochs", async function () {
      await deploy(0);
      let previous = await controller.currentEpochRate();
      for (let i = 1; i <= 5; i++) {
        await time.increase(EPOCH);
        const current = await controller.currentEpochRate();
        expect(current).to.be.lte(previous);
        previous = current;
      }
    });
  });

  // -------------------------------------------------------------------------
  // timeUntilNextHalving
  // -------------------------------------------------------------------------

  describe("timeUntilNextHalving", function () {
    it("returns epoch duration just after epoch zero start", async function () {
      await deploy(0);
      // Small slack because `deploy` itself consumed a block or two.
      const t = await controller.timeUntilNextHalving();
      expect(t).to.be.gt(EPOCH - 60);
      expect(t).to.be.lte(EPOCH);
    });

    it("returns ~0 just before a halving", async function () {
      await deploy(0);
      await time.increase(EPOCH - 5);
      expect(await controller.timeUntilNextHalving()).to.be.lte(10);
    });
  });

  // -------------------------------------------------------------------------
  // mintAuthorized — access control
  // -------------------------------------------------------------------------

  describe("mintAuthorized access control", function () {
    it("reverts when called by a non-distributor", async function () {
      await deploy(0);
      await time.increase(3600);
      await expect(
        controller.connect(other).mintAuthorized(ONE_FTNS)
      ).to.be.revertedWithCustomError(controller, "NotDistributor");
    });

    it("reverts when minting is paused", async function () {
      await deploy(0);
      await time.increase(3600);
      await controller.connect(owner).pauseMinting();
      await expect(
        controller.connect(distributor).mintAuthorized(ONE_FTNS)
      ).to.be.revertedWithCustomError(controller, "MintingIsPaused");
    });

    it("reverts on zero amount", async function () {
      await deploy(0);
      await time.increase(3600);
      await expect(
        controller.connect(distributor).mintAuthorized(0)
      ).to.be.revertedWithCustomError(controller, "ZeroAmount");
    });
  });

  // -------------------------------------------------------------------------
  // mintAuthorized — rate + cap enforcement
  // -------------------------------------------------------------------------

  describe("mintAuthorized rate + cap", function () {
    it("accepts a mint within the rate limit and forwards to the token", async function () {
      await deploy(0);
      await time.increase(100);

      // Allowed ≈ BASELINE_RATE × ~100s (with some slack for test block time).
      const amount = BASELINE_RATE * 50n;
      await controller.connect(distributor).mintAuthorized(amount);

      expect(await mockToken.totalMinted()).to.equal(amount);
      expect(await mockToken.lastRecipient()).to.equal(distributor.address);
      expect(await controller.mintedToDate()).to.equal(amount);
    });

    it("rejects a mint exceeding the per-call rate limit", async function () {
      await deploy(0);
      await time.increase(10);

      // Ask for far more than rate × elapsed.
      const huge = BASELINE_RATE * 10_000n;
      await expect(
        controller.connect(distributor).mintAuthorized(huge)
      ).to.be.revertedWithCustomError(controller, "ExceedsRateLimit");
    });

    it("rejects a mint exceeding the lifetime cap", async function () {
      // Deploy with a tiny mintCap so we can hit it quickly.
      const tinyCap = BASELINE_RATE * 10n;
      await deploy(0, tinyCap);
      await time.increase(100);

      await controller.connect(distributor).mintAuthorized(tinyCap);
      expect(await controller.mintedToDate()).to.equal(tinyCap);

      await time.increase(100);
      await expect(
        controller.connect(distributor).mintAuthorized(1n)
      ).to.be.revertedWithCustomError(controller, "ExceedsMintCap");
    });

    it("emits Minted with the current epoch and rate", async function () {
      await deploy(0);
      await time.increase(100);

      const amount = BASELINE_RATE * 50n;
      await expect(controller.connect(distributor).mintAuthorized(amount))
        .to.emit(controller, "Minted")
        .withArgs(distributor.address, amount, 0, BASELINE_RATE);
    });

    it("updates lastMintTimestamp so the next call's elapsed starts fresh", async function () {
      await deploy(0);
      await time.increase(100);
      await controller.connect(distributor).mintAuthorized(BASELINE_RATE * 50n);

      const t1 = await controller.lastMintTimestamp();

      await time.increase(200);
      await controller.connect(distributor).mintAuthorized(BASELINE_RATE * 100n);

      const t2 = await controller.lastMintTimestamp();
      expect(t2).to.be.gt(t1);
    });
  });

  // -------------------------------------------------------------------------
  // Governance
  // -------------------------------------------------------------------------

  describe("setAuthorizedDistributor", function () {
    it("only callable by owner", async function () {
      await deploy(0);
      await expect(
        controller.connect(other).setAuthorizedDistributor(other.address)
      ).to.be.revertedWithCustomError(controller, "OwnableUnauthorizedAccount");
    });

    it("rejects zero address", async function () {
      await deploy(0);
      await expect(
        controller.connect(owner).setAuthorizedDistributor(ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(controller, "InvalidAddress");
    });

    it("emits DistributorUpdated on change", async function () {
      await deploy(0);
      await expect(
        controller.connect(owner).setAuthorizedDistributor(other.address)
      )
        .to.emit(controller, "DistributorUpdated")
        .withArgs(distributor.address, other.address);
    });
  });

  describe("pause / resume", function () {
    it("pauseMinting is owner-only", async function () {
      await deploy(0);
      await expect(
        controller.connect(other).pauseMinting()
      ).to.be.revertedWithCustomError(controller, "OwnableUnauthorizedAccount");
    });

    it("cannot pause twice", async function () {
      await deploy(0);
      await controller.connect(owner).pauseMinting();
      await expect(
        controller.connect(owner).pauseMinting()
      ).to.be.revertedWithCustomError(controller, "AlreadyPaused");
    });

    it("cannot resume when not paused", async function () {
      await deploy(0);
      await expect(
        controller.connect(owner).resumeMinting()
      ).to.be.revertedWithCustomError(controller, "NotPaused");
    });

    it("pause then resume restores minting", async function () {
      await deploy(0);
      await time.increase(3600);

      await controller.connect(owner).pauseMinting();
      await expect(
        controller.connect(distributor).mintAuthorized(ONE_FTNS)
      ).to.be.revertedWithCustomError(controller, "MintingIsPaused");

      await controller.connect(owner).resumeMinting();
      await controller.connect(distributor).mintAuthorized(ONE_FTNS);
      expect(await mockToken.totalMinted()).to.equal(ONE_FTNS);
    });

    it("emits MintingPaused and MintingResumed", async function () {
      await deploy(0);
      await expect(controller.connect(owner).pauseMinting())
        .to.emit(controller, "MintingPaused")
        .withArgs(owner.address);
      await expect(controller.connect(owner).resumeMinting())
        .to.emit(controller, "MintingResumed")
        .withArgs(owner.address);
    });
  });
});
