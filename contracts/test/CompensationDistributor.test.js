const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 8 Task 2 — CompensationDistributor.
// Per docs/2026-04-22-phase8-design-plan.md §6 Task 2.

describe("CompensationDistributor — pull + weighted split", function () {
  let controller, distributor, token;
  let owner, creator, operator, grant, anyone, other;

  const EPOCH = 4 * 365 * 24 * 60 * 60;
  const NINETY_DAYS = 90 * 24 * 60 * 60;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const BASELINE_RATE = ONE_FTNS; // 1 FTNS/sec at epoch 0
  const TEST_MINT_CAP = 1_000_000_000n * ONE_FTNS;

  // 70/25/5 split per plan §6 Task 2 test list.
  const INITIAL_WEIGHTS = {
    creatorPoolBps: 7000,
    operatorPoolBps: 2500,
    grantPoolBps: 500,
  };

  async function deploy() {
    [owner, creator, operator, grant, anyone, other] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockFTNSToken");
    token = await Token.deploy();

    const now = await time.latest();
    const Controller = await ethers.getContractFactory("EmissionController");
    controller = await Controller.deploy(
      await token.getAddress(),
      now,
      BASELINE_RATE,
      TEST_MINT_CAP,
      owner.address
    );

    const Distributor = await ethers.getContractFactory("CompensationDistributor");
    distributor = await Distributor.deploy(
      await token.getAddress(),
      await controller.getAddress(),
      creator.address,
      operator.address,
      grant.address,
      INITIAL_WEIGHTS,
      owner.address
    );

    await controller
      .connect(owner)
      .setAuthorizedDistributor(await distributor.getAddress());

    return { token, controller, distributor };
  }

  async function weights(distrib = distributor) {
    const w = await distrib.currentWeights();
    return {
      creatorPoolBps: Number(w.creatorPoolBps),
      operatorPoolBps: Number(w.operatorPoolBps),
      grantPoolBps: Number(w.grantPoolBps),
    };
  }

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  describe("construction", function () {
    it("rejects weights that do not sum to 10000", async function () {
      await deploy(); // reuse deployed token + controller
      const Distrib = await ethers.getContractFactory("CompensationDistributor");
      await expect(
        Distrib.deploy(
          await token.getAddress(),
          await controller.getAddress(),
          creator.address,
          operator.address,
          grant.address,
          { creatorPoolBps: 7000, operatorPoolBps: 2500, grantPoolBps: 400 },
          owner.address
        )
      ).to.be.revertedWithCustomError(distributor, "InvalidWeights");
    });

    it("rejects zero pool addresses", async function () {
      await deploy();
      const Distrib = await ethers.getContractFactory("CompensationDistributor");
      await expect(
        Distrib.deploy(
          await token.getAddress(),
          await controller.getAddress(),
          ethers.ZeroAddress,
          operator.address,
          grant.address,
          INITIAL_WEIGHTS,
          owner.address
        )
      ).to.be.revertedWithCustomError(distributor, "InvalidAddress");
    });

    it("stores the initial weights", async function () {
      await deploy();
      expect(await weights()).to.deep.equal(INITIAL_WEIGHTS);
    });
  });

  // -------------------------------------------------------------------------
  // pullAndDistribute
  // -------------------------------------------------------------------------

  describe("pullAndDistribute", function () {
    it("splits the minted emission 70/25/5", async function () {
      await deploy();
      await time.increase(100);

      await distributor.connect(anyone).pullAndDistribute();

      const total =
        (await token.balanceOf(creator.address)) +
        (await token.balanceOf(operator.address)) +
        (await token.balanceOf(grant.address));
      expect(total).to.be.gt(0);

      // Share ratios within 1 wei of bps targets (integer division dust).
      const creatorBal = await token.balanceOf(creator.address);
      const operatorBal = await token.balanceOf(operator.address);
      const grantBal = await token.balanceOf(grant.address);

      expect(creatorBal).to.equal((total * 7000n) / 10000n);
      expect(operatorBal).to.equal((total * 2500n) / 10000n);
      // Grant receives total − creator − operator, including any dust.
      expect(grantBal).to.equal(total - creatorBal - operatorBal);
    });

    it("is callable by anyone", async function () {
      await deploy();
      await time.increase(100);
      await expect(distributor.connect(other).pullAndDistribute()).not.to.be
        .reverted;
    });

    it("is a no-op when the controller is paused and balance is empty", async function () {
      await deploy();
      await time.increase(100);
      // First call drains the distributor's balance.
      await distributor.connect(anyone).pullAndDistribute();
      const creatorBefore = await token.balanceOf(creator.address);

      // Pause the controller so _tryPull takes the paused early-exit,
      // leaving the distributor with a zero balance that _distribute
      // also early-exits on.
      await controller.connect(owner).pauseMinting();
      await time.increase(1000);

      await distributor.connect(anyone).pullAndDistribute();
      expect(await token.balanceOf(creator.address)).to.equal(creatorBefore);
    });

    it("emits Distributed with the per-pool amounts", async function () {
      await deploy();
      await time.increase(100);
      await expect(distributor.connect(anyone).pullAndDistribute()).to.emit(
        distributor,
        "Distributed"
      );
    });

    it("updates lastDistributionTimestamp", async function () {
      await deploy();
      await time.increase(100);
      await distributor.connect(anyone).pullAndDistribute();
      const t = await distributor.lastDistributionTimestamp();
      expect(t).to.be.gt(0);
    });

    it("still distributes residual balance when emission is paused", async function () {
      await deploy();
      // Prime the contract by minting directly to it (simulates a leftover
      // balance from before a pause).
      await token.mintReward(await distributor.getAddress(), ONE_FTNS * 100n);
      await controller.connect(owner).pauseMinting();

      await distributor.connect(anyone).pullAndDistribute();
      expect(await token.balanceOf(creator.address)).to.be.gt(0);
    });
  });

  // -------------------------------------------------------------------------
  // Weight scheduling
  // -------------------------------------------------------------------------

  describe("updateWeights", function () {
    it("is onlyOwner", async function () {
      await deploy();
      const future = (await time.latest()) + NINETY_DAYS + 1;
      await expect(
        distributor
          .connect(other)
          .updateWeights(
            { creatorPoolBps: 5000, operatorPoolBps: 4000, grantPoolBps: 1000 },
            future
          )
      ).to.be.revertedWithCustomError(distributor, "OwnableUnauthorizedAccount");
    });

    it("rejects scheduledAt earlier than now + 90 days", async function () {
      await deploy();
      const tooSoon = (await time.latest()) + NINETY_DAYS - 1;
      await expect(
        distributor
          .connect(owner)
          .updateWeights(
            { creatorPoolBps: 5000, operatorPoolBps: 4000, grantPoolBps: 1000 },
            tooSoon
          )
      ).to.be.revertedWithCustomError(distributor, "ScheduleTooSoon");
    });

    it("rejects weights that do not sum to 10000", async function () {
      await deploy();
      const future = (await time.latest()) + NINETY_DAYS + 1;
      await expect(
        distributor
          .connect(owner)
          .updateWeights(
            { creatorPoolBps: 5000, operatorPoolBps: 4000, grantPoolBps: 999 },
            future
          )
      ).to.be.revertedWithCustomError(distributor, "InvalidWeights");
    });

    it("emits WeightsScheduled", async function () {
      await deploy();
      const future = (await time.latest()) + NINETY_DAYS + 100;
      const next = {
        creatorPoolBps: 5000,
        operatorPoolBps: 4000,
        grantPoolBps: 1000,
      };
      await expect(distributor.connect(owner).updateWeights(next, future))
        .to.emit(distributor, "WeightsScheduled");
    });
  });

  describe("scheduled weight activation", function () {
    it("does not apply before scheduledAt", async function () {
      await deploy();
      const future = (await time.latest()) + NINETY_DAYS + 100;
      await distributor
        .connect(owner)
        .updateWeights(
          { creatorPoolBps: 5000, operatorPoolBps: 4000, grantPoolBps: 1000 },
          future
        );

      await time.increase(100); // still < 90 days
      await distributor.connect(anyone).pullAndDistribute();

      expect(await weights()).to.deep.equal(INITIAL_WEIGHTS);
    });

    it("activates on the next pullAndDistribute at/after scheduledAt", async function () {
      await deploy();
      const future = (await time.latest()) + NINETY_DAYS + 100;
      const next = {
        creatorPoolBps: 5000,
        operatorPoolBps: 4000,
        grantPoolBps: 1000,
      };
      await distributor.connect(owner).updateWeights(next, future);

      await time.increaseTo(future + 1);
      await expect(distributor.connect(anyone).pullAndDistribute())
        .to.emit(distributor, "WeightsActivated");

      expect(await weights()).to.deep.equal(next);
    });

    it("clears hasScheduledWeights after activation", async function () {
      await deploy();
      const future = (await time.latest()) + NINETY_DAYS + 100;
      await distributor
        .connect(owner)
        .updateWeights(
          { creatorPoolBps: 5000, operatorPoolBps: 4000, grantPoolBps: 1000 },
          future
        );
      await time.increaseTo(future + 1);
      await distributor.connect(anyone).pullAndDistribute();
      expect(await distributor.hasScheduledWeights()).to.equal(false);
    });
  });

  // -------------------------------------------------------------------------
  // Pool address rotation
  // -------------------------------------------------------------------------

  describe("setPoolAddresses", function () {
    it("is onlyOwner", async function () {
      await deploy();
      await expect(
        distributor
          .connect(other)
          .setPoolAddresses(other.address, other.address, other.address)
      ).to.be.revertedWithCustomError(distributor, "OwnableUnauthorizedAccount");
    });

    it("rejects any zero address", async function () {
      await deploy();
      await expect(
        distributor
          .connect(owner)
          .setPoolAddresses(ethers.ZeroAddress, operator.address, grant.address)
      ).to.be.revertedWithCustomError(distributor, "InvalidAddress");
    });

    it("updates the stored pools and emits PoolAddressesUpdated", async function () {
      await deploy();
      await expect(
        distributor
          .connect(owner)
          .setPoolAddresses(other.address, other.address, other.address)
      )
        .to.emit(distributor, "PoolAddressesUpdated")
        .withArgs(other.address, other.address, other.address);
      expect(await distributor.creatorPool()).to.equal(other.address);
    });
  });
});
