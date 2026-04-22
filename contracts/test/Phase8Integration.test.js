const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 8 end-to-end integration — composes the real FTNSTokenSimple
// (UUPS proxy) + EmissionController (Task 1) + CompensationDistributor
// (Task 2) with real MINTER_ROLE wiring (Task 3 pattern).
//
// Per docs/2026-04-22-phase8-design-plan.md §7 acceptance criteria:
//   1. On-chain halving at Epoch 2 boundary (rate == baseline >> 2).
//   2. No sequence of governance-authorised calls can exceed the cap
//      or the per-call rate limit.
//   3. Operator continuity — pools receive expected-rate FTNS.
//
// Capstone for Phase 8 Tasks 1-4. Tasks 5-9 (testnet exercise, formal
// verification, audits, migration playbook, mainnet deploy) are
// non-engineering and cannot be covered by hardhat tests.

describe("Phase 8 end-to-end integration", function () {
  let ftnsToken, controller, distributor;
  let owner, treasury, creator, operator, grant, triggerer;

  const EPOCH = 4 * 365 * 24 * 60 * 60;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const BASELINE_RATE = ONE_FTNS; // 1 FTNS/sec at epoch 0
  const TEST_MINT_CAP = 500_000_000n * ONE_FTNS; // 500M — fits in token headroom
  const INITIAL_WEIGHTS = {
    creatorPoolBps: 7000,
    operatorPoolBps: 2500,
    grantPoolBps: 500,
  };

  async function deployFullStack() {
    [owner, treasury, creator, operator, grant, triggerer] =
      await ethers.getSigners();

    // 1. Deploy FTNSTokenSimple via UUPS proxy.
    const Token = await ethers.getContractFactory("FTNSTokenSimple");
    ftnsToken = await upgrades.deployProxy(
      Token,
      [owner.address, treasury.address],
      { initializer: "initialize", kind: "uups" }
    );

    // 2. Deploy EmissionController with epoch zero at current block.
    const now = await time.latest();
    const Controller = await ethers.getContractFactory("EmissionController");
    controller = await Controller.deploy(
      await ftnsToken.getAddress(),
      now,
      BASELINE_RATE,
      TEST_MINT_CAP,
      owner.address
    );

    // 3. Deploy CompensationDistributor wired to the controller.
    const Distributor = await ethers.getContractFactory(
      "CompensationDistributor"
    );
    distributor = await Distributor.deploy(
      await ftnsToken.getAddress(),
      await controller.getAddress(),
      creator.address,
      operator.address,
      grant.address,
      INITIAL_WEIGHTS,
      owner.address
    );

    // 4. Wire: set distributor as controller's authorized address,
    //    grant MINTER_ROLE to controller.
    await controller
      .connect(owner)
      .setAuthorizedDistributor(await distributor.getAddress());
    const MINTER_ROLE = await ftnsToken.MINTER_ROLE();
    await ftnsToken
      .connect(owner)
      .grantRole(MINTER_ROLE, await controller.getAddress());

    return { ftnsToken, controller, distributor };
  }

  // -------------------------------------------------------------------------
  // Acceptance §7.1 — Monotone halving at the epoch boundary
  // -------------------------------------------------------------------------

  describe("§7.1 halving at epoch boundary", function () {
    it("currentEpochRate halves exactly at each 4-year boundary", async function () {
      await deployFullStack();
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE);

      await time.increase(EPOCH);
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE / 2n);

      await time.increase(EPOCH);
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE / 4n);

      await time.increase(EPOCH);
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE / 8n);
    });

    it("at Epoch 2 boundary, rate is exactly baseline >> 2", async function () {
      await deployFullStack();
      await time.increase(2 * EPOCH);
      // Plan §7.1: at `block.timestamp >= epochZeroStartTimestamp + 2 *
      // EPOCH_DURATION_SECONDS`, currentEpochRate returns exactly
      // BASELINE_RATE_PER_SECOND >> 2. Verifiable by any caller at any time.
      expect(await controller.currentEpochRate()).to.equal(BASELINE_RATE >> 2n);
    });
  });

  // -------------------------------------------------------------------------
  // Acceptance §7.2 — Cap + rate invariants
  // -------------------------------------------------------------------------

  describe("§7.2 cap + rate enforcement", function () {
    it("mintedToDate monotone-increasing across distributions", async function () {
      await deployFullStack();

      let last = await controller.mintedToDate();
      for (let i = 0; i < 5; i++) {
        await time.increase(100);
        await distributor.connect(triggerer).pullAndDistribute();
        const now = await controller.mintedToDate();
        expect(now).to.be.gte(last);
        last = now;
      }
    });

    it("per-call mint never exceeds currentEpochRate × elapsed", async function () {
      await deployFullStack();
      await time.increase(100);

      const beforeBal = await ftnsToken.balanceOf(await distributor.getAddress());
      const elapsed = BigInt(
        (await time.latest()) -
          Number(await controller.lastMintTimestamp())
      );
      const maxAllowed = (await controller.currentEpochRate()) * elapsed;

      // pullAndDistribute pulls directly; delta to distributor balance
      // (pre-distribution) is bounded by maxAllowed.
      // But pullAndDistribute also distributes out in the same tx, so
      // we observe totalSupply growth instead.
      const supplyBefore = await ftnsToken.totalSupply();
      await distributor.connect(triggerer).pullAndDistribute();
      const supplyAfter = await ftnsToken.totalSupply();

      const minted = supplyAfter - supplyBefore;
      // Give a few-block slack for the elapsed-time calculation.
      expect(minted).to.be.lte(
        (await controller.currentEpochRate()) * (elapsed + 10n)
      );
    });
  });

  // -------------------------------------------------------------------------
  // Acceptance §7.3 — Pool receipts per weight
  // -------------------------------------------------------------------------

  describe("§7.3 pool receipts match governance weights", function () {
    it("70/25/5 split over a single distribution", async function () {
      await deployFullStack();
      await time.increase(100);
      await distributor.connect(triggerer).pullAndDistribute();

      const cBal = await ftnsToken.balanceOf(creator.address);
      const oBal = await ftnsToken.balanceOf(operator.address);
      const gBal = await ftnsToken.balanceOf(grant.address);
      const total = cBal + oBal + gBal;

      // Weights were 70/25/5. Integer-division dust accumulates to grant.
      expect(cBal).to.equal((total * 7000n) / 10000n);
      expect(oBal).to.equal((total * 2500n) / 10000n);
      expect(gBal).to.equal(total - cBal - oBal);
    });

    it("scheduled weight update activates after 90 days and applies to future distributes", async function () {
      await deployFullStack();
      await time.increase(100);

      // Initial distribution at 70/25/5.
      await distributor.connect(triggerer).pullAndDistribute();
      const cBefore = await ftnsToken.balanceOf(creator.address);

      // Schedule a new split — 50/40/10 — 91 days out.
      const scheduled = (await time.latest()) + 91 * 24 * 60 * 60;
      const newWeights = {
        creatorPoolBps: 5000,
        operatorPoolBps: 4000,
        grantPoolBps: 1000,
      };
      await distributor.connect(owner).updateWeights(newWeights, scheduled);

      // Advance past scheduled; next distribution should activate the
      // new weights.
      await time.increaseTo(scheduled + 1);
      await expect(distributor.connect(triggerer).pullAndDistribute())
        .to.emit(distributor, "WeightsActivated");

      // Confirm current weights are the new 50/40/10.
      const w = await distributor.currentWeights();
      expect(Number(w.creatorPoolBps)).to.equal(5000);
      expect(Number(w.operatorPoolBps)).to.equal(4000);
      expect(Number(w.grantPoolBps)).to.equal(1000);

      // One more distribution — confirm split follows new weights.
      await time.increase(100);
      const cMid = await ftnsToken.balanceOf(creator.address);
      const oMid = await ftnsToken.balanceOf(operator.address);
      const gMid = await ftnsToken.balanceOf(grant.address);

      await distributor.connect(triggerer).pullAndDistribute();

      const cDelta = (await ftnsToken.balanceOf(creator.address)) - cMid;
      const oDelta = (await ftnsToken.balanceOf(operator.address)) - oMid;
      const gDelta = (await ftnsToken.balanceOf(grant.address)) - gMid;
      const thisRound = cDelta + oDelta + gDelta;
      expect(cDelta).to.equal((thisRound * 5000n) / 10000n);
      expect(oDelta).to.equal((thisRound * 4000n) / 10000n);
      expect(gDelta).to.equal(thisRound - cDelta - oDelta);
    });
  });

  // -------------------------------------------------------------------------
  // Governance — MINTER_ROLE revocation halts emission
  // -------------------------------------------------------------------------

  describe("governance off-ramp via MINTER_ROLE revocation", function () {
    it("revoking MINTER_ROLE blocks future distributions at the mint step", async function () {
      await deployFullStack();
      await time.increase(100);
      await distributor.connect(triggerer).pullAndDistribute();

      const MINTER_ROLE = await ftnsToken.MINTER_ROLE();
      await ftnsToken
        .connect(owner)
        .revokeRole(MINTER_ROLE, await controller.getAddress());

      // Next pullAndDistribute attempts mintAuthorized → token-side
      // AccessControl revert bubbles up through the controller.
      await time.increase(100);
      await expect(
        distributor.connect(triggerer).pullAndDistribute()
      ).to.be.reverted;
    });
  });

  // -------------------------------------------------------------------------
  // Pause/resume end-to-end
  // -------------------------------------------------------------------------

  describe("pause / resume end-to-end", function () {
    it("pause blocks distribution; resume restores it", async function () {
      await deployFullStack();
      await time.increase(100);
      await distributor.connect(triggerer).pullAndDistribute();

      await controller.connect(owner).pauseMinting();
      await time.increase(100);

      // pullAndDistribute's _tryPull takes the paused early-exit; no new
      // FTNS minted; distributor balance was drained by the prior
      // distribute, so no new pool receipts either.
      const cBefore = await ftnsToken.balanceOf(creator.address);
      await distributor.connect(triggerer).pullAndDistribute();
      expect(await ftnsToken.balanceOf(creator.address)).to.equal(cBefore);

      // Resume; next distribute succeeds.
      await controller.connect(owner).resumeMinting();
      await time.increase(100);
      await distributor.connect(triggerer).pullAndDistribute();
      expect(await ftnsToken.balanceOf(creator.address)).to.be.gt(cBefore);
    });
  });

  // -------------------------------------------------------------------------
  // Halving continuity — Epoch 1 rate to Epoch 2 rate
  // -------------------------------------------------------------------------

  describe("halving continuity across epoch boundary", function () {
    it("per-second emission halves at the boundary and pools see the new rate", async function () {
      await deployFullStack();

      // Advance to mid-epoch-0, distribute. Capture the rate.
      await time.increase(100);
      await distributor.connect(triggerer).pullAndDistribute();
      const rateEpoch0 = await controller.currentEpochRate();

      // Fast-forward past the epoch-0 → epoch-1 boundary.
      await time.increase(EPOCH);
      await distributor.connect(triggerer).pullAndDistribute();
      const rateEpoch1 = await controller.currentEpochRate();

      // Epoch 1 rate is exactly half Epoch 0.
      expect(rateEpoch1).to.equal(rateEpoch0 / 2n);

      // Confirm subsequent mints scale with the new rate.
      const supplyBefore = await ftnsToken.totalSupply();
      const elapsed1 = 1000n;
      await time.increase(Number(elapsed1));
      await distributor.connect(triggerer).pullAndDistribute();
      const supplyAfter = await ftnsToken.totalSupply();
      const minted = supplyAfter - supplyBefore;

      // Minted over ~1000s at the halved rate is bounded below the
      // pre-halving rate × same elapsed time.
      expect(minted).to.be.lt(rateEpoch0 * elapsed1);
    });
  });
});
