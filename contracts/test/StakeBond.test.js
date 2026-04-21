const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Phase 7 Task 1 — StakeBond bond/requestUnbond/withdraw lifecycle.
// Slashing + bounty accrual are Task 2.

describe("StakeBond — lifecycle", function () {
  let bond, token;
  let owner, provider, other, slasher;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_UNBOND_DELAY = 7 * 24 * 60 * 60; // 7 days
  const STANDARD_STAKE = 5_000n * ONE_FTNS;
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const CRITICAL_STAKE = 50_000n * ONE_FTNS;

  // Slash-rate basis-points per Phase 7 §3.2.
  const STANDARD_SLASH_BPS = 5000;   // 50%
  const PREMIUM_SLASH_BPS = 10000;   // 100%

  beforeEach(async function () {
    [owner, provider, other, slasher] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Bond = await ethers.getContractFactory("StakeBond");
    bond = await Bond.deploy(
      owner.address,
      await token.getAddress(),
      DEFAULT_UNBOND_DELAY
    );
    await bond.waitForDeployment();

    // Fund the provider + approve StakeBond for a generous amount.
    await token.mint(provider.address, CRITICAL_STAKE * 3n);
    await token
      .connect(provider)
      .approve(await bond.getAddress(), CRITICAL_STAKE * 3n);
  });

  describe("constructor", function () {
    it("sets owner, token, and unbond delay", async function () {
      expect(await bond.owner()).to.equal(owner.address);
      expect(await bond.ftns()).to.equal(await token.getAddress());
      expect(await bond.unbondDelaySeconds()).to.equal(DEFAULT_UNBOND_DELAY);
    });

    it("reverts on zero FTNS address", async function () {
      const Bond = await ethers.getContractFactory("StakeBond");
      await expect(
        Bond.deploy(owner.address, ethers.ZeroAddress, DEFAULT_UNBOND_DELAY)
      ).to.be.revertedWithCustomError(bond, "ZeroAddress");
    });

    it("reverts on unbond delay below minimum", async function () {
      const Bond = await ethers.getContractFactory("StakeBond");
      await expect(
        Bond.deploy(owner.address, await token.getAddress(), 3600)
      ).to.be.revertedWithCustomError(bond, "InvalidUnbondDelay");
    });

    it("reverts on unbond delay above maximum", async function () {
      const Bond = await ethers.getContractFactory("StakeBond");
      const tooLong = 60 * 24 * 60 * 60; // 60 days
      await expect(
        Bond.deploy(owner.address, await token.getAddress(), tooLong)
      ).to.be.revertedWithCustomError(bond, "InvalidUnbondDelay");
    });
  });

  describe("bond", function () {
    it("accepts a standard-tier bond and transfers FTNS", async function () {
      const bondAddr = await bond.getAddress();
      const before = await token.balanceOf(bondAddr);

      await expect(
        bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS)
      )
        .to.emit(bond, "Bonded")
        .withArgs(
          provider.address,
          STANDARD_STAKE,
          STANDARD_SLASH_BPS,
          anyUint()
        );

      const stake = await bond.stakeOf(provider.address);
      expect(stake.amount).to.equal(STANDARD_STAKE);
      expect(stake.status).to.equal(1); // BONDED
      expect(stake.tier_slash_rate_bps).to.equal(STANDARD_SLASH_BPS);
      expect(stake.unbond_eligible_at).to.equal(0);

      expect(await token.balanceOf(bondAddr)).to.equal(before + STANDARD_STAKE);
    });

    it("rejects zero-amount bond", async function () {
      await expect(
        bond.connect(provider).bond(0, STANDARD_SLASH_BPS)
      ).to.be.revertedWithCustomError(bond, "ZeroAmount");
    });

    it("rejects slash rate above 100% (10000 bps)", async function () {
      await expect(
        bond.connect(provider).bond(STANDARD_STAKE, 10001)
      ).to.be.revertedWithCustomError(bond, "InvalidSlashRateBps");
    });

    it("rejects double-bond from already-BONDED provider", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);

      await expect(
        bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS)
      ).to.be.revertedWithCustomError(bond, "AlreadyBonded");
    });

    it("rejects bond from UNBONDING provider", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);
      await bond.connect(provider).requestUnbond();

      await expect(
        bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS)
      ).to.be.revertedWithCustomError(bond, "AlreadyBonded");
    });

    it("allows re-bond after full withdraw", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);
      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY + 10);
      await bond.connect(provider).withdraw();

      // Now fully withdrawn; re-bond with different tier.
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      const stake = await bond.stakeOf(provider.address);
      expect(stake.status).to.equal(1); // BONDED
      expect(stake.amount).to.equal(PREMIUM_STAKE);
      expect(stake.tier_slash_rate_bps).to.equal(PREMIUM_SLASH_BPS);
    });
  });

  describe("requestUnbond", function () {
    beforeEach(async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
    });

    it("transitions BONDED → UNBONDING and sets eligibility time", async function () {
      const blockTimeBefore = await time.latest();
      await expect(bond.connect(provider).requestUnbond())
        .to.emit(bond, "UnbondRequested");

      const stake = await bond.stakeOf(provider.address);
      expect(stake.status).to.equal(2); // UNBONDING
      expect(Number(stake.unbond_eligible_at)).to.be.gte(
        blockTimeBefore + DEFAULT_UNBOND_DELAY
      );
    });

    it("reverts when called twice (already UNBONDING)", async function () {
      await bond.connect(provider).requestUnbond();
      await expect(
        bond.connect(provider).requestUnbond()
      ).to.be.revertedWithCustomError(bond, "NotBonded");
    });

    it("reverts when provider has never bonded", async function () {
      await expect(
        bond.connect(other).requestUnbond()
      ).to.be.revertedWithCustomError(bond, "NotBonded");
    });
  });

  describe("withdraw", function () {
    beforeEach(async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
    });

    it("reverts before requestUnbond is called", async function () {
      await expect(
        bond.connect(provider).withdraw()
      ).to.be.revertedWithCustomError(bond, "NotUnbonding");
    });

    it("reverts immediately after requestUnbond (delay not elapsed)", async function () {
      await bond.connect(provider).requestUnbond();
      await expect(
        bond.connect(provider).withdraw()
      ).to.be.revertedWithCustomError(bond, "UnbondDelayNotElapsed");
    });

    it("reverts shortly before delay elapses", async function () {
      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY - 60);
      await expect(
        bond.connect(provider).withdraw()
      ).to.be.revertedWithCustomError(bond, "UnbondDelayNotElapsed");
    });

    it("succeeds after delay + returns full stake to provider", async function () {
      const providerBalBefore = await token.balanceOf(provider.address);

      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY + 10);

      await expect(bond.connect(provider).withdraw())
        .to.emit(bond, "Withdrawn")
        .withArgs(provider.address, PREMIUM_STAKE);

      const stake = await bond.stakeOf(provider.address);
      expect(stake.status).to.equal(3); // WITHDRAWN
      expect(stake.amount).to.equal(0);

      expect(await token.balanceOf(provider.address)).to.equal(
        providerBalBefore + PREMIUM_STAKE
      );
    });

    it("reverts on double-withdraw", async function () {
      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY + 10);
      await bond.connect(provider).withdraw();

      await expect(
        bond.connect(provider).withdraw()
      ).to.be.revertedWithCustomError(bond, "NotUnbonding");
    });
  });

  describe("effectiveTier", function () {
    it("returns 'open' for unbonded providers", async function () {
      expect(await bond.effectiveTier(other.address)).to.equal("open");
    });

    it("returns 'standard' for 5K-FTNS bond", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);
      expect(await bond.effectiveTier(provider.address)).to.equal("standard");
    });

    it("returns 'premium' for 25K-FTNS bond", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      expect(await bond.effectiveTier(provider.address)).to.equal("premium");
    });

    it("returns 'critical' for 50K-FTNS bond", async function () {
      await bond.connect(provider).bond(CRITICAL_STAKE, PREMIUM_SLASH_BPS);
      expect(await bond.effectiveTier(provider.address)).to.equal("critical");
    });

    it("returns 'open' for sub-5K bond", async function () {
      await token
        .connect(provider)
        .approve(await bond.getAddress(), 100n * ONE_FTNS);
      await bond.connect(provider).bond(100n * ONE_FTNS, 0);
      expect(await bond.effectiveTier(provider.address)).to.equal("open");
    });

    it("returns 'open' during UNBONDING regardless of amount", async function () {
      await bond.connect(provider).bond(CRITICAL_STAKE, PREMIUM_SLASH_BPS);
      expect(await bond.effectiveTier(provider.address)).to.equal("critical");

      await bond.connect(provider).requestUnbond();
      expect(await bond.effectiveTier(provider.address)).to.equal("open");
    });

    it("returns 'open' after WITHDRAWN", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY + 10);
      await bond.connect(provider).withdraw();

      expect(await bond.effectiveTier(provider.address)).to.equal("open");
    });
  });

  describe("governance: setUnbondDelay", function () {
    it("owner can update the delay within bounds", async function () {
      const newDelay = 14 * 24 * 60 * 60;
      await expect(bond.connect(owner).setUnbondDelay(newDelay))
        .to.emit(bond, "UnbondDelayUpdated")
        .withArgs(DEFAULT_UNBOND_DELAY, newDelay);
      expect(await bond.unbondDelaySeconds()).to.equal(newDelay);
    });

    it("non-owner cannot update the delay", async function () {
      await expect(
        bond.connect(provider).setUnbondDelay(14 * 24 * 60 * 60)
      ).to.be.reverted;
    });

    it("reverts on delay below minimum", async function () {
      await expect(
        bond.connect(owner).setUnbondDelay(3600)
      ).to.be.revertedWithCustomError(bond, "InvalidUnbondDelay");
    });

    it("reverts on delay above maximum", async function () {
      await expect(
        bond.connect(owner).setUnbondDelay(60 * 24 * 60 * 60)
      ).to.be.revertedWithCustomError(bond, "InvalidUnbondDelay");
    });

    it("does NOT retroactively affect in-flight UNBONDING stakes", async function () {
      // Provider requests unbond under the default 7d delay.
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond.connect(provider).requestUnbond();
      const stakeBefore = await bond.stakeOf(provider.address);
      const oldEligibleAt = stakeBefore.unbond_eligible_at;

      // Governance raises the delay to 14d AFTER provider requested unbond.
      await bond.connect(owner).setUnbondDelay(14 * 24 * 60 * 60);

      // Provider's unbond_eligible_at is unchanged — they keep their
      // original 7d window.
      const stakeAfter = await bond.stakeOf(provider.address);
      expect(stakeAfter.unbond_eligible_at).to.equal(oldEligibleAt);

      // Withdrawal succeeds after original 7d.
      await time.increase(DEFAULT_UNBOND_DELAY + 10);
      await bond.connect(provider).withdraw();
    });
  });

  describe("governance: setSlasher", function () {
    it("owner can set the slasher", async function () {
      await expect(bond.connect(owner).setSlasher(slasher.address))
        .to.emit(bond, "SlasherUpdated")
        .withArgs(ethers.ZeroAddress, slasher.address);
      expect(await bond.slasher()).to.equal(slasher.address);
    });

    it("non-owner cannot set the slasher", async function () {
      await expect(
        bond.connect(provider).setSlasher(slasher.address)
      ).to.be.reverted;
    });

    it("accepts zero-address slasher (disables slashing)", async function () {
      await bond.connect(owner).setSlasher(slasher.address);
      await bond.connect(owner).setSlasher(ethers.ZeroAddress);
      expect(await bond.slasher()).to.equal(ethers.ZeroAddress);
    });
  });
});


// ── Phase 7 Task 2: slash() + bounty accrual ──────────────────────

describe("StakeBond — slash + bounty", function () {
  let bond, token;
  let owner, provider, challenger, slasher, foundation, other;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_UNBOND_DELAY = 7 * 24 * 60 * 60;
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const STANDARD_STAKE = 5_000n * ONE_FTNS;
  const STANDARD_SLASH_BPS = 5000;
  const PREMIUM_SLASH_BPS = 10000;
  const REASON_ID = ethers.keccak256(ethers.toUtf8Bytes("batch-1"));

  beforeEach(async function () {
    [owner, provider, challenger, slasher, foundation, other] =
      await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Bond = await ethers.getContractFactory("StakeBond");
    bond = await Bond.deploy(
      owner.address,
      await token.getAddress(),
      DEFAULT_UNBOND_DELAY
    );
    await bond.waitForDeployment();

    // Authorize the `slasher` signer to call slash().
    await bond.connect(owner).setSlasher(slasher.address);

    // Fund + approve the provider's stake.
    await token.mint(provider.address, PREMIUM_STAKE * 2n);
    await token
      .connect(provider)
      .approve(await bond.getAddress(), PREMIUM_STAKE * 2n);
  });

  describe("slash — authorization", function () {
    beforeEach(async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
    });

    it("reverts when called by non-slasher", async function () {
      await expect(
        bond
          .connect(other)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.be.revertedWithCustomError(bond, "CallerNotSlasher");
    });

    it("reverts when called by the owner (owner is not the slasher)", async function () {
      await expect(
        bond
          .connect(owner)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.be.revertedWithCustomError(bond, "CallerNotSlasher");
    });

    it("succeeds when called by the authorized slasher", async function () {
      await expect(
        bond
          .connect(slasher)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.emit(bond, "Slashed");
    });

    it("old slasher loses authorization after setSlasher rotation", async function () {
      await bond.connect(owner).setSlasher(other.address);
      await expect(
        bond
          .connect(slasher)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.be.revertedWithCustomError(bond, "CallerNotSlasher");
    });
  });

  describe("slash — amount + split", function () {
    it("slashes 50% of standard stake; splits 70/30 challenger/foundation", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);
      // 50% of 5K FTNS = 2.5K FTNS total slash
      // 70% → challenger bounty = 1.75K FTNS
      // 30% → Foundation reserve = 0.75K FTNS
      const expectedSlash = (STANDARD_STAKE * 5000n) / 10000n;
      const expectedChallengerBounty = (expectedSlash * 7000n) / 10000n;
      const expectedFoundationShare = expectedSlash - expectedChallengerBounty;

      await expect(
        bond
          .connect(slasher)
          .slash(provider.address, challenger.address, REASON_ID)
      )
        .to.emit(bond, "Slashed")
        .withArgs(
          provider.address,
          challenger.address,
          REASON_ID,
          expectedSlash,
          expectedChallengerBounty,
          expectedFoundationShare
        );

      // Stake amount reduced by slash.
      const stake = await bond.stakeOf(provider.address);
      expect(stake.amount).to.equal(STANDARD_STAKE - expectedSlash);

      // Pools credited.
      expect(await bond.slashedBountyPayable(challenger.address)).to.equal(
        expectedChallengerBounty
      );
      expect(await bond.foundationReserveBalance()).to.equal(
        expectedFoundationShare
      );
    });

    it("slashes 100% of premium stake", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);

      const stake = await bond.stakeOf(provider.address);
      expect(stake.amount).to.equal(0);

      // 70% → challenger = 17.5K FTNS
      // 30% → Foundation = 7.5K FTNS
      expect(await bond.slashedBountyPayable(challenger.address)).to.equal(
        (PREMIUM_STAKE * 7000n) / 10000n
      );
      expect(await bond.foundationReserveBalance()).to.equal(
        (PREMIUM_STAKE * 3000n) / 10000n
      );
    });

    it("routes 100% to Foundation when challenger == provider (self-slash)", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(provider.address, provider.address, REASON_ID);

      expect(await bond.slashedBountyPayable(provider.address)).to.equal(0);
      expect(await bond.foundationReserveBalance()).to.equal(PREMIUM_STAKE);
    });

    it("routes 100% to Foundation when challenger is zero-address", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(provider.address, ethers.ZeroAddress, REASON_ID);

      expect(await bond.foundationReserveBalance()).to.equal(PREMIUM_STAKE);
    });

    it("permits slashing during UNBONDING state", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond.connect(provider).requestUnbond();

      const stakeBefore = await bond.stakeOf(provider.address);
      expect(stakeBefore.status).to.equal(2); // UNBONDING

      await expect(
        bond
          .connect(slasher)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.emit(bond, "Slashed");

      const stakeAfter = await bond.stakeOf(provider.address);
      expect(stakeAfter.amount).to.equal(0); // 100% slashed
      expect(stakeAfter.status).to.equal(2); // still UNBONDING
    });

    it("rejects slashing a WITHDRAWN stake", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);
      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY + 10);
      await bond.connect(provider).withdraw();

      await expect(
        bond
          .connect(slasher)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.be.revertedWithCustomError(bond, "NotSlashable");
    });

    it("rejects slashing a never-bonded provider", async function () {
      await expect(
        bond
          .connect(slasher)
          .slash(other.address, challenger.address, REASON_ID)
      ).to.be.revertedWithCustomError(bond, "NotSlashable");
    });

    it("second slash operates on remaining stake", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);

      // First slash: 50% of 5K = 2.5K remaining 2.5K
      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);
      let stake = await bond.stakeOf(provider.address);
      expect(stake.amount).to.equal(STANDARD_STAKE / 2n);

      // Second slash: 50% of 2.5K = 1.25K remaining 1.25K
      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);
      stake = await bond.stakeOf(provider.address);
      expect(stake.amount).to.equal(STANDARD_STAKE / 4n);
    });

    it("rejects slash when stake fully drained", async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);
      // 100% slashed; amount = 0. Second slash: NothingToSlash.
      await expect(
        bond
          .connect(slasher)
          .slash(provider.address, challenger.address, REASON_ID)
      ).to.be.revertedWithCustomError(bond, "NothingToSlash");
    });
  });

  describe("slash + withdraw interaction", function () {
    it("post-slash withdraw returns only the remaining amount", async function () {
      await bond.connect(provider).bond(STANDARD_STAKE, STANDARD_SLASH_BPS);
      const providerBalBefore = await token.balanceOf(provider.address);

      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);
      // 50% slashed; 2.5K FTNS remains in the stake.

      await bond.connect(provider).requestUnbond();
      await time.increase(DEFAULT_UNBOND_DELAY + 10);
      await bond.connect(provider).withdraw();

      const providerBalAfter = await token.balanceOf(provider.address);
      expect(providerBalAfter - providerBalBefore).to.equal(STANDARD_STAKE / 2n);
    });
  });

  describe("claimBounty", function () {
    beforeEach(async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);
    });

    it("transfers accumulated bounty to the challenger", async function () {
      const expectedBounty = (PREMIUM_STAKE * 7000n) / 10000n;
      const balBefore = await token.balanceOf(challenger.address);

      await expect(bond.connect(challenger).claimBounty())
        .to.emit(bond, "BountyClaimed")
        .withArgs(challenger.address, expectedBounty);

      expect(await token.balanceOf(challenger.address)).to.equal(
        balBefore + expectedBounty
      );
      expect(await bond.slashedBountyPayable(challenger.address)).to.equal(0);
    });

    it("is idempotent — second claim reverts cleanly", async function () {
      await bond.connect(challenger).claimBounty();
      await expect(
        bond.connect(challenger).claimBounty()
      ).to.be.revertedWithCustomError(bond, "NothingToClaim");
    });

    it("reverts for someone with zero balance", async function () {
      await expect(
        bond.connect(other).claimBounty()
      ).to.be.revertedWithCustomError(bond, "NothingToClaim");
    });

    it("accumulates across multiple slashes", async function () {
      // First slash already fired in beforeEach. Bond a second provider
      // so we can accumulate another slash credited to the same challenger.
      await token.mint(other.address, PREMIUM_STAKE);
      await token
        .connect(other)
        .approve(await bond.getAddress(), PREMIUM_STAKE);
      await bond.connect(other).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(other.address, challenger.address, REASON_ID);

      // Claim should pay out sum of both slashes' bounties.
      const expected = 2n * ((PREMIUM_STAKE * 7000n) / 10000n);
      await bond.connect(challenger).claimBounty();
      expect(await token.balanceOf(challenger.address)).to.be.gte(expected);
    });
  });

  describe("drainFoundationReserve", function () {
    beforeEach(async function () {
      await bond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
      await bond
        .connect(slasher)
        .slash(provider.address, challenger.address, REASON_ID);
      // Foundation reserve now holds 30% of PREMIUM_STAKE.
    });

    it("reverts when foundation wallet not set", async function () {
      await expect(
        bond.connect(owner).drainFoundationReserve()
      ).to.be.revertedWithCustomError(
        bond,
        "FoundationReserveWalletNotSet"
      );
    });

    it("transfers full reserve to the configured wallet", async function () {
      await bond
        .connect(owner)
        .setFoundationReserveWallet(foundation.address);

      const expectedReserve = (PREMIUM_STAKE * 3000n) / 10000n;
      const balBefore = await token.balanceOf(foundation.address);

      await expect(bond.connect(owner).drainFoundationReserve())
        .to.emit(bond, "FoundationReserveDrained")
        .withArgs(foundation.address, expectedReserve);

      expect(await token.balanceOf(foundation.address)).to.equal(
        balBefore + expectedReserve
      );
      expect(await bond.foundationReserveBalance()).to.equal(0);
    });

    it("reverts on empty reserve (already drained)", async function () {
      await bond
        .connect(owner)
        .setFoundationReserveWallet(foundation.address);
      await bond.connect(owner).drainFoundationReserve();
      await expect(
        bond.connect(owner).drainFoundationReserve()
      ).to.be.revertedWithCustomError(bond, "FoundationReserveEmpty");
    });

    it("non-owner cannot drain", async function () {
      await bond
        .connect(owner)
        .setFoundationReserveWallet(foundation.address);
      await expect(
        bond.connect(provider).drainFoundationReserve()
      ).to.be.reverted;
    });
  });

  describe("setFoundationReserveWallet", function () {
    it("owner can set the wallet", async function () {
      await expect(
        bond.connect(owner).setFoundationReserveWallet(foundation.address)
      )
        .to.emit(bond, "FoundationReserveWalletUpdated")
        .withArgs(ethers.ZeroAddress, foundation.address);
      expect(await bond.foundationReserveWallet()).to.equal(
        foundation.address
      );
    });

    it("non-owner cannot set the wallet", async function () {
      await expect(
        bond.connect(provider).setFoundationReserveWallet(foundation.address)
      ).to.be.reverted;
    });
  });
});

function anyUint() {
  return (value) => typeof value === "bigint" || typeof value === "number";
}
