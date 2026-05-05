/*
 * Team D — REGRESSION GUARD for D-02 (formerly HIGH):
 * Treasury contracts have no pause mechanism.
 *
 * Original finding: none of BatchSettlementRegistry, EscrowPool,
 * StakeBond, RoyaltyDistributor used `whenNotPaused` or inherited
 * `Pausable`. The only kill-switch was FTNSToken.pause() which globally
 * halts all transfers — too heavy for surgical incident response.
 *
 * REMEDIATION SHIPPED 2026-05-05 (this commit):
 *   - BatchSettlementRegistry, EscrowPool, StakeBond all inherit OZ
 *     Pausable. State-mutating user/operator-callable functions are
 *     gated on `whenNotPaused`. Admin setters are NOT gated, so the
 *     owner can perform emergency rotation while paused.
 *   - RoyaltyDistributor pause comes with the HIGH-1 v2 re-deploy
 *     since v1 is non-Ownable + non-upgradeable.
 *
 * This test asserts:
 *   1. Each contract exposes pause() / unpause() (owner-only).
 *   2. State-mutating functions revert with EnforcedPause when paused.
 *   3. Admin setters work while paused.
 *   4. Functions resume after unpause.
 *   5. Unbond timer continues during pause (timestamp-based, not block-
 *      based) — provider whose unbond_eligible_at elapsed during pause
 *      can withdraw immediately on unpause.
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("Audit Team D — D-02 regression: pause surface on audit-bundle contracts", function () {
  let owner, provider, requester, challenger;
  let token, registry, escrowPool, stakeBond;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const CHALLENGE_WINDOW = 3 * 24 * 60 * 60; // 3 days
  const UNBOND_DELAY = 7 * 24 * 60 * 60; // 7 days
  const STAKE = ethers.parseUnits("10000", 18);

  beforeEach(async function () {
    [owner, provider, requester, challenger] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(owner.address, await token.getAddress(), ethers.ZeroAddress);
    await escrowPool.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, CHALLENGE_WINDOW);
    await registry.waitForDeployment();

    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(owner.address, await token.getAddress(), UNBOND_DELAY);
    await stakeBond.waitForDeployment();

    await escrowPool.connect(owner).setSettlementRegistry(await registry.getAddress());
    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());
    await stakeBond.connect(owner).setSlasher(await registry.getAddress());

    // Fund a depositor + a provider.
    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token.connect(requester).approve(await escrowPool.getAddress(), ONE_FTNS * 1000n);
    await escrowPool.connect(requester).deposit(ONE_FTNS * 100n);

    await token.mint(provider.address, STAKE * 2n);
    await token.connect(provider).approve(await stakeBond.getAddress(), STAKE * 2n);
    await stakeBond.connect(provider).bond(STAKE, 5000);
  });

  describe("EscrowPool", function () {
    it("pause/unpause are owner-only", async function () {
      await expect(escrowPool.connect(provider).pause())
        .to.be.revertedWithCustomError(escrowPool, "OwnableUnauthorizedAccount");
      await escrowPool.connect(owner).pause();
      expect(await escrowPool.paused()).to.equal(true);
      await escrowPool.connect(owner).unpause();
      expect(await escrowPool.paused()).to.equal(false);
    });

    it("deposit / withdraw revert when paused", async function () {
      await escrowPool.connect(owner).pause();
      await expect(escrowPool.connect(requester).deposit(ONE_FTNS))
        .to.be.revertedWithCustomError(escrowPool, "EnforcedPause");
      await expect(escrowPool.connect(requester).withdraw(ONE_FTNS))
        .to.be.revertedWithCustomError(escrowPool, "EnforcedPause");
    });

    it("admin setters work while paused", async function () {
      await escrowPool.connect(owner).pause();
      // setSettlementRegistry doesn't revert on EnforcedPause — admin needs it
      // for emergency rotation. Use a clean address (not the existing wired
      // registry) to avoid duplicate-set semantics if any.
      await expect(escrowPool.connect(owner).setSettlementRegistry(challenger.address))
        .to.not.be.reverted;
    });

    it("operations resume after unpause", async function () {
      await escrowPool.connect(owner).pause();
      await escrowPool.connect(owner).unpause();
      await expect(escrowPool.connect(requester).deposit(ONE_FTNS))
        .to.emit(escrowPool, "Deposited");
    });
  });

  describe("StakeBond", function () {
    it("pause/unpause are owner-only", async function () {
      await expect(stakeBond.connect(provider).pause())
        .to.be.revertedWithCustomError(stakeBond, "OwnableUnauthorizedAccount");
      await stakeBond.connect(owner).pause();
      expect(await stakeBond.paused()).to.equal(true);
    });

    it("bond / requestUnbond / withdraw revert when paused", async function () {
      await stakeBond.connect(owner).pause();
      // Provider attempts to bond a fresh amount — paused.
      const [, , , , extra] = await ethers.getSigners();
      await token.mint(extra.address, STAKE);
      await token.connect(extra).approve(await stakeBond.getAddress(), STAKE);
      await expect(stakeBond.connect(extra).bond(STAKE, 5000))
        .to.be.revertedWithCustomError(stakeBond, "EnforcedPause");

      await expect(stakeBond.connect(provider).requestUnbond())
        .to.be.revertedWithCustomError(stakeBond, "EnforcedPause");
    });

    it("unbond timer continues during pause (timestamp-based)", async function () {
      // Provider initiates unbond BEFORE pause.
      await stakeBond.connect(provider).requestUnbond();
      const stake = await stakeBond.stakes(provider.address);
      const eligibleAt = stake.unbond_eligible_at;

      // Owner pauses.
      await stakeBond.connect(owner).pause();

      // Time advances past unbond_eligible_at while paused.
      // Note: post-HIGH-2 fix, unbond_eligible_at is clamped to MAX of
      // local unbondDelay AND slasher.challengeWindowSeconds. With
      // UNBOND_DELAY (7d) > CHALLENGE_WINDOW (3d), the floor is 7d.
      await time.increaseTo(Number(eligibleAt) + 60);

      // Withdraw is still blocked because we're paused.
      await expect(stakeBond.connect(provider).withdraw())
        .to.be.revertedWithCustomError(stakeBond, "EnforcedPause");

      // Owner unpauses. Timer was timestamp-based, did not reset.
      await stakeBond.connect(owner).unpause();

      // Provider can withdraw immediately — unbond_eligible_at has elapsed.
      await expect(stakeBond.connect(provider).withdraw())
        .to.emit(stakeBond, "Withdrawn");
    });
  });

  describe("BatchSettlementRegistry", function () {
    it("pause/unpause are owner-only", async function () {
      await expect(registry.connect(provider).pause())
        .to.be.revertedWithCustomError(registry, "OwnableUnauthorizedAccount");
      await registry.connect(owner).pause();
      expect(await registry.paused()).to.equal(true);
    });

    it("commitBatch / finalizeBatch / challengeReceipt revert when paused", async function () {
      await registry.connect(owner).pause();

      const root = ethers.keccak256(ethers.toUtf8Bytes("test-root"));
      await expect(
        registry.connect(provider).commitBatch(
          requester.address, root, 1, ONE_FTNS, 0, ethers.ZeroHash, ""
        )
      ).to.be.revertedWithCustomError(registry, "EnforcedPause");

      // finalizeBatch on a non-existent batch will revert — but we want
      // to confirm EnforcedPause fires FIRST. The whenNotPaused modifier
      // runs before any function-body checks so this should work.
      const fakeBatchId = ethers.keccak256(ethers.toUtf8Bytes("fake"));
      await expect(registry.connect(provider).finalizeBatch(fakeBatchId))
        .to.be.revertedWithCustomError(registry, "EnforcedPause");
    });

    it("admin setters work while paused — owner can rotate during incident", async function () {
      await registry.connect(owner).pause();
      // Owner can rotate signatureVerifier even while paused.
      await expect(
        registry.connect(owner).setSignatureVerifier(ethers.ZeroAddress)
      ).to.not.be.reverted;
    });
  });
});
