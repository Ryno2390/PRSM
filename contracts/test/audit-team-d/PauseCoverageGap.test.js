// Team D — D-02 REGRESSION GUARD (formerly PoC for "no pause surface").
//
// Original finding asserted that none of BatchSettlementRegistry,
// EscrowPool, StakeBond, RoyaltyDistributor exposed a pause mechanism —
// the "pause covers the full attack surface" invariant (#3) was FALSE.
//
// REMEDIATION SHIPPED 2026-05-05:
//   - BSR + EscrowPool + StakeBond all inherit OZ Pausable.
//   - State-mutating functions gated on `whenNotPaused`.
//   - Admin setters intentionally NOT gated (rotation during incident).
//   - RoyaltyDistributor pause comes with the HIGH-1 v2 re-deploy
//     (v1 is non-Ownable + non-upgradeable).
//
// This test now asserts the inverse: the three audit-bundle contracts
// DO expose pause()/unpause(), the source does contain whenNotPaused,
// and the live-mainnet-deployed RoyaltyDistributor v1 still does NOT
// (documenting the v2-redeploy follow-up).

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("[Team D] D-02 regression: pause surface present on audit-bundle contracts", function () {
  let registry, pool, stakeBond, royaltyDistributor, provReg, token;
  let owner;

  beforeEach(async function () {
    [owner] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, 3 * 24 * 60 * 60);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    pool = await Pool.deploy(owner.address, await token.getAddress(), await registry.getAddress());
    await pool.waitForDeployment();

    const StakeBond = await ethers.getContractFactory("StakeBond");
    stakeBond = await StakeBond.deploy(owner.address, await token.getAddress(), 7 * 24 * 60 * 60, await registry.getAddress());
    await stakeBond.waitForDeployment();

    const ProvReg = await ethers.getContractFactory("ProvenanceRegistry");
    provReg = await ProvReg.deploy();
    await provReg.waitForDeployment();

    const RD = await ethers.getContractFactory("RoyaltyDistributor");
    royaltyDistributor = await RD.deploy(
      await token.getAddress(),
      await provReg.getAddress(),
      owner.address,
      owner.address
    );
    await royaltyDistributor.waitForDeployment();
  });

  it("BatchSettlementRegistry NOW exposes pause()/unpause()/paused()", async function () {
    expect(registry.pause).to.not.be.undefined;
    expect(registry.unpause).to.not.be.undefined;
    expect(registry.paused).to.not.be.undefined;
    expect(await registry.paused()).to.equal(false);
  });

  it("EscrowPool NOW exposes pause()/unpause()/paused()", async function () {
    expect(pool.pause).to.not.be.undefined;
    expect(pool.unpause).to.not.be.undefined;
    expect(pool.paused).to.not.be.undefined;
    expect(await pool.paused()).to.equal(false);
  });

  it("StakeBond NOW exposes pause()/unpause()/paused()", async function () {
    expect(stakeBond.pause).to.not.be.undefined;
    expect(stakeBond.unpause).to.not.be.undefined;
    expect(stakeBond.paused).to.not.be.undefined;
    expect(await stakeBond.paused()).to.equal(false);
  });

  it("RoyaltyDistributor v1 (live mainnet) still has no pause — pending HIGH-1 v2 re-deploy", async function () {
    // Documents the deferred fix per consolidated.md §6 + §3 HIGH-1.
    // RoyaltyDistributor at 0x3E82...D6c2 is non-Ownable + non-upgradeable.
    // v2 re-deploy with both the burn implementation AND Pausable comes
    // in Week 2 of the remediation sprint.
    expect(royaltyDistributor.pause).to.be.undefined;
    expect(royaltyDistributor.paused).to.be.undefined;
  });

  it("source contains whenNotPaused on at least one state-mutating fn in each audit-bundle contract", async function () {
    const fs = require("fs");
    const path = require("path");
    const root = path.resolve(__dirname, "../../contracts");
    for (const f of [
      "BatchSettlementRegistry.sol",
      "EscrowPool.sol",
      "StakeBond.sol",
    ]) {
      const src = fs.readFileSync(path.join(root, f), "utf8");
      expect(src.includes("whenNotPaused"), `${f} should have whenNotPaused`).to.equal(true);
    }
  });
});
