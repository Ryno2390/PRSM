// Team D — D4 PoC: Pause-coverage gap. The four core treasury contracts
// (BatchSettlementRegistry, EscrowPool, StakeBond, RoyaltyDistributor) have
// NO pause mechanism. There is no admin button to halt value movement during
// an exploit-in-progress. The only kill-switch is FTNSToken's pause, which
// blocks ALL transfers globally including legitimate user activity, and is
// administratively heavy.
//
// This test demonstrates that the "pause cover the full attack surface"
// invariant (team-prompt §Stated invariants #3) is FALSE for the treasury
// layer — there is no such surface.

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("[Team D] D4 — No pause surface on treasury contracts", function () {
  let registry, pool, stakeBond, royaltyDistributor, provReg, token;
  let owner;

  beforeEach(async function () {
    [owner] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    pool = await Pool.deploy(owner.address, await token.getAddress(), ethers.ZeroAddress);
    await pool.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, 3 * 24 * 60 * 60);
    await registry.waitForDeployment();

    const StakeBond = await ethers.getContractFactory("StakeBond");
    stakeBond = await StakeBond.deploy(owner.address, await token.getAddress(), 7 * 24 * 60 * 60);
    await stakeBond.waitForDeployment();

    const ProvReg = await ethers.getContractFactory("ProvenanceRegistry");
    provReg = await ProvReg.deploy();
    await provReg.waitForDeployment();

    const RD = await ethers.getContractFactory("RoyaltyDistributor");
    royaltyDistributor = await RD.deploy(
      await token.getAddress(),
      await provReg.getAddress(),
      owner.address
    );
    await royaltyDistributor.waitForDeployment();
  });

  it("BatchSettlementRegistry exposes no pause()/unpause()", async function () {
    expect(registry.pause).to.be.undefined;
    expect(registry.paused).to.be.undefined;
  });

  it("EscrowPool exposes no pause()/unpause()", async function () {
    expect(pool.pause).to.be.undefined;
    expect(pool.paused).to.be.undefined;
  });

  it("StakeBond exposes no pause()/unpause()", async function () {
    expect(stakeBond.pause).to.be.undefined;
    expect(stakeBond.paused).to.be.undefined;
  });

  it("RoyaltyDistributor exposes no pause()/unpause()", async function () {
    expect(royaltyDistributor.pause).to.be.undefined;
    expect(royaltyDistributor.paused).to.be.undefined;
  });

  it("source contains no whenNotPaused modifier on any state-mutating fn", async function () {
    const fs = require("fs");
    const path = require("path");
    const root = path.resolve(__dirname, "../../contracts");
    for (const f of [
      "BatchSettlementRegistry.sol",
      "EscrowPool.sol",
      "StakeBond.sol",
      "RoyaltyDistributor.sol",
    ]) {
      const src = fs.readFileSync(path.join(root, f), "utf8");
      // No literal "whenNotPaused" appears in any treasury contract.
      expect(src.includes("whenNotPaused"), `${f} unexpectedly has whenNotPaused`).to.equal(false);
    }
  });
});
