// Team D — D11 PoC: Owner mutates cross-wire mid-flight.
// The BatchSettlementRegistry / EscrowPool / StakeBond owner can swap the
// cross-references at any time with NO timelock and NO governance notice
// enforcement. This breaks "in-flight" batches in three concrete ways:
//
// 1. setEscrowPool while batches PENDING -> finalize fails / unrelated pool
// 2. setSettlementRegistry on EscrowPool -> existing registry's
//    finalizations revert with CallerNotRegistry
// 3. setSlasher on StakeBond -> challenges already in flight no longer slash
//
// Each is documented as governance-only with off-chain notice (PRSM-GOV-1
// §10.3 14-day notice), but the contract enforces NO on-chain delay.
// A compromised owner key (or a Foundation multisig with quorum) can move
// instantly. Combined with no pause (D4), there is no recovery window.

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("[Team D] D11 — Cross-wire owner mutation breaks in-flight batches", function () {
  let registry, pool, pool2, token, owner, provider, requester;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const WINDOW = 3 * 24 * 60 * 60;

  beforeEach(async function () {
    [owner, provider, requester] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    pool = await Pool.deploy(owner.address, await token.getAddress(), await registry.getAddress());
    await pool.waitForDeployment();

    pool2 = await Pool.deploy(owner.address, await token.getAddress(), await registry.getAddress());
    await pool2.waitForDeployment();

    await registry.connect(owner).setEscrowPool(await pool.getAddress());

    await token.mint(requester.address, ONE_FTNS * 100n);
    await token.connect(requester).approve(await pool.getAddress(), ONE_FTNS * 100n);
    await pool.connect(requester).deposit(ONE_FTNS * 100n);
  });

  it("REGRESSION (MEDIUM D-03): setEscrowPool after commit does NOT re-route in-flight batches — per-batch escrowPool snapshot is load-bearing", async function () {
    const root = ethers.keccak256(ethers.toUtf8Bytes("root"));
    const tx = await registry.connect(provider).commitBatch(
      requester.address, root, 1, ONE_FTNS * 10n, 0, ethers.ZeroHash, ""
    );
    const r = await tx.wait();
    const batchId = r.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // Owner reroutes to a different (empty) pool MID-FLIGHT.
    // Pre-D-03: this would have soft-bricked finalize against pool2.
    // Post-D-03: finalize reads b.escrowPoolAtCommit (= original pool)
    // and settles cleanly against it, ignoring the live mutable pointer.
    await registry.connect(owner).setEscrowPool(await pool2.getAddress());

    // Wait the challenge window.
    await time.increase(WINDOW + 1);

    // Finalize succeeds against the snapshot pool, not the live one.
    await expect(registry.finalizeBatch(batchId)).to.emit(registry, "BatchFinalized");

    // Funds moved from the original pool's accounting, not pool2.
    expect(await pool.balances(requester.address)).to.equal(ONE_FTNS * 90n); // 100 - 10
    expect(await pool2.balances(requester.address)).to.equal(0n);
  });

  it("REGRESSION (HIGH-6): owner cannot re-point pool.settlementRegistry mid-flight — field is immutable, attack vector closed", async function () {
    const root = ethers.keccak256(ethers.toUtf8Bytes("root2"));
    const tx = await registry.connect(provider).commitBatch(
      requester.address, root, 1, ONE_FTNS * 10n, 0, ethers.ZeroHash, ""
    );
    const r = await tx.wait();
    const batchId = r.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // Pre-fix: owner could have called pool.setSettlementRegistry(evil) here
    // to break finalize. Post-fix: setter does not exist; low-level call to
    // the old selector reverts because the function is not in the bytecode.
    expect(pool.setSettlementRegistry).to.be.undefined;
    const evilRegistry = ethers.Wallet.createRandom().address;
    const nukedSelector = ethers.id("setSettlementRegistry(address)").slice(0, 10);
    const data = nukedSelector + "000000000000000000000000" + evilRegistry.slice(2);
    await expect(
      owner.sendTransaction({ to: await pool.getAddress(), data })
    ).to.be.reverted;

    // settlementRegistry value is unchanged — still the legitimate registry.
    expect(await pool.settlementRegistry()).to.equal(await registry.getAddress());

    // Wait the challenge window — finalize succeeds because the cross-wire
    // could not be tampered with.
    await time.increase(WINDOW + 1);
    await expect(registry.finalizeBatch(batchId)).to.not.be.reverted;
  });
});
