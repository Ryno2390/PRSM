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

    const Pool = await ethers.getContractFactory("EscrowPool");
    pool = await Pool.deploy(owner.address, await token.getAddress(), ethers.ZeroAddress);
    await pool.waitForDeployment();

    pool2 = await Pool.deploy(owner.address, await token.getAddress(), ethers.ZeroAddress);
    await pool2.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, WINDOW);
    await registry.waitForDeployment();

    await pool.connect(owner).setSettlementRegistry(await registry.getAddress());
    await pool2.connect(owner).setSettlementRegistry(await registry.getAddress());
    await registry.connect(owner).setEscrowPool(await pool.getAddress());

    await token.mint(requester.address, ONE_FTNS * 100n);
    await token.connect(requester).approve(await pool.getAddress(), ONE_FTNS * 100n);
    await pool.connect(requester).deposit(ONE_FTNS * 100n);
  });

  it("setEscrowPool after commit: finalize attempts settle from EMPTY new pool, reverts", async function () {
    const root = ethers.keccak256(ethers.toUtf8Bytes("root"));
    const tx = await registry.connect(provider).commitBatch(
      requester.address, root, 1, ONE_FTNS * 10n, 0, ethers.ZeroHash, ""
    );
    const r = await tx.wait();
    const batchId = r.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // Owner reroutes to a different (empty) pool MID-FLIGHT, with no notice.
    await registry.connect(owner).setEscrowPool(await pool2.getAddress());

    // Wait the challenge window.
    await time.increase(WINDOW + 1);

    // Finalize tries to pull from pool2 — requester has no balance there.
    await expect(
      registry.finalizeBatch(batchId)
    ).to.be.revertedWithCustomError(pool2, "InsufficientBalance");

    // Provider's settlement is now soft-bricked: the batch is stuck in
    // PENDING with the wrong pool wired. Owner intervention required.
  });

  it("setSettlementRegistry on pool: legitimate registry's finalize reverts CallerNotRegistry", async function () {
    const root = ethers.keccak256(ethers.toUtf8Bytes("root2"));
    const tx = await registry.connect(provider).commitBatch(
      requester.address, root, 1, ONE_FTNS * 10n, 0, ethers.ZeroHash, ""
    );
    const r = await tx.wait();
    const batchId = r.logs.find(l => l.fragment && l.fragment.name === "BatchCommitted").args[0];

    // Owner of pool changes its registry pointer to an arbitrary address.
    const evilRegistry = ethers.Wallet.createRandom().address;
    await pool.connect(owner).setSettlementRegistry(evilRegistry);

    await time.increase(WINDOW + 1);

    // Real registry can no longer settle.
    await expect(
      registry.finalizeBatch(batchId)
    ).to.be.revertedWithCustomError(pool, "CallerNotRegistry");
  });
});
