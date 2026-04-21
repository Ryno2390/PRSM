const { expect } = require("chai");
const { ethers } = require("hardhat");

// Phase 3.1 Task 2 — EscrowPool unit tests. Focus: per-requester balance
// accounting + authorization boundary on settleFromRequester.

describe("EscrowPool", function () {
  let pool, token;
  let owner, requester, other, registry, impostor;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const INITIAL_MINT = ONE_FTNS * 10000n;

  beforeEach(async function () {
    [owner, requester, other, registry, impostor] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    pool = await Pool.deploy(
      owner.address,
      await token.getAddress(),
      registry.address // use a signer address as a stand-in registry
    );
    await pool.waitForDeployment();

    await token.mint(requester.address, INITIAL_MINT);
    await token.mint(other.address, INITIAL_MINT);
  });

  describe("constructor", function () {
    it("sets owner, token, and registry", async function () {
      expect(await pool.owner()).to.equal(owner.address);
      expect(await pool.ftns()).to.equal(await token.getAddress());
      expect(await pool.settlementRegistry()).to.equal(registry.address);
    });

    it("allows zero initial registry (to be set later)", async function () {
      const Pool = await ethers.getContractFactory("EscrowPool");
      const p = await Pool.deploy(
        owner.address,
        await token.getAddress(),
        ethers.ZeroAddress
      );
      await p.waitForDeployment();
      expect(await p.settlementRegistry()).to.equal(ethers.ZeroAddress);
    });

    it("reverts on zero FTNS address", async function () {
      const Pool = await ethers.getContractFactory("EscrowPool");
      await expect(
        Pool.deploy(owner.address, ethers.ZeroAddress, registry.address)
      ).to.be.revertedWithCustomError(pool, "ZeroAddress");
    });
  });

  describe("deposit", function () {
    it("credits the caller's balance + transfers tokens in", async function () {
      const amount = ONE_FTNS * 100n;
      await token.connect(requester).approve(await pool.getAddress(), amount);

      const poolAddr = await pool.getAddress();
      const before = await token.balanceOf(poolAddr);

      await expect(pool.connect(requester).deposit(amount))
        .to.emit(pool, "Deposited")
        .withArgs(requester.address, amount, amount);

      expect(await pool.balanceOf(requester.address)).to.equal(amount);
      expect(await token.balanceOf(poolAddr)).to.equal(before + amount);
      expect(await token.balanceOf(requester.address)).to.equal(INITIAL_MINT - amount);
    });

    it("accumulates across multiple deposits", async function () {
      await token.connect(requester).approve(await pool.getAddress(), ONE_FTNS * 500n);
      await pool.connect(requester).deposit(ONE_FTNS * 100n);
      await pool.connect(requester).deposit(ONE_FTNS * 50n);
      await pool.connect(requester).deposit(ONE_FTNS * 25n);
      expect(await pool.balanceOf(requester.address)).to.equal(ONE_FTNS * 175n);
    });

    it("tracks per-depositor balances independently", async function () {
      await token.connect(requester).approve(await pool.getAddress(), ONE_FTNS * 100n);
      await token.connect(other).approve(await pool.getAddress(), ONE_FTNS * 200n);
      await pool.connect(requester).deposit(ONE_FTNS * 100n);
      await pool.connect(other).deposit(ONE_FTNS * 200n);
      expect(await pool.balanceOf(requester.address)).to.equal(ONE_FTNS * 100n);
      expect(await pool.balanceOf(other.address)).to.equal(ONE_FTNS * 200n);
    });

    it("reverts on zero amount", async function () {
      await expect(
        pool.connect(requester).deposit(0)
      ).to.be.revertedWithCustomError(pool, "ZeroAmount");
    });

    it("reverts when approval is insufficient", async function () {
      // No approval granted.
      await expect(
        pool.connect(requester).deposit(ONE_FTNS * 100n)
      ).to.be.reverted;
    });
  });

  describe("withdraw", function () {
    beforeEach(async function () {
      await token
        .connect(requester)
        .approve(await pool.getAddress(), ONE_FTNS * 500n);
      await pool.connect(requester).deposit(ONE_FTNS * 500n);
    });

    it("returns FTNS to the caller + reduces their balance", async function () {
      const amount = ONE_FTNS * 200n;
      const mintedToRequester = await token.balanceOf(requester.address);

      await expect(pool.connect(requester).withdraw(amount))
        .to.emit(pool, "Withdrawn")
        .withArgs(requester.address, amount, ONE_FTNS * 300n);

      expect(await pool.balanceOf(requester.address)).to.equal(ONE_FTNS * 300n);
      expect(await token.balanceOf(requester.address)).to.equal(
        mintedToRequester + amount
      );
    });

    it("allows full withdrawal", async function () {
      await pool.connect(requester).withdraw(ONE_FTNS * 500n);
      expect(await pool.balanceOf(requester.address)).to.equal(0);
    });

    it("reverts on zero amount", async function () {
      await expect(
        pool.connect(requester).withdraw(0)
      ).to.be.revertedWithCustomError(pool, "ZeroAmount");
    });

    it("reverts when amount exceeds balance", async function () {
      await expect(
        pool.connect(requester).withdraw(ONE_FTNS * 1000n)
      ).to.be.revertedWithCustomError(pool, "InsufficientBalance");
    });

    it("cannot withdraw from another user's balance", async function () {
      // `other` has no deposit; their attempt to withdraw fails.
      await expect(
        pool.connect(other).withdraw(ONE_FTNS * 100n)
      ).to.be.revertedWithCustomError(pool, "InsufficientBalance");
    });
  });

  describe("settleFromRequester — authorization", function () {
    beforeEach(async function () {
      await token
        .connect(requester)
        .approve(await pool.getAddress(), ONE_FTNS * 500n);
      await pool.connect(requester).deposit(ONE_FTNS * 500n);
    });

    it("succeeds when called by the registered registry", async function () {
      const amount = ONE_FTNS * 100n;
      const otherBefore = await token.balanceOf(other.address);

      await expect(
        pool
          .connect(registry)
          .settleFromRequester(requester.address, other.address, amount)
      )
        .to.emit(pool, "Settled")
        .withArgs(requester.address, other.address, amount, registry.address);

      expect(await pool.balanceOf(requester.address)).to.equal(ONE_FTNS * 400n);
      expect(await token.balanceOf(other.address)).to.equal(otherBefore + amount);
    });

    it("reverts when called by any other address", async function () {
      await expect(
        pool
          .connect(impostor)
          .settleFromRequester(requester.address, other.address, ONE_FTNS)
      ).to.be.revertedWithCustomError(pool, "CallerNotRegistry");
    });

    it("reverts when called by the owner (owner is not the registry)", async function () {
      await expect(
        pool
          .connect(owner)
          .settleFromRequester(requester.address, other.address, ONE_FTNS)
      ).to.be.revertedWithCustomError(pool, "CallerNotRegistry");
    });
  });

  describe("settleFromRequester — validations", function () {
    beforeEach(async function () {
      await token
        .connect(requester)
        .approve(await pool.getAddress(), ONE_FTNS * 500n);
      await pool.connect(requester).deposit(ONE_FTNS * 500n);
    });

    it("reverts on zero amount", async function () {
      await expect(
        pool
          .connect(registry)
          .settleFromRequester(requester.address, other.address, 0)
      ).to.be.revertedWithCustomError(pool, "ZeroAmount");
    });

    it("reverts on zero recipient", async function () {
      await expect(
        pool
          .connect(registry)
          .settleFromRequester(requester.address, ethers.ZeroAddress, ONE_FTNS)
      ).to.be.revertedWithCustomError(pool, "ZeroAddress");
    });

    it("reverts when requester's balance is insufficient", async function () {
      await expect(
        pool
          .connect(registry)
          .settleFromRequester(
            requester.address,
            other.address,
            ONE_FTNS * 1000n
          )
      ).to.be.revertedWithCustomError(pool, "InsufficientBalance");
    });

    it("reverts when requester has never deposited", async function () {
      await expect(
        pool
          .connect(registry)
          .settleFromRequester(impostor.address, other.address, ONE_FTNS)
      ).to.be.revertedWithCustomError(pool, "InsufficientBalance");
    });
  });

  describe("governance", function () {
    it("owner can update the settlement registry", async function () {
      await expect(pool.connect(owner).setSettlementRegistry(impostor.address))
        .to.emit(pool, "SettlementRegistryUpdated")
        .withArgs(registry.address, impostor.address);
      expect(await pool.settlementRegistry()).to.equal(impostor.address);
    });

    it("non-owner cannot update the registry", async function () {
      await expect(
        pool.connect(requester).setSettlementRegistry(impostor.address)
      ).to.be.reverted;
    });

    it("old registry loses authorization after an update", async function () {
      await token
        .connect(requester)
        .approve(await pool.getAddress(), ONE_FTNS * 100n);
      await pool.connect(requester).deposit(ONE_FTNS * 100n);

      // Swap registry.
      await pool.connect(owner).setSettlementRegistry(impostor.address);

      // Old registry can no longer settle.
      await expect(
        pool
          .connect(registry)
          .settleFromRequester(requester.address, other.address, ONE_FTNS)
      ).to.be.revertedWithCustomError(pool, "CallerNotRegistry");

      // New registry can.
      await expect(
        pool
          .connect(impostor)
          .settleFromRequester(requester.address, other.address, ONE_FTNS)
      ).to.emit(pool, "Settled");
    });

    it("owner can update the FTNS token address", async function () {
      const Token2 = await ethers.getContractFactory("MockERC20");
      const newToken = await Token2.deploy();
      await newToken.waitForDeployment();

      const oldAddr = await token.getAddress();
      await expect(pool.connect(owner).setFtnsToken(await newToken.getAddress()))
        .to.emit(pool, "FtnsTokenUpdated")
        .withArgs(oldAddr, await newToken.getAddress());
      expect(await pool.ftns()).to.equal(await newToken.getAddress());
    });

    it("non-owner cannot update the FTNS token", async function () {
      const Token2 = await ethers.getContractFactory("MockERC20");
      const newToken = await Token2.deploy();
      await expect(
        pool.connect(requester).setFtnsToken(await newToken.getAddress())
      ).to.be.reverted;
    });

    it("reverts when setting FTNS token to zero address", async function () {
      await expect(
        pool.connect(owner).setFtnsToken(ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(pool, "ZeroAddress");
    });
  });

  describe("reentrancy + misc", function () {
    it("balanceOf matches public mapping", async function () {
      await token.connect(requester).approve(await pool.getAddress(), ONE_FTNS);
      await pool.connect(requester).deposit(ONE_FTNS);
      expect(await pool.balanceOf(requester.address)).to.equal(
        await pool.balances(requester.address)
      );
    });
  });
});
