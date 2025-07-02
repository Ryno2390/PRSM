const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");

describe("FTNSToken", function () {
  let ftnsToken;
  let owner, treasury, user1, user2, governance;

  const INITIAL_SUPPLY = ethers.utils.parseEther("100000000"); // 100M
  const MAX_SUPPLY = ethers.utils.parseEther("1000000000"); // 1B

  beforeEach(async function () {
    [owner, treasury, user1, user2, governance] = await ethers.getSigners();

    // Deploy FTNS Token
    const FTNSToken = await ethers.getContractFactory("FTNSToken");
    ftnsToken = await upgrades.deployProxy(
      FTNSToken,
      [owner.address, treasury.address],
      { initializer: "initialize", kind: "uups" }
    );
    await ftnsToken.deployed();
  });

  describe("Deployment", function () {
    it("Should set the correct name and symbol", async function () {
      expect(await ftnsToken.name()).to.equal("PRSM Fungible Tokens for Node Support");
      expect(await ftnsToken.symbol()).to.equal("FTNS");
    });

    it("Should mint initial supply to treasury", async function () {
      expect(await ftnsToken.balanceOf(treasury.address)).to.equal(INITIAL_SUPPLY);
      expect(await ftnsToken.totalSupply()).to.equal(INITIAL_SUPPLY);
    });

    it("Should set correct roles", async function () {
      const adminRole = await ftnsToken.DEFAULT_ADMIN_ROLE();
      const minterRole = await ftnsToken.MINTER_ROLE();
      
      expect(await ftnsToken.hasRole(adminRole, owner.address)).to.be.true;
      expect(await ftnsToken.hasRole(minterRole, owner.address)).to.be.true;
    });
  });

  describe("Balance Management", function () {
    beforeEach(async function () {
      // Transfer some tokens to user1 for testing
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("1000"));
    });

    it("Should track liquid balance correctly", async function () {
      const balance = await ftnsToken.balanceOf(user1.address);
      const liquidBalance = await ftnsToken.liquidBalance(user1.address);
      
      expect(liquidBalance).to.equal(balance);
    });

    it("Should lock tokens correctly", async function () {
      const lockAmount = ethers.utils.parseEther("100");
      const duration = 30 * 24 * 3600; // 30 days
      
      await ftnsToken.connect(user1).lockTokens(lockAmount, duration);
      
      expect(await ftnsToken.lockedBalance(user1.address)).to.equal(lockAmount);
      expect(await ftnsToken.liquidBalance(user1.address)).to.equal(
        ethers.utils.parseEther("900")
      );
    });

    it("Should not allow transfer of locked tokens", async function () {
      const lockAmount = ethers.utils.parseEther("900");
      const duration = 30 * 24 * 3600;
      
      await ftnsToken.connect(user1).lockTokens(lockAmount, duration);
      
      // Should not be able to transfer more than liquid balance
      await expect(
        ftnsToken.connect(user1).transfer(user2.address, ethers.utils.parseEther("200"))
      ).to.be.revertedWith("Transfer exceeds liquid balance");
    });
  });

  describe("Staking for Governance", function () {
    beforeEach(async function () {
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("10000"));
    });

    it("Should stake tokens for governance", async function () {
      const stakeAmount = ethers.utils.parseEther("1000");
      const duration = 365 * 24 * 3600; // 1 year
      
      await ftnsToken.connect(user1).stakeForGovernance(stakeAmount, duration);
      
      expect(await ftnsToken.stakedBalance(user1.address)).to.equal(stakeAmount);
      
      // Check voting power (should be higher due to time multiplier)
      const votingPower = await ftnsToken.getVotingPower(user1.address);
      expect(votingPower).to.be.gt(stakeAmount);
    });

    it("Should not allow unstaking before duration", async function () {
      const stakeAmount = ethers.utils.parseEther("1000");
      const duration = 365 * 24 * 3600;
      
      await ftnsToken.connect(user1).stakeForGovernance(stakeAmount, duration);
      
      await expect(
        ftnsToken.connect(user1).unstakeTokens(stakeAmount)
      ).to.be.revertedWith("Staking period not ended");
    });
  });

  describe("Context Allocation", function () {
    beforeEach(async function () {
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("1000"));
    });

    it("Should allocate context", async function () {
      const allocateAmount = ethers.utils.parseEther("100");
      
      await ftnsToken.connect(user1).allocateContext(allocateAmount);
      
      const [allocated, used] = await ftnsToken.getContextInfo(user1.address);
      expect(allocated).to.equal(allocateAmount);
      expect(used).to.equal(0);
      
      // Should be locked
      expect(await ftnsToken.lockedBalance(user1.address)).to.equal(allocateAmount);
    });

    it("Should use context (minter role)", async function () {
      const allocateAmount = ethers.utils.parseEther("100");
      const useAmount = ethers.utils.parseEther("50");
      
      await ftnsToken.connect(user1).allocateContext(allocateAmount);
      
      // Use context (as minter)
      await ftnsToken.connect(owner).useContext(user1.address, useAmount);
      
      const [allocated, used] = await ftnsToken.getContextInfo(user1.address);
      expect(allocated).to.equal(allocateAmount.sub(useAmount));
      expect(used).to.equal(useAmount);
      
      // Tokens should be burned
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(
        ethers.utils.parseEther("950")
      );
    });
  });

  describe("Rewards and Minting", function () {
    it("Should mint rewards (minter role)", async function () {
      const rewardAmount = ethers.utils.parseEther("100");
      
      await ftnsToken.connect(owner).mintReward(user1.address, rewardAmount);
      
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(rewardAmount);
    });

    it("Should not exceed max supply", async function () {
      const excessiveAmount = MAX_SUPPLY; // Would exceed max supply
      
      await expect(
        ftnsToken.connect(owner).mintReward(user1.address, excessiveAmount)
      ).to.be.revertedWith("Would exceed max supply");
    });

    it("Should distribute dividends", async function () {
      const recipients = [user1.address, user2.address];
      const amounts = [ethers.utils.parseEther("50"), ethers.utils.parseEther("75")];
      
      await ftnsToken.connect(owner).distributeDividends(recipients, amounts);
      
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(amounts[0]);
      expect(await ftnsToken.balanceOf(user2.address)).to.equal(amounts[1]);
    });
  });

  describe("Administrative Functions", function () {
    it("Should pause and unpause", async function () {
      await ftnsToken.connect(owner).pause();
      
      // Should not be able to transfer when paused
      await expect(
        ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("100"))
      ).to.be.revertedWith("Pausable: paused");
      
      await ftnsToken.connect(owner).unpause();
      
      // Should work after unpause
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("100"));
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(ethers.utils.parseEther("100"));
    });

    it("Should burn tokens", async function () {
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("100"));
      
      const burnAmount = ethers.utils.parseEther("50");
      await ftnsToken.connect(owner).burnFrom(user1.address, burnAmount);
      
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(ethers.utils.parseEther("50"));
    });
  });

  describe("Account Info", function () {
    it("Should return comprehensive account info", async function () {
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.utils.parseEther("1000"));
      
      // Lock some tokens
      await ftnsToken.connect(user1).lockTokens(ethers.utils.parseEther("200"), 30 * 24 * 3600);
      
      // Stake some tokens
      await ftnsToken.connect(user1).stakeForGovernance(ethers.utils.parseEther("300"), 365 * 24 * 3600);
      
      // Allocate context
      await ftnsToken.connect(user1).allocateContext(ethers.utils.parseEther("100"));
      
      const accountInfo = await ftnsToken.getAccountInfo(user1.address);
      
      expect(accountInfo.liquid).to.equal(ethers.utils.parseEther("400")); // 1000 - 200 - 300 - 100
      expect(accountInfo.locked).to.equal(ethers.utils.parseEther("300")); // 200 + 100 context
      expect(accountInfo.staked).to.equal(ethers.utils.parseEther("300"));
      expect(accountInfo.total).to.equal(ethers.utils.parseEther("1000"));
      expect(accountInfo.contextAllocated).to.equal(ethers.utils.parseEther("100"));
    });
  });

  describe("Access Control", function () {
    it("Should not allow non-minter to mint", async function () {
      await expect(
        ftnsToken.connect(user1).mintReward(user2.address, ethers.utils.parseEther("100"))
      ).to.be.revertedWith("AccessControl:");
    });

    it("Should not allow non-pauser to pause", async function () {
      await expect(
        ftnsToken.connect(user1).pause()
      ).to.be.revertedWith("AccessControl:");
    });

    it("Should not allow non-admin to upgrade", async function () {
      const FTNSTokenV2 = await ethers.getContractFactory("FTNSToken");
      
      await expect(
        upgrades.upgradeProxy(ftnsToken.address, FTNSTokenV2.connect(user1))
      ).to.be.revertedWith("AccessControl:");
    });
  });
});