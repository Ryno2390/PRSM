/**
 * Comprehensive Test Suite for FTNSTokenSimple Contract
 * 
 * Critical Security Component - Requires 80% test coverage
 * Tests all security-critical functions for PRSM's core token contract
 */

const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

describe("FTNSTokenSimple", function () {
  // Test fixture for contract deployment
  async function deployFTNSTokenFixture() {
    const [owner, treasury, user1, user2, attacker, newAdmin] = await ethers.getSigners();

    // Deploy FTNSTokenSimple
    const FTNSTokenSimple = await ethers.getContractFactory("FTNSTokenSimple");
    const ftnsToken = await upgrades.deployProxy(
      FTNSTokenSimple,
      [owner.address, treasury.address],
      { initializer: "initialize", kind: "uups" }
    );

    return { ftnsToken, owner, treasury, user1, user2, attacker, newAdmin, FTNSTokenSimple };
  }

  describe("Contract Deployment and Initialization", function () {
    it("Should deploy with correct name and symbol", async function () {
      const { ftnsToken } = await loadFixture(deployFTNSTokenFixture);
      
      expect(await ftnsToken.name()).to.equal("PRSM Fungible Tokens for Node Support");
      expect(await ftnsToken.symbol()).to.equal("FTNS");
    });

    it("Should have correct decimals", async function () {
      const { ftnsToken } = await loadFixture(deployFTNSTokenFixture);
      
      expect(await ftnsToken.decimals()).to.equal(18);
    });

    it("Should mint initial supply to treasury", async function () {
      const { ftnsToken, treasury } = await loadFixture(deployFTNSTokenFixture);
      
      const expectedInitialSupply = ethers.parseEther("100000000"); // 100M tokens
      expect(await ftnsToken.balanceOf(treasury.address)).to.equal(expectedInitialSupply);
      expect(await ftnsToken.totalSupply()).to.equal(expectedInitialSupply);
    });

    it("Should set correct maximum supply", async function () {
      const { ftnsToken } = await loadFixture(deployFTNSTokenFixture);
      
      const expectedMaxSupply = ethers.parseEther("1000000000"); // 1B tokens
      expect(await ftnsToken.MAX_SUPPLY()).to.equal(expectedMaxSupply);
    });

    it("Should initialize with contract not paused", async function () {
      const { ftnsToken } = await loadFixture(deployFTNSTokenFixture);
      
      expect(await ftnsToken.paused()).to.be.false;
    });

    it("Should prevent double initialization", async function () {
      const { ftnsToken, owner, treasury } = await loadFixture(deployFTNSTokenFixture);
      
      await expect(
        ftnsToken.initialize(owner.address, treasury.address)
      ).to.be.revertedWith("Initializable: contract is already initialized");
    });
  });

  describe("Role-Based Access Control", function () {
    it("Should grant all roles to initial owner", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      const adminRole = await ftnsToken.DEFAULT_ADMIN_ROLE();
      const minterRole = await ftnsToken.MINTER_ROLE();
      const pauserRole = await ftnsToken.PAUSER_ROLE();
      const burnerRole = await ftnsToken.BURNER_ROLE();
      
      expect(await ftnsToken.hasRole(adminRole, owner.address)).to.be.true;
      expect(await ftnsToken.hasRole(minterRole, owner.address)).to.be.true;
      expect(await ftnsToken.hasRole(pauserRole, owner.address)).to.be.true;
      expect(await ftnsToken.hasRole(burnerRole, owner.address)).to.be.true;
    });

    it("Should allow admin to grant roles", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const minterRole = await ftnsToken.MINTER_ROLE();
      
      await ftnsToken.connect(owner).grantRole(minterRole, user1.address);
      expect(await ftnsToken.hasRole(minterRole, user1.address)).to.be.true;
    });

    it("Should allow admin to revoke roles", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const minterRole = await ftnsToken.MINTER_ROLE();
      
      // Grant then revoke
      await ftnsToken.connect(owner).grantRole(minterRole, user1.address);
      await ftnsToken.connect(owner).revokeRole(minterRole, user1.address);
      
      expect(await ftnsToken.hasRole(minterRole, user1.address)).to.be.false;
    });

    it("Should prevent non-admin from granting roles", async function () {
      const { ftnsToken, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      
      const minterRole = await ftnsToken.MINTER_ROLE();
      
      await expect(
        ftnsToken.connect(user1).grantRole(minterRole, user2.address)
      ).to.be.revertedWith(/AccessControl: account .* is missing role/);
    });

    it("Should allow role holders to renounce their own roles", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      const minterRole = await ftnsToken.MINTER_ROLE();
      
      await ftnsToken.connect(owner).renounceRole(minterRole, owner.address);
      expect(await ftnsToken.hasRole(minterRole, owner.address)).to.be.false;
    });
  });

  describe("Minting Functionality", function () {
    it("Should allow minter to mint tokens", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const mintAmount = ethers.parseEther("1000");
      await ftnsToken.connect(owner).mintReward(user1.address, mintAmount);
      
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(mintAmount);
    });

    it("Should increase total supply when minting", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const initialSupply = await ftnsToken.totalSupply();
      const mintAmount = ethers.parseEther("1000");
      
      await ftnsToken.connect(owner).mintReward(user1.address, mintAmount);
      
      expect(await ftnsToken.totalSupply()).to.equal(initialSupply + mintAmount);
    });

    it("Should prevent minting beyond max supply", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const maxSupply = await ftnsToken.MAX_SUPPLY();
      const currentSupply = await ftnsToken.totalSupply();
      const excessiveAmount = maxSupply - currentSupply + ethers.parseEther("1");
      
      await expect(
        ftnsToken.connect(owner).mintReward(user1.address, excessiveAmount)
      ).to.be.revertedWith("Would exceed max supply");
    });

    it("Should prevent non-minter from minting", async function () {
      const { ftnsToken, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      
      const mintAmount = ethers.parseEther("1000");
      
      await expect(
        ftnsToken.connect(user1).mintReward(user2.address, mintAmount)
      ).to.be.revertedWith(/AccessControl: account .* is missing role/);
    });

    it("Should allow minting to zero address to fail", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      const mintAmount = ethers.parseEther("1000");
      
      await expect(
        ftnsToken.connect(owner).mintReward(ethers.ZeroAddress, mintAmount)
      ).to.be.revertedWith("ERC20: mint to the zero address");
    });

    it("Should emit Transfer event when minting", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const mintAmount = ethers.parseEther("1000");
      
      await expect(
        ftnsToken.connect(owner).mintReward(user1.address, mintAmount)
      ).to.emit(ftnsToken, "Transfer")
       .withArgs(ethers.ZeroAddress, user1.address, mintAmount);
    });
  });

  describe("Burning Functionality", function () {
    beforeEach(async function () {
      // Setup: Give user1 some tokens to burn
      const { ftnsToken, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
    });

    it("Should allow burner to burn tokens from address", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(ftnsToken.treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      const burnAmount = ethers.parseEther("500");
      const initialBalance = await ftnsToken.balanceOf(user1.address);
      
      await ftnsToken.connect(owner).burnFrom(user1.address, burnAmount);
      
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(initialBalance - burnAmount);
    });

    it("Should decrease total supply when burning", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      const initialSupply = await ftnsToken.totalSupply();
      const burnAmount = ethers.parseEther("500");
      
      await ftnsToken.connect(owner).burnFrom(user1.address, burnAmount);
      
      expect(await ftnsToken.totalSupply()).to.equal(initialSupply - burnAmount);
    });

    it("Should prevent burning more than balance", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      const excessiveAmount = ethers.parseEther("1500");
      
      await expect(
        ftnsToken.connect(owner).burnFrom(user1.address, excessiveAmount)
      ).to.be.revertedWith("ERC20: burn amount exceeds balance");
    });

    it("Should prevent non-burner from burning", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      const burnAmount = ethers.parseEther("500");
      
      await expect(
        ftnsToken.connect(user2).burnFrom(user1.address, burnAmount)
      ).to.be.revertedWith(/AccessControl: account .* is missing role/);
    });

    it("Should emit Transfer event when burning", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      const burnAmount = ethers.parseEther("500");
      
      await expect(
        ftnsToken.connect(owner).burnFrom(user1.address, burnAmount)
      ).to.emit(ftnsToken, "Transfer")
       .withArgs(user1.address, ethers.ZeroAddress, burnAmount);
    });
  });

  describe("Pausable Functionality", function () {
    it("Should allow pauser to pause the contract", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      expect(await ftnsToken.paused()).to.be.true;
    });

    it("Should allow pauser to unpause the contract", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      await ftnsToken.connect(owner).unpause();
      
      expect(await ftnsToken.paused()).to.be.false;
    });

    it("Should prevent transfers when paused", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      
      await expect(
        ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("100"))
      ).to.be.revertedWith("Pausable: paused");
    });

    it("Should prevent approvals when paused", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      
      await expect(
        ftnsToken.connect(treasury).approve(user1.address, ethers.parseEther("100"))
      ).to.be.revertedWith("Pausable: paused");
    });

    it("Should prevent minting when paused", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      
      await expect(
        ftnsToken.connect(owner).mintReward(user1.address, ethers.parseEther("100"))
      ).to.be.revertedWith("Pausable: paused");
    });

    it("Should prevent burning when paused", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      // Give user1 tokens first
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      await ftnsToken.connect(owner).pause();
      
      await expect(
        ftnsToken.connect(owner).burnFrom(user1.address, ethers.parseEther("100"))
      ).to.be.revertedWith("Pausable: paused");
    });

    it("Should allow transfers after unpausing", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      await ftnsToken.connect(owner).unpause();
      
      // Should work after unpause
      await expect(
        ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("100"))
      ).to.not.be.reverted;
      
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(ethers.parseEther("100"));
    });

    it("Should prevent non-pauser from pausing", async function () {
      const { ftnsToken, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      await expect(
        ftnsToken.connect(user1).pause()
      ).to.be.revertedWith(/AccessControl: account .* is missing role/);
    });

    it("Should emit Paused event when pausing", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      await expect(
        ftnsToken.connect(owner).pause()
      ).to.emit(ftnsToken, "Paused")
       .withArgs(owner.address);
    });

    it("Should emit Unpaused event when unpausing", async function () {
      const { ftnsToken, owner } = await loadFixture(deployFTNSTokenFixture);
      
      await ftnsToken.connect(owner).pause();
      
      await expect(
        ftnsToken.connect(owner).unpause()
      ).to.emit(ftnsToken, "Unpaused")
       .withArgs(owner.address);
    });
  });

  describe("ERC20 Standard Functionality", function () {
    beforeEach(async function () {
      // Setup: Give users some tokens for testing
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      await ftnsToken.connect(treasury).transfer(user2.address, ethers.parseEther("3000"));
    });

    it("Should allow token transfers", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      
      const transferAmount = ethers.parseEther("1000");
      const initialBalance = await ftnsToken.balanceOf(user2.address);
      
      await ftnsToken.connect(user1).transfer(user2.address, transferAmount);
      
      expect(await ftnsToken.balanceOf(user2.address)).to.equal(initialBalance + transferAmount);
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(ethers.parseEther("4000"));
    });

    it("Should handle approvals correctly", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      
      const approveAmount = ethers.parseEther("2000");
      
      await ftnsToken.connect(user1).approve(user2.address, approveAmount);
      
      expect(await ftnsToken.allowance(user1.address, user2.address)).to.equal(approveAmount);
    });

    it("Should allow transferFrom with proper allowance", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      
      const approveAmount = ethers.parseEther("2000");
      const transferAmount = ethers.parseEther("1500");
      
      await ftnsToken.connect(user1).approve(user2.address, approveAmount);
      await ftnsToken.connect(user2).transferFrom(user1.address, user2.address, transferAmount);
      
      expect(await ftnsToken.balanceOf(user2.address)).to.equal(transferAmount);
      expect(await ftnsToken.allowance(user1.address, user2.address)).to.equal(approveAmount - transferAmount);
    });

    it("Should prevent transferFrom without sufficient allowance", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      
      const transferAmount = ethers.parseEther("1000");
      
      await expect(
        ftnsToken.connect(user2).transferFrom(user1.address, user2.address, transferAmount)
      ).to.be.revertedWith("ERC20: insufficient allowance");
    });

    it("Should prevent transfer of more than balance", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      const excessiveAmount = ethers.parseEther("1500");
      
      await expect(
        ftnsToken.connect(user1).transfer(user2.address, excessiveAmount)
      ).to.be.revertedWith("ERC20: transfer amount exceeds balance");
    });

    it("Should emit Transfer events", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      
      const transferAmount = ethers.parseEther("1000");
      
      await expect(
        ftnsToken.connect(user1).transfer(user2.address, transferAmount)
      ).to.emit(ftnsToken, "Transfer")
       .withArgs(user1.address, user2.address, transferAmount);
    });

    it("Should emit Approval events", async function () {
      const { ftnsToken, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("5000"));
      
      const approveAmount = ethers.parseEther("2000");
      
      await expect(
        ftnsToken.connect(user1).approve(user2.address, approveAmount)
      ).to.emit(ftnsToken, "Approval")
       .withArgs(user1.address, user2.address, approveAmount);
    });
  });

  describe("UUPS Upgradeability", function () {
    it("Should allow admin to authorize upgrades", async function () {
      const { ftnsToken, owner, FTNSTokenSimple } = await loadFixture(deployFTNSTokenFixture);
      
      // Deploy new implementation
      const FTNSTokenSimpleV2 = await ethers.getContractFactory("FTNSTokenSimple");
      
      // Should not revert for admin
      await expect(
        upgrades.upgradeProxy(ftnsToken, FTNSTokenSimpleV2.connect(owner))
      ).to.not.be.reverted;
    });

    it("Should prevent non-admin from authorizing upgrades", async function () {
      const { ftnsToken, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const FTNSTokenSimpleV2 = await ethers.getContractFactory("FTNSTokenSimple");
      
      await expect(
        upgrades.upgradeProxy(ftnsToken, FTNSTokenSimpleV2.connect(user1))
      ).to.be.revertedWith(/AccessControl: account .* is missing role/);
    });

    it("Should preserve state after upgrade", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      // Give user1 some tokens
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      const balanceBefore = await ftnsToken.balanceOf(user1.address);
      
      // Upgrade contract
      const FTNSTokenSimpleV2 = await ethers.getContractFactory("FTNSTokenSimple");
      await upgrades.upgradeProxy(ftnsToken, FTNSTokenSimpleV2.connect(owner));
      
      // State should be preserved
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(balanceBefore);
      expect(await ftnsToken.name()).to.equal("PRSM Fungible Tokens for Node Support");
    });
  });

  describe("Security and Edge Cases", function () {
    it("Should handle zero value transfers", async function () {
      const { ftnsToken, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      // Zero transfer should be allowed
      await expect(
        ftnsToken.connect(treasury).transfer(user1.address, 0)
      ).to.not.be.reverted;
    });

    it("Should handle zero value approvals", async function () {
      const { ftnsToken, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      await expect(
        ftnsToken.connect(treasury).approve(user1.address, 0)
      ).to.not.be.reverted;
    });

    it("Should prevent transfer to zero address", async function () {
      const { ftnsToken, treasury } = await loadFixture(deployFTNSTokenFixture);
      
      await expect(
        ftnsToken.connect(treasury).transfer(ethers.ZeroAddress, ethers.parseEther("100"))
      ).to.be.revertedWith("ERC20: transfer to the zero address");
    });

    it("Should handle role admin changes", async function () {
      const { ftnsToken, owner, newAdmin } = await loadFixture(deployFTNSTokenFixture);
      
      const adminRole = await ftnsToken.DEFAULT_ADMIN_ROLE();
      
      // Grant admin role to new admin
      await ftnsToken.connect(owner).grantRole(adminRole, newAdmin.address);
      
      // New admin should be able to grant roles
      const minterRole = await ftnsToken.MINTER_ROLE();
      await expect(
        ftnsToken.connect(newAdmin).grantRole(minterRole, newAdmin.address)
      ).to.not.be.reverted;
    });

    it("Should support ERC165 interface detection", async function () {
      const { ftnsToken } = await loadFixture(deployFTNSTokenFixture);
      
      // Should support AccessControl interface
      const accessControlInterfaceId = "0x7965db0b";
      expect(await ftnsToken.supportsInterface(accessControlInterfaceId)).to.be.true;
    });
  });

  describe("Integration and Real-world Scenarios", function () {
    it("Should handle multiple operations in sequence", async function () {
      const { ftnsToken, owner, treasury, user1, user2 } = await loadFixture(deployFTNSTokenFixture);
      
      // Complex sequence of operations
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("10000"));
      await ftnsToken.connect(owner).mintReward(user2.address, ethers.parseEther("5000"));
      await ftnsToken.connect(user1).approve(user2.address, ethers.parseEther("3000"));
      await ftnsToken.connect(user2).transferFrom(user1.address, user2.address, ethers.parseEther("2000"));
      await ftnsToken.connect(owner).burnFrom(user2.address, ethers.parseEther("1000"));
      
      // Verify final state
      expect(await ftnsToken.balanceOf(user1.address)).to.equal(ethers.parseEther("8000"));
      expect(await ftnsToken.balanceOf(user2.address)).to.equal(ethers.parseEther("6000"));
    });

    it("Should handle emergency pause scenario", async function () {
      const { ftnsToken, owner, treasury, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      // Normal operation
      await ftnsToken.connect(treasury).transfer(user1.address, ethers.parseEther("1000"));
      
      // Emergency pause
      await ftnsToken.connect(owner).pause();
      
      // All operations should be blocked
      await expect(ftnsToken.connect(user1).transfer(treasury.address, ethers.parseEther("100")))
        .to.be.revertedWith("Pausable: paused");
      await expect(ftnsToken.connect(owner).mintReward(user1.address, ethers.parseEther("100")))
        .to.be.revertedWith("Pausable: paused");
      
      // Resume operations
      await ftnsToken.connect(owner).unpause();
      
      // Should work again
      await expect(ftnsToken.connect(user1).transfer(treasury.address, ethers.parseEther("100")))
        .to.not.be.reverted;
    });

    it("Should handle role transfer scenario", async function () {
      const { ftnsToken, owner, newAdmin } = await loadFixture(deployFTNSTokenFixture);
      
      const adminRole = await ftnsToken.DEFAULT_ADMIN_ROLE();
      const minterRole = await ftnsToken.MINTER_ROLE();
      
      // Transfer admin role
      await ftnsToken.connect(owner).grantRole(adminRole, newAdmin.address);
      await ftnsToken.connect(owner).renounceRole(adminRole, owner.address);
      
      // New admin should have control
      await expect(
        ftnsToken.connect(newAdmin).grantRole(minterRole, newAdmin.address)
      ).to.not.be.reverted;
      
      // Old admin should not have control
      await expect(
        ftnsToken.connect(owner).grantRole(minterRole, owner.address)
      ).to.be.revertedWith(/AccessControl: account .* is missing role/);
    });

    it("Should handle large scale minting approaching max supply", async function () {
      const { ftnsToken, owner, user1 } = await loadFixture(deployFTNSTokenFixture);
      
      const maxSupply = await ftnsToken.MAX_SUPPLY();
      const currentSupply = await ftnsToken.totalSupply();
      const remainingSupply = maxSupply - currentSupply;
      
      // Mint close to max supply
      const largeAmount = remainingSupply - ethers.parseEther("1000");
      await ftnsToken.connect(owner).mintReward(user1.address, largeAmount);
      
      // Should be close to max supply
      expect(await ftnsToken.totalSupply()).to.be.closeTo(maxSupply - ethers.parseEther("1000"), ethers.parseEther("1"));
      
      // Should prevent exceeding max supply
      await expect(
        ftnsToken.connect(owner).mintReward(user1.address, ethers.parseEther("2000"))
      ).to.be.revertedWith("Would exceed max supply");
    });
  });
});