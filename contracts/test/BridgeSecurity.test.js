/**
 * Comprehensive Test Suite for Bridge Security Contract
 * 
 * Tests multi-signature verification for cross-chain bridge operations
 * including EIP-712 typed data signing, replay protection, and threshold validation.
 */

const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

describe("BridgeSecurity", function () {
  // Test fixture for contract deployment
  async function deployBridgeSecurityFixture() {
    const [admin, validator1, validator2, validator3, validator4, validator5, user1, user2, attacker] = 
      await ethers.getSigners();

    // Initial validators (3-of-5 threshold)
    const initialValidators = [
      validator1.address,
      validator2.address,
      validator3.address,
      validator4.address,
      validator5.address
    ];

    // Deploy BridgeSecurity
    const BridgeSecurity = await ethers.getContractFactory("BridgeSecurity");
    const bridgeSecurity = await upgrades.deployProxy(
      BridgeSecurity,
      [admin.address, 3, initialValidators],
      { initializer: "initialize", kind: "uups" }
    );

    return { 
      bridgeSecurity, 
      admin, 
      validator1, validator2, validator3, validator4, validator5,
      user1, user2, attacker,
      initialValidators,
      BridgeSecurity 
    };
  }

  // Helper to create EIP-712 typed data for BridgeMessage
  function createBridgeMessageTypes() {
    return {
      BridgeMessage: [
        { name: "recipient", type: "address" },
        { name: "amount", type: "uint256" },
        { name: "sourceChainId", type: "uint256" },
        { name: "sourceTxId", type: "bytes32" },
        { name: "nonce", type: "uint256" }
      ]
    };
  }

  // Helper to get EIP-712 domain
  async function getDomain(contract) {
    return {
      name: "PRSM Bridge Security",
      version: "1",
      chainId: (await ethers.provider.getNetwork()).chainId,
      verifyingContract: await contract.getAddress()
    };
  }

  // Helper to create a bridge message
  function createBridgeMessage(recipient, amount, sourceChainId, sourceTxId, nonce) {
    return {
      recipient,
      amount,
      sourceChainId,
      sourceTxId,
      nonce
    };
  }

  // Helper to sign a bridge message
  async function signBridgeMessage(signer, contract, message) {
    const domain = await getDomain(contract);
    const types = createBridgeMessageTypes();
    
    const signature = await signer.signTypedData(domain, types, message);
    const { r, s, v } = ethers.Signature.from(signature);
    
    return {
      validator: signer.address,
      r,
      s,
      v
    };
  }

  // Helper to create multiple signatures
  async function createSignatures(contract, message, signers) {
    const signatures = [];
    for (const signer of signers) {
      const sig = await signBridgeMessage(signer, contract, message);
      signatures.push(sig);
    }
    return signatures;
  }

  describe("Contract Deployment and Initialization", function () {
    it("Should deploy with correct initial configuration", async function () {
      const { bridgeSecurity, admin, initialValidators } = await loadFixture(deployBridgeSecurityFixture);
      
      expect(await bridgeSecurity.signatureThreshold()).to.equal(3);
      expect(await bridgeSecurity.totalValidators()).to.equal(5);
      
      // Check all validators are registered
      for (const validator of initialValidators) {
        expect(await bridgeSecurity.isValidator(validator)).to.be.true;
      }
    });

    it("Should set correct admin role", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      const adminRole = await bridgeSecurity.DEFAULT_ADMIN_ROLE();
      expect(await bridgeSecurity.hasRole(adminRole, admin.address)).to.be.true;
    });

    it("Should revert with zero admin address", async function () {
      const BridgeSecurity = await ethers.getContractFactory("BridgeSecurity");
      
      await expect(
        upgrades.deployProxy(
          BridgeSecurity,
          [ethers.ZeroAddress, 3, []],
          { initializer: "initialize", kind: "uups" }
        )
      ).to.be.reverted;
    });

    it("Should revert with invalid threshold", async function () {
      const [admin, validator1] = await ethers.getSigners();
      const BridgeSecurity = await ethers.getContractFactory("BridgeSecurity");
      
      // Threshold of 0
      await expect(
        upgrades.deployProxy(
          BridgeSecurity,
          [admin.address, 0, [validator1.address]],
          { initializer: "initialize", kind: "uups" }
        )
      ).to.be.reverted;

      // Threshold greater than validators
      await expect(
        upgrades.deployProxy(
          BridgeSecurity,
          [admin.address, 5, [validator1.address]],
          { initializer: "initialize", kind: "uups" }
        )
      ).to.be.reverted;
    });

    it("Should prevent double initialization", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.initialize(admin.address, 3, [])
      ).to.be.revertedWith("Initializable: contract is already initialized");
    });
  });

  describe("Validator Management", function () {
    it("Should allow admin to add validator", async function () {
      const { bridgeSecurity, admin, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await bridgeSecurity.connect(admin).addValidator(user1.address);
      
      expect(await bridgeSecurity.isValidator(user1.address)).to.be.true;
      expect(await bridgeSecurity.totalValidators()).to.equal(6);
    });

    it("Should emit ValidatorAdded event", async function () {
      const { bridgeSecurity, admin, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(bridgeSecurity.connect(admin).addValidator(user1.address))
        .to.emit(bridgeSecurity, "ValidatorAdded")
        .withArgs(user1.address, 6);
    });

    it("Should revert when adding zero address as validator", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(admin).addValidator(ethers.ZeroAddress)
      ).to.be.reverted;
    });

    it("Should revert when adding duplicate validator", async function () {
      const { bridgeSecurity, admin, validator1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(admin).addValidator(validator1.address)
      ).to.be.reverted;
    });

    it("Should allow admin to remove validator", async function () {
      const { bridgeSecurity, admin, validator1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await bridgeSecurity.connect(admin).removeValidator(validator1.address);
      
      expect(await bridgeSecurity.isValidator(validator1.address)).to.be.false;
      expect(await bridgeSecurity.totalValidators()).to.equal(4);
    });

    it("Should emit ValidatorRemoved event", async function () {
      const { bridgeSecurity, admin, validator1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(bridgeSecurity.connect(admin).removeValidator(validator1.address))
        .to.emit(bridgeSecurity, "ValidatorRemoved")
        .withArgs(validator1.address, 4);
    });

    it("Should revert when removing non-existent validator", async function () {
      const { bridgeSecurity, admin, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(admin).removeValidator(user1.address)
      ).to.be.reverted;
    });

    it("Should adjust threshold when validators removed below threshold", async function () {
      const { bridgeSecurity, admin, validator1, validator2, validator3, validator4 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      // Remove validators until threshold needs adjustment
      await bridgeSecurity.connect(admin).removeValidator(validator1.address);
      await bridgeSecurity.connect(admin).removeValidator(validator2.address);
      
      // Threshold should still be 3 with 3 validators
      expect(await bridgeSecurity.signatureThreshold()).to.equal(3);
      
      // Remove one more - threshold should adjust to 2
      await bridgeSecurity.connect(admin).removeValidator(validator3.address);
      expect(await bridgeSecurity.signatureThreshold()).to.equal(2);
    });

    it("Should prevent non-admin from adding validators", async function () {
      const { bridgeSecurity, user1, user2 } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(user1).addValidator(user2.address)
      ).to.be.reverted;
    });
  });

  describe("Signature Threshold Management", function () {
    it("Should allow admin to update threshold", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      await bridgeSecurity.connect(admin).setSignatureThreshold(4);
      
      expect(await bridgeSecurity.signatureThreshold()).to.equal(4);
    });

    it("Should emit SignatureThresholdUpdated event", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(bridgeSecurity.connect(admin).setSignatureThreshold(4))
        .to.emit(bridgeSecurity, "SignatureThresholdUpdated")
        .withArgs(3, 4);
    });

    it("Should revert with zero threshold", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(admin).setSignatureThreshold(0)
      ).to.be.reverted;
    });

    it("Should revert with threshold exceeding validator count", async function () {
      const { bridgeSecurity, admin } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(admin).setSignatureThreshold(10)
      ).to.be.reverted;
    });

    it("Should prevent non-admin from updating threshold", async function () {
      const { bridgeSecurity, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      await expect(
        bridgeSecurity.connect(user1).setSignatureThreshold(2)
      ).to.be.reverted;
    });
  });

  describe("EIP-712 Message Hashing", function () {
    it("Should compute consistent message hash", async function () {
      const { bridgeSecurity, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1, // Ethereum mainnet
        ethers.id("test-tx-123"),
        1
      );
      
      const hash1 = await bridgeSecurity.hashBridgeMessage(message);
      const hash2 = await bridgeSecurity.hashBridgeMessage(message);
      
      expect(hash1).to.equal(hash2);
      expect(hash1).to.have.length(66); // 32 bytes as hex string with 0x prefix
    });

    it("Should produce different hashes for different messages", async function () {
      const { bridgeSecurity, user1, user2 } = await loadFixture(deployBridgeSecurityFixture);
      
      const message1 = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const message2 = createBridgeMessage(
        user2.address, // Different recipient
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const hash1 = await bridgeSecurity.hashBridgeMessage(message1);
      const hash2 = await bridgeSecurity.hashBridgeMessage(message2);
      
      expect(hash1).to.not.equal(hash2);
    });

    it("Should include nonce in hash for replay protection", async function () {
      const { bridgeSecurity, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      const message1 = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const message2 = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        2 // Different nonce
      );
      
      const hash1 = await bridgeSecurity.hashBridgeMessage(message1);
      const hash2 = await bridgeSecurity.hashBridgeMessage(message2);
      
      expect(hash1).to.not.equal(hash2);
    });

    it("Should include source chain ID in hash", async function () {
      const { bridgeSecurity, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      const message1 = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1, // Ethereum
        ethers.id("test-tx-123"),
        1
      );
      
      const message2 = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        137, // Polygon
        ethers.id("test-tx-123"),
        1
      );
      
      const hash1 = await bridgeSecurity.hashBridgeMessage(message1);
      const hash2 = await bridgeSecurity.hashBridgeMessage(message2);
      
      expect(hash1).to.not.equal(hash2);
    });
  });

  describe("Signature Verification", function () {
    it("Should verify valid signatures from registered validators", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.true;
    });

    it("Should emit BridgeTransactionVerified event", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      await expect(
        bridgeSecurity.verifyBridgeSignatures(message, signatures)
      ).to.emit(bridgeSecurity, "BridgeTransactionVerified");
    });

    it("Should reject signatures below threshold", async function () {
      const { bridgeSecurity, validator1, validator2, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      // Only 2 signatures, threshold is 3
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2]
      );
      
      await expect(
        bridgeSecurity.verifyBridgeSignatures(message, signatures)
      ).to.be.reverted;
    });

    it("Should reject signatures from non-validators", async function () {
      const { bridgeSecurity, validator1, validator2, user1, attacker } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      // Attacker is not a validator
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, attacker]
      );
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      // Should fail because only 2 valid signatures
      expect(isValid).to.be.false;
    });

    it("Should reject duplicate signatures from same validator", async function () {
      const { bridgeSecurity, validator1, validator2, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      // Create signature from validator1 twice
      const sig1 = await signBridgeMessage(validator1, bridgeSecurity, message);
      const sig2 = await signBridgeMessage(validator2, bridgeSecurity, message);
      const sig3 = await signBridgeMessage(validator1, bridgeSecurity, message); // Duplicate!
      
      const signatures = [sig1, sig2, sig3];
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      // Should fail because only 2 unique valid signatures
      expect(isValid).to.be.false;
    });

    it("Should reject invalid cryptographic signatures", async function () {
      const { bridgeSecurity, validator1, validator2, user1, attacker } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      // Create valid signatures
      const sig1 = await signBridgeMessage(validator1, bridgeSecurity, message);
      const sig2 = await signBridgeMessage(validator2, bridgeSecurity, message);
      
      // Create a forged signature (attacker signs but claims it's from validator3)
      const forgedSig = await signBridgeMessage(attacker, bridgeSecurity, message);
      forgedSig.validator = (await ethers.getSigners())[4].address; // validator3's address
      
      const signatures = [sig1, sig2, forgedSig];
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.false;
    });

    it("Should prevent replay attacks", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      // First verification should succeed
      const [messageHash1, isValid1] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      expect(isValid1).to.be.true;
      
      // Second verification should fail (already processed)
      await expect(
        bridgeSecurity.verifyBridgeSignatures(message, signatures)
      ).to.be.reverted;
    });

    it("Should mark transaction as processed after verification", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const messageHash = await bridgeSecurity.hashBridgeMessage(message);
      
      // Should not be processed initially
      expect(await bridgeSecurity.processedBridgeTx(messageHash)).to.be.false;
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      await bridgeSecurity.verifyBridgeSignatures(message, signatures);
      
      // Should be processed now
      expect(await bridgeSecurity.processedBridgeTx(messageHash)).to.be.true;
    });

    it("Should mark nonce as used after verification", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const sourceChainId = 1;
      const nonce = 1;
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        sourceChainId,
        ethers.id("test-tx-123"),
        nonce
      );
      
      // Nonce should not be used initially
      expect(await bridgeSecurity.usedNonces(sourceChainId, nonce)).to.be.false;
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      await bridgeSecurity.verifyBridgeSignatures(message, signatures);
      
      // Nonce should be used now
      expect(await bridgeSecurity.usedNonces(sourceChainId, nonce)).to.be.true;
    });
  });

  describe("Check Signatures (View Function)", function () {
    it("Should return correct validation status without state changes", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      const [isValid, validCount] = await bridgeSecurity.checkBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.true;
      expect(validCount).to.equal(3);
    });

    it("Should not modify state when checking signatures", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const messageHash = await bridgeSecurity.hashBridgeMessage(message);
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      // Check signatures (view call)
      await bridgeSecurity.checkBridgeSignatures(message, signatures);
      
      // Transaction should still be unprocessed
      expect(await bridgeSecurity.processedBridgeTx(messageHash)).to.be.false;
      
      // Now verify for real
      await bridgeSecurity.verifyBridgeSignatures(message, signatures);
      
      // Now it should be processed
      expect(await bridgeSecurity.processedBridgeTx(messageHash)).to.be.true;
    });

    it("Should return false for already processed transactions", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      // Verify first time
      await bridgeSecurity.verifyBridgeSignatures(message, signatures);
      
      // Check should return false (already processed)
      const [isValid, validCount] = await bridgeSecurity.checkBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.false;
      expect(validCount).to.equal(0);
    });
  });

  describe("Domain Separator", function () {
    it("Should return consistent domain separator", async function () {
      const { bridgeSecurity } = await loadFixture(deployBridgeSecurityFixture);
      
      const domain1 = await bridgeSecurity.domainSeparator();
      const domain2 = await bridgeSecurity.domainSeparator();
      
      expect(domain1).to.equal(domain2);
    });

    it("Should have correct domain separator format", async function () {
      const { bridgeSecurity } = await loadFixture(deployBridgeSecurityFixture);
      
      const domain = await bridgeSecurity.domainSeparator();
      
      // Domain separator should be 32 bytes
      expect(domain).to.have.length(66); // 0x + 64 hex chars
    });
  });

  describe("Configuration View Functions", function () {
    it("Should return correct configuration", async function () {
      const { bridgeSecurity } = await loadFixture(deployBridgeSecurityFixture);
      
      const [threshold, validators] = await bridgeSecurity.getConfiguration();
      
      expect(threshold).to.equal(3);
      expect(validators).to.equal(5);
    });
  });

  describe("Edge Cases and Security", function () {
    it("Should handle exact threshold signatures", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      // Exactly 3 signatures (the threshold)
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.true;
    });

    it("Should handle more than threshold signatures", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, validator4, validator5, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      // 5 signatures (more than threshold of 3)
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3, validator4, validator5]
      );
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.true;
    });

    it("Should reject empty signatures array", async function () {
      const { bridgeSecurity, user1 } = await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      await expect(
        bridgeSecurity.verifyBridgeSignatures(message, [])
      ).to.be.reverted;
    });

    it("Should handle large amounts correctly", async function () {
      const { bridgeSecurity, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        user1.address,
        ethers.MaxUint256, // Maximum possible amount
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.true;
    });

    it("Should handle zero address recipient", async function () {
      const { bridgeSecurity, validator1, validator2, validator3 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      const message = createBridgeMessage(
        ethers.ZeroAddress,
        ethers.parseEther("100"),
        1,
        ethers.id("test-tx-123"),
        1
      );
      
      const signatures = await createSignatures(
        bridgeSecurity,
        message,
        [validator1, validator2, validator3]
      );
      
      // Should still verify signatures (validation happens at bridge level)
      const [messageHash, isValid] = await bridgeSecurity.verifyBridgeSignatures(
        message,
        signatures
      );
      
      expect(isValid).to.be.true;
    });
  });

  describe("Upgradeability", function () {
    it("Should allow admin to upgrade contract", async function () {
      const { bridgeSecurity, admin, BridgeSecurity } = await loadFixture(deployBridgeSecurityFixture);
      
      // Deploy new implementation
      const BridgeSecurityV2 = await ethers.getContractFactory("BridgeSecurity");
      const newImplementation = await BridgeSecurityV2.deploy();
      
      // Upgrade
      await expect(
        bridgeSecurity.connect(admin).upgradeTo(await newImplementation.getAddress())
      ).to.not.be.reverted;
    });

    it("Should prevent non-admin from upgrading", async function () {
      const { bridgeSecurity, attacker, BridgeSecurity } = await loadFixture(deployBridgeSecurityFixture);
      
      const BridgeSecurityV2 = await ethers.getContractFactory("BridgeSecurity");
      const newImplementation = await BridgeSecurityV2.deploy();
      
      await expect(
        bridgeSecurity.connect(attacker).upgradeTo(await newImplementation.getAddress())
      ).to.be.reverted;
    });

    it("Should preserve state after upgrade", async function () {
      const { bridgeSecurity, admin, validator1, validator2, validator3, user1 } = 
        await loadFixture(deployBridgeSecurityFixture);
      
      // Record initial state
      const initialThreshold = await bridgeSecurity.signatureThreshold();
      const initialValidators = await bridgeSecurity.totalValidators();
      
      // Upgrade
      const BridgeSecurityV2 = await ethers.getContractFactory("BridgeSecurity");
      const newImplementation = await BridgeSecurityV2.deploy();
      await bridgeSecurity.connect(admin).upgradeTo(await newImplementation.getAddress());
      
      // Check state preserved
      expect(await bridgeSecurity.signatureThreshold()).to.equal(initialThreshold);
      expect(await bridgeSecurity.totalValidators()).to.equal(initialValidators);
    });
  });
});
