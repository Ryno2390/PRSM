const { expect } = require("chai");
const { ethers } = require("hardhat");

// Phase 7-storage Task 6 — KeyDistribution.sol.
// Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 6.

describe("KeyDistribution — payment-gated key release", function () {
  let keyDist, royalty;
  let owner, publisher, consumer, other;

  const CONTENT_HASH =
    "0x1111111111111111111111111111111111111111111111111111111111111111";
  const ENCRYPTED_KEY = "0x" + "ab".repeat(96); // 96-byte opaque envelope
  const FEE = ethers.parseUnits("1", 18); // 1 FTNS

  async function deploy() {
    [owner, publisher, consumer, other] = await ethers.getSigners();

    const Royalty = await ethers.getContractFactory("MockRoyaltyVerifier");
    royalty = await Royalty.deploy();

    const KeyDist = await ethers.getContractFactory("KeyDistribution");
    keyDist = await KeyDist.deploy(owner.address);

    return { keyDist, royalty };
  }

  async function deposit(contentHash = CONTENT_HASH, fee = FEE) {
    return keyDist
      .connect(publisher)
      .depositKey(contentHash, ENCRYPTED_KEY, await royalty.getAddress(), fee);
  }

  // -------------------------------------------------------------------------
  // depositKey
  // -------------------------------------------------------------------------

  describe("depositKey", function () {
    it("stores the record with publisher address", async function () {
      await deploy();
      await deposit();
      const r = await keyDist.records(CONTENT_HASH);
      expect(r.publisher).to.equal(publisher.address);
      expect(r.active).to.equal(true);
      expect(r.releaseFeeFtnsWei).to.equal(FEE);
    });

    it("emits KeyDeposited", async function () {
      await deploy();
      await expect(deposit())
        .to.emit(keyDist, "KeyDeposited")
        .withArgs(
          CONTENT_HASH,
          publisher.address,
          await royalty.getAddress(),
          FEE
        );
    });

    it("rejects empty encryptedKey", async function () {
      await deploy();
      await expect(
        keyDist
          .connect(publisher)
          .depositKey(
            CONTENT_HASH,
            "0x",
            await royalty.getAddress(),
            FEE
          )
      ).to.be.revertedWithCustomError(keyDist, "EmptyEncryptedKey");
    });

    it("rejects zero royalty address", async function () {
      await deploy();
      await expect(
        keyDist
          .connect(publisher)
          .depositKey(CONTENT_HASH, ENCRYPTED_KEY, ethers.ZeroAddress, FEE)
      ).to.be.revertedWithCustomError(keyDist, "InvalidAddress");
    });

    it("rejects zero fee", async function () {
      await deploy();
      await expect(
        keyDist
          .connect(publisher)
          .depositKey(
            CONTENT_HASH,
            ENCRYPTED_KEY,
            await royalty.getAddress(),
            0
          )
      ).to.be.revertedWithCustomError(keyDist, "ZeroFee");
    });

    it("rejects duplicate deposit on same contentHash", async function () {
      await deploy();
      await deposit();
      await expect(deposit()).to.be.revertedWithCustomError(
        keyDist,
        "KeyAlreadyDeposited"
      );
    });
  });

  // -------------------------------------------------------------------------
  // release — happy path
  // -------------------------------------------------------------------------

  describe("release happy path", function () {
    it("emits KeyReleased with the encrypted key blob when payment is verified", async function () {
      await deploy();
      await deposit();
      await royalty.setPaid(consumer.address, CONTENT_HASH, FEE, true);

      await expect(keyDist.connect(consumer).release(CONTENT_HASH, consumer.address))
        .to.emit(keyDist, "KeyReleased")
        .withArgs(CONTENT_HASH, consumer.address, ENCRYPTED_KEY);
    });

    it("allows a third party to trigger release on behalf of the payer", async function () {
      await deploy();
      await deposit();
      await royalty.setPaid(consumer.address, CONTENT_HASH, FEE, true);

      // `other` triggers the release, but the event recipient is still
      // `consumer`. Third parties cannot leak keys to arbitrary addresses.
      await expect(
        keyDist.connect(other).release(CONTENT_HASH, consumer.address)
      )
        .to.emit(keyDist, "KeyReleased")
        .withArgs(CONTENT_HASH, consumer.address, ENCRYPTED_KEY);
    });

    it("supports release to multiple distinct consumers", async function () {
      await deploy();
      await deposit();
      await royalty.setPaid(consumer.address, CONTENT_HASH, FEE, true);
      await royalty.setPaid(other.address, CONTENT_HASH, FEE, true);

      await expect(keyDist.connect(consumer).release(CONTENT_HASH, consumer.address))
        .to.emit(keyDist, "KeyReleased");
      await expect(keyDist.connect(other).release(CONTENT_HASH, other.address))
        .to.emit(keyDist, "KeyReleased");
    });

    it("is permissive on repeated release calls for the same recipient", async function () {
      await deploy();
      await deposit();
      await royalty.setPaid(consumer.address, CONTENT_HASH, FEE, true);
      await keyDist.connect(consumer).release(CONTENT_HASH, consumer.address);
      // Second release emits a second event — plan §8.5: on-chain
      // revocation is impossible and re-emission is harmless.
      await expect(
        keyDist.connect(consumer).release(CONTENT_HASH, consumer.address)
      ).to.emit(keyDist, "KeyReleased");
    });
  });

  // -------------------------------------------------------------------------
  // release — failure paths
  // -------------------------------------------------------------------------

  describe("release failure paths", function () {
    it("reverts PaymentNotVerified when the royalty verifier returns false", async function () {
      await deploy();
      await deposit();
      // Payment not set → verifier returns false.
      await expect(
        keyDist.connect(consumer).release(CONTENT_HASH, consumer.address)
      ).to.be.revertedWithCustomError(keyDist, "PaymentNotVerified");
    });

    it("reverts KeyNotFound for a content hash that was never deposited", async function () {
      await deploy();
      const UNKNOWN =
        "0x9999999999999999999999999999999999999999999999999999999999999999";
      await expect(
        keyDist.connect(consumer).release(UNKNOWN, consumer.address)
      ).to.be.revertedWithCustomError(keyDist, "KeyNotFound");
    });

    it("reverts with zero recipient", async function () {
      await deploy();
      await deposit();
      await expect(
        keyDist.connect(consumer).release(CONTENT_HASH, ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(keyDist, "InvalidAddress");
    });

    it("does not leak keys when payment is made against the wrong fee amount", async function () {
      await deploy();
      await deposit();
      // Payer paid a different fee amount — verifyPayment returns false.
      await royalty.setPaid(consumer.address, CONTENT_HASH, ethers.parseUnits("2", 18), true);
      await expect(
        keyDist.connect(consumer).release(CONTENT_HASH, consumer.address)
      ).to.be.revertedWithCustomError(keyDist, "PaymentNotVerified");
    });
  });

  // -------------------------------------------------------------------------
  // deauthorize
  // -------------------------------------------------------------------------

  describe("deauthorize", function () {
    it("is publisher-only", async function () {
      await deploy();
      await deposit();
      await expect(
        keyDist.connect(other).deauthorize(CONTENT_HASH)
      ).to.be.revertedWithCustomError(keyDist, "NotPublisher");
    });

    it("reverts for never-deposited content", async function () {
      await deploy();
      await expect(
        keyDist.connect(publisher).deauthorize(CONTENT_HASH)
      ).to.be.revertedWithCustomError(keyDist, "KeyNotFound");
    });

    it("blocks future release calls", async function () {
      await deploy();
      await deposit();
      await royalty.setPaid(consumer.address, CONTENT_HASH, FEE, true);

      // First release works.
      await keyDist.connect(consumer).release(CONTENT_HASH, consumer.address);

      await keyDist.connect(publisher).deauthorize(CONTENT_HASH);

      // Subsequent release blocked — even with payment verified.
      await expect(
        keyDist.connect(consumer).release(CONTENT_HASH, consumer.address)
      ).to.be.revertedWithCustomError(keyDist, "KeyNotActive");
    });

    it("emits KeyDeauthorized", async function () {
      await deploy();
      await deposit();
      await expect(keyDist.connect(publisher).deauthorize(CONTENT_HASH))
        .to.emit(keyDist, "KeyDeauthorized")
        .withArgs(CONTENT_HASH, publisher.address);
    });
  });
});
