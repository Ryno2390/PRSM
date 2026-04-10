const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ProvenanceRegistry", function () {
  let registry;
  let owner, creator, otherCreator;

  beforeEach(async function () {
    [owner, creator, otherCreator] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("ProvenanceRegistry");
    registry = await Registry.deploy();
    await registry.waitForDeployment();
  });

  describe("registerContent", function () {
    it("registers new content with creator address as msg.sender", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("test-content"));
      const royaltyRateBps = 800; // 8%
      const metadataUri = "ipfs://QmTest";

      await expect(
        registry.connect(creator).registerContent(contentHash, royaltyRateBps, metadataUri)
      )
        .to.emit(registry, "ContentRegistered")
        .withArgs(contentHash, creator.address, royaltyRateBps, metadataUri);

      const content = await registry.contents(contentHash);
      expect(content.creator).to.equal(creator.address);
      expect(content.royaltyRateBps).to.equal(royaltyRateBps);
      expect(content.metadataUri).to.equal(metadataUri);
      expect(content.registeredAt).to.be.gt(0);
    });

    it("rejects duplicate registration", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("dup"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://A");
      await expect(
        registry.connect(otherCreator).registerContent(contentHash, 500, "ipfs://B")
      ).to.be.revertedWith("Already registered");
    });

    it("rejects royalty rate above MAX_ROYALTY_RATE_BPS (9800)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("over"));
      await expect(
        registry.connect(creator).registerContent(contentHash, 9801, "ipfs://X")
      ).to.be.revertedWith("Rate exceeds max");
    });

    it("rejects royalty rate above 100% (10001)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("very-high"));
      await expect(
        registry.connect(creator).registerContent(contentHash, 10001, "ipfs://X")
      ).to.be.reverted;
    });

    it("accepts royalty rate exactly at MAX_ROYALTY_RATE_BPS (9800)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("max"));
      await registry.connect(creator).registerContent(contentHash, 9800, "ipfs://M");
      const c = await registry.contents(contentHash);
      expect(c.royaltyRateBps).to.equal(9800);
    });

    it("exposes MAX_ROYALTY_RATE_BPS as a public constant", async function () {
      expect(await registry.MAX_ROYALTY_RATE_BPS()).to.equal(9800);
    });

    it("accepts zero royalty rate (free content)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("free"));
      await registry.connect(creator).registerContent(contentHash, 0, "ipfs://free");
      const content = await registry.contents(contentHash);
      expect(content.royaltyRateBps).to.equal(0);
    });
  });

  describe("isRegistered", function () {
    it("returns false for unregistered content", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("none"));
      expect(await registry.isRegistered(contentHash)).to.equal(false);
    });

    it("returns true after registration", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("yes"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://Y");
      expect(await registry.isRegistered(contentHash)).to.equal(true);
    });
  });

  describe("getCreatorAndRate", function () {
    it("returns just creator and rate, no metadata", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("slim"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

      const [c, rate] = await registry.getCreatorAndRate(contentHash);
      expect(c).to.equal(creator.address);
      expect(rate).to.equal(800);
    });

    it("returns zero address for unregistered content", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("missing-slim"));
      const [c, rate] = await registry.getCreatorAndRate(contentHash);
      expect(c).to.equal(ethers.ZeroAddress);
      expect(rate).to.equal(0);
    });
  });

  describe("transferContentOwnership", function () {
    it("allows current creator to transfer", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("xfer"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

      await expect(
        registry.connect(creator).transferContentOwnership(contentHash, otherCreator.address)
      )
        .to.emit(registry, "OwnershipTransferred")
        .withArgs(contentHash, creator.address, otherCreator.address);

      const content = await registry.contents(contentHash);
      expect(content.creator).to.equal(otherCreator.address);
    });

    it("rejects transfer by non-owner", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("xfer2"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        registry.connect(otherCreator).transferContentOwnership(contentHash, otherCreator.address)
      ).to.be.revertedWith("Not creator");
    });

    it("rejects transfer to zero address", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("xfer3"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        registry.connect(creator).transferContentOwnership(contentHash, ethers.ZeroAddress)
      ).to.be.revertedWith("Zero address");
    });
  });
});
