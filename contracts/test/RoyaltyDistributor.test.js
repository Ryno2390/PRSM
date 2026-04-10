const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("RoyaltyDistributor", function () {
  let registry, token, distributor;
  let admin, treasury, creator, servingNode, payer;
  const ONE = ethers.parseUnits("1", 18);

  beforeEach(async function () {
    [admin, treasury, creator, servingNode, payer] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("ProvenanceRegistry");
    registry = await Registry.deploy();
    await registry.waitForDeployment();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Distributor = await ethers.getContractFactory("RoyaltyDistributor");
    distributor = await Distributor.deploy(
      await token.getAddress(),
      await registry.getAddress(),
      treasury.address
    );
    await distributor.waitForDeployment();

    // Mint payer some tokens and approve distributor
    await token.mint(payer.address, ONE * 1000n);
    await token.connect(payer).approve(await distributor.getAddress(), ONE * 1000n);
  });

  describe("distributeRoyalty", function () {
    it("splits payment 8% creator / 2% network / 90% serving node", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("split-test"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

      const gross = ONE * 100n; // 100 FTNS
      const expectedCreator = (gross * 800n) / 10000n; // 8 FTNS
      const expectedNetwork = (gross * 200n) / 10000n; // 2 FTNS
      const expectedNode = gross - expectedCreator - expectedNetwork; // 90 FTNS

      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross)
      )
        .to.emit(distributor, "RoyaltyPaid")
        .withArgs(
          contentHash,
          payer.address,
          creator.address,
          servingNode.address,
          expectedCreator,
          expectedNetwork,
          expectedNode
        );

      expect(await token.balanceOf(creator.address)).to.equal(expectedCreator);
      expect(await token.balanceOf(treasury.address)).to.equal(expectedNetwork);
      expect(await token.balanceOf(servingNode.address)).to.equal(expectedNode);
    });

    it("reverts if content not registered", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("missing"));
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, ONE * 10n)
      ).to.be.revertedWith("Not registered");
    });

    it("reverts if gross is zero", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("zero"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, 0)
      ).to.be.revertedWith("Zero amount");
    });

    it("reverts if serving node is zero address", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("zeronode"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, ethers.ZeroAddress, ONE * 10n)
      ).to.be.revertedWith("Zero serving node");
    });

    it("handles 0% royalty (creator gets nothing, all to node + treasury)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("free"));
      await registry.connect(creator).registerContent(contentHash, 0, "ipfs://F");

      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

      expect(await token.balanceOf(creator.address)).to.equal(0);
      expect(await token.balanceOf(treasury.address)).to.equal((gross * 200n) / 10000n);
      expect(await token.balanceOf(servingNode.address)).to.equal(gross - (gross * 200n) / 10000n);
    });

    it("handles max royalty (creator + network = 100%, node gets 0)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("max"));
      await registry.connect(creator).registerContent(contentHash, 9800, "ipfs://M");

      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

      expect(await token.balanceOf(creator.address)).to.equal((gross * 9800n) / 10000n);
      expect(await token.balanceOf(treasury.address)).to.equal((gross * 200n) / 10000n);
      expect(await token.balanceOf(servingNode.address)).to.equal(0);
    });

    it("reverts if creator + network fee exceed 100%", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("over"));
      await registry.connect(creator).registerContent(contentHash, 9900, "ipfs://O");

      const gross = ONE * 100n;
      // creator share = 99, network = 2, sum = 101 > 100 → revert
      await expect(
        distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross)
      ).to.be.revertedWith("Rate plus fee exceeds 100%");
    });
  });

  describe("preview", function () {
    it("returns the same split distributeRoyalty would apply", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("preview"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://P");

      const gross = ONE * 50n;
      const [c, n, s] = await distributor.preview(contentHash, gross);
      expect(c).to.equal((gross * 800n) / 10000n);
      expect(n).to.equal((gross * 200n) / 10000n);
      expect(s).to.equal(gross - c - n);
    });
  });
});
