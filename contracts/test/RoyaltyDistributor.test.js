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

    // Phase 1.1: the registry now caps rates at MAX_ROYALTY_RATE_BPS = 9800
    // so the distributor's "Rate plus fee exceeds 100%" require is no longer
    // reachable from a real registered content hash. The require is kept as
    // defense-in-depth (e.g. for a hypothetical buggy/malicious registry).
    // To verify the invariant, registering at the max+1 rate fails at the
    // registry, which transitively prevents the distributor revert.
    it("registry cap prevents the distributor's overflow revert (invariant)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("invariant"));
      await expect(
        registry.connect(creator).registerContent(contentHash, 9801, "ipfs://X")
      ).to.be.revertedWith("Rate exceeds max");
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

  describe("metadataUri gas griefing regression (Phase 1.1 P2 #7)", function () {
    it("payment gas is independent of metadataUri size", async function () {
      const smallHash = ethers.keccak256(ethers.toUtf8Bytes("small-meta"));
      const bigHash = ethers.keccak256(ethers.toUtf8Bytes("big-meta"));
      const smallMeta = "ipfs://X";
      const bigMeta = "ipfs://" + "Q".repeat(8000); // ~8KB

      await registry.connect(creator).registerContent(smallHash, 800, smallMeta);
      await registry.connect(creator).registerContent(bigHash, 800, bigMeta);

      const gross = ONE * 1n;

      // Warmup pass: prime the token recipient slots so cold-vs-warm SSTORE
      // doesn't dominate the measurement. Each distribute touches creator,
      // treasury, and servingNode balances; the first touch is ~22100 gas
      // and subsequent touches are ~5000.
      await distributor.connect(payer).distributeRoyalty(smallHash, servingNode.address, gross);
      await distributor.connect(payer).distributeRoyalty(bigHash, servingNode.address, gross);

      // Measurement pass: now that all token slots are warm, the only
      // difference between small-meta and big-meta should be the metadata
      // storage access. With the slim getCreatorAndRate getter, that
      // difference is zero.
      const txSmall = await distributor
        .connect(payer)
        .distributeRoyalty(smallHash, servingNode.address, gross);
      const txBig = await distributor
        .connect(payer)
        .distributeRoyalty(bigHash, servingNode.address, gross);

      const rcSmall = await txSmall.wait();
      const rcBig = await txBig.wait();

      // If the distributor were reading metadataUri, the 8KB difference
      // would cost ~525k extra gas (250 SLOADs at ~2100 each). With the
      // slim getter, the diff must be near zero (allow tiny variance).
      const diff = Number(rcBig.gasUsed - rcSmall.gasUsed);
      expect(Math.abs(diff)).to.be.lessThan(1000);
    });
  });
});
