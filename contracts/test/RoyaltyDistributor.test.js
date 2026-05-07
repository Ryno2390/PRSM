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
      treasury.address,
      admin.address
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

      // L2 audit MEDIUM D-04 fix: pull-payment. distributeRoyalty
      // credits per-recipient `claimable` balances; recipients pull via
      // claim(). Direct token balances are unchanged until claim().
      expect(await token.balanceOf(creator.address)).to.equal(0n);
      expect(await token.balanceOf(treasury.address)).to.equal(0n);
      expect(await token.balanceOf(servingNode.address)).to.equal(0n);
      expect(await distributor.claimable(creator.address)).to.equal(expectedCreator);
      expect(await distributor.claimable(treasury.address)).to.equal(expectedNetwork);
      expect(await distributor.claimable(servingNode.address)).to.equal(expectedNode);

      // Each recipient claims independently.
      await expect(distributor.connect(creator).claim())
        .to.emit(distributor, "RoyaltyClaimed")
        .withArgs(creator.address, expectedCreator);
      expect(await token.balanceOf(creator.address)).to.equal(expectedCreator);
      expect(await distributor.claimable(creator.address)).to.equal(0n);

      await distributor.connect(treasury).claim();
      await distributor.connect(servingNode).claim();
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

      expect(await distributor.claimable(creator.address)).to.equal(0);
      expect(await distributor.claimable(treasury.address)).to.equal((gross * 200n) / 10000n);
      expect(await distributor.claimable(servingNode.address)).to.equal(gross - (gross * 200n) / 10000n);
    });

    it("handles max royalty (creator + network = 100%, node gets 0)", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("max"));
      await registry.connect(creator).registerContent(contentHash, 9800, "ipfs://M");

      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

      expect(await distributor.claimable(creator.address)).to.equal((gross * 9800n) / 10000n);
      expect(await distributor.claimable(treasury.address)).to.equal((gross * 200n) / 10000n);
      expect(await distributor.claimable(servingNode.address)).to.equal(0);
    });

    it("REGRESSION (MEDIUM D-04): claims are independent — one recipient's claim() does not affect another's accrual", async function () {
      // The load-bearing property of pull-payment: distribute credits
      // each recipient's claimable balance independently. A claim() by
      // any one recipient zeroes only their own slot. If recipient X
      // never claims (or their address is a contract that reverts on
      // receive), recipients Y and Z still get paid in full when they
      // claim. Pre-fix push-payment would have failed the entire
      // distribute on a single reverting recipient.
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("isolated-claim"));
      await registry.connect(creator).registerContent(contentHash, 800, "ipfs://I");

      const gross = ONE * 100n;
      const expectedCreator = (gross * 800n) / 10000n;
      const expectedNetwork = (gross * 200n) / 10000n;
      const expectedNode = gross - expectedCreator - expectedNetwork;

      await distributor.connect(payer).distributeRoyalty(
        contentHash, servingNode.address, gross,
      );

      // Treasury claims first.
      await distributor.connect(treasury).claim();
      expect(await token.balanceOf(treasury.address)).to.equal(expectedNetwork);
      expect(await distributor.claimable(treasury.address)).to.equal(0n);
      // Creator + servingNode untouched.
      expect(await distributor.claimable(creator.address)).to.equal(expectedCreator);
      expect(await distributor.claimable(servingNode.address)).to.equal(expectedNode);

      // Treasury double-claim reverts (idempotent zero-balance guard).
      await expect(distributor.connect(treasury).claim()).to.be.revertedWith("Nothing to claim");

      // Creator + node can still claim normally.
      await distributor.connect(creator).claim();
      await distributor.connect(servingNode).claim();
      expect(await token.balanceOf(creator.address)).to.equal(expectedCreator);
      expect(await token.balanceOf(servingNode.address)).to.equal(expectedNode);
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

  // -------------------------------------------------------------------------
  // L4 self-audit A-08 — recoverStranded
  // -------------------------------------------------------------------------
  //
  // Donations sent via direct ftns.transfer(distributor, X) bypass
  // distributeRoyalty and are not credited to any claimable[] slot.
  // Pre-fix they sat forever; post-fix the owner can sweep the strict
  // excess `balance(this) - totalClaimable` without disturbing real
  // claimable balances.

  describe("A-08 recoverStranded", function () {
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("a-08-content"));

    async function registerCreator(rateBps) {
      await registry.connect(creator).registerContent(
        contentHash,
        rateBps,
        "ipfs://a-08",
      );
    }

    it("totalClaimable starts at zero", async function () {
      expect(await distributor.totalClaimable()).to.equal(0);
    });

    it("totalClaimable tracks distributeRoyalty credits in lockstep", async function () {
      await registerCreator(5000); // 50% creator
      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(
        contentHash,
        servingNode.address,
        gross,
      );
      // creator: 50, network: 2, node: 48 — total 100 = gross
      expect(await distributor.totalClaimable()).to.equal(gross);
      expect(await token.balanceOf(await distributor.getAddress())).to.equal(gross);
    });

    it("totalClaimable decrements on claim() in lockstep with balance", async function () {
      await registerCreator(5000);
      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(
        contentHash, servingNode.address, gross,
      );
      const creatorAmt = await distributor.claimable(creator.address);
      await distributor.connect(creator).claim();
      expect(await distributor.claimable(creator.address)).to.equal(0);
      expect(await distributor.totalClaimable()).to.equal(gross - creatorAmt);
      expect(await token.balanceOf(await distributor.getAddress())).to.equal(gross - creatorAmt);
    });

    it("recoverStranded recovers a direct donation when no claimable balance exists", async function () {
      const donation = ONE * 7n;
      await token.mint(payer.address, donation);
      await token.connect(payer).transfer(await distributor.getAddress(), donation);
      // Pre-recover: totalClaimable=0, balance=donation
      expect(await distributor.totalClaimable()).to.equal(0);
      expect(await token.balanceOf(await distributor.getAddress())).to.equal(donation);

      const beforeBal = await token.balanceOf(admin.address);
      await expect(distributor.connect(admin).recoverStranded(admin.address))
        .to.emit(distributor, "StrandedRecovered")
        .withArgs(admin.address, donation);
      expect(await token.balanceOf(admin.address)).to.equal(beforeBal + donation);
      expect(await token.balanceOf(await distributor.getAddress())).to.equal(0);
    });

    it("recoverStranded recovers ONLY the excess when claimable balances exist", async function () {
      await registerCreator(5000);
      const gross = ONE * 100n;
      await distributor.connect(payer).distributeRoyalty(
        contentHash, servingNode.address, gross,
      );
      // Now donate on top.
      const donation = ONE * 13n;
      await token.mint(payer.address, donation);
      await token.connect(payer).transfer(await distributor.getAddress(), donation);

      const totalClaimableBefore = await distributor.totalClaimable();
      const balBefore = await token.balanceOf(await distributor.getAddress());
      expect(balBefore - totalClaimableBefore).to.equal(donation);

      const recipient = ethers.Wallet.createRandom().address;
      await expect(distributor.connect(admin).recoverStranded(recipient))
        .to.emit(distributor, "StrandedRecovered")
        .withArgs(recipient, donation);

      // claimable balances UNTOUCHED, totalClaimable UNTOUCHED
      expect(await distributor.totalClaimable()).to.equal(totalClaimableBefore);
      expect(await token.balanceOf(await distributor.getAddress()))
        .to.equal(totalClaimableBefore);
      expect(await token.balanceOf(recipient)).to.equal(donation);

      // All claims still work post-recovery
      await distributor.connect(creator).claim();
      await distributor.connect(servingNode).claim();
      await distributor.connect(treasury).claim();
      expect(await token.balanceOf(await distributor.getAddress())).to.equal(0);
      expect(await distributor.totalClaimable()).to.equal(0);
    });

    it("recoverStranded reverts NothingToRecover when balance == totalClaimable", async function () {
      await registerCreator(5000);
      await distributor.connect(payer).distributeRoyalty(
        contentHash, servingNode.address, ONE * 100n,
      );
      // No donation; balance exactly equals totalClaimable.
      await expect(distributor.connect(admin).recoverStranded(admin.address))
        .to.be.revertedWithCustomError(distributor, "NothingToRecover");
    });

    it("recoverStranded reverts NothingToRecover on a fresh empty contract", async function () {
      await expect(distributor.connect(admin).recoverStranded(admin.address))
        .to.be.revertedWithCustomError(distributor, "NothingToRecover");
    });

    it("recoverStranded reverts ZeroAddress on to == 0", async function () {
      await token.mint(payer.address, ONE);
      await token.connect(payer).transfer(await distributor.getAddress(), ONE);
      await expect(distributor.connect(admin).recoverStranded(ethers.ZeroAddress))
        .to.be.revertedWithCustomError(distributor, "ZeroAddress");
    });

    it("recoverStranded is onlyOwner", async function () {
      await token.mint(payer.address, ONE);
      await token.connect(payer).transfer(await distributor.getAddress(), ONE);
      await expect(distributor.connect(payer).recoverStranded(payer.address))
        .to.be.revertedWithCustomError(distributor, "OwnableUnauthorizedAccount")
        .withArgs(payer.address);
    });

    it("ownership transfers via Ownable2Step pendingOwner -> acceptOwnership", async function () {
      // Initial owner is `admin` per beforeEach.
      expect(await distributor.owner()).to.equal(admin.address);
      // Transfer to creator (pending until accepted).
      await distributor.connect(admin).transferOwnership(creator.address);
      expect(await distributor.owner()).to.equal(admin.address);
      expect(await distributor.pendingOwner()).to.equal(creator.address);
      // Old owner can still call until acceptance.
      await token.mint(payer.address, ONE);
      await token.connect(payer).transfer(await distributor.getAddress(), ONE);
      await distributor.connect(admin).recoverStranded(admin.address);
      // Accept; creator is now owner.
      await distributor.connect(creator).acceptOwnership();
      expect(await distributor.owner()).to.equal(creator.address);
      // Old owner now reverts.
      await token.mint(payer.address, ONE);
      await token.connect(payer).transfer(await distributor.getAddress(), ONE);
      await expect(distributor.connect(admin).recoverStranded(admin.address))
        .to.be.revertedWithCustomError(distributor, "OwnableUnauthorizedAccount");
      // New owner can.
      await distributor.connect(creator).recoverStranded(creator.address);
    });
  });
});
