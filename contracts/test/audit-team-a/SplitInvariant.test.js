const { expect } = require("chai");
const { ethers } = require("hardhat");

// Team A — Economic Audit
//
// Finding A-01 PoC: RoyaltyDistributor split deviates from canonical
// PRSM-TOK-1 §8.1 invariant.
//
// PRSM-TOK-1 §8.1 (`docs/2026-04-21-prsm-tok-1-ftns-tokenomics.md`,
// lines 300-311) specifies:
//   burn:          20.0%
//   treasury:       1.6%   (= 2% network fee × 80% of remainder)
//   creator:        6.4%   (= 8% royalty × 80% of remainder)
//   serving node: 72.0%
//   ─────
//   total:        100.0%
//
// "The 20% burn is taken off the top; the remaining 80% is split per the
//  royalty/treasury/operator ratios established in Phase 1.1."
//
// Team A audit-team-prompt §"Stated invariants" #1: "Payment split is
// exactly 20/6.4/72/1.6 bps for burn/creator/node/treasury. Sum is exactly
// 10000 bps. Any deviation under any caller-controllable input is a
// finding."
//
// The deployed contract (RoyaltyDistributor.sol) does NOT burn anything.
// `distributeRoyalty(gross)` computes:
//   creatorAmt = gross * rateBps / 10000
//   networkAmt = gross * 200 / 10000          (= 2% of gross, NOT 1.6%)
//   nodeAmt    = gross - creator - network
//
// Even when the registry stores rateBps=640 (the value §8.1 anticipates as
// the "effective" creator share post-burn), the resulting split is:
//   creator: 6.4%   (matches §8.1 by accident)
//   treasury: 2.0%  (NOT 1.6% — 0.4% over spec → 25% over-treasury)
//   serving node: 91.6%  (NOT 72% — 19.6% over spec)
//   burn: 0.0%      (NOT 20% — invariant violated)
//
// This test PINS the divergence so it surfaces in audit. The remediation
// is either: (a) update tokenomics §8.1 to match contract behavior and
// remove the burn-on-use claim from the canonical spec, or (b) update the
// contract to burn 20% off the top before splitting the remaining 80%.

describe("AUDIT-TEAM-A — A01 RoyaltyDistributor split breaks PRSM-TOK-1 §8.1", function () {
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
      await token.getAddress(), await registry.getAddress(), treasury.address
    );
    await distributor.waitForDeployment();

    await token.mint(payer.address, ONE * 100000n);
    await token.connect(payer).approve(await distributor.getAddress(), ONE * 100000n);
  });

  it("burn is permanently 0 — TOK-1 §8.1 requires 20%", async function () {
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("burn-test"));
    // Per §8.4 the user-set rate is the effective post-burn rate. Use 800
    // (8%) which §8.4 says yields 6.4% of gross — but only AFTER a 20%
    // burn that this contract does not perform.
    await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

    const gross = ONE * 1000n; // 1000 FTNS
    const totalSupplyBefore = await token.totalSupply();

    await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

    const totalSupplyAfter = await token.totalSupply();

    // §8.1 mandates 20% of gross is burned. ZERO supply was destroyed.
    expect(totalSupplyAfter).to.equal(totalSupplyBefore);

    // Headline numeric proof: with §8.1 invariant, treasury should hold
    // 1.6% = 16 FTNS. Contract instead pays 2.0% = 20 FTNS — a 25%
    // over-payment to the network treasury at every transaction.
    const treasuryBal = await token.balanceOf(treasury.address);
    expect(treasuryBal).to.equal((gross * 200n) / 10000n); // 20 FTNS, contract behavior
    expect(treasuryBal).to.not.equal((gross * 160n) / 10000n); // 16 FTNS, §8.1 spec
  });

  it("serving node is over-paid by ~19.6 percentage points vs §8.1", async function () {
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("node-overpay"));
    await registry.connect(creator).registerContent(contentHash, 800, "ipfs://X");

    const gross = ONE * 10000n; // 10,000 FTNS
    await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);

    const nodeBal = await token.balanceOf(servingNode.address);

    // §8.1 says serving node = 72% = 7,200 FTNS.
    // Contract pays node = gross - creator(8%) - network(2%) = 90% = 9,000 FTNS.
    expect(nodeBal).to.equal(ONE * 9000n);
    expect(nodeBal).to.not.equal(ONE * 7200n);

    // Over-payment magnitude: 1,800 FTNS per 10,000 = 18% over §8.1.
    const overpayment = nodeBal - ONE * 7200n;
    expect(overpayment).to.equal(ONE * 1800n);
  });

  it("creator share equals 6.4% only by coincidence — burn invariant still violated", async function () {
    // If the team's defense is "set rateBps to the effective post-burn
    // rate (640 = 6.4%) and §8.1 §8.4 invariants are satisfied", the
    // creator share matches but treasury+node are still wrong AND no
    // burn happens. This pins the residual mismatch.
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("creator-effective"));
    await registry.connect(creator).registerContent(contentHash, 640, "ipfs://X");

    const gross = ONE * 10000n;
    const supplyBefore = await token.totalSupply();
    await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);
    const supplyAfter = await token.totalSupply();

    // Creator now matches §8.1's 6.4% = 640 FTNS.
    expect(await token.balanceOf(creator.address)).to.equal(ONE * 640n);
    // Treasury still mismatches (2% vs 1.6%).
    expect(await token.balanceOf(treasury.address)).to.equal(ONE * 200n); // 2%
    // Serving node still mismatches (91.6% vs 72%).
    expect(await token.balanceOf(servingNode.address)).to.equal(ONE * 9160n);
    // Burn still 0.
    expect(supplyAfter).to.equal(supplyBefore);
  });
});
