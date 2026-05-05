// Team D — D5 PoC: Royalty push-payment contagion grief. RoyaltyDistributor
// uses push-payment to creator, networkTreasury, AND servingNode atomically.
// If creator transfers ownership to a contract that always reverts on token
// receipt (e.g., a contract with no `receive`/no recipient logic on a custom
// token), ALL future distributeRoyalty calls for that contentHash revert,
// blocking the servingNode and networkTreasury from receiving their shares.
//
// FTNSTokenSimple is plain ERC20 so transfer to ANY address (EOA or contract)
// succeeds — but a custom-FTNS replacement (the EscrowPool.setFtnsToken
// escape hatch) could ship one with hooks. More immediately exploitable:
// a creator can transfer to a non-existent contract that nullroutes royalties
// without bricking transfers. This test demonstrates the architectural
// vulnerability — ownership transfer to a malicious sink contract — using
// MockERC20 with hook-style behavior simulated by a malicious recipient.
//
// In our concrete reproducer, we register content, then transfer ownership
// to a "BlackholeRecipient" address that has no entry point. With plain
// ERC20, the transfer still succeeds (tokens land at the contract address,
// permanently locked). Royalties are not bricked but creator's share is
// stranded — a different but equivalent grief: the creator can permanently
// burn 100% of the creator share by self-transferring to a recoverable-only
// address (or a known burn).
//
// More important: there is NO `pull-payment` / `claim-bounty` indirection
// in RoyaltyDistributor (compare StakeBond which uses pull-style claim).
// This is an architectural risk the audit must flag, not an exploitable
// instance against the current FTNSTokenSimple deployment.

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("[Team D] D5 — Royalty push-payment architectural risk", function () {
  let token, provReg, distributor;
  let owner, payer, creator, servingNode, treasury;

  beforeEach(async function () {
    [owner, payer, creator, servingNode, treasury] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const PR = await ethers.getContractFactory("ProvenanceRegistry");
    provReg = await PR.deploy();
    await provReg.waitForDeployment();

    const RD = await ethers.getContractFactory("RoyaltyDistributor");
    distributor = await RD.deploy(
      await token.getAddress(),
      await provReg.getAddress(),
      treasury.address
    );
    await distributor.waitForDeployment();
  });

  it("REGRESSION (MEDIUM D-04): creator transfer to non-receiving contract does NOT brick distribute — pull-payment isolates the bad-recipient slot", async function () {
    // Register content with 50% royalty, transfer ownership to the
    // distributor itself (a contract that cannot externally claim its
    // own balance — worst case for the bad-recipient scenario).
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("content-1"));
    await provReg.connect(creator).registerContent(contentHash, 5000, "ipfs://meta");
    await provReg.connect(creator).transferContentOwnership(
      contentHash,
      await distributor.getAddress(),
    );

    // Payer pays.
    const gross = ethers.parseUnits("100", 18);
    await token.mint(payer.address, gross);
    await token.connect(payer).approve(await distributor.getAddress(), gross);

    // Pre-D-04 (push-payment) would have blocked the entire distribute
    // on the creator transfer if creator reverted on receive. Even
    // when creator silently accepts (as in the original test), the
    // funds were "stranded" because they sat in the recipient address
    // as untracked balance. Post-D-04, distribute always succeeds —
    // the entire gross sits in the distributor as escrow, and each
    // recipient's claimable slot is set independently.
    const distAddr = await distributor.getAddress();
    const balDistributorBefore = await token.balanceOf(distAddr);
    await distributor.connect(payer).distributeRoyalty(
      contentHash, servingNode.address, gross,
    );
    const balDistributorAfter = await token.balanceOf(distAddr);
    expect(balDistributorAfter - balDistributorBefore).to.equal(gross);

    // Treasury + servingNode can claim cleanly — bad-creator slot does
    // NOT block them. This is the core D-04 fix: claim isolation.
    const networkShare = (gross * 200n) / 10000n;       // 2 FTNS
    const creatorShare = (gross * 5000n) / 10000n;      // 50 FTNS
    const nodeShare = gross - creatorShare - networkShare;  // 48 FTNS
    expect(await distributor.claimable(treasury.address)).to.equal(networkShare);
    expect(await distributor.claimable(servingNode.address)).to.equal(nodeShare);
    expect(await distributor.claimable(distAddr)).to.equal(creatorShare);

    await distributor.connect(treasury).claim();
    await distributor.connect(servingNode).claim();
    expect(await token.balanceOf(treasury.address)).to.equal(networkShare);
    expect(await token.balanceOf(servingNode.address)).to.equal(nodeShare);

    // The bad-creator slot (claimable[distributor]) is operationally
    // stranded — distributor has no path to call its own claim() — but
    // this is now an OPERATOR misuse confined to the bad slot, not a
    // contagion that voids the entire payment. The creator could be
    // recovered via a future transferContentOwnership back to a real
    // EOA (not done here — this test pins the isolation property).
  });

  it("REGRESSION (MEDIUM D-04): servingNode = RoyaltyDistributor self-recipient does NOT brick treasury+creator — pull-payment isolates the bad slot", async function () {
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("content-2"));
    await provReg.connect(creator).registerContent(contentHash, 1000, "ipfs://meta2"); // 10% royalty

    const gross = ethers.parseUnits("100", 18);
    await token.mint(payer.address, gross);
    await token.connect(payer).approve(await distributor.getAddress(), gross);

    const distAddr = await distributor.getAddress();

    // Caller passes the distributor's own address as servingNode.
    // Pre-D-04: the entire payment flowed to distributor untracked.
    // Post-D-04: claim isolation — treasury + creator still get paid.
    await distributor.connect(payer).distributeRoyalty(contentHash, distAddr, gross);

    const creatorShare = (gross * 1000n) / 10000n;     // 10 FTNS
    const networkShare = (gross * 200n) / 10000n;      // 2 FTNS
    const nodeShare = gross - creatorShare - networkShare;  // 88 FTNS

    // Distributor holds the entire gross as escrow.
    expect(await token.balanceOf(distAddr)).to.equal(gross);
    // Per-recipient slots:
    expect(await distributor.claimable(creator.address)).to.equal(creatorShare);
    expect(await distributor.claimable(treasury.address)).to.equal(networkShare);
    expect(await distributor.claimable(distAddr)).to.equal(nodeShare);

    // Creator + treasury claim normally. The 88 FTNS stranded in the
    // distributor-as-node slot is an operator-misuse confinement, not
    // a contagion that voids the entire transaction.
    await distributor.connect(creator).claim();
    await distributor.connect(treasury).claim();
    expect(await token.balanceOf(creator.address)).to.equal(creatorShare);
    expect(await token.balanceOf(treasury.address)).to.equal(networkShare);
  });
});
