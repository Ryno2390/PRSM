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

  it("creator transfer to non-receiving contract permanently strands creator-share", async function () {
    // Register content with 50% royalty.
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("content-1"));
    await provReg.connect(creator).registerContent(contentHash, 5000, "ipfs://meta");

    // Transfer ownership to the RoyaltyDistributor itself — a known address
    // that holds no logic to forward FTNS onward. (Realistically, a creator
    // could transfer to any contract that locks tokens.)
    await provReg.connect(creator).transferContentOwnership(
      contentHash,
      await distributor.getAddress()
    );

    // Payer pays.
    const gross = ethers.parseUnits("100", 18);
    await token.mint(payer.address, gross);
    await token.connect(payer).approve(await distributor.getAddress(), gross);

    const balDistributorBefore = await token.balanceOf(await distributor.getAddress());
    await distributor.connect(payer).distributeRoyalty(contentHash, servingNode.address, gross);
    const balDistributorAfter = await token.balanceOf(await distributor.getAddress());

    // 50% of 100 = 50 stranded inside RoyaltyDistributor with no recovery
    // function (RoyaltyDistributor has no withdraw/sweep). Tokens are lost
    // forever.
    const creatorShare = ethers.parseUnits("50", 18);
    expect(balDistributorAfter - balDistributorBefore).to.equal(creatorShare);

    // The contract has no admin / withdraw / sweep — these tokens are dead.
    const fns = ["withdraw", "sweep", "rescue", "recover", "drain"];
    for (const fn of fns) {
      expect(distributor[fn], `unexpected ${fn} present`).to.be.undefined;
    }
  });

  it("servingNode = RoyaltyDistributor itself causes self-trap of node share", async function () {
    const contentHash = ethers.keccak256(ethers.toUtf8Bytes("content-2"));
    await provReg.connect(creator).registerContent(contentHash, 1000, "ipfs://meta2"); // 10% royalty

    const gross = ethers.parseUnits("100", 18);
    await token.mint(payer.address, gross);
    await token.connect(payer).approve(await distributor.getAddress(), gross);

    // Caller passes the distributor's own address as servingNode — caller
    // is griefing themselves but the RoyaltyDistributor accepts it (no
    // self-recipient check).
    const balBefore = await token.balanceOf(await distributor.getAddress());
    await distributor.connect(payer).distributeRoyalty(contentHash, await distributor.getAddress(), gross);
    const balAfter = await token.balanceOf(await distributor.getAddress());

    // 88% of gross (100 - 10 creator - 2 network = 88) is stranded.
    const stranded = ethers.parseUnits("88", 18);
    expect(balAfter - balBefore).to.equal(stranded);
  });
});
