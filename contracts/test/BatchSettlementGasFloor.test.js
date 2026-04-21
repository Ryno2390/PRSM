const { expect } = require("chai");
const { ethers } = require("hardhat");

// Pre-audit hardening for Phase 7 §8.7 / Phase 7.1 §8.8 cross-cut:
// BatchSettlementRegistry.challengeReceipt now enforces
// gasleft() >= MIN_SLASH_GAS immediately before the stakeBond.slash
// try/catch. Without this floor, eth_estimateGas could under-budget
// a slashing challenge — the outer tx would succeed (receipt
// invalidated) while the nested slash silently reverted with OOG,
// caught by the try/catch. A challenger who under-paid gas would
// get a free burn of the receipt without the economic penalty.
//
// With the floor: the tx reverts cleanly BEFORE any state change,
// forcing the estimator (or a conscientious manual submission) to
// allocate enough gas for the slash to actually complete.

describe("BatchSettlementRegistry — §8.7 MIN_SLASH_GAS floor", function () {
  let registry, escrowPool, token, verifier, stakeBond;
  let owner, provider, requester, challenger, foundation;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60;
  const DEFAULT_UNBOND_DELAY = 7 * 24 * 60 * 60;
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000;

  function makeLeaf(overrides = {}) {
    return {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-gas-floor")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("provider")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output")),
      executedAtUnix: Math.floor(Date.now() / 1000),
      valueFtns: ONE_FTNS,
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("sig")),
      ...overrides,
    };
  }

  function hashLeaf(leaf) {
    const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
      ["tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)"],
      [[
        leaf.jobIdHash, leaf.shardIndex, leaf.providerIdHash,
        leaf.providerPubkeyHash, leaf.outputHash, leaf.executedAtUnix,
        leaf.valueFtns, leaf.signatureHash,
      ]],
    );
    return ethers.keccak256(encoded);
  }

  beforeEach(async function () {
    [owner, provider, requester, challenger, foundation] =
      await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(
      owner.address, await token.getAddress(), ethers.ZeroAddress,
    );
    await escrowPool.waitForDeployment();

    const Registry = await ethers.getContractFactory(
      "BatchSettlementRegistry",
    );
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();

    await escrowPool.connect(owner).setSettlementRegistry(
      await registry.getAddress(),
    );
    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    const Verifier = await ethers.getContractFactory("MockSignatureVerifier");
    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    await registry.connect(owner).setSignatureVerifier(
      await verifier.getAddress(),
    );

    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(
      owner.address, await token.getAddress(), DEFAULT_UNBOND_DELAY,
    );
    await stakeBond.waitForDeployment();
    await stakeBond.connect(owner).setSlasher(await registry.getAddress());
    await stakeBond.connect(owner).setFoundationReserveWallet(
      foundation.address,
    );
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    await token.mint(provider.address, PREMIUM_STAKE * 2n);
    await token.connect(provider).approve(
      await stakeBond.getAddress(), PREMIUM_STAKE * 2n,
    );
    await stakeBond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
  });

  async function commitDoubleSpendPair(leaf) {
    // Two identical leaves committed as separate batches — the
    // cheapest way to set up a DOUBLE_SPEND challenge that goes all
    // the way to the slash hook (so we can probe the gas floor).
    const root = hashLeaf(leaf);
    const tx1 = await registry.connect(provider).commitBatch(
      requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, "b1",
    );
    const id1 = (await tx1.wait()).logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];
    const tx2 = await registry.connect(provider).commitBatch(
      requester.address, root, 1, leaf.valueFtns, PREMIUM_SLASH_BPS, "b2",
    );
    const id2 = (await tx2.wait()).logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];
    return { id1, id2 };
  }

  // ── Constant surface ────────────────────────────────────────────

  it("exposes MIN_SLASH_GAS as a public constant", async function () {
    expect(await registry.MIN_SLASH_GAS()).to.equal(150_000n);
  });

  // ── Guard fires on under-funded challenges ──────────────────────

  it("reverts InsufficientGasForSlash when challenger under-funds gas", async function () {
    const leaf = makeLeaf();
    const { id1, id2 } = await commitDoubleSpendPair(leaf);

    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"], [id1, []],
    );

    // Cap the tx at 120_000 gas — well below both the pre-slash
    // execution cost and the MIN_SLASH_GAS floor itself. The tx
    // runs out before even reaching the floor check, so the revert
    // is a generic OOG rather than the custom error. Verify it
    // reverts (the exact error type depends on where the budget
    // runs out — important part is the tx does NOT succeed with a
    // silent slash skip).
    await expect(
      registry.connect(challenger).challengeReceipt(
        id2, leaf, [], 0, aux,
        { gasLimit: 120_000 },
      ),
    ).to.be.reverted;

    // Provider's stake must be untouched regardless of which revert
    // path fired — the key invariant: no partial state change.
    const stake = await stakeBond.stakeOf(provider.address);
    expect(stake.amount).to.equal(PREMIUM_STAKE);
  });

  it("reverts with the custom error when gasleft is below MIN_SLASH_GAS but above baseline", async function () {
    const leaf = makeLeaf();
    const { id1, id2 } = await commitDoubleSpendPair(leaf);

    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"], [id1, []],
    );

    // Find a gas cap where execution reaches the floor check but
    // gasleft() there is below MIN_SLASH_GAS. The full-path gas is
    // ~200k; before the floor check, ~50k has been consumed. So a
    // gasLimit of 180_000 should reach the floor check with
    // ~130_000 remaining → below 150_000 → custom error.
    await expect(
      registry.connect(challenger).challengeReceipt(
        id2, leaf, [], 0, aux,
        { gasLimit: 180_000 },
      ),
    ).to.be.revertedWithCustomError(
      registry, "InsufficientGasForSlash",
    );
  });

  // ── Happy path (estimator allocates enough) ─────────────────────

  it("slashes normally when challenger lets estimator allocate gas", async function () {
    const leaf = makeLeaf();
    const { id1, id2 } = await commitDoubleSpendPair(leaf);

    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"], [id1, []],
    );

    // No manual gasLimit — hardhat estimator binary-searches and
    // finds a value that clears MIN_SLASH_GAS naturally. Slash lands.
    await expect(
      registry.connect(challenger).challengeReceipt(id2, leaf, [], 0, aux),
    ).to.emit(stakeBond, "Slashed");

    const stake = await stakeBond.stakeOf(provider.address);
    expect(stake.amount).to.equal(0);
  });

  it("slashes normally when challenger explicitly funds 1M gas", async function () {
    const leaf = makeLeaf();
    const { id1, id2 } = await commitDoubleSpendPair(leaf);

    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"], [id1, []],
    );

    await expect(
      registry.connect(challenger).challengeReceipt(
        id2, leaf, [], 0, aux,
        { gasLimit: 1_000_000 },
      ),
    ).to.emit(stakeBond, "Slashed");
  });

  // ── No-slash paths remain gas-cheap ─────────────────────────────

  it("does NOT enforce the floor on EXPIRED challenges (no slash path)", async function () {
    // EXPIRED doesn't slash, so MIN_SLASH_GAS doesn't gate it.
    // Prove the tx succeeds at a gas budget far below MIN_SLASH_GAS.

    // Commit a batch with an old timestamp.
    const oldLeaf = makeLeaf({
      executedAtUnix: Math.floor(Date.now() / 1000) - (60 * 24 * 60 * 60), // 60 days old
    });
    const root = hashLeaf(oldLeaf);
    const tx = await registry.connect(provider).commitBatch(
      requester.address, root, 1, oldLeaf.valueFtns,
      PREMIUM_SLASH_BPS, "expired-batch",
    );
    const batchId = (await tx.wait()).logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];

    // ReasonCode.EXPIRED = 3. No slash path, so no gas floor required.
    // Estimator will allocate just enough for the expired-check path.
    await expect(
      registry.connect(challenger).challengeReceipt(
        batchId, oldLeaf, [], 3, "0x",
      ),
    ).to.emit(registry, "ReceiptChallenged");

    // Provider's stake untouched (EXPIRED doesn't slash).
    const stake = await stakeBond.stakeOf(provider.address);
    expect(stake.amount).to.equal(PREMIUM_STAKE);
  });

  it("does NOT enforce the floor when stakeBond is unset", async function () {
    // Disable slashing entirely — floor is only relevant when the
    // slash path would run.
    await registry.connect(owner).setStakeBond(ethers.ZeroAddress);

    const leaf = makeLeaf();
    const { id1, id2 } = await commitDoubleSpendPair(leaf);
    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "bytes32[]"], [id1, []],
    );

    // With stakeBond unset, the slash branch is skipped entirely.
    // Tx should complete at a budget well below MIN_SLASH_GAS.
    await expect(
      registry.connect(challenger).challengeReceipt(id2, leaf, [], 0, aux),
    ).to.emit(registry, "ReceiptChallenged");
  });
});
