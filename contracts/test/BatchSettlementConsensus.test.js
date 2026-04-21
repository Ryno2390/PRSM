const { expect } = require("chai");
const { ethers } = require("hardhat");

// Phase 7.1 Task 3 (revised during Task 7 E2E): CONSENSUS_MISMATCH uses
// a CROSS-BATCH proof, mirroring DOUBLE_SPEND's structure. In Phase 7.1's
// k-of-n redundant-execution model, each provider commits their own
// batch of their own receipts; the challenge against a minority
// provider uses a majority provider's (different) batch as proof of
// disagreement. Slashing naturally hits b.provider — the committer of
// the challenged (minority) batch.

describe("BatchSettlementRegistry — Phase 7.1 CONSENSUS_MISMATCH", function () {
  let registry, escrowPool, token, verifier, stakeBond;
  let owner, providerMajority, providerMinority, requester, foundation, other;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60;
  const DEFAULT_UNBOND_DELAY = 7 * 24 * 60 * 60;
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000;
  const CONSENSUS_MISMATCH = 5;

  function makeLeaf(overrides = {}) {
    return {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-phase7.1")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("provider")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output-default")),
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

  function encodeAux(conflictingBatchId, majorityProof, majorityLeaf) {
    return ethers.AbiCoder.defaultAbiCoder().encode(
      [
        "bytes32",
        "bytes32[]",
        "tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)",
      ],
      [
        conflictingBatchId,
        majorityProof,
        [
          majorityLeaf.jobIdHash, majorityLeaf.shardIndex,
          majorityLeaf.providerIdHash, majorityLeaf.providerPubkeyHash,
          majorityLeaf.outputHash, majorityLeaf.executedAtUnix,
          majorityLeaf.valueFtns, majorityLeaf.signatureHash,
        ],
      ],
    );
  }

  async function deployStack() {
    [owner, providerMajority, providerMinority, requester, foundation, other] =
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

    // Both providers bond premium stake so both are slashable if caught.
    for (const p of [providerMajority, providerMinority]) {
      await token.mint(p.address, PREMIUM_STAKE * 2n);
      await token.connect(p).approve(
        await stakeBond.getAddress(), PREMIUM_STAKE * 2n,
      );
      await stakeBond.connect(p).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
    }
  }

  beforeEach(async function () {
    await deployStack();
  });

  /**
   * Commit a single-leaf batch for `signer`. Returns {batchId, leafHash}.
   * In Phase 7.1's k-of-n flow, each participating provider commits
   * their own batch of their own receipts. Proof for a 1-leaf tree is
   * empty — the leaf hash IS the root.
   */
  async function commitSingleLeafBatch(signer, leaf) {
    const root = hashLeaf(leaf);
    const tx = await registry.connect(signer).commitBatch(
      requester.address, root, 1, leaf.valueFtns,
      PREMIUM_SLASH_BPS, "ipfs://" + signer.address.slice(0, 8),
    );
    const r = await tx.wait();
    const batchId = r.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted",
    ).args[0];
    return { batchId, leafHash: hashLeaf(leaf) };
  }

  // ── Happy path ──────────────────────────────────────────────────

  it("slashes the minority provider on legitimate CONSENSUS_MISMATCH", async function () {
    const majorityLeaf = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("A")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("C")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });

    // Each provider commits their own batch of their own receipt.
    const majorityBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minorityBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    // 1-leaf proof is empty (leaf hash IS the root).
    const aux = encodeAux(majorityBatch.batchId, [], majorityLeaf);

    const preMajorityStake = await stakeBond.stakeOf(providerMajority.address);
    const preMinorityStake = await stakeBond.stakeOf(providerMinority.address);
    expect(preMajorityStake.amount).to.equal(PREMIUM_STAKE);
    expect(preMinorityStake.amount).to.equal(PREMIUM_STAKE);

    await expect(
      registry.connect(requester).challengeReceipt(
        minorityBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
        { gasLimit: 1_000_000 },
      ),
    ).to.emit(stakeBond, "Slashed");

    // Minority's stake drained 100%; majority's stake untouched.
    const postMajorityStake = await stakeBond.stakeOf(providerMajority.address);
    const postMinorityStake = await stakeBond.stakeOf(providerMinority.address);
    expect(postMajorityStake.amount).to.equal(PREMIUM_STAKE);
    expect(postMinorityStake.amount).to.equal(0);
  });

  // ── Authorization ───────────────────────────────────────────────

  it("rejects when caller is not the minority batch's requester", async function () {
    const majorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    const aux = encodeAux(majBatch.batchId, [], majorityLeaf);

    // `other` is not the batch's requester — MVP rejects.
    await expect(
      registry.connect(other).challengeReceipt(
        minBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "CallerNotRequester");
  });

  // ── Same-batch (self-referential) rejection ─────────────────────

  it("rejects when conflictingBatchId equals the batch being challenged", async function () {
    // A provider can't disagree with themselves inside one batch.
    const leaf = makeLeaf();
    const batch = await commitSingleLeafBatch(providerMinority, leaf);

    const aux = encodeAux(batch.batchId, [], leaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        batch.batchId, leaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  // ── Disagreement-semantics guards ───────────────────────────────

  it("rejects when leaves are for different shards", async function () {
    const majorityLeaf = makeLeaf({
      shardIndex: 0,
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      shardIndex: 1,   // different shard — cross-challenge is meaningless
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    const aux = encodeAux(majBatch.batchId, [], majorityLeaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        minBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  it("rejects when leaves are from different jobs", async function () {
    const majorityLeaf = makeLeaf({
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-A")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-B")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    const aux = encodeAux(majBatch.batchId, [], majorityLeaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        minBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  it("rejects when outputs are equal (no real disagreement)", async function () {
    const sameHash = ethers.keccak256(ethers.toUtf8Bytes("SAME"));
    const majorityLeaf = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("A")),
      outputHash: sameHash,
    });
    const minorityLeaf = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("B")),
      outputHash: sameHash,
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    const aux = encodeAux(majBatch.batchId, [], majorityLeaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        minBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  // ── Proof verification ──────────────────────────────────────────

  it("rejects when majorityProof does not verify against conflicting batch root", async function () {
    const majorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    // Fake proof (non-empty sibling when the real proof is empty).
    const bogusSibling = ethers.keccak256(ethers.toUtf8Bytes("BOGUS"));
    const aux = encodeAux(majBatch.batchId, [bogusSibling], majorityLeaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        minBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  it("reverts when conflicting batch does not exist", async function () {
    const leaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const minBatch = await commitSingleLeafBatch(providerMinority, leaf);

    const fakeBatchId = ethers.keccak256(ethers.toUtf8Bytes("DOES_NOT_EXIST"));
    const fakeLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const aux = encodeAux(fakeBatchId, [], fakeLeaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        minBatch.batchId, leaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ConflictingBatchNotCommitted");
  });

  // ── 70/30 split preserved ───────────────────────────────────────

  it("slashed FTNS split matches 70% challenger / 30% Foundation", async function () {
    const majorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    const aux = encodeAux(majBatch.batchId, [], majorityLeaf);
    await registry.connect(requester).challengeReceipt(
      minBatch.batchId, minorityLeaf, [],
      CONSENSUS_MISMATCH, aux,
      { gasLimit: 1_000_000 },
    );

    const expectedBounty = (PREMIUM_STAKE * 7000n) / 10000n;
    const expectedFoundation = PREMIUM_STAKE - expectedBounty;
    expect(await stakeBond.slashedBountyPayable(requester.address))
      .to.equal(expectedBounty);
    expect(await stakeBond.foundationReserveBalance())
      .to.equal(expectedFoundation);
  });

  // ── Best-effort slashing preserved ──────────────────────────────

  it("challenge still succeeds when stakeBond is unset", async function () {
    await registry.connect(owner).setStakeBond(ethers.ZeroAddress);

    const majorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const minorityLeaf = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const majBatch = await commitSingleLeafBatch(
      providerMajority, majorityLeaf,
    );
    const minBatch = await commitSingleLeafBatch(
      providerMinority, minorityLeaf,
    );

    const aux = encodeAux(majBatch.batchId, [], majorityLeaf);
    await expect(
      registry.connect(requester).challengeReceipt(
        minBatch.batchId, minorityLeaf, [],
        CONSENSUS_MISMATCH, aux,
      ),
    ).to.emit(registry, "ReceiptChallenged");

    const minStake = await stakeBond.stakeOf(providerMinority.address);
    expect(minStake.amount).to.equal(PREMIUM_STAKE); // untouched
  });
});
