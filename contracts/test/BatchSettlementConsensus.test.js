const { expect } = require("chai");
const { ethers } = require("hardhat");

// Phase 7.1 Task 3 — BatchSettlementRegistry CONSENSUS_MISMATCH path.
//
// A successful k-of-n redundant-execution dispatch can produce a batch
// with two receipts for the SAME shard from different providers whose
// outputs disagree. The requester then challenges the minority leaf via
// ReasonCode.CONSENSUS_MISMATCH, passing the majority leaf + proof as
// auxData. The contract verifies:
//   1. caller is the batch requester (MVP authorization)
//   2. the hash bindings in auxData match the leaf fields
//   3. the majority and minority hashes are genuinely different
//   4. the majority leaf is in the same batch's Merkle tree
// On success, the receipt is invalidated AND the slash hook fires.

describe("BatchSettlementRegistry — Phase 7.1 CONSENSUS_MISMATCH", function () {
  let registry, escrowPool, token, verifier, stakeBond;
  let owner, provider, requester, challenger, foundation;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60;
  const DEFAULT_UNBOND_DELAY = 7 * 24 * 60 * 60;
  const PREMIUM_STAKE = 25_000n * ONE_FTNS;
  const PREMIUM_SLASH_BPS = 10000;
  const CONSENSUS_MISMATCH = 5; // ReasonCode enum value

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

  // OpenZeppelin MerkleProof.verify hashes pairs in sorted order.
  function pairHash(a, b) {
    const aBuf = ethers.getBytes(a);
    const bBuf = ethers.getBytes(b);
    const sorted = Buffer.compare(
      Buffer.from(aBuf), Buffer.from(bBuf)
    ) < 0 ? [a, b] : [b, a];
    return ethers.keccak256(
      ethers.concat([sorted[0], sorted[1]])
    );
  }

  function buildTwoLeafTree(leafMajority, leafMinority) {
    const hMaj = hashLeaf(leafMajority);
    const hMin = hashLeaf(leafMinority);
    const root = pairHash(hMaj, hMin);
    // In a 2-leaf tree each leaf's proof is the sibling's hash.
    return {
      root,
      proofForMajority: [hMin],
      proofForMinority: [hMaj],
      majorityHash: hMaj,
      minorityHash: hMin,
    };
  }

  function encodeAux(
    majorityAgreedHash,
    minorityClaimedHash,
    majorityProof,
    majorityLeaf,
  ) {
    return ethers.AbiCoder.defaultAbiCoder().encode(
      [
        "bytes32",
        "bytes32",
        "bytes32[]",
        "tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)",
      ],
      [
        majorityAgreedHash,
        minorityClaimedHash,
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

  async function deployPhase71Stack() {
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

    // Fund + bond the provider.
    await token.mint(provider.address, PREMIUM_STAKE * 2n);
    await token.connect(provider).approve(
      await stakeBond.getAddress(), PREMIUM_STAKE * 2n,
    );
    await stakeBond.connect(provider).bond(PREMIUM_STAKE, PREMIUM_SLASH_BPS);
  }

  beforeEach(async function () {
    await deployPhase71Stack();
  });

  async function commitTwoLeafBatch(leafMajority, leafMinority) {
    const tree = buildTwoLeafTree(leafMajority, leafMinority);
    const tx = await registry.connect(provider).commitBatch(
      requester.address,
      tree.root,
      2,
      leafMajority.valueFtns + leafMinority.valueFtns,
      PREMIUM_SLASH_BPS,
      "ipfs://consensus-batch",
    );
    const r = await tx.wait();
    const batchId = r.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted",
    ).args[0];
    return { batchId, tree };
  }

  // ── Happy path ──────────────────────────────────────────────────

  it("slashes minority provider on legitimate CONSENSUS_MISMATCH", async function () {
    const leafMaj = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("providerA")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("providerC")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const aux = encodeAux(
      leafMaj.outputHash, leafMin.outputHash,
      tree.proofForMajority, leafMaj,
    );

    const stakeBefore = await stakeBond.stakeOf(provider.address);
    expect(stakeBefore.amount).to.equal(PREMIUM_STAKE);

    await expect(
      registry.connect(requester).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.emit(stakeBond, "Slashed");

    const stakeAfter = await stakeBond.stakeOf(provider.address);
    expect(stakeAfter.amount).to.equal(0);
  });

  // ── Binding failures ────────────────────────────────────────────

  it("rejects when minorityClaimedHash in aux != leaf.outputHash", async function () {
    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    // Flip the minority hash in auxData to something that doesn't
    // match the leaf we're challenging.
    const bogusMinorityHash = ethers.keccak256(ethers.toUtf8Bytes("BOGUS"));
    const aux = encodeAux(
      leafMaj.outputHash, bogusMinorityHash,
      tree.proofForMajority, leafMaj,
    );

    await expect(
      registry.connect(requester).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  it("rejects when majorityAgreedHash in aux != majorityLeaf.outputHash", async function () {
    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const bogusMajorityHash = ethers.keccak256(ethers.toUtf8Bytes("NOPE"));
    const aux = encodeAux(
      bogusMajorityHash, leafMin.outputHash,
      tree.proofForMajority, leafMaj,
    );

    await expect(
      registry.connect(requester).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  // ── No actual disagreement ──────────────────────────────────────

  it("rejects when majority and minority output hashes are equal", async function () {
    // Both receipts claim the same output hash — there's no dispute
    // here; either both are wrong (DOUBLE_SPEND territory) or neither
    // is wrong. CONSENSUS_MISMATCH is the wrong tool.
    const sameHash = ethers.keccak256(ethers.toUtf8Bytes("SAME"));
    const leafMaj = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("A")),
      outputHash: sameHash,
    });
    const leafMin = makeLeaf({
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("B")),
      outputHash: sameHash,
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const aux = encodeAux(
      sameHash, sameHash, tree.proofForMajority, leafMaj,
    );
    await expect(
      registry.connect(requester).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  // ── Merkle proof failures ───────────────────────────────────────

  it("rejects when majority proof does not verify against batch root", async function () {
    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    // Use a fake proof (sibling hash that isn't in this batch).
    const bogusSibling = ethers.keccak256(ethers.toUtf8Bytes("BOGUS_SIBLING"));
    const aux = encodeAux(
      leafMaj.outputHash, leafMin.outputHash,
      [bogusSibling], leafMaj,
    );

    await expect(
      registry.connect(requester).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");
  });

  // ── Authorization (MVP: requester-only) ─────────────────────────

  it("rejects when caller is not the batch requester", async function () {
    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const aux = encodeAux(
      leafMaj.outputHash, leafMin.outputHash,
      tree.proofForMajority, leafMaj,
    );
    // `challenger` is a different signer than `requester` — Phase 7.1
    // MVP forbids this. Phase 7.1x introduces consensus_group_id so
    // any holder of both receipts can challenge.
    await expect(
      registry.connect(challenger).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.be.revertedWithCustomError(registry, "CallerNotRequester");
  });

  // ── Best-effort slashing (Phase 7 invariant preserved) ──────────

  it("challenge succeeds even if slash is skipped (stakeBond unset)", async function () {
    // Disable the slash hook by setting stakeBond to zero address.
    await registry.connect(owner).setStakeBond(ethers.ZeroAddress);

    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const aux = encodeAux(
      leafMaj.outputHash, leafMin.outputHash,
      tree.proofForMajority, leafMaj,
    );

    // Receipt still invalidated; stake untouched.
    await expect(
      registry.connect(requester).challengeReceipt(
        batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      ),
    ).to.emit(registry, "ReceiptChallenged");

    const stakeAfter = await stakeBond.stakeOf(provider.address);
    expect(stakeAfter.amount).to.equal(PREMIUM_STAKE); // unchanged
  });

  it("challenge succeeds even when provider has no stake", async function () {
    // Unbond + withdraw before challenge → slash silently skipped via
    // try/catch but receipt invalidation stands.
    await stakeBond.connect(provider).requestUnbond();
    // Fast-forward past the unbond delay.
    await ethers.provider.send(
      "evm_increaseTime", [DEFAULT_UNBOND_DELAY + 1],
    );
    await ethers.provider.send("evm_mine", []);
    await stakeBond.connect(provider).withdraw();
    const s = await stakeBond.stakeOf(provider.address);
    expect(s.amount).to.equal(0);

    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });

    // Need to commit the batch BEFORE withdraw — provider can't commit
    // with 0 amount in some flows. Actually commitBatch doesn't check
    // stake at all; so we commit now.
    // NOTE: re-ordering here — the test's intent is "provider with no
    // stake at challenge time". commitBatch doesn't gate on stake, so
    // this is fine.
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const aux = encodeAux(
      leafMaj.outputHash, leafMin.outputHash,
      tree.proofForMajority, leafMaj,
    );

    // Challenge succeeds (emits ReceiptChallenged); no Slashed event.
    const tx = await registry.connect(requester).challengeReceipt(
      batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      { gasLimit: 1_000_000 }, // ensure slash-path has headroom per Phase 7 §8.7
    );
    const rcpt = await tx.wait();
    const slashedEvents = rcpt.logs.filter((l) => {
      try {
        return stakeBond.interface.parseLog(l)?.name === "Slashed";
      } catch (_) { return false; }
    });
    expect(slashedEvents.length).to.equal(0);
  });

  // ── 70/30 split preserved for CONSENSUS_MISMATCH ────────────────

  it("slashed FTNS split matches the 70% challenger / 30% Foundation rule", async function () {
    const leafMaj = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_OK")),
    });
    const leafMin = makeLeaf({
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("HASH_CHEAT")),
    });
    const { batchId, tree } = await commitTwoLeafBatch(leafMaj, leafMin);

    const aux = encodeAux(
      leafMaj.outputHash, leafMin.outputHash,
      tree.proofForMajority, leafMaj,
    );

    await registry.connect(requester).challengeReceipt(
      batchId, leafMin, tree.proofForMinority, CONSENSUS_MISMATCH, aux,
      { gasLimit: 1_000_000 },
    );

    const expectedBounty = (PREMIUM_STAKE * 7000n) / 10000n;
    const expectedFoundation = PREMIUM_STAKE - expectedBounty;
    expect(await stakeBond.slashedBountyPayable(requester.address))
      .to.equal(expectedBounty);
    expect(await stakeBond.foundationReserveBalance())
      .to.equal(expectedFoundation);
  });
});
