/*
 * Team C — REGRESSION GUARD for C-INT-01 (formerly CRITICAL):
 * Adversarial slashing via unbound `signingMessage` in the
 * INVALID_SIGNATURE challenge path.
 *
 * REMEDIATION SHIPPED 2026-05-05 (commit will be tagged after CI):
 *   - ReceiptLeaf extended with `signingMessageHash` field (option A).
 *   - `_handleInvalidSignature` auxData now (publicKey, signature) only.
 *   - Verifier consumes `leaf.signingMessageHash` directly; challenger
 *     can no longer pick the message.
 *
 * Original attack (now blocked):
 *   1. Reuse the real (pubkey, signature) bytes (their keccak256 hashes
 *      match the leaf-committed providerPubkeyHash and signatureHash).
 *   2. Pick ANY arbitrary `signingMessage` whose Ed25519 verification
 *      under (pubkey, signature) returns FALSE.
 *   3. Submit challenge → verifier returns false → !valid was true →
 *      provider slashed at full bond rate, attacker takes 70% bounty.
 *
 * After remediation: attacker can no longer supply `signingMessage` at
 * all (auxData layout no longer carries it). The verifier checks the
 * REAL signature against `leaf.signingMessageHash` — which is the actual
 * canonical message the provider signed. The verifier returns true,
 * the challenge fails, the provider is NOT slashed, and the challenger's
 * INVALID_SIGNATURE attempt reverts with `ChallengeNotProven`.
 *
 * This file is now a regression guard: the test asserts that the legit
 * INVALID_SIGNATURE path (against a real, committed signature) FAILS as
 * "challenge not proven" because the signature is in fact valid.
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");
const nacl = require("tweetnacl");

describe("Audit Team C — C-INT-01: INVALID_SIGNATURE adversarial slashing", function () {
  let registry, escrowPool, token, verifier, stakeBond;
  let owner, provider, requester, attacker;
  const ONE_FTNS = ethers.parseUnits("1", 18);
  const DEFAULT_WINDOW = 3 * 24 * 60 * 60; // 3 days
  const TIER_CRITICAL_BPS = 10000; // 100% slash rate
  const BOND_AMOUNT = ethers.parseUnits("50000", 18);

  function makeLeaf(overrides) {
    const leaf = {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("job-real-1")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("provider-id-1")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("provider-pubkey-default")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output-bytes")),
      executedAtUnix: Math.floor(Date.now() / 1000),
      valueFtns: ONE_FTNS,
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("provider-sig-default")),
      // L4 self-audit INFO-3 (C-04): non-degenerate default. Tests that
      // care about the signingMessageHash binding override this; tests
      // that don't were previously inheriting `ZeroHash`, which is a
      // valid bytes32 but also the value an uninitialised storage slot
      // would produce. Future test authors copying this factory shouldn't
      // land on a degenerate input by default.
      signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("default-signing-message")),
      ...overrides,
    };
    return leaf;
  }

  function hashLeaf(leaf) {
    const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
      [
        "tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32,bytes32)",
      ],
      [
        [
          leaf.jobIdHash,
          leaf.shardIndex,
          leaf.providerIdHash,
          leaf.providerPubkeyHash,
          leaf.outputHash,
          leaf.executedAtUnix,
          leaf.valueFtns,
          leaf.signatureHash,
          leaf.signingMessageHash,
        ],
      ]
    );
    return ethers.keccak256(encoded);
  }

  beforeEach(async function () {
    [owner, provider, requester, attacker] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("MockERC20");
    token = await Token.deploy();
    await token.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(owner.address, await token.getAddress(), await registry.getAddress());
    await escrowPool.waitForDeployment();

    // Wire the REAL Ed25519Verifier (not a mock) — the bug exists with the
    // real verifier just as it did with the mock; the mock just made it
    // visible-by-construction.
    const Verifier = await ethers.getContractFactory("Ed25519Verifier");
    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    await registry.connect(owner).setSignatureVerifier(await verifier.getAddress());

    // Wire StakeBond with the registry as immutable slasher (HIGH-7).
    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(owner.address, await token.getAddress(), 1 * 24 * 60 * 60, await registry.getAddress());
    await stakeBond.waitForDeployment();
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    // Provider bonds at the critical (100%) tier.
    await token.mint(provider.address, BOND_AMOUNT);
    await token.connect(provider).approve(await stakeBond.getAddress(), BOND_AMOUNT);
    await stakeBond.connect(provider).bond(BOND_AMOUNT, TIER_CRITICAL_BPS);

    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token.connect(requester).approve(await escrowPool.getAddress(), ONE_FTNS * 1000n);
    await escrowPool.connect(requester).deposit(ONE_FTNS * 1000n);
  });

  it("REGRESSION: post-fix, attacker cannot forge an INVALID_SIGNATURE challenge — the verifier consults leaf.signingMessageHash, not a challenger-supplied message", async function () {
    // ── Step 1: provider produces a REAL Ed25519-signed receipt ─────────
    const seed = new Uint8Array(32).fill(0x42);
    const kp = nacl.sign.keyPair.fromSeed(seed);
    const realPubkey = Buffer.from(kp.publicKey);
    // Provider's canonical 32-byte messageHash. Whatever the provider
    // actually signs is bound into the leaf.
    const realMessageBytes = new Uint8Array(32);
    for (let j = 0; j < 32; j++) realMessageBytes[j] = (0x10 + j) & 0xff;
    const realSignature = Buffer.from(nacl.sign.detached(realMessageBytes, kp.secretKey));

    // ── Sanity: the real signature verifies under the real message. ────
    {
      const ok = await verifier.verify(
        ethers.hexlify(realMessageBytes),
        ethers.hexlify(realSignature),
        ethers.hexlify(realPubkey)
      );
      expect(ok, "real signature should verify").to.equal(true);
    }

    // ── Step 2: provider commits a leaf binding pubkey + signature
    //          + signingMessageHash. The signingMessageHash field IS
    //          the real message (verifier wants bytes32 directly). ──────
    const leaf = makeLeaf({
      providerPubkeyHash: ethers.keccak256(realPubkey),
      signatureHash: ethers.keccak256(realSignature),
      signingMessageHash: ethers.hexlify(realMessageBytes),
    });
    const root = hashLeaf(leaf);

    const commitTx = await registry
      .connect(provider)
      .commitBatch(requester.address, root, 1, leaf.valueFtns, TIER_CRITICAL_BPS, ethers.ZeroHash, "ipfs://test");
    const commitRcpt = await commitTx.wait();
    const batchId = commitRcpt.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];

    const stakeBefore = (await stakeBond.stakes(provider.address)).amount;
    expect(stakeBefore).to.equal(BOND_AMOUNT);

    // ── Step 3: ATTACK ATTEMPT (now blocked) ───────────────────────────
    // Attacker submits the real (pubkey, signature) bytes. They CANNOT
    // supply a chosen signingMessage anymore — auxData is now only
    // (publicKey, signature). The contract uses leaf.signingMessageHash
    // for verification — which IS the real message. The verifier returns
    // true. _handleInvalidSignature returns !valid == false. Challenge
    // is not proven; the call reverts with ChallengeNotProven.
    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes", "bytes"],
      [realPubkey, realSignature]
    );

    await expect(
      registry
        .connect(attacker)
        .challengeReceipt(batchId, leaf, [], 1 /* INVALID_SIGNATURE */, aux, {
          gasLimit: 4_000_000n,
        })
    ).to.be.revertedWithCustomError(registry, "ChallengeNotProven");

    // Provider's stake is intact — no slashing fired.
    const stakeAfter = (await stakeBond.stakes(provider.address)).amount;
    expect(stakeAfter).to.equal(BOND_AMOUNT, "Provider's stake intact post-fix");

    // No bounty paid out.
    expect(await stakeBond.slashedBountyPayable(attacker.address)).to.equal(0n);

    // The receipt is NOT invalidated.
    const batch = await registry.getBatch(batchId);
    expect(batch.invalidatedValueFTNS).to.equal(0n);
  });

  it("post-fix POSITIVE path: a genuinely-invalid signature (signingMessageHash mismatch with what was signed) DOES still allow legitimate slashing", async function () {
    // Demonstrates the fix doesn't disable the legitimate INVALID_SIGNATURE
    // path: a malicious provider who commits a leaf whose
    // signingMessageHash was not actually signed CAN still be slashed.
    const seed = new Uint8Array(32).fill(0x42);
    const kp = nacl.sign.keyPair.fromSeed(seed);
    const pubkey = Buffer.from(kp.publicKey);

    // Provider signs message A but commits a leaf claiming message B.
    const signedMessageBytes = new Uint8Array(32).fill(0xaa);
    const signature = Buffer.from(nacl.sign.detached(signedMessageBytes, kp.secretKey));

    const claimedMessageBytes = new Uint8Array(32).fill(0xbb); // different
    const leaf = makeLeaf({
      providerPubkeyHash: ethers.keccak256(pubkey),
      signatureHash: ethers.keccak256(signature),
      signingMessageHash: ethers.hexlify(claimedMessageBytes), // mismatch
    });
    const root = hashLeaf(leaf);

    const commitTx = await registry
      .connect(provider)
      .commitBatch(requester.address, root, 1, leaf.valueFtns, TIER_CRITICAL_BPS, ethers.ZeroHash, "ipfs://test");
    const commitRcpt = await commitTx.wait();
    const batchId = commitRcpt.logs.find(
      (l) => l.fragment && l.fragment.name === "BatchCommitted"
    ).args[0];

    // Anyone can challenge — signature does NOT verify against
    // leaf.signingMessageHash → verifier returns false → !valid is true →
    // challenge succeeds → provider slashed.
    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes", "bytes"],
      [pubkey, signature]
    );

    await expect(
      registry
        .connect(attacker)
        .challengeReceipt(batchId, leaf, [], 1 /* INVALID_SIGNATURE */, aux, {
          gasLimit: 4_000_000n,
        })
    ).to.emit(registry, "ReceiptChallenged");

    const stakeAfter = (await stakeBond.stakes(provider.address)).amount;
    expect(stakeAfter).to.equal(0n, "Malicious provider IS slashed post-fix");
  });
});
