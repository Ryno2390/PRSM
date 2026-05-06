const { expect } = require("chai");
const { ethers } = require("hardhat");

// PRSM-PROV-1 Item 7 T7.1 — ProvenanceRegistryV2 Hardhat tests.
//
// Covers the new embeddingCommitment + fingerprintKind fields and the
// dispute-helper view functions. Backwards-compatible v1 paths
// (ownership transfer, royalty cap, duplicate-rejection) are mirrored
// to confirm the contract is a strict v1 superset.

describe("ProvenanceRegistryV2", function () {
  let registry;
  let owner, creator, otherCreator, attacker;

  // Reusable kind-tag hashes — kept consistent with the Python wrapper.
  const KIND_TEXT = ethers.keccak256(ethers.toUtf8Bytes("text-vector"));
  const KIND_IMAGE = ethers.keccak256(ethers.toUtf8Bytes("image-phash"));

  // Build a synthetic embedding commitment matching the v2 wrapper's
  // canonical formula: keccak256(model_id || uint32_be(dim) || vector_bytes).
  function makeCommitment(modelId, dim, vector) {
    const modelIdBytes = ethers.toUtf8Bytes(modelId);
    const dimBe = new Uint8Array(4);
    new DataView(dimBe.buffer).setUint32(0, dim, false); // big-endian
    const concat = new Uint8Array(modelIdBytes.length + 4 + vector.length);
    concat.set(modelIdBytes, 0);
    concat.set(dimBe, modelIdBytes.length);
    concat.set(vector, modelIdBytes.length + 4);
    return ethers.keccak256(concat);
  }

  beforeEach(async function () {
    [owner, creator, otherCreator, attacker] = await ethers.getSigners();
    const Registry = await ethers.getContractFactory("ProvenanceRegistryV2");
    registry = await Registry.deploy();
    await registry.waitForDeployment();
  });

  describe("registerContent — v2 fields", function () {
    it("stores and emits the embedding commitment + kind", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("paper-A"));
      const commitment = makeCommitment(
        "openai-text-embedding-ada-002",
        1536,
        new Uint8Array([1, 2, 3, 4, 5]),
      );

      await expect(
        registry.connect(creator).registerContent(
          contentHash,
          800,
          "ipfs://QmAbstract",
          commitment,
          KIND_TEXT,
        ),
      )
        .to.emit(registry, "ContentRegistered")
        .withArgs(
          contentHash,
          creator.address,
          800,
          commitment,
          KIND_TEXT,
          "ipfs://QmAbstract",
        );

      const [storedCommitment, storedKind] =
        await registry.getEmbeddingCommitment(contentHash);
      expect(storedCommitment).to.equal(commitment);
      expect(storedKind).to.equal(KIND_TEXT);
    });

    it("accepts zero commitment for byte-hash-only content", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("opaque-blob"));
      await registry.connect(creator).registerContent(
        contentHash,
        500,
        "ipfs://QmBlob",
        ethers.ZeroHash,
        ethers.ZeroHash,
      );

      const [c, k] = await registry.getEmbeddingCommitment(contentHash);
      expect(c).to.equal(ethers.ZeroHash);
      expect(k).to.equal(ethers.ZeroHash);
    });

    it("preserves the v1 metadataUri + creator + rate semantics", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("paper-B"));
      const commitment = makeCommitment(
        "all-MiniLM-L6-v2",
        384,
        new Uint8Array([7, 7, 7]),
      );

      await registry.connect(creator).registerContent(
        contentHash,
        1500,
        "ipfs://QmB",
        commitment,
        KIND_TEXT,
      );

      const [c, rate] = await registry.getCreatorAndRate(contentHash);
      expect(c).to.equal(creator.address);
      expect(rate).to.equal(1500);
      expect(await registry.isRegistered(contentHash)).to.equal(true);
    });

    it("rejects duplicate registration regardless of commitment", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("dup"));
      const c1 = makeCommitment("m1", 4, new Uint8Array([0]));
      const c2 = makeCommitment("m2", 4, new Uint8Array([1]));

      await registry.connect(creator).registerContent(
        contentHash,
        800,
        "ipfs://A",
        c1,
        KIND_TEXT,
      );
      await expect(
        registry.connect(otherCreator).registerContent(
          contentHash,
          500,
          "ipfs://B",
          c2,
          KIND_TEXT,
        ),
      ).to.be.revertedWith("Already registered");
    });

    it("rejects royalty rate above MAX_ROYALTY_RATE_BPS", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("over"));
      await expect(
        registry.connect(creator).registerContent(
          contentHash,
          9801,
          "ipfs://X",
          ethers.ZeroHash,
          ethers.ZeroHash,
        ),
      ).to.be.revertedWith("Rate exceeds max");
    });

    it("supports per-kind tags for binary content", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("photo"));
      const commitment = ethers.keccak256(ethers.toUtf8Bytes("phash-bytes"));

      await registry.connect(creator).registerContent(
        contentHash,
        500,
        "ipfs://photo",
        commitment,
        KIND_IMAGE,
      );

      const [, kind] = await registry.getEmbeddingCommitment(contentHash);
      expect(kind).to.equal(KIND_IMAGE);
    });
  });

  describe("verifyEmbeddingCommitment", function () {
    it("returns true when claimed commitment matches on-chain", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("paper-V"));
      const vec = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const commitment = makeCommitment("test-model", 8, vec);

      await registry.connect(creator).registerContent(
        contentHash,
        800,
        "ipfs://V",
        commitment,
        KIND_TEXT,
      );

      expect(
        await registry.verifyEmbeddingCommitment(contentHash, commitment),
      ).to.equal(true);
    });

    it("returns false when claimed commitment differs", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("paper-W"));
      const real = makeCommitment("test-model", 8, new Uint8Array([1, 2, 3]));
      const fake = makeCommitment("test-model", 8, new Uint8Array([9, 9, 9]));

      await registry.connect(creator).registerContent(
        contentHash,
        800,
        "ipfs://W",
        real,
        KIND_TEXT,
      );

      expect(
        await registry.verifyEmbeddingCommitment(contentHash, fake),
      ).to.equal(false);
    });

    it("returns false for content with zero commitment, regardless of claim", async function () {
      // Zero commitment must NEVER verify — even if the claim is also
      // zero. Otherwise a claimant could "win" disputes against
      // byte-hash-only content with a zero-vector forgery.
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("legacy-paper"));
      await registry.connect(creator).registerContent(
        contentHash,
        500,
        "ipfs://legacy",
        ethers.ZeroHash,
        ethers.ZeroHash,
      );

      expect(
        await registry.verifyEmbeddingCommitment(contentHash, ethers.ZeroHash),
      ).to.equal(false);
      const someClaim = ethers.keccak256(ethers.toUtf8Bytes("anything"));
      expect(
        await registry.verifyEmbeddingCommitment(contentHash, someClaim),
      ).to.equal(false);
    });

    it("returns false for unregistered contentHash", async function () {
      const ghost = ethers.keccak256(ethers.toUtf8Bytes("never-registered"));
      const claim = ethers.keccak256(ethers.toUtf8Bytes("any"));
      expect(
        await registry.verifyEmbeddingCommitment(ghost, claim),
      ).to.equal(false);
    });
  });

  describe("transferContentOwnership — preserves v1 semantics", function () {
    let contentHash;
    let commitment;

    beforeEach(async function () {
      contentHash = ethers.keccak256(ethers.toUtf8Bytes("transferable"));
      commitment = makeCommitment("m", 4, new Uint8Array([9, 9]));
      await registry.connect(creator).registerContent(
        contentHash,
        800,
        "ipfs://T",
        commitment,
        KIND_TEXT,
      );
    });

    it("creator can transfer ownership", async function () {
      await expect(
        registry
          .connect(creator)
          .transferContentOwnership(contentHash, otherCreator.address),
      )
        .to.emit(registry, "OwnershipTransferred")
        .withArgs(contentHash, creator.address, otherCreator.address);

      const [newCreator] = await registry.getCreatorAndRate(contentHash);
      expect(newCreator).to.equal(otherCreator.address);
    });

    it("transfer leaves embeddingCommitment untouched", async function () {
      await registry
        .connect(creator)
        .transferContentOwnership(contentHash, otherCreator.address);
      const [c, k] = await registry.getEmbeddingCommitment(contentHash);
      expect(c).to.equal(commitment);
      expect(k).to.equal(KIND_TEXT);
    });

    it("non-creator cannot transfer", async function () {
      await expect(
        registry
          .connect(attacker)
          .transferContentOwnership(contentHash, attacker.address),
      ).to.be.revertedWith("Not creator");
    });

    it("rejects transfer to zero address", async function () {
      await expect(
        registry
          .connect(creator)
          .transferContentOwnership(contentHash, ethers.ZeroAddress),
      ).to.be.revertedWith("Zero address");
    });
  });

  describe("storage layout — append-only over v1 shape", function () {
    it("contents() returns the v1-prefixed fields plus the new ones", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("layout"));
      const commitment = makeCommitment("m", 4, new Uint8Array([1]));
      await registry.connect(creator).registerContent(
        contentHash,
        700,
        "ipfs://layout",
        commitment,
        KIND_TEXT,
      );

      // Public mapping returns fields in struct declaration order:
      // creator, royaltyRateBps, registeredAt, embeddingCommitment,
      // fingerprintKind, metadataUri.
      const c = await registry.contents(contentHash);
      expect(c.creator).to.equal(creator.address);
      expect(c.royaltyRateBps).to.equal(700);
      expect(c.registeredAt).to.be.gt(0);
      expect(c.embeddingCommitment).to.equal(commitment);
      expect(c.fingerprintKind).to.equal(KIND_TEXT);
      expect(c.metadataUri).to.equal("ipfs://layout");
    });
  });
});
