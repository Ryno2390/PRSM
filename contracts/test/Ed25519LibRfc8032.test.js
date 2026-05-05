/*
 * Ed25519Lib — RFC 8032 §7.1 conformance test vectors.
 *
 * Source: https://datatracker.ietf.org/doc/html/rfc8032#section-7.1
 *
 * These are the canonical reference vectors for Ed25519. Any compliant
 * verifier MUST accept all four signatures as valid for their respective
 * (publicKey, message) pairs. This file runs all four against the
 * Ed25519LibHarness — i.e., directly against the underlying crypto math,
 * not the 32-byte-messageHash production wrapper.
 *
 * Why a separate harness test:
 * - Production Ed25519Verifier restricts m to a fixed 32 bytes
 *   (the keccak256 messageHash); RFC 8032 vectors use variable-length
 *   messages from 0 to 1023 bytes.
 * - Running the canonical vectors directly proves spec-conformance of the
 *   library at the level a cryptography auditor will inspect.
 *
 * Status of this test:
 * - L3 pre-engagement deliverable per audits/decisions/L3-ed25519-decision.md §5.
 * - Output also archived to audits/findings/L3-crypto/test-vectors-ed25519.md
 *   after the test suite passes.
 *
 * If any vector here fails, the lib has a spec-conformance regression and
 * MUST be remediated before any specialist audit is engaged. Engaging a
 * vendor against a spec-broken lib wastes both their time and our budget.
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");

// ─── RFC 8032 §7.1 Test 1 ─────────────────────────────────────────────
// ALGORITHM: Ed25519
// SECRET KEY:
//   9d61b19deffd5a60ba844af492ec2cc4 4449c5697b326919703bac031cae7f60
// PUBLIC KEY:
//   d75a980182b10ab7d54bfed3c964073a 0ee172f3daa62325af021a68f707511a
// MESSAGE (length 0 bytes):
// SIGNATURE:
//   e5564300c360ac729086e2cc806e828a 84877f1eb8e5d974d873e06522490155
//   5fb8821590a33bacc61e39701cf9b46b d25bf5f0595bbe24655141438e7a100b
const TEST_1 = {
  name: "Test 1 (empty message)",
  publicKey: "0xd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
  message: "0x",
  signature: "0xe5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
};

// ─── RFC 8032 §7.1 Test 2 ─────────────────────────────────────────────
// ALGORITHM: Ed25519
// SECRET KEY:
//   4ccd089b28ff96da9db6c346ec114e0f 5b8a319f35aba624da8cf6ed4fb8a6fb
// PUBLIC KEY:
//   3d4017c3e843895a92b70aa74d1b7ebc 9c982ccf2ec4968cc0cd55f12af4660c
// MESSAGE (length 1 byte): 72
// SIGNATURE:
//   92a009a9f0d4cab8720e820b5f642540 a2b27b5416503f8fb3762223ebdb69da
//   085ac1e43e15996e458f3613d0f11d8c 387b2eaeb4302aeeb00d291612bb0c00
const TEST_2 = {
  name: "Test 2 (1-byte message: 0x72)",
  publicKey: "0x3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
  message: "0x72",
  signature: "0x92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
};

// ─── RFC 8032 §7.1 Test 3 ─────────────────────────────────────────────
// ALGORITHM: Ed25519
// SECRET KEY:
//   c5aa8df43f9f837bedb7442f31dcb7b1 66d38535076f094b85ce3a2e0b4458f7
// PUBLIC KEY:
//   fc51cd8e6218a1a38da47ed00230f058 0816ed13ba3303ac5deb911548908025
// MESSAGE (length 2 bytes): af 82
// SIGNATURE:
//   6291d657deec24024827e69c3abe01a3 0ce548a284743a445e3680d7db5ac3ac
//   18ff9b538d16f290ae67f760984dc659 4a7c15e9716ed28dc027beceea1ec40a
const TEST_3 = {
  name: "Test 3 (2-byte message: 0xaf 0x82)",
  publicKey: "0xfc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
  message: "0xaf82",
  signature: "0x6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
};

// ─── RFC 8032 §7.1 Test SHA(abc) ──────────────────────────────────────
// ALGORITHM: Ed25519
// SECRET KEY:
//   833fe62409237b9d62ec77587520911e 9a759cec1d19755b7da901b96dca3d42
// PUBLIC KEY:
//   ec172b93ad5e563bf4932c70e1245034 c35467ef2efd4d64ebf819683467e2bf
// MESSAGE (length 64 bytes — SHA-512 of "abc"):
//   ddaf35a193617aba cc417349ae204131 12e6fa4e89a97ea2 0a9eeee64b55d39a
//   2192992a274fc1a8 36ba3c23a3feebbd 454d4423643ce80e 2a9ac94fa54ca49f
// SIGNATURE:
//   dc2a4459e7369633 a52b1bf277839a00 201009a3efbf3ecb 69bea2186c26b589
//   09351fc9ac90b3ec fdfbc7c66431e030 3dca179c138ac17a d9bef1177331a704
const TEST_SHA_ABC = {
  name: "Test SHA(abc) (64-byte message)",
  publicKey: "0xec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
  message: "0xddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
  signature: "0xdc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704",
};

const RFC_8032_VECTORS = [TEST_1, TEST_2, TEST_3, TEST_SHA_ABC];

function splitSignature(sig) {
  const bytes = ethers.getBytes(sig);
  return {
    r: ethers.hexlify(bytes.slice(0, 32)),
    s: ethers.hexlify(bytes.slice(32, 64)),
  };
}

describe("Ed25519Lib — RFC 8032 §7.1 conformance", function () {
  let harness;

  before(async function () {
    const Harness = await ethers.getContractFactory("Ed25519LibHarness");
    harness = await Harness.deploy();
    await harness.waitForDeployment();
  });

  describe("Canonical RFC vectors must verify", function () {
    for (const v of RFC_8032_VECTORS) {
      it(v.name, async function () {
        // Some of these vectors take significant gas; allow up to 60s timeout.
        this.timeout(60000);
        const { r, s } = splitSignature(v.signature);
        const ok = await harness.verify(v.publicKey, r, s, v.message);
        expect(ok, `RFC 8032 vector "${v.name}" failed verification`).to.equal(true);
      });
    }
  });

  describe("Bit-flip rejection on RFC vectors", function () {
    // Each canonical vector, with one bit flipped in the signature, must be
    // rejected. This catches mutation bugs in the verifier (e.g., always-true
    // returns).
    for (const v of RFC_8032_VECTORS) {
      it(`rejects bit-flipped signature for ${v.name}`, async function () {
        this.timeout(60000);
        const sigBytes = ethers.getBytes(v.signature);
        sigBytes[0] ^= 0x01;
        const tampered = ethers.hexlify(sigBytes);
        const { r, s } = splitSignature(tampered);
        const ok = await harness.verify(v.publicKey, r, s, v.message);
        expect(ok, `tampered ${v.name} accepted!`).to.equal(false);
      });
    }
  });

  describe("Wrong-key rejection on RFC vectors", function () {
    it("rejects Test 1 signature against Test 2 public key", async function () {
      this.timeout(60000);
      const { r, s } = splitSignature(TEST_1.signature);
      const ok = await harness.verify(TEST_2.publicKey, r, s, TEST_1.message);
      expect(ok).to.equal(false);
    });

    it("rejects Test 2 signature against Test 3 public key", async function () {
      this.timeout(60000);
      const { r, s } = splitSignature(TEST_2.signature);
      const ok = await harness.verify(TEST_3.publicKey, r, s, TEST_2.message);
      expect(ok).to.equal(false);
    });
  });

  describe("Wrong-message rejection on RFC vectors", function () {
    it("rejects Test 2 signature against Test 3 message", async function () {
      this.timeout(60000);
      const { r, s } = splitSignature(TEST_2.signature);
      const ok = await harness.verify(TEST_2.publicKey, r, s, TEST_3.message);
      expect(ok).to.equal(false);
    });
  });
});
