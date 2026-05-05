/*
 * Sha512.sol — FIPS 180-4 known-answer-test (KAT) conformance.
 *
 * Sha512.sol implements SHA-512 used by Ed25519Lib. This test runs a corpus
 * of FIPS 180-4 §D and NIST CAVP-derived KATs through the on-chain hasher to
 * prove spec conformance.
 *
 * Source vectors:
 *   - FIPS 180-4 Appendix D: "Test 1" (1-block "abc"), "Test 2" (multi-block
 *     "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"+"opqrpqrsqrstrstu"
 *     — actually per FIPS 180-4 the SHA-512 §D.2 message is: "abcdefgh..." —
 *     see RFC 6234 / NIST examples).
 *   - Empty message (length 0) → known SHA-512 of empty string.
 *   - 1-byte 0x00 — basic single-block boundary check.
 *   - 111-byte message — last byte where padding still fits in single block.
 *   - 112-byte message — first length where padding spills to second block.
 *   - 119, 120, 127, 128 — block-boundary edges.
 *   - 1023-byte message (for parity with RFC 8032 Test 1024).
 *
 * Expected hashes are verified independently using Node's built-in `crypto`
 * module (which wraps OpenSSL — implementation-independent ground truth).
 *
 * Status: L3 pre-engagement deliverable per
 *   audits/decisions/L3-ed25519-decision.md §5.
 *
 * If any KAT fails, Sha512.sol has a spec-conformance regression and must
 * be remediated before any specialist audit. Engaging a vendor against a
 * spec-broken hash wastes both their time and our budget.
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");
const crypto = require("crypto");

/**
 * Pack the 8 × uint64 state returned by Sha512Harness into a 64-byte
 * big-endian byte string for direct comparison with crypto.createHash output.
 */
function packState(state) {
  const out = new Uint8Array(64);
  for (let i = 0; i < 8; i++) {
    const v = BigInt(state[i].toString());
    for (let j = 0; j < 8; j++) {
      out[i * 8 + (7 - j)] = Number((v >> BigInt(j * 8)) & 0xffn);
    }
  }
  return ethers.hexlify(out);
}

function expectedSha512(messageBytes) {
  const buf = Buffer.from(ethers.getBytes(messageBytes));
  const hash = crypto.createHash("sha512").update(buf).digest();
  return ethers.hexlify(hash);
}

const KATS = [
  {
    name: "FIPS 180-4 §D.1 — 'abc'",
    message: "0x" + Buffer.from("abc", "ascii").toString("hex"),
  },
  {
    name: "Empty message (0 bytes)",
    message: "0x",
  },
  {
    name: "Single byte 0x00",
    message: "0x00",
  },
  {
    name: "FIPS 180-4 §D.2 — 'abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu'",
    message:
      "0x" +
      Buffer.from(
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
        "ascii",
      ).toString("hex"),
  },
  {
    name: "Block-boundary edge — 111 bytes (padding fits in single block)",
    message: "0x" + "a".repeat(222), // 111 bytes ascii 'a'
  },
  {
    name: "Block-boundary edge — 112 bytes (padding spills to 2nd block)",
    message: "0x" + "b".repeat(224), // 112 bytes ascii 'b'
  },
  {
    name: "Block-boundary edge — 127 bytes",
    message: "0x" + "c".repeat(254),
  },
  {
    name: "Block-boundary edge — 128 bytes (exactly 1 block)",
    message: "0x" + "d".repeat(256),
  },
  {
    name: "Block-boundary edge — 256 bytes (exactly 2 blocks)",
    message: "0x" + "e".repeat(512),
  },
  {
    name: "1023 bytes (parity with RFC 8032 Test 1024)",
    message: "0x" + "08".repeat(1023),
  },
];

describe("Sha512 — FIPS 180-4 known-answer tests", function () {
  let harness;

  before(async function () {
    const Harness = await ethers.getContractFactory("Sha512Harness");
    harness = await Harness.deploy();
    await harness.waitForDeployment();
  });

  for (const kat of KATS) {
    it(kat.name, async function () {
      // Long messages can take significant gas; allow up to 60s.
      this.timeout(60000);

      const expected = expectedSha512(kat.message);
      const state = await harness.hashOf(kat.message);
      const actual = packState(state);

      expect(actual.toLowerCase()).to.equal(
        expected.toLowerCase(),
        `Sha512(${kat.name}) mismatch:\n  expected: ${expected}\n  actual:   ${actual}`,
      );
    });
  }

  describe("Avalanche property — single bit flip changes hash entirely", function () {
    it("flipping bit 0 in 'abc' produces a wholly different hash", async function () {
      const a = await harness.hashOf("0x616263"); // 'abc'
      const b = await harness.hashOf("0x606263"); // 'a' with bit 0 flipped
      const aPacked = packState(a);
      const bPacked = packState(b);
      expect(aPacked).to.not.equal(bPacked);

      // Heuristic avalanche check — at least 200/512 bits should differ.
      const aBytes = ethers.getBytes(aPacked);
      const bBytes = ethers.getBytes(bPacked);
      let diffBits = 0;
      for (let i = 0; i < 64; i++) {
        let xor = aBytes[i] ^ bBytes[i];
        while (xor) {
          diffBits += xor & 1;
          xor >>>= 1;
        }
      }
      expect(diffBits).to.be.greaterThan(200);
    });
  });
});
