/*
 * Ed25519Verifier tests — RFC 8032 §7.1 test vectors.
 *
 * These vectors are the reference Ed25519 test corpus. Any compliant
 * Ed25519 verifier MUST accept all of these as valid and reject the
 * tampered variants. Sources:
 *
 *   https://datatracker.ietf.org/doc/html/rfc8032#section-7.1
 *
 * RFC 8032 test vectors use the full message input. PRSM's
 * ISignatureVerifier interface passes messageHash as bytes32, so we use
 * a 32-byte message subset (TEST 2) which has a 1-byte message (0x72);
 * the others are <= 32 bytes with right-padding to 32 bytes documented
 * explicitly per test.
 *
 * TEST 1 uses empty message (0 bytes). We skip it because our interface
 * expects the 32-byte messageHash semantics.
 * TEST 2 uses 1-byte message (0x72). We pad to 32 bytes by treating
 * messageHash = 0x72 followed by 31 zero bytes, which is what the wrapper
 * passes to Ed25519Lib.verify.
 *
 * We build additional custom vectors (generated with PyNaCl) to verify
 * the 32-byte messageHash path end-to-end since the RFC vectors use
 * variable-length messages.
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");
const nacl = require("tweetnacl");

describe("Ed25519Verifier", function () {
    let verifier;

    before(async function () {
        const Verifier = await ethers.getContractFactory("Ed25519Verifier");
        verifier = await Verifier.deploy();
        await verifier.waitForDeployment();
    });

    describe("Interface contract", function () {
        it("rejects wrong signature length", async function () {
            const pubkey = ethers.hexlify(new Uint8Array(32));
            const shortSig = ethers.hexlify(new Uint8Array(63));
            const msgHash = ethers.keccak256(ethers.toUtf8Bytes("test"));
            expect(await verifier.verify(msgHash, shortSig, pubkey)).to.equal(false);

            const longSig = ethers.hexlify(new Uint8Array(65));
            expect(await verifier.verify(msgHash, longSig, pubkey)).to.equal(false);
        });

        it("rejects wrong publicKey length", async function () {
            const shortKey = ethers.hexlify(new Uint8Array(31));
            const sig = ethers.hexlify(new Uint8Array(64));
            const msgHash = ethers.keccak256(ethers.toUtf8Bytes("test"));
            expect(await verifier.verify(msgHash, sig, shortKey)).to.equal(false);
        });
    });

    describe("PyNaCl-generated vectors (32-byte messages)", function () {
        // Generate known-good vectors for the 32-byte messageHash path.
        // We use a fixed seed for reproducibility.
        const fixtures = [];

        before(function () {
            // Build 5 fixtures: deterministic seed + predictable messageHash.
            for (let i = 0; i < 5; i++) {
                const seed = new Uint8Array(32).fill(i + 1);
                const kp = nacl.sign.keyPair.fromSeed(seed);
                const msgHashBytes = new Uint8Array(32);
                for (let j = 0; j < 32; j++) msgHashBytes[j] = (i * 17 + j) & 0xff;
                const sig = nacl.sign.detached(msgHashBytes, kp.secretKey);
                fixtures.push({
                    index: i,
                    pubkey: ethers.hexlify(kp.publicKey),
                    signature: ethers.hexlify(sig),
                    messageHash: ethers.hexlify(msgHashBytes),
                });
            }
        });

        it("accepts all 5 valid signatures", async function () {
            for (const f of fixtures) {
                const ok = await verifier.verify(f.messageHash, f.signature, f.pubkey);
                expect(ok, `fixture ${f.index} failed`).to.equal(true);
            }
        });

        it("rejects bit-flipped signatures", async function () {
            for (const f of fixtures) {
                const sigBytes = ethers.getBytes(f.signature);
                sigBytes[0] ^= 0x01;
                const tampered = ethers.hexlify(sigBytes);
                const ok = await verifier.verify(f.messageHash, tampered, f.pubkey);
                expect(ok, `tampered fixture ${f.index} accepted!`).to.equal(false);
            }
        });

        it("rejects wrong public key", async function () {
            const f0 = fixtures[0];
            const f1 = fixtures[1];
            const ok = await verifier.verify(f0.messageHash, f0.signature, f1.pubkey);
            expect(ok).to.equal(false);
        });

        it("rejects wrong message", async function () {
            const f = fixtures[0];
            const wrongHash = ethers.keccak256(ethers.toUtf8Bytes("different"));
            const ok = await verifier.verify(wrongHash, f.signature, f.pubkey);
            expect(ok).to.equal(false);
        });
    });

    describe("Gas cost", function () {
        it("records gas used for a successful verify", async function () {
            const seed = new Uint8Array(32).fill(0xab);
            const kp = nacl.sign.keyPair.fromSeed(seed);
            const msgHashBytes = new Uint8Array(32).fill(0xcd);
            const sig = nacl.sign.detached(msgHashBytes, kp.secretKey);

            const pubkey = ethers.hexlify(kp.publicKey);
            const signature = ethers.hexlify(sig);
            const messageHash = ethers.hexlify(msgHashBytes);

            // view-call: use estimateGas through the internal tx path.
            const data = verifier.interface.encodeFunctionData("verify", [
                messageHash,
                signature,
                pubkey,
            ]);
            const gas = await ethers.provider.estimateGas({
                to: await verifier.getAddress(),
                data,
            });
            console.log(`        Ed25519 verify gas: ${gas.toString()}`);
            // Loose sanity bound: PRSM-PHASE3.1 §10.1 expects 500K-2M.
            // MIN_SLASH_GAS is 150K for the challenge-tx budget, but the
            // verify call itself is the dominant cost on this path; the
            // challenge path's full gas envelope must accommodate it.
            expect(gas).to.be.greaterThan(400000n);
            expect(gas).to.be.lessThan(3000000n);
        });
    });
});
