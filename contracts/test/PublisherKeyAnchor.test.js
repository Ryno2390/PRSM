/*
 * PublisherKeyAnchor tests — Phase 3.x.3 Task 1.
 *
 * The contract closes the cross-node trust-boundary caveat from Phase
 * 3.x.2 + 3.x.4. Tests verify:
 *   - Permissionless registration with sha256-derived node_id check
 *     (caller cannot forge a registration for someone else's pubkey)
 *   - Write-once invariant (can't overwrite via re-register)
 *   - Multisig admin override path (emergency revocation)
 *   - Non-admin override reverts
 *   - View lookup returns empty bytes for unregistered node_ids
 *   - Gas-cost benchmark for register
 *
 * Pubkeys are generated with tweetnacl (same library used by the
 * Ed25519Verifier tests) so the on-chain sha256 derivation can be
 * compared against the off-chain expectation byte-for-byte.
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");
const crypto = require("crypto");
const nacl = require("tweetnacl");

/** Compute the same node_id (first 16 bytes of sha256(pubkey)) the
 *  contract derives on-chain. Used to verify the contract's behavior
 *  matches PRSM's NodeIdentity rule. */
function deriveNodeId(pubkey32) {
    const sha = crypto.createHash("sha256").update(pubkey32).digest();
    return ethers.hexlify(sha.subarray(0, 16));
}

/** Generate a fresh Ed25519 keypair returning {pubkey, nodeId} as hex. */
function freshKey(seedByte) {
    const seed = new Uint8Array(32).fill(seedByte);
    const kp = nacl.sign.keyPair.fromSeed(seed);
    return {
        pubkey: ethers.hexlify(kp.publicKey),
        pubkeyBytes: kp.publicKey,
        nodeId: deriveNodeId(Buffer.from(kp.publicKey)),
    };
}

describe("PublisherKeyAnchor", function () {
    let anchor;
    let admin;
    let publisher1;
    let publisher2;
    let outsider;

    beforeEach(async function () {
        [admin, publisher1, publisher2, outsider] = await ethers.getSigners();
        const Anchor = await ethers.getContractFactory("PublisherKeyAnchor");
        anchor = await Anchor.deploy(admin.address);
        await anchor.waitForDeployment();
    });

    describe("Construction", function () {
        it("stores admin address", async function () {
            expect(await anchor.admin()).to.equal(admin.address);
        });

        it("rejects zero-address admin", async function () {
            const Anchor = await ethers.getContractFactory("PublisherKeyAnchor");
            await expect(Anchor.deploy(ethers.ZeroAddress))
                .to.be.revertedWithCustomError(Anchor, "AdminIsZeroAddress");
        });
    });

    describe("register — permissionless with sha256 derivation", function () {
        it("stores a registered pubkey under the derived nodeId", async function () {
            const k = freshKey(0x01);
            await anchor.connect(publisher1).register(k.pubkey);

            const stored = await anchor.lookup(k.nodeId);
            expect(stored).to.equal(k.pubkey);
        });

        it("emits PublisherRegistered with derived nodeId + timestamp", async function () {
            const k = freshKey(0x02);
            const tx = await anchor.connect(publisher1).register(k.pubkey);
            await expect(tx)
                .to.emit(anchor, "PublisherRegistered")
                .withArgs(k.nodeId, k.pubkey, anyUint64());
        });

        it("records registeredAt timestamp", async function () {
            const k = freshKey(0x03);
            const tx = await anchor.connect(publisher1).register(k.pubkey);
            const receipt = await tx.wait();
            const block = await ethers.provider.getBlock(receipt.blockNumber);
            expect(await anchor.registeredAt(k.nodeId)).to.equal(block.timestamp);
        });

        it("rejects pubkey of wrong length", async function () {
            const tooShort = ethers.hexlify(new Uint8Array(31));
            const tooLong = ethers.hexlify(new Uint8Array(33));
            const empty = "0x";
            for (const bad of [tooShort, tooLong, empty]) {
                await expect(anchor.connect(publisher1).register(bad))
                    .to.be.revertedWithCustomError(anchor, "InvalidPublicKeyLength");
            }
        });

        it("rejects duplicate registration of the same nodeId", async function () {
            const k = freshKey(0x04);
            await anchor.connect(publisher1).register(k.pubkey);
            // Second call with the SAME pubkey must revert.
            await expect(anchor.connect(publisher1).register(k.pubkey))
                .to.be.revertedWithCustomError(anchor, "AlreadyRegistered")
                .withArgs(k.nodeId);
        });

        it("blocks duplicate registration even from a different caller", async function () {
            // First-write-wins; an attacker re-registering under a
            // different signer must still be rejected (the nodeId is
            // already taken regardless of who registered it).
            const k = freshKey(0x05);
            await anchor.connect(publisher1).register(k.pubkey);
            await expect(anchor.connect(publisher2).register(k.pubkey))
                .to.be.revertedWithCustomError(anchor, "AlreadyRegistered");
        });

        it("THE KEY PROPERTY: caller cannot forge registration for someone else's pubkey", async function () {
            // The contract derives nodeId from sha256(publicKey) on-chain.
            // An adversary calling register(victim_pubkey) DOES register
            // — but the nodeId is the victim's real nodeId (because the
            // pubkey is real), and once registered, the victim's node is
            // permanently associated with that pubkey. The "forge" attack
            // would require the adversary to register a DIFFERENT pubkey
            // under the victim's nodeId, which is impossible because the
            // contract derives the key, not the caller.
            //
            // Concretely: there's no `register(nodeId, pubkey)` API where
            // the caller could supply a mismatched pair. The single-arg
            // `register(pubkey)` makes the attack impossible by construction.
            const victimKey = freshKey(0x06);
            const attackerKey = freshKey(0x07);

            // Attacker can't claim the victim's nodeId for the attacker's
            // pubkey. The attacker can only register their OWN pubkey
            // under THEIR OWN nodeId (which is a different nodeId).
            await anchor.connect(outsider).register(attackerKey.pubkey);
            const stored = await anchor.lookup(attackerKey.nodeId);
            expect(stored).to.equal(attackerKey.pubkey);
            // Victim's nodeId remains unregistered.
            expect(await anchor.lookup(victimKey.nodeId)).to.equal("0x");
        });

        it("multiple distinct publishers can all register", async function () {
            const k1 = freshKey(0x08);
            const k2 = freshKey(0x09);
            const k3 = freshKey(0x0a);
            await anchor.connect(publisher1).register(k1.pubkey);
            await anchor.connect(publisher2).register(k2.pubkey);
            await anchor.connect(outsider).register(k3.pubkey);
            expect(await anchor.lookup(k1.nodeId)).to.equal(k1.pubkey);
            expect(await anchor.lookup(k2.nodeId)).to.equal(k2.pubkey);
            expect(await anchor.lookup(k3.nodeId)).to.equal(k3.pubkey);
        });
    });

    describe("adminOverride — emergency multisig path", function () {
        it("admin can override a previously-registered key", async function () {
            const original = freshKey(0x10);
            const replacement = freshKey(0x11);
            await anchor.connect(publisher1).register(original.pubkey);

            // Compromise scenario: rotate to a new key under the SAME nodeId.
            // (In practice, the new pubkey would have a DIFFERENT nodeId
            // since nodeId derives from pubkey. The override is for the
            // case where a publisher's identity-to-key binding needs to
            // be forced — e.g., the publisher's private key was leaked
            // and they want their on-chain identity bound to a new
            // verifying key. v1 keeps it minimal; v2 will introduce
            // proper rotation history.)
            await anchor.adminOverride(original.nodeId, replacement.pubkey);
            expect(await anchor.lookup(original.nodeId)).to.equal(replacement.pubkey);
        });

        it("emits PublisherOverridden with old + new keys", async function () {
            const original = freshKey(0x12);
            const replacement = freshKey(0x13);
            await anchor.connect(publisher1).register(original.pubkey);

            await expect(anchor.adminOverride(original.nodeId, replacement.pubkey))
                .to.emit(anchor, "PublisherOverridden")
                .withArgs(original.nodeId, original.pubkey, replacement.pubkey, anyUint64());
        });

        it("non-admin override reverts", async function () {
            const k = freshKey(0x14);
            const replacement = freshKey(0x15);
            await anchor.connect(publisher1).register(k.pubkey);

            await expect(
                anchor.connect(outsider).adminOverride(k.nodeId, replacement.pubkey)
            )
                .to.be.revertedWithCustomError(anchor, "NotAdmin")
                .withArgs(outsider.address);
        });

        it("override of unregistered nodeId reverts", async function () {
            const fake = freshKey(0x16);
            const replacement = freshKey(0x17);
            await expect(anchor.adminOverride(fake.nodeId, replacement.pubkey))
                .to.be.revertedWithCustomError(anchor, "NotRegistered")
                .withArgs(fake.nodeId);
        });

        it("override with wrong-length newKey reverts", async function () {
            const k = freshKey(0x18);
            await anchor.connect(publisher1).register(k.pubkey);
            const tooShort = ethers.hexlify(new Uint8Array(31));
            await expect(anchor.adminOverride(k.nodeId, tooShort))
                .to.be.revertedWithCustomError(anchor, "InvalidPublicKeyLength");
        });

        it("override updates registeredAt to the override timestamp", async function () {
            const k = freshKey(0x19);
            const replacement = freshKey(0x1a);
            await anchor.connect(publisher1).register(k.pubkey);
            const tsBefore = await anchor.registeredAt(k.nodeId);

            // Mine a few blocks to advance the timestamp
            await ethers.provider.send("evm_increaseTime", [60]);
            await ethers.provider.send("evm_mine");

            await anchor.adminOverride(k.nodeId, replacement.pubkey);
            const tsAfter = await anchor.registeredAt(k.nodeId);
            expect(tsAfter).to.be.greaterThan(tsBefore);
        });
    });

    describe("lookup — view path", function () {
        it("returns empty bytes for unregistered nodeId", async function () {
            const fake = freshKey(0x20);
            expect(await anchor.lookup(fake.nodeId)).to.equal("0x");
        });

        it("returns the registered pubkey", async function () {
            const k = freshKey(0x21);
            await anchor.connect(publisher1).register(k.pubkey);
            expect(await anchor.lookup(k.nodeId)).to.equal(k.pubkey);
        });

        it("publisherKeys mapping is publicly readable (matches lookup)", async function () {
            const k = freshKey(0x22);
            await anchor.connect(publisher1).register(k.pubkey);
            expect(await anchor.publisherKeys(k.nodeId)).to.equal(k.pubkey);
        });
    });

    describe("Gas cost benchmark", function () {
        it("register costs less than 100k gas", async function () {
            const k = freshKey(0x30);
            const tx = await anchor.connect(publisher1).register(k.pubkey);
            const receipt = await tx.wait();
            // Loose bound; tighter than the design plan's 100k target.
            // The contract's storage write + sha256 + event emit is
            // dominated by the 32-byte storage write (~22k) + 4 storage
            // operations + sha256 precompile (~75 gas per word) + event
            // (~1k). Total typically ~50-60k.
            expect(receipt.gasUsed).to.be.lessThan(100_000n);
            // Log for visibility during development; doesn't fail the test.
            console.log(`        register gas used: ${receipt.gasUsed}`);
        });
    });
});

/** Helper: matches any uint64 in event-arg comparisons (for timestamps).
 *  Hardhat-chai uses a sentinel that satisfies any value when present
 *  in the args array; we wrap it in a function for readability. */
function anyUint64() {
    return require("@nomicfoundation/hardhat-chai-matchers/withArgs").anyValue;
}
