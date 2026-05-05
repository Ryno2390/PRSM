const { expect } = require("chai");
const { ethers } = require("hardhat");

// Phase 3.1 Task 5 parity check: verify that the Solidity-side leaf
// hash matches the Python-side hash for identical input.
//
// Python reference values are computed in
// `tests/unit/test_settlement_merkle.py` + the sanity script in the
// Task 5 commit message. They must match this file's expected hashes.
//
// The canonical input:
//   leaf.jobIdHash             = keccak256("test-job")
//   leaf.shardIndex            = 0
//   leaf.providerIdHash        = keccak256("test-provider")
//   leaf.providerPubkeyHash    = keccak256("test-pubkey")
//   leaf.outputHash            = keccak256("test-output")
//   leaf.executedAtUnix        = 1700000000
//   leaf.valueFtns             = 10**18
//   leaf.signatureHash         = keccak256("test-signature")
//   leaf.signingMessageHash    = keccak256("test-signing-msg")
//
// Expected leaf hash = Solidity keccak256(abi.encode(leaf))
//                    = Python keccak(abi_encode(tuple(...)))
//                    = 0x572b6e5a854385da2e97f9e18542df44d45b75123d40041c9c0960488aa6762a
//
// The leaf grew by one bytes32 field (signingMessageHash) per L2 audit
// C-INT-01 remediation. The previous parity hash
// (0x1683ea2c92a15b838b3d767faff1ad8151b9597e9e1f3f4353eef6424ed04832) is
// retained in audits/findings/consolidated.md §2 CRIT-1 as the pre-fix
// reference.

describe("Solidity ↔ Python ReceiptLeaf parity", function () {
  it("produces the expected leaf hash for the canonical fixture", async function () {
    const leaf = {
      jobIdHash: ethers.keccak256(ethers.toUtf8Bytes("test-job")),
      shardIndex: 0,
      providerIdHash: ethers.keccak256(ethers.toUtf8Bytes("test-provider")),
      providerPubkeyHash: ethers.keccak256(ethers.toUtf8Bytes("test-pubkey")),
      outputHash: ethers.keccak256(ethers.toUtf8Bytes("test-output")),
      executedAtUnix: 1700000000,
      valueFtns: 10n ** 18n,
      signatureHash: ethers.keccak256(ethers.toUtf8Bytes("test-signature")),
      signingMessageHash: ethers.keccak256(ethers.toUtf8Bytes("test-signing-msg")),
    };

    // Compute Solidity-equivalent hash via ethers.js ABI encoder.
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
    const solidityHash = ethers.keccak256(encoded);

    // MUST match the Python-computed hash.
    const expectedFromPython =
      "0x572b6e5a854385da2e97f9e18542df44d45b75123d40041c9c0960488aa6762a";

    expect(solidityHash).to.equal(expectedFromPython);
  });
});
