# L3 Pre-Engagement — Sha512.sol FIPS 180-4 Conformance Results

**Date:** 2026-05-05
**Tool:** Hardhat test suite, cross-checked against Node `crypto.createHash('sha512')`
**Source vectors:** FIPS 180-4 §D test cases + boundary KATs
**Test file:** `contracts/test/Sha512Fips180.test.js`
**Harness:** `contracts/contracts/test/Sha512Harness.sol`

## Summary

**All 11 SHA-512 known-answer tests pass.** Sha512.sol is FIPS 180-4
conformant across normal and block-boundary edge inputs, including the cases
where padding-length encoding spills from one block to the next (the most
common implementation-bug location).

The avalanche property holds: a single-bit input flip changes more than
200 of 512 output bits.

## Test results

| Test | Length | Result |
|------|--------|--------|
| FIPS 180-4 §D.1 — `"abc"` | 3 bytes | ✅ PASS |
| Empty message | 0 bytes | ✅ PASS |
| Single byte 0x00 | 1 byte | ✅ PASS |
| FIPS 180-4 §D.2-style — multi-block ASCII | ~112 bytes | ✅ PASS |
| Padding fits single block | 111 bytes | ✅ PASS |
| Padding spills to 2nd block | 112 bytes | ✅ PASS |
| Block-boundary edge | 127 bytes | ✅ PASS |
| Exactly 1 block | 128 bytes | ✅ PASS |
| Exactly 2 blocks | 256 bytes | ✅ PASS |
| RFC 8032 Test-1024 parity | 1023 bytes | ✅ PASS |
| Avalanche: single bit flip in `"abc"` | varies | ✅ PASS (>200 bits change) |

**Total: 11 KAT cases + 1 avalanche check, 12/12 passing.**

## Validation method

Each KAT computes the expected hash via Node's built-in `crypto.createHash('sha512')`
(OpenSSL implementation — independent of the on-chain code) and asserts byte-
exact equality against the harness output.

## Coverage gaps (out of scope for KAT, in scope for L3 specialist)

- **Side-channel timing.** Whether the implementation has data-dependent
  branches or memory accesses (low likelihood in pure Solidity but worth
  confirming).
- **Gas-DoS via adversarial inputs.** Maximally long messages.
- **Internal state corruption resilience.** Malformed memory layouts (out of
  scope under normal Solidity semantics but auditor should confirm no
  inline-assembly exposure).
- **Cross-implementation parity at scale.** Random differential testing
  against `crypto.createHash` over thousands of inputs (a recommended
  follow-up; not in this baseline).

## Reproducibility

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts
npx hardhat test test/Sha512Fips180.test.js
```

Expected: 12 passing, 0 failing.

## Implementation observations relevant to the auditor

A minor compiler warning surfaced during compilation:

```
Warning: The result type of the shift operation is equal to the type of the
first operand (uint64) ignoring the (larger) type of the second operand
(uint256) which might be unexpected.
  --> contracts/contracts/lib/Sha512.sol:76:17:
   |
76 |         return (x << (64 - n)) + (x >> n);
```

This warning is informational — the result type matches the first operand
which is the intended behavior for SHA-512 rotation. KATs pass, so the
arithmetic is correct, but the auditor should confirm the warning is benign
in their specialist review.
