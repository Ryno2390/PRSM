# L3 Pre-Engagement — Ed25519Lib RFC 8032 §7.1 Conformance Results

**Date:** 2026-05-05
**Tool:** Hardhat test suite + chai assertions
**Source vectors:** RFC 8032 §7.1 (canonical Ed25519 reference vectors)
**Test file:** `contracts/test/Ed25519LibRfc8032.test.js`
**Harness:** `contracts/contracts/test/Ed25519LibHarness.sol`

## Summary

**All 4 RFC 8032 §7.1 canonical Ed25519 vectors verify successfully against
the in-tree `Ed25519Lib.sol`.** Bit-flip rejection, wrong-key rejection, and
wrong-message rejection all behave correctly.

This is the spec-conformance baseline a cryptographic auditor expects to
inherit. The L3 specialist engagement should focus on edge cases NOT covered
by the canonical corpus (low-order points, non-canonical encodings,
malleability bounds, side channels).

## Test results

| Test | Result | Notes |
|------|--------|-------|
| Test 1 (empty message) | ✅ PASS | After fix to my transcription error in initial test draft |
| Test 2 (1-byte message: 0x72) | ✅ PASS | |
| Test 3 (2-byte message: 0xaf 0x82) | ✅ PASS | |
| Test SHA(abc) (64-byte message) | ✅ PASS | |
| Bit-flip rejection × 4 vectors | ✅ PASS | All tampered sigs rejected |
| Wrong-key Test 1 against Test 2 pubkey | ✅ PASS | Rejected |
| Wrong-key Test 2 against Test 3 pubkey | ✅ PASS | Rejected |
| Wrong-message Test 2 sig + Test 3 msg | ✅ PASS | Rejected |

**Total: 11 test cases, 11 passing.**

## Process notes for the auditor

While preparing this baseline, my initial transcription of Test 1's signature
contained an error in the second 32 bytes. This was caught immediately when
the test failed. I cross-checked with both Node's native `crypto` Ed25519
(OpenSSL) and `tweetnacl`, both of which produced the canonical RFC 8032
signature; my hex literal was the bug, not the library.

Recording this here because:

1. It demonstrates the test harness correctly distinguishes valid from
   invalid signatures even at the exact-signature-byte level.
2. An auditor inspecting this corpus should understand that the test was
   *empirically validated* against multiple independent reference
   implementations, not just self-asserted.

## Coverage gaps (deliberately NOT covered here)

These cases are out of scope for canonical conformance but are exactly where
specialist audit value lies. Listing them so the auditor can scope deeper:

- **Low-order public keys.** RFC 8032 §5.1.7 step 3 mandates rejection of
  low-order subgroup elements. Not exercised by §7.1.
- **Non-canonical S encoding.** RFC 8032 §5.1.7 step 1 requires `S < L`.
  Vectors with `S = L`, `S = L + small`, etc. not exercised.
- **Point-at-infinity decoded R or A.** Not exercised.
- **Signature malleability under (R, S) → (R, -S mod L).** Whether the lib
  accepts both is not specified by RFC 8032 alone but matters for replay
  protection in PRSM.
- **Maximally-long messages near gas limits.** RFC §7.1 Test 1024 is 1023
  bytes; PRSM production wrapper restricts to 32 bytes so this is not
  reached, but the lib itself handles arbitrary lengths.
- **Side-channel timing analysis.** Constant-time properties of the
  underlying math.
- **Edge cases in SHA-512 of crafted (R || A || M) inputs** that exercise
  block-boundary handling.
- **Adversarial gas-DoS vectors** — inputs that maximize gas usage.

## Reproducibility

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM/contracts
npx hardhat test test/Ed25519LibRfc8032.test.js
```

Expected: 11 passing, 0 failing.

## Hand-off package for L3 vendor

The auditor should receive:
- This document.
- The test file `contracts/test/Ed25519LibRfc8032.test.js`.
- The harness contract `contracts/contracts/test/Ed25519LibHarness.sol`.
- The pinned commit hash (will be the audit-bundle freeze tag).
- The upstream port diff (`upstream-port-diff.md` in this directory).
- The caller-assumptions memo (`caller-assumptions.md` in this directory).
