# L3 Pre-Engagement — Upstream Port Diff

**Date:** 2026-05-05
**Source upstream:** [chengwenxi/Ed25519](https://github.com/chengwenxi/Ed25519) on GitHub
**Upstream branch:** `main`
**Upstream files:**
- `contracts/libraries/Ed25519.sol` (883 LoC)
- `contracts/libraries/Sha512.sol` (304 LoC)

**Local files:**
- `contracts/contracts/lib/Ed25519Lib.sol` (887 LoC, +4 from upstream)
- `contracts/contracts/lib/Sha512.sol` (328 LoC, +24 from upstream)

**License:** Apache-2.0 (preserved from upstream).

## Summary

**The PRSM port is mechanical and minimal.** It bumps the Solidity pragma
from 0.6.8 to 0.8.22, renames the library, switches to a named import, and
wraps arithmetic-heavy functions in `unchecked { ... }` blocks (required
because Solidity 0.8.0+ enables checked arithmetic by default).

**No algorithmic changes. No constants modified. No control flow changed.
No new logic introduced.** The port is faithful at the transformation level.

This is the strongest possible posture for the L3 specialist audit: any
finding in the underlying math is inherent to the chengwenxi upstream, not
introduced by PRSM. The auditor's review reduces to (a) verify the port is
faithful (~30 min of diff inspection), and (b) audit the underlying math (the
actual specialist work).

## Diff: Ed25519Lib.sol

Total: 4 line-pair changes, 7 lines added, 3 deleted.

```diff
2c2
< pragma solidity ^0.6.8;
---
> pragma solidity ^0.8.22;
4c4
< import "./Sha512.sol";
---
> import {Sha512} from "./Sha512.sol";
6c6
< library Ed25519 {
---
> library Ed25519Lib {
8a9
>         unchecked {
273a275
>             }
281a284
>         unchecked {
882a886
>             }
```

**Interpretation:**
- Lines 2, 4, 6: cosmetic (pragma, named import, library rename). Zero
  semantic impact.
- Lines 9 + 275: `unchecked` opens around the body of `pow22501()`, closing
  brace at the function's end. This is required for 0.8.x to preserve
  upstream's pre-0.8.0 wrapping arithmetic semantics.
- Lines 284 + 886: same pattern around `verify()`.

**Risk assessment:** `unchecked` blocks remove overflow/underflow reverts. In
the upstream 0.6.x code, no such reverts existed (pre-0.8.0 default). The
wrapping arithmetic semantics are intentional in elliptic-curve field
operations (mod p arithmetic uses `mulmod`/`addmod` opcodes which already
handle modular reduction). Wrapping `pow22501` and `verify` in `unchecked`
preserves the upstream behavior exactly.

**Auditor verification path:** Confirm that no operation inside the
`unchecked` blocks could produce an unintended overflow that would be
silently masked. The math operates mod `p = 2^255 - 19` via `mulmod` and
explicit reductions; native uint256 wrapping is semantically irrelevant where
modular reduction is explicit. This is a high-confidence delta.

## Diff: Sha512.sol

Total: 13 line-pair changes, 24 lines added, 0 deleted.

```diff
2c2
< pragma solidity ^0.6.8;
---
> pragma solidity ^0.8.22;
[12 × `unchecked { ... }` block insertions across helper functions
 (preprocess, ROTR, SHR, Ch, Maj, sigma0, sigma1, gamma0, gamma1, hash) and
 the FuncVar struct manipulator]
```

**Interpretation:**
- Line 2: pragma bump.
- 12 helper functions wrapped in `unchecked` for the same reason as in
  Ed25519Lib. SHA-512 operates on 64-bit lanes via uint64 arithmetic;
  wrapping is semantically correct because SHA-512 is defined modulo
  2^64 throughout.

**Risk assessment:** Wrapping arithmetic is the SPECIFIED behavior of
SHA-512. The `unchecked` blocks restore exactly that. No semantic change.

**Auditor verification path:** Confirm each `unchecked` block's range matches
exactly what the upstream 0.6.x code did under default arithmetic. KAT pass
(see `test-vectors-sha512.md`) provides empirical confirmation.

## Summary table

| Category | Upstream | PRSM | Delta | Risk |
|----------|----------|------|-------|------|
| Pragma | ^0.6.8 | ^0.8.22 | Bump | None — Solidity-level, no semantic change |
| Library naming | `Ed25519` | `Ed25519Lib` | Rename | None — naming hygiene |
| Imports | global | named | Idiom | None |
| Arithmetic semantics | implicit-wrapping (0.6.x default) | explicit-wrapping via `unchecked` | Behavioral parity | Low — restores upstream behavior |
| Algorithm | RFC 8032 §5.1.7 | RFC 8032 §5.1.7 | None | N/A |
| Constants | RFC 8032 + Curve25519 | identical | None | N/A |
| Control flow | per upstream | per upstream | None | N/A |

## Auditor scope reduction

This diff document **reduces the auditor's scope** from "audit a 1,200 LoC
hand-rolled Ed25519+SHA-512 from scratch" to "audit chengwenxi/Ed25519 (the
upstream) plus confirm a faithful 4+13-line port."

That is materially cheaper. Vendors estimating the engagement should price
accordingly:

- Faithful-port confirmation: ~0.5–1 day.
- Underlying math review (chengwenxi/Ed25519 + Sha512): the bulk of the
  engagement, 1–3 weeks.

If the auditor prefers, they may treat the upstream as their primary review
target and PRSM's port as a thin wrapper. PRSM accepts either framing.

## Upstream maintenance status (FYI for the auditor)

Per a search of the upstream repo as of 2026-05-05:
- 2 commits total on `main`
- 11 stars, 3 forks
- No published independent audit
- Apache-2.0 licensed
- Test corpus exists in upstream's `test/` directory but does not include
  the canonical RFC 8032 §7.1 vectors (PRSM has added these — see
  `test-vectors-ed25519.md`)

This is the strongest argument for engaging a specialist: the upstream has
not been independently audited. Any findings the L3 vendor produces also
benefit upstream (and PRSM should consider upstreaming fixes).
