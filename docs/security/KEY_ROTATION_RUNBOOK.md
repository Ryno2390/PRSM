# PRSM Key Rotation & Compromise-Recovery Runbook

**Version:** 1.0 (initial)
**Date:** 2026-05-05
**Status:** Pre-mainnet — published before incident per Vision §14 commitment
**Audience:** Foundation council, signing-authority holders, founder/deputy-founder, on-call engineering
**Owner:** Foundation council (post-formation); founder until provisioned
**Related:** [`EXPLOIT_RESPONSE_PLAYBOOK.md`](EXPLOIT_RESPONSE_PLAYBOOK.md) (on-chain exploits) · [`SECURITY_RUNBOOK.md`](SECURITY_RUNBOOK.md) (traditional infosec) · [`audits/AUDIT_PLAN.md`](../../audits/AUDIT_PLAN.md) §5 L6c

## 1. Purpose

This runbook defines pre-committed procedures for three key-management
emergencies that DO NOT involve active on-chain exploitation but DO require
fast, deliberate action:

- **Scenario A: Lost or destroyed hardware wallet** (one of the 3 Foundation
  Safe signers). Common causes: device failure, theft, accidental erasure,
  natural disaster.
- **Scenario B: Suspected deployer-key compromise** during or shortly after a
  deploy ceremony, before ownership-transfer to the Safe completes.
- **Scenario C: Suspected Safe-owner seed leak** (one of the 3 Safe signers'
  24-word recovery seeds is believed to have been observed, photographed,
  scanned, or copied).

Each scenario has different urgency, blast radius, and recovery path. This
runbook presents them in parallel so the responder can immediately identify
which applies.

## 2. Why these are pre-committed

Improvised key-rotation under stress almost always introduces NEW key-
management vulnerabilities (rotating to a fresh key on a contaminated machine,
recovering from seed in a way that leaks the new seed, etc.). This runbook
exists so the responder follows a procedure that has been reviewed when calm.

## 3. Roles

- **Founder** (currently Ryne Schultz): incident commander; runs ceremony in
  scenarios A and B; assists in C.
- **Deputy-founder** (per D7 succession protocol — see `docs/2026-04-?-d7-...`):
  succeeds founder in any scenario where founder is unavailable or
  compromised.
- **Foundation council** (post-formation): ratification of any rotation that
  changes Safe owner-set or threshold.
- **Other 2 Safe signers**: must be reachable for any 2-of-3 Safe transaction.
- **Bystander**: any ceremony has a bystander witness recording timestamps
  and outcomes. Bystander does NOT have key access; their role is forensic.

## 4. Pre-conditions for executing this runbook

Before you start any of the three scenarios:

1. **Confirm the trigger condition.** False positives are common (a wallet
   "lost" that's actually in a drawer). Spend 30 minutes searching before
   executing rotation; rotation is destructive (in the sense that the
   rotated-out signer is permanently demoted) and must not be triggered on
   suspicion alone.
2. **Verify identity of any signer claiming compromise.** A social-engineering
   attack would manifest as someone claiming "my seed is leaked, please add
   my new address as signer." Verify out-of-band before acting.
3. **Notify all other Safe signers.** Two-of-three threshold means any
   rotation is itself a 2-of-3 transaction; the surviving signers must be
   available.
4. **Open an incident timeline document.** Plain text, plain timestamps.
   Capture every action, every confirmation, every block number. The
   bystander records this; if no bystander, founder records.

## 5. Scenario A — Lost or destroyed hardware wallet

### Severity: MEDIUM

Treasury is not at risk. The Safe still has 2 of 3 functional signers (assuming
only one device is lost), threshold = 2, so signing power continues
uninterrupted. The risk is **future lock-out** if a second device is lost
before rotation completes.

### A.1 Immediate actions (Hour 0–24)

1. ☐ Confirm the device is unrecoverable. Check usual locations, safe deposit
   box, drawers, travel bag.
2. ☐ Check whether the seed phrase is recoverable from a different physical
   location (the metal seed plate, the paper backup, the deputy-founder copy
   if applicable). **If the seed is recoverable, restore the seed onto a new
   compatible device — this is not a rotation, it's a recovery.** Skip §A.2.
3. ☐ If both device AND all seed copies are unrecoverable: this is a true
   rotation event. Continue to §A.2.
4. ☐ Notify the other 2 Safe signers. Verify their devices + seeds remain
   secure (an attacker who took one might attempt the others).

### A.2 Rotation procedure (Day 1–7)

1. ☐ Acquire a replacement hardware wallet — **DIFFERENT VENDOR** from the
   lost one (per L6 multi-vendor diversity policy). Current Safe signers use
   Ledger Nano S Plus, Trezor Safe 3, OneKey Classic 1S Pure. Replace the lost
   one with whichever brand the surviving signers don't already use, OR with
   the same brand if the lost one is the unique sole-vendor choice.
2. ☐ Generate a fresh seed on the new device. **Air-gapped setup** —
   workstation off the network, camera disabled, no other devices in the
   room. Bystander present.
3. ☐ Record the seed on a metal seed plate (per multi-sig setup protocol).
   Store the plate in a different physical location than any other Safe
   signer's seed.
4. ☐ Capture the new device's first receive address.
5. ☐ Update `~/.prsm-foundation-private/Multi-Sig_Addresses.txt` with the new
   address. (chmod 600. Out-of-repo. Never committed.)
6. ☐ Construct a Safe transaction that simultaneously:
   - Adds the new address as a Safe owner (via `addOwnerWithThreshold`).
   - Removes the lost device's address as a Safe owner (via `removeOwner`).
   - Keeps threshold = 2.

   Use [Safe app UI](https://app.safe.global/) → Settings → Owners → Manage
   Owners → "Replace owner" (this is a single atomic transaction).
7. ☐ The 2 surviving signers sign the transaction. Threshold reached.
   Transaction executes.
8. ☐ Verify on Basescan that:
   - `getOwners()` returns the 3 expected addresses (2 unchanged + 1 new).
   - `getThreshold()` returns 2.
   - The lost-device address is NOT present.
9. ☐ Run `scripts/foundation-safe-health-check.py` against the Safe address
   to confirm the rotation succeeded end-to-end. Health check should report
   3 owners with the new address, threshold 2.
10. ☐ Update audit memory and Foundation council record. Notify any
    third-parties tracking the signer-set (e.g., L4 auditor, L8 counsel,
    any DEX listing forms).

### A.3 Post-rotation review (Week 1–2)

- ☐ Document what was lost, when, and why in incident timeline.
- ☐ Update operational hygiene checklist (L6e — see
  [`L6E_OPS_HYGIENE_REVIEW.md`](L6E_OPS_HYGIENE_REVIEW.md)) if the loss
  revealed a process gap.
- ☐ Verify deputy-founder has up-to-date access information for the new
  signer set (D7 protocol).

## 6. Scenario B — Suspected deployer-key compromise mid-ceremony

### Severity: HIGH (potentially CRITICAL — depends on ceremony state)

The deployer key is the disposable single-use key generated for a deploy
ceremony (per the disposable-deployer pattern documented in
`docs/2026-05-04-task8-deploy-ceremony-lessons.md`). Compromise scenarios:

- Private key was inadvertently exposed (e.g., pasted in chat, screen-shared).
- Deployer machine showed signs of malware mid-ceremony.
- Deployer's home network was suspected to be MITM'd.
- Funded deployer received unexpected transactions before ownership-transfer.

### B.1 Critical question — has ownership-transfer to the Safe completed?

This is the gating question that determines blast radius:

- **YES, ownership has transferred.** Deployer key compromise is now a
  spent-key issue. Deployer can no longer modify any deployed contract.
  Treasury is safe. Severity drops to "investigate the leak vector but no
  active fund-loss risk." Continue at §B.3.
- **NO, ownership has NOT transferred.** Deployer key still controls the
  newly-deployed contracts. Severity is HIGH or CRITICAL. Continue
  immediately at §B.2.

### B.2 Pre-handoff compromise — abort and redeploy

1. ☐ **DO NOT continue the ceremony.** Do not transfer ownership from the
   compromised deployer.
2. ☐ Sweep all funds from the deployer address back to a clean foundation
   address — `scripts/sweep-deployer.py` (committed, tested). Even if the
   key is exposed, sweep first to deny the attacker gas funding for any
   action they might attempt.
3. ☐ Mark the just-deployed contracts as **abandoned**. Update the deploy
   manifest with status=abandoned and reason=deployer-key-compromise.
4. ☐ Announce internally that the abandoned addresses are NOT canonical.
   Do NOT publish them in any user-facing channel; do NOT update Forta bot
   contract registry to point at them.
5. ☐ **Generate a fresh disposable deployer key on a clean machine.**
   - Fresh OS install or known-clean machine.
   - Air-gapped key generation.
   - Bystander present.
   - Use Python `eth_account.Account.create()` or equivalent.
   - Verify `0x` prefix on output (per L1 lesson from 2026-05-04 ceremony).
6. ☐ Re-execute the deploy ceremony from scratch with the fresh key.
   Reference `scripts/rehearse-deploy.sh` for the canonical sequence; do
   not skip preflight checks.
7. ☐ Run `verify-audit-bundle-deployment.js` (or `verify-provenance-deployment.js`
   for Phase 1.3) post-handoff with `EXPECTED_OWNER` set to the Foundation
   Safe address. Confirm all owner() calls return the Safe.
8. ☐ Investigate the leak vector. Document in incident timeline.

### B.3 Post-handoff compromise — defensive cleanup

1. ☐ Verify ownership did transfer cleanly to the Safe — re-run
   `verify-audit-bundle-deployment.js` with `EXPECTED_OWNER=<Safe>`. If any
   contract still shows the deployer as owner, treat as §B.2.
2. ☐ Sweep any residual funds from the deployer address.
3. ☐ Mark the deployer address as RETIRED in
   `~/.prsm-foundation-private/Multi-Sig_Addresses.txt` with timestamp and
   reason.
4. ☐ Investigate the leak vector. Even though no fund-loss occurred, the
   leak is forensically important — the same attack pattern could target a
   future ceremony.
5. ☐ Update deploy ceremony lessons doc if the leak revealed a
   process gap.

## 7. Scenario C — Suspected Safe-owner seed leak

### Severity: HIGH

Different from Scenario A in that the seed is now believed to be in
adversary hands (not just lost). The compromised signer's signing power must
be removed before the attacker can join 1 honest signer in a 2-of-3 to drain
the treasury.

### C.1 Trigger conditions (any one is sufficient)

- Seed phrase was photographed by an unknown party.
- Seed plate was discovered to be in a non-secure location.
- Signer's home was burgled and the seed plate is missing or possibly viewed.
- Signer suspects their device has been physically tampered with.
- A failed phishing attempt against the signer included specifics that
  suggest prior reconnaissance of the seed.
- Signer cannot rule out a sophisticated remote-key-extraction attack.

### C.2 Time-critical first-hour actions (Hour 0–1)

The attacker needs ONE collaborator + the threshold to drain. They have one
of three signing keys; they need to either compromise a second, or trick a
second into signing a malicious tx. **Therefore: every minute the
compromised key remains a valid Safe owner is a minute the attacker has to
operationalize a second-signer attack.**

1. ☐ Founder + remaining 2 signers convene immediately (synchronous video,
   not async chat).
2. ☐ Confirm the leak claim. If the compromised signer is making the claim:
   verify their identity (out-of-band — known voice, agreed challenge phrase,
   etc.). Social engineering of THIS step is itself an attack vector.
3. ☐ **PREPARE a 2-of-3 Safe transaction that removes the compromised
   signer.** Do NOT execute yet (see C.3 first if there's any Safe-held
   liquidity that could be drained in the time it takes to also rotate).
4. ☐ Open a Forta alert subscription on the Safe address (if not already
   active). Any pending tx will be visible.

### C.3 Optional — preemptive treasury move (Hour 1–2)

If the Safe holds material balance and the leak is fresh enough that no
attacker tx has been seen yet:

- ☐ Construct an additional Safe tx that moves funds to a fresh, clean
  multi-sig with the 2 uncompromised signers + a NEW signer (using a
  freshly-generated hardware wallet).
- ☐ This is more complex than rotation alone — it requires standing up a
  brand-new Safe. Only do this if (a) treasury balance > $X (to be defined
  by Foundation council; suggested floor $1M USD-equiv) AND (b) the leak is
  fresh enough that you believe you have a window before attacker action.

For Phase 1.3 mainnet today (2026-05-05), Foundation Safe holds modest fee
accumulation. **Recommend C.4 rotation, not C.3 treasury move.**

### C.4 Standard rotation (Hour 1–24)

Execute Scenario A's §A.2 rotation procedure with the modification that
the compromised signer's address is the one being removed. The 2
uncompromised signers sign the rotation tx; the compromised signer does NOT
participate.

After execution:

- ☐ Verify on-chain — compromised address is NO LONGER in `getOwners()`.
- ☐ Run `foundation-safe-health-check.py` to confirm new state.
- ☐ Update `~/.prsm-foundation-private/Multi-Sig_Addresses.txt`.
- ☐ Notify Foundation council, L4 auditor, L8 counsel, integrators.

### C.5 Post-incident (Day 1–14)

- ☐ Forensic investigation of the leak vector. Was it physical (burglary),
  remote (malware), social (phishing)?
- ☐ Update threat model if a new vector is discovered.
- ☐ Update deputy-founder access information.
- ☐ Document in `docs/security/incidents/` (private repo if creating
  identifying details about the compromised signer).

## 8. Cross-scenario invariants

Regardless of which scenario applies:

- **Never broadcast any signer address change** until on-chain verification
  confirms the transaction landed and the new owner-set is what's expected.
  Premature broadcast can be exploited by an attacker who has compromised
  the broadcasting channel.
- **Never store a fresh seed on the same machine** that ran the rotation
  procedure. Rotation is performed from a workstation; seed material is
  never present on it.
- **Bystander witness, not just signer.** Even for routine rotations, having
  one non-signer present prevents the rotation itself from becoming a single
  point of social-engineering failure.
- **Update the documentation IMMEDIATELY after rotation.** Stale docs that
  reference rotated-out signers are themselves a security risk (someone
  consulting old docs might attempt to coordinate a transaction with a
  retired signer who can no longer participate).

## 9. Drill cadence

This runbook should be **drilled** annually, folded into the Q4 L6e
quarterly hygiene review (see
[`L6E_OPS_HYGIENE_REVIEW.md`](L6E_OPS_HYGIENE_REVIEW.md) §5). Annual
cadence reflects the cost of a full rotation drill vs. the lighter
quarterly checklist. Drill procedure:

- Use a testnet Safe (Base Sepolia) with the same threshold structure.
- Simulate Scenario A (lost device) + execute full A.2 rotation.
- Verify health-check script reports correct state.
- Time the procedure end-to-end. Log opportunities to compress the timeline.

Document each drill outcome in `docs/security/drills/`.

## 10. Operationalization status

Engineering-side complete:
- ✅ Multi-sig addresses file location documented (out-of-repo).
- ✅ Sweep-deployer script committed and tested (`scripts/sweep-deployer.py`).
- ✅ Foundation Safe health-check script
  (`scripts/foundation-safe-health-check.py`).
- ✅ Verify-audit-bundle-deployment script with EXPECTED_OWNER guard
  (`contracts/scripts/verify-audit-bundle-deployment.js`).

Pending operational items (founder-side):
- ☐ Annual rotation drill on Base Sepolia testnet (next: 2027-05).
- ☐ Foundation council ratification of this runbook.
- ☐ Deputy-founder briefing per D7 protocol.
- ☐ Bystander witness role formalized (which signers / non-signers are
  bystander-eligible).
- ☐ Out-of-band identity verification protocol formalized (challenge phrases
  per signer pair).
