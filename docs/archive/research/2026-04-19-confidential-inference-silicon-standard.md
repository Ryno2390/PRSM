# PRSM Confidential Inference Silicon Standard — Planning Doc

**Status:** Pre-draft / scope-setting. Not a spec yet.
**Owner:** PRSM Foundation (standard); Prismatica (first implementer).
**Date opened:** 2026-04-19
**Horizon:** 5-10 years from first silicon tape-out to production-qualified meganode deployment.

---

## Why this document exists

Phase 2's launch UX thesis (see `docs/2026-04-12-phase2-remote-compute-plan.md`) is carried by T3 cloud-arbitrage nodes running on rented H100s with NVIDIA Confidential Compute (CC). That stack is adequate for Tier A / B inference workloads where the model publisher is comfortable with "the hyperscaler and NVIDIA are both honest, and H100 CC has not yet been publicly broken."

It is **not** adequate for frontier AI labs weighing whether to publish $500M-training-cost SOTA weights to a permissionless market. Research track R8 (see `docs/2026-04-14-phase4plus-research-track.md`) enumerates the threat model in full. The residual risk after H100 CC is:

1. Side-channel attacks (cache timing, power analysis, memory-bus probes, Rowhammer) against H100 CC — ongoing academic research.
2. The hyperscaler operating the physical machine can always access VRAM via host-side primitives; PRSM's operator identity is not the underlying trust root.
3. NVIDIA signs attestation quotes. If NVIDIA is compromised, compelled, or simply wrong, the whole defense collapses.
4. H100 CC has no public open specification; independent audit is limited; the attestation root is proprietary.

**This document scopes a silicon standard that addresses the residual risk, is owned by the PRSM Foundation, is openly published, and is implementable by any qualified fab/designer.** Prismatica will be a first implementer and early test venue, not the standard-setter.

## What the standard must guarantee

A compliant chip, when integrated into a PRSM meganode, MUST provide the following guarantees, architecturally — not by operational promise:

1. **Weight sealing.** Model weights are stored in chip-internal memory encrypted to a per-chip hardware root-of-trust. The root never leaves the die. Any attempt to read plaintext weights requires either the chip's cooperation via an attested inference call or physical destruction of the chip.

2. **Attested execution.** The chip signs an attestation quote over: (a) the firmware/ROM version currently running, (b) the cryptographic hash of the model weights sealed on the chip, (c) a nonce provided by the model publisher or the PRSM dispatcher, (d) the output produced for a given input. The attestation key is anchored in the chip's hardware root.

3. **Tamper evidence.** Physical intrusion into the chip package (decapsulation, probing, laser fault injection, focused-ion-beam edits) destroys either the chip's operation or the sealed weights with a publisher-acceptable probability (target: ≥99.99%). This follows established HSM / Secure Enclave design patterns — tamper meshes, active shields, fuses, environmental sensors.

4. **Side-channel resistance at specified thresholds.** Power analysis, electromagnetic emanations, timing channels, and cache channels MUST NOT leak sufficient information to reconstruct weights after N inferences, for a specified N and reconstruction-fidelity bound. Concrete numbers to be defined by the Foundation's spec working group based on academic side-channel literature.

5. **On-chain-verifiable attestation.** The attestation signature chain terminates at a root certificate registered in PRSM's ProvenanceRegistry (or a dedicated AttestationRegistry contract). Any PRSM client can verify an inference attestation against on-chain state without trusting any off-chain CA. Per-chip identity certificates are issued by the manufacturer and cross-signed by the Foundation's compliance process; revocation is on-chain.

6. **No vendor-proprietary hooks.** The chip's attestation mechanism MUST NOT require any off-chain API operated by a single vendor (including the manufacturer, Prismatica, or the Foundation itself) to be available at inference time. Network independence is a design requirement.

## What the standard must NOT lock in

To remain open and implementable by multiple parties, the standard MUST avoid:

- **Specific manufacturing processes, node sizes, or fabs.** A compliant chip at 7nm, 5nm, or 3nm is all fine. A compliant chip fabbed at TSMC, Samsung, Intel Foundry, or a future entrant is all fine.
- **Specific ISA for the compute units.** Candidates include open instruction sets (RISC-V with tensor extensions), proprietary licensed cores (ARM Neoverse), or clean-sheet designs. The standard specifies the *security boundary* and *attestation protocol*, not the compute microarchitecture.
- **Specific cryptographic primitives beyond algorithm class.** "Lattice-based KEM conforming to FIPS 203" leaves room for implementation flexibility. "Use this specific library" does not.
- **Mandatory vendor-specific signed blobs.** The attestation root MUST be the chip's manufacturer; the chip MUST NOT require signed runtime code from Prismatica, the Foundation, or anyone else to function.

## Governance separation

| Responsibility | Owner |
|---|---|
| Threat model maintenance | PRSM Foundation |
| Compliance test suite (hardware + protocol) | PRSM Foundation |
| Certification process (review, approval, revocation) | PRSM Foundation, with independent auditor reviewers |
| On-chain AttestationRegistry contract | PRSM Foundation (deployed on Base; governed by existing PRSM governance process) |
| First chip design and tape-out | Prismatica (or alternate; any qualified party may pursue first implementation) |
| Subsequent chip designs | Any party; compliance is the gate, not brand |
| Meganode hardware integration | Prismatica (for its T4 operations); any other party for theirs |
| Chip-per-unit pricing and commercial terms | Each implementer sets its own |

**The Foundation commits to not developing chips itself.** Foundation activity on the silicon track is limited to: standard maintenance, compliance testing, registry operation, and grants to academic red-teams to attack candidate designs. This is the structural guarantee that the Foundation's authority is adversarial and corrective, not competitive with implementers.

**Prismatica's commitments as first implementer:**

1. Tape-out the first compliant chip at a manufacturing process and cost structure that allows a meganode operator to earn FTNS arbitrage at steady state.
2. Contribute test vectors, side-channel-attack findings, and hardware debug data back to the Foundation's compliance test suite as a condition of certification — failure to contribute can trigger decertification.
3. Commit, in writing, not to seek exclusivity or any form of preferential Foundation treatment. Public, auditable commitment.
4. Open-source the chip's firmware and any attestation agent code. Circuit-level designs may remain proprietary for trade-secret reasons, but the security boundary must be fully specified in the standard and auditable externally.

## Timeline (speculative, to be refined)

| Milestone | Target year | Gating |
|---|---|---|
| Foundation convenes spec working group | 2027 | Contingent on R8 research maturing to a defense-stack recommendation |
| Draft 0.1 of silicon standard published | 2028 | |
| First implementer tape-out announced | 2029-2030 | Contingent on spec maturity + implementer capital |
| First certified chip available for integration | 2030-2031 | |
| First Prismatica T4 meganode deployment on certified silicon | 2031-2032 | |
| Second independent implementer certified | 2032-2033 | Critical for breaking Prismatica monopoly |
| Frontier lab publishes SOTA weights to PRSM citing silicon assurance | 2033+ | The market-truth signal |

Timeline is aggressive for a greenfield silicon program and assumes adequate capital (likely $100-500M across implementers, amortized across the lifetime of certified chips). If the timeline slips to the longer end, PRSM operates in the interim on the TEE-based stack (H100 CC + R8 layer-1 defenses) with the understanding that frontier lab acceptance is partial.

## Fallback positioning

If custom silicon proves infeasible (capital, expertise, timeline), the defense stack degrades gracefully:

- **R8 layers 1-4 without layer 5 (custom silicon):** H100 CC + fingerprint detection + output watermarking + cryptographic sharding. This is probably sufficient for "nearly frontier" labs to publish (12-18 months behind true SOTA). Market remains viable but the very-top-frontier stays off PRSM.
- **R8 layers 1-2 only:** the minimum viable anti-exfiltration stack for Tier B confidential inference. Sufficient for mid-tier proprietary models (e.g., enterprise-specialized fine-tunes), probably not for flagship frontier releases.

The value of the silicon standard is partly architectural (close the side-channel and trust-root gaps) and partly *economic signaling* — announcing the roadmap and governance model now makes frontier labs willing to engage in conversations they would otherwise dismiss. A well-specified, well-governed, open standard has strategic value even before any silicon exists.

## Cross-references

- **R8 (Anti-Exfiltration Architecture for Frontier-Model Inference):** `docs/2026-04-14-phase4plus-research-track.md`. The research track that produces the threat model and defense-stack evaluation this standard is built on.
- **Phase 2 Launch UX thesis and Line item C (TEE attestation):** `docs/2026-04-12-phase2-remote-compute-plan.md`. The near-term TEE baseline this standard complements, not replaces.
- **Historical confidential-compute spec:** `docs/CONFIDENTIAL_COMPUTE_SPEC.md`. Adjacent but distinct scope (user data confidentiality, not model-weight confidentiality).
- **Glossary:** `docs/glossary.md`. Note: the compute-verification tiers A/B/C defined there need an extension for attested-silicon-backed inference (call it "Tier C+" or a new naming scheme). Flag for glossary revision when this standard advances past draft.
- **Risk Register entry G5 (TEE side-channel):** `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/My Vault/Agent-Shared/Risk_Register.md`. The current TEE-only risk that this standard ultimately retires.

## Next actions (for when R8 promotion triggers fire)

1. Foundation convenes a spec working group (5-7 members: security researchers, HSM designers, confidential-compute standards experts, a frontier-lab liaison, a cryptographer, a governance lawyer).
2. Working group produces a draft threat model matching R8's research findings.
3. Compliance test suite drafted in parallel with spec — never spec without tests.
4. Prismatica signs a public commitment to the governance terms above before any Foundation-granted standard-setting seat.
5. Standard published as a numbered Foundation RFC, subject to community review period, then adopted or revised.

Until promotion, this document is a placeholder to hold position on the architecture and governance commitments, so that "we will not let Prismatica become the new NVIDIA" is an auditable record, not just a conversational claim.
