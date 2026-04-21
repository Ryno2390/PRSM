# PRSM-CIS-1: Confidential Inference Silicon Standard

**Document identifier:** PRSM-CIS-1
**Version:** 0.1 Draft
**Status:** Pre-working-group draft for community review. Not yet normative. Implementers should treat this as scoping material, not a certification target.
**Date:** 2026-04-21
**Editor:** PRSM Foundation (drafting authority pending working-group convocation)
**Supersedes:** none (first issue)

**Preceding materials:**
- `docs/2026-04-19-confidential-inference-silicon-standard.md` — original scope-setting doc. Retained as companion context.
- `docs/2026-04-14-phase4plus-research-track.md` §R8 — threat model origin and research dependencies.

---

## 1. Preface

### 1.1 Purpose

This standard specifies the architectural requirements for a Confidential Inference Silicon (CIS) device: a semiconductor integrated circuit capable of performing AI model inference on plaintext weights while guaranteeing, by construction, that those weights cannot be exfiltrated by any party with physical access to the chip, its host system, or the hosting data center.

A compliant chip — when integrated into a PRSM meganode according to this standard — provides assurances sufficient for a frontier AI lab to publish proprietary SOTA model weights to an otherwise-permissionless inference marketplace. Absent this standard or equivalent assurance, PRSM's inference market is effectively restricted to open-weights models, cap­ping the network's addressable market.

### 1.2 Scope

This document specifies:

- (§4) The threat model the standard is designed to defeat.
- (§5–§11) The architectural requirements a compliant chip MUST satisfy.
- (§12) The attestation protocol and wire format.
- (§13) The on-chain registry contracts that anchor attestation.
- (§14) The compliance test suite framework and how certified labs conduct conformance testing.
- (§15) The certification process, revocation procedure, and governance model.
- (§16) Conformance levels (C1 / C2 / C3) with graduated requirements.

This document does NOT specify:

- Specific instruction-set architectures for the compute units.
- Specific manufacturing processes, nodes, or fabs.
- Specific cryptographic libraries (only primitive classes and standards).
- Specific pricing, commercial terms, or business arrangements for any implementer.
- The PRSM Foundation's internal budgeting or grant process.

### 1.3 Normative language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119.

### 1.4 Versioning and amendment

PRSM-CIS-1 is the first issue. Subsequent versions (PRSM-CIS-1.x, PRSM-CIS-2, etc.) are adopted by Foundation governance process documented in §15.4. Implementations certified against one version remain certified against that version until its deprecation period expires; the Foundation publishes a transition schedule of at least 24 months for breaking revisions.

---

## 2. Normative references

The following documents, in whole or in part, are normatively referenced in this standard.

- **FIPS PUB 140-3** — Security Requirements for Cryptographic Modules.
- **FIPS PUB 203** — Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM).
- **FIPS PUB 204** — Module-Lattice-Based Digital Signature Algorithm (ML-DSA).
- **RFC 9334** — Remote Attestation Procedures (RATS) Architecture.
- **RFC 8152** — CBOR Object Signing and Encryption (COSE).
- **ISO/IEC 19790:2025** — Security requirements for cryptographic modules.
- **NIST SP 800-90B** — Recommendation for the Entropy Sources Used for Random Bit Generation.
- **TCG D-RTM Specification** — Trusted Computing Group Dynamic Root of Trust for Measurement.
- **IEEE 1735-2023** — Recommended Practice for Encryption and Management of Electronic Design Intellectual Property.
- **PRSM-GOV-1** (pending) — PRSM Foundation Governance Charter.

Referenced standards are binding at their versions in effect on the date of a given implementation's certification submission. Subsequent standard revisions may trigger a grace period for re-certification; see §15.5.

---

## 3. Terminology

For the purposes of this document, the following definitions apply. These complement, but do not replace, definitions in `docs/glossary.md`.

**Attestation Quote.** A signed CBOR-encoded object produced by a CIS device asserting that a specified inference occurred inside the chip, under a named firmware version, with weights matching a named hash, producing a named output hash for a named input hash, under a named nonce.

**Chip Provisioning.** The manufacturer-controlled one-time process during which a CIS device is assigned its Hardware Root of Trust key material. Provisioning occurs inside a certified Hardware Security Module at a certified fab, before the chip is packaged.

**Compliance.** A chip is compliant with PRSM-CIS-1 if and only if it passes the full conformance suite at its target conformance level (§16) and holds current certification from the PRSM Foundation.

**Confidential Inference.** Inference performed on plaintext model weights where the weights are never exposed outside the chip's secure boundary.

**Frontier Publisher.** An AI model publisher whose model represents or is within 12 months of the state-of-the-art. The archetypal customer this standard is designed to satisfy.

**Hardware Root of Trust (HRoT).** The chip-internal key material, derived from physical entropy at provisioning and never leaving the die, that anchors all attestation signatures and weight-encryption keys for this chip.

**Meganode.** A PRSM network participant operating CIS-compliant hardware at production scale (multiple chips, redundant power, certified data-center integration). Meganodes are expected to be the supply backbone for frontier inference workloads on PRSM.

**Model Publisher.** The entity that owns the rights to a model's weights and authorizes its sealing to a specific set of CIS chips.

**Sealed Weights.** Model weights stored in chip memory encrypted under a key derived from the chip's HRoT. Sealed weights can only be decrypted by the chip that sealed them.

**Tamper Event.** A detected physical or environmental condition indicating an attempt to extract chip-internal state. See §9.

---

## 4. Threat model

### 4.1 Adversary classes

PRSM-CIS-1 is designed against the following adversary classes, in ascending order of capability.

**A0 — Unauthorized remote attacker.** Has network access to the inference service but no physical access and no insider relationship with the operator or manufacturer. Can submit inference requests and observe responses. Can read public blockchain state. **Defeated by standard cryptographic protocol (out of direct chip scope; addressed by PRSM-CIS-1's on-chain attestation binding).**

**A1 — Casual insider.** Has physical access to the meganode data center and operational-level access to the host systems. Cannot decapsulate chips or run invasive physical attacks. Example: a rogue employee of the meganode operator or colocated data-center staff. **Defeated by weight sealing (§8) and runtime attestation (§12).**

**A2 — Resourceful insider.** Has physical access AND capability to perform non-destructive physical probing: bus tapping, power-rail analysis, electromagnetic sensing, thermal imaging. Does not have budget or expertise for circuit-level invasive attacks. Example: a small team with $50k-$500k of test equipment. **Defeated by side-channel resistance (§10), tamper evidence (§9), and HRoT confidentiality (§7).**

**A3 — Nation-state or competitor with physical invasive capability.** Has access to: decapsulation tooling, focused-ion-beam (FIB) stations, scanning electron microscopes, laser fault injection. Budget in the $1M-$100M range per chip. Able to destroy many chips to extract one working key. Example: intelligence agency or well-funded competitor lab. **Defeated by tamper mesh + active shield (§9), confidential assembly chain-of-custody, and key-space compartmentation (§7).**

**A4 — Manufacturer compromise.** The chip manufacturer itself is compromised, coerced, or malicious. **PARTIALLY DEFEATED by multi-manufacturer certification (§15.3), independent audit of physical design (§9.3), and publicly-verifiable attestation chain (§12). Residual trust in the manufacturer for correct provisioning cannot be eliminated in first-generation CIS; mitigated by requiring ≥2 independent certified manufacturers.**

**A5 — Foundation compromise.** The PRSM Foundation itself is compromised and issues fraudulent certifications. **DEFEATED by on-chain governance (§15.4), public audit logs (§14.7), and the 2-of-3 multi-sig requirement for certification actions (§15.4).**

### 4.2 Attack chains explicitly addressed

The standard is designed to defeat, specifically, the following composed attacks:

**Attack chain AC-1 (VRAM exfiltration, repurpose with perturbation):**
1. Meganode operator runs CIS inference for model M with weights W.
2. Operator attempts to read W from chip-external memory (HBM, DRAM) during inference.
3. Operator applies perturbation δ below any fingerprint-detection threshold.
4. Operator re-registers W+δ as model M' on PRSM ProvenanceRegistry.
5. Operator charges sub-publisher-price, capturing demand.

Defeated at step 2: weights never exist plaintext outside the chip's secure boundary (§8). Fingerprinting (R8 layer 3) is a complementary economic defense but is NOT relied upon by PRSM-CIS-1 — the chip itself makes step 2 fail.

**Attack chain AC-2 (attestation forgery):**
1. Operator runs stolen weights W on uncertified hardware.
2. Operator generates a forged attestation claiming CIS-chip execution.
3. Requester verifies attestation, believes execution was confidential, pays.

Defeated at step 2: the attestation signature MUST chain to an HRoT key anchored in the on-chain AttestationRegistry (§13). A forged attestation by an uncertified party has no valid root.

**Attack chain AC-3 (side-channel weight reconstruction):**
1. Operator runs many inferences through a CIS chip they control.
2. Operator measures power, EM, or timing side channels during inference.
3. Operator reconstructs weights or partial weights from observations.

Defeated by §10 requirements: the chip MUST NOT leak weight information at a rate sufficient to reconstruct weights after N inferences, where N is specified per conformance level.

**Attack chain AC-4 (cold-boot / FIB on inactive chip):**
1. Operator physically seizes a CIS chip that is powered off but contains sealed weights.
2. Operator decapsulates and uses FIB to extract the HRoT key.
3. Operator decrypts sealed weights offline.

Defeated by §9 requirements: HRoT key material MUST be stored in a form that is destroyed by tamper-mesh breach. Decapsulation MUST trigger tamper-mesh activation with ≥99.99% probability at C2 and ≥99.999% at C3.

### 4.3 Attacks explicitly out of scope

PRSM-CIS-1 does NOT attempt to defend against:

- **Timing-channel attacks via the inference-output distribution itself.** If the model's output-token distribution leaks information about the weights, that's a model-design issue addressed by output watermarking (R8 layer 4) and differential privacy on outputs, not a chip-design issue.
- **Sybil attacks on the PRSM network layer.** Addressed by Phase 7 staking contracts.
- **Economic attacks via below-cost pricing.** A compliant operator is not prevented from running at a loss; this is an economic-layer concern, not a confidentiality concern.
- **Backdoors in the model itself.** Model trustworthiness is the publisher's concern, not the chip's. The chip attests to correct execution of whatever weights were sealed; it does not verify model behavior.
- **Firmware bugs in publisher-provided inference code.** The chip provides a runtime; the publisher's inference code runs inside it. Publisher-code bugs are the publisher's responsibility.

---

## 5. Security architecture overview

### 5.1 Security boundary

The CIS security boundary is the physical chip die together with its tamper-detection enclosure. Everything within the boundary is considered trusted by this standard; everything outside is considered potentially adversarial, including:

- The chip's package leads.
- The printed-circuit board.
- System memory (DRAM / HBM), even when cryptographically attached.
- The host CPU and operating system.
- Data-center network, power, and cooling infrastructure.

The boundary MUST be maintained by hardware only. No external software (including the host OS, hypervisor, or data-center management plane) can be trusted to preserve the boundary.

### 5.2 Functional block requirements

A compliant CIS chip MUST contain, within the security boundary:

1. **Hardware Root of Trust subsystem** (§7).
2. **Secure random bit generator** conforming to NIST SP 800-90B at an entropy rate sufficient for FIPS 203 / 204 key generation.
3. **Sealed-weight storage** with weight-encryption keys derived from HRoT (§8).
4. **Inference compute fabric** capable of the numerical operations required by the target model class. Architecture-agnostic; may be a tensor-core array, a systolic array, dataflow engine, or other microarchitecture.
5. **Attestation signing unit** with FIPS 204 (ML-DSA) or equivalent post-quantum signature capability.
6. **Secure boot + firmware measurement chain** (§11).
7. **Tamper-detection sensor network** (§9).
8. **Side-channel-hardened cryptographic accelerators** for weight-encryption, attestation signing, and any intermediate crypto operations (§10).
9. **Diagnostic / debug interface** that is permanently disabled post-provisioning (§11.4).

The chip MAY contain, as implementation choices:

- Multiple compute tiles for performance.
- Specialized blocks for attention, MoE routing, sparsity acceleration, or other model-architecture-specific optimizations.
- Optional I/O acceleration (PCIe endpoints, NVLink-equivalent).

The chip MUST NOT contain:

- Any hardware-level backdoor, vendor-reserved debug channel, or "safe mode" that bypasses the security boundary post-provisioning.
- Any manufacturer-side remote-attestation challenge-response capability that operates without the chip's normal attestation flow.

### 5.3 Lifecycle states

A CIS chip operates in exactly one of the following states at any moment:

1. **Pre-provisioning.** Chip is unpowered or in one-time-programmable (OTP) fuse-programming mode. No HRoT key exists yet.
2. **Provisioning.** HRoT is being generated from on-die entropy and sealed into secure OTP. Debug interfaces are active. Provisioning occurs only within a certified HSM at a certified fab.
3. **Provisioned.** HRoT is sealed. The chip has a unique manufacturer-signed identity certificate. Debug interfaces are permanently disabled by fuse-blowing. The chip is available for customer programming.
4. **Weight-sealing.** Publisher authorizes sealing of specific weights to this specific chip via an attested key-exchange protocol (§8.4). Sealed weights are written to the chip's internal encrypted storage.
5. **Operational.** Chip is performing inference. Sealed weights are decrypted on-the-fly during inference; never exposed outside the boundary.
6. **Tampered.** A tamper event has been detected. Chip MUST destroy its HRoT key and sealed weights. Chip MUST signal its tampered state via attestation (a tampered chip can still emit a signed "tampered" statement until its signing key is destroyed, which SHOULD happen within 100 microseconds of tamper detection).
7. **Retired.** Operator has cryptographically erased sealed weights and revoked the chip's on-chain registration. Chip may physically persist but is cryptographically inert.

Transitions between states are unidirectional except: (Operational → Tampered) is irreversible; (Tampered → Retired) is operator-initiated cleanup; (Operational → Retired) is clean shutdown.

---

## 6. Conformance levels

PRSM-CIS-1 defines three conformance levels with graduated security requirements. An implementation claims compliance at a specific level; higher levels subsume lower levels.

### 6.1 Conformance C1 (Baseline)

**Intended deployment:** Tier A / B inference (open-weights fine-tunes, mid-tier proprietary models). Analogous security posture to NVIDIA H100 CC with PRSM-specific attestation binding.

**Requirements summary:**
- HRoT in secure OTP memory (§7).
- Basic tamper detection: package intrusion switch + decapsulation-triggered fuse (§9.1).
- Side-channel resistance against A1 + A2 adversaries at first-order power/EM analysis (§10.2).
- Weight-exfiltration lower bound: ≥2²⁰ inferences before weight-reconstruction confidence exceeds 50% under documented attack (§10.3).
- Attestation signed with FIPS 204 ML-DSA at any NIST-approved parameter set.

### 6.2 Conformance C2 (Frontier-Acceptable)

**Intended deployment:** Frontier model publishers comfortable with 12-18-months-old SOTA weights. The assurance level at which most Phase 3 meganodes should operate by 2030-2031.

**Requirements above C1:**
- Active tamper mesh with ≥8 independent sensor channels (§9.2).
- Tamper-detection response time ≤10ms (HRoT key destroyed within this window).
- Side-channel resistance against A2 adversary at second-order power + EM + cache analysis (§10.2).
- Weight-exfiltration lower bound: ≥2³⁰ inferences (§10.3).
- Physical penetration test from ≥2 independent certified labs (§14.4).
- Formally-verified attestation-signing unit (§11.3).
- Multi-manufacturer availability — the same implementation MUST be buildable by ≥2 independent fabs on the Foundation's certified-fab list.

### 6.3 Conformance C3 (Current-SOTA-Acceptable)

**Intended deployment:** Current-SOTA frontier models. The assurance level at which Foundation-certified meganodes can demand the very-top-of-market publisher engagements.

**Requirements above C2:**
- All A3-adversary (invasive-physical) defenses (§9.3).
- Weight-exfiltration lower bound: ≥2⁴⁰ inferences under C2 + A3 attacks combined (§10.3).
- Side-channel resistance against any combination of power + EM + timing + acoustic + thermal channels at third-order analysis.
- Destructive tamper-mesh testing by ≥3 independent certified labs (§14.4).
- Formal hardware-design verification of the security-boundary-enforcing subsystems.
- Publisher-controllable weight-re-keying: the chip MUST support periodic re-encryption of sealed weights under a new per-session key material on publisher command, with attestation proving the old key was destroyed (§8.5).
- Published design review: complete logical-level design disclosure to ≥5 independent security reviewers under NDA, with review findings published in redacted form.

### 6.4 Level declaration and migration

An implementation MAY target multiple conformance levels simultaneously (e.g., a single chip may be certified at C1 for general use and C2 for specific publisher engagements based on operational controls). The target level MUST be declared in the attestation quote of every inference (§12.2), allowing the verifier to reject if the publisher's policy requires higher than claimed.

Implementations certified at C1 MAY be retrofitted to C2 only if the physical design supports the required tamper-mesh sensor density; otherwise the implementation is forever-C1 and a new chip revision is required for higher levels.

---

## 7. Hardware Root of Trust

### 7.1 HRoT generation

The HRoT key material MUST be generated exactly once, during chip provisioning, from on-die physical entropy sources that pass NIST SP 800-90B at a minimum of 8 bits of entropy per 10 bits of raw output (80% efficiency). Acceptable entropy sources include:

- Ring-oscillator jitter (recommended primary source).
- Thermal-noise-sampled comparators.
- Metastability-resolution sampling.
- On-die quantum-level effects (future; not required).

The HRoT MUST consist of at least 256 bits of effective entropy at C1, 384 bits at C2, 512 bits at C3.

The HRoT MUST be written only to chip-internal physically-unclonable-function (PUF) storage or to fused secure OTP. It MUST NOT be written to any memory that is accessible by any interface outside the security boundary, even transiently.

### 7.2 HRoT key hierarchy

From the HRoT, the following keys MUST be derivable via deterministic KDF (HKDF-SHA-384 or equivalent):

- **K_attest** — long-lived attestation-signing private key. ML-DSA at NIST-3 or higher. MUST NOT ever leave the chip.
- **K_seal** — weight-encryption key for weights sealed to this chip. AES-256-GCM or higher. MUST NOT ever leave the chip.
- **K_session(i)** — per-inference-session ephemeral keys, derived from K_seal and a publisher-provided nonce.
- **K_manufacturer_cert** — the manufacturer's signing key that signed this chip's identity certificate at provisioning. Public portion available for verification; private portion held by the manufacturer, not the chip.

K_attest and K_seal MUST be mathematically independent — compromise of one MUST NOT reveal the other even under quantum attacks. The KDF context strings used to derive them are public but MUST be domain-separated.

### 7.3 HRoT destruction

HRoT key material MUST be destroyed (zeroized and physically scrubbed from fuse / PUF storage to the extent physically possible) upon:

1. Tamper detection (§9).
2. Operator-initiated retirement (operator must authenticate to the chip via a provisioned retirement credential).
3. End-of-life per the manufacturer's published lifespan (e.g., fuse-degradation threshold reached).
4. Certification revocation reaching this specific chip's serial number (§15.5).

Destruction MUST be completed within 100 microseconds of the trigger event for C2+, or within 1ms for C1. Successful destruction MUST be verifiable via a subsequent failed-attestation: a chip whose HRoT has been destroyed cannot sign any valid attestation, and the chip SHOULD (when possible) emit a final signed "destroyed" statement as its last act of signing capability.

---

## 8. Weight sealing

### 8.1 Sealing protocol overview

Weight sealing is the process by which a model publisher authorizes specific weights to be stored and executed on a specific CIS chip. Weights never exist plaintext outside the publisher's control systems and the target chip's security boundary.

The protocol is:

1. Publisher obtains the target chip's attestation of its current state (firmware version, HRoT-derived public key fingerprint). Attestation is signed by K_attest and verifiable against the chip's manufacturer certificate registered in the on-chain AttestationRegistry.

2. Publisher generates a one-time session key `K_transport` and encrypts the model weights under `K_transport` using AES-256-GCM or equivalent.

3. Publisher encrypts `K_transport` under the chip's attestation-derived public key (a KEM operation using FIPS 203 ML-KEM-768 or higher) AND an auxiliary per-session public key derived from K_seal. Binding to K_seal ensures the wrapped `K_transport` can ONLY be decapsulated by this specific chip.

4. Publisher transmits `{encrypted_weights, encrypted_K_transport, publisher_signature, publisher_policy}` to the chip.

5. Chip verifies publisher_signature against the publisher's registered identity, decapsulates `K_transport`, decrypts the weights into chip-internal secure storage encrypted under `K_seal`, verifies the weight hash matches publisher_policy, and emits an attestation of successful sealing.

### 8.2 Sealed-weight storage requirements

Sealed weights MUST be stored encrypted at rest under `K_seal`. The encryption MUST use an authenticated-encryption scheme (AES-256-GCM, ChaCha20-Poly1305, or equivalent AEAD). Storage media MAY be:

- On-die SRAM (highest performance, lowest capacity).
- On-die eDRAM or FeRAM (medium performance, medium capacity).
- Tightly-coupled in-package HBM with chip-side AEAD transparent on read (highest capacity, acceptable performance given AEAD overhead).

If sealed weights are stored in HBM or any memory outside the die, the HBM MUST be in the same tamper-detection zone as the die; removing the HBM MUST trigger the same tamper event as decapsulating the die.

### 8.3 Memory encryption requirements

During inference, weights are decrypted on-the-fly into the compute fabric. Decryption MUST occur on-die. At no time may plaintext weights be driven onto any chip-external interface — this includes debug ports, DFT scan chains, mesh bus lines that cross the tamper boundary, or diagnostic registers.

The chip MUST implement memory-encryption integrity protection: any attempt to modify encrypted weights in off-die HBM MUST be detected on read (by authentication tag verification) and cause immediate inference halt plus tamper event.

### 8.4 Publisher authorization

Each sealing operation MUST be authorized by the publisher via a signed sealing certificate bound to:

- Publisher's long-lived identity (registered in the on-chain AttestationRegistry).
- Target chip's serial number.
- Weight hash.
- Policy: allowed inference rate, max inference count, expiration time, allowed customer set (see §8.5).
- Publisher signature under FIPS 204 ML-DSA.

The chip MUST enforce the policy. Exceeding inference count or expiration MUST cause the chip to refuse further inference with the sealed weights. Re-sealing under new policy is permitted; it requires a fresh sealing certificate.

### 8.5 Re-keying and revocation

At C3, the publisher MUST be able to command re-keying: the chip re-derives `K_seal'` from a new per-session component, re-encrypts all sealed weights, destroys the old ciphertexts, and emits attestation of the completed re-key. This limits cryptographic exposure in the event of a suspected key compromise.

A publisher MUST be able to revoke a sealing by posting a revocation record to the AttestationRegistry. After a revocation is finalized (typically within 1 hour on Base), any chip attempting inference with the revoked weights will fail publisher-signature verification on the per-inference policy check and the inference will not execute. The chip MAY destroy the sealed weights proactively upon observing the revocation (recommended; required at C3).

---

## 9. Tamper evidence

### 9.1 Tamper requirements at C1

The chip MUST detect and respond to:

- **Package decapsulation** — mechanical or chemical opening of the chip package. Detection via breaking of a wire-bond loop, a capacitance mesh embedded in the mold compound, or a cured-epoxy conductive mesh.
- **Voltage-rail manipulation** — supply voltage outside specified operating range. Detection via on-die voltage monitor.
- **Clock-frequency manipulation** — clock signal outside specified operating range. Detection via on-die PLL lock-loss + frequency comparator.
- **Temperature excursion** — die temperature outside specified operating range. Detection via on-die thermal sensor.
- **Reset-line manipulation** — attempts to hold the chip in reset while accessing other interfaces.

Detection of any tamper event at C1 MUST trigger the C1 tamper response: K_attest and K_seal are destroyed; the chip halts all operations; it emits (if still capable) a final signed tamper statement.

### 9.2 Tamper requirements at C2

In addition to all C1 requirements:

- **Active mesh shield** covering 100% of the die top surface. Mesh MUST have ≥8 independent conductive paths each continuously monitored for continuity and impedance changes. Mesh MUST be constructed such that ion-beam or laser-cutting a single path triggers detection.
- **Backside-thinning detection** via on-die optical sensors or by exploiting substrate-side irregularities that disrupt die-attach resonance.
- **Clock-glitching detection** beyond simple frequency monitoring: detection of subcycle perturbations via redundant clock-domain crossings.
- **Laser-fault-injection detection** via on-die photo-sensors positioned to detect the characteristic emissions of LFI lasers.
- **Electromagnetic-fault-injection detection** via on-die magnetic-field sensors.

Tamper response time MUST be ≤10ms (from event to HRoT destruction).

### 9.3 Tamper requirements at C3

In addition to all C2 requirements:

- **Tamper-resistant package** rated per FIPS 140-3 Level 4.
- **Environmental randomization of sensor thresholds** — tamper sensors do not all trigger at the same fixed threshold, so a calibrated attack cannot probe each one individually.
- **Delayed-destruction capability** — the chip MAY delay the visible response to a tamper event by a publisher-configured randomized interval (up to 10 seconds), during which it executes a set of honeypot-like decoys while actually having destroyed the real keys. Defeats "immediate cessation" as a signal to the attacker.
- **Inside-die tamper-detection distribution** — tamper-response logic replicated across ≥3 independent die regions, each powered from separate power rails, each capable of triggering destruction independently.
- **Destructive validation** — during certification testing (§14.4), ≥100 sample chips MUST be subjected to destructive physical attacks and demonstrate ≥99.999% tamper-detection success rate.

---

## 10. Side-channel resistance

### 10.1 Threat model for side channels

Adversaries with physical proximity to the chip (A2 and above) can measure:

- **Power consumption** — time-series via shunt resistors or current probes.
- **Electromagnetic emanations** — near-field or far-field probes picking up switching activity.
- **Cache timing** — if the chip has shared caches, timing-based eviction-set analysis.
- **Bus timing** — timing of memory-bus transactions.
- **Acoustic emissions** — ultrasonic signatures of CPU activity.
- **Thermal patterns** — infrared imaging of hot spots corresponding to active compute units.

The standard's threat: these channels leak information about the secret weights (or intermediate activations carrying weight information) at a rate measurable in bits-per-inference. The goal is to reduce this rate to the point that reconstructing useful weights requires more inferences than an attacker is likely to obtain.

### 10.2 Mitigation requirements by level

**At C1:**
- Constant-time cryptographic operations for attestation-signing and weight-decryption. First-order power/EM analysis MUST NOT reveal key material in fewer than 2²⁰ operations.
- Randomized inference scheduling to break fixed-pattern observations: inference ordering across multiple requests randomized within publisher-specified latency budgets.
- Power-rail filtering at package level to reduce the bandwidth of external power-trace measurements.

**At C2:**
- All C1 requirements.
- Second-order side-channel resistance: masking (boolean or arithmetic) for the critical cryptographic operations, with mask refreshes per operation. Second-order attacks MUST NOT reveal key material in fewer than 2³⁰ operations.
- Cache-partitioning: if shared caches are used, inference-relevant cache lines MUST be isolated in dedicated partitions flushed between inferences from different sessions.
- EM shielding: die-level Faraday enclosure as part of the active mesh (§9.2).

**At C3:**
- All C2 requirements.
- Third-order side-channel resistance: masking schemes with at least 3 shares per secret, with threshold implementations of all critical operations. Third-order attacks MUST NOT reveal key material in fewer than 2⁴⁰ operations.
- No shared caches across security domains; per-session isolated cache hierarchies.
- Thermal obfuscation: auxiliary heating elements controlled to maintain uniform die thermal signature regardless of compute activity.
- Acoustic dampening in package.

### 10.3 Weight-exfiltration lower bound

The primary quantitative security metric: the number of inferences `N` an A2+ adversary must observe, under the composed side-channel attacks of §10.2, before their weight-reconstruction confidence (measured as, e.g., top-1 weight prediction accuracy) exceeds a 50% threshold.

| Level | Minimum N |
|---|---|
| C1 | 2²⁰ (~10⁶ inferences) |
| C2 | 2³⁰ (~10⁹ inferences) |
| C3 | 2⁴⁰ (~10¹² inferences) |

Certification labs (§14) MUST measure this metric directly via adversarial evaluation with state-of-the-art side-channel-analysis toolchains against each chip revision seeking certification.

---

## 11. Firmware, secure boot, and debug

### 11.1 Firmware

The chip's firmware (the code running on the secure-boot-measured attestation unit) MUST be:

- Open-source and publicly reviewable. Binary blobs without source are not permitted in the firmware image.
- Deterministically reproducible from source — the same source produces the same binary byte-for-byte on a specified toolchain.
- Signed by the manufacturer. Firmware update is allowed but requires a signed update image. The chip maintains a monotonic firmware-version counter and will not roll back.
- Measured into a hash chain anchored at the HRoT. Current firmware hash is included in every attestation quote (§12.2).

The specific firmware functionality is implementation-defined, but at minimum it MUST:

- Handle attestation quote generation.
- Manage weight-sealing protocol (§8).
- Enforce publisher policy.
- Handle tamper-response protocols (§9).
- Schedule inference operations on the compute fabric.

The firmware MUST NOT:

- Accept any code submitted by an inference-request caller for execution in the secure domain.
- Export debug information that includes weight material or key material.
- Implement any functionality whose authorization depends on a check against a remote Foundation or manufacturer server (the chip MUST function offline after provisioning).

### 11.2 Secure boot

At each power-on, the chip MUST execute a secure boot sequence:

1. Power-on-reset completes; HRoT subsystem initializes from fuse/PUF state.
2. Boot ROM (mask-programmed, immutable) begins execution.
3. Boot ROM measures firmware image (SHA-384 or equivalent). Measurement is extended into a boot-time hash chain.
4. Boot ROM verifies firmware signature against the manufacturer's firmware-signing key (whose public part is in another OTP fuse set).
5. On verification success, Boot ROM transfers control to firmware; the firmware-boot measurement is retained as part of attestation state.
6. On verification failure, chip enters a halt state and emits a signed boot-failure attestation (using its pre-firmware signing path) before destruction of ephemeral state.

### 11.3 Formal verification requirements

At C2+, the attestation-signing unit and the secure-boot code MUST be formally verified against a specification of their intended behavior. The specification MUST be public; the verification proof MUST be reproducible by independent auditors. Tools acceptable for verification include Coq, Isabelle/HOL, TLA+, and equivalent formal-methods systems.

### 11.4 Debug interface lifecycle

Pre-provisioned chips MAY have debug interfaces (JTAG, SWD, scan chain). These are used during manufacturing test and provisioning. Upon completion of provisioning:

- Debug interface fuses MUST be blown permanently, disabling all debug access.
- The chip MUST verify at every boot that debug-disable fuses are intact; tampering with the fuse state MUST trigger a tamper event.

No post-provisioning debug channel is permitted, including "authenticated debug" schemes that require manufacturer cooperation. The chip is either sealed or it is not; there is no middle state that bypasses the security boundary.

---

## 12. Attestation protocol

### 12.1 Attestation quote format

An Attestation Quote is a CBOR (COSE_Sign1) object with the following claims:

```
AttestationQuote = {
    "iss": manufacturer_chip_id,            # serial + manufacturer ID
    "firmware_hash": hex-encoded SHA-384,   # current firmware measurement
    "firmware_version": uint,                # monotonic firmware counter
    "sealed_weight_hash": hex-encoded SHA-384, # hash of currently-sealed weights
    "publisher_id": hex-encoded pubkey_hash,   # issuing publisher
    "policy_hash": hex-encoded SHA-384,      # publisher policy at sealing
    "input_hash": hex-encoded SHA-384,       # hash of the inference input
    "output_hash": hex-encoded SHA-384,      # hash of the inference output
    "nonce": hex-encoded bytes,              # caller-provided randomness
    "executed_at_unix": uint,                # wall-clock from chip's secure clock
    "conformance_level": "C1" | "C2" | "C3",
    "chip_identity_cert": byte_string,       # manufacturer-signed chip cert
}

signature = ML-DSA-Sign(K_attest, CBOR-encoded AttestationQuote claims)
```

The quote is returned as the COSE_Sign1 envelope; verifiers extract claims and verify signature against the attestation public key derived from HRoT and registered in the AttestationRegistry.

### 12.2 Verification procedure

A verifier (typically a PRSM requester or a marketplace orchestrator, see PRSM Phase 3 design) MUST:

1. Parse the CBOR attestation envelope.
2. Extract `chip_identity_cert` and verify it against the manufacturer's registered certificate in the AttestationRegistry. Reject on revocation or expiration.
3. Derive the attestation verification key from the chip identity certificate.
4. Verify the COSE_Sign1 signature using ML-DSA.
5. Check `firmware_hash` against the publisher's policy (publisher may allow only specific firmware versions).
6. Check `sealed_weight_hash` against the expected model weight hash for this inference.
7. Check `input_hash` matches the actual input sent.
8. Check `output_hash` matches the actual output received.
9. Check `nonce` matches the requester's challenge.
10. Check `executed_at_unix` is within an acceptable time window.
11. Check `conformance_level` meets or exceeds publisher/requester policy.

If any check fails, the inference receipt MUST NOT be accepted and the accompanying FTNS escrow MUST be refunded to the requester (integrating with Phase 2's escrow-state-machine semantics).

### 12.3 Replay and freshness

The `nonce` claim MUST be a requester-provided fresh random value. Chips MUST reject attestation-quote requests that do not carry a fresh nonce (the same nonce used in a prior quote within the chip's short-term nonce window MUST be rejected; implementations MAY use Bloom filters or cryptographic-hash-based deduplication).

The `executed_at_unix` claim is produced from the chip's secure time source; this source MUST be maintained with drift ≤1 second per day and MUST resist manipulation from outside the security boundary.

---

## 13. On-chain attestation registry

### 13.1 AttestationRegistry contract

The PRSM Foundation MUST deploy and operate an AttestationRegistry smart contract on Base mainnet (or a successor chain the governance process approves). The contract provides:

- **Manufacturer registration.** A certified chip manufacturer MUST post a long-lived manufacturer certificate with their public key, official legal entity identifier, and bond deposit. The registry MAY require manufacturers to post a bond slashable upon certification revocation.
- **Chip certificate registration.** Upon provisioning, manufacturers post chip identity certificates — one per chip — as Merkle-tree batch commitments to control gas costs. Individual chip certificates are revealed on demand via Merkle proof.
- **Publisher registration.** Model publishers register their long-lived identity keys.
- **Revocation.** Both manufacturers and publishers may post revocations; chip-level revocation (individual chips) MUST be supported at Merkle-proof level.
- **Audit trail.** Every registration and revocation emits an on-chain event for public consumption.

### 13.2 Contract architecture

```solidity
interface AttestationRegistry {
    function registerManufacturer(
        bytes32 publicKeyHash,
        string calldata legalEntityId,
        uint256 bondAmount
    ) external;

    function postChipCertBatch(
        bytes32 merkleRoot,
        uint256 chipCount
    ) external;

    function revokeChip(
        bytes32 chipCertHash,
        bytes32[] calldata merkleProof,
        string calldata revocationReason
    ) external;

    function registerPublisher(
        bytes32 publicKeyHash,
        string calldata metadata
    ) external;

    function isManufacturerActive(bytes32 publicKeyHash) external view returns (bool);
    function isChipRevoked(bytes32 chipCertHash) external view returns (bool);
    function isPublisherActive(bytes32 publicKeyHash) external view returns (bool);
}
```

### 13.3 Binding to PRSM provenance

The AttestationRegistry is separate from the existing ProvenanceRegistry (Phase 1.3) to keep contract responsibilities narrow. Models that are attested via PRSM-CIS-1 record references to both:

- `ProvenanceRegistry.content_cid` — where the model's opaque encrypted weights are stored.
- `AttestationRegistry.publisher_id` — who authorized sealing.

A PRSM requester dispatching inference via Phase 3's MarketplaceOrchestrator and requiring CIS-backed inference will consult both registries before accepting a receipt.

---

## 14. Compliance testing

### 14.1 Certification labs

Testing MUST be performed by labs certified by the PRSM Foundation via a separate laboratory accreditation process. Minimum qualifications:

- Demonstrated experience with FIPS 140-3 or Common Criteria EAL 5+ testing.
- Independence: the lab MUST NOT be owned or controlled by any chip manufacturer seeking certification.
- Geographic diversity: the Foundation SHALL maintain a list of certified labs across ≥3 jurisdictions to limit regulatory-capture risk.

### 14.2 Test suite categories

The compliance test suite covers:

1. **Functional tests.** Attestation correctness, weight-sealing correctness, policy enforcement — run as software-level tests against the chip's external interfaces.
2. **Cryptographic tests.** FIPS test vectors for all implemented algorithms, KDF correctness, random-number-generation quality.
3. **Side-channel tests** (§14.3).
4. **Tamper tests** (§14.4).
5. **Longevity tests.** Stress testing across temperature, voltage, humidity to verify that tamper sensors remain calibrated across environmental ranges.
6. **Firmware verification.** Secure-boot chain audit, firmware-update rollback-resistance tests.

### 14.3 Side-channel test methodology

Side-channel testing MUST be conducted according to the Test Vector Leakage Assessment (TVLA) methodology or equivalent. For each conformance level, the lab MUST:

1. Instrument the chip at the specified measurement bandwidth (power: ≥GSPS; EM: ≥GHz; timing: cycle-level).
2. Execute N inferences with varying inputs and fixed weights; separately with fixed inputs and varying weights.
3. Apply mutual-information analysis and T-test-based leakage detection.
4. Report the number of traces required for any first/second/third-order attack to distinguish weights from random at statistical-significance thresholds.
5. Verify the required `N` threshold per §10.3 is met.

### 14.4 Tamper test methodology

For each conformance level, the lab MUST perform the following attacks on sample chips:

**At C1:** package decapsulation attempts × 10 samples. Pass criterion: ≥95% tamper-event detection.

**At C2:** all C1 + laser-fault injection × 20 samples; focused-ion-beam probing × 10 samples; backside thinning × 10 samples. Pass criterion: ≥99% detection.

**At C3:** all C2, scaled to 100+ samples, plus specific tests for:
- Side-channel-guided tamper (combining power analysis with physical probing).
- Slow-rate tamper (tamper attempts spread over hours/days to bypass simple "threshold" sensors).
- Environmental-variation tamper (attempting to move sensor thresholds via temperature or voltage drift).

Pass criterion: ≥99.999% detection.

### 14.5 Red-team exercise

For C3 certification, the Foundation MUST commission a dedicated red-team engagement with at least two independent academic security research groups. Red teams are given non-destructive pre-certification chips and up to 6 months to find exfiltration paths. All findings are documented; the implementation MUST address any exfiltration achieved at any level of resource below the C3 adversary model before certification is granted.

### 14.6 Publication of test results

Summary test reports MUST be published by the Foundation for every certified chip. Full reports MAY be redacted of information that would aid attackers without serving the standard's transparency goals; redactions are reviewed by the Foundation's independent audit committee before publication.

### 14.7 Continuous monitoring

After certification, the Foundation MAY:

- Commission periodic re-testing (annually at C2+, bi-annually at C1) to confirm no regressions.
- Monitor public academic publications for new side-channel or physical attacks against the certified design. A qualifying new attack triggers a mandatory retest or recertification.

---

## 15. Certification and governance

### 15.1 Certification process

An implementer seeking certification:

1. Submits a design package including: architectural specification, implementation description, cryptographic parameter choices, target conformance level, bill-of-materials.
2. Submits sample chips (≥10 at C1, ≥50 at C2, ≥100 at C3) and firmware source to the chosen certified lab.
3. Lab conducts testing per §14.
4. Lab submits test report to Foundation certification committee.
5. Committee reviews and votes (≥2/3 supermajority required). Committee members publicly disclose any commercial relationships with the implementer; members with conflicts recuse.
6. Upon approval, implementation is added to the Foundation's registry of certified designs. Manufacturer-signing key is registered in the AttestationRegistry.
7. Certification is valid for the period specified by the conformance level (C1: 3 years; C2: 2 years; C3: 1 year with mandatory annual re-review).

### 15.2 Certification cost

The Foundation sets certification fees on a cost-recovery basis. Fees MUST cover:

- Laboratory testing costs (largest component, may be several hundred thousand dollars at C3).
- Committee review time.
- Registry operation and on-chain gas costs.
- Audit committee oversight.

Fees MUST NOT subsidize the Foundation's general operating budget — any surplus is returned to certification fee-payers pro rata or refunded.

### 15.3 Multi-manufacturer requirement at C2+

C2 and C3 certifications REQUIRE that the certified design be independently implementable by ≥2 manufacturers. The Foundation MAY grant provisional single-manufacturer C2 certification with a mandatory 24-month transition to a second manufacturer; failure to qualify a second manufacturer within the transition period triggers downgrade to C1.

The rationale: single-manufacturer dependency is a structural risk on the order of the concerns this standard is designed to defeat. A chip that only one fab can make is effectively hostage to that fab's continued willingness and ability.

### 15.4 Foundation governance of certification decisions

The certification committee:

- Has exactly 7 voting members.
- Members are elected by Foundation governance process (see PRSM-GOV-1 once ratified).
- Member terms are 3 years, staggered, with a 2-term limit.
- At least 2 members MUST be from PRSM's Phase 7+ staked-participant class (to preserve community voice).
- At least 2 members MUST be from academic security research institutions.
- At least 1 member MUST be a qualified cryptographer (not dual-hatted as industry).
- No more than 1 member may be affiliated with any single first-implementer entity (including Prismatica).

Committee decisions require a supermajority AND a 2-of-3 multi-sig execution of the registration transaction on Base. The 2-of-3 signers are committee-elected officers; keys are held on hardware wallets with published public identities.

All committee votes are public. All committee correspondence is published with a 90-day delay (to allow routine operations to complete without leakage but to preserve transparency in the long run).

### 15.5 Revocation

Certification revocation MAY be initiated by:

- Discovery of a post-certification vulnerability not addressable by firmware update.
- Manufacturer malfeasance or corporate governance change that invalidates the trust basis.
- Failure to cooperate with a Foundation-initiated re-test.

Revocation is posted to the AttestationRegistry; active chips retain functionality until their currently-sealed weights expire, but no new sealings under the revoked certification are valid. Publishers with ongoing engagements receive notification and a migration window.

---

## 16. Open issues and research dependencies

The following items are identified as requiring further research before this standard can be finalized. They are ordered by blocking priority.

### 16.1 Side-channel thresholds — concrete values

The specific N thresholds in §10.3 (2²⁰ / 2³⁰ / 2⁴⁰) are placeholders based on first-principles security estimates. Final values MUST be set in consultation with academic side-channel experts after review of the latest attack literature. Target for resolution: working-group draft v0.2.

### 16.2 Post-quantum transition

This standard specifies ML-DSA (FIPS 204) and ML-KEM (FIPS 203) as the post-quantum primitives. These are first-generation post-quantum standards and have an expected deprecation horizon of 10-20 years. The standard's versioning process (§1.4) MUST accommodate timely migration to successor algorithms. A migration-planning annex will be added in v0.2 or v0.3.

### 16.3 Hardware-entropy source characterization

The 80%-efficiency NIST SP 800-90B entropy requirement (§7.1) is a floor, but the effective entropy rate of on-die sources varies significantly with process node and design. Target for resolution: characterization study across ≥3 process nodes before v1.0.

### 16.4 Formally-verified attestation-signing unit reference implementation

A Foundation-sponsored reference implementation of the attestation-signing unit — formally verified and open-source — would substantially reduce certification cost for C2 implementations. Target: commission after working-group ratifies v0.2.

### 16.5 Performance overhead characterization

Side-channel mitigations, especially at C3 (third-order masking, thermal obfuscation, etc.), impose performance overhead. For operators (meganode builders) to make rational build-vs-buy decisions, the Foundation SHOULD publish characterization data on the overhead of each mitigation. Target: post v0.2 before first-implementer tape-out commitments.

### 16.6 Compatibility with non-PRSM confidential compute

This standard does not attempt to be interoperable with NVIDIA Confidential Compute, AMD SEV-SNP, Intel TDX, or other existing confidential-computing standards. Bridge formats may be specified in a companion PRSM-CIS-INT document if demand emerges. Target: opportunistic; not blocking.

### 16.7 Supply-chain integrity

This standard addresses chip-level tamper defense but does not mandate provenance-chain protection from wafer production through shipment. A companion Supply-Chain Integrity annex MAY specify: trusted-foundry arrangements, in-transit tamper-evident packaging, wafer-level per-chip cryptographic identity bound at lithography time. Target: v0.3.

### 16.8 Firmware update authority

The current spec (§11.1) gives manufacturers authority to sign firmware updates. This creates a potential choke point: a compelled or compromised manufacturer could push malicious firmware. Mitigations under consideration: multi-signer firmware updates (manufacturer + Foundation + publisher); formal-verification requirement on firmware changes; rate-limited update cadence. Target for resolution: v0.2.

### 16.9 Chip retirement and e-waste

Retired chips contain (until their HRoT is destroyed) material that could leak to an attacker. A mandatory cryptographic erasure-plus-attestation protocol for retirement is described in §5.3 but physical destruction recommendations for end-of-life chips are still TBD. Environmental considerations intersect with security considerations here. Target: v0.3.

### 16.10 Economic model for compliance costs

A manufacturer pursuing C3 certification will incur costs on the order of $50M-$200M (design, fab NRE, certification, maintenance). These costs must be recoverable in per-chip pricing for the standard to be commercially viable. The Foundation SHOULD publish, alongside v1.0, an economic-model white paper clarifying expected per-chip cost, expected operating volumes, and the FTNS-revenue economics that make this recoverable for first implementers. Target: v1.0.

---

## 17. Contributors and acknowledgements

To be populated as working group members are named.

## 18. Change log

**v0.1 (2026-04-21):** Initial draft for community review. Authored by Claude Opus 4.7 on behalf of the PRSM Foundation (editor pending working-group convocation). This version is for discussion only and carries no certification authority.

---

**End of PRSM-CIS-1 v0.1 Draft.**
