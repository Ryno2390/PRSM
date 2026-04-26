---
name: 🔌 Node Operator Issue
about: Report problems running a PRSM node (bootstrap, NAT, performance, settlement)
title: '[Node Op] '
labels: ['node-operator', 'needs triage']
assignees: ''
---

## 🔌 Node Operator Issue

If your issue is on the SDK / MCP / consumer-of-PRSM side, please use the **Bug Report** template instead.

This template is for problems running and maintaining a PRSM node — peer discovery, NAT traversal, FTNS settlement, hardware tier classification, storage proofs, or anything else specific to the node-operator path.

### 📋 Summary

<!-- One sentence: what is your node doing or failing to do? -->

### 🔍 Environment

**PRSM version:**
<!-- Output of `prsm --version` or `pip show prsm-network | grep Version` -->

**Operating system:**
<!-- macOS 14.5 / Ubuntu 24.04 / Windows 11 / etc. -->

**Hardware tier (from `prsm node benchmark`):**
- [ ] T1 (mobile / IoT)
- [ ] T2 (consumer GPU / Apple Silicon)
- [ ] T3 (high-end desktop / cloud GPU)
- [ ] T4 (datacenter)

**TFLOPS (FP32):** <!-- from prsm node benchmark output -->

**TEE backend (from `prsm node benchmark`):**
- [ ] Software (no hardware TEE)
- [ ] SGX
- [ ] TDX
- [ ] SEV-SNP
- [ ] TrustZone (ARM)
- [ ] Apple Secure Enclave

**Network:**
- [ ] Residential (consumer ISP, NAT'd)
- [ ] Cloud (specify provider): _______
- [ ] VPS / dedicated server

**Bootstrap node configuration:**
- [ ] Default (`bootstrap1.prsm-network.com:8765`)
- [ ] Custom: _______

### 🐛 Issue Category

- [ ] Cannot connect to bootstrap
- [ ] Peer discovery / NAT traversal failure
- [ ] Hardware benchmark wrong / missing capability
- [ ] FTNS settlement not arriving
- [ ] Storage proof challenges failing
- [ ] Compute job dispatch errors
- [ ] Marketplace listing not appearing
- [ ] On-chain registration / role grant errors
- [ ] Sepolia testnet specific
- [ ] Other: _______

### 🔄 Steps to Reproduce

1. <!-- First step -->
2. <!-- Second step -->
3. <!-- ... -->

### 📊 Logs

<!--
Paste relevant log output. Redact wallet addresses ONLY if they're tied to
significant FTNS balances; node-identity addresses are usually fine to share.
DO NOT paste your seed phrase or hardware-wallet recovery codes EVER.
-->

```
[paste node logs here]
```

### 🤔 What I've Tried

- [ ] Restarted the node
- [ ] Re-ran `prsm node benchmark`
- [ ] Checked `prsm node status` for ring health
- [ ] Verified network connectivity to bootstrap
- [ ] Checked Discord `#node-operators` for similar reports
- [ ] Other: _______

### 💡 Expected vs. Actual Behavior

**Expected:**
<!-- What did you think would happen? -->

**Actual:**
<!-- What actually happened? -->

### 🎯 Severity (your assessment)

- [ ] Blocking — node is unusable
- [ ] High — major function broken but workaround exists
- [ ] Medium — degraded experience
- [ ] Low — minor issue / polish

### 🤝 Are you willing to help debug?

- [ ] Yes — I can run additional diagnostic commands
- [ ] Yes — I can grant temporary access to a maintainer for live debugging
- [ ] Limited — I can respond to specific questions
- [ ] No — please diagnose from this report alone

---

**Tip:** Many node-operator issues are first answered in Discord `#node-operators` ([https://discord.gg/R8dhCBCUp3](https://discord.gg/R8dhCBCUp3)) before being formalized as GitHub issues. If your problem is time-sensitive, ping there first.
