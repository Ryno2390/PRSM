# PRSM Pause-Transaction Templates

**Purpose:** Pre-built Safe Transaction templates for emergency contract pause / unpause / upgrade operations during P0 incidents. Closes Exploit Response Playbook §11 readiness checklist item.

Per Exploit Playbook §6 pause authorization procedure, the on-call engineer broadcasts the pause Tx to the council Signal group at T+2-3 minutes. With these templates, that step is **fill-in-the-amount-and-broadcast**, not build-from-scratch under pressure.

---

## What's pausable, what's not

PRSM has 13 deployed/deployable smart contracts. Only **4 of 13** have administrative pause or upgrade levers:

| Contract | Pause | Upgrade | Notes |
|---|---|---|---|
| `FTNSTokenSimple.sol` | `pause()` (PAUSER_ROLE) | UUPS via `_authorizeUpgrade` | Token-level pause stops all transfers |
| `FTNSBridge.sol` | `pause()` (BridgeAdmin) | UUPS | Bridge-only pause; doesn't affect Base-side |
| `EmissionController.sol` | `pauseMinting()` (Owner) | None | Stops new emissions; vested grants unaffected |
| `BridgeSecurity.sol` | None | UUPS | Upgrade only (signature verification updates) |

The other **9 contracts are intentionally immutable** — `RoyaltyDistributor`, `EscrowPool`, `BatchSettlementRegistry`, `ProvenanceRegistry`, `CompensationDistributor`, `Ed25519Verifier`, `KeyDistribution`, `StakeBond`, `StorageSlashing`. Pause was deliberately omitted from these because it would qualify the Tokenomics §10 invariants — "creator royalties enforced on-chain" reads stronger as an unconditional guarantee than as "enforced unless we pause."

For incidents affecting an immutable contract, see [`NON_PAUSE_RESPONSE.md`](./NON_PAUSE_RESPONSE.md) for the alternate response menu (public communication, oracle-pull, vendor coordination, etc.).

---

## Template index

| Template | Contract | Use when |
|---|---|---|
| [`safe-tx/ftns-token-pause.json`](./safe-tx/ftns-token-pause.json) | `FTNSTokenSimple` | Active drain on FTNS supply; unauthorized mint detected; severe market manipulation |
| [`safe-tx/ftns-token-unpause.json`](./safe-tx/ftns-token-unpause.json) | `FTNSTokenSimple` | After patched contract deployed + auditor sign-off + 24h Sepolia bake-in |
| [`safe-tx/ftns-token-upgrade.json`](./safe-tx/ftns-token-upgrade.json) | `FTNSTokenSimple` | UUPS upgrade to deploy patch (post-audit) |
| [`safe-tx/ftns-bridge-pause.json`](./safe-tx/ftns-bridge-pause.json) | `FTNSBridge` | Bridge exploit; double-spend across L1/L2; signature-verification failure |
| [`safe-tx/ftns-bridge-unpause.json`](./safe-tx/ftns-bridge-unpause.json) | `FTNSBridge` | After bridge patch + cross-chain reconciliation |
| [`safe-tx/ftns-bridge-upgrade.json`](./safe-tx/ftns-bridge-upgrade.json) | `FTNSBridge` | UUPS upgrade |
| [`safe-tx/emission-controller-pause.json`](./safe-tx/emission-controller-pause.json) | `EmissionController` | Halving-bug detected; emission rate anomaly; spurious mint authorization |
| [`safe-tx/emission-controller-resume.json`](./safe-tx/emission-controller-resume.json) | `EmissionController` | After patched emission logic + auditor sign-off |
| [`safe-tx/bridge-security-upgrade.json`](./safe-tx/bridge-security-upgrade.json) | `BridgeSecurity` | UUPS upgrade for signature-verification logic |

---

## How to use during a P0 incident

### Pre-incident (one-time setup)

1. **Council members import templates into Safe Web UI** at `app.safe.global` for the relevant Safe wallet
2. **Verify Safe address + threshold** matches the council operational multi-sig (3-of-5 per Q6 ratification)
3. **Bookmark the Safe URL** for fast access during incident
4. **Pre-load gas tank** — Safe must have ≥0.1 ETH on Base for transaction fees

### During P0 incident — pause workflow (per Exploit Playbook §6)

```
T+0      P0 declared by on-call engineer or council member
T+0-2    War room convenes; confirms pause is correct response
T+2      On-call engineer:
         a. Identifies affected contract from incident type
         b. Opens corresponding pause template JSON
         c. Imports to Safe Tx Builder via "Load Transaction" button
         d. Verifies contract address matches deployed mainnet address
         e. Broadcasts Safe Tx hash to council Signal group
         f. Posts in #war-room-active Discord channel
T+3-15   Council members cosign:
         - Open Safe URL on their device
         - Verify Tx hash matches engineer's broadcast
         - Confirm intent (do not blindly cosign)
         - Cosign — counts toward 3-of-5 threshold
T+15     Pause executes when 3rd signature lands; on-chain confirmation
T+15+    Public announcement per Exploit Playbook §3 (T+30 min)
```

### Failure modes and contingencies

**If Safe gas tank empty:**
1. Any council member sends ≥0.1 ETH to Safe address
2. Wait 1 confirmation
3. Resume cosigning

**If 3-of-5 quorum cannot be reached in 30 minutes:**
1. Page all 5 council members via every available channel
2. Public broadcast acknowledging pause attempt and reason for delay
3. At T+60 with no quorum: engage incident response firm per Exploit Playbook §5 escalation

**If pause cannot be invoked because contract is immutable:**
1. See [`NON_PAUSE_RESPONSE.md`](./NON_PAUSE_RESPONSE.md)
2. Pause is not an option — proceed to alternate containment

---

## Template format

Templates follow the **Safe Transaction Builder** JSON format (importable via `app.safe.global` Tx Builder UI). Each template includes:

- `version` — Safe Tx Builder schema version (1.0)
- `chainId` — target network (8453 = Base mainnet, 84532 = Base Sepolia)
- `createdAt` — template creation timestamp
- `meta` — human-readable metadata (name, description, transaction count)
- `transactions` — array of transactions to execute atomically

The `transactions` array contains:
- `to` — contract address
- `value` — ETH value (always "0" for pause/upgrade)
- `data` — pre-encoded call data (function selector + parameters)
- `contractMethod` — ABI fragment for verification
- `contractInputsValues` — human-readable parameters

Templates are **address-parameterized**: contract addresses default to mainnet placeholders (`0x0000…`); replace with actual deployed addresses before import. Sepolia variants in [`safe-tx-sepolia/`](./safe-tx-sepolia/) (post-Phase-1.3-deploy population).

---

## Template generation

Templates are hand-built rather than auto-generated to ensure:
- Function selectors are correct (matched against compiled contract ABI)
- Parameters use canonical zero values (no accidental misconfiguration)
- Human-readable description matches actual on-chain behavior
- No dependency on a build step under incident pressure

For each contract, the generation procedure was:
1. Confirm function exists in deployed contract source
2. Compute function selector via `keccak256("functionName(types)")[:4]`
3. Encode parameters per ABI spec
4. Build Safe Tx JSON with placeholders
5. Document in this README

---

## Pre-mainnet checklist

- [ ] Safe wallet address confirmed (mainnet + Sepolia)
- [ ] All 5 council members imported templates into Safe UI
- [ ] Safe gas tank funded ≥0.1 ETH on Base mainnet
- [ ] Tabletop exercise §1 (active drain) executed pause Tx within 15 min target
- [ ] Sepolia variants generated post-Phase-1.3 deploy
- [ ] Mainnet variants generated post-Phase-1.3 mainnet deploy
- [ ] All templates verified against contract ABIs in CI

---

## Related documents

- [Exploit Response Playbook](../../../docs/security/EXPLOIT_RESPONSE_PLAYBOOK.md) §6 — pause authorization procedure
- [Operational annex](https://github.com/prsm-network/prsm-private) (private repo) §5 — war-room logistics
- [`NON_PAUSE_RESPONSE.md`](./NON_PAUSE_RESPONSE.md) — response menu when pause not applicable

---

## Versioning

- **0.1 (2026-04-26):** Initial templates for 4 pausable/upgradeable contracts. Address placeholders pending Phase 1.3 mainnet deploy.
