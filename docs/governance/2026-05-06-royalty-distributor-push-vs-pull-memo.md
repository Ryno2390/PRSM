# RoyaltyDistributor settlement pattern — push vs. pull

**Status:** Memo for council consideration — no resolution proposed yet.
**Issued:** 2026-05-06
**Author:** Ryne Schultz, Founder
**Subject:** Recommendation that the council formally accept the current Base mainnet `RoyaltyDistributor` push-payment behavior as the production policy, with a 90-day re-evaluation trigger, rather than redeploying to the Sepolia pull-payment variant.

---

## 1. The fork in the road

Two `RoyaltyDistributor` variants are currently in production use:

| Variant | Deployed | Address | Settlement |
|---------|----------|---------|------------|
| Mainnet | 2026-05-04 | `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` | **Push** — funds routed directly to recipient EOAs in the distribute tx |
| Sepolia | 2026-05-05 | `0xB790045ff826C76fe02DBc54a6ef0021951Fd892` | **Pull** — `claimable[address]` mapping accumulates; recipients call `claim()` |

Both variants enforce the same canonical 3-way split (10% creator / 2% network fee / 88% serving node) at the same `NETWORK_FEE_BPS = 200` and `MAX_ROYALTY_RATE_BPS = 9800` constants. The economic mechanic is identical. The fork is purely about *settlement* — how funds reach the three recipients.

Discovery context: the divergence was found during the 2026-05-06 mainnet bring-up Path A trace (`docs/2026-05-06-canonical-workflow-base-mainnet-trace.md`) when the trace script's `claimable()` reader reverted on mainnet. The on-chain payment had already landed correctly via push; the read-side verification failed because mainnet has no `claimable()` function.

This memo presents the trade-offs and recommends the council formally adopt one variant as the production policy rather than tolerating an indefinite Sepolia/mainnet behavioral asymmetry.

## 2. Trade-offs

### 2.1 Push pattern (mainnet status quo)

**Pros:**
- **Single tx, no claim step.** End-to-end royalty distribution completes in one signature. UX is materially simpler — recipients just see funds arrive in their wallet without any post-distribute action.
- **Lower aggregate gas.** Push does N transfers in one tx (`distributeRoyalty` → 3× internal Transfer). Pull does 1 inbound transfer + N outbound `claim()` txs, each paying the per-tx 21k base + ERC-20 transfer ~50k = ~71k gas/recipient. For three recipients, push is ~150k vs. pull ~150k + 3×71k = ~363k aggregate. **2.4× gas multiplier on pull.**
- **Immediate finality.** Recipient sees tokens at the same block as the distribute. No "pending" state.
- **No zombie balances.** Tokens never sit in escrow waiting on a recipient who forgot to claim, never claims, or has lost their key. With pull, an unclaimed `claimable[address]` balance is administratively dust forever.
- **Contract simpler.** Bytecode smaller; surface area for bugs smaller. The pull variant adds the `claimable` mapping, the `claim()` function, and per-claim accounting, all of which are additional state and code paths to audit.

**Cons:**
- **No recovery from address-typo errors.** If `ProvenanceRegistry.transferContentOwnership(content_hash, wrong_address)` is ever called and a distribute fires before the error is noticed, the creator's 10% goes to the wrong address. Push has no admin override to redirect after the fact.
- **No mid-flight recovery from compromised recipient.** If a serving-node EOA's key is leaked between distribute issuance and recipient receipt — though the window is one block, ~2s on Base — the push sends funds directly to the compromised address. With pull, an admin (post-rollout) could intercept and reroute via a roles-gated `redirectClaimable(from, to)` operation.
- **Recipient must accept ERC-20 with no receive contract.** Push requires the recipient to be an EOA or a contract that can hold ERC-20. Almost universally the case for PRSM today (creators/nodes/treasury are all EOAs or simple multisigs), but pull would tolerate more exotic recipients.

### 2.2 Pull pattern (Sepolia)

**Pros:**
- **Address-error recovery path.** Funds sit in `claimable[wrong_address]` until the wrong recipient (or the admin, if the contract has roles for it) claims. The Sepolia variant *does not* currently have an admin redirect — but adding one is a smaller surgery on a pull contract than retrofitting recovery into a push contract.
- **Recipient pays gas.** Distribute initiator does not subsidize per-recipient transfer gas. Useful at scale if there are many tiny royalties — recipient batches claims when economical.
- **Contract escrow buffer.** Funds are held by the RD contract between distribute and claim. Useful for atomicity if multiple distributes are pipelined: claims process the cumulative state, not per-distribute.

**Cons:**
- **2.4× aggregate gas in steady state.** Each recipient pays ~71k gas to claim. Across many recipients across many distributes, the gas cost compounds materially. At scale ($X/yr in royalty volume), this is a meaningful drag on creator/node economics.
- **Claim step is UX friction.** Creator earns royalty in tx N → realizes tokens are there in tx N+1 (claim). Two-step UX is a documented source of user confusion in pull-payment patterns.
- **Zombie balances.** A creator who loses their seed phrase before claiming → balance is permanently stuck in `claimable[creator_address]`. The pull pattern needs a post-deadline sweep mechanism (which the Sepolia variant doesn't currently have) to recover dust.
- **Larger surface area.** More code, more state, more audit. The 2026-05-05 T6.2 D-04 refactor that introduced the pull pattern on Sepolia added the `claimable` mapping, the `claim()` function, and supporting accounting — about ~150 LoC of additional Solidity.

### 2.3 Asymmetric situation — mainnet path-dependence

**Critical context:** The mainnet RoyaltyDistributor is *already deployed* with the push pattern. Switching requires:

1. Deploy a new pull-pattern RoyaltyDistributor at a fresh address.
2. Migrate ProvenanceRegistry references (no-op — content_hash is keyed on the registry, not the distributor).
3. Update `prsm/config/networks.py:MAINNET.royalty_distributor` to the new address.
4. Update all client code that reads `claimable[]` (currently the Sepolia trace script can already do this).
5. **Cannot redirect in-flight distributes.** Any RoyaltyPaid event already on-chain stays on the old contract; new distributes go to the new contract. There is no reasonable way to consolidate.

The 2026-05-06 mainnet trace put 0.2 FTNS into the Foundation Safe via push. That tx is done. A redeploy doesn't undo it; it just means *future* distributes go through a different contract. So the cost is migration overhead + a contract bifurcation footnote in audit trail, not lost funds.

## 3. Recommendation

**Recommend the council formally adopt the push pattern as the production policy on Base mainnet, with a 90-day re-evaluation trigger.**

The reasoning:

1. **Path of least resistance.** Mainnet is already push. No redeploy. No migration. No client churn.
2. **Steady-state economics favor push.** 2.4× gas multiplier on pull would compound noticeably as PRSM scales.
3. **Recovery scenarios are theoretical today.** PRSM has not yet had a single recipient-address typo or mid-flight key compromise on mainnet. The recovery advantage of pull is buying insurance against zero observed events; the gas advantage of push is paying real bps every distribute.
4. **Sepolia stays pull.** Sepolia is a testbed; the asymmetry is acceptable as long as the trace tooling handles both variants (which it does, post-2026-05-06 patch). Treating Sepolia as a "what if we'd gone pull" reference allows ongoing measurement of the divergence cost.

### 3.1 Re-evaluation trigger

The council shall re-evaluate this decision if any of the following occur within 90 days of resolution adoption:

- One or more documented incidents of address-typo-loss on mainnet RoyaltyDistributor (creator or serving-node royalty going to a wrong recipient because of upstream error).
- One or more documented incidents of mid-flight recipient compromise where a pull contract's redirect path would have prevented loss.
- A material change in Base mainnet gas economics (e.g., 10× gas-price spike that shifts the gas-cost calculus).
- A specific governance event requiring contract-level pause/revoke of pending royalties (e.g., regulatory injunction, fork-recovery scenario).

If none of those trigger by 2026-08-04, the recommendation matures into the steady-state policy.

### 3.2 Open question — should Sepolia keep pull?

The Sepolia variant cost real engineering time to implement (T6.2 D-04 refactor). Tearing it out would discard that work. Two options:

- **Keep both:** Sepolia pull, mainnet push. Asymmetry tolerated as a "what if we'd gone pull" reference. Trace tooling handles both. **Recommended.**
- **Force parity:** redeploy Sepolia to push to match mainnet. Achieves clean parity but discards the pull experiment.

The memo defers this to council preference; my own recommendation is keep both.

## 4. Resolution path forward

If the council accepts §3, the next step is a formal resolution PRSM-CR-2026-05-06-4 with the following operative clauses:

- Adopt push as the production policy on Base mainnet for `RoyaltyDistributor`.
- 90-day re-evaluation trigger as defined in §3.1.
- Sepolia retains the pull variant as a live-comparison reference.

I have not yet drafted that resolution; this memo is the input for the discussion. If the council wants me to draft, that's a follow-up step.

## 5. Non-decisions

This memo does NOT recommend or oppose:

- Modifying the canonical 10/2/88 split or any underlying bps constants.
- Modifying `networkTreasury` (immutable on the deployed contract regardless of variant).
- Migrating `RoyaltyDistributor` to a fully different architecture (e.g., a periodic settlement contract with batch distributes). That would be a separate, larger conversation.
- Bridging the asymmetry by adding a `claimable()` shim to the mainnet contract (would require an upgrade or redeploy; same complexity as a full pattern switch).

## 6. References

- Mainnet RoyaltyDistributor: `0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2` (Basescan)
- Sepolia RoyaltyDistributor: `0xB790045ff826C76fe02DBc54a6ef0021951Fd892` (sepolia.basescan.org)
- 2026-05-06 mainnet trace: `docs/2026-05-06-canonical-workflow-base-mainnet-trace.md`
- 2026-05-06 Sepolia trace: `docs/2026-05-06-canonical-workflow-base-sepolia-trace.md`
- Trace script (handles both variants): `scripts/exercise_canonical_workflow_base_sepolia.py`
- T6.2 D-04 pull-payment refactor (Sepolia): commit-trail under PRSM-PROV-1 Item 6
