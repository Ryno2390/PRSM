# PRSM Canonical Workflow — Base Sepolia E2E Runbook

**Status:** Stage 1 (preparation + scoping) complete 2026-05-06.
Stage 2 (live execution) is the next-session deliverable.

**Goal:** Demonstrate the canonical 8-step user workflow from
`PRSM_Vision.md` §4 running end-to-end against Base Sepolia, with
real on-chain transactions producing Basescan-verifiable Royalty
Distribution events. Sepolia first per founder direction; mainnet
follows once the Sepolia trace is documented and reviewed.

The *single*-claim being proven is **the cross-section that has
never been demonstrated end-to-end against real chain money**:
publisher-uploads-content → consumer-queries → SPRK-routes-to-
compute-node → on-chain-royalty-payment-splits-three-ways → tx
hashes pasteable into Basescan.

---

## 1. Network configuration (already in repo)

`prsm/config/networks.py` already pins the Base Sepolia testnet
configuration with all audit-bundle contracts deployed:

| Contract | Base Sepolia address |
|---|---|
| FTNS Token | `0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a` |
| ProvenanceRegistry (v1) | `0x2911f9a0a02896486CdF59d6d369764841DC0eA4` |
| RoyaltyDistributor | `0xB790045ff826C76fe02DBc54a6ef0021951Fd892` |
| Foundation Safe (deployer EOA) | `0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0` |
| EscrowPool | `0x4BDf07b2BB23176469bdEFca2B103AdB3DCb3dd2` |
| StakeBond | `0xDea103f33503BC7e73Ea447d43b2Cd7E2710D20A` |
| ProvenanceRegistryV2 (PRSM-PROV-1 Item 7) | `0xe75F0c24a9e63B63456d170d99F03Ab7fC3450A7` (deployed today) |

RPC: Alchemy Base Sepolia endpoint at
`$BASE_SEPOLIA_RPC_URL` (sourced from
`~/.prsm/testnet-deployer.env`).

Chain ID: 84532. Explorer: https://sepolia.basescan.org.

## 2. Stage 2 — execution plan

The workflow has eight steps per Vision §4. Each is mapped to
specific code paths in this repo + a verification check.

### Step 0: Pre-flight

- [ ] Load `~/.prsm/testnet-deployer.env`.
- [ ] Verify deployer ETH balance > 0.01 (gas for ~5 txs).
- [ ] Verify deployer FTNS balance — if zero, mint via the FTNS
      Sepolia token's MINTER_ROLE-holding admin or pull from the
      existing Sepolia treasury. ~100 FTNS sufficient for a small
      royalty exercise.

### Step 1: Publisher uploads a small dataset

**Code path:** `prsm.user_content_manager.UserContentManager.upload_content`
or directly via `prsm.node.content_uploader.ContentUploader`. Uses
the existing IPFS or local-content path; the Sepolia exercise
DOES NOT require BitTorrent seeding for the royalty leg to fire.

**Verification:** content CID saved to local upload record; the
ContentUploader's `provenance_client` (a `ProvenanceRegistryClient`
constructed from `PRSM_PROVENANCE_REGISTRY_ADDRESS=0x2911f9...0eA4`)
emits a `ContentRegistered` event on the v1 registry.

**Basescan check:** filter v1 ProvenanceRegistry events at
`0x2911f9a0a02896486CdF59d6d369764841DC0eA4` for the new
`ContentRegistered(contentHash, creator, royaltyRateBps)` event with
`creator = deployer address`.

### Step 2: LLM compiles intent → DSL → SPRK

**Code path:** `prsm.compute.inference.mcp` MCP server exposes
`prsm_inference`. For the Sepolia exercise, we skip the LLM round
trip and construct the SPRK directly via
`prsm.compute.collaboration.execution.sprk_executor` to keep the
trace deterministic. Real LLM-driven SPRK construction is a
follow-on demo.

### Step 3: Semantic sharding (skipped for v1 trace)

Single-node demo; cross-node sharding via
EmbeddingDHT is exercised in `test_dht_components.py` already.
Skipped here to keep the trace tight.

### Step 4–5: SPRK execution

**Code path:** `prsm.compute.parallax.executor.ParallaxScheduledExecutor`
with the existing in-process Wasmtime runtime. A simple "count
words in this content" SPRK is sufficient evidence for the trace.

### Step 6: On-chain royalty settlement (THE LOAD-BEARING STEP)

**Code path:**

```python
from prsm.economy.web3.royalty_distributor import RoyaltyDistributorClient

client = RoyaltyDistributorClient(
    rpc_url=os.environ["BASE_SEPOLIA_RPC_URL"],
    distributor_address="0xB790045ff826C76fe02DBc54a6ef0021951Fd892",
    ftns_token_address="0xF8d0c1AE75441d3C3Dd2A2420C0789043916412a",
    private_key=os.environ["PRIVATE_KEY"],
)
tx_hash, status = client.distribute_royalty(
    content_hash=content_hash_bytes_32,
    serving_node=deployer_address,  # self-as-node for v1 trace
    gross=10 * 10**18,  # 10 FTNS gross
)
```

**What happens on-chain (per `RoyaltyDistributor.sol` contract):**
1. Pull `gross` FTNS from payer (the deployer wallet).
2. Read creator + royaltyRateBps from v1 ProvenanceRegistry for
   `content_hash`.
3. Atomically split the gross — creator share, network treasury
   2%, serving-node share — and emit a single `RoyaltyPaid` event
   carrying all three amounts.

**Verification (THE basescan SCREENSHOT THE DEMO IS BUILT AROUND):**

- Filter `RoyaltyPaid(contentHash, payer, creator, servingNode,
  gross, creatorAmount, networkAmount, servingNodeAmount)` events at
  `0xB790045ff826C76fe02DBc54a6ef0021951Fd892`.
- Three transfers visible:
  - FTNS → creator address (creator's share at the registered rate)
  - FTNS → foundation treasury (2% per RoyaltyDistributor.NETWORK_FEE_BPS)
  - FTNS → serving node address (remaining share)
- Single tx hash links to Basescan with all three internal transfers.

### Step 7: Answer returned

Out of scope for this trace. The on-chain leg (step 6) is the
proof; the response delivery is mechanical.

## 3. Smoke script (Stage 2 deliverable)

A new file `scripts/exercise_canonical_workflow_base_sepolia.py`
will mirror the structure of
`scripts/exercise_provenance_registry_v2_sepolia.py` (which already
ships the RPC-replica retry pattern). The script will:

1. Read env vars: `PRSM_RPC_URL`, `PRSM_PRIVATE_KEY`,
   `PRSM_PROVENANCE_REGISTRY_ADDRESS`,
   `PRSM_ROYALTY_DISTRIBUTOR_ADDRESS`, `PRSM_FTNS_TOKEN_ADDRESS`.
2. Compute a deterministic `content_hash` from a small in-memory
   payload.
3. Register the content on the v1 ProvenanceRegistry with the
   deployer as creator and a 10% royalty rate (1000 bps).
4. Approve + distribute 10 FTNS via the RoyaltyDistributor.
5. Decode the resulting `RoyaltyPaid` event from the receipt.
6. Print Basescan tx URLs + the three split amounts.
7. Exit 0 on success.

## 4. Open questions to resolve before Stage 2

1. **FTNS balance on the deployer wallet.** Per the testnet-deployer
   env, the wallet is funded with ~0.025 ETH for gas. FTNS
   balance unknown. If zero, need to mint or transfer in. The
   testnet FTNS token has MINTER_ROLE; the deployer is presumably
   the role holder per `prsm/config/networks.py:118` ("deployer EOA
   per §9 ratified decision").
2. **Per-content royalty rate.** Pre-registering the test content
   with `royaltyRateBps = 1000` (10%) yields a clean three-way
   split for the screenshot: 9.8 FTNS gross gets 1.0 to creator,
   0.2 to treasury, 8.8 to node (after 2% network fee on the 90%
   non-royalty pool — exact math from `RoyaltyDistributor.sol`
   payment-distribution comment in `PRSM_Tokenomics.md` §5.1).

## 5. Stage 3 — mainnet bring-up (post-Sepolia)

Once the Sepolia trace is documented and posted, the same script
runs against Base mainnet with `PRSM_RPC_URL=https://mainnet.base.org`
and the mainnet contract addresses already pinned in `MAINNET`
config. Mainnet adds:

- Foundation Safe is the actual hardware-multisig (0x91b0...5791),
  not the deployer EOA. Step 6 settlement still works for
  arbitrary payers; only the *treasury* address differs.
- ProvenanceRegistryV2 mainnet deploy follows the council
  resolution `PRSM-CR-2026-05-06-2` once the deploy-script guard
  is removed in a follow-on PR.

---

## 6. What this runbook does NOT cover

- **Multi-node SPRK dispatch** (Vision §4 step 5). Proven in
  `test_dht_components.py` 3-node E2E + Phase 3.x.7 multi-node
  tensor-parallel tests. Re-running cross-node here would not
  produce different evidence than the existing passing test
  surface.
- **Tier B/C content confidentiality** (Vision §2). Phase 7
  delivery; out of scope until then per `PRSM_Vision.md` §13.
- **Aggregator-mediated settlement** (Vision §4 step 6, "staked
  aggregator"). v1 trace runs settlement directly; aggregator
  routing is a Phase 3 marketplace feature.

The runbook is intentionally narrow: prove the on-chain royalty
flow works against real Base Sepolia money, with documented
Basescan-pasteable evidence. Everything else has been proven in
unit tests.
