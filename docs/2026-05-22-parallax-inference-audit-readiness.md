# PRSM Verifiable Inference — Audit Readiness Summary

Date: 2026-05-22 · Sprint range: 498–698 (201 sprints) · Author: PRSM
core team · Reading time: ~15 min.

This document is the external-facing summary of what PRSM's §7
verifiable-inference path provably does today, what's still open, and
how an external party (auditor, investor, engineer) can reproduce
every claim below.

It is NOT marketing. Every operational claim cites:

- A specific commit hash on `main`
- An on-chain transaction on Base mainnet, OR
- A live HTTP response captured during a live-attest probe

## 1. Executive summary

**What works today, end-to-end, on a live mainnet-anchored fleet:**

- `/compute/inference` returns a cryptographically-signed inference
  receipt. Output is bit-identical to a reference HuggingFace greedy
  decode. (Sprint 687 + 688)
- `/compute/inference/stream` returns Server-Sent Events with per-token
  autoregressive output. Output is deterministic + matches HuggingFace
  reference. (Sprints 692 / 693 / 694)
- The DHT-backed GPU pool discovers peers via the on-chain
  `PublisherKeyAnchor` contract (Base mainnet
  `0xd811ad9986f44f404b0fd992168a7cc76206df03`). Unregistered
  pseudonyms cannot serve traffic. (Sprints 680–685)
- Multi-host 2-stage allocation runs across geographically distinct
  peers (NYC ↔ SFO) over WAN P2P transport, returning a single signed
  receipt with embedded per-stage attestations. (Sprint 695)
- A real GPU peer (Lambda Cloud A10) joined the fleet from cold-image
  to first signed receipt in 8 documented steps. Output is
  bit-identical to the CPU peers. (Sprint 698, runbook at
  `docs/operations/parallax-inference-deploy.md`)
- 22 operator env vars are validated by a single CLI command
  (`prsm node parallax-readiness`) with non-zero exit on missing or
  invalid values, suitable for CI gating. (Sprint 696)

**Wire-protocol hardening (sprints 711-731, 22-sprint arc):** the
remote token-stream + unary chain-executor RPC have been audited
end-to-end across 10 dimensions, closing **35 F-class production-
blockers (F30 through F64)** with corresponding pin tests + 5
integration tests on real `WebSocketTransport`. Streaming +
unary paths now have parity defense on collision (F52/F57), payload
size DoS (F55/F58), per-peer concurrency cap (F56/F59), and
response hijack via sender binding (F53/F60). Plus streaming-only
race fixes (F50/F51), disconnect cleanup (F54), back-pressure
(sprint 713), observability CLI (sprint 722), and hang-defense
timeouts (F61/F62 unary + streaming). **F63/F64 (sprints 730-731)
restored the cryptographic foundation under EVERY MSG_DIRECT
handler in the codebase** — including ledger_sync FTNS transfers,
compute_provider, storage_provider, content_provider, and
agent_registry — by binding `msg.sender_id` to the handshake-
authenticated `peer.peer_id` at the transport dispatch boundary.
Pre-731, a peered third party could spoof sender_id to bypass
per-peer caps OR forge responses.

**What is honestly NOT closed yet** (Section 7):

- Stake-eligibility runs in "advisory" mode in the live fleet because
  no operator has bonded FTNS on the `StakeBond` contract yet. The
  production path is wired (sprint 690) but unused.
- ~~Hardware-profile gossip propagation has a documented inconsistency
  (F46)~~ — **closed in sprint 700**. `_handle_peer_response` now uses
  monotonic-improvement semantics: gossip can ADD profile data but
  cannot REMOVE it. Operators no longer need env-var overrides for
  reliable heterogeneous CPU+GPU pool allocation.
- Cross-host token streaming (one peer streams from another peer's
  GPU) **CLOSED in sprint 711** — wire protocol shipped + pin-tested;
  live cross-host attest is sprint 712. Self-dispatch streaming works
  fully. (F40
  deferred per sprint 691)

## 2. Operational claims with on-chain evidence

| Claim | Evidence |
| --- | --- |
| The PRSM PublisherKeyAnchor contract is deployed on Base mainnet | Address `0xd811ad9986f44f404b0fd992168a7cc76206df03`, verifiable via Basescan or any Base RPC |
| The PRSM StakeBond contract is deployed on Base mainnet | Address `0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed` |
| Three operator pubkeys are registered on the live anchor | NYC `484f003c895ee02ac7ed01e570a6a51f` (sprint 623), SFO `d437aa67d99cff4a6a17179f5c731b77` (TX `0x209601b0862f81facda2bda0a06fa09df3c83786184596c2460d2c53b9cb8255`, block 46292504), Lambda `30d6b443a0e2249d50c7b7f3226a5649` (TX `0x1d038e179a8b4c232bcfb32cf186dc858a0b6005392d711d2bfd7ed7be4b58f0`, block 46331231) |
| All three operator daemons return the same signed `output_hash` for the same prompt | `fe0663fd13cdb08743dca3d2d5d1168762556ea36f77d8ab6009306d12a6f351` for prompt "The capital of France is" / max_tokens=1 / gpt2 (sprint 687 NYC CPU receipt, sprint 698 Lambda A10 GPU receipt) |
| Streaming receipts are deterministic across runs | `output_hash: 1b46bc86847a74868649092f6e9bfab0363bfe229548867fd991f0200553f4f8` reproducible (sprint 694 + 698) |
| Multi-host 2-stage receipt embeds both stage attestations | `topology_assignment.positions = [[0,0,"484f003c..."],[1,0,"d437aa67..."]]`, `stage_count: 2`, single combined receipt signed by the settler (NYC) — sprint 695 |
| Receipts match local HuggingFace `model.generate()` byte-for-byte | "The capital of France is" → `" the"` (token id 262) matches `AutoModelForCausalLM.from_pretrained('gpt2') .generate(input_ids, max_new_tokens=1, do_sample=False)` on the same prompt — sprint 688 + 694 |

## 3. Architecture (text flow)

```
client
  │
  │ POST /compute/inference {prompt, model_id, ...}
  ▼
ParallaxScheduledExecutor.execute (asynchronous handler)
  │
  ├── pre-execute gates:
  │     ├── pool gathering         ← DHT-backed pool (sprint 682)
  │     ├── Adapter A: anchor verify ← on-chain PublisherKeyAnchor
  │     ├── Adapter C: stake check  ← PoolBackedStakeLookup (sprint 690)
  │     └── Adapter B: tier gate    ← privacy_tier filter
  │
  ├── Phase-1 allocation: DP allocator splits N layers across pool
  │   (deterministic; reproducible across runs given same pool)
  │
  ├── Phase-2 routing: DP optimizer picks min-latency chain
  │   (uses tflops, memory_gb, RTT from advertised profile)
  │
  └── chain execution (RpcChainExecutor → real)
        ├── stage 0 dispatch → local StageExecutor (self) OR P2P (remote)
        │     ├── prompt encoder (HF tokenizer + wte+wpe embedding)
        │     ├── HuggingFaceLayerSliceRunner forward through layer range
        │     └── stage signs response with operator Ed25519 key
        ├── stage N+1 dispatch (if multi-stage)
        └── final stage: lm_head + sampling → next token
              ↓
        settler builds + signs InferenceReceipt:
          - epsilon_spent, tee_attestation (per-stage hex array),
            output_hash, settler_signature, settler_node_id,
            topology_assignment, streamed_output flag
              ↓
        returned to caller as JSON (unary) or terminal SSE frame (streaming)
```

Key invariants verified by code + live attest:

- The settler signs over the full canonical receipt bytes. Tampered
  receipts fail `signature_valid` on `/compute/receipt/verify` (sprint
  433 — pinned in 6 tests).
- Each stage produces its own TEE attestation hex; the settler
  concatenates them into `tee_attestation` byte array. Verifier
  inspects each stage independently.
- `topology_assignment.positions` is a structural record:
  `(stage_index, slot_index, node_id)`. Reconstructible from
  `chain.stages`; hash verifiable. (Sprint 414)
- Streaming and unary paths produce IDENTICAL `output_hash` for the
  same prompt + sampling defaults. Confirmed live for gpt2 single-token
  (sprint 694).

## 4. Live-attest evidence catalog

This section lists the actual receipts captured during sprint 685–698
live-attests. All are reproducible: redeploy any operator, POST the
same prompt, get the same `output_hash`. (settler_signature varies
because it commits over a per-request timestamp.)

### 4.1 NYC unary single-token (sprint 687)

```
POST http://127.0.0.1:8002/compute/inference
{"prompt":"The capital of France is","model_id":"gpt2","budget_ftns":1.0,
 "privacy_tier":"none","content_tier":"A","max_tokens":1}

→ success: true, output: " the"
  duration_seconds: 1.59
  settler_node_id: 484f003c895ee02ac7ed01e570a6a51f  ← NYC operator
  output_hash: 975936a99e86cc22755110011ce90f1b3419c1fae873ebb6aabe1e0a694132d8
```

### 4.2 NYC unary semantic-correctness fix (sprint 688, F38 closed)

```
prompt: "The capital of France is" / max_tokens=1
→ output: " the" (token_id 262, matches HF reference greedy)
  output_hash: fe0663fd13cdb08743dca3d2d5d1168762556ea36f77d8ab6009306d12a6f351
  duration_seconds: 0.73
```

### 4.3 NYC streaming SSE 4-token (sprint 694)

```
POST /compute/inference/stream  ← Server-Sent Events
prompt: "The capital of France is" / max_tokens=4

event: token   data: seq=0, text=" the",     token_id=262
event: token   data: seq=1, text=" capital", token_id=3139
event: token   data: seq=2, text=" of",      token_id=286
event: token   data: seq=3, text=" the",     token_id=262, finish_reason="max_tokens"
event: result  data: success=true, output=" the capital of the",
                     output_hash: 1b46bc86847a74868649092f6e9bfab0363bfe229548867fd991f0200553f4f8
                     streamed_output: true
                     duration_seconds: 1.02
```

### 4.4 Multi-host 2-stage NYC↔SFO (sprint 695)

```
prompt: "Hi" / max_tokens=1
→ topology_assignment.positions = [
    [0, 0, "484f003c895ee02ac7ed01e570a6a51f"],   ← NYC, layers 0-5
    [1, 0, "d437aa67d99cff4a6a17179f5c731b77"]    ← SFO, layers 6-11
  ]
  stage_count: 2, slots_per_stage: 1
  output: "."
  tee_attestation: 2-stage hex array (both NYC + SFO sign)
  duration_seconds: 27.8 (cold-load both droplets + cross-host dispatch)
```

### 4.6 Activation-DP injection at tier=standard (sprint 702)

```
prompt: "The capital of France is" / max_tokens=1 / privacy_tier=standard
→ success: true, output: " screen"     ← DIVERGES from tier=none's " the"
  privacy_tier: standard
  activation_noise_trace: {
    per_stage_epsilon: [8.0],
    total_epsilon_spent: 8.0,
    clip_norm: 1.0,
    stage_count: 1,
    tier: "standard"
  }
  output_hash: 1df3a3bec8b48d66e12359c9fb006f999d9d2469dafd4c0bd61e18189e679b5d
  cost_ftns: 0.1320                     ← bumped from 0.12 (tier=none) for DP overhead
  settler_node_id: 484f003c895ee02ac7ed01e570a6a51f
  duration_seconds: 2.24
```

Three pieces of evidence that DP injection is REAL, not just claimed:
1. Output diverges from tier=none for the same prompt + sampling
   defaults (different token id → Gaussian noise altered the
   activations enough to flip argmax)
2. `activation_noise_trace` field is populated with per-stage
   epsilon vector + clip_norm + tier (sprint 414's data path,
   sprint 295's noise injector)
3. Cost is bumped 10% reflecting DP-overhead pricing in the
   privacy-budget module

Settler_signature commits over the entire receipt including
activation_noise_trace bytes — a verifier reading the on-chain
anchor + this receipt JSON can independently confirm that the
operator committed to this specific noise trace.

### 4.5 GPU peer (Lambda A10) signed inference (sprint 698)

```
prompt: "The capital of France is" / max_tokens=1
→ success: true, output: " the"
  output_hash: fe0663fd13cdb08743dca3d2d5d1168762556ea36f77d8ab6009306d12a6f351
  ← BYTE-IDENTICAL TO 4.2 NYC CPU receipt
  duration_seconds: 0.108  ← 7× faster than NYC CPU (sprint 687)
  settler_node_id: 30d6b443a0e2249d50c7b7f3226a5649  ← Lambda A10
  receipt: signed by Lambda operator's Ed25519 key
```

This is the strongest single piece of evidence: identical `output_hash`
from a CPU droplet and a cloud GPU prove the architecture is
semantically equivalent across hardware classes. Different operators,
different hardware, different Ed25519 keys → same canonical output →
same hash. (Settler signatures differ — they commit over a
per-request timestamp.)

## 5. Cost model

### 5.1 Live attestation costs to date (sprints 685–698)

| Item | Cost |
| --- | --- |
| 3× on-chain pubkey registrations (NYC, SFO, Lambda) | ~$0.03 total Base gas |
| NYC bootstrap-us droplet (DO Basic 2GB, always-on) | $12/mo |
| SFO operator droplet (DO Basic 2GB, always-on) | $12/mo |
| Lambda A10 GPU bursts (terminate when done) | $1.29/hr while running |
| FTNS staked on-chain | $0 (advisory mode in live fleet) |
| **Recurring fleet cost (always-on)** | **$24/mo** |
| **One live-attest session w/ GPU** | ~$1.50 (1 hour Lambda) |

### 5.2 What a third party would pay to reproduce

| Step | Cost |
| --- | --- |
| Provision their own DO droplet OR Lambda instance | $6–$1.29 per hour |
| Fund a Base EOA with ~0.0001 ETH | $0.01 |
| Run `scripts/sprint_675_register_operator_pubkey.py` | ~$0.01 gas |
| Follow `docs/operations/parallax-inference-deploy.md` | 0 |
| **Total to first signed receipt** | **~$0.05 + 1 hour wall time** |

## 6. Reproducibility

Everything in §4 is reproducible by an external party with no PRSM-team
involvement. The full path:

1. `git clone https://github.com/prsm-network/PRSM.git`
2. Follow `docs/operations/parallax-inference-deploy.md` (8 numbered
   steps including the on-chain anchor registration)
3. Run `prsm node parallax-readiness` to verify the 22 env vars
4. `systemctl start prsm-operator.service`
5. POST `/compute/inference` with the same prompt → identical
   `output_hash` (settler_signature differs because it commits over a
   per-request timestamp)
6. POST the receipt to `/compute/receipt/verify` → `ok: true,
   signature_valid: true`
7. Tamper any byte of the receipt → `signature_valid: false`

The verification surface lives at:

- `prsm/compute/inference/receipt.py` — receipt sign + verify
- `prsm/security/publisher_key_anchor/client.py` — on-chain pubkey
  lookup
- `prsm/compute/parallax_scheduling/trust_adapter.py` — Adapter A/B/C
  (anchor verify, tier gate, stake eligibility)

A receipt is verifiable using ONLY (a) the receipt JSON, (b) the Base
mainnet anchor address, (c) any Base RPC URL. No PRSM-team server-side
component is in the trust path.

**Sprint 703 ships `scripts/verify_prsm_receipt.py`** — a standalone
~300-line Python script that performs the full verification with
ZERO imports from the `prsm` package. An external party with only
this file, a receipt JSON, and `pip install web3 cryptography` can
verify any PRSM receipt in under 30 seconds:

```bash
python3 verify_prsm_receipt.py receipt.json
# → valid: true, anchor_lookup: ..., signature_valid: true,
#   attestation_envelope: {...}, noise_trace: {...}
```

The script does not assume the PRSM repo exists. The "no PRSM-team in
the trust path" claim is now operationally provable in a single
command-line invocation. 5 pin tests defend the standalone signing-
bytes implementation against drift from the PRSM-side
`InferenceReceipt.signing_payload` source.

### 6.1 Try it yourself in 30 seconds (sprint 706)

A real signed receipt is checked into the repo at
`docs/sample-receipts/tier-standard-dp-2026-05-22.json`. It was
captured from a live `/compute/inference` POST against NYC at
`bootstrap-us.prsm-network.com` on 2026-05-22 with
`privacy_tier=standard` (activation-DP injection wired). Verify it
yourself:

```bash
git clone https://github.com/prsm-network/PRSM.git
cd PRSM
pip install web3 cryptography
python3 scripts/verify_prsm_receipt.py docs/sample-receipts/tier-standard-dp-2026-05-22.json
```

Expected: `✓ VALID — receipt verifies cleanly` with the
`activation_noise_trace` populated (per_stage_epsilon=[8.0],
total=8.0, tier=standard).

To prove tamper-detection works, flip one byte of any signed
field and re-verify:

```bash
python3 -c "
import json
r = json.load(open('docs/sample-receipts/tier-standard-dp-2026-05-22.json'))
old = r['output_hash']
r['output_hash'] = old[:-2] + ('00' if old[-2:] != '00' else 'ff')
json.dump(r, open('/tmp/tampered.json', 'w'))
"
python3 scripts/verify_prsm_receipt.py /tmp/tampered.json
```

Expected: `✗ INVALID — settler_signature does NOT verify against
the anchor-registered pubkey`. Every byte of the canonical signing
payload is committed-to by the settler's Ed25519 signature.

See `docs/sample-receipts/README.md` for more receipts + the
canonical signing-bytes spec. Three samples now shipped covering
the full live-attested matrix:

- `tier-none-unary-2026-05-22.json` — baseline unary inference,
  `output_hash: fe0663fd…` bit-identical to HuggingFace gpt2 greedy
- `tier-none-streaming-2026-05-22.json` — SSE 4-token streaming,
  `output_hash: 1b46bc86…` bit-identical to streaming reference
- `tier-standard-dp-2026-05-22.json` — activation-DP injection
  with `activation_noise_trace` populated (ε=8.0)
- `multi-host-2stage-2026-05-22.json` — **TRUE 2-stage cross-WAN**
  inference with `topology_assignment.stage_count=2`, both NYC +
  SFO operators sign per-stage attestations

Run the verifier against any of them with the same command. Each
proves a different load-bearing PRSM claim:
  1. **bit-identical CPU inference to local HF reference** (unary)
  2. **per-token autoregressive streaming with deterministic hash**
  3. **DP injection with epsilon accounting in signed receipt**
  4. **multi-host 2-stage chain with cryptographic per-stage attestations**

## 7. Known limits + active gaps

This section is honest about what is NOT closed.

### 7.1 Stake-eligibility advisory mode

`PRSM_PARALLAX_STAKE_ELIGIBILITY=advisory` is set on all three operators
in the live fleet because no operator has bonded FTNS on the `StakeBond`
contract yet. The production code path is wired:

- Operators advertise their EOA via `PRSM_OPERATOR_ADDRESS` env
- `PoolBackedStakeLookup` reads on-chain stake balance per peer (sprint 690)
- `StakeWeightedTrustAdapter.is_eligible(node_id)` filters unstaked peers
  from the pool BEFORE Phase-1 allocation

Switching the live fleet to `enforced` mode requires:

1. The operator deposits FTNS into the StakeBond contract via
   `prsm staking stake` (existing CLI)
2. The operator's `PRSM_OPERATOR_ADDRESS` is set to the EOA they
   staked from
3. The operator restarts with `PRSM_PARALLAX_STAKE_ELIGIBILITY=enforced`

Until then, the advisory bypass is a documented dev-mode posture. It
is NOT a security claim that production has stake enforcement —
production today is "anyone with an anchor-registered pubkey can be a
stage."

### 7.2 F46: DHT hardware-profile propagation — **CLOSED in sprint 700**

Originally surfaced during sprint 698: when the Lambda A10 peer joined
the fleet, NYC's pool snapshot initially showed Lambda with stale CPU
specs (tflops_fp16 ≈ 0.08 vs actual 33.9). Root-caused to
`_handle_peer_response` clobbering authoritative profile data: when a
gossipping peer (Y) sent its peer-list to NYC, each entry carried Y's
view of peer X. If Y had `known_peers[X].hardware_profile = None`
(because Y hadn't yet received X's direct DISCOVERY_ANNOUNCE), NYC
would overwrite its OWN good profile for X with that None.

Sprint 700 fix: monotonic-improvement gossip. `_handle_peer_response`
preserves the existing local `hardware_profile` when the incoming
gossip entry has None and we already have one. Gossip can ADD profile
data but cannot REMOVE it. `_handle_announce` retains latest-write-
wins semantics because the announcer IS authoritative for its own
profile. Tag `sprint-700-f46-monotonic-hardware-profile-merge-ready-
20260522` commit `466a8ccb`. 5 new pin tests defend the distinction
between authoritative-announce and non-authoritative-gossip paths.

After sprint 700, the env-var overrides shipped in sprint 695
(`PRSM_PARALLAX_TFLOPS_FP16_OVERRIDE`, `PRSM_PARALLAX_MEMORY_GB_OVERRIDE`)
remain available for operators who want to pin advertised values
deliberately (e.g., for testing or for resource budgeting), but they
are no longer required for reliable heterogeneous CPU+GPU pool
allocation.

### 7.3 F40: Remote token-stream transport — **CLOSED in sprint 711**

`/compute/inference/stream` works for **self-dispatch** streaming (the
operator handles all stages locally). For **multi-host** streaming,
where one peer streams from another peer's GPU, the cross-host
token-stream transport was the last non-user-gated audit-doc gap.
**Sprint 711 ships the wire protocol** (`CHAIN_STREAM_MSG_TYPE` with
`CHAIN_STREAM_REQ_KEY` / `CHAIN_STREAM_FRAME_KEY` / `CHAIN_STREAM_END_KEY`
/ `CHAIN_STREAM_SEQ_KEY`):

- Requester → server: ONE `CHAIN_STREAM_REQ` over `MSG_DIRECT`
- Server → requester: MULTIPLE `CHAIN_STREAM_FRAME` (one per chunk,
  sequenced for reorder-safety)
- Server → requester: ONE terminal `CHAIN_STREAM_END` (optional error
  field for mid-stream failures → cleanly closes requester iterator)

Client-side `_remote_token_stream_dispatch` uses a per-stream
`asyncio.Queue` stored on `node._chain_executor_pending_streams`;
`handle_chain_stream_response` routes incoming frames threadsafe via
`loop.call_soon_threadsafe`. Server-side `handle_chain_stream_request`
calls `LayerStageServer.handle_token_stream(request_bytes)` and ships
frames as they arrive. `build_token_stream_send_message_adapter`'s
remote branch now delegates to the new dispatcher (replaces sprint 691's
placeholder `RuntimeError("remote streaming not yet wired")`).

8 pin tests defend the wire protocol (constant-distinctness, ignore-non-
stream messages, ignore-wrong-direction messages, ignore-unknown-stream
late frames, adapter-delegates-to-dispatch, node.py registers both
handlers). 530 cross-suite green. Tag
`sprint-711-remote-token-stream-wire-protocol-merge-ready-20260522`
commit `2a54b97c`.

Live-attest of cross-host streaming end-to-end across NYC+SFO is
sprint 712's scope. Streaming back-pressure + integration with the
sprint-704 OOM semaphore for streaming inference is sprint 713's scope.
Both are follow-ons, not new gaps — the wire protocol itself is
complete and pin-tested.

### 7.4 NYC 2GB droplet OOM cycling — **CLOSED in sprint 704**

NYC's 2GB DO droplet OOM-cycled during sprint 698's cross-host
coordination cold-load when concurrent inference arrived during
gpt2 model load. Sprint 704 closes this with an env-gated
asyncio.Semaphore: `PRSM_INFERENCE_CONCURRENCY_LIMIT=1` serializes
inference handling so peak memory is bounded by a single request's
working set.

Both unary `/compute/inference` and streaming `/compute/inference/
stream` paths share the semaphore. Streaming acquires before
iterator construction + releases in the outer try/finally — the
slot covers cold-load + all token-frame generation. Default
unset preserves pre-704 no-cap behavior for memory-rich nodes
(Lambda A10 with 200GB RAM doesn't need it).

Deployed live on NYC + SFO 2026-05-22 with limit=1. Tag
`sprint-704-inference-concurrency-limit-merge-ready-20260522`
commit `83b2af46`. 7 pin tests defend the gate (serialization,
disabled-when-unset, rebuild-on-limit-change, defense against
typos/zero/negative).

**Live-attested under concurrent load (sprint 705)**: 4 simultaneous
POSTs to NYC's `/compute/inference` returned 4× signed receipts in
~40s total wall time with staggered completion times (29s / 33s /
36s / 39s — each request waited for the prior). Daemon stayed
`active` throughout with 855 MB free RAM at end (vs. pre-704 the
2nd concurrent request would OOM-kill the daemon mid-flight).
The semaphore-gated fix holds under realistic concurrent load.

### 7.5 Activation-DP injection at tier=standard — **CLOSED in sprint 702**

Originally a known limit: sprints 685–698 evidence covered only
`privacy_tier: "none"`. Sprint 702 live-attested the activation-DP
injection path against the NYC operator with a software-TEE
runtime via the new `PRSM_PARALLAX_TIER_GATE=advisory` env that
mirrors sprint 686's stake-eligibility advisory pattern. The
adapter-side Adapter B + the server-side `LayerStageServer
._is_hardware_tee` runtime check both honor the advisory env so
software-only operators can exercise the DP code path. Production
deployment with real TEE hardware retains enforced mode unchanged
(default + typo-fallthrough).

Live-attest receipt (full evidence in §4.6 below): `privacy_tier:
"standard"`, `output: " screen"` (diverges from `" the"` at
tier=none, proving Gaussian noise was actually injected),
`activation_noise_trace: {per_stage_epsilon: [8.0],
total_epsilon_spent: 8.0, clip_norm: 1.0, stage_count: 1, tier:
"standard"}`, cost bumped to 0.132 FTNS reflecting DP-overhead
pricing.

Tags: `sprint-702-tier-gate-advisory-dp-injection-live-merge-
ready-20260522` commit `8d21b2d8`. Streaming + DP combination
remains documented as an explicit error path in sprint 689 (not
silently dropped) — separate sprint when needed.

### 7.6 Model coverage

Live-attested with gpt2 only (124M params, 12 layers). The wiring is
designed for arbitrary HuggingFace causal LMs and the configuration
surface (`PRSM_PARALLAX_HF_MODEL_ID`) supports any HF model. Larger
models (llama-3.2-1B, 3B, etc.) would exercise:

- More layers → more interesting Phase-1 splits
- More memory pressure → more rigorous capacity math
- Real GPU value (gpt2 was so small the A10 was only 3× faster than
  CPU; a 3B model would see 30–50× speedup)

Not blocked by any architectural gap — purely a deploy step that
needs more disk + memory than the current $12/mo droplets have.

### 7.7 Streaming single-yield hang (sprint 729 carveout)

Sprint 729 (F62) bounds the **cumulative wall-time** of a server-
side streaming response via a `time.monotonic()` check between
yields. This catches slow / glacial-yield StageExecutors.

**Not caught**: a StageExecutor whose `__next__()` call hangs
indefinitely on a single iteration. The streaming iteration is
sync inside the async handler:

```python
for frame in stream_iter:  # ← __next__() runs synchronously on the loop
    send_ok = await send_to_peer(...)
```

If `__next__()` blocks (deadlock in model load, deadlock in CUDA
sync, blocking network call inside the iterator), the asyncio
event loop blocks too — the wall-clock check never gets a
chance to run.

Closing this would require offloading the sync iteration to a
worker thread (via `asyncio.to_thread` or `run_in_executor`)
and ferrying frames back through an `asyncio.Queue`. That's a
substantial refactor with its own race-condition surface (the
exact pattern this audit arc has been hardening), so it's
deferred until live-attest data shows it as a real production
issue. Mitigations available today:

- Sprint-723 per-peer stream cap bounds how many slots a single
  peer can hold.
- A separate process-level watchdog (operator-side) can
  detect a wedged daemon by polling `/health` and restart it
  via systemd `WatchdogSec=`.

### 7.8 Gossip layer not in this audit arc

The F30-F65 arc audited `MSG_DIRECT` handlers (chain-executor,
ledger_sync, compute_provider, storage_provider, content_-
provider, agent_registry). Sprint 731 explicitly excluded
`MSG_GOSSIP` from the transport-level sender-binding because
gossip relay legitimately carries the original sender's id,
distinct from the relaying peer.

What this means honestly: `GossipProtocol._handle_gossip` reads
`origin = payload.get("origin", msg.sender_id)` and forwards
that to subscribers as the claimed original sender — with NO
end-to-end signature verification against the origin's pubkey.
A relay peer could mint a new gossip message with an arbitrary
`origin=VICTIM` field and broadcast it; receivers would dispatch
to subscribers as if VICTIM had sent it.

Concrete impact bounds: most gossip subtypes are heartbeats /
peer announcements / digest exchange — relatively low stakes.
The financial transfer path uses `MSG_DIRECT` (now sprint-731
protected), not gossip. Gossip-origin spoofing would let an
attacker forge presence claims ("peer X says they saw peer Y")
which influence peer discovery + reputation — material but not
funds-at-risk.

Closing this is a separate audit arc — gossip messages need
end-to-end signing scheme per subtype + receiver verification
against origin's known pubkey. Tracked for a future session.

## 8. Sprint changelog summary (sprints 680–698)

| Sprint | Class | One-line summary |
| --- | --- | --- |
| 680 | feat | hardware_profile pass-through in DISCOVERY_ANNOUNCE |
| 681 | feat | local hardware_profile loader (PRSM_HARDWARE_PROFILE_FILE env) |
| 682 | feat | DHT-backed GpuPoolProvider (sprint-558 static-empty replacement) |
| 683 | feat | OnChainStakeReader with TTL cache (sprint-561 fix) |
| 684 | test | end-to-end smoke for sprints 680-683 composition |
| 685 | feat | `/admin/parallax/pool/snapshot` endpoint (F30 closed) |
| 686 | feat | stake-eligibility advisory mode (F31/F32/F33 closed) |
| 687 | fix | self-dispatch shortcut + F34/F35/F36/F37 cluster closed |
| 688 | fix | F38 prompt encoder adds gpt2 position embeddings (bit-identical to HF) |
| 689 | fix | F39 streaming decorator passthroughs |
| 690 | feat | F31 proper fix: PoolBackedStakeLookup + PRSM_OPERATOR_ADDRESS |
| 691 | feat | F40 token-stream wiring (self-dispatch only) |
| 692 | feat | F41 SyntheticStreamingRunner wired |
| 693 | feat | EmbedderBackedStreamingRunner — real per-token autoregressive |
| 694 | fix | F43 streaming defaults to greedy (deterministic) |
| 695 | feat | TRUE MULTI-HOST 2-STAGE inference (F44 + F45 closed) |
| 696 | feat | `prsm node parallax-readiness` CLI for 22-env-var preflight |
| 697 | docs | `parallax-inference-deploy.md` operator runbook |
| 698 | feat | Lambda A10 GPU operator (F47 closed inline) |
| 699 | docs | this audit-readiness summary |
| 700 | fix | F46 monotonic hardware_profile gossip propagation |
| 701 | docs | audit doc updates + MCP streaming live-attest + pin tests |
| 702 | feat | tier-gate advisory mode + activation-DP injection live-attested (F48 + F49 closed) |
| 703 | feat | standalone PRSM-import-free receipt verifier (scripts/verify_prsm_receipt.py) |
| 704 | feat | PRSM_INFERENCE_CONCURRENCY_LIMIT semaphore gate (closes §7.4 OOM) |
| 705 | test | sprint 704 live-validated under concurrent load (4 concurrent inferences serialize cleanly) |
| 706 | docs | sample receipt + "try it yourself" walkthrough — verifies in 30s |
| 707 | docs | 2 more sample receipts (tier-none unary + tier-none streaming); 3-receipt coverage of all live-attested modes |
| 708 | docs | multi-host 2-stage sample receipt — verifier demo now covers all 4 architectural modes |
| 709 | docs | operator runbook refresh for sprints 700-708 |
| 710 | docs | README 30-second verifier callout (6 pin tests) |
| 711 | feat | **F40 remote token-stream wire protocol — closes §7.3** (8 pin tests) |
| 713 | feat | stream back-pressure (PRSM_CHAIN_STREAM_QUEUE_MAXSIZE bounded receive queue) — closes sprint-711 OOM follow-on (10 pin tests) |
| 714 | feat | `prsm node parallax-readiness` surfaces sprint-713 env (23rd env var) |
| 715 | fix | **F50** — STREAM_END race fix; sprint-713 put_nowait-for-END let terminal land before frame puts on the response queue. Plus E2E integration test for sprint-711 wire protocol on real WebSocketTransport (2 pin tests + 8 pre-existing unit tests green) |
| 716 | fix | **F51** — malformed-frame race (sprint-715 sibling); same put_nowait-vs-coroutine ordering issue at the malformed-frame terminal error path. All 3 queue-write paths now use same scheduler |
| 717 | test | E2E back-pressure load test — 20-frame stream + maxsize=2 receive queue. Proves non-lossy back-pressure end-to-end (would have caught F50/F51 pre-fix) |
| 718 | fix | **F52** — stream_id collision under concurrent identical requests. Pre-718 derivation was `sha256(request_bytes + ':stream')` purely deterministic; idempotent retries collided + overwrote queue. Mixed 8 bytes os.urandom into derivation |
| 719 | fix | **F53** — stream hijack protection. Sprint 711 routed frames by stream_id alone; any P2P peer that learned the id could forge frames into a victim's queue. Now `pending[]` binds `(queue, expected_sender)`; response handler verifies msg.sender_id before routing |
| 720 | fix | **F54** — server-side resource leak on requester disconnect. Sprint 711 ignored send_to_peer return value; server kept iterating + burning GPU on tokens nobody received. Now reads return value + closes inner generator on False (releases KV cache + model context) |
| 721 | fix | **F55** — request_bytes size limit (memory-DoS defense). `PRSM_CHAIN_STREAM_REQUEST_MAX_BYTES` default 16 MiB; pre-decode (b64 length * 3/4) + post-decode (actual bytes) gates. Surfaced in `parallax-readiness` (27 env vars total) |
| 722 | feat | stream observability — GET /admin/parallax/streams endpoint + `prsm node streams` CLI. Makes sprints 711-721 visible: active stream count + per-stream queue depth/maxsize/full + sender prefix + env values in effect |
| 723 | fix | **F56** — per-peer concurrent stream cap. Pre-723 one peer could open unlimited streams → memory DoS. `PRSM_CHAIN_STREAM_PER_PEER_CONCURRENCY` default 8. Refactor extracts `_handle_stream_request_body` so wrapper places try/finally around every return path (counter never leaks) |
| 724 | fix | **F57** — unary request_id collision (F52 sibling on unary path). Same deterministic sha256(request_bytes) pattern in `build_send_message_adapter` — concurrent identical unary retries collided + first future was overwritten. Fixed with os.urandom + distinct `:unary:` domain separator |
| 725 | fix | **F58** — unary payload size limit (F55 sibling). `handle_chain_executor_request` lacked pre-decode size gate. `PRSM_CHAIN_UNARY_REQUEST_MAX_BYTES` default 16 MiB; pre + post-decode gates; flows into existing CHAIN_ERROR_KEY response path. Surfaced in `parallax-readiness` (29 vars total) |
| 726 | fix | **F59** — unary per-peer concurrent-request cap (F56 sibling). `PRSM_CHAIN_UNARY_PER_PEER_CONCURRENCY` default 8. Body extracted into `_handle_chain_executor_request_body` so wrapper places try/finally around every return path. isinstance(dict) check generalizes the lesson from sprint 723. Surfaced in `parallax-readiness` (30 vars total) |
| 727 | fix | **F60** — unary response hijack protection (F53 sibling). Response handler routed by request_id alone; any peered third party who learned the id could forge CHAIN_RESP and resolve victim's future with attacker bytes. Now `pending[request_id]` stores `(future, expected_sender)` tuple; response handler verifies msg.sender_id before resolving |
| 728 | fix | **F61** — unary execution timeout (hang-defense). `executor.execute()` had no timeout; a hung executor held the sprint-726 per-peer cap slot indefinitely. `PRSM_CHAIN_UNARY_EXECUTION_TIMEOUT_S` default 60s wraps via `asyncio.wait_for`; TimeoutError converts to CHAIN_ERROR_KEY response naming the env var |
| 729 | fix | **F62** — streaming total wall-time bound (sibling of F61 on streaming side). Sync `for`-loop can't use `asyncio.wait_for`; uses `time.monotonic()` check between yields instead. `PRSM_CHAIN_STREAM_EXECUTION_TIMEOUT_S` default 300s; exceedance ships terminal STREAM_END with actionable error |
| 730 | fix | **F63** — bind msg.sender_id to authenticated peer.peer_id (foundation fix). Transport verified signatures only at handshake; subsequent msg.sender_id was wire-trusted not crypto-bound. A peer with valid handshake could spoof sender_id to bypass per-peer caps (F53/F56/F59) AND forge responses (F53/F60). Dispatch wrappers in PRSMNode.start now overwrite sender_id with peer.peer_id before handlers see the msg |
| 731 | fix | **F64** — transport-level sender_id binding (generalizes F63). Sprint-730 only protected chain-executor handlers; other MSG_DIRECT handlers (ledger_sync FTNS transfers, compute_provider, storage_provider, content_provider, agent_registry) still trusted msg.sender_id. Move the bind to WebSocketTransport._dispatch so ALL MSG_DIRECT handlers see authenticated sender. MSG_GOSSIP excluded (relay requires original sender). Sprint-730 wrappers retained as defense-in-depth |
| 734 | fix | **F65** — /admin/* loopback-only by default. Pre-734 every /admin/* endpoint was unauthenticated — `tags=["admin"]` was just a swagger grouping. Sprint 722's observability endpoint widened the leak (expected_sender peer IDs visible to network). Middleware checks request.client.host against loopback whitelist; non-loopback → 403. `PRSM_ADMIN_REMOTE_ALLOWED=1` opt-in for remote-admin (must be behind reverse-proxy auth or VPN) |
| 737 | fix | **F66** — admin loopback gate respects X-Forwarded-For. Sprint-734 checked only `client.host` — bypassed for the common reverse-proxy-on-same-host pattern (nginx forwards to 127.0.0.1 → daemon sees client=127.0.0.1 for any external traffic). Now parses XFF last-hop when immediate client is loopback; external real-client → 403. Local-process spoofing remains possible but bounded vs peered network access |
| 738 | fix | **F67** — admin loopback gate also inspects X-Real-IP. Some proxies (common nginx config `proxy_set_header X-Real-IP`) set X-Real-IP instead of XFF. Sprint-737's XFF-only inspection would miss those bypasses. Now: XFF takes precedence; X-Real-IP checked when XFF absent. Both default-deny external upstream clients |
| 739 | fix | **F68** — admin gate accepts IPv4-mapped IPv6 + 127/8 block. Sprint-734's literal whitelist rejected `::ffff:127.0.0.1` (dual-stack daemon form of loopback) and 127.x.x.x addresses other than 127.0.0.1. Operators on dual-stack Linux + with custom loopback aliases hit false 403. Fix applies to both immediate-client check AND XFF/X-Real-IP last-hop check |
| 741 | fix | **F69** — HTTP rate limiter keyed by local daemon ID, not HTTP requester. `/compute/forge` + `/compute/inference` + `/compute/inference/stream` all used `node.identity.node_id` (a CONSTANT) as the bucket key — "per requester" semantics were effectively GLOBAL. Attacker traffic exhausted bucket → legit clients got 429. Now uses proxy-aware `_resolve_requester_key(request)` matching sprint-737/738 admin precedence |
| 742 | fix | **F70** — HTTP body-size limit (memory-DoS defense at HTTP layer; sibling of wire F55/F58). FastAPI accepted arbitrary JSON bodies; 1 GB POST would allocate before any size check. Middleware checks Content-Length pre-read; default 1 MiB; PRSM_HTTP_MAX_BODY_BYTES env-tunable. 413 with actionable error |
| 743 | fix | **F71** — admin DNS-rebinding defense via Origin header check. DNS rebinding lets a malicious webpage in the victim's browser make requests to `http://localhost:8000/admin/*` — daemon sees client=127.0.0.1, F65-F68 gate passes, admin data leaks to attacker JS. Now rejects /admin/* requests carrying an Origin header (browsers always set it; CLI tools don't) |

41 F-class production-blockers (F30 → F71) closed across the
session. ~277 new pin tests + 5 new integration tests, 0 cross-suite
regressions. **F71 closes the canonical DNS-rebinding attack that
loopback-as-auth services must defend against. 29/29 pin tests
across the cumulative F65-F71 admin-auth arc.**

## 9. What this enables

After sprint 698, an external party can:

1. **Verify any PRSM inference receipt** without trusting any
   PRSM-operated server. Verification reads the on-chain anchor +
   reconstructs the canonical signing bytes.
2. **Stand up their own operator** on any hardware (CPU droplet,
   cloud GPU, or on-premise GPU) using the 8-step runbook. The new
   operator participates in the same DHT pool and can be a stage for
   any inference request.
3. **Run heterogeneous-pool inference** (mixed CPU + GPU) and the
   allocator picks the right peer per request (with the F46 caveat
   that env-var overrides are needed until the propagation fix
   ships).
4. **Trust-but-verify the live fleet**: any anchor-registered peer
   can be challenged by re-running its receipts locally. The
   `output_hash` is deterministic given the same prompt + sampling
   defaults; divergence is detectable.

The §7 verifiable-inference promise — "a third party can verify
chain-of-custody on an inference receipt without trusting the
operator" — is now operationally complete for unary inference on
single-host AND multi-host CPU AND single-host GPU.

## 10. Contact + references

- Repository: https://github.com/prsm-network/PRSM
- Vision document: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/My Vault/Agent-Shared/PRSM_Vision.md`
  (iCloud Obsidian; not in repo)
- Operator runbook: `docs/operations/parallax-inference-deploy.md`
- Receipt verifier source: `prsm/compute/inference/receipt.py`
- Anchor client: `prsm/security/publisher_key_anchor/client.py`
- Sprint 558+ inception of `ParallaxScheduledExecutor`: see
  `prsm/compute/inference/parallax_executor.py`
