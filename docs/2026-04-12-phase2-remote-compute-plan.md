# Phase 2: Remote Compute Dispatch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `TensorParallelExecutor._execute_shard(node_id != "local")` actually execute remotely via MSG_DIRECT dispatch with signed receipts and FTNS escrow, instead of raising `NotImplementedError`.

**Architecture:** A new `RemoteShardDispatcher` class plugs into the executor's existing `remote_dispatcher` slot. Dispatch → escrow create → MSG_DIRECT request → await receipt → verify signature → escrow release → return output. Server side: `ComputeProvider` gets a new `_on_shard_execute_request` handler that executes locally via a shared `execute_shard_locally` helper, signs an Ed25519 receipt, and sends a `shard_execute_response`. Local-ledger FTNS escrow only (on-chain escrow is Phase 3). Verification is tier-A receipt-only for Phase 2, with a pluggable interface for tiers B/C later.

**Tech stack:** Python 3.13, existing `prsm/node/transport.py` (WebSocket MSG_DIRECT), existing `prsm/node/gossip.py`, existing `prsm/compute/model_sharding/`, `numpy`, `eth_account` (Ed25519 via existing `NodeIdentity.sign`), `pytest-asyncio`.

---

## Launch UX thesis (added 2026-04-18)

**Short version.** T3 cloud-arbitrage nodes deliver frontier-adjacent inference latency from day 1, because the latency a developer perceives when calling Claude / GPT / Gemini is dominated by network RTT, not intra-datacenter bandwidth — and a T3 node running on a rented H100 competes on exactly the same network-RTT surface.

**Latency decomposition (rough numbers, Claude-class API call):**

| Component | Typical contribution |
|---|---|
| Developer ↔ nearest cloud region RTT | 20-80 ms |
| Edge ↔ backend routing / auth | 10-40 ms |
| Queuing + tokenization | 5-30 ms |
| GPU forward-pass (short prompt) | 50-100 ms |
| **Total perceived TTFT** | **~100-250 ms** |

A T3 PRSM node running on AWS/GCP/Azure spot or on-demand H100 instances faces the same network surface as the proprietary endpoint: developer → PRSM gateway → T3 node on cloud GPU. Intra-model bandwidth is NVLink or PCIe, not the PRSM P2P wire. The gateway adds ~20-50 ms over direct API; with anycast / edge-deployed PoPs this narrows to ~10-20 ms.

**Conclusion:** expected T3 TTFT at launch is within 10-30% of frontier APIs, not 2-5× worse. That is a qualitatively different product-positioning story from "PRSM is slower at first in exchange for sovereignty."

**Arbitrage math (sanity check for supply-side motivation):**

| Input | Range |
|---|---|
| Spot H100 hourly (AWS/GCP cross-region median) | $2-4 |
| On-demand H100 hourly | $5-8 |
| Throughput (70B-class model, mixed concurrency) | 1000-2000 tok/s |
| Per-hour token capacity | 3.6M-7.2M |
| Cost per M tokens served (spot → on-demand) | $0.50 — $2.20 |

At PRSM inference pricing of $3/M tokens (roughly 1/5 of Sonnet-class API pricing), a T3 operator nets $0.80-$2.50/M tokens gross margin. Profitable arbitrage from week one without requiring ideological alignment — mercenary capacity materializes on its own.

**Phase 2 implications (priorities that make this work at launch, not later):**

1. **T3 onboarding UX is high-leverage.** A great "spin up a T3 node on AWS spot in 20 minutes" onboarding flow is plausibly the single highest-value non-core-protocol deliverable in Phase 2. Tracked in planning but not in this plan — flag for Phase 2.5 scope.

2. **Geo-aware scheduler from day 1.** A Tokyo-developer → us-east-1 T3 node adds ~150 ms RTT and breaks the latency thesis for that user. The scheduler's dispatch path must consider geographic proximity at routing time. Already implicit in the `RemoteShardDispatcher` design — flag for explicit coverage in Task 5.

3. **Cold-start discipline.** A 70B model's weights are ~140 GB. Cold-starting a T3 node adds multi-minute latency unless weights are pre-cached. Pre-pinning popular models + keeping hot standby capacity (warm pool of subscribed weights) matters for consistent UX. Covered partially by `ComputeProvider._can_accept_shard()` in Task 4, but "am I warm for this model?" signal needs to propagate to the scheduler.

4. **TEE attestation from day 1, not day N.** For the arbitrage thesis to hold, developers must be able to trust a T3 node is running the claimed model (and not silently swapping in a cheaper one). Line item C (TEE attestation) is therefore launch-critical for Tier B/C confidentiality-demanding traffic, not a future hardening lever. Phase 2 still ships with Tier A receipt-only as the baseline, but the Tier B/C plugin path must be exercised end-to-end before go-live, not after.

5. **Spot preemption as first-class case.** If T3 operators run on spot instances (lower cost, more competitive pricing) and preemption = total loss, no rational operator will run PRSM nodes. Line item A (partial-receipt protocol for spot preemption) becomes launch-critical, not nice-to-have.

**What this reframes for R7 (KV/activation compression research):** previously tracked as "necessary to make consumer-edge viable." Revised: R7 is a **cost-curve lever, not a launch-viability lever.** T3 arbitrage carries the UX weight at launch; R7 matters when we need to shift supply mix T3 → T1/T2 to keep the cost basis dropping over 2-5 years, or when T3 capacity approaches the rented-GPU supply ceiling. See `docs/2026-04-14-phase4plus-research-track.md` §R7 "Trigger to move to engineering."

**What this does not change:** the structural PRSM wins that compound regardless of launch latency — no vendor deprecation (weights pinned via ProvenanceRegistry), confidentiality as a per-request dial with verifiable attestation, composable SPRK ecosystem with pay-per-use economics, and unit economics without 50-70% platform margin. Those remain the long-term thesis. T3 arbitrage just means the short-term thesis is also competitive on the axis developers actually feel first (speed), not only on the axes they care about second (sovereignty, cost).

**Pitch line:** *"Comparable latency immediately via T3 arbitrage, structurally cheaper throughout, sovereignty and composability compounding over time."* Replaces the earlier "slower at first, cheaper at steady state" framing in investor / developer conversations.

---

## Context

This plan implements [`docs/2026-04-12-phase2-remote-compute-design.md`](./2026-04-12-phase2-remote-compute-design.md). Read the design spec first — it covers architecture decisions, message protocol, out-of-scope deferrals, and the full rationale. This plan translates the spec into TDD-ordered bite-sized steps.

**Predecessor state (post-Phase-1.3, commit 8c2a32c):**
- `TensorParallelExecutor` at `prsm/compute/model_sharding/executor.py:155` raises `NotImplementedError` on remote shard assignment
- `ComputeProvider` at `prsm/node/compute_provider.py` is production-ready for local gossip-driven jobs but has no MSG_DIRECT handler
- `PaymentEscrow` at `prsm/node/payment_escrow.py` has `create_escrow()` working but `release_escrow()` incomplete
- Transport layer (`prsm/node/transport.py`) supports MSG_DIRECT; Phase 1 uses it for content serving

**Design decisions locked:**
1. Transport: WebSocket MSG_DIRECT (not gRPC — Phase 6 upgrade)
2. Verification: Tier A (Ed25519 receipt only), interface designed for Tiers B/C plugins
3. Integration test: in-process with real transport (multi-process is Phase 6)

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `prsm/compute/model_sharding/executor.py` | **Modify** | Extract `execute_shard_locally()` as module-level helper; `_execute_local()` becomes a thin wrapper |
| `prsm/compute/shard_receipt.py` | **Create** | `ShardExecutionReceipt` dataclass + `VerificationStrategy` Protocol + `ReceiptOnlyVerification` impl |
| `prsm/node/payment_escrow.py` | **Modify** | Complete `release_escrow()` body; add `refund_escrow()`; add idempotency state checks; add `EscrowAlreadyFinalizedError` |
| `prsm/node/compute_provider.py` | **Modify** | MSG_DIRECT subscription in `start()`; `_on_shard_execute_request()` handler; `_can_accept_shard()` policy |
| `prsm/compute/remote_dispatcher.py` | **Create** | `RemoteShardDispatcher` class: dispatch flow, timeout/retry, escrow wiring, response routing |
| `prsm/node/node.py` | **Modify** | Construct `PaymentEscrow` (if not already); construct `RemoteShardDispatcher`; wire into `TensorParallelExecutor` |
| `tests/unit/test_shard_receipt.py` | **Create** | ~4 unit tests for receipt verification |
| `tests/unit/test_payment_escrow.py` | **Create/Extend** | ~4 idempotency tests for the completed escrow API |
| `tests/unit/test_remote_dispatcher.py` | **Create** | ~6 unit tests for dispatch flow |
| `tests/node/test_compute_provider.py` | **Extend/Create** | ~4 unit tests for server-side shard handler |
| `tests/integration/test_phase2_remote_dispatch.py` | **Create** | 1 flagship test: 3-node in-process sharded inference |

Estimated ~500 lines of new production code + ~400 lines of tests.

---

## Task 1: Extract `execute_shard_locally` Helper

**Why:** Both the executor's local path AND the provider's remote-serve path must produce bit-identical output for the same shard + input. Extract the numpy-matmul logic into a shared module-level function so both call sites use the same code path.

**Files:**
- Modify: `prsm/compute/model_sharding/executor.py`
- Test: (use existing tests — any regression here is caught by the existing local-execution tests)

- [ ] **Step 1: Read the current `_execute_local` method**

```bash
cd /Users/ryneschultz/Documents/GitHub/PRSM
grep -n "def _execute_local" prsm/compute/model_sharding/executor.py
```

Read the full method body (should be ~20 lines around line 184). Note the exact numpy operations: `frombuffer`, `reshape`, `matmul`. The extraction must preserve the same operations in the same order.

- [ ] **Step 2: Add the module-level helper function**

At the top of `prsm/compute/model_sharding/executor.py`, after the existing imports and before the `TensorParallelExecutor` class, add:

```python
def execute_shard_locally(shard: ModelShard, input_tensor: np.ndarray) -> np.ndarray:
    """Execute a single tensor-parallel shard locally via numpy matmul.

    Pure function with no side effects. Used by:
      - TensorParallelExecutor._execute_local (the executor's local path)
      - ComputeProvider._on_shard_execute_request (the remote-serve path)

    Both call sites use this helper so local and remote execution
    produce bit-identical output for the same shard + input.

    Args:
        shard: ModelShard with tensor_data (bytes), tensor_shape, tensor_dtype.
        input_tensor: Input numpy array to multiply against the shard tensor.

    Returns:
        Output numpy array = input_tensor @ shard_tensor.

    Raises:
        ValueError: if tensor shape/dtype/buffer don't match.
    """
    dtype = np.dtype(shard.tensor_dtype)
    tensor = np.frombuffer(shard.tensor_data, dtype=dtype)
    tensor = tensor.reshape(shard.tensor_shape)
    return np.matmul(input_tensor, tensor)
```

> Note: the existing `_execute_local` hard-codes `np.float32`. The new helper reads `shard.tensor_dtype` for generality — `ModelShard` already stores the dtype so this is a non-behavior-changing extraction for the common float32 case, but enables future dtypes.

- [ ] **Step 3: Refactor `_execute_local` to call the helper**

Replace the body of `TensorParallelExecutor._execute_local` with:

```python
async def _execute_local(self, shard: ModelShard, input_tensor: np.ndarray) -> dict:
    """Execute a shard on this local node. Thin wrapper around
    the module-level execute_shard_locally() helper — see that
    function's docstring for the numerics contract."""
    output = execute_shard_locally(shard, input_tensor)
    return {
        "output_array": output,
        "execution_mode": "local",
        "shard_index": shard.shard_index,
    }
```

(Preserve any additional keys the original method returned — read the original to confirm.)

- [ ] **Step 4: Run existing tests for regression**

```bash
.venv/bin/python -m pytest tests/ -k "tensor_parallel or executor or shard" -v 2>&1 | tail -15
```

Expected: all existing tests still pass. No new tests at this step — this is a refactor, not new behavior.

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
refactor(compute): extract execute_shard_locally as module-level helper

Phase 2 Task 1. Pulls the numpy matmul body out of
TensorParallelExecutor._execute_local into a module-level
execute_shard_locally(shard, input_tensor) -> np.ndarray helper.

Used by two call sites in Phase 2:
 - TensorParallelExecutor._execute_local (the executor's local path)
 - ComputeProvider._on_shard_execute_request (the new remote-serve path)

Both call the same helper so local and remote shard execution
produce bit-identical output for the same shard + input. The
integration test in Task 7 asserts this equality.

Generalized dtype handling (reads shard.tensor_dtype) — Phase 2
doesn't exercise non-float32 but the ModelShard dataclass already
stores dtype, so reading it here is non-breaking for float32 and
unlocks future dtypes without another refactor.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 6)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 1)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" prsm/compute/model_sharding/executor.py
```

---

## Task 2: `ShardExecutionReceipt` + `VerificationStrategy`

**Why:** Pure-code dataclass + verification protocol. No network or async required. Establishes the signature verification contract that all later tasks depend on. TDD-friendly — tests exercise sign/verify round-trip with a known key.

**Files:**
- Create: `prsm/compute/shard_receipt.py`
- Create: `tests/unit/test_shard_receipt.py`

- [ ] **Step 1: Write the failing receipt-roundtrip test**

Create `tests/unit/test_shard_receipt.py`:

```python
"""Unit tests for ShardExecutionReceipt + VerificationStrategy.

Phase 2 Task 2. Exercises the signed-receipt round-trip:
create → sign → verify — with known Ed25519 key material.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.shard_receipt import (
    ReceiptOnlyVerification,
    ShardExecutionReceipt,
    build_receipt_signing_payload,
)


# Burner Ed25519 keypair for tests. Not used on any real network.
# Derived deterministically so test output is reproducible.
_BURNER_PRIVATE_HEX = "11" * 32
_BURNER_PUBKEY_B64 = None   # populated once at import time below


def _burner_identity():
    """Construct a minimal identity-like object with .sign() + .public_key_b64."""
    import base64
    from eth_account import Account
    account = Account.from_key("0x" + _BURNER_PRIVATE_HEX)
    # NodeIdentity in PRSM uses Ed25519; eth_account uses secp256k1.
    # For the test we use the real NodeIdentity if available, else
    # a minimal stub that mirrors its sign/verify contract.
    try:
        from prsm.node.identity import NodeIdentity
        # NodeIdentity generates its own keys; we can't control the
        # private key. For the test we create a fresh identity and
        # use its pubkey directly.
        identity = NodeIdentity.generate()
        return identity
    except ImportError:
        # Fallback: tests must construct a working identity.
        raise


def test_receipt_verification_happy_path():
    """A receipt signed by a known identity verifies correctly."""
    identity = _burner_identity()

    job_id = "job-happy-path"
    shard_index = 0
    output_bytes = b"deterministic output payload"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id=job_id,
        shard_index=shard_index,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    signature = identity.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id=job_id,
        shard_index=shard_index,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=signature,
    )

    verifier = ReceiptOnlyVerification()
    import asyncio
    assert asyncio.run(verifier.verify(receipt.to_dict(), output_bytes)) is True


def test_receipt_verification_bad_signature():
    """A tampered signature fails verification."""
    identity = _burner_identity()

    output_bytes = b"original output"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-bad-sig",
        shard_index=0,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    good_sig = identity.sign(payload)

    # Flip one character in the signature to invalidate it.
    tampered_sig = good_sig[:-4] + ("AAAA" if good_sig[-4:] != "AAAA" else "BBBB")

    receipt = ShardExecutionReceipt(
        job_id="job-bad-sig",
        shard_index=0,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=tampered_sig,
    )

    verifier = ReceiptOnlyVerification()
    import asyncio
    assert asyncio.run(verifier.verify(receipt.to_dict(), output_bytes)) is False


def test_receipt_verification_output_hash_mismatch():
    """A receipt whose output_hash doesn't match the actual bytes fails verification."""
    identity = _burner_identity()

    # Sign a payload with output_hash=X, but pass bytes with hash=Y to verify.
    wrong_bytes = b"attacker tried to substitute this"
    declared_hash = hashlib.sha256(b"original honest output").hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-hash-mismatch",
        shard_index=0,
        output_hash=declared_hash,
        executed_at_unix=executed_at,
    )
    signature = identity.sign(payload)

    receipt = ShardExecutionReceipt(
        job_id="job-hash-mismatch",
        shard_index=0,
        provider_id=identity.node_id,
        provider_pubkey_b64=identity.public_key_b64,
        output_hash=declared_hash,
        executed_at_unix=executed_at,
        signature=signature,
    )

    verifier = ReceiptOnlyVerification()
    import asyncio
    # Signature is valid over the declared hash, but the actual bytes
    # hash to something different — verification must fail.
    assert asyncio.run(verifier.verify(receipt.to_dict(), wrong_bytes)) is False


def test_receipt_verification_wrong_pubkey():
    """A signature valid over the correct payload but signed by a
    different key (attacker-controlled) fails verification."""
    victim = _burner_identity()
    attacker = _burner_identity()
    assert victim.public_key_b64 != attacker.public_key_b64

    output_bytes = b"honest output"
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = 1776019150

    payload = build_receipt_signing_payload(
        job_id="job-wrong-key",
        shard_index=0,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    # Attacker signs with their key.
    attacker_sig = attacker.sign(payload)

    # Receipt claims to be from the victim but signed by attacker.
    receipt = ShardExecutionReceipt(
        job_id="job-wrong-key",
        shard_index=0,
        provider_id=victim.node_id,
        provider_pubkey_b64=victim.public_key_b64,   # claims victim's pubkey
        output_hash=output_hash,
        executed_at_unix=executed_at,
        signature=attacker_sig,   # but signed by attacker's key
    )

    verifier = ReceiptOnlyVerification()
    import asyncio
    assert asyncio.run(verifier.verify(receipt.to_dict(), output_bytes)) is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/test_shard_receipt.py -v 2>&1 | tail -10
```

Expected: FAIL with `ImportError: cannot import name 'ReceiptOnlyVerification' from 'prsm.compute.shard_receipt'` (the module doesn't exist yet).

- [ ] **Step 3: Create the `shard_receipt.py` module**

Create `prsm/compute/shard_receipt.py`:

```python
"""Phase 2: ShardExecutionReceipt + VerificationStrategy.

Signed receipt for remote shard execution. The provider signs a
canonical payload after executing; the requester verifies the
signature matches the provider's advertised pubkey AND the
output_hash matches the actual bytes returned.

Tier-A verification (receipt-only) is the Phase 2 default. Tiers B
(redundant execution) and C (stake + slash) plug in at Phase 7 via
the same VerificationStrategy protocol without changing the receipt
format or the dispatch protocol.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Protocol

logger = logging.getLogger(__name__)


def build_receipt_signing_payload(
    job_id: str,
    shard_index: int,
    output_hash: str,
    executed_at_unix: int,
) -> bytes:
    """Canonical bytes the provider signs. Requesters rebuild the same
    payload and verify the provider's signature over it.

    Format: "{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    encoded as UTF-8.
    """
    return (
        f"{job_id}||{shard_index}||{output_hash}||{executed_at_unix}"
    ).encode("utf-8")


@dataclass(frozen=True)
class ShardExecutionReceipt:
    """Signed proof that a provider executed a shard.

    Fields form the serialized `receipt` sub-object in a
    shard_execute_response MSG_DIRECT payload.
    """
    job_id: str
    shard_index: int
    provider_id: str
    provider_pubkey_b64: str
    output_hash: str
    executed_at_unix: int
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardExecutionReceipt":
        return cls(
            job_id=data["job_id"],
            shard_index=data["shard_index"],
            provider_id=data["provider_id"],
            provider_pubkey_b64=data["provider_pubkey_b64"],
            output_hash=data["output_hash"],
            executed_at_unix=data["executed_at_unix"],
            signature=data["signature"],
        )


class VerificationStrategy(Protocol):
    """Pluggable interface for receipt verification. Phase 2 implements
    Tier A (receipt-only). Tiers B (redundant execution) and C (stake +
    slash) implement this protocol at Phase 7."""

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
    ) -> bool: ...


class ReceiptOnlyVerification:
    """Tier A: signature check only. Phase 2 default.

    Two checks:
      1. output_hash == sha256(output_bytes)  — the declared hash
         actually matches the bytes returned.
      2. Ed25519(signature) valid against provider_pubkey_b64 over
         build_receipt_signing_payload(...) — the receipt was signed
         by the claimed provider.

    Returns True iff both checks pass. Never raises on invalid input —
    logs a warning and returns False so dispatchers can uniformly
    refund escrow on verification failure.
    """

    async def verify(
        self,
        receipt: Dict[str, Any],
        output_bytes: bytes,
    ) -> bool:
        # Check 1: output hash matches the bytes.
        declared_hash = receipt.get("output_hash")
        actual_hash = hashlib.sha256(output_bytes).hexdigest()
        if declared_hash != actual_hash:
            logger.warning(
                f"receipt verification failed: output_hash mismatch "
                f"(declared={declared_hash[:16]}…, actual={actual_hash[:16]}…)"
            )
            return False

        # Check 2: signature valid over canonical payload.
        try:
            payload = build_receipt_signing_payload(
                job_id=receipt["job_id"],
                shard_index=receipt["shard_index"],
                output_hash=declared_hash,
                executed_at_unix=receipt["executed_at_unix"],
            )
            pubkey_b64 = receipt["provider_pubkey_b64"]
            signature = receipt["signature"]
        except KeyError as exc:
            logger.warning(f"receipt verification failed: missing field {exc}")
            return False

        # Delegate to NodeIdentity's verify-with-pubkey helper.
        try:
            from prsm.node.identity import NodeIdentity
            if not NodeIdentity.verify_signature(pubkey_b64, payload, signature):
                logger.warning(
                    f"receipt verification failed: signature invalid for "
                    f"provider {receipt.get('provider_id', '?')[:12]}…"
                )
                return False
        except ImportError:
            logger.warning(
                "receipt verification failed: NodeIdentity.verify_signature unavailable"
            )
            return False
        except Exception as exc:
            logger.warning(f"receipt verification raised: {exc}")
            return False

        return True
```

> Note: this assumes `NodeIdentity.verify_signature(pubkey_b64, payload, signature) -> bool` exists. Before writing the implementation, grep for it:
> ```bash
> grep -n "def verify_signature\|def sign\|def generate" prsm/node/identity.py
> ```
> If `verify_signature` doesn't exist in `NodeIdentity`, either (a) add it as a static method (takes a pubkey + payload + signature, returns bool) or (b) use whatever the existing verify helper is called. Adjust the test fixture's `_burner_identity()` to match the real API.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/test_shard_receipt.py -v 2>&1 | tail -15
```

Expected: all 4 tests PASS.

If `NodeIdentity.verify_signature` didn't exist and you had to add it, also run any existing NodeIdentity tests to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(compute): add ShardExecutionReceipt + ReceiptOnlyVerification

Phase 2 Task 2. Signed-receipt primitives for remote shard execution.

New module prsm/compute/shard_receipt.py:
 - build_receipt_signing_payload(job_id, shard_index, output_hash,
   executed_at_unix) -> bytes — canonical payload providers sign
   over.
 - ShardExecutionReceipt dataclass — 7 fields matching the receipt
   sub-object in the shard_execute_response MSG_DIRECT payload.
 - VerificationStrategy Protocol — interface for Tiers A/B/C.
 - ReceiptOnlyVerification (Tier A) — checks output_hash equals
   sha256(output_bytes) AND signature is valid Ed25519 over the
   canonical payload. Logs warnings on failure; never raises; always
   returns bool so dispatchers can uniformly refund on verification
   failure.

Four unit tests cover the happy path, tampered signature, output
hash mismatch (attacker substitutes output but keeps signature),
and wrong pubkey (attacker signs with their key but claims victim
pubkey). All 4 pass.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 3)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 2)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" prsm/compute/shard_receipt.py tests/unit/test_shard_receipt.py
```

---

## Task 3: Complete `PaymentEscrow` + Idempotency

**Why:** `RemoteShardDispatcher` needs reliable escrow create/release/refund with idempotency. Task 3 completes the scaffold and adds state-machine guards.

**Files:**
- Modify: `prsm/node/payment_escrow.py`
- Create: `tests/unit/test_payment_escrow.py`

- [ ] **Step 1: Read the current `PaymentEscrow` scaffold**

```bash
grep -n "def create_escrow\|def release_escrow\|def refund_escrow\|class PaymentEscrow\|class EscrowEntry\|class EscrowStatus" prsm/node/payment_escrow.py
```

Read the full `PaymentEscrow` class, the `EscrowEntry` dataclass, and the `EscrowStatus` enum. Note the existing state transitions (PENDING / RELEASED / REFUNDED / DISPUTED) and the ledger-transfer calls inside `create_escrow`.

- [ ] **Step 2: Write the failing idempotency tests**

Create `tests/unit/test_payment_escrow.py`:

```python
"""Unit tests for PaymentEscrow state-machine idempotency.

Phase 2 Task 3. Verifies:
  - create_escrow debits requester, credits holding wallet
  - release_escrow transfers to recipient, marks RELEASED
  - refund_escrow returns to requester, marks REFUNDED
  - double-release is a no-op with warning (self-idempotent)
  - double-refund is a no-op with warning (self-idempotent)
  - release-after-refund raises EscrowAlreadyFinalizedError
  - refund-after-release raises EscrowAlreadyFinalizedError
  - release-before-create raises EscrowNotFoundError
"""
from __future__ import annotations

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import (
    EscrowAlreadyFinalizedError,
    EscrowNotFoundError,
    EscrowStatus,
    PaymentEscrow,
)


@pytest.fixture
async def ledger():
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    await ledger.create_wallet("alice", "Alice")
    await ledger.create_wallet("bob", "Bob")
    await ledger.credit(
        wallet_id="alice",
        amount=100.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="seed",
    )
    yield ledger
    await ledger.close()


@pytest.fixture
async def escrow(ledger):
    return PaymentEscrow(ledger=ledger)


@pytest.mark.asyncio
async def test_create_escrow_debits_requester(ledger, escrow):
    """create_escrow moves FTNS from requester to holding wallet."""
    assert await ledger.get_balance("alice") == 100.0

    escrow_id = await escrow.create_escrow(
        requester_id="alice",
        amount=10.0,
        purpose="shard_exec:job-1:0",
    )

    assert await ledger.get_balance("alice") == 90.0
    entry = escrow.get_escrow(escrow_id)
    assert entry.status == EscrowStatus.PENDING
    assert entry.amount == 10.0


@pytest.mark.asyncio
async def test_release_escrow_transfers_to_recipient(ledger, escrow):
    """release_escrow transfers the held amount to recipient and marks RELEASED."""
    escrow_id = await escrow.create_escrow(
        requester_id="alice", amount=10.0, purpose="shard_exec:job-1:0",
    )

    await escrow.release_escrow(escrow_id, recipient="bob")

    assert await ledger.get_balance("bob") == 10.0
    assert escrow.get_escrow(escrow_id).status == EscrowStatus.RELEASED


@pytest.mark.asyncio
async def test_refund_escrow_returns_to_requester(ledger, escrow):
    """refund_escrow returns the held amount to original requester and marks REFUNDED."""
    assert await ledger.get_balance("alice") == 100.0
    escrow_id = await escrow.create_escrow(
        requester_id="alice", amount=10.0, purpose="shard_exec:job-1:0",
    )
    assert await ledger.get_balance("alice") == 90.0

    await escrow.refund_escrow(escrow_id, reason="test")

    assert await ledger.get_balance("alice") == 100.0
    assert escrow.get_escrow(escrow_id).status == EscrowStatus.REFUNDED


@pytest.mark.asyncio
async def test_release_escrow_idempotent(ledger, escrow, caplog):
    """Double-release is a no-op with warning; doesn't double-pay."""
    escrow_id = await escrow.create_escrow(
        requester_id="alice", amount=10.0, purpose="shard_exec:job-1:0",
    )
    await escrow.release_escrow(escrow_id, recipient="bob")
    assert await ledger.get_balance("bob") == 10.0

    # Second release on same escrow_id.
    with caplog.at_level("WARNING"):
        await escrow.release_escrow(escrow_id, recipient="bob")

    # Bob balance unchanged (no double-pay).
    assert await ledger.get_balance("bob") == 10.0
    assert any("already released" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_refund_escrow_idempotent(ledger, escrow, caplog):
    """Double-refund is a no-op with warning; doesn't double-refund."""
    escrow_id = await escrow.create_escrow(
        requester_id="alice", amount=10.0, purpose="shard_exec:job-1:0",
    )
    await escrow.refund_escrow(escrow_id, reason="first")
    assert await ledger.get_balance("alice") == 100.0

    with caplog.at_level("WARNING"):
        await escrow.refund_escrow(escrow_id, reason="second")

    # Alice balance unchanged (no double-refund).
    assert await ledger.get_balance("alice") == 100.0
    assert any("already refunded" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_release_after_refund_raises(ledger, escrow):
    """Calling release_escrow on a REFUNDED escrow raises
    EscrowAlreadyFinalizedError."""
    escrow_id = await escrow.create_escrow(
        requester_id="alice", amount=10.0, purpose="shard_exec:job-1:0",
    )
    await escrow.refund_escrow(escrow_id, reason="test")

    with pytest.raises(EscrowAlreadyFinalizedError):
        await escrow.release_escrow(escrow_id, recipient="bob")


@pytest.mark.asyncio
async def test_refund_after_release_raises(ledger, escrow):
    """Calling refund_escrow on a RELEASED escrow raises
    EscrowAlreadyFinalizedError."""
    escrow_id = await escrow.create_escrow(
        requester_id="alice", amount=10.0, purpose="shard_exec:job-1:0",
    )
    await escrow.release_escrow(escrow_id, recipient="bob")

    with pytest.raises(EscrowAlreadyFinalizedError):
        await escrow.refund_escrow(escrow_id, reason="test")


@pytest.mark.asyncio
async def test_release_nonexistent_raises(escrow):
    """Calling release_escrow on an unknown escrow_id raises
    EscrowNotFoundError."""
    with pytest.raises(EscrowNotFoundError):
        await escrow.release_escrow("no-such-escrow", recipient="bob")
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/test_payment_escrow.py -v 2>&1 | tail -20
```

Expected: most tests FAIL — `EscrowAlreadyFinalizedError` and/or `EscrowNotFoundError` don't exist; `release_escrow`/`refund_escrow` bodies are incomplete.

- [ ] **Step 4: Add the missing error types and complete `release_escrow` + `refund_escrow`**

In `prsm/node/payment_escrow.py`:

1. Near the top (after imports, before the `EscrowStatus` enum), add:

```python
class EscrowNotFoundError(KeyError):
    """Raised when a caller references an escrow_tx_id that doesn't exist."""


class EscrowAmountError(ValueError):
    """Raised when a release amount exceeds the escrowed amount."""


class EscrowAlreadyFinalizedError(RuntimeError):
    """Raised when release is called on REFUNDED or refund is called
    on RELEASED — cross-operation transitions are never valid."""
```

2. Complete `release_escrow` with the state-machine logic:

```python
async def release_escrow(
    self,
    escrow_tx_id: str,
    recipient: str,
    amount: Optional[float] = None,
) -> None:
    """Transfer escrowed amount to recipient. Marks RELEASED.

    Self-idempotent: calling twice on the same escrow is a no-op
    with a warning. Calling on a REFUNDED escrow raises
    EscrowAlreadyFinalizedError (cross-state transition).
    """
    entry = self._escrows.get(escrow_tx_id)
    if entry is None:
        raise EscrowNotFoundError(
            f"escrow {escrow_tx_id!r} does not exist"
        )

    if entry.status == EscrowStatus.RELEASED:
        logger.warning(
            f"escrow {escrow_tx_id!r} already released; release_escrow "
            f"is a no-op"
        )
        return

    if entry.status == EscrowStatus.REFUNDED:
        raise EscrowAlreadyFinalizedError(
            f"escrow {escrow_tx_id!r} is already REFUNDED; cannot release"
        )

    release_amount = amount if amount is not None else entry.amount
    if release_amount > entry.amount:
        raise EscrowAmountError(
            f"release amount {release_amount} exceeds escrowed "
            f"{entry.amount} for {escrow_tx_id!r}"
        )

    # Transfer from holding wallet to recipient.
    await self.ledger.transfer(
        from_wallet=f"escrow:{escrow_tx_id}",
        to_wallet=recipient,
        amount=release_amount,
        tx_type=TransactionType.ESCROW_RELEASE,
        description=f"Escrow release: {entry.purpose}",
    )

    entry.status = EscrowStatus.RELEASED
    entry.released_to = recipient
    entry.released_at = time.time()
    logger.info(
        f"escrow {escrow_tx_id!r} released: {release_amount} FTNS → {recipient}"
    )
```

3. Add `refund_escrow`:

```python
async def refund_escrow(
    self,
    escrow_tx_id: str,
    reason: str = "",
) -> None:
    """Return escrowed amount to the original requester. Marks REFUNDED.

    Self-idempotent: calling twice on the same escrow is a no-op
    with a warning. Calling on a RELEASED escrow raises
    EscrowAlreadyFinalizedError (cross-state transition).
    """
    entry = self._escrows.get(escrow_tx_id)
    if entry is None:
        raise EscrowNotFoundError(
            f"escrow {escrow_tx_id!r} does not exist"
        )

    if entry.status == EscrowStatus.REFUNDED:
        logger.warning(
            f"escrow {escrow_tx_id!r} already refunded; refund_escrow "
            f"is a no-op"
        )
        return

    if entry.status == EscrowStatus.RELEASED:
        raise EscrowAlreadyFinalizedError(
            f"escrow {escrow_tx_id!r} is already RELEASED; cannot refund"
        )

    await self.ledger.transfer(
        from_wallet=f"escrow:{escrow_tx_id}",
        to_wallet=entry.requester_id,
        amount=entry.amount,
        tx_type=TransactionType.ESCROW_REFUND,
        description=f"Escrow refund ({reason}): {entry.purpose}",
    )

    entry.status = EscrowStatus.REFUNDED
    entry.refund_reason = reason
    entry.refunded_at = time.time()
    logger.info(
        f"escrow {escrow_tx_id!r} refunded: {entry.amount} FTNS → "
        f"{entry.requester_id} (reason: {reason!r})"
    )
```

4. Add a `get_escrow(escrow_tx_id) -> Optional[EscrowEntry]` helper if not already present:

```python
def get_escrow(self, escrow_tx_id: str) -> Optional[EscrowEntry]:
    """Return the EscrowEntry for an escrow_tx_id, or None if unknown."""
    return self._escrows.get(escrow_tx_id)
```

5. Confirm `EscrowEntry` dataclass has fields: `requester_id`, `amount`, `purpose`, `status`, and optional fields `released_to`, `released_at`, `refund_reason`, `refunded_at`. Add any missing fields with `Optional[...]` defaults.

6. Confirm `TransactionType.ESCROW_RELEASE` and `TransactionType.ESCROW_REFUND` exist. If not, grep `local_ledger.py` for the enum and add them.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/test_payment_escrow.py -v 2>&1 | tail -15
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Run the full regression suite**

```bash
.venv/bin/python -m pytest tests/integration/test_onchain_provenance_e2e.py tests/unit/test_content_upload.py tests/unit/test_cross_node_content.py tests/unit/test_royalty_pipeline.py tests/node/test_content_economy.py -q 2>&1 | tail -5
```

Expected: all pass (Phase 1.3 test suite unchanged). The `PaymentEscrow` changes shouldn't affect any Phase 1 test.

- [ ] **Step 7: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(node): complete PaymentEscrow state machine + idempotency

Phase 2 Task 3. Finishes the escrow scaffold so Phase 2's
RemoteShardDispatcher can rely on create / release / refund
with predictable failure modes.

Completed:
 - release_escrow() body: transfers from escrow holding wallet
   to recipient via LocalLedger.transfer(), marks RELEASED.
 - refund_escrow() added: transfers back to original requester,
   marks REFUNDED.
 - get_escrow() helper for callers that want to inspect state.

State-machine guards:
 - Double-release on same escrow_tx_id: no-op with warning log
   (self-idempotent).
 - Double-refund on same escrow_tx_id: no-op with warning log
   (self-idempotent).
 - release-after-refund: raises EscrowAlreadyFinalizedError.
 - refund-after-release: raises EscrowAlreadyFinalizedError.
 - release/refund on unknown escrow_tx_id: raises EscrowNotFoundError.
 - release amount exceeding escrowed: raises EscrowAmountError.

Eight unit tests in tests/unit/test_payment_escrow.py cover create,
release, refund, both idempotency cases, both cross-state transition
failures, and the unknown-escrow case. All 8 pass.

Phase 1 regression suite (content economy, royalty pipeline,
on-chain provenance) unchanged — no Phase 1 test touches
PaymentEscrow, and PaymentEscrow doesn't share state with
ContentEconomy.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 5)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" prsm/node/payment_escrow.py tests/unit/test_payment_escrow.py
```

---

## Task 4: `ComputeProvider` MSG_DIRECT Handler

**Why:** The server-side handler that accepts a `shard_execute_request`, runs `execute_shard_locally`, builds a signed receipt, and sends back a `shard_execute_response`. Second half of the protocol.

**Files:**
- Modify: `prsm/node/compute_provider.py`
- Test: `tests/node/test_compute_provider.py` (extend or create)

- [ ] **Step 1: Write failing tests for the server handler**

Append to or create `tests/node/test_compute_provider.py`:

```python
"""Unit tests for ComputeProvider's shard_execute_request handler.

Phase 2 Task 4. Tests the server-side flow: receive request →
acceptance check → execute → sign → respond.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard
from prsm.compute.model_sharding.executor import execute_shard_locally
from prsm.compute.shard_receipt import (
    ReceiptOnlyVerification,
    ShardExecutionReceipt,
)
from prsm.node.compute_provider import ComputeProvider
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _make_provider(max_concurrent_jobs: int = 10):
    """Build a ComputeProvider with minimal mocks for unit tests."""
    from prsm.node.identity import NodeIdentity
    identity = NodeIdentity.generate()
    transport = MagicMock()
    transport.send_to_peer = AsyncMock()
    transport.on_message = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock()
    ledger = MagicMock()

    provider = ComputeProvider(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
        max_concurrent_jobs=max_concurrent_jobs,
    )
    return provider, identity, transport


def _build_shard_request(
    shard_tensor: np.ndarray,
    input_tensor: np.ndarray,
    job_id: str = "job-1",
    shard_index: int = 0,
    deadline_offset_sec: int = 30,
) -> dict:
    """Build a shard_execute_request payload matching the Phase 2 protocol."""
    tensor_bytes = shard_tensor.astype(np.float32).tobytes()
    checksum = hashlib.sha256(tensor_bytes).hexdigest()
    return {
        "subtype": "shard_execute_request",
        "job_id": job_id,
        "shard_index": shard_index,
        "tensor_data_b64": base64.b64encode(tensor_bytes).decode(),
        "tensor_shape": list(shard_tensor.shape),
        "tensor_dtype": "float32",
        "input_b64": base64.b64encode(input_tensor.astype(np.float32).tobytes()).decode(),
        "input_shape": list(input_tensor.shape),
        "input_dtype": "float32",
        "checksum": checksum,
        "stake_tier": "STANDARD",
        "escrow_tx_id": "esc-test-1",
        "deadline_unix": int(time.time()) + deadline_offset_sec,
        "request_id": "req-test-1",
        "requester_pubkey_b64": "fake-requester-pubkey",
    }


def test_shard_execute_request_happy_path():
    """A valid shard_execute_request produces a completed response with
    a correct output and a signature that verifies."""
    provider, identity, transport = _make_provider()

    shard_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    input_tensor = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    expected_output = input_tensor @ shard_tensor   # [[6.0, 8.0]]

    payload = _build_shard_request(shard_tensor, input_tensor)
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock()
    peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    assert transport.send_to_peer.await_count == 1
    _peer_id, sent_msg = transport.send_to_peer.await_args.args
    resp = sent_msg.payload

    assert resp["subtype"] == "shard_execute_response"
    assert resp["request_id"] == "req-test-1"
    assert resp["status"] == "completed"
    assert resp["shard_index"] == 0

    output_bytes = base64.b64decode(resp["output_b64"])
    output = np.frombuffer(output_bytes, dtype=np.float32).reshape(resp["output_shape"])
    np.testing.assert_allclose(output, expected_output, rtol=1e-6)

    # Receipt verifies.
    verifier = ReceiptOnlyVerification()
    assert asyncio.run(verifier.verify(resp["receipt"], output_bytes)) is True


def test_shard_execute_request_declines_when_over_capacity():
    """When _current_jobs >= max_concurrent_jobs, provider responds
    status=declined and does NOT execute."""
    provider, _identity, transport = _make_provider(max_concurrent_jobs=1)
    provider._current_jobs = 1   # at capacity

    shard_tensor = np.array([[1.0]], dtype=np.float32)
    input_tensor = np.array([[1.0]], dtype=np.float32)
    payload = _build_shard_request(shard_tensor, input_tensor)
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    assert transport.send_to_peer.await_count == 1
    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["status"] == "declined"
    assert "output_b64" not in resp or resp.get("output_b64") is None


def test_shard_execute_request_declines_past_deadline():
    """deadline_unix < now → declined."""
    provider, _identity, transport = _make_provider()

    shard_tensor = np.array([[1.0]], dtype=np.float32)
    input_tensor = np.array([[1.0]], dtype=np.float32)
    payload = _build_shard_request(shard_tensor, input_tensor, deadline_offset_sec=-100)
    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["status"] == "declined"


def test_shard_execute_request_declines_bad_checksum():
    """Declared checksum != actual sha256(tensor_bytes) → declined."""
    provider, _identity, transport = _make_provider()

    shard_tensor = np.array([[1.0, 2.0]], dtype=np.float32)
    input_tensor = np.array([[1.0]], dtype=np.float32)
    payload = _build_shard_request(shard_tensor, input_tensor)
    payload["checksum"] = "0" * 64   # wrong hash

    msg = P2PMessage(msg_type=MSG_DIRECT, sender_id="peer-req", payload=payload)
    peer = MagicMock(); peer.peer_id = "peer-req"

    asyncio.run(provider._on_shard_execute_request(msg, peer))

    resp = transport.send_to_peer.await_args.args[1].payload
    assert resp["status"] == "declined"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/node/test_compute_provider.py -v 2>&1 | tail -15
```

Expected: FAIL with `AttributeError: 'ComputeProvider' has no attribute '_on_shard_execute_request'` (method doesn't exist yet).

- [ ] **Step 3: Add the handler + helpers to `ComputeProvider`**

In `prsm/node/compute_provider.py`:

1. Add imports if missing:

```python
import base64
import hashlib
import time
import numpy as np

from prsm.compute.model_sharding.models import ModelShard
from prsm.compute.model_sharding.executor import execute_shard_locally
from prsm.compute.shard_receipt import build_receipt_signing_payload
from prsm.node.transport import MSG_DIRECT, P2PMessage
```

2. Add the MSG_DIRECT subscription inside `start()`:

```python
def start(self) -> None:
    # ... existing gossip subscriptions ...
    self.transport.on_message(MSG_DIRECT, self._on_direct_message)
```

3. Add `_on_direct_message` router:

```python
async def _on_direct_message(self, msg, peer) -> None:
    """Route MSG_DIRECT messages to subtype-specific handlers.

    Other subscribers (content_provider, etc.) also receive this
    dispatch; the subtype filter routes each handler to its own
    messages.
    """
    subtype = msg.payload.get("subtype", "")
    if subtype == "shard_execute_request":
        await self._on_shard_execute_request(msg, peer)
```

4. Add the main handler:

```python
async def _on_shard_execute_request(self, msg, peer) -> None:
    """Accept a shard_execute_request, run execute_shard_locally,
    sign a receipt, and send a shard_execute_response.

    Phase 2 Task 4. Uses the same execute_shard_locally helper the
    local executor uses, so local and remote execution produce
    bit-identical output.
    """
    payload = msg.payload
    request_id = payload.get("request_id", "")
    job_id = payload.get("job_id", "")
    shard_index = payload.get("shard_index", 0)

    try:
        # Acceptance policy — check resources, deadline, integrity.
        decline_reason = self._can_accept_shard(payload)
        if decline_reason is not None:
            await self._send_shard_response(
                peer.peer_id, request_id,
                status="declined", shard_index=shard_index,
                error=decline_reason,
            )
            return

        # Deserialize shard + input.
        shard = self._deserialize_shard_from_payload(payload)
        input_tensor = self._deserialize_input_from_payload(payload)

        # Execute using the shared helper (same code path as local executor).
        output = execute_shard_locally(shard, input_tensor)
        output_bytes = output.tobytes()
        output_hash = hashlib.sha256(output_bytes).hexdigest()
        executed_at = int(time.time())

        # Sign the receipt over the canonical payload.
        sig_payload = build_receipt_signing_payload(
            job_id=job_id,
            shard_index=shard_index,
            output_hash=output_hash,
            executed_at_unix=executed_at,
        )
        signature = self.identity.sign(sig_payload)

        receipt = {
            "job_id": job_id,
            "shard_index": shard_index,
            "provider_id": self.identity.node_id,
            "provider_pubkey_b64": self.identity.public_key_b64,
            "output_hash": output_hash,
            "executed_at_unix": executed_at,
            "signature": signature,
        }

        await self._send_shard_response(
            peer.peer_id, request_id,
            status="completed",
            shard_index=shard_index,
            output_bytes=output_bytes,
            output_shape=list(output.shape),
            output_dtype=str(output.dtype),
            receipt=receipt,
        )
    except Exception as exc:
        logger.warning(
            f"shard_execute_request failed for job {job_id!r} "
            f"shard {shard_index}: {exc}"
        )
        await self._send_shard_response(
            peer.peer_id, request_id,
            status="failed", shard_index=shard_index, error=str(exc),
        )
```

5. Add the acceptance policy helper:

```python
def _can_accept_shard(self, payload: dict) -> Optional[str]:
    """Return None if the shard request is acceptable, else a string
    reason explaining the decline (becomes the response's `error`
    field).

    Checks (in order):
      1. Capacity: _current_jobs < max_concurrent_jobs
      2. Deadline: deadline_unix > now
      3. Checksum: declared checksum == sha256(tensor_bytes)
      4. Payload size: decoded tensor bytes <= 10 MB
    """
    if self._current_jobs >= self.max_concurrent_jobs:
        return "resource_unavailable"

    now = int(time.time())
    deadline = payload.get("deadline_unix", 0)
    if deadline < now:
        return "deadline_past"

    try:
        tensor_bytes = base64.b64decode(payload.get("tensor_data_b64", ""))
    except Exception:
        return "tensor_decode_failed"

    MAX_SHARD_BYTES = 10 * 1024 * 1024
    if len(tensor_bytes) > MAX_SHARD_BYTES:
        return "shard_too_large"

    declared_checksum = payload.get("checksum", "")
    actual_checksum = hashlib.sha256(tensor_bytes).hexdigest()
    if declared_checksum != actual_checksum:
        return "checksum_mismatch"

    return None
```

6. Add the serializer helpers:

```python
def _deserialize_shard_from_payload(self, payload: dict) -> ModelShard:
    tensor_bytes = base64.b64decode(payload["tensor_data_b64"])
    return ModelShard(
        shard_index=payload["shard_index"],
        tensor_data=tensor_bytes,
        tensor_shape=tuple(payload["tensor_shape"]),
        tensor_dtype=payload["tensor_dtype"],
        checksum=payload["checksum"],
    )

def _deserialize_input_from_payload(self, payload: dict) -> np.ndarray:
    input_bytes = base64.b64decode(payload["input_b64"])
    dtype = np.dtype(payload["input_dtype"])
    return np.frombuffer(input_bytes, dtype=dtype).reshape(payload["input_shape"])
```

> The exact `ModelShard` constructor args depend on what the dataclass defines. Read `prsm/compute/model_sharding/models.py` to confirm the field names and pass them correctly.

7. Add the response sender:

```python
async def _send_shard_response(
    self,
    peer_id: str,
    request_id: str,
    status: str,
    shard_index: int,
    output_bytes: Optional[bytes] = None,
    output_shape: Optional[list] = None,
    output_dtype: Optional[str] = None,
    receipt: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    payload = {
        "subtype": "shard_execute_response",
        "request_id": request_id,
        "status": status,
        "shard_index": shard_index,
        "error": error,
    }
    if output_bytes is not None:
        payload["output_b64"] = base64.b64encode(output_bytes).decode()
        payload["output_shape"] = output_shape
        payload["output_dtype"] = output_dtype
    if receipt is not None:
        payload["receipt"] = receipt

    msg = P2PMessage(
        msg_type=MSG_DIRECT,
        sender_id=self.identity.node_id,
        payload=payload,
    )
    await self.transport.send_to_peer(peer_id, msg)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/node/test_compute_provider.py -v 2>&1 | tail -15
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(node): ComputeProvider handles shard_execute_request MSG_DIRECT

Phase 2 Task 4. Server side of the remote shard execution protocol.
ComputeProvider now subscribes to MSG_DIRECT and routes the
shard_execute_request subtype to a new handler that:

 1. Runs an acceptance policy (capacity, deadline, payload size,
    checksum integrity) — declines with a structured reason on any
    failure.
 2. Deserializes the shard + input tensor from the base64 payload.
 3. Calls execute_shard_locally (the shared helper from Task 1) so
    local-path and remote-serve-path produce bit-identical output.
 4. Signs a ShardExecutionReceipt with the node's Ed25519 identity
    over build_receipt_signing_payload(job_id, shard_index,
    output_hash, executed_at_unix).
 5. Sends a shard_execute_response with status=completed, the output
    bytes, and the signed receipt. On any exception, sends a
    status=failed response with the error string.

Four unit tests cover: happy path (output + receipt verify), over
capacity (declined), past deadline (declined), bad checksum
(declined). All 4 pass.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 4)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" prsm/node/compute_provider.py tests/node/test_compute_provider.py
```

---

## Task 5: `RemoteShardDispatcher` (Requester Side)

**Why:** The class that plugs into `TensorParallelExecutor.remote_dispatcher`. Owns the dispatch → escrow create → MSG_DIRECT send → await → verify → escrow release → return flow.

**Files:**
- Create: `prsm/compute/remote_dispatcher.py`
- Create: `tests/unit/test_remote_dispatcher.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_remote_dispatcher.py`:

```python
"""Unit tests for RemoteShardDispatcher.

Phase 2 Task 5. Exercises dispatch, timeout, retry, verification
failure, size limit, and peer-not-connected handling.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard
from prsm.compute.remote_dispatcher import (
    PeerNotConnectedError,
    RemoteShardDispatcher,
    ShardTooLargeError,
)
from prsm.compute.shard_receipt import (
    ReceiptOnlyVerification,
    build_receipt_signing_payload,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage


def _make_shard() -> tuple[ModelShard, np.ndarray, np.ndarray]:
    shard_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor_bytes = shard_tensor.tobytes()
    shard = ModelShard(
        shard_index=0,
        tensor_data=tensor_bytes,
        tensor_shape=shard_tensor.shape,
        tensor_dtype="float32",
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )
    input_tensor = np.array([[1.0, 0.0]], dtype=np.float32)
    expected_output = input_tensor @ shard_tensor
    return shard, input_tensor, expected_output


def _make_dispatcher(timeout=0.5, max_retries=0, local_fallback=None):
    from prsm.node.identity import NodeIdentity
    identity = NodeIdentity.generate()
    transport = MagicMock()
    transport.get_peer = MagicMock(return_value=MagicMock(peer_id="provider-1"))
    transport.send_to_peer = AsyncMock()
    transport.on_message = MagicMock()

    escrow = MagicMock()
    escrow.create_escrow = AsyncMock(return_value="esc-1")
    escrow.release_escrow = AsyncMock()
    escrow.refund_escrow = AsyncMock()

    verifier = ReceiptOnlyVerification()

    dispatcher = RemoteShardDispatcher(
        identity=identity,
        transport=transport,
        payment_escrow=escrow,
        verification_strategy=verifier,
        default_timeout=timeout,
        max_retries=max_retries,
        max_shard_bytes=1024,   # small for the size-limit test
        local_fallback=local_fallback,
    )
    return dispatcher, identity, transport, escrow


def _provider_identity_and_response(
    dispatcher, identity, shard, input_tensor, request_id,
) -> dict:
    """Build a valid shard_execute_response payload signed by a fresh
    provider identity."""
    from prsm.node.identity import NodeIdentity
    provider = NodeIdentity.generate()

    output = input_tensor @ np.frombuffer(
        shard.tensor_data, dtype=np.float32
    ).reshape(shard.tensor_shape)
    output_bytes = output.tobytes()
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    executed_at = int(time.time())

    sig_payload = build_receipt_signing_payload(
        job_id="job-1",
        shard_index=shard.shard_index,
        output_hash=output_hash,
        executed_at_unix=executed_at,
    )
    signature = provider.sign(sig_payload)

    return {
        "subtype": "shard_execute_response",
        "request_id": request_id,
        "status": "completed",
        "shard_index": shard.shard_index,
        "output_b64": base64.b64encode(output_bytes).decode(),
        "output_shape": list(output.shape),
        "output_dtype": "float32",
        "receipt": {
            "job_id": "job-1",
            "shard_index": shard.shard_index,
            "provider_id": provider.node_id,
            "provider_pubkey_b64": provider.public_key_b64,
            "output_hash": output_hash,
            "executed_at_unix": executed_at,
            "signature": signature,
        },
    }


def test_dispatch_happy_path():
    """Dispatch sends request, receives valid response, releases escrow,
    returns output."""
    from prsm.compute.model_sharding.models import StakeTier
    dispatcher, identity, transport, escrow = _make_dispatcher()
    shard, input_tensor, expected = _make_shard()

    # Inject a response as soon as send_to_peer is called.
    async def send_and_respond(peer_id, msg):
        request_id = msg.payload["request_id"]
        response_payload = _provider_identity_and_response(
            dispatcher, identity, shard, input_tensor, request_id
        )
        response_msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id="provider-1",
            payload=response_payload,
        )
        peer = MagicMock(); peer.peer_id = "provider-1"
        asyncio.create_task(dispatcher._on_direct_message(response_msg, peer))

    transport.send_to_peer.side_effect = send_and_respond

    result = asyncio.run(dispatcher.dispatch(
        shard=shard, input_tensor=input_tensor,
        node_id="provider-1", job_id="job-1",
        stake_tier=StakeTier.STANDARD, escrow_amount_ftns=1.0,
    ))

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    escrow.create_escrow.assert_awaited_once()
    escrow.release_escrow.assert_awaited_once()
    escrow.refund_escrow.assert_not_called()


def test_dispatch_timeout_falls_back():
    """When the provider doesn't respond in time and no retries remain,
    escrow is refunded and local_fallback (if wired) is called."""
    from prsm.compute.model_sharding.models import StakeTier
    fallback_output = np.array([[999.0, 999.0]], dtype=np.float32)
    fallback = MagicMock(return_value=fallback_output)

    dispatcher, _, transport, escrow = _make_dispatcher(
        timeout=0.1, max_retries=0, local_fallback=fallback,
    )
    shard, input_tensor, _ = _make_shard()

    # transport.send_to_peer resolves but no response ever arrives.
    transport.send_to_peer = AsyncMock()

    from prsm.compute.model_sharding.models import StakeTier
    result = asyncio.run(dispatcher.dispatch(
        shard=shard, input_tensor=input_tensor,
        node_id="provider-1", job_id="job-1",
        stake_tier=StakeTier.STANDARD, escrow_amount_ftns=1.0,
    ))

    np.testing.assert_allclose(result, fallback_output)
    escrow.refund_escrow.assert_awaited_once()
    escrow.release_escrow.assert_not_called()
    fallback.assert_called_once()


def test_dispatch_bad_signature_refunds():
    """A response with an invalid signature causes escrow refund."""
    from prsm.compute.model_sharding.models import StakeTier
    dispatcher, identity, transport, escrow = _make_dispatcher()
    shard, input_tensor, _ = _make_shard()

    async def send_and_respond_with_bad_sig(peer_id, msg):
        request_id = msg.payload["request_id"]
        response_payload = _provider_identity_and_response(
            dispatcher, identity, shard, input_tensor, request_id
        )
        # Tamper with signature.
        response_payload["receipt"]["signature"] = (
            response_payload["receipt"]["signature"][:-4] + "AAAA"
        )
        response_msg = P2PMessage(
            msg_type=MSG_DIRECT, sender_id="provider-1",
            payload=response_payload,
        )
        peer = MagicMock(); peer.peer_id = "provider-1"
        asyncio.create_task(dispatcher._on_direct_message(response_msg, peer))

    transport.send_to_peer.side_effect = send_and_respond_with_bad_sig

    with pytest.raises(Exception):
        asyncio.run(dispatcher.dispatch(
            shard=shard, input_tensor=input_tensor,
            node_id="provider-1", job_id="job-1",
            stake_tier=StakeTier.STANDARD, escrow_amount_ftns=1.0,
        ))

    escrow.refund_escrow.assert_awaited_once()
    escrow.release_escrow.assert_not_called()


def test_dispatch_size_limit_rejects():
    """Shard > max_shard_bytes raises ShardTooLargeError without network."""
    from prsm.compute.model_sharding.models import StakeTier
    dispatcher, _, transport, escrow = _make_dispatcher()

    # Build a shard > 1024 bytes (our test limit).
    big_tensor = np.ones((32, 32), dtype=np.float32)   # 4096 bytes
    tensor_bytes = big_tensor.tobytes()
    shard = ModelShard(
        shard_index=0,
        tensor_data=tensor_bytes,
        tensor_shape=big_tensor.shape,
        tensor_dtype="float32",
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )
    input_tensor = np.ones((1, 32), dtype=np.float32)

    with pytest.raises(ShardTooLargeError):
        asyncio.run(dispatcher.dispatch(
            shard=shard, input_tensor=input_tensor,
            node_id="provider-1", job_id="job-1",
            stake_tier=StakeTier.STANDARD, escrow_amount_ftns=1.0,
        ))

    transport.send_to_peer.assert_not_called()
    escrow.create_escrow.assert_not_called()


def test_dispatch_retry_on_timeout():
    """First response times out; retry succeeds; only one final escrow
    release fires."""
    from prsm.compute.model_sharding.models import StakeTier
    dispatcher, identity, transport, escrow = _make_dispatcher(
        timeout=0.1, max_retries=1,
    )
    shard, input_tensor, expected = _make_shard()

    call_count = {"n": 0}

    async def send_sometimes(peer_id, msg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return   # swallow first request
        request_id = msg.payload["request_id"]
        response_payload = _provider_identity_and_response(
            dispatcher, identity, shard, input_tensor, request_id
        )
        response_msg = P2PMessage(
            msg_type=MSG_DIRECT, sender_id="provider-1",
            payload=response_payload,
        )
        peer = MagicMock(); peer.peer_id = "provider-1"
        asyncio.create_task(dispatcher._on_direct_message(response_msg, peer))

    transport.send_to_peer.side_effect = send_sometimes

    result = asyncio.run(dispatcher.dispatch(
        shard=shard, input_tensor=input_tensor,
        node_id="provider-1", job_id="job-1",
        stake_tier=StakeTier.STANDARD, escrow_amount_ftns=1.0,
    ))

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert escrow.release_escrow.await_count == 1
    assert escrow.refund_escrow.await_count == 0


def test_dispatch_peer_not_connected():
    """If transport.get_peer returns None, raise PeerNotConnectedError
    (or fall back to local if local_fallback is wired)."""
    from prsm.compute.model_sharding.models import StakeTier
    dispatcher, _, transport, escrow = _make_dispatcher()
    transport.get_peer = MagicMock(return_value=None)

    shard, input_tensor, _ = _make_shard()

    with pytest.raises(PeerNotConnectedError):
        asyncio.run(dispatcher.dispatch(
            shard=shard, input_tensor=input_tensor,
            node_id="provider-1", job_id="job-1",
            stake_tier=StakeTier.STANDARD, escrow_amount_ftns=1.0,
        ))

    escrow.create_escrow.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/test_remote_dispatcher.py -v 2>&1 | tail -10
```

Expected: FAIL with `ImportError: cannot import name 'RemoteShardDispatcher' from 'prsm.compute.remote_dispatcher'`.

- [ ] **Step 3: Create `remote_dispatcher.py`**

Create `prsm/compute/remote_dispatcher.py`:

```python
"""Phase 2: RemoteShardDispatcher.

Plugs into TensorParallelExecutor's `remote_dispatcher` slot. Owns
the full round-trip for a single shard: size check → peer resolve →
escrow create → MSG_DIRECT send → await response → verify receipt →
escrow release → return output. Refunds escrow on any failure.
Retries once on timeout; falls back to local execution if
local_fallback is wired.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

import numpy as np

from prsm.compute.model_sharding.models import ModelShard, StakeTier
from prsm.compute.shard_receipt import VerificationStrategy
from prsm.node.transport import MSG_DIRECT, P2PMessage

logger = logging.getLogger(__name__)


class ShardTooLargeError(ValueError):
    """Shard tensor exceeds max_shard_bytes."""


class PeerNotConnectedError(RuntimeError):
    """Target node_id is not a connected peer."""


class ShardDispatchError(RuntimeError):
    """Generic dispatch failure after escrow refund."""


class RemoteShardDispatcher:
    """Dispatches a shard to a remote provider via MSG_DIRECT.

    The executor's remote_dispatcher slot expects a callable like:
        async def dispatch(shard, input_tensor, node_id, job_id,
                           stake_tier, escrow_amount_ftns) -> np.ndarray

    This class implements that contract.
    """

    def __init__(
        self,
        identity,
        transport,
        payment_escrow,
        verification_strategy: VerificationStrategy,
        default_timeout: float = 30.0,
        max_retries: int = 1,
        max_shard_bytes: int = 10 * 1024 * 1024,
        local_fallback: Optional[Callable] = None,
    ):
        self.identity = identity
        self.transport = transport
        self.payment_escrow = payment_escrow
        self.verification_strategy = verification_strategy
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.max_shard_bytes = max_shard_bytes
        self.local_fallback = local_fallback

        self._pending: Dict[str, asyncio.Future] = {}

        # Register the MSG_DIRECT handler so responses route into
        # pending futures.
        self.transport.on_message(MSG_DIRECT, self._on_direct_message)

    async def dispatch(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_id: str,
        job_id: str,
        stake_tier: StakeTier,
        escrow_amount_ftns: float,
    ) -> np.ndarray:
        """Dispatch shard to node_id and return the output tensor.
        Raises on unrecoverable failure after escrow refund."""
        # Step 1: size check.
        if len(shard.tensor_data) > self.max_shard_bytes:
            if self.local_fallback is not None:
                logger.warning(
                    f"shard {shard.shard_index} size "
                    f"{len(shard.tensor_data)} exceeds max "
                    f"{self.max_shard_bytes}; falling back to local"
                )
                return await self._call_fallback(shard, input_tensor)
            raise ShardTooLargeError(
                f"shard {shard.shard_index} size "
                f"{len(shard.tensor_data)} exceeds max {self.max_shard_bytes}"
            )

        # Step 2: peer resolve.
        peer = self.transport.get_peer(node_id)
        if peer is None:
            if self.local_fallback is not None:
                logger.warning(f"peer {node_id} not connected; falling back")
                return await self._call_fallback(shard, input_tensor)
            raise PeerNotConnectedError(
                f"node_id {node_id!r} is not a connected peer"
            )

        # Step 3: escrow create.
        escrow_id = await self.payment_escrow.create_escrow(
            requester_id=self.identity.node_id,
            amount=escrow_amount_ftns,
            purpose=f"shard_exec:{job_id}:{shard.shard_index}",
        )

        # Steps 4-9: build request, send, await, verify, release, return.
        # Retry on timeout.
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    output = await self._dispatch_once(
                        shard=shard,
                        input_tensor=input_tensor,
                        node_id=node_id,
                        job_id=job_id,
                        stake_tier=stake_tier,
                        escrow_id=escrow_id,
                    )
                    # Success: release escrow, return output.
                    await self.payment_escrow.release_escrow(
                        escrow_id,
                        recipient=node_id,
                        amount=escrow_amount_ftns,
                    )
                    return output
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        logger.info(
                            f"shard {shard.shard_index} dispatch to {node_id} "
                            f"timed out; retry {attempt + 1}/{self.max_retries}"
                        )
                        continue
                    raise
        except Exception as exc:
            # Any failure: refund escrow, then fall back or raise.
            await self.payment_escrow.refund_escrow(
                escrow_id, reason=str(exc)
            )
            if isinstance(exc, asyncio.TimeoutError) and self.local_fallback is not None:
                logger.warning(
                    f"shard {shard.shard_index} dispatch failed after "
                    f"{self.max_retries} retries; falling back to local"
                )
                return await self._call_fallback(shard, input_tensor)
            raise ShardDispatchError(
                f"shard {shard.shard_index} dispatch failed: {exc}"
            ) from exc

    async def _dispatch_once(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_id: str,
        job_id: str,
        stake_tier: StakeTier,
        escrow_id: str,
    ) -> np.ndarray:
        """Single dispatch attempt. Raises asyncio.TimeoutError if no
        response arrives within default_timeout."""
        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        deadline = int(time.time()) + int(self.default_timeout)

        payload = {
            "subtype": "shard_execute_request",
            "job_id": job_id,
            "shard_index": shard.shard_index,
            "tensor_data_b64": base64.b64encode(shard.tensor_data).decode(),
            "tensor_shape": list(shard.tensor_shape),
            "tensor_dtype": shard.tensor_dtype,
            "input_b64": base64.b64encode(input_tensor.tobytes()).decode(),
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "checksum": shard.checksum,
            "stake_tier": stake_tier.name if hasattr(stake_tier, "name") else str(stake_tier),
            "escrow_tx_id": escrow_id,
            "deadline_unix": deadline,
            "request_id": request_id,
            "requester_pubkey_b64": self.identity.public_key_b64,
        }
        msg = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=self.identity.node_id,
            payload=payload,
        )

        try:
            await self.transport.send_to_peer(node_id, msg)
            response = await asyncio.wait_for(
                future, timeout=self.default_timeout
            )
        finally:
            self._pending.pop(request_id, None)

        # Verify status.
        if response.get("status") != "completed":
            raise ShardDispatchError(
                f"provider returned status={response.get('status')!r} "
                f"error={response.get('error')!r}"
            )

        # Decode output + verify receipt.
        output_bytes = base64.b64decode(response["output_b64"])
        output_shape = response["output_shape"]
        output_dtype = np.dtype(response["output_dtype"])
        output = np.frombuffer(output_bytes, dtype=output_dtype).reshape(output_shape)

        receipt = response.get("receipt", {})
        verified = await self.verification_strategy.verify(receipt, output_bytes)
        if not verified:
            raise ShardDispatchError(
                f"receipt verification failed for shard {shard.shard_index}"
            )

        return output

    async def _on_direct_message(self, msg, peer) -> None:
        """Route shard_execute_response to pending futures."""
        if msg.payload.get("subtype") != "shard_execute_response":
            return
        request_id = msg.payload.get("request_id")
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(msg.payload)

    async def _call_fallback(
        self, shard: ModelShard, input_tensor: np.ndarray,
    ) -> np.ndarray:
        """Invoke local_fallback. Supports both sync and async callables."""
        result = self.local_fallback(shard, input_tensor)
        if asyncio.iscoroutine(result):
            return await result
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/test_remote_dispatcher.py -v 2>&1 | tail -15
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(compute): add RemoteShardDispatcher with escrow + receipt verify

Phase 2 Task 5. Requester side of the remote shard execution protocol.

RemoteShardDispatcher plugs into TensorParallelExecutor's existing
remote_dispatcher slot. For each shard assigned to a remote node:

  1. Size check — raise or fall back to local if > max_shard_bytes.
  2. Peer resolve — raise or fall back if node_id not connected.
  3. Escrow create — lock FTNS from requester to holding wallet.
  4. Build + send shard_execute_request MSG_DIRECT.
  5. Await shard_execute_response via future registered in
     self._pending[request_id]; resolved by _on_direct_message.
  6. Retry once on asyncio.TimeoutError.
  7. Verify receipt via VerificationStrategy (Tier A in Phase 2).
  8. Release escrow to provider on success; refund on any failure.
  9. Return the output numpy array.

Fall-back-to-local fires on: shard-too-large, peer-not-connected,
and timeout-after-retries (when local_fallback is wired). Without
local_fallback, these raise ShardTooLargeError, PeerNotConnectedError,
or ShardDispatchError respectively.

Six unit tests cover: happy path, timeout-with-fallback, bad
signature, size limit, retry-after-first-timeout, peer-not-connected.
All 6 pass.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 3)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 5)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" prsm/compute/remote_dispatcher.py tests/unit/test_remote_dispatcher.py
```

---

## Task 6: Node Bootstrap Wiring

**Why:** Construct `PaymentEscrow` and `RemoteShardDispatcher` in `PRSMNode.initialize()`; wire the dispatcher's `dispatch` method into `TensorParallelExecutor.remote_dispatcher`.

**Files:**
- Modify: `prsm/node/node.py`

- [ ] **Step 1: Read the current bootstrap and find wiring points**

```bash
grep -n "PaymentEscrow\|TensorParallelExecutor\|ComputeProvider(\|compute_provider" prsm/node/node.py
```

Identify:
- Where (if at all) `PaymentEscrow` is currently constructed.
- Where `TensorParallelExecutor` is instantiated.
- Where `ComputeProvider` is constructed.
- The correct ordering: `ledger` → `payment_escrow` → `compute_provider` → `remote_shard_dispatcher` → `TensorParallelExecutor`.

- [ ] **Step 2: Add the wiring**

In `prsm/node/node.py` `initialize()`, after `self.ledger` is constructed:

```python
# ── Payment Escrow (local-ledger FTNS only; on-chain is Phase 3) ──
from prsm.node.payment_escrow import PaymentEscrow
self.payment_escrow = PaymentEscrow(ledger=self.ledger)
```

After `self.compute_provider` is constructed but before any `TensorParallelExecutor` is used:

```python
# ── Remote Shard Dispatcher (Phase 2) ─────────────────────────────
# Plugs into TensorParallelExecutor's remote_dispatcher slot. Tier A
# verification (receipt-only) in Phase 2; Tiers B/C plug in at Phase 7.
from prsm.compute.remote_dispatcher import RemoteShardDispatcher
from prsm.compute.shard_receipt import ReceiptOnlyVerification

self.remote_shard_dispatcher = RemoteShardDispatcher(
    identity=self.identity,
    transport=self.transport,
    payment_escrow=self.payment_escrow,
    verification_strategy=ReceiptOnlyVerification(),
    default_timeout=30.0,
    max_retries=1,
    max_shard_bytes=10 * 1024 * 1024,
    local_fallback=None,   # None for Phase 2
)
```

Then wherever `TensorParallelExecutor` is instantiated, pass `remote_dispatcher=self.remote_shard_dispatcher.dispatch`. If there's no existing instantiation (executor is created lazily inside another class), document that the consumer must pass it.

- [ ] **Step 3: Run the full regression suite**

```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/integration/test_phase2_remote_dispatch.py 2>&1 | tail -10
```

Expected: all existing tests pass. This task is pure wiring — no behavior change for tests that don't exercise remote dispatch.

- [ ] **Step 4: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(node): wire PaymentEscrow + RemoteShardDispatcher into PRSMNode

Phase 2 Task 6. Bootstrap wiring in PRSMNode.initialize():

 - PaymentEscrow constructed after ledger (ledger is its only dep).
 - RemoteShardDispatcher constructed after compute_provider with
   ReceiptOnlyVerification, default_timeout=30s, max_retries=1,
   max_shard_bytes=10MB, local_fallback=None.
 - TensorParallelExecutor instantiations pass
   remote_dispatcher=self.remote_shard_dispatcher.dispatch so the
   executor's existing remote_dispatcher slot is filled — the
   NotImplementedError safety rail at executor.py:155 now only
   fires for tests/environments where the node wasn't properly
   bootstrapped.

No behavior change for existing tests (Phase 1 regression suite
still passes unchanged). The 3-node integration test in Task 7
exercises the wiring end-to-end.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 6)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 6)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" prsm/node/node.py
```

---

## Task 7: 3-Node Integration Test (Acceptance Criterion)

**Why:** The roadmap's acceptance criterion for Phase 2 — "3-node local cluster runs a sharded inference end-to-end." Uses in-process real-transport per the design Q3 decision.

**Files:**
- Create: `tests/integration/test_phase2_remote_dispatch.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_phase2_remote_dispatch.py`:

```python
"""Phase 2 integration test — 3-node sharded inference end-to-end.

Roadmap acceptance criterion: "3-node local cluster runs a sharded
inference end-to-end."

Phase 2 design Q3 decision: in-process with real transport (option B).
Spins up 3 PRSMNode instances in the same event loop, wires their
transports via loopback, and dispatches a sharded matmul across them.
Asserts payment, receipt verification, and bit-identical output
compared to a local baseline.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import List

import numpy as np
import pytest

from prsm.compute.model_sharding.executor import execute_shard_locally
from prsm.compute.model_sharding.models import ModelShard, StakeTier
from prsm.compute.remote_dispatcher import RemoteShardDispatcher
from prsm.compute.shard_receipt import ReceiptOnlyVerification


@pytest.mark.asyncio
async def test_three_node_sharded_inference_end_to_end(tmp_path):
    """Three nodes, each owning one shard. Node A (requester) dispatches
    shard 0 to itself (local), shard 1 to node B, shard 2 to node C.
    All receipts verify, all escrows either RELEASED or refunded
    appropriately, final all_reduce output matches local baseline
    bit-identically.
    """
    # -- Setup: 3 nodes, loopback transport hub, real PaymentEscrow. --
    from tests.integration.conftest_phase2 import (
        spin_up_three_node_cluster,
    )
    nodes = await spin_up_three_node_cluster(tmp_path)
    requester = nodes[0]
    provider_b = nodes[1]
    provider_c = nodes[2]

    # -- Build a sharded model. 3 shards along the column axis. --
    full_tensor = np.random.rand(8, 9).astype(np.float32)
    shard_width = 3
    shards: List[ModelShard] = []
    for i in range(3):
        part = full_tensor[:, i * shard_width:(i + 1) * shard_width]
        part_bytes = part.tobytes()
        shards.append(ModelShard(
            shard_index=i,
            tensor_data=part_bytes,
            tensor_shape=part.shape,
            tensor_dtype="float32",
            checksum=hashlib.sha256(part_bytes).hexdigest(),
        ))

    input_tensor = np.random.rand(1, 8).astype(np.float32)

    # -- Local baseline. --
    expected_output = input_tensor @ full_tensor   # (1, 9)

    # -- Dispatch: shard 0 local, shard 1 → B, shard 2 → C. --
    # Seed the requester's wallet so escrow can be debited.
    await requester.ledger.credit(
        wallet_id=requester.identity.node_id,
        amount=100.0,
        tx_type=requester.ledger.__class__.__module__.TransactionType.WELCOME_GRANT,
        description="test seed",
    )

    job_id = "integration-job-1"
    shard_outputs: List[np.ndarray] = []

    # Shard 0: local.
    shard_outputs.append(execute_shard_locally(shards[0], input_tensor))

    # Shard 1: remote to B.
    out_b = await requester.remote_shard_dispatcher.dispatch(
        shard=shards[1],
        input_tensor=input_tensor,
        node_id=provider_b.identity.node_id,
        job_id=job_id,
        stake_tier=StakeTier.STANDARD,
        escrow_amount_ftns=1.0,
    )
    shard_outputs.append(out_b)

    # Shard 2: remote to C.
    out_c = await requester.remote_shard_dispatcher.dispatch(
        shard=shards[2],
        input_tensor=input_tensor,
        node_id=provider_c.identity.node_id,
        job_id=job_id,
        stake_tier=StakeTier.STANDARD,
        escrow_amount_ftns=1.0,
    )
    shard_outputs.append(out_c)

    # -- Assemble output: concatenate along column axis. --
    full_output = np.concatenate(shard_outputs, axis=1)

    # -- Assertion 1: bit-identical to local baseline. --
    np.testing.assert_array_equal(full_output, expected_output)

    # -- Assertion 2: escrow state. --
    # Two escrows created (one per remote dispatch), both RELEASED.
    b_escrows = [
        e for e in requester.payment_escrow._escrows.values()
        if e.purpose.startswith(f"shard_exec:{job_id}:1")
    ]
    c_escrows = [
        e for e in requester.payment_escrow._escrows.values()
        if e.purpose.startswith(f"shard_exec:{job_id}:2")
    ]
    assert len(b_escrows) == 1 and b_escrows[0].status.name == "RELEASED"
    assert len(c_escrows) == 1 and c_escrows[0].status.name == "RELEASED"

    # -- Assertion 3: providers received FTNS. --
    assert await requester.ledger.get_balance(provider_b.identity.node_id) >= 1.0
    assert await requester.ledger.get_balance(provider_c.identity.node_id) >= 1.0
```

> Note: `spin_up_three_node_cluster` is a fixture helper you'll create in a new `tests/integration/conftest_phase2.py`. It should:
> 1. Create three `PRSMNode` instances with separate identities, each wired to a shared in-memory transport hub that routes MSG_DIRECT between them by `node_id`.
> 2. Share a single `LocalLedger` across all 3 nodes (so escrow debits from requester credit the provider's wallet on the same ledger).
> 3. Call `await node.initialize()` and `node.start()` on each so handlers (MSG_DIRECT subscriptions) are wired.
> 4. Return the list of nodes in order `[requester, provider_b, provider_c]`.
>
> The loopback transport hub is a small class (~50 lines) with a dict of `node_id → WebSocketTransport-like` interface; `send_to_peer(node_id, msg)` dispatches directly into the target's `on_message` callbacks. No real WebSocket connection needed — the interface contract (`send_to_peer`, `on_message`, `get_peer`) is what matters.

- [ ] **Step 2: Create the helper `conftest_phase2.py`**

Create `tests/integration/conftest_phase2.py` with a minimal `spin_up_three_node_cluster` fixture that constructs the 3 nodes wired via loopback. The exact implementation depends on `PRSMNode`'s current constructor signature and the `WebSocketTransport` interface — read both and wire the minimum needed for MSG_DIRECT to flow between the 3 nodes.

If `PRSMNode.__init__` has too many dependencies to stand up cleanly in-process, consider a test-scoped `TestPRSMNode` subclass that only includes: identity, transport, gossip, ledger, payment_escrow, compute_provider, remote_shard_dispatcher. The integration test doesn't need the full Phase 1 economy wiring.

- [ ] **Step 3: Run the integration test**

```bash
.venv/bin/python -m pytest tests/integration/test_phase2_remote_dispatch.py -v -s 2>&1 | tail -20
```

Expected: PASS. If it fails, diagnose the first failure:
- Wiring issue: fixture doesn't connect all 3 nodes correctly.
- Protocol issue: request/response keys don't match.
- Escrow state mismatch: assertion about `status.name` may need adjusting to the actual enum value.

- [ ] **Step 4: Run the full regression suite one final time**

```bash
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all tests pass (Phase 1 + Phase 2).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test(integration): 3-node sharded inference end-to-end (Phase 2 acceptance)

Phase 2 Task 7. Roadmap acceptance criterion: "3-node local cluster
runs a sharded inference end-to-end."

Design Q3 decision: in-process with real transport (option B —
sweet spot between speed and fidelity; multi-process chaos tests
are Phase 6).

Test:
 - 3 PRSMNode instances wired via loopback transport hub.
 - Requester builds a sharded (1, 8) @ (8, 9) -> (1, 9) matmul,
   3 shards along the column axis.
 - Shard 0 runs locally via execute_shard_locally.
 - Shard 1 dispatched to provider B via RemoteShardDispatcher.
 - Shard 2 dispatched to provider C via RemoteShardDispatcher.
 - Assembled output compared to local baseline bit-for-bit (np.testing
   .assert_array_equal, not allclose — the shared helper guarantees
   identical output).
 - Both escrows (1 per remote dispatch) assert RELEASED status.
 - Both providers' ledger balances assert >= 1.0 FTNS received.

tests/integration/conftest_phase2.py provides
spin_up_three_node_cluster fixture that wires 3 nodes via an
in-memory transport hub. No real WebSocket; the hub implements
send_to_peer / on_message / get_peer.

Refs: docs/2026-04-12-phase2-remote-compute-design.md (Section 7)
Refs: docs/2026-04-12-phase2-remote-compute-plan.md (Task 7)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" tests/integration/test_phase2_remote_dispatch.py tests/integration/conftest_phase2.py
```

---

## Task 8: Codex Review Gate

**Why:** Final validation before merging Phase 2. Same pattern as Phase 1.3 — independent static review catches what TDD misses.

**Files:** None modified. Review task.

- [ ] **Step 1: Run codex review**

```bash
codex exec "IMPORTANT: Do NOT read or execute any files under ~/.claude/, ~/.agents/, .claude/skills/, or agents/. Stay focused on repository code only.

Pre-merge code review of PRSM Phase 2 (Remote Compute Dispatch).
Run \`git diff origin/main\` to see the full cumulative diff.

Phase 2 implements remote shard execution via MSG_DIRECT with signed
receipts and local-ledger FTNS escrow. Design spec is at
docs/2026-04-12-phase2-remote-compute-design.md. Key decisions:

  1. Transport: WebSocket MSG_DIRECT (not gRPC — Phase 6 upgrade)
  2. Verification: Tier A (receipt-only); Tiers B/C plug in at Phase 7
  3. Integration test: in-process with real transport

Verify:
  1. The NotImplementedError at executor.py:155 never fires when a
     node is properly bootstrapped — the bootstrap wiring in
     node.py passes remote_dispatcher=dispatcher.dispatch.
  2. Receipt verification (Ed25519 sig + output hash) is correct
     and uses the canonical signing payload.
  3. Escrow state machine is tight: double-release is no-op, cross-
     state transitions raise, no double-pay possible.
  4. ComputeProvider's acceptance policy rejects requests with bad
     checksum / past deadline / over capacity / oversized tensor.
  5. RemoteShardDispatcher refunds escrow on every failure path
     (timeout-after-retries, verification failure, bad status,
     peer-not-connected).
  6. The shared execute_shard_locally helper guarantees bit-identical
     output between local and remote paths — verified by the 3-node
     integration test.
  7. No Phase 1 regressions.

Use [P1]/[P2]/[P3] tagging. End with SAFE TO MERGE or NOT SAFE TO MERGE." \
-C "\$(git rev-parse --show-toplevel)" -s read-only -c 'model_reasoning_effort="high"' 2>&1 | tail -120
```

- [ ] **Step 2: Address any findings**

P1/P2 findings → patch + retest + re-run codex until clean. P3 findings → log for Phase 2.1 or Phase 3 follow-up, not blocking.

- [ ] **Step 3: Tag the clean-review commit**

Once codex returns "SAFE TO MERGE":

```bash
git tag phase2-merge-ready-$(date +%Y%m%d) -m "Phase 2 remote compute dispatch — codex gate passed"
git push origin main
git push origin --tags
```

Phase 2 done. Per the roadmap: Phase 3 (Marketplace) is next, and it can reuse the dispatcher/receipt/escrow infrastructure this phase built.

---

## Self-Review Checklist

After all 8 tasks complete:

- [ ] All unit tests pass: ~18 new tests across receipt, escrow, dispatcher, provider.
- [ ] 3-node integration test passes with bit-identical local/remote output.
- [ ] Phase 1 regression suite unchanged: 100+ tests still pass.
- [ ] `NotImplementedError` at `executor.py:155` never fires in any test with a bootstrapped node.
- [ ] Codex review returns SAFE TO MERGE.
- [ ] No `TODO`, `FIXME`, or `pass # placeholder` markers in any new or modified file.
- [ ] No new files in repo root.
- [ ] Every commit message references `docs/2026-04-12-phase2-remote-compute-{design,plan}.md`.

## Estimated Scope

- 8 commits
- 3 new production files (`shard_receipt.py`, `remote_dispatcher.py`, test helpers)
- 4 modified production files (`executor.py`, `payment_escrow.py`, `compute_provider.py`, `node.py`)
- 5 new test files / sections (~18 unit tests + 1 integration test)
- Total: ~500 LoC new production code + ~400 LoC tests
- Same shape as Phase 1.3 Tasks 1-3 combined, smaller than the full Phase 1.3 surface.

After Phase 2 ships, per the roadmap: "After Phase 1, re-audit before starting Phase 2" — but Phase 2 has already started. The audit-gap should happen after Phase 2 ships (re-audit before Phase 3).

---

## Addendum: Vision-Doc-Derived Requirements (Added 2026-04-14)

The PRSM Vision document (`PRSM_Vision.md`) has evolved during the Phase 1.3 bake-in period to include two sections that impose new requirements on Phase 2 remote compute dispatch:

- **Section 7: Private Inference — The Zero-Trust Compute Layer.** Positions PRSM as a regulated-industry-grade inference substrate with sharded weights, TEE-attested compute, and on-chain receipts.
- **Section 6 subsection: The four-tier supply architecture.** Introduces T3 (professional arbitrage) operators running PRSM nodes on rented cloud GPUs, with spot preemption as a first-class concern.

The Phase 2 design and plan documents above precede these Vision additions. Before Phase 2 ships, the following line items must be incorporated — either into Phase 2 scope directly, or explicitly deferred to a named successor phase with a written rationale.

### Line item A: Spot preemption handling in signed-receipt protocol

**Vision doc reference:** Section 6 ("Spot preemption handling") and Section 7 ("on-chain receipts").

**Problem:** T3 operators run on cloud spot instances that can be preempted mid-inference. The current `ShardExecutionReceipt` design (Task 2 above) assumes complete-or-abandon semantics. Preemption is neither: the operator did honest work up to preemption, then the cloud provider killed the pod through no fault of the operator.

**Required behavior:**

1. **Partial-completion credit.** If a node is preempted after completing `k` of `n` tensor-parallel shards, it receives credit for the `k` completed shards (partial FTNS payment), not zero.
2. **Re-routing.** The `RemoteShardDispatcher` must detect preemption (timeout with specific signal from the node's final heartbeat, if any) and re-dispatch the incomplete shards to another node without treating the preempted node as malicious.
3. **No slashing for preemption.** Slashing applies only to provable abandonment (node signs on to a job, then goes silent with no preemption signal) or to verified malicious output (receipt mismatch, Section 7 activation-attestation failure). Preemption is neither.

**Acceptance criterion:** New integration test — 3-node job where one node is killed mid-execution (simulated via `kill -9` or pod eviction API), the job completes successfully via re-routing, the preempted node receives partial FTNS credit, and no slashing event is recorded against it.

**Recommendation:** incorporate into Phase 2 scope. Without this, T3 supply-tier adoption is blocked because no rational cloud operator will run PRSM nodes where preemption = total loss.

### Line item B: Activation-inversion mitigation primitives

**Vision doc reference:** Section 7, "Honest limits" — "Activation-inversion attacks can partially reconstruct input prompts from early-layer activations. Mitigations include topology rotation per inference and activation-layer TEE attestation, both of which are in the Phase 2+ roadmap."

**Problem:** If the same set of nodes repeatedly handles the same early transformer layers across many inferences from the same user, they accumulate enough activation observations to mount reconstruction attacks. A rotating topology — where each inference is randomly assigned to a different subset of nodes across the full shard set — breaks this accumulation.

**Required behavior:**

1. **Per-inference topology randomization.** The `RemoteShardDispatcher` selects the node assignment for each shard from a randomized subset of eligible nodes. Consecutive inferences from the same requester do not reuse the same node-to-shard mapping.
2. **Unlinkability.** The dispatch protocol does not expose to any participating node (a) which other nodes are handling adjacent shards, or (b) the identity of the requester. Nodes see only their assigned shard, the upstream activation, and a dispatch token.
3. **Compatible with preemption re-routing (line item A).** When re-routing, the replacement node must come from a fresh random subset, not the next node in a deterministic ring.

**Acceptance criterion:** 100 consecutive inferences from the same requester with the same prompt prefix result in <10% node-assignment overlap across inferences (measured across the top-K early-layer nodes where inversion attacks are most effective).

**Recommendation:** design in Phase 2 (the dispatcher is being built now; adding topology randomization later is substantially harder than shipping it correctly on day one). Implementation can be scope-capped to "per-inference random selection over eligible pool"; more sophisticated privacy-preserving routing (onion-routed shard assignment, etc.) can defer.

### Line item C: TEE attestation at inference granularity

**Vision doc reference:** Section 7, "TEE-attested compute. Each SPRK executes in a hardware-isolated enclave."

**Problem:** Phase 2 Rings 7-10 (already shipped) provide TEE runtime support at the node level — a node declares itself TEE-capable at join time. Section 7 requires something stronger: per-inference attestation that *this specific shard execution* occurred inside a valid, unrevoked TEE, with fresh quote verification. Node-level claims ("I'm running SGX, trust me") are insufficient for the regulated-industry tier.

**Required behavior:**

1. **Per-inference quote.** `ShardExecutionReceipt` optionally carries a TEE attestation quote bound to the specific execution (input hash, output hash, shard id, timestamp nonce).
2. **Quote verification.** The dispatcher verifies the quote against a current attestation service (Intel DCAP, AMD KDS, etc.) before accepting the receipt for high-sensitivity jobs.
3. **Tier gating.** Requesters can specify a job-level requirement: "accept receipts only from nodes that include a verified TEE quote for this execution." Non-TEE nodes are excluded from these jobs.
4. **Revocation handling.** The dispatcher refuses receipts bearing quotes from revoked TEE instances (known-broken SGX platforms, etc.).

**Acceptance criterion:** Integration test — a tier-gated job dispatched to a pool containing both TEE and non-TEE nodes only accepts results from TEE nodes, all receipts carry verified quotes, and the verification failure path (simulated expired quote) correctly rejects the receipt and re-dispatches.

**Recommendation:** design in Phase 2, ship implementation in Phase 2.1 or Phase 3. The `ShardExecutionReceipt` schema from Task 2 must reserve the attestation field now; the verification logic can land later without breaking the wire format.

### Line item D: Cross-references

- Update `docs/2026-04-12-phase2-remote-compute-design.md` to reference Vision doc Section 7 in its "Security model" section.
- Add explicit note to Task 2's `ShardExecutionReceipt` schema: reserved field for TEE attestation quote (optional in Phase 2, required in Phase 2.1+).
- Add explicit note to Task 5's `RemoteShardDispatcher`: topology randomization is in-scope; see line item B.
- Add explicit note to Task 7's integration test: preemption handling test is in-scope; see line item A.

### Scope impact

**In scope for Phase 2 (this plan):**
- Line item A (preemption handling) — integrate into Task 2, Task 5, Task 7.
- Line item B (topology randomization) — integrate into Task 5.
- Line item C (TEE attestation schema field only, verification deferred) — integrate into Task 2.

**Deferred to Phase 2.1 (new follow-on phase, not yet planned):**
- Line item C verification logic and tier-gating enforcement.

**Estimated scope impact:** +1 commit, +150 LoC production, +100 LoC tests. New total: ~9 commits, ~650 LoC production, ~500 LoC tests. Still smaller than Phase 1.3's full surface.

### Naming clarification: compute verification tiers vs. content confidentiality tiers (Added 2026-04-15)

A terminology collision exists in the broader PRSM documentation and must be clarified explicitly in any Phase 2 materials or inline comments that reference tiers:

- **Compute verification tiers (A/B/C).** Used throughout this plan and in the codebase for the verification-strength spectrum applied to remote compute dispatch: Tier A is receipt-only (cheap; current Phase 2 scope), Tier B is redundant execution consensus (expensive; deferred to Phase 7), Tier C is stake-slash verification (requires on-chain slashing contract; deferred to Phase 7).
- **Content confidentiality tiers (A/B/C).** Used in `PRSM_Vision.md` §2 and §7 for the confidentiality-strength spectrum applied to stored content: Tier A is public content (current Phase 1 scope), Tier B is encryption-before-sharding (deferred to Phase 7), Tier C is encryption + Reed-Solomon erasure coding + Shamir-split keys (deferred to Phase 7).

These are **two independent tier systems applied to two different concerns (compute vs. data).** The naming collision is unfortunate but the concepts are orthogonal. When disambiguation is needed in code, comments, or communications, prefer the fully-qualified terms "compute verification Tier A/B/C" and "content confidentiality Tier A/B/C" rather than bare "Tier A/B/C."

Phase 7 delivers both sets of Tier B and C simultaneously because both depend on similar cryptographic and consensus infrastructure (stake management, on-chain proofs, erasure coding). The Phase 7 plan documentation will include an explicit section disambiguating the two tier systems and confirming they do not share code paths — they are independent subsystems co-located in the same phase delivery for scheduling convenience, not logical coupling.

### Cross-reference: R7 research track — KV/activation compression (Added 2026-04-16)

The Phase 2 activation-streaming path (Task 5 `RemoteShardDispatcher` and downstream SPRK execution) transports activation tensors between pipeline-parallel and tensor-parallel stages in plaintext at native model precision. This is the primary driver of the 9000× bandwidth handicap documented in Risk Register G3 and PRSM_Vision §7.

**Research track R7** (`docs/2026-04-14-phase4plus-research-track.md`) investigates KV cache and activation-tensor compression (rotation-based quantization, Johnson-Lindenstrauss residual corrections, established KV-quant schemes like KIVI / KVQuant / QuaRot) as an efficiency-side mitigation. R7 is research, not Phase 2 engineering — activation streaming in Phase 2 ships uncompressed, as currently designed.

**Interactions with Phase 2 line items that any future R7-driven engineering phase must consider:**

- **Line item B (topology randomization).** Compression schemes that rely on cross-stage statistics (e.g., calibrated quantization buckets) may weaken under per-inference topology rotation. R7 must be measured with Phase 2 topology randomization enabled, not against a fixed topology.
- **Line item C (TEE attestation).** If compression executes inside the enclave, the enclave's attestation surface grows (compression library is part of the trusted codebase). If compression executes post-enclave, plaintext activations still exist in enclave memory — no privacy regression but also no privacy gain. R7 engineering phase must choose explicitly.
- **Ring 9 DP noise (already shipped).** Composition of quantization noise with calibrated DP noise is not obvious; R7 must budget them jointly or demonstrate independence.
- **Activation-inversion threat model (Risk Register G5, research track R3).** Quantization may reduce or may increase inversion reconstructability. R7 results must be validated against the R3 red-team before any production rollout.

**No Phase 2 scope impact.** Phase 2 ships as planned with plaintext activation streaming. R7 is tracked as parallel research; any engineering integration is a future phase.
