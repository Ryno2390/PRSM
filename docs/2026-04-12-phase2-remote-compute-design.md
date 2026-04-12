# Phase 2: Remote Compute Dispatch — Design Spec

> **Status:** Design approved 2026-04-12. Implementation plan pending.
>
> **Roadmap link:** Phase 2 of [2026-04-10-audit-gap-roadmap.md](./2026-04-10-audit-gap-roadmap.md) — "Remote Compute Dispatch (Ring 8 completion)."
>
> **Predecessor:** Phase 1.3 (on-chain provenance) — completes the economic rails Phase 2 builds on.
>
> **For agentic workers:** This is the validated design. Next step: invoke the `superpowers:writing-plans` skill to generate a step-by-step implementation plan from this spec.

## Goal

Make `TensorParallelExecutor._execute_shard(node_id != "local")` actually execute remotely instead of raising `NotImplementedError`. The remote node runs the shard, returns a signed receipt, and gets paid via FTNS escrow. Timeout, retry, fallback, and payment verification all work end-to-end.

## Acceptance criteria

1. A sharded inference runs end-to-end across 3 in-process PRSM nodes with real WebSocket transport.
2. Each remote shard fires a signed `ShardExecutionReceipt` that the requester verifies (Ed25519 signature + output hash).
3. Each successful shard releases its local-ledger FTNS escrow to the provider; each failure (timeout, declined, bad signature) refunds to the requester.
4. Local execution is bit-identical between local-path and remote-serve-path — both call the same `execute_shard_locally` helper.
5. The `NotImplementedError` safety rail at `executor.py:155` never fires in any test where the node was wired with a `RemoteShardDispatcher`.

## Key decisions (Q&A during brainstorming)

| Decision | Choice | Reason |
|---|---|---|
| Transport protocol | WebSocket MSG_DIRECT (option A) | Reuses existing transport; no new dependencies; upgrade path to gRPC in Phase 6 |
| Verification strategy | Tiered, receipt-only (tier A) in Phase 2 (option D) | Interface designed for tiers B (redundant execution) and C (stake+slash) to plug in later without protocol changes |
| 3-node test fidelity | In-process with real transport (option B) | Sweet spot between speed and fidelity; multi-process chaos tests are Phase 6 |

## Current state audit (pre-Phase-2)

| Component | File | State |
|---|---|---|
| `NotImplementedError` for remote shards | `prsm/compute/model_sharding/executor.py:155` | Scaffold — `remote_dispatcher` slot exists but nothing injected |
| Ring 8 sharding (models + sharder) | `prsm/compute/model_sharding/models.py`, `sharder.py` | Working locally (numpy matmul) |
| ComputeProvider | `prsm/node/compute_provider.py` | Production-ready for gossip-driven local jobs; no MSG_DIRECT handler yet |
| Transport layer | `prsm/node/transport.py` | WebSocket gossip + MSG_DIRECT both working (Phase 1 uses MSG_DIRECT for content) |
| Gossip subtypes | `prsm/node/gossip.py` | `GOSSIP_JOB_*` and `GOSSIP_ESCROW_*` defined, fire-and-forget only |
| PaymentEscrow | `prsm/node/payment_escrow.py` | Scaffold — `create_escrow()` works, `release_escrow()` body incomplete |
| Result consensus | `prsm/node/result_consensus.py` | `ProviderResult` struct reused for receipt shape |
| Executor integration | `prsm/node/node.py` | No `RemoteShardDispatcher` constructed or injected |

Delta estimate: ~500 lines of new code across 3–4 files. Most of the infrastructure exists; Phase 2 is about wiring the pieces together through the `RemoteShardDispatcher`.

## Architecture

```
Requester Node                              Provider Node
─────────────                               ──────────────
TensorParallelExecutor
  → RemoteShardDispatcher
    → PaymentEscrow.create_escrow()
    → MSG_DIRECT: shard_execute_request ──→ ComputeProvider._on_shard_request()
                                              → deserialize shard
                                              → execute_shard_locally()
                                              → sign(output_hash)
    ← MSG_DIRECT: shard_execute_response ←── return (output, receipt)
    → verification_strategy.verify(receipt)
    → PaymentEscrow.release_escrow()
    → return output to executor
  → all_reduce(local_outputs + remote_outputs)
  → final result
```

### New components (3 files)

1. `prsm/compute/remote_dispatcher.py` — `RemoteShardDispatcher` class
2. `prsm/compute/shard_receipt.py` — `ShardExecutionReceipt` dataclass + `VerificationStrategy` protocol + `ReceiptOnlyVerification` impl
3. `prsm/compute/model_sharding/executor.py` — new module-level `execute_shard_locally()` helper (extracted from `_execute_local`)

### Modified components

1. `prsm/node/compute_provider.py` — add MSG_DIRECT subscription + `_on_shard_execute_request` handler + `_can_accept_shard` policy check
2. `prsm/node/payment_escrow.py` — complete `release_escrow()` body + add idempotency checks
3. `prsm/node/node.py` — construct and inject `RemoteShardDispatcher` into `TensorParallelExecutor`

## Message protocol

Two new MSG_DIRECT subtypes on the existing transport. JSON payloads, base64-encoded tensor bytes.

### `shard_execute_request` (requester → provider)

```python
{
    "subtype": "shard_execute_request",
    "job_id": "<uuid>",
    "shard_index": 0,
    "tensor_data_b64": "<base64 numpy bytes>",
    "tensor_shape": [128, 512],
    "tensor_dtype": "float32",
    "input_b64": "<base64 input tensor>",
    "input_shape": [1, 128],
    "input_dtype": "float32",
    "checksum": "<sha256 of tensor_data>",
    "stake_tier": "STANDARD",
    "escrow_tx_id": "<requester's local escrow ID>",
    "deadline_unix": 1776019200,
    "request_id": "<uuid for response matching>",
    "requester_pubkey_b64": "<requester Ed25519 pubkey>",
}
```

### `shard_execute_response` (provider → requester)

```python
{
    "subtype": "shard_execute_response",
    "request_id": "<matches request>",
    "status": "completed",   # or "failed" / "declined"
    "shard_index": 0,
    "output_b64": "<base64 output tensor>",
    "output_shape": [1, 512],
    "output_dtype": "float32",
    "receipt": {
        "job_id": "<uuid>",
        "shard_index": 0,
        "provider_id": "<provider node_id>",
        "provider_pubkey_b64": "<provider Ed25519 pubkey>",
        "output_hash": "<sha256 of output bytes>",
        "executed_at_unix": 1776019150,
        "signature": "<Ed25519 over (job_id || shard_index || output_hash || executed_at)>",
    },
    "error": null,   # populated on status="failed"
}
```

### Payload sizing

- PRSM shards are typically KB to low MB (numpy arrays).
- Base64 encoding inflates by 33%.
- WebSocket frame limit is 1 MB default, configurable to 16 MB.
- Dispatcher refuses shards > 10 MB decoded size (configurable via `max_shard_bytes`) and logs a warning.
- Larger shards are a Phase 6 gRPC-streaming concern.

### Signature scheme

Provider signs `keccak256(job_id || shard_index || output_hash || executed_at_unix)` with their Ed25519 identity key. Requester verifies against the provider's gossip-advertised pubkey (same key material PRSM already uses for node identity).

## RemoteShardDispatcher interface

```python
class RemoteShardDispatcher:
    def __init__(
        self,
        identity: NodeIdentity,
        transport: WebSocketTransport,
        payment_escrow: PaymentEscrow,
        verification_strategy: VerificationStrategy,
        default_timeout: float = 30.0,
        max_retries: int = 1,
        max_shard_bytes: int = 10 * 1024 * 1024,
        local_fallback: Optional[Callable] = None,
    ): ...

    async def dispatch(
        self,
        shard: ModelShard,
        input_tensor: np.ndarray,
        node_id: str,
        job_id: str,
        stake_tier: StakeTier,
        escrow_amount_ftns: float,
    ) -> np.ndarray: ...
```

### `dispatch()` flow

1. **Size check.** If shard serialized size > `max_shard_bytes`, log warning + fall back to local (if `local_fallback` is wired) or raise `ShardTooLargeError`.
2. **Peer resolution.** Look up `node_id` via `transport.get_peer()`. If not connected, fall back or raise `PeerNotConnectedError`.
3. **Escrow.** `payment_escrow.create_escrow(requester_id, amount=escrow_amount_ftns, purpose=f"shard_exec:{job_id}:{shard.shard_index}")` → `escrow_tx_id`.
4. **Build request.** Construct `shard_execute_request` payload with fresh `request_id`, `deadline_unix = now + default_timeout`.
5. **Register pending response.** `self._pending[request_id] = asyncio.Future()`.
6. **Send.** `await transport.send_to_peer(node_id, P2PMessage(MSG_DIRECT, payload))`.
7. **Await with timeout.** `response = await asyncio.wait_for(future, timeout=default_timeout)`.
   - On `TimeoutError` + retries remaining: retry once with fresh `request_id`.
   - On `TimeoutError` + retries exhausted: refund escrow + fall back or raise.
8. **Verify receipt.** `verification_strategy.verify(receipt, output_bytes)`. On failure: refund escrow + raise.
9. **Release escrow.** `payment_escrow.release_escrow(escrow_tx_id, recipient=node_id, amount=escrow_amount_ftns)`.
10. **Return output.** Decode `output_b64` via `output_shape` + `output_dtype`.

### Response handler (subscribed to MSG_DIRECT at __init__)

```python
async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection):
    if msg.payload.get("subtype") != "shard_execute_response":
        return
    request_id = msg.payload.get("request_id")
    future = self._pending.pop(request_id, None)
    if future and not future.done():
        future.set_result(msg.payload)
```

Fire-and-forget router into pending futures. Mirrors Phase 1's `content_response` pattern.

## Verification strategy interface

```python
class VerificationStrategy(Protocol):
    async def verify(self, receipt: dict, output_bytes: bytes) -> bool: ...


class ReceiptOnlyVerification:
    """Tier A: signature check only. Phase 2 default."""
    async def verify(self, receipt: dict, output_bytes: bytes) -> bool:
        # 1. Verify output_hash == sha256(output_bytes)
        # 2. Verify Ed25519(receipt["signature"]) against receipt["provider_pubkey_b64"]
        #    over keccak256(job_id || shard_index || output_hash || executed_at_unix)
        ...


# Future (NOT Phase 2):
# class RedundantExecutionVerification: ...   # tier B, Phase 7
# class StakeSlashVerification: ...           # tier C, Phase 7
```

## ComputeProvider extension (server side)

### MSG_DIRECT subscription in `start()`

```python
def start(self) -> None:
    # existing gossip subscriptions...
    self.transport.on_message(MSG_DIRECT, self._on_direct_message)

async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection):
    subtype = msg.payload.get("subtype", "")
    if subtype == "shard_execute_request":
        await self._on_shard_execute_request(msg, peer)
    # other subtypes handled by other subscribers — subtype filter routes correctly
```

### `_on_shard_execute_request` handler

```python
async def _on_shard_execute_request(self, msg: P2PMessage, peer: PeerConnection):
    payload = msg.payload
    request_id = payload.get("request_id")
    job_id = payload.get("job_id")
    shard_index = payload.get("shard_index")

    try:
        if not self._can_accept_shard(payload):
            await self._send_shard_response(
                peer.peer_id, request_id,
                status="declined", shard_index=shard_index,
                error="resource_unavailable",
            )
            return

        shard = self._deserialize_shard(payload)
        input_tensor = self._deserialize_input(payload)

        output = await self._execute_shard_locally(shard, input_tensor)

        output_bytes = output.tobytes()
        output_hash = hashlib.sha256(output_bytes).hexdigest()
        executed_at = int(time.time())
        sig_payload = (
            f"{job_id}||{shard_index}||{output_hash}||{executed_at}"
        ).encode()
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
    except Exception as e:
        logger.warning(f"shard_execute_request failed: {e}")
        await self._send_shard_response(
            peer.peer_id, request_id,
            status="failed", shard_index=shard_index, error=str(e),
        )
```

### Acceptance policy `_can_accept_shard`

Checks (in order):

1. `self._current_jobs < self.max_concurrent_jobs`
2. Decoded shard size ≤ configured limit
3. Current time < `deadline_unix` (minus expected execution time)
4. Shard checksum matches declared checksum (integrity)
5. (Future) stake_tier ≤ this node's advertised tier

Any failure → respond with `status="declined"`. Requester's dispatcher treats declined like a timeout — refunds escrow, falls back or retries.

### Local execution reuse

Extract `TensorParallelExecutor._execute_local()` body into a module-level `execute_shard_locally(shard, input_tensor) -> np.ndarray` in `prsm/compute/model_sharding/executor.py`. Both the executor's local path AND the compute_provider's remote-serve path call the same helper, guaranteeing bit-identical local and remote execution.

## Escrow wiring

### Contract for Phase 2

```python
async def create_escrow(
    self,
    requester_id: str,
    amount: float,
    purpose: str,   # "shard_exec:<job_id>:<shard_index>"
) -> str:
    """Debit requester, credit escrow holding wallet. Returns escrow_tx_id.
    Raises InsufficientBalanceError if requester can't cover amount."""

async def release_escrow(
    self,
    escrow_tx_id: str,
    recipient: str,
    amount: Optional[float] = None,   # defaults to full escrowed amount
) -> None:
    """Transfer from escrow holding wallet to recipient. Marks escrow
    RELEASED. Calling release twice on the same escrow_tx_id is a
    no-op with a warning log (self-idempotent). Calling release on
    an already-REFUNDED escrow raises EscrowAlreadyFinalizedError."""

async def refund_escrow(
    self,
    escrow_tx_id: str,
    reason: str = "",
) -> None:
    """Return escrowed amount to original requester. Marks escrow
    REFUNDED. Used on dispatch timeout, declined request, or receipt
    verification failure. Calling refund twice on the same escrow_tx_id
    is a no-op with a warning log (self-idempotent). Calling refund on
    an already-RELEASED escrow raises EscrowAlreadyFinalizedError."""
```

### State machine

`PENDING → RELEASED` (success path) or `PENDING → REFUNDED` (failure path). Both terminal; no further transitions. Idempotency via state check at the top of each operation.

### Failure modes

- Release-before-create: raises `EscrowNotFoundError`
- Amount mismatch (release > escrowed): raises `EscrowAmountError`
- Ledger transfer fails mid-operation: escrow stays PENDING; caller retries or refunds
- Release after refund (or vice versa): raises `EscrowAlreadyFinalizedError`

### No on-chain integration in Phase 2

All escrow is local-ledger FTNS via `LocalLedger.transfer()`. On-chain compute escrow is Phase 3 (Marketplace) territory. Phase 1's `_try_onchain_distribute` is for content royalties, not compute.

## Executor integration

### Bootstrap in `prsm/node/node.py`

```python
# In PRSMNode.initialize(), after content_provider + compute_provider wired:
from prsm.compute.remote_dispatcher import RemoteShardDispatcher
from prsm.compute.shard_receipt import ReceiptOnlyVerification

self.remote_shard_dispatcher = RemoteShardDispatcher(
    identity=self.identity,
    transport=self.transport,
    payment_escrow=self.payment_escrow,
    verification_strategy=ReceiptOnlyVerification(),
    default_timeout=30.0,
    max_retries=1,
    local_fallback=None,   # None for Phase 2
)
```

### TensorParallelExecutor wiring

Wherever `TensorParallelExecutor` is currently instantiated, pass `remote_dispatcher=self.remote_shard_dispatcher.dispatch`. The existing `remote_dispatcher` callable slot satisfies the contract.

### `execute_shard_locally` helper (shared)

```python
# New module-level function in prsm/compute/model_sharding/executor.py:
def execute_shard_locally(shard: ModelShard, input_tensor: np.ndarray) -> np.ndarray:
    """Pure numpy matmul for a single shard. Used by both the local
    executor path AND the compute_provider's remote-serve path, so
    local and remote execution produce bit-identical results.
    """
    tensor = np.frombuffer(shard.tensor_data, dtype=np.float32)
    tensor = tensor.reshape(shard.tensor_shape)
    return np.matmul(input_tensor, tensor)


# _execute_local becomes a thin wrapper:
async def _execute_local(self, shard, input_tensor):
    output = execute_shard_locally(shard, input_tensor)
    return {"output_array": output, "execution_mode": "local", ...}
```

### Parallel execution loop

Existing `asyncio.gather()` logic unchanged — remote dispatches await independently so one slow node doesn't block the rest.

## Test strategy

### Unit tests

**`tests/unit/test_remote_dispatcher.py`** (new, ~6 tests):

- `test_dispatch_happy_path` — mock transport, register fake response via future, assert escrow released + output returned
- `test_dispatch_timeout_falls_back` — mock transport swallows request, assert escrow refunded + local_fallback called (when wired)
- `test_dispatch_bad_signature_refunds` — inject response with invalid Ed25519 signature, assert escrow refunded + error raised
- `test_dispatch_size_limit_rejects` — shard > max_shard_bytes, assert no network send + local fallback fires
- `test_dispatch_retry_on_timeout` — first response times out, second succeeds, assert single final escrow release
- `test_dispatch_peer_not_connected` — `transport.get_peer()` returns None, assert local fallback or raise

**`tests/unit/test_shard_receipt.py`** (new, ~4 tests):

- `test_receipt_verification_happy_path` — signed receipt over known payload, verify passes
- `test_receipt_verification_bad_signature` — tampered signature, verify fails
- `test_receipt_verification_output_hash_mismatch` — signature valid but output bytes don't match declared hash, verify fails
- `test_receipt_verification_wrong_pubkey` — signature valid over correct payload but signed by different key, verify fails

**`tests/node/test_compute_provider.py`** (extend or create, ~4 tests):

- `test_shard_execute_request_happy_path` — mock transport delivery, assert response carries correct output + signed receipt
- `test_shard_execute_request_declines_when_over_capacity` — jobs at max, assert `status="declined"`
- `test_shard_execute_request_declines_past_deadline` — `deadline_unix < now`, assert declined
- `test_shard_execute_request_declines_bad_checksum` — shard checksum mismatch, assert declined

**Escrow idempotency** (extend existing escrow tests, ~2 tests):

- `test_release_escrow_idempotent` — double-release logs warning, doesn't double-pay
- `test_refund_after_release_errors` — post-release refund raises `EscrowAlreadyFinalizedError`

### Integration test

**`tests/integration/test_phase2_remote_dispatch.py`** (new, 1 test):

- `test_three_node_sharded_inference_end_to_end` — roadmap acceptance criterion. Spin up 3 `PRSMNode` instances in-process via `asyncio.gather()`, each with real transport + gossip + ledger connected via a loopback hub. Node A generates a sharded model + input, assigns shard 0 locally, shard 1 to node B, shard 2 to node C. Dispatch via executor. Assert:
  - Each remote shard fires a real `shard_execute_request` MSG_DIRECT
  - Each provider executes and returns a signed receipt
  - Receipts verify on the requester
  - Escrow: 2 escrows created, 2 released to correct providers, none left in PENDING
  - Final `all_reduce` output matches a local-only baseline (bit-identical via the shared `execute_shard_locally` helper)

### Total test count

~15 new tests. Mirrors Phase 1's TDD discipline — every new module has dedicated unit tests, plus one flagship integration test that proves the whole flow end-to-end.

## Out of scope

### Deferred to Phase 3 (Marketplace)

- On-chain compute escrow contract (`ComputeEscrow.sol`). Phase 2 uses local-ledger FTNS only.
- Bidding / auction for shard execution. Phase 2 uses direct `node_id` assignment — the executor picks the node, dispatcher sends to it. Bidding sits *above* the dispatcher in Phase 3.
- Cross-node reputation scoring beyond what's already in `ContentDiscovery`.

### Deferred to Phase 6 (P2P hardening)

- gRPC or libp2p-stream upgrade. Phase 2 ships MSG_DIRECT; the dispatcher/receipt/escrow interfaces are stable enough that a transport swap doesn't change higher-level logic.
- Multi-process integration test. Phase 2 uses in-process loopback.
- NAT traversal / relay for remote compute peers behind firewalls.

### Deferred to Phase 7 (Slashing + storage hardening)

- Tier B verification (redundant execution for consensus). Interface slot exists; impl is Phase 7.
- Tier C verification (stake + slashing). Interface slot exists; impl is Phase 7 alongside storage slashing.
- On-chain `ComputeSlashing.sol` contract.

### Explicitly NOT Phase 2 even though adjacent

- Streaming output for large tensor results. Max 10 MB decoded size, drop shards over that with a warning. Larger shards are Phase 6 gRPC-streaming.
- GPU acceleration on the provider side. `ComputeProvider` detects GPU but the shard executor path is CPU/numpy. Future task adds GPU numpy/cupy behind a feature flag.
- Model weight distribution. Phase 2 assumes the requester ships tensor bytes inline with the request. Phase 3+ adds a "fetch by hash" path from content-addressed store.
- Job cancellation mid-execution. If the requester times out, the provider finishes and the response is dropped. Cooperative cancellation is Phase 6.

## File structure

| File | Status | Purpose |
|---|---|---|
| `prsm/compute/remote_dispatcher.py` | **Create** | `RemoteShardDispatcher` class; dispatch flow, timeout/retry, escrow wiring, response routing |
| `prsm/compute/shard_receipt.py` | **Create** | `ShardExecutionReceipt` dataclass, `VerificationStrategy` protocol, `ReceiptOnlyVerification` impl |
| `prsm/compute/model_sharding/executor.py` | **Modify** | Extract `execute_shard_locally` as module-level helper; keep `_execute_local` as thin wrapper |
| `prsm/node/compute_provider.py` | **Modify** | Add MSG_DIRECT subscription, `_on_shard_execute_request` handler, `_can_accept_shard` policy |
| `prsm/node/payment_escrow.py` | **Modify** | Complete `release_escrow` body, add `refund_escrow`, add idempotency checks |
| `prsm/node/node.py` | **Modify** | Construct `RemoteShardDispatcher` in `initialize()`; wire into `TensorParallelExecutor` |
| `tests/unit/test_remote_dispatcher.py` | **Create** | ~6 unit tests for dispatch flow |
| `tests/unit/test_shard_receipt.py` | **Create** | ~4 unit tests for receipt verification |
| `tests/node/test_compute_provider.py` | **Extend/Create** | ~4 unit tests for server-side handler |
| `tests/integration/test_phase2_remote_dispatch.py` | **Create** | 1 integration test — 3-node sharded inference |

Estimated ~500 lines of new code + ~400 lines of tests.
