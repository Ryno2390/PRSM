# Ring 8 — "The Shield" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A proprietary model can be sharded across the network using tensor parallelism, with randomized pipeline assignment and stake-weighted trust ensuring no single node (or colluding group) can reconstruct the full model.

**Architecture:** A `ModelSharder` splits model weight tensors into N partitions. A `PipelineRandomizer` assigns shards to nodes randomly per inference, enforcing minimum pool size. A `TensorParallelExecutor` coordinates parallel execution with ring all-reduce synchronization. A `CollisionDetector` compares diversified pipeline outputs to detect tampering. Extended staking tiers govern pipeline participation.

**Tech Stack:** `numpy` (already a dependency) for tensor operations. No new external dependencies. Actual PyTorch model loading is deferred — we work at the tensor/numpy level for the infrastructure.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/compute/model_sharding/__init__.py` | Package exports |
| Create | `prsm/compute/model_sharding/models.py` | `ModelShard`, `ShardedModel`, `PipelineConfig`, `PipelineStakeTier` |
| Create | `prsm/compute/model_sharding/sharder.py` | `ModelSharder` — tensor partitioning |
| Create | `prsm/compute/model_sharding/randomizer.py` | `PipelineRandomizer` — random node assignment |
| Create | `prsm/compute/model_sharding/executor.py` | `TensorParallelExecutor` — coordinated parallel inference |
| Create | `prsm/compute/model_sharding/collision_detector.py` | `CollisionDetector` — diversified pipeline comparison |
| Modify | `prsm/node/node.py` | Wire Ring 8 into PRSMNode |
| Create | `tests/unit/test_model_sharding.py` | Sharder + models tests |
| Create | `tests/unit/test_pipeline_security.py` | Randomizer + collision detection tests |
| Create | `tests/integration/test_ring8_shield.py` | End-to-end smoke test |

---

### Task 1: Model Sharding Data Models

**Files:**
- Create: `prsm/compute/model_sharding/__init__.py`
- Create: `prsm/compute/model_sharding/models.py`
- Test: `tests/unit/test_model_sharding.py`

- [ ] **Step 1:** `mkdir -p prsm/compute/model_sharding`

- [ ] **Step 2: Write tests, implement, commit**

`tests/unit/test_model_sharding.py` covers:
- PipelineStakeTier enum values and thresholds
- ModelShard creation and to_dict roundtrip
- ShardedModel creation with multiple shards
- ShardedModel.get_shard_by_index
- PipelineConfig defaults and custom

`prsm/compute/model_sharding/models.py` contains:

- **`PipelineStakeTier`** enum: OPEN(0 FTNS, 0% slash), STANDARD(5000, 50%), PREMIUM(25000, 100%), CRITICAL(50000, 100% + ban)
- **`ModelShard`** dataclass: shard_id, model_id, shard_index, total_shards, tensor_data (numpy bytes), tensor_shape (tuple), layer_range (tuple of ints), size_bytes, checksum. Methods: to_dict(), from_dict()
- **`ShardedModel`** dataclass: model_id, model_name, total_shards, shards (List[ModelShard]), stake_tier (PipelineStakeTier), created_at. Methods: get_shard_by_index(i), total_size_bytes (property)
- **`PipelineConfig`** dataclass: parallelism_degree (int, 4), min_pool_size (int, 20), require_tee (bool, False), privacy_level (str, "standard"), stake_tier (PipelineStakeTier, STANDARD), enable_diversified_pipeline (bool, False), max_latency_ms (int, 5000)

Commit: `"feat(ring8): ModelShard, ShardedModel, PipelineConfig, PipelineStakeTier models"`

---

### Task 2: Model Sharder + Pipeline Randomizer

**Files:**
- Create: `prsm/compute/model_sharding/sharder.py`
- Create: `prsm/compute/model_sharding/randomizer.py`
- Test: `tests/unit/test_model_sharding.py` (append)
- Test: `tests/unit/test_pipeline_security.py`

**ModelSharder:**
- `shard_tensor(tensor: np.ndarray, n_shards: int) -> List[np.ndarray]` — splits along largest dimension
- `shard_model(model_id, model_name, weight_tensors: Dict[str, np.ndarray], n_shards: int, stake_tier) -> ShardedModel`
- `reassemble_tensor(shards: List[np.ndarray]) -> np.ndarray` — concatenates shards back

**PipelineRandomizer:**
- `__init__(self, min_pool_size=20)`
- `assign_pipeline(shard_count, available_nodes: List[Dict], require_tee=False) -> List[Dict]` — randomly selects nodes, enforces min pool, returns assignment list
- `_validate_pool(available_nodes, shard_count)` — raises if pool too small
- `_entropy_score(assignments_history) -> float` — measures randomization quality over time

Tests cover: tensor splitting/reassembly, correct shard count, pool size enforcement, randomization produces different assignments, TEE filtering, entropy scoring.

TWO commits:
- `"feat(ring8): ModelSharder — tensor partitioning with checksum verification"`
- `"feat(ring8): PipelineRandomizer — randomized node assignment with pool enforcement"`

---

### Task 3: Tensor Parallel Executor + Collision Detector

**Files:**
- Create: `prsm/compute/model_sharding/executor.py`
- Create: `prsm/compute/model_sharding/collision_detector.py`
- Test: `tests/unit/test_pipeline_security.py` (append)

**TensorParallelExecutor:**
- `__init__(self, confidential_executor, pipeline_config)`
- `async execute_parallel(sharded_model, input_data, node_assignments) -> Dict` — fans out shard execution, collects results, performs all-reduce aggregation
- `_all_reduce(shard_outputs: List[np.ndarray]) -> np.ndarray` — ring all-reduce (sum/average)

**CollisionDetector:**
- `__init__(self, dp_epsilon=8.0, tolerance_multiplier=3.0)`
- `compare_pipelines(output_a: bytes, output_b: bytes) -> Tuple[bool, float]` — returns (match, divergence_score). Accounts for DP noise variance.
- `detect_collision(outputs: List[bytes]) -> Dict` — compares all pairs, returns {match: bool, divergence_scores, flagged_nodes}

Tests cover: parallel execution produces result, all-reduce aggregation, collision detector accepts matching outputs, rejects divergent outputs, DP noise tolerance.

TWO commits:
- `"feat(ring8): TensorParallelExecutor — coordinated shard execution with all-reduce"`
- `"feat(ring8): CollisionDetector — diversified pipeline output comparison"`

---

### Task 4: Node Integration + Smoke Test + Version Bump + Publish

**Files:**
- Modify: `prsm/node/node.py`
- Update: `prsm/compute/model_sharding/__init__.py`
- Create: `tests/integration/test_ring8_shield.py`
- Modify: `prsm/__init__.py`, `pyproject.toml`

Wire `TensorParallelExecutor` into PRSMNode after Ring 7 block. Smoke test covers: model sharding roundtrip, randomized pipeline assignment, parallel execution, collision detection, all Ring 1-8 imports.

Bump to 0.33.0, push, publish.

Commit: `"chore: bump version to 0.33.0 for Ring 8 — The Shield (model sharding + collusion resistance)"`
