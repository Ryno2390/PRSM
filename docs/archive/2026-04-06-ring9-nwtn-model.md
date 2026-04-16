# Ring 9 — "The Mind" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the infrastructure for fine-tuning and deploying NWTN as an open-weight model: training data pipeline (collecting and exporting AgentTrace corpus), model registry for deployment, and a NWTN model service that integrates with Ring 8's sharding infrastructure.

**Architecture:** A `TrainingPipeline` collects AgentTrace data from Ring 5, validates and exports it as a fine-tuning dataset. A `ModelRegistry` manages trained model metadata and deployment state. A `NWTNModelService` registers the fine-tuned model as a backend and serves inference via Ring 8's tensor parallel executor. The actual fine-tuning step is an offline process — this ring builds the tooling around it.

**Tech Stack:** Existing PRSM infrastructure. No new external dependencies.

**Scope note:** This ring builds training pipeline infrastructure and deployment tooling. Actual model training (GPU compute, hyperparameter search) happens offline once sufficient AgentTrace corpus is collected.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/compute/nwtn/training/__init__.py` | Package exports |
| Create | `prsm/compute/nwtn/training/models.py` | `TrainingConfig`, `TrainingCorpus`, `ModelCard` |
| Create | `prsm/compute/nwtn/training/pipeline.py` | `TrainingPipeline` — collect, validate, export traces |
| Create | `prsm/compute/nwtn/training/model_service.py` | `NWTNModelService` — deploy + serve fine-tuned model |
| Modify | `prsm/node/node.py` | Wire Ring 9 into PRSMNode |
| Create | `tests/unit/test_training_pipeline.py` | Training data + pipeline tests |
| Create | `tests/integration/test_ring9_mind.py` | End-to-end smoke test |

---

### Task 1: Training Data Models + Pipeline

Create `prsm/compute/nwtn/training/` package with models and pipeline.

**models.py** contains:
- **`TrainingConfig`** dataclass: base_model (str, "meta-llama/Llama-3.1-8B"), epochs (int, 3), batch_size (int, 4), learning_rate (float, 2e-5), max_seq_length (int, 4096), lora_rank (int, 16), min_corpus_size (int, 100), export_format (str, "jsonl")
- **`TrainingCorpus`** dataclass: corpus_id (str), traces (List[Dict]), total_traces (int property), created_at. Methods: add_trace(trace_dict), validate() -> Tuple[bool, List[str]], export_jsonl() -> str, stats() -> Dict
- **`ModelCard`** dataclass: model_id (str), model_name (str), base_model (str), version (str), training_config (TrainingConfig), corpus_stats (Dict), metrics (Dict — accuracy, pcu_prediction_error, etc), created_at, deployed (bool). Methods: to_dict(), from_dict()
- **`DeploymentStatus`** str enum: REGISTERED, SHARDING, DEPLOYED, SERVING, RETIRED

**pipeline.py** contains `TrainingPipeline`:
- `__init__(self, config: TrainingConfig = None)`
- `ingest_traces(traces: List) -> int` — adds traces from AgentForge.traces, returns count added
- `validate_corpus() -> Tuple[bool, List[str]]` — checks min size, required fields, data quality
- `export_dataset(output_path: str = None) -> str` — exports as JSONL for fine-tuning
- `get_corpus_stats() -> Dict` — trace count, route distribution, avg complexity, etc
- `create_model_card(model_id, model_name, metrics) -> ModelCard` — generates card after training

**Tests** cover: TrainingConfig defaults, corpus add/validate/export, pipeline ingest from forge traces, validation catches missing fields, export produces valid JSONL, model card creation, deployment status enum.

Commit: `"feat(ring9): TrainingPipeline — trace collection, validation, JSONL export for fine-tuning"`

---

### Task 2: NWTN Model Service

Create `prsm/compute/nwtn/training/model_service.py` with `NWTNModelService`:

- `__init__(self, model_registry=None, tensor_executor=None)`
- `register_model(model_card: ModelCard) -> str` — registers model metadata
- `deploy_model(model_id: str, weight_tensors: Dict[str, np.ndarray] = None, n_shards: int = 4) -> Dict` — shards model via Ring 8's ModelSharder, returns deployment info
- `get_model_status(model_id: str) -> Dict` — returns deployment status + stats
- `list_models() -> List[Dict]` — all registered models
- `retire_model(model_id: str) -> bool` — marks model as retired

**Tests** cover: register model, deploy model with sharding, get status, list models, retire model.

Commit: `"feat(ring9): NWTNModelService — model registration, sharded deployment, lifecycle management"`

---

### Task 3: Node Integration + Smoke Test + Version Bump + Publish

Wire into node.py, create smoke test, bump to 0.34.0, push, publish.

Commit: `"chore: bump version to 0.34.0 for Ring 9 — The Mind (NWTN model infrastructure)"`
