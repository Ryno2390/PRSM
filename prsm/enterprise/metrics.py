"""Sprint 318d — enterprise metrics.

Module-level singleton REGISTRY + canonical metric
instances that orchestrator code increments on key
events. Operators scrape via
`/admin/enterprise/metrics` (Prometheus text exposition)
or print via `bringup metrics-snapshot` (one-shot CLI).

The metric names below follow the Prometheus naming
convention: `prsm_<subsystem>_<unit>` with `_total`
suffix for counters. Adding a new metric: import REGISTRY,
call `REGISTRY.counter(name, description)` or
`REGISTRY.gauge(name, description)`, hold the returned
instance at module scope, increment on the event.
"""
from __future__ import annotations

from prsm.enterprise.metrics_registry import (
    Counter,
    Gauge,
    MetricsRegistry,
)


# ── Singleton registry ─────────────────────────────


REGISTRY = MetricsRegistry()


# ── §7 federated learning ──────────────────────────


FL_JOBS_PROPOSED: Counter = REGISTRY.counter(
    "fl_jobs_proposed_total",
    (
        "Number of federated-learning jobs proposed via "
        "FederatedLearningOrchestrator.propose_job"
    ),
)

FL_ROUNDS_AGGREGATED: Counter = REGISTRY.counter(
    "fl_rounds_aggregated_total",
    (
        "Number of FL rounds successfully aggregated "
        "(FedAvg or FedMedian)"
    ),
)

FL_WORKER_UPDATES_ACCEPTED: Counter = REGISTRY.counter(
    "fl_worker_updates_accepted_total",
    (
        "Number of gradient updates accepted from "
        "workers (signed updates included)"
    ),
)

FL_WORKER_UPDATES_REJECTED: Counter = REGISTRY.counter(
    "fl_worker_updates_rejected_total",
    (
        "Number of gradient updates refused — bad "
        "signature / unregistered worker / wrong round / "
        "duplicate / scope mismatch"
    ),
)


# ── §7 federated inference (pipeline orchestrator) ─


PIPELINE_INFERENCE_PROPOSED: Counter = REGISTRY.counter(
    "pipeline_inference_jobs_proposed_total",
    (
        "Number of pipeline inference jobs proposed via "
        "PipelineInferenceOrchestrator.propose_job"
    ),
)

PIPELINE_INFERENCE_COMPLETED: Counter = REGISTRY.counter(
    "pipeline_inference_completed_total",
    (
        "Number of pipeline inference rounds completed "
        "with signed receipts"
    ),
)

PIPELINE_INFERENCE_FAILED: Counter = REGISTRY.counter(
    "pipeline_inference_failed_total",
    (
        "Number of pipeline inference rounds that "
        "failed (stage runner raised, chain broken, "
        "etc.)"
    ),
)


# ── §7 layer-2 $CORP capability ────────────────────


CORP_CAPABILITIES_REDEEMED: Counter = REGISTRY.counter(
    "corp_capabilities_redeemed_total",
    (
        "Number of $CORP capability redemptions accepted "
        "(dual-signature verified, quota OK, not "
        "replayed)"
    ),
)

CORP_CAPABILITIES_REJECTED: Counter = REGISTRY.counter(
    "corp_capabilities_rejected_total",
    (
        "Number of $CORP capability redemptions refused"
    ),
)


# ── §7 layer-1 recipient encryption ────────────────


CONTENT_UPLOADS_ENCRYPTED: Counter = REGISTRY.counter(
    "content_uploads_encrypted_total",
    (
        "Number of /content/upload calls with "
        "recipient encryption enabled (OR-decrypt or "
        "threshold)"
    ),
)


# ── §14 incident lifecycle ─────────────────────────


INCIDENT_OPENED: Counter = REGISTRY.counter(
    "incident_opened_total",
    (
        "Number of §14 incidents opened (by IncidentResponse"
        ".open)"
    ),
)


# ── Subsystem health gauges ────────────────────────


FL_JOBS_PENDING: Gauge = REGISTRY.gauge(
    "fl_jobs_pending",
    (
        "Current count of FL jobs in PROPOSED status "
        "(snapshot; updated on read)"
    ),
)

PIPELINE_JOBS_PENDING: Gauge = REGISTRY.gauge(
    "pipeline_jobs_pending",
    (
        "Current count of pipeline inference jobs not "
        "yet executed (snapshot; updated on read)"
    ),
)
