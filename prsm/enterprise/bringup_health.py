"""Sprint 318c — operator runtime health check.

Sprint 318's `bringup status` is STATIC: it answers
"are env vars set + well-formed?". This module is
DYNAMIC: it answers "do the configured persistence dirs
exist + are they writable? Do the configured keypairs
actually load? Can each subsystem the env vars enable
round-trip a small operation?".

Status answers "is the deployment WIRED?" — health
answers "does the deployment WORK?". Both must pass
before pointing real customers at the node.

The CLI subcommand `bringup health` wraps `run_health_
checks()`, prints the per-check summary, returns rc=0
only if all checks pass (so CI / deploy scripts can
fail-fast).
"""
from __future__ import annotations

import base64
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── CheckResult ────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    ok: bool
    diagnostic: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "diagnostic": self.diagnostic,
        }


# ── HealthCheckOutcome ─────────────────────────────


@dataclass
class HealthCheckOutcome:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def summary(self) -> str:
        lines = [
            "PRSM Enterprise Deployment — Health Check",
            "",
        ]
        for c in self.checks:
            symbol = "✓" if c.ok else "✗"
            lines.append(
                f"  [{symbol}] {c.name:50}  "
                f"{c.diagnostic}"
            )
        lines.append("")
        if self.ok:
            lines.append(
                "✓ All checks passed — deployment "
                "appears healthy."
            )
        else:
            failed = sum(1 for c in self.checks if not c.ok)
            lines.append(
                f"✗ {failed} check(s) failed — fix the "
                f"flagged items before serving real "
                f"traffic."
            )
        return "\n".join(lines)


# ── Individual checks ─────────────────────────────


def check_persistence_dir_writable(
    *, var_name: str, path: Optional[str],
) -> CheckResult:
    if path is None or not path.strip():
        return CheckResult(
            name=var_name, ok=True,
            diagnostic=(
                "not configured (env var unset; "
                "subsystem disabled)"
            ),
        )
    p = Path(path)
    # Create the leaf dir if missing — fresh-mount
    # deploys shouldn't fail just because a sub-directory
    # didn't exist yet
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return CheckResult(
            name=var_name, ok=False,
            diagnostic=(
                f"could not create directory {path!r}: "
                f"{type(e).__name__}: {e}"
            ),
        )
    # Probe writability with a temp file
    probe = p / f".prsm-health-probe-{uuid.uuid4().hex}"
    try:
        probe.write_text("ok")
        probe.unlink()
    except Exception as e:
        return CheckResult(
            name=var_name, ok=False,
            diagnostic=(
                f"directory {path!r} not writable: "
                f"{type(e).__name__}: {e}"
            ),
        )
    return CheckResult(
        name=var_name, ok=True,
        diagnostic=f"writable at {path!r}",
    )


def check_keypair_format(
    *, var_name: str, value: Optional[str],
    key_type: str,
) -> CheckResult:
    if key_type not in ("ed25519", "x25519"):
        raise ValueError(
            f"unknown key_type {key_type!r}; expected "
            f"'ed25519' or 'x25519'"
        )
    if value is None or not value.strip():
        return CheckResult(
            name=var_name, ok=True,
            diagnostic=(
                "not configured (env var unset; "
                "feature disabled)"
            ),
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as e:
        return CheckResult(
            name=var_name, ok=False,
            diagnostic=f"not valid base64: {e}",
        )
    if len(raw) != 32:
        return CheckResult(
            name=var_name, ok=False,
            diagnostic=(
                f"decoded length {len(raw)} != 32 bytes "
                f"(expected raw {key_type} privkey)"
            ),
        )
    # Confirm the key can actually load via the
    # cryptography lib — catches subtle issues like
    # invalid X25519 scalars that base64+length wouldn't
    # detect.
    try:
        if key_type == "ed25519":
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            Ed25519PrivateKey.from_private_bytes(raw)
        else:
            from cryptography.hazmat.primitives.asymmetric.x25519 import (
                X25519PrivateKey,
            )
            X25519PrivateKey.from_private_bytes(raw)
    except Exception as e:
        return CheckResult(
            name=var_name, ok=False,
            diagnostic=(
                f"cryptography rejected the privkey: "
                f"{e}"
            ),
        )
    return CheckResult(
        name=var_name, ok=True,
        diagnostic=(
            f"valid {key_type} 32-byte privkey "
            f"(loaded ok)"
        ),
    )


def check_subsystem_round_trip(
    *, subsystem: str, persist_dir: Optional[str],
) -> CheckResult:
    """Wire up the named subsystem at the configured path
    + do a small operation. If anything in the import
    chain or filesystem layer is broken, this surfaces
    the failure with a clear diagnostic."""
    valid_subsystems = (
        "fl_orchestrator", "pipeline_orchestrator",
    )
    if subsystem not in valid_subsystems:
        raise ValueError(
            f"unknown subsystem {subsystem!r}; expected "
            f"one of {valid_subsystems}"
        )
    if persist_dir is None or not persist_dir.strip():
        return CheckResult(
            name=f"{subsystem}_roundtrip", ok=True,
            diagnostic=(
                "not configured (persistence dir unset; "
                "subsystem disabled)"
            ),
        )
    try:
        if subsystem == "fl_orchestrator":
            from prsm.enterprise.federated_learning import (
                FederatedLearningOrchestrator,
            )
            orch = FederatedLearningOrchestrator(
                persist_dir=Path(persist_dir),
            )
            # Just instantiating is the test — if env+path+
            # imports all line up, this works
            _ = orch.list_jobs()
        elif subsystem == "pipeline_orchestrator":
            from prsm.compute.inference.pipeline_orchestrator import (
                PipelineInferenceOrchestrator,
            )
            from prsm.enterprise.federated_learning import (
                generate_worker_keypair,
            )
            # Need a privkey for construction. Use a
            # throwaway one — this is just a wiring check.
            priv, _ = generate_worker_keypair()
            orch = PipelineInferenceOrchestrator(
                orchestrator_privkey_b64=priv,
                persist_dir=Path(persist_dir),
            )
            _ = orch.list_jobs()
    except Exception as e:
        return CheckResult(
            name=f"{subsystem}_roundtrip", ok=False,
            diagnostic=(
                f"subsystem failed to wire: "
                f"{type(e).__name__}: {e}"
            ),
        )
    return CheckResult(
        name=f"{subsystem}_roundtrip", ok=True,
        diagnostic=(
            f"subsystem wired ok at {persist_dir!r}"
        ),
    )


# ── Aggregate ──────────────────────────────────────


_KEYPAIR_TYPE_FOR_VAR = {
    # Sprint 308b — worker signs gradients via Ed25519
    "PRSM_FEDERATED_WORKER_PRIVKEY": "ed25519",
    # Sprint 308c — orchestrator unseals transport-
    # encrypted gradients via X25519
    "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY": (
        "x25519"
    ),
    # Sprint 312 — pipeline orchestrator signs receipts
    # via Ed25519
    "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY": "ed25519",
}


_PERSISTENCE_DIR_VARS = (
    "PRSM_DISCLOSURE_INTAKE_DIR",
    "PRSM_INCIDENT_RESPONSE_DIR",
    "PRSM_UPGRADE_ORCHESTRATOR_DIR",
    "PRSM_CORP_CAPABILITY_DIR",
    "PRSM_FEDERATED_LEARNING_DIR",
    "PRSM_PIPELINE_ORCHESTRATOR_DIR",
)


def run_health_checks() -> HealthCheckOutcome:
    """Run every health check applicable to the current
    env. Returns a HealthCheckOutcome aggregating all
    results."""
    outcome = HealthCheckOutcome()

    # 1) Persistence-dir writability
    for var in _PERSISTENCE_DIR_VARS:
        outcome.checks.append(
            check_persistence_dir_writable(
                var_name=var,
                path=os.environ.get(var, "").strip() or None,
            )
        )

    # 2) Keypair format + load
    for var, key_type in _KEYPAIR_TYPE_FOR_VAR.items():
        outcome.checks.append(
            check_keypair_format(
                var_name=var,
                value=os.environ.get(var, "").strip() or None,
                key_type=key_type,
            )
        )

    # 3) Subsystem round-trip (only if persist dir set)
    outcome.checks.append(
        check_subsystem_round_trip(
            subsystem="fl_orchestrator",
            persist_dir=(
                os.environ.get(
                    "PRSM_FEDERATED_LEARNING_DIR", "",
                ).strip() or None
            ),
        )
    )
    outcome.checks.append(
        check_subsystem_round_trip(
            subsystem="pipeline_orchestrator",
            persist_dir=(
                os.environ.get(
                    "PRSM_PIPELINE_ORCHESTRATOR_DIR", "",
                ).strip() or None
            ),
        )
    )

    return outcome
