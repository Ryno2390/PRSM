"""Sprint 318 — operator deployment foundation.

The §7 enterprise + federated inference stack now spans
~15 env vars across keypairs, persistence paths, and
endpoints. Each sprint shipped its env var in isolation —
operators wiring an end-to-end deployment needed a single
place that:
  1. Enumerates every required + optional env var
  2. Validates a current deployment (status / missing
     vars / malformed values)
  3. Generates a starter config with fresh keypairs

This module is that single place. The CLI in
prsm/enterprise/bringup_cli.py wraps it with `status`
and `generate` subcommands; sprint 318a layers a
Dockerfile + docker-compose on top.

The env-var catalog is the load-bearing primitive — it
documents what an operator must set, which sprint shipped
it, and what default makes sense. Operators copy-paste
the rendered .env file, fill in vendor-specific values
(RPC URL, etc.), and have a deployable config.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


class EnterpriseConfigValidationError(Exception):
    """Raised when validate() detects a misconfigured or
    incomplete deployment."""


# ── Spec catalog ────────────────────────────────────


@dataclass(frozen=True)
class EnvVarSpec:
    """A single env var the §7 enterprise + federated
    inference stack depends on."""

    name: str
    sprint: str  # which sprint introduced it
    description: str
    required: bool
    is_keypair: bool = False  # X25519/Ed25519 32-byte base64
    is_path: bool = False  # filesystem directory
    default_suffix: Optional[str] = None  # for path generation


_SPECS: List[EnvVarSpec] = [
    # ── Keypair env vars ────────────────────────────
    EnvVarSpec(
        name="PRSM_FEDERATED_WORKER_PRIVKEY",
        sprint="308b",
        description=(
            "Ed25519 privkey (base64, 32 bytes) used by "
            "this node when acting as a federated-learning "
            "WORKER (signs gradient updates). Pair with "
            "the worker's pubkey registered on the "
            "orchestrator via /admin/federated/worker-key."
        ),
        required=True, is_keypair=True,
    ),
    EnvVarSpec(
        name="PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
        sprint="308c",
        description=(
            "X25519 privkey (base64, 32 bytes) used by "
            "this node when acting as a federated-learning "
            "ORCHESTRATOR to unseal encrypted gradients "
            "from workers. Required only if jobs declare "
            "transport_pubkey_b64."
        ),
        required=False, is_keypair=True,
    ),
    EnvVarSpec(
        name="PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY",
        sprint="312",
        description=(
            "Ed25519 privkey (base64, 32 bytes) used by "
            "this node when acting as the PIPELINE "
            "INFERENCE ORCHESTRATOR (signs "
            "PipelineInferenceReceipts). Pair with the "
            "orchestrator's pubkey for verifier "
            "distribution."
        ),
        required=True, is_keypair=True,
    ),
    # ── Persistence dirs ────────────────────────────
    EnvVarSpec(
        name="PRSM_DISCLOSURE_INTAKE_DIR",
        sprint="300",
        description=(
            "Filesystem dir for the §14 responsible-"
            "disclosure intake's persistent state."
        ),
        required=False, is_path=True,
        default_suffix="disclosure",
    ),
    EnvVarSpec(
        name="PRSM_INCIDENT_RESPONSE_DIR",
        sprint="301",
        description=(
            "Filesystem dir for the §14 incident-response "
            "playbook's lifecycle records."
        ),
        required=False, is_path=True,
        default_suffix="incident",
    ),
    EnvVarSpec(
        name="PRSM_UPGRADE_ORCHESTRATOR_DIR",
        sprint="303",
        description=(
            "Filesystem dir for the §14 UUPS upgrade "
            "orchestrator's proposal records."
        ),
        required=False, is_path=True,
        default_suffix="upgrade",
    ),
    EnvVarSpec(
        name="PRSM_CORP_CAPABILITY_DIR",
        sprint="306",
        description=(
            "Filesystem dir for the §7 layer-2 $CORP "
            "capability store (issuers, ledger, "
            "consumed counters)."
        ),
        required=False, is_path=True,
        default_suffix="corp",
    ),
    EnvVarSpec(
        name="PRSM_FEDERATED_LEARNING_DIR",
        sprint="308",
        description=(
            "Filesystem dir for the §7 federated-learning "
            "orchestrator's job + round records."
        ),
        required=True, is_path=True,
        default_suffix="fl",
    ),
    EnvVarSpec(
        name="PRSM_PIPELINE_ORCHESTRATOR_DIR",
        sprint="312",
        description=(
            "Filesystem dir for the pipeline inference "
            "orchestrator's job + receipt records."
        ),
        required=True, is_path=True,
        default_suffix="pipeline",
    ),
    # ── Toggles / addresses ─────────────────────────
    EnvVarSpec(
        name="PRSM_PIPELINE_STAGE_RUNNER_ENABLED",
        sprint="313",
        description=(
            "Set to '1' to enable this node to serve as "
            "a REMOTE PIPELINE STAGE WORKER for other "
            "nodes' orchestrators (exposes "
            "/compute/inference/pipeline/stage). "
            "Default disabled."
        ),
        required=False,
    ),
    EnvVarSpec(
        name="PRSM_INSURANCE_FUND_ADDRESS",
        sprint="299",
        description=(
            "Foundation insurance-fund wallet address "
            "(0x-prefixed). The §14 insurance fund "
            "tracker reads on-chain balance from this; "
            "compose_recovery_transfer_tx targets it."
        ),
        required=False,
    ),
    EnvVarSpec(
        name="PRSM_NETWORK",
        sprint="(pre-298)",
        description=(
            "Network identifier — 'base-mainnet' or "
            "'base-sepolia'. Resolves to canonical RPC + "
            "contract addresses via prsm.config.networks."
        ),
        required=False,
    ),
]


def list_env_var_specs() -> List[EnvVarSpec]:
    return list(_SPECS)


# ── EnterpriseConfig ────────────────────────────────


@dataclass
class EnterpriseConfig:
    values: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EnterpriseConfig":
        loaded: Dict[str, str] = {}
        for spec in _SPECS:
            raw = os.environ.get(spec.name, "").strip()
            if raw:
                loaded[spec.name] = raw
        return cls(values=loaded)

    def get(self, name: str) -> Optional[str]:
        return self.values.get(name)

    def validate(self) -> None:
        missing: List[str] = []
        for spec in _SPECS:
            if not spec.required:
                continue
            value = self.values.get(spec.name)
            if not value:
                missing.append(spec.name)
        if missing:
            raise EnterpriseConfigValidationError(
                "missing required env var(s): "
                + ", ".join(missing)
            )
        # Per-var format validation for set values
        for spec in _SPECS:
            value = self.values.get(spec.name)
            if not value:
                continue
            if spec.is_keypair:
                try:
                    raw = base64.b64decode(
                        value, validate=True,
                    )
                except Exception as e:
                    raise EnterpriseConfigValidationError(
                        f"{spec.name} privkey is not "
                        f"valid base64: {e}"
                    )
                if len(raw) != 32:
                    raise EnterpriseConfigValidationError(
                        f"{spec.name} privkey must be "
                        f"32 bytes (got {len(raw)})"
                    )

    def summary(self) -> str:
        """Human-readable per-var status. Operator-facing
        — gets printed by `bringup status`."""
        lines = [
            "PRSM Enterprise Deployment — Configuration "
            "Status",
            "",
        ]
        for spec in _SPECS:
            value = self.values.get(spec.name)
            tag = "REQUIRED" if spec.required else "optional"
            if value:
                masked = (
                    value[:8] + "…"
                    if len(value) > 8 else value
                )
                status = f"✓ set ({masked})"
            else:
                status = (
                    "✗ MISSING"
                    if spec.required else "· unset"
                )
            lines.append(
                f"  [{tag:8}] {spec.name:50}  {status}"
            )
            lines.append(
                f"             (sprint {spec.sprint}): "
                f"{spec.description[:80]}..."
                if len(spec.description) > 80
                else (
                    f"             (sprint {spec.sprint}): "
                    f"{spec.description}"
                )
            )
        return "\n".join(lines)


# ── Starter config generation ──────────────────────


def generate_starter_config(
    base_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Produce a complete starter config dict that
    validates out of the box. Generates fresh keypairs
    for every keypair var + creates persistence-dir
    values under base_dir.

    The output is suitable for piping to a .env file via
    render_env_file().
    """
    base = (
        Path(base_dir) if base_dir is not None
        else Path("/var/lib/prsm")
    )
    out: Dict[str, str] = {}

    for spec in _SPECS:
        if spec.is_keypair:
            from prsm.enterprise.federated_learning import (
                generate_worker_keypair,
            )
            # Both Ed25519 (worker/orchestrator) and X25519
            # (transport) keypair env vars are 32-byte
            # base64; generate_worker_keypair returns
            # Ed25519. For X25519-flavored vars, use the
            # X25519 generator from sprint 308c.
            if "TRANSPORT" in spec.name:
                from prsm.enterprise.federated_learning import (
                    generate_transport_keypair,
                )
                priv, _ = generate_transport_keypair()
            else:
                priv, _ = generate_worker_keypair()
            out[spec.name] = priv
        elif spec.is_path and spec.default_suffix:
            out[spec.name] = str(
                base / spec.default_suffix,
            )
        # Toggles + addresses: leave unset; operator fills
        # them in per their deployment context
    return out


# ── .env file rendering ─────────────────────────────


def render_env_file(values: Dict[str, str]) -> str:
    """Render a .env-style file template with one block
    per spec. Variables in `values` get active
    assignments; ones not in `values` are commented out
    so the operator can see + uncomment as needed."""
    lines: List[str] = [
        "# PRSM Enterprise Deployment — generated by "
        "prsm-enterprise-bringup",
        "# Each variable below carries the sprint that "
        "introduced it + a description.",
        "",
    ]
    for spec in _SPECS:
        tag = "[REQUIRED]" if spec.required else "[optional]"
        lines.append(
            f"# {tag} {spec.name} (sprint {spec.sprint})"
        )
        # Word-wrap description at ~70 chars
        desc = spec.description
        while desc:
            chunk = desc[:70]
            if len(desc) > 70:
                # Break at last space
                last_space = chunk.rfind(" ")
                if last_space > 0:
                    chunk = desc[:last_space]
                    desc = desc[last_space + 1:]
                else:
                    desc = desc[70:]
            else:
                desc = ""
            lines.append(f"#   {chunk}")
        value = values.get(spec.name)
        if value:
            lines.append(f"{spec.name}={value}")
        else:
            lines.append(f"# {spec.name}=")
        lines.append("")
    return "\n".join(lines)
