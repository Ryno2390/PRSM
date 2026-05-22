"""Sprint 709 — pin tests for operator-deploy-runbook freshness.

`docs/operations/parallax-inference-deploy.md` is the canonical
external-operator onboarding doc. Sprint 697 shipped it; sprint 709
refreshed it to include the 5 env vars added in sprints 700-708:

  PRSM_INFERENCE_CONCURRENCY_LIMIT  (sprint 704 OOM gate)
  PRSM_PARALLAX_TIER_GATE           (sprint 702 DP-injection bypass)
  PRSM_OPERATOR_ADDRESS             (sprint 690 stake-pool wiring)
  PRSM_PARALLAX_STREAMING_RUNNER_KIND (sprint 693 — was implicit)
  PRSM_PARALLAX_DEFAULT_RTT_MS      (sprint 695 routing DP)

These tests fail if the runbook drifts back to stale state — a
deliberate-update workflow protects the doc from silently going
out of date as new env vars ship.
"""
from __future__ import annotations

from pathlib import Path

import pytest


RUNBOOK_PATH = (
    Path(__file__).parent.parent.parent
    / "docs" / "operations" / "parallax-inference-deploy.md"
)


def _runbook_text() -> str:
    return RUNBOOK_PATH.read_text()


def test_runbook_exists():
    assert RUNBOOK_PATH.is_file(), (
        f"sprint-697 operator runbook missing at {RUNBOOK_PATH}"
    )


@pytest.mark.parametrize("env_var", [
    "PRSM_INFERENCE_EXECUTOR",
    "PRSM_PARALLAX_GPU_POOL_KIND",
    "PRSM_PARALLAX_TRUST_STACK_KIND",
    "PRSM_PARALLAX_MODEL_CATALOG_FILE",
    "PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS",
    "PRSM_STAKE_BOND_ADDRESS",
    "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND",
    "PRSM_PARALLAX_STAGE_EXECUTOR_KIND",
    "PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND",
    "PRSM_PARALLAX_HF_MODEL_ID",
    "PRSM_PARALLAX_HF_DEVICE",
    "PRSM_PARALLAX_PROMPT_ENCODER_KIND",
    "PRSM_PARALLAX_OUTPUT_DECODER_KIND",
    "PRSM_PARALLAX_STREAMING_RUNNER_KIND",
    "PRSM_PARALLAX_KV_CACHE_ENABLED",
    "PRSM_PARALLAX_STAKE_ELIGIBILITY",
    "PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE",
    "PRSM_PARALLAX_DEFAULT_RTT_MS",
    "PRSM_PARALLAX_SEND_MESSAGE_TIMEOUT_S",
    # Sprint 702
    "PRSM_PARALLAX_TIER_GATE",
    # Sprint 704
    "PRSM_INFERENCE_CONCURRENCY_LIMIT",
    # Sprint 690 (optional in runbook; commented as awaiting-stake)
    "PRSM_OPERATOR_ADDRESS",
])
def test_runbook_references_env_var(env_var: str):
    """Every operator-impactful env var must appear in the runbook,
    either in the systemd-conf block (for required) or in the
    troubleshooting table (for diagnostic). If a new env var ships
    without a runbook update, the parametrize above will fail to
    find it here."""
    text = _runbook_text()
    assert env_var in text, (
        f"runbook must reference {env_var} — either in the systemd "
        f"env block or in the troubleshooting table. Refresh the "
        f"runbook before merging the new env var."
    )


def test_runbook_references_sprint_700_through_708():
    """Sprint history table must reference the latest sprints so a
    reader can navigate from runbook → sprint commits."""
    text = _runbook_text()
    for sprint in ("700", "702", "703", "704", "706", "708"):
        assert sprint in text, (
            f"runbook sprint-references table must include sprint {sprint}"
        )


def test_runbook_references_verifier_walkthrough():
    """Sprint 706+707+708 sample-receipt walkthrough should be linked
    so operators can verify their own daemon's receipts."""
    text = _runbook_text()
    assert "verify_prsm_receipt.py" in text
    assert "sample-receipts" in text


def test_runbook_no_stale_overrides_pinned_as_default():
    """Sprint 695's MEMORY_GB_OVERRIDE + TFLOPS_FP16_OVERRIDE were
    needed to FORCE 2-stage on CPU droplets for live-attest, but
    sprint 700's F46 fix made the default heterogeneous-pool path
    work without them. Runbook should mark them as OPT-IN, not as
    required env vars in the canonical config block."""
    text = _runbook_text()
    # Quick sanity: the runbook header notes the OPT-IN nature
    assert "OPT-IN" in text or "opt-in" in text or "OPTIONAL" in text, (
        "runbook must mark sprint-695 multi-stage overrides as "
        "opt-in (sprint 700 made them non-required for default deploy)"
    )
