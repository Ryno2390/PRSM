"""
Phase 5 Test Suite Completeness Verification
============================================

Ensures no test files have stale module-level skips.
All remaining module-level skips must reference a specific missing item,
not a generic 'Module dependencies not yet fully implemented' message.
"""

import ast
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent

# Files that are intentionally deferred with specific reasons
KNOWN_DEFERRED = {
    "test_real_data_integration.py",
    "test_phase7_integration.py",
    "test_full_spectrum_integration.py",
    "test_hybrid_architecture_integration.py",
    "test_integration_suite_runner.py",
    "test_ftns_concurrency_integration.py",
    "test_openai_free_tier.py",
    "test_openai_real_integration.py",
    "test_openai_integration.py",
    "test_governance.py",
    "test_150k_papers_provenance.py",
    "test_nwtn_provenance_integration.py",
    "test_phase5_completeness.py",  # This file itself
    "test_consensus_integration.py",
    "simple_performance_test.py",
    "test_runner.py",
    "test_performance_benchmarks_alt.py",
}


def test_no_vague_module_level_skips():
    """
    All remaining module-level skips must reference a specific missing item,
    not a generic 'Module dependencies not yet fully implemented' message.
    """
    vague_message = "Module dependencies not yet fully implemented"
    offenders = []

    for f in REPO_ROOT.glob("tests/**/*.py"):
        # Skip known deferred files
        if f.name in KNOWN_DEFERRED:
            continue

        try:
            text = f.read_text()
        except Exception:
            continue

        if vague_message in text and "allow_module_level=True" in text:
            offenders.append(str(f.relative_to(REPO_ROOT)))

    assert not offenders, (
        f"These test files still have vague module-level skips:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_known_deferred_have_specific_messages():
    """
    Verify that all known deferred files have specific skip messages.
    """
    generic_patterns = [
        "Module dependencies not yet fully implemented",
    ]

    for filename in KNOWN_DEFERRED:
        # Skip the test file itself (it references the pattern in docstrings)
        if filename == "test_phase5_completeness.py":
            continue

        filepath = REPO_ROOT / "tests" / filename
        if not filepath.exists():
            # Check in subdirectories
            matches = list(REPO_ROOT.glob(f"tests/**/{filename}"))
            if not matches:
                continue
            filepath = matches[0]

        try:
            text = filepath.read_text()
        except Exception:
            continue

        # If file has a module-level skip, it should NOT have generic message
        if "allow_module_level=True" in text:
            for pattern in generic_patterns:
                assert pattern not in text, (
                    f"Deferred file {filename} has generic skip message. "
                    f"Should have specific reason for deferral."
                )


# test_breakthrough_mode_exports removed in v1.6.1:
# prsm.compute.nwtn.breakthrough_modes was deleted in v1.6.0 (legacy AGI framework).


def test_ftns_service_constants():
    """Verify FTNS service constants are exported."""
    from prsm.economy.tokenomics.ftns_service import (
        BASE_NWTN_FEE,
        CONTEXT_UNIT_COST,
        ARCHITECT_DECOMPOSITION_COST,
        COMPILER_SYNTHESIS_COST,
        AGENT_COSTS,
        REWARD_PER_MB,
        MODEL_CONTRIBUTION_REWARD,
        SUCCESSFUL_TEACHING_REWARD,
    )

    assert BASE_NWTN_FEE > 0
    assert CONTEXT_UNIT_COST > 0
    assert len(AGENT_COSTS) > 0


def test_budget_manager_classes():
    """Verify budget manager classes are available."""
    from decimal import Decimal
    from prsm.economy.tokenomics.ftns_budget_manager import (
        FTNSBudgetManager,
        SpendingCategory,
        BudgetStatus,
        BudgetAllocation,
        BudgetPrediction,
        BudgetAlert,
    )

    # Test BudgetPrediction can be instantiated
    prediction = BudgetPrediction(
        estimated_total_cost=Decimal("10.0"),
        confidence_score=0.8,
        query_complexity=0.5,
        category_estimates={}
    )
    assert prediction.estimated_total_cost == Decimal("10.0")

    # Test BudgetAlert can be instantiated
    alert = BudgetAlert(
        alert_id="test-alert",
        user_id="test-user",
        alert_type="warning",
        message="Test alert",
        current_spend=Decimal("50.0"),
        budget_limit=Decimal("100.0"),
        percentage_used=0.5
    )
    assert alert.alert_id == "test-alert"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
