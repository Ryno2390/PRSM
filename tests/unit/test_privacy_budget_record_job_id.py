"""Sprint 263 — PrivacyBudgetTracker record_spend signature fix.

Pre-fix the api.py callsites passed job_id positionally as the
third arg where the tracker expected model_id, so audit records
had job_id strings in the model_id field and the actual model_id
was lost.

This sprint:
  - Adds job_id field to PrivacySpend
  - Changes record_spend(epsilon, operation, ...) so position 3
    is now job_id; model_id moves to kwarg
  - Preserves backwards-compat for callers passing 3 positional
    args (their value lands in the correct job_id slot)
"""
from __future__ import annotations

import pytest

from prsm.security.privacy_budget import (
    PrivacyBudgetTracker, PrivacySpend,
)


def test_privacy_spend_has_job_id_field():
    s = PrivacySpend(
        epsilon=1.0, operation="inference",
        job_id="job-abc", model_id="m1",
    )
    assert s.job_id == "job-abc"
    assert s.model_id == "m1"


def test_record_spend_positional_arg3_is_now_job_id():
    """Pre-263 callers passed (epsilon, "inference", job_id);
    post-263 the third positional arg lands in the job_id slot,
    correctly. model_id is now kwarg-only."""
    t = PrivacyBudgetTracker(max_epsilon=100.0)
    t.record_spend(8.0, "inference", "job-xyz")
    report = t.get_audit_report()
    spend = report["spends"][0]
    assert spend["job_id"] == "job-xyz"
    assert spend["model_id"] == ""  # not supplied


def test_record_spend_with_explicit_model_id():
    t = PrivacyBudgetTracker(max_epsilon=100.0)
    t.record_spend(
        8.0, "inference", "job-xyz",
        model_id="mock-llama-3-8b",
    )
    spend = t.get_audit_report()["spends"][0]
    assert spend["job_id"] == "job-xyz"
    assert spend["model_id"] == "mock-llama-3-8b"


def test_audit_report_surfaces_timestamp():
    """Sprint 263 also adds timestamp to the audit dict for
    operator log-correlation."""
    t = PrivacyBudgetTracker(max_epsilon=100.0)
    t.record_spend(8.0, "inference", "job-x")
    spend = t.get_audit_report()["spends"][0]
    assert "timestamp" in spend
    assert isinstance(spend["timestamp"], float)
    assert spend["timestamp"] > 0


def test_negative_epsilon_still_rejected():
    """Pre-263 guard preserved."""
    t = PrivacyBudgetTracker(max_epsilon=100.0)
    assert t.record_spend(-5.0, "inference", "job-x") is False
    assert t.get_audit_report()["num_operations"] == 0


def test_nan_epsilon_still_rejected():
    import math
    t = PrivacyBudgetTracker(max_epsilon=100.0)
    assert t.record_spend(math.nan, "inference", "job-x") is False


def test_inf_epsilon_still_rejected():
    import math
    t = PrivacyBudgetTracker(max_epsilon=100.0)
    assert t.record_spend(math.inf, "inference", "job-x") is False
