"""Sprint 481 — audit checklist startup-warning demotion pin.

Pre-fix: SecurityCheck.__post_init__ emitted WARNING for every
instance with `auto_check=True, check_function=None`. The
DEFAULT_CHECKS list contains 36 such intentional scaffolds
(each waiting on a concrete check_function impl) → 36 WARNING
lines per daemon startup.

These are not real faults — they're known TODO scaffolding. A
WARNING-level signal on intentional state pollutes operator
log dashboards keyed on WARNING count.

Sprint 481 demoted to DEBUG. Operators can still surface the
pending-impl roster via SecurityAuditor's audit-trail API.

These pins fire if a future refactor re-introduces WARNING.
"""
from __future__ import annotations

import logging

from prsm.security.audit_checklist import (
    SecurityCheck,
    CheckCategory,
    CheckSeverity,
)


def test_scaffold_check_does_not_warn(caplog):
    """auto_check=True + check_function=None scaffold must NOT
    emit WARNING — operator log dashboards alarm on WARNING
    count and pre-fix would alarm spuriously every restart."""
    with caplog.at_level(logging.DEBUG):
        SecurityCheck(
            check_id="test_scaffold",
            name="Test Scaffold",
            description="placeholder",
            category=CheckCategory.AUTHENTICATION,
            severity=CheckSeverity.HIGH,
            auto_check=True,
            check_function=None,
        )
    warning_logs = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    assert warning_logs == [], (
        f"intentional scaffold must not warn; got: "
        f"{[r.getMessage() for r in warning_logs]}"
    )


def test_audit_checklist_source_uses_debug_not_warning():
    """Source-of-truth pin: __post_init__ in
    prsm/security/audit_checklist.py must call logger.debug
    on the scaffold-detected branch, NOT logger.warning.
    Pre-fix used warning; the fix is at this exact line.

    (Caplog can't reliably capture structlog output — pin
    the source instead.)"""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "security" / "audit_checklist.py"
    ).read_text()
    # Find the __post_init__ block.
    idx = src.find("def __post_init__")
    assert idx >= 0
    body = src[idx:idx + 1200]
    assert "logger.debug" in body, (
        "audit-checklist scaffold log must use logger.debug, "
        "not logger.warning"
    )
    # The buggy pattern must NOT reappear.
    assert "logger.warning" not in body, (
        "sprint 481 regression: logger.warning was "
        "re-introduced in __post_init__ — would re-pollute "
        "operator logs with 36 lines per startup"
    )


def test_check_with_function_does_not_log():
    """Sanity: a properly-implemented check (with a real
    check_function) emits no scaffold-related log."""
    import logging as _logging
    handler_records = []

    class _Capture(_logging.Handler):
        def emit(self, record):
            handler_records.append(record)

    cap = _Capture()
    cap.setLevel(_logging.DEBUG)
    root = _logging.getLogger("prsm.security.audit_checklist")
    root.addHandler(cap)
    try:
        SecurityCheck(
            check_id="test_with_fn",
            name="Test With Function",
            description="real impl",
            category=CheckCategory.AUTHENTICATION,
            severity=CheckSeverity.HIGH,
            auto_check=True,
            check_function=lambda: True,
        )
    finally:
        root.removeHandler(cap)
    scaffold_logs = [
        r for r in handler_records
        if "pending check_function" in r.getMessage()
    ]
    assert scaffold_logs == [], (
        "implemented check should not emit scaffold log"
    )
