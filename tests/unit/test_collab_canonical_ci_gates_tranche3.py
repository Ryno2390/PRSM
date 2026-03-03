"""P2 Tranche 3 CI-style regression gates for canonical collaboration interfaces."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_CANONICAL_SUITES = [
    "tests/unit/test_collab_observability_tranche1.py",
    "tests/unit/test_collab_canonical_boundaries_tranche1.py",
    "tests/unit/test_collab_canonical_boundaries_tranche2.py",
    "tests/unit/test_collab_canonical_ci_gates_tranche3.py",
]

_PRIMARY_CANONICAL_DISPATCH_SUITES = [
    "tests/security/test_sprint4_collab_bridge.py",
    "tests/unit/test_collab_canonical_boundaries_tranche1.py",
    "tests/unit/test_collab_canonical_boundaries_tranche2.py",
]

_CANONICAL_COLLABORATION_OWNERSHIP_SUITES = [
    "tests/security/test_sprint4_collaboration.py",
    "tests/security/test_sprint4_collab_bridge.py",
    "tests/unit/test_collab_observability_tranche1.py",
    "tests/unit/test_collab_canonical_boundaries_tranche1.py",
]


def _read(relative_path: str) -> str:
    return (_ROOT / relative_path).read_text(encoding="utf-8")


def test_required_canonical_collaboration_suites_exist() -> None:
    """Required canonical collaboration suites must exist and be discoverable by pytest."""
    missing = [path for path in _REQUIRED_CANONICAL_SUITES if not (_ROOT / path).exists()]
    assert not missing, f"Missing required canonical collaboration suite(s): {missing}"


def test_collaboration_ownership_tests_keep_canonical_manager_dispatch_as_primary_path() -> None:
    """Primary collaboration ownership suites must stay on CollaborationManager dispatch path."""
    for relative_path in _PRIMARY_CANONICAL_DISPATCH_SUITES:
        source = _read(relative_path)
        assert "manager.dispatch_session(" in source or "dispatch_session(" in source, (
            f"{relative_path} must exercise canonical CollaborationManager.dispatch_session path"
        )


def test_collaboration_ownership_suites_do_not_use_federation_execution_as_primary_path() -> None:
    """Collaboration ownership suites must not rely on federation execution entrypoints."""
    for relative_path in _CANONICAL_COLLABORATION_OWNERSHIP_SUITES:
        source = _read(relative_path)
        assert "coordinate_distributed_execution(" not in source, (
            f"{relative_path} must not use federation coordinate_distributed_execution as collaboration primary path"
        )


def test_federation_collaboration_entrypoints_remain_compatibility_only_and_gated() -> None:
    """Compatibility-only federation collaboration entrypoints must stay fenced in boundary tests."""
    tranche2_source = _read("tests/unit/test_collab_canonical_boundaries_tranche2.py")

    assert "test_federation_p2p_execution_entrypoint_emits_compatibility_fence_warning" in tranche2_source
    assert "test_enhanced_federation_execution_entrypoint_emits_compatibility_fence_warning" in tranche2_source
    assert "Compatibility-only collaboration entrypoint used" in tranche2_source
    assert "CollaborationManager.dispatch_session" in tranche2_source


def test_canonical_boundary_suites_exercise_manager_bridge_dispatch() -> None:
    """Canonical boundary suites must continue asserting manager bridge dispatch coverage."""
    tranche1_source = _read("tests/unit/test_collab_canonical_boundaries_tranche1.py")
    tranche2_source = _read("tests/unit/test_collab_canonical_boundaries_tranche2.py")

    assert tranche1_source.count("manager.dispatch_session(") >= 3
    assert "test_dispatch_remains_on_canonical_manager_bridge_not_federation" in tranche2_source
    assert tranche2_source.count("manager.dispatch_session(") >= 1
