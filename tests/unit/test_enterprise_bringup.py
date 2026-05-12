"""Sprint 318 — operator deployment foundation.

The §7 enterprise + federated inference stack now spans
~15 env vars across keypairs, persistence paths, and
endpoints. Each sprint shipped its env var in isolation —
operators wiring an end-to-end deployment need a single
place that:
  1. Enumerates every required + optional env var
  2. Validates a current deployment (status / missing
     vars / malformed values)
  3. Generates a starter config with fresh keypairs

This module is that single place. Sprint 318a layers
Dockerfile + docker-compose on top; this sprint ships the
Python primitives + CLI that the Docker bringup will
invoke.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from prsm.enterprise.bringup import (
    EnterpriseConfig,
    EnterpriseConfigValidationError,
    EnvVarSpec,
    generate_starter_config,
    list_env_var_specs,
    render_env_file,
)


# ── EnvVarSpec catalog ─────────────────────────────


def test_env_var_specs_enumerate_keypair_vars():
    specs = {s.name: s for s in list_env_var_specs()}
    # The five keypair env vars introduced across the §7
    # enterprise + federated-inference arc
    for name in (
        "PRSM_FEDERATED_WORKER_PRIVKEY",
        "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
        "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY",
    ):
        assert name in specs, (
            f"env var spec missing: {name}"
        )


def test_env_var_specs_enumerate_persistence_dirs():
    specs = {s.name: s for s in list_env_var_specs()}
    for name in (
        "PRSM_DISCLOSURE_INTAKE_DIR",
        "PRSM_INCIDENT_RESPONSE_DIR",
        "PRSM_UPGRADE_ORCHESTRATOR_DIR",
        "PRSM_CORP_CAPABILITY_DIR",
        "PRSM_FEDERATED_LEARNING_DIR",
        "PRSM_PIPELINE_ORCHESTRATOR_DIR",
    ):
        assert name in specs


def test_env_var_specs_marked_required_or_optional():
    """Every spec must declare whether it's required for
    a base enterprise deployment. Operators rely on the
    distinction when reading status output."""
    for spec in list_env_var_specs():
        assert isinstance(spec.required, bool)


def test_env_var_specs_have_descriptions():
    """Each spec carries an operator-facing description
    so `bringup status` prints something useful."""
    for spec in list_env_var_specs():
        assert spec.description
        assert len(spec.description) > 10


# ── EnterpriseConfig.from_env ──────────────────────


def test_from_env_loads_known_vars(monkeypatch):
    monkeypatch.setenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", "AAAA",
    )
    monkeypatch.setenv(
        "PRSM_FEDERATED_LEARNING_DIR", "/tmp/fl",
    )
    cfg = EnterpriseConfig.from_env()
    assert cfg.get("PRSM_FEDERATED_WORKER_PRIVKEY") == (
        "AAAA"
    )
    assert cfg.get("PRSM_FEDERATED_LEARNING_DIR") == (
        "/tmp/fl"
    )


def test_from_env_unset_returns_none(monkeypatch):
    monkeypatch.delenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", raising=False,
    )
    cfg = EnterpriseConfig.from_env()
    assert cfg.get("PRSM_FEDERATED_WORKER_PRIVKEY") is None


def test_from_env_ignores_empty_strings(monkeypatch):
    """An empty string env value is the same as unset for
    deployment purposes — refuse to treat '' as a valid
    privkey."""
    monkeypatch.setenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", "",
    )
    cfg = EnterpriseConfig.from_env()
    assert cfg.get("PRSM_FEDERATED_WORKER_PRIVKEY") is None


# ── EnterpriseConfig.validate ──────────────────────


def test_validate_reports_missing_required():
    cfg = EnterpriseConfig(values={})
    with pytest.raises(
        EnterpriseConfigValidationError,
    ) as exc_info:
        cfg.validate()
    msg = str(exc_info.value)
    # The error must enumerate which required vars are
    # missing — operator can't fix what they can't see
    assert "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY" in msg


def test_validate_passes_when_required_set():
    """With every required var set to plausible values
    (valid 32-byte base64 for keypair vars; any string
    for others), validate() doesn't raise."""
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    required = {}
    for s in list_env_var_specs():
        if not s.required:
            continue
        required[s.name] = (
            valid_key if s.is_keypair else "x"
        )
    cfg = EnterpriseConfig(values=required)
    cfg.validate()  # no raise


def test_validate_optional_vars_unset_doesnt_fail():
    """Optional env vars unset is fine; only required
    ones gate validation."""
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    required = {}
    for s in list_env_var_specs():
        if not s.required:
            continue
        required[s.name] = (
            valid_key if s.is_keypair else "x"
        )
    cfg = EnterpriseConfig(values=required)
    cfg.validate()


def test_validate_keypair_format_when_present():
    """If a keypair env var IS set, its value must be a
    valid base64-decoded 32-byte string. A malformed
    privkey now is operator confusion that will silently
    explode at request time — refuse early."""
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    base_required = {
        s.name: valid_key if s.is_keypair else "x"
        for s in list_env_var_specs() if s.required
    }
    base_required["PRSM_FEDERATED_WORKER_PRIVKEY"] = (
        "not-valid-key"
    )
    cfg = EnterpriseConfig(values=base_required)
    with pytest.raises(
        EnterpriseConfigValidationError,
        match="privkey|base64|32",
    ):
        cfg.validate()


def test_validate_keypair_accepts_valid_base64_32():
    """A real X25519 / Ed25519 32-byte privkey value
    encodes to 44 chars of base64 (with padding). Valid
    keypair values must not trip validation."""
    valid_key = base64.b64encode(b"\x00" * 32).decode()
    required_with_key = {
        s.name: valid_key if s.is_keypair else "x"
        for s in list_env_var_specs() if s.required
    }
    cfg = EnterpriseConfig(values=required_with_key)
    cfg.validate()  # no raise


# ── EnterpriseConfig.summary ───────────────────────


def test_summary_reports_per_var_status():
    cfg = EnterpriseConfig(values={
        "PRSM_FEDERATED_WORKER_PRIVKEY": "set",
    })
    s = cfg.summary()
    assert "PRSM_FEDERATED_WORKER_PRIVKEY" in s
    # Should distinguish set vs missing visually
    assert "set" in s.lower() or "✓" in s or "OK" in s
    assert (
        "missing" in s.lower()
        or "unset" in s.lower()
        or "MISSING" in s
        or "✗" in s
    )


def test_summary_distinguishes_required_from_optional():
    cfg = EnterpriseConfig(values={})
    s = cfg.summary()
    # Required vars should be flagged differently from
    # optional ones so operators know which ones gate
    # bringup
    assert (
        "required" in s.lower()
        or "REQUIRED" in s
    )


# ── generate_starter_config ────────────────────────


def test_generate_starter_produces_valid_keypairs():
    starter = generate_starter_config()
    # Every keypair env var must be set to a valid 32-byte
    # base64 value
    for spec in list_env_var_specs():
        if "PRIVKEY" not in spec.name:
            continue
        if spec.name not in starter:
            continue  # spec optional + not generated
        raw = base64.b64decode(starter[spec.name])
        assert len(raw) == 32


def test_generate_starter_creates_persistence_paths(tmp_path):
    """generate_starter_config(base_dir=...) sets each
    persistence-dir env var to a path under base_dir."""
    starter = generate_starter_config(base_dir=tmp_path)
    for spec in list_env_var_specs():
        if not spec.name.endswith("_DIR"):
            continue
        if spec.name not in starter:
            continue
        path = Path(starter[spec.name])
        # All persistence dirs live under base_dir
        assert str(tmp_path) in str(path)


def test_generate_starter_passes_validation():
    """A starter config should be a valid deployment on
    its own — operators copy-paste it and bringup
    succeeds without further edits to required fields."""
    starter = generate_starter_config()
    cfg = EnterpriseConfig(values=starter)
    cfg.validate()


def test_generate_starter_keypairs_unique_per_call():
    """Two starter configs produced sequentially must
    have DIFFERENT keypairs (each operator/each deploy
    gets its own keys)."""
    a = generate_starter_config()
    b = generate_starter_config()
    for spec in list_env_var_specs():
        if "PRIVKEY" not in spec.name:
            continue
        if spec.name not in a or spec.name not in b:
            continue
        assert a[spec.name] != b[spec.name]


# ── render_env_file ─────────────────────────────────


def test_render_env_file_includes_all_specs():
    """A rendered .env file template should have a line
    per known env var, with comments + the value."""
    starter = generate_starter_config()
    rendered = render_env_file(starter)
    for spec in list_env_var_specs():
        assert spec.name in rendered


def test_render_env_file_has_comments():
    """Each line should be prefixed with a comment line
    carrying the spec's description — operator reading
    the file knows what each var does."""
    rendered = render_env_file({})
    # Count comment lines vs assignment lines
    comment_lines = [
        l for l in rendered.splitlines()
        if l.startswith("#") and not l.strip() == "#"
    ]
    # At least one comment per spec (could be multi-line)
    assert len(comment_lines) >= len(
        list_env_var_specs(),
    )


def test_render_env_file_missing_values_commented_out():
    """Variables NOT in the supplied values dict should
    appear commented out (so the operator can see them
    without them being active in the env)."""
    rendered = render_env_file({})
    for spec in list_env_var_specs():
        # Either: var doesn't appear at all (no), or
        # appears commented out (yes)
        active = f"\n{spec.name}=" in rendered
        commented = f"\n#{spec.name}=" in rendered or (
            f"# {spec.name}=" in rendered
        )
        assert not active or commented, (
            f"{spec.name} should be commented when no "
            f"value supplied"
        )


def test_render_env_file_set_values_active():
    """Variables WITH supplied values should appear as
    active assignments."""
    rendered = render_env_file({
        "PRSM_FEDERATED_LEARNING_DIR": "/var/fl",
    })
    assert (
        "PRSM_FEDERATED_LEARNING_DIR=/var/fl"
        in rendered
    )


# ── CLI smoke ──────────────────────────────────────


def test_cli_status_prints_summary(
    capsys, monkeypatch,
):
    monkeypatch.delenv(
        "PRSM_FEDERATED_WORKER_PRIVKEY", raising=False,
    )
    from prsm.enterprise.bringup_cli import main
    rc = main(["status"])
    out = capsys.readouterr().out
    # rc==1 is correct when required vars are missing —
    # CI / scripts can rely on the non-zero exit to fail
    # a deploy. The test asserts the summary printed.
    assert rc == 1
    assert "PRSM_FEDERATED_WORKER_PRIVKEY" in out
    assert "MISSING" in out or "✗" in out


def test_cli_generate_produces_env_file(
    capsys,
):
    from prsm.enterprise.bringup_cli import main
    rc = main(["generate"])
    out = capsys.readouterr().out
    assert rc == 0
    # Output should be a usable .env file template
    assert "PRSM_PIPELINE_ORCHESTRATOR_PRIVKEY" in out
    # And it should validate as a deployable config
    lines = [
        l.strip() for l in out.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    values = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    cfg = EnterpriseConfig(values=values)
    cfg.validate()


def test_cli_unknown_command_returns_nonzero(capsys):
    from prsm.enterprise.bringup_cli import main
    rc = main(["nope"])
    assert rc != 0
