"""Sprint 809 — `prsm node init --recommended` pre-fills defaults.

Sprint 800 ships a template with bare `VARNAME=` placeholders.
For operators following the documented production path, those
placeholders force them to look up "what's the right value for
PRSM_INFERENCE_EXECUTOR / PRSM_PARALLAX_GPU_POOL_KIND / etc."
that have STABLE recommended values.

Sprint 809 adds a `--recommended` flag that fills in sensible
production defaults inline. Operators get a near-complete env
file out of the box; they only need to fill in operator-
specific values (PRSM_OPERATOR_ADDRESS, PRSM_PARALLAX_HF_MODEL_ID).

Default lines for recommended-filled vars include a comment:
  # recommended: <value>
  PRSM_FOO=value

So operators know which were auto-filled vs left placeholder.

Pin tests:
- --recommended flag exists in --help.
- Without --recommended, sprint 800 behavior preserved (bare
  `VAR=` lines).
- With --recommended, PRSM_INFERENCE_EXECUTOR=parallax appears.
- With --recommended, PRSM_PARALLAX_GPU_POOL_KIND=dht-backed.
- With --recommended, PRSM_PARALLAX_TRUST_STACK_KIND=production.
- Operator-specific vars (PRSM_OPERATOR_ADDRESS) stay placeholder
  (no recommended default for those).
- "# recommended" annotation comment present.
- --recommended + --dry-run still works (no write).
"""
from __future__ import annotations

from pathlib import Path
from click.testing import CliRunner


def _invoke(args=None, env=None):
    from prsm.cli import node as _node_group
    runner = CliRunner()
    return runner.invoke(
        _node_group, ["init"] + (args or []), env=env or {},
    )


# ---- Flag exists -----------------------------------------------


def test_recommended_flag_in_help():
    from prsm.cli import node as _node_group
    runner = CliRunner()
    init_cmd = _node_group.commands["init"]
    result = runner.invoke(init_cmd, ["--help"])
    assert result.exit_code == 0
    assert "--recommended" in result.output


# ---- Sprint 800 behavior preserved when flag absent -----------


def test_without_recommended_bare_placeholders_preserved(
    tmp_path: Path,
):
    """No --recommended → sprint 800 behavior: bare VAR= lines,
    no recommended-value annotations."""
    target = tmp_path / "operator.env"
    result = _invoke(["--output", str(target)])
    assert result.exit_code == 0
    content = target.read_text()
    # Sprint 800 placeholder lines for the production trio
    assert "PRSM_INFERENCE_EXECUTOR=\n" in content
    assert "PRSM_PARALLAX_GPU_POOL_KIND=\n" in content
    assert "PRSM_PARALLAX_TRUST_STACK_KIND=\n" in content
    # No recommended annotations
    assert "# recommended:" not in content


# ---- --recommended fills production values -------------------


def test_recommended_fills_inference_executor(tmp_path: Path):
    target = tmp_path / "operator.env"
    result = _invoke([
        "--output", str(target), "--recommended",
    ])
    assert result.exit_code == 0
    content = target.read_text()
    assert "PRSM_INFERENCE_EXECUTOR=parallax" in content


def test_recommended_fills_pool_kind(tmp_path: Path):
    target = tmp_path / "operator.env"
    _invoke(["--output", str(target), "--recommended"])
    content = target.read_text()
    assert "PRSM_PARALLAX_GPU_POOL_KIND=dht-backed" in content


def test_recommended_fills_trust_stack(tmp_path: Path):
    target = tmp_path / "operator.env"
    _invoke(["--output", str(target), "--recommended"])
    content = target.read_text()
    assert "PRSM_PARALLAX_TRUST_STACK_KIND=production" in content


# ---- Operator-specific vars stay placeholder -----------------


def test_operator_address_stays_placeholder(tmp_path: Path):
    """No recommended default for operator-specific values.
    PRSM_OPERATOR_ADDRESS depends on the operator's ETH key;
    a daemon-provided default would be wrong / dangerous."""
    target = tmp_path / "operator.env"
    _invoke(["--output", str(target), "--recommended"])
    content = target.read_text()
    assert "PRSM_OPERATOR_ADDRESS=\n" in content


# ---- Recommended annotation visible --------------------------


def test_recommended_annotation_marks_auto_filled_lines(
    tmp_path: Path,
):
    """Operator can grep the file to see which lines were
    auto-filled vs left for them to fill — the annotation
    comment surfaces this."""
    target = tmp_path / "operator.env"
    _invoke(["--output", str(target), "--recommended"])
    content = target.read_text()
    assert "# recommended:" in content


# ---- Dry-run still works -----------------------------------


def test_recommended_dry_run_emits_to_stdout(tmp_path: Path):
    target = tmp_path / "operator.env"
    result = _invoke([
        "--output", str(target),
        "--recommended", "--dry-run",
    ])
    assert result.exit_code == 0
    assert not target.exists()
    # stdout has the filled vars
    assert "PRSM_INFERENCE_EXECUTOR=parallax" in result.output
