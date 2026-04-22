"""
Tests for Sub-phase 10.6: AI-Agent-Centric CLI

Coverage:
  1. Helper functions — _agent_output, _agent_error, session meta persistence
  2. status --format json — machine-readable system status
  3. node info --format json — machine-readable node info
  4. nwtn agent-team start — session creation, JSON output
  5. nwtn agent-team status — session state query
  6. nwtn agent-team whiteboard — whiteboard read
  7. nwtn agent-team ledger — ledger read
  8. nwtn agent-team checkpoint — branch review
  9. nwtn agent-team branches — branch listing
  10. nwtn agent-team synthesise — nightly synthesis
"""

from __future__ import annotations

import json
import subprocess as _subprocess_module
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import main

# Capture real Popen at import time (before conftest session mock is applied).
_REAL_POPEN = _subprocess_module.Popen

# Force early import of modules that use _REAL_POPEN so they capture the real
# Popen during test collection (before the session-scoped subprocess mock runs).
# Without this, lazy imports inside CLI commands would capture the mock instead.
import prsm.compute.nwtn.team.branch_manager as _bm_mod  # noqa: E402
_bm_mod._REAL_POPEN = _REAL_POPEN  # belt-and-suspenders: ensure correct reference


# ======================================================================
# Helpers
# ======================================================================

def invoke(args: List[str], input: str = None) -> tuple:
    """Invoke the CLI and return (result, parsed_json_or_None)."""
    runner = CliRunner()
    result = runner.invoke(main, args, input=input, catch_exceptions=False)
    parsed = None
    if result.output.strip():
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError:
            pass
    return result, parsed


def _make_whiteboard_store(tmp_path, session_id="sess-test"):
    """Create an in-memory whiteboard store with some entries."""
    import asyncio
    from prsm.compute.nwtn.whiteboard import WhiteboardStore
    from prsm.compute.nwtn.bsc import (
        PromotionDecision, ChunkMetadata, FilterDecision, KLFilterResult, DedupResult,
    )
    from datetime import datetime, timezone

    db_path = tmp_path / "wb.db"

    async def _setup():
        store = WhiteboardStore(db_path)
        await store.open()
        await store.create_session(session_id)
        for i in range(3):
            await store.write(PromotionDecision(
                promoted=True,
                chunk=f"Finding {i}: important discovery.",
                metadata=ChunkMetadata(
                    source_agent=f"agent/coder-20260326",
                    session_id=session_id,
                    timestamp=datetime.now(timezone.utc),
                ),
                surprise_score=0.7 + i * 0.05,
                raw_perplexity=70.0,
                similarity_score=0.1,
                kl_result=KLFilterResult(
                    decision=FilterDecision.PROMOTE, score=0.7,
                    epsilon=0.55, reason="test",
                ),
                dedup_result=DedupResult(
                    is_redundant=False, max_similarity=0.1,
                    most_similar_index=None, reason="test",
                ),
                reason="test",
            ))
        await store.close()

    asyncio.run(_setup())
    return db_path


# ======================================================================
# 1. Helper functions
# ======================================================================

class TestHelpers:
    def test_agent_output_valid_json(self, capsys):
        from prsm.cli import _agent_output
        _agent_output({"ok": True, "value": 42})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["value"] == 42

    def test_agent_error_exits_nonzero(self):
        from prsm.cli import _agent_error
        import pytest
        with pytest.raises(SystemExit) as exc:
            _agent_error("something went wrong")
        assert exc.value.code != 0

    def test_save_and_load_session_meta(self, tmp_path):
        from prsm.cli import _save_session_meta, _load_session_meta, _SESSIONS_DIR
        import prsm.cli as cli_module

        # Patch the sessions dir to use tmp_path
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        try:
            _save_session_meta("test-sess", {"goal": "Build X", "status": "active"})
            meta = _load_session_meta("test-sess")
            assert meta is not None
            assert meta["goal"] == "Build X"
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_load_session_meta_returns_none_for_missing(self):
        from prsm.cli import _load_session_meta
        assert _load_session_meta("nonexistent-session-xyz") is None


# ======================================================================
# 2. status --format json
# ======================================================================

class TestStatusCommand:
    def test_status_text_format(self):
        result, _ = invoke(["status"])
        assert result.exit_code == 0
        # Rich table output — not JSON
        assert "PRSM" in result.output or "Component" in result.output or result.output

    def test_status_json_format(self):
        result, data = invoke(["status", "--format", "json"])
        assert result.exit_code == 0
        assert data is not None
        assert data["ok"] is True
        assert "components" in data

    def test_status_json_has_nwtn_component(self):
        _, data = invoke(["status", "--format", "json"])
        assert "nwtn" in data["components"]

    def test_status_json_has_ftns_component(self):
        _, data = invoke(["status", "--format", "json"])
        assert "ftns" in data["components"]


# ======================================================================
# 3. node info --format json (mocked node identity)
# ======================================================================

class TestNodeInfoCommand:
    def test_node_info_json_no_identity(self):
        """With no node identity, json mode should emit an error JSON."""
        with patch("prsm.node.identity.load_node_identity", return_value=None), \
             patch("prsm.node.config.NodeConfig.load", return_value=MagicMock(
                 identity_path=Path("/tmp/nonexistent"),
                 display_name="test", roles=[], p2p_port=8765,
                 api_port=8000, data_dir="/tmp", bootstrap_nodes=[],
             )):
            result, data = invoke(["node", "info", "--format", "json"])
        assert result.exit_code != 0 or (data is not None and data.get("ok") is False)

    def test_node_info_json_with_identity(self):
        mock_identity = MagicMock()
        mock_identity.node_id = "node-abc123"
        mock_identity.public_key_b64 = "A" * 64

        from enum import Enum
        class FakeRole(Enum):
            COMPUTE = "compute"

        mock_config = MagicMock()
        mock_config.identity_path = Path("/tmp/fake")
        mock_config.display_name = "Test Node"
        mock_config.roles = [FakeRole.COMPUTE]
        mock_config.p2p_port = 8765
        mock_config.api_port = 8000
        mock_config.data_dir = "/data"
        mock_config.bootstrap_nodes = ["ws://bootstrap1:8765"]

        with patch("prsm.node.identity.load_node_identity", return_value=mock_identity), \
             patch("prsm.node.config.NodeConfig.load", return_value=mock_config):
            result, data = invoke(["node", "info", "--format", "json"])

        assert result.exit_code == 0
        assert data is not None
        assert data["ok"] is True
        assert data["node_id"] == "node-abc123"
        assert data["p2p_port"] == 8765
        assert "compute" in data["roles"]


# ======================================================================
# 4. nwtn agent-team start
# ======================================================================

class TestAgentTeamStart:
    def _patch_team_start(self, tmp_path):
        """Return context managers that mock team creation."""
        return [
            patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                  ".create_team_branches", new=AsyncMock(return_value={})),
        ]

    def test_start_creates_session_meta(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR

        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        try:
            with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                       ".create_team_branches", new=AsyncMock(return_value={})):
                result, data = invoke([
                    "nwtn", "agent-team", "start",
                    "Build the NWTN BSC system",
                    "--tech", "Python",
                    "--format", "json",
                ])

            assert result.exit_code == 0, result.output
            assert data is not None
            assert data["ok"] is True
            assert "session_id" in data
            assert data["goal"] == "Build the NWTN BSC system"
            assert len(data["team"]) > 0
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_start_json_contains_team_branches(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        try:
            with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                       ".create_team_branches", new=AsyncMock(return_value={})):
                _, data = invoke([
                    "nwtn", "agent-team", "start", "Goal X", "--format", "json",
                ])
            assert data["ok"] is True
            for member in data["team"]:
                assert "role" in member
                assert "branch" in member
                assert "model" in member
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_start_text_format(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        try:
            with patch("prsm.compute.nwtn.team.branch_manager.BranchManager"
                       ".create_team_branches", new=AsyncMock(return_value={})):
                result, _ = invoke([
                    "nwtn", "agent-team", "start", "Goal Y",
                    "--format", "text",
                ])
            assert result.exit_code == 0
            assert "Session" in result.output or "started" in result.output.lower()
        finally:
            cli_module._SESSIONS_DIR = orig


# ======================================================================
# 5. nwtn agent-team status
# ======================================================================

class TestAgentTeamStatus:
    def _write_session(self, tmp_path, session_id="s-test"):
        import prsm.cli as cli_module
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        cli_module._save_session_meta(session_id, {
            "session_id": session_id,
            "goal": "Test goal",
            "status": "active",
            "meta_plan_title": "Test Plan",
            "created_at": "2026-03-26T10:00:00Z",
            "team": [{"role": "coder", "branch": "agent/coder-20260326",
                       "model": "anthropic/claude-sonnet-4-6", "agent_name": "Coder"}],
            "db_path": str(tmp_path / "wb.db"),
            "ledger_dir": str(tmp_path / "ledger"),
        })

    def test_status_known_session(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        self._write_session(tmp_path)
        try:
            result, data = invoke(["nwtn", "agent-team", "status", "s-test", "--format", "json"])
            assert result.exit_code == 0
            assert data["ok"] is True
            assert data["session_id"] == "s-test"
            assert data["goal"] == "Test goal"
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_status_unknown_session(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        try:
            result, data = invoke(["nwtn", "agent-team", "status", "no-such-id", "--format", "json"])
            assert result.exit_code != 0
            assert data is not None
            assert data["ok"] is False
        finally:
            cli_module._SESSIONS_DIR = orig


# ======================================================================
# 6. nwtn agent-team whiteboard
# ======================================================================

class TestAgentTeamWhiteboard:
    def test_whiteboard_json(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"

        sid = "wb-sess"
        db_path = _make_whiteboard_store(tmp_path, session_id=sid)
        cli_module._save_session_meta(sid, {
            "session_id": sid, "db_path": str(db_path),
        })

        try:
            result, data = invoke(
                ["nwtn", "agent-team", "whiteboard", sid, "--format", "json"]
            )
            assert result.exit_code == 0
            assert data["ok"] is True
            assert data["entry_count"] == 3
            assert "compressed_state" in data
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_whiteboard_with_entries_flag(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"

        sid = "wb-sess2"
        db_path = _make_whiteboard_store(tmp_path, session_id=sid)
        cli_module._save_session_meta(sid, {"session_id": sid, "db_path": str(db_path)})

        try:
            _, data = invoke(
                ["nwtn", "agent-team", "whiteboard", sid, "--entries", "--format", "json"]
            )
            assert data["ok"] is True
            assert "entries" in data
            assert len(data["entries"]) == 3
        finally:
            cli_module._SESSIONS_DIR = orig


# ======================================================================
# 7. nwtn agent-team ledger
# ======================================================================

class TestAgentTeamLedger:
    def _make_ledger(self, ledger_dir):
        from prsm.compute.nwtn.synthesis import ProjectLedger, LedgerSigner, NarrativeSynthesizer
        from prsm.compute.nwtn.whiteboard.schema import WhiteboardSnapshot
        import asyncio

        ledger = ProjectLedger(ledger_dir=ledger_dir, project_title="Test")
        ledger.load()
        signer = LedgerSigner()
        signer.load_or_generate()

        async def _write():
            snap = WhiteboardSnapshot(session_id="s", entries=[])
            synth = NarrativeSynthesizer(backend_registry=None)
            result = await synth.synthesise(snap)
            ledger.append(result, signer)

        asyncio.run(_write())
        return ledger

    def test_ledger_json(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"

        ldir = tmp_path / "ledger"
        self._make_ledger(ldir)
        sid = "ledger-sess"
        cli_module._save_session_meta(sid, {
            "session_id": sid, "ledger_dir": str(ldir),
        })

        try:
            result, data = invoke(
                ["nwtn", "agent-team", "ledger", sid, "--format", "json"]
            )
            assert result.exit_code == 0
            assert data["ok"] is True
            assert data["entry_count"] == 1
            assert len(data["entries"]) == 1
            assert "onboarding_context" in data
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_ledger_verify_flag(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"

        ldir = tmp_path / "ledger2"
        self._make_ledger(ldir)
        sid = "ledger-sess2"
        cli_module._save_session_meta(sid, {"session_id": sid, "ledger_dir": str(ldir)})

        try:
            _, data = invoke(
                ["nwtn", "agent-team", "ledger", sid, "--verify", "--format", "json"]
            )
            assert data["ok"] is True
            assert data["chain_valid"] is True
            assert "latest_chain_hash" in data
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_ledger_no_args_errors(self, tmp_path):
        result, data = invoke(["nwtn", "agent-team", "ledger", "--format", "json"])
        assert result.exit_code != 0 or (data is not None and not data.get("ok", True))

    def test_ledger_dir_option(self, tmp_path):
        ldir = tmp_path / "direct_ledger"
        self._make_ledger(ldir)
        result, data = invoke([
            "nwtn", "agent-team", "ledger",
            "--ledger-dir", str(ldir),
            "--format", "json",
        ])
        assert result.exit_code == 0
        assert data["entry_count"] == 1


# ======================================================================
# 8. nwtn agent-team checkpoint
# ======================================================================

class TestAgentTeamCheckpoint:
    @pytest.fixture
    def git_repo(self, tmp_path):
        """Create a minimal git repo using real Popen."""
        import os
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"

        repo = tmp_path / "repo"
        repo.mkdir()
        for cmd in [
            ["git", "-C", str(repo), "init"],
            ["git", "-C", str(repo), "config", "user.email", "t@prsm.ai"],
            ["git", "-C", str(repo), "config", "user.name", "Test"],
        ]:
            _REAL_POPEN(cmd, stdout=_subprocess_module.PIPE,
                        stderr=_subprocess_module.PIPE, env=env).communicate()
        (repo / "README.md").write_text("test\n")
        _REAL_POPEN(["git", "-C", str(repo), "add", "."],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "commit", "-m", "init"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "branch", "-M", "main"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        return repo

    def test_checkpoint_empty_branch_rejected(self, tmp_path, git_repo):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        sid = "cp-sess"
        cli_module._save_session_meta(sid, {
            "session_id": sid, "goal": "Build X",
            "meta_plan_title": "Test Plan", "repo_path": str(git_repo),
        })

        # Create an agent branch (empty — no commits)
        import os
        env = dict(os.environ); env["GIT_TERMINAL_PROMPT"] = "0"
        _REAL_POPEN(
            ["git", "-C", str(git_repo), "checkout", "-b", "agent/coder-20260326"],
            stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env
        ).communicate()
        _REAL_POPEN(
            ["git", "-C", str(git_repo), "checkout", "main"],
            stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env
        ).communicate()

        try:
            result, data = invoke([
                "nwtn", "agent-team", "checkpoint",
                "agent/coder-20260326", sid,
                "--repo-path", str(git_repo),
                "--format", "json",
            ])
            assert result.exit_code == 0
            assert data["ok"] is True
            assert data["approved"] is False
            assert "empty" in data["reason"].lower()
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_checkpoint_with_commits_approved(self, tmp_path, git_repo):
        import prsm.cli as cli_module
        import os
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"
        sid = "cp-sess2"
        cli_module._save_session_meta(sid, {
            "session_id": sid, "goal": "Build X",
            "meta_plan_title": "Test Plan", "repo_path": str(git_repo),
        })

        env = dict(os.environ); env["GIT_TERMINAL_PROMPT"] = "0"
        # Create branch with a commit
        _REAL_POPEN(["git", "-C", str(git_repo), "checkout", "-b", "agent/tester-20260326"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        (git_repo / "work.py").write_text("class Work: pass\n")
        _REAL_POPEN(["git", "-C", str(git_repo), "add", "."],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(git_repo), "commit", "-m", "work done"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(git_repo), "checkout", "main"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()

        try:
            result, data = invoke([
                "nwtn", "agent-team", "checkpoint",
                "agent/tester-20260326", sid,
                "--repo-path", str(git_repo),
                "--format", "json",
            ])
            assert result.exit_code == 0
            assert data["ok"] is True
            assert data["approved"] is True
        finally:
            cli_module._SESSIONS_DIR = orig


# ======================================================================
# 9. nwtn agent-team branches
# ======================================================================

class TestAgentTeamBranches:
    @pytest.fixture
    def git_repo_with_branches(self, tmp_path):
        import os
        env = dict(os.environ); env["GIT_TERMINAL_PROMPT"] = "0"
        repo = tmp_path / "repo"
        repo.mkdir()
        for cmd in [
            ["git", "-C", str(repo), "init"],
            ["git", "-C", str(repo), "config", "user.email", "t@prsm.ai"],
            ["git", "-C", str(repo), "config", "user.name", "T"],
        ]:
            _REAL_POPEN(cmd, stdout=_subprocess_module.PIPE,
                        stderr=_subprocess_module.PIPE, env=env).communicate()
        (repo / "README.md").write_text("test\n")
        _REAL_POPEN(["git", "-C", str(repo), "add", "."],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "commit", "-m", "init"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "branch", "-M", "main"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        # Add two agent branches
        for b in ["agent/coder-20260326", "agent/security-20260326"]:
            _REAL_POPEN(["git", "-C", str(repo), "checkout", "-b", b],
                        stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "checkout", "main"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        return repo

    def test_branches_json(self, git_repo_with_branches):
        result, data = invoke([
            "nwtn", "agent-team", "branches",
            "--repo-path", str(git_repo_with_branches),
            "--format", "json",
        ])
        assert result.exit_code == 0
        assert data["ok"] is True
        assert data["branch_count"] == 2
        branch_names = [b["branch"] for b in data["branches"]]
        assert "agent/coder-20260326" in branch_names
        assert "agent/security-20260326" in branch_names

    def test_branches_empty_repo(self, tmp_path):
        import os
        env = dict(os.environ); env["GIT_TERMINAL_PROMPT"] = "0"
        repo = tmp_path / "empty"
        repo.mkdir()
        _REAL_POPEN(["git", "-C", str(repo), "init"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        (repo / "f").write_text("x")
        _REAL_POPEN(["git", "-C", str(repo), "add", "."],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "config", "user.email", "t@t.com"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "config", "user.name", "T"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()
        _REAL_POPEN(["git", "-C", str(repo), "commit", "-m", "i"],
                    stdout=_subprocess_module.PIPE, stderr=_subprocess_module.PIPE, env=env).communicate()

        result, data = invoke([
            "nwtn", "agent-team", "branches",
            "--repo-path", str(repo),
            "--format", "json",
        ])
        assert result.exit_code == 0
        assert data["branch_count"] == 0


# ======================================================================
# 10. nwtn agent-team synthesise
# ======================================================================

class TestAgentTeamSynthesise:
    def test_synthesise_json(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"

        sid = "synth-sess"
        db_path = _make_whiteboard_store(tmp_path, session_id=sid)
        ldir = tmp_path / "ledger"
        key_file = tmp_path / "ledger.key"
        cli_module._save_session_meta(sid, {
            "session_id": sid,
            "db_path": str(db_path),
            "ledger_dir": str(ldir),
            "key_file": str(key_file),
            "meta_plan_title": "Test Project",
        })

        try:
            result, data = invoke(
                ["nwtn", "agent-team", "synthesise", sid, "--format", "json"]
            )
            assert result.exit_code == 0, result.output
            assert data["ok"] is True
            assert data["entry_index"] == 0
            assert data["chain_valid"] is True
            assert data["whiteboard_entries_synthesised"] == 3
        finally:
            cli_module._SESSIONS_DIR = orig

    def test_synthesise_creates_ledger_files(self, tmp_path):
        import prsm.cli as cli_module
        orig = cli_module._SESSIONS_DIR
        cli_module._SESSIONS_DIR = tmp_path / "sessions"

        sid = "synth-sess2"
        db_path = _make_whiteboard_store(tmp_path, session_id=sid)
        ldir = tmp_path / "ledger2"
        cli_module._save_session_meta(sid, {
            "session_id": sid,
            "db_path": str(db_path),
            "ledger_dir": str(ldir),
            "key_file": str(tmp_path / "k.key"),
            "meta_plan_title": "Test",
        })

        try:
            invoke(["nwtn", "agent-team", "synthesise", sid, "--format", "json"])
            assert (ldir / "project_ledger.md").exists()
            assert (ldir / "project_ledger.json").exists()
        finally:
            cli_module._SESSIONS_DIR = orig
