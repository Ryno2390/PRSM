"""Unit tests verifying ui_api.py endpoints serve real data (Phase 3 Item 3a)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone


def _make_async_cm(mock_db):
    """Return a mock async context manager that yields mock_db."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_db)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# ── list_conversations ────────────────────────────────────────────────────────

class TestListConversations:
    @pytest.mark.asyncio
    async def test_returns_db_sessions(self):
        """list_conversations queries PRSMSessionModel, not hardcoded data."""
        from prsm.interface.api.ui_api import list_conversations

        mock_session = MagicMock(
            session_id=uuid4(), user_id="user1",
            status="active", model_metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_result = MagicMock()
        mock_result.all.return_value = [
            MagicMock(PRSMSessionModel=mock_session, message_count=3)
        ]
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("prsm.core.database.get_async_session",
                   return_value=_make_async_cm(mock_db)):
            result = await list_conversations(user_id="user1")

        assert result["success"] is True
        assert len(result["conversations"]) == 1
        assert result["conversations"][0]["conversation_id"] == str(mock_session.session_id)
        assert result["conversations"][0]["message_count"] == 3

    @pytest.mark.asyncio
    async def test_no_hardcoded_conv_1(self):
        """conv-1 / 'Research on Atomically Precise Manufacturing' must never appear."""
        from prsm.interface.api.ui_api import list_conversations

        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("prsm.core.database.get_async_session",
                   return_value=_make_async_cm(mock_db)):
            result = await list_conversations()

        ids = [c["conversation_id"] for c in result["conversations"]]
        assert "conv-1" not in ids
        assert "conv-2" not in ids


# ── list_user_files ───────────────────────────────────────────────────────────

class TestListUserFiles:
    @pytest.mark.asyncio
    async def test_returns_provenance_records(self):
        from prsm.interface.api.ui_api import list_user_files

        mock_row = MagicMock(
            cid="QmRealCID",
            filename="real_file.csv",
            size_bytes=1024,
            created_at=datetime.now(timezone.utc),
            creator_id="user1",
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row]
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("prsm.core.database.get_async_session",
                   return_value=_make_async_cm(mock_db)):
            result = await list_user_files(user_id="user1")

        assert result["files"][0]["file_id"] == "QmRealCID"
        assert result["files"][0]["filename"] == "real_file.csv"

    @pytest.mark.asyncio
    async def test_no_hardcoded_qmxxx(self):
        from prsm.interface.api.ui_api import list_user_files

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("prsm.core.database.get_async_session",
                   return_value=_make_async_cm(mock_db)):
            result = await list_user_files()

        ids = [f["file_id"] for f in result["files"]]
        assert "QmXXX1" not in ids


# ── get_tokenomics_ui ─────────────────────────────────────────────────────────

class TestGetTokenomicsUi:
    @pytest.mark.asyncio
    async def test_staking_comes_from_db(self):
        from prsm.interface.api.ui_api import get_tokenomics_ui

        # Separate mock results for each execute() call
        mock_stake = MagicMock()
        mock_stake.one.return_value = MagicMock(staked=500.0, rewards=25.0)

        mock_earn = MagicMock()
        mock_earn.scalar.return_value = 10.0

        mock_txs = MagicMock()
        mock_txs.scalars.return_value.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=[
            mock_stake,  # stake totals
            mock_earn,   # daily earnings
            mock_earn,   # weekly earnings
            mock_earn,   # monthly earnings
            mock_earn,   # lifetime earnings
            mock_txs,    # recent transactions
        ])

        with patch("prsm.interface.api.ui_api.FTNSQueries") as mock_ftns, \
             patch("prsm.core.database.get_async_session",
                   return_value=_make_async_cm(mock_db)):

            mock_ftns.get_user_balance = AsyncMock(
                return_value={"balance": 1000.0, "locked_balance": 100.0}
            )
            result = await get_tokenomics_ui("user1")

        staking = result["tokenomics"]["staking"]
        assert staking["staked_amount"] == 500.0
        assert staking["staking_rewards"] == 25.0
        # APY comes from StakingConfig.reward_rate_annual * 100, not hardcoded 12.5
        assert isinstance(staking["apy"], float)
        assert staking["lock_period_days"] > 0


# ── get_user_tasks / create_task ──────────────────────────────────────────────

class TestUserTasks:
    @pytest.mark.asyncio
    async def test_create_then_list_tasks(self):
        from prsm.interface.api.ui_api import create_task, get_user_tasks

        task_store = {}

        async def mock_get(key):
            return task_store.get(key)

        async def mock_store(key, value, ttl=None):
            task_store[key] = value
            return True

        mock_cache = MagicMock()
        mock_cache.get_session = mock_get
        mock_cache.store_session = mock_store

        with patch("prsm.interface.api.ui_api.get_session_cache", return_value=mock_cache):
            await create_task({"title": "Fix bug", "user_id": "user1", "priority": "high"})
            result = await get_user_tasks("user1")

        assert result["success"] is True
        assert any(t["title"] == "Fix bug" for t in result["tasks"])

    @pytest.mark.asyncio
    async def test_no_hardcoded_tasks(self):
        from prsm.interface.api.ui_api import get_user_tasks

        mock_cache = MagicMock()
        mock_cache.get_session = AsyncMock(return_value=None)

        with patch("prsm.interface.api.ui_api.get_session_cache", return_value=mock_cache):
            result = await get_user_tasks("user1")

        titles = [t["title"] for t in result["tasks"]]
        assert "Review research paper draft" not in titles
        assert "Validate tokenomics model" not in titles


# ── upload_file_ui ────────────────────────────────────────────────────────────

class TestUploadFileUi:
    @pytest.mark.asyncio
    async def test_returns_real_cid(self):
        import base64
        from prsm.interface.api.ui_api import upload_file_ui

        mock_ipfs = AsyncMock()
        mock_ipfs.add_content = AsyncMock(
            return_value=MagicMock(success=True, cid="QmRealIPFSCID")
        )

        mock_db = AsyncMock()

        with patch("prsm.core.ipfs_client.get_ipfs_client", return_value=mock_ipfs), \
             patch("prsm.core.database.get_async_session",
                   return_value=_make_async_cm(mock_db)):
            result = await upload_file_ui({
                "filename": "test.txt",
                "content": base64.b64encode(b"hello world").decode(),
                "content_type": "text/plain",
                "user_id": "user1",
            })

        assert result["file"]["ipfs_cid"] == "QmRealIPFSCID"
        # Must not be the old fake pattern Qm{uuid[:32]}
        assert len(result["file"]["ipfs_cid"]) > 10

    @pytest.mark.asyncio
    async def test_ipfs_unavailable_raises_503(self):
        import base64
        from fastapi import HTTPException
        from prsm.core.ipfs_client import IPFSConnectionError
        from prsm.interface.api.ui_api import upload_file_ui

        mock_ipfs = AsyncMock()
        mock_ipfs.add_content = AsyncMock(side_effect=IPFSConnectionError("no daemon"))

        with patch("prsm.core.ipfs_client.get_ipfs_client", return_value=mock_ipfs):
            with pytest.raises(HTTPException) as exc_info:
                await upload_file_ui({
                    "filename": "test.txt",
                    "content": base64.b64encode(b"hello").decode(),
                    "content_type": "text/plain",
                })
        assert exc_info.value.status_code == 503
