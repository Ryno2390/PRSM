"""Unit tests for ContentPublisher + ContentRetriever (PR 2a, Tier A only)."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from prsm.compute.inference.models import ContentTier
from prsm.node.content_publisher import (
    ContentPublisher,
    ContentRetriever,
    PublishedContent,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeManifest:
    infohash: str
    name: str
    total_size: int


@dataclass
class _FakeDownloadResult:
    success: bool
    error: Optional[str] = None


class _FakeBitTorrentProvider:
    """Captures seed_content calls; returns a deterministic manifest."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.next_return: Optional[_FakeManifest] = None

    async def seed_content(
        self,
        path: Path,
        name: Optional[str] = None,
        provenance_id: Optional[str] = None,
        piece_length: int = 262144,
    ) -> Optional[_FakeManifest]:
        self.calls.append({
            "path": Path(path),
            "name": name,
            "provenance_id": provenance_id,
            "piece_length": piece_length,
        })
        if self.next_return is not None:
            return self.next_return
        # Default: synthesize a manifest from the path's content.
        data = Path(path).read_bytes()
        infohash = hashlib.sha256(data + b"::infohash").hexdigest()[:40]
        return _FakeManifest(
            infohash=infohash,
            name=name or Path(path).name,
            total_size=len(data),
        )


class _FakeBitTorrentRequester:
    """Writes the requested file into save_path; returns success."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.payload_for: Dict[str, bytes] = {}
        self.next_result: Optional[_FakeDownloadResult] = None
        self.extra_files: int = 0  # set >0 to simulate multi-file torrents

    async def request_content(
        self,
        infohash: str,
        save_path: Path,
        timeout: Optional[float] = None,
        progress_callback: Any = None,
    ) -> _FakeDownloadResult:
        self.calls.append({
            "infohash": infohash,
            "save_path": Path(save_path),
            "timeout": timeout,
        })
        if self.next_result is not None:
            return self.next_result

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        payload = self.payload_for.get(infohash, b"")
        (save_path / "content.bin").write_bytes(payload)
        for i in range(self.extra_files):
            (save_path / f"extra-{i}.bin").write_bytes(b"x")
        return _FakeDownloadResult(success=True)


# ---------------------------------------------------------------------------
# ContentPublisher
# ---------------------------------------------------------------------------


class TestContentPublisher:
    @pytest.fixture
    def staging_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "staging"

    @pytest.fixture
    def provider(self) -> _FakeBitTorrentProvider:
        return _FakeBitTorrentProvider()

    @pytest.fixture
    def publisher(
        self, provider: _FakeBitTorrentProvider, staging_dir: Path
    ) -> ContentPublisher:
        return ContentPublisher(bt_provider=provider, staging_dir=staging_dir)

    @pytest.mark.asyncio
    async def test_publish_tier_a_returns_handle(
        self, publisher: ContentPublisher, provider: _FakeBitTorrentProvider
    ) -> None:
        data = b"hello PRSM"
        result = await publisher.publish(data, provenance_id="0xprov-1")

        assert isinstance(result, PublishedContent)
        assert result.torrent_infohash != ""
        assert result.staged_path.exists()
        assert result.staged_path.read_bytes() == data
        assert result.manifest.total_size == len(data)
        assert len(provider.calls) == 1
        assert provider.calls[0]["provenance_id"] == "0xprov-1"

    @pytest.mark.asyncio
    async def test_publish_uses_sha256_filename(
        self, publisher: ContentPublisher
    ) -> None:
        data = b"deterministic"
        result = await publisher.publish(data, provenance_id="0xp")

        expected_name = hashlib.sha256(data).hexdigest()
        assert result.staged_path.name == expected_name

    @pytest.mark.asyncio
    async def test_publish_idempotent_for_same_bytes(
        self, publisher: ContentPublisher, provider: _FakeBitTorrentProvider
    ) -> None:
        data = b"same content"
        first = await publisher.publish(data, provenance_id="0xp")
        second = await publisher.publish(data, provenance_id="0xp")

        # Same staged path; same infohash (synthesized from content).
        assert first.staged_path == second.staged_path
        assert first.torrent_infohash == second.torrent_infohash
        # Two seed_content calls — BT layer is responsible for its own dedup.
        assert len(provider.calls) == 2

    @pytest.mark.asyncio
    async def test_publish_tier_b_raises_not_implemented(
        self, publisher: ContentPublisher
    ) -> None:
        with pytest.raises(NotImplementedError, match="PR 2b"):
            await publisher.publish(
                b"x", provenance_id="0xp", tier=ContentTier.B
            )

    @pytest.mark.asyncio
    async def test_publish_tier_c_raises_not_implemented(
        self, publisher: ContentPublisher
    ) -> None:
        with pytest.raises(NotImplementedError, match="PR 2b"):
            await publisher.publish(
                b"x", provenance_id="0xp", tier=ContentTier.C
            )

    @pytest.mark.asyncio
    async def test_publish_seed_failure_preserves_staged_file(
        self,
        publisher: ContentPublisher,
        provider: _FakeBitTorrentProvider,
        staging_dir: Path,
    ) -> None:
        provider.next_return = None  # default behaviour
        # Override provider to return None (simulating seed failure).

        class _FailingProvider(_FakeBitTorrentProvider):
            async def seed_content(self, *args: Any, **kwargs: Any) -> None:
                self.calls.append(kwargs)
                return None

        failing = _FailingProvider()
        pub = ContentPublisher(bt_provider=failing, staging_dir=staging_dir)
        data = b"will fail to seed"

        with pytest.raises(RuntimeError, match="seed_content returned None"):
            await pub.publish(data, provenance_id="0xp")

        # Staged file is preserved for retry.
        staged = staging_dir / hashlib.sha256(data).hexdigest()
        assert staged.exists()
        assert staged.read_bytes() == data

    @pytest.mark.asyncio
    async def test_publish_custom_name_passed_through(
        self, publisher: ContentPublisher, provider: _FakeBitTorrentProvider
    ) -> None:
        await publisher.publish(
            b"x", provenance_id="0xp", name="dataset-v1"
        )
        assert provider.calls[0]["name"] == "dataset-v1"


# ---------------------------------------------------------------------------
# ContentRetriever
# ---------------------------------------------------------------------------


class TestContentRetriever:
    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "cache"

    @pytest.fixture
    def requester(self) -> _FakeBitTorrentRequester:
        return _FakeBitTorrentRequester()

    @pytest.fixture
    def retriever(
        self, requester: _FakeBitTorrentRequester, cache_dir: Path
    ) -> ContentRetriever:
        return ContentRetriever(bt_requester=requester, cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_fetch_returns_payload(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        infohash = "deadbeef"
        requester.payload_for[infohash] = b"the bytes we asked for"

        result = await retriever.fetch(infohash)

        assert result == b"the bytes we asked for"
        assert len(requester.calls) == 1
        assert requester.calls[0]["infohash"] == infohash

    @pytest.mark.asyncio
    async def test_fetch_propagates_timeout(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        requester.payload_for["x"] = b""
        await retriever.fetch("x", timeout=12.5)
        assert requester.calls[0]["timeout"] == 12.5

    @pytest.mark.asyncio
    async def test_fetch_failure_raises(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        requester.next_result = _FakeDownloadResult(
            success=False, error="peer timeout"
        )
        with pytest.raises(RuntimeError, match="peer timeout"):
            await retriever.fetch("nope")

    @pytest.mark.asyncio
    async def test_fetch_empty_dir_raises(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        # success=True but the requester writes no files.
        class _EmptyRequester(_FakeBitTorrentRequester):
            async def request_content(self, *args: Any, **kwargs: Any):
                save_path = Path(kwargs.get("save_path") or args[1])
                save_path.mkdir(parents=True, exist_ok=True)
                return _FakeDownloadResult(success=True)

        empty = _EmptyRequester()
        ret = ContentRetriever(bt_requester=empty, cache_dir=retriever.cache_dir)
        with pytest.raises(FileNotFoundError):
            await ret.fetch("empty-result")

    @pytest.mark.asyncio
    async def test_fetch_multifile_raises_not_implemented(
        self,
        retriever: ContentRetriever,
        requester: _FakeBitTorrentRequester,
    ) -> None:
        requester.payload_for["multi"] = b"primary"
        requester.extra_files = 2
        with pytest.raises(NotImplementedError, match="PR 2b"):
            await retriever.fetch("multi")
