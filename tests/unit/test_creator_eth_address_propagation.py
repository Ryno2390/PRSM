"""Sprint 244 — verify creator_eth_address propagates through:
   ContentUploadRequest → upload_text() → UploadedContent
   ContentIndex.advertise → ContentRecord → /content/{cid}.
"""
from __future__ import annotations

import pytest

from prsm.node.content_uploader import UploadedContent


def test_uploaded_content_has_field():
    uc = UploadedContent(
        content_id="c1",
        filename="x.txt",
        size_bytes=10,
        content_hash="00" * 32,
        creator_id="creator-a",
        creator_eth_address="0x" + "a" * 40,
    )
    assert uc.creator_eth_address == "0x" + "a" * 40


def test_uploaded_content_field_optional():
    uc = UploadedContent(
        content_id="c1",
        filename="x.txt",
        size_bytes=10,
        content_hash="00" * 32,
        creator_id="creator-a",
    )
    assert uc.creator_eth_address is None


def test_content_record_has_field():
    from prsm.node.content_index import ContentRecord
    r = ContentRecord(
        cid="c1",
        filename="x.txt",
        size_bytes=10,
        content_hash="00" * 32,
        creator_id="creator-a",
        creator_eth_address="0x" + "b" * 40,
    )
    assert r.creator_eth_address == "0x" + "b" * 40


@pytest.mark.asyncio
async def test_content_index_ingests_field_from_advertise():
    """ContentIndex._on_content_advertise wires the gossip
    payload through to ContentRecord. New field carried through."""
    from unittest.mock import MagicMock
    from prsm.node.content_index import ContentIndex

    idx = ContentIndex(gossip=MagicMock())
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c1",
            "provider_id": "peer-a",
            "filename": "x.txt",
            "size_bytes": 10,
            "content_hash": "00" * 32,
            "creator_id": "creator-a",
            "creator_eth_address": "0x" + "c" * 40,
        },
        origin="peer-a",
    )
    record = idx.lookup("c1")
    assert record is not None
    assert record.creator_eth_address == "0x" + "c" * 40


@pytest.mark.asyncio
async def test_content_index_handles_missing_field():
    """Pre-sprint-244 peers won't include the field. Ingest gracefully."""
    from unittest.mock import MagicMock
    from prsm.node.content_index import ContentIndex

    idx = ContentIndex(gossip=MagicMock())
    await idx._on_content_advertise(
        subtype="content.advertise",
        data={
            "cid": "c2",
            "provider_id": "peer-a",
            "filename": "x.txt",
            "size_bytes": 10,
            "content_hash": "00" * 32,
            "creator_id": "creator-a",
            # creator_eth_address omitted
        },
        origin="peer-a",
    )
    record = idx.lookup("c2")
    assert record is not None
    assert record.creator_eth_address is None
