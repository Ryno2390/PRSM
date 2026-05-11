"""Sprint 245 — verify content advertise gossip carries creator_eth_address.

The three GOSSIP_CONTENT_ADVERTISE publish sites in upload() now
all include the creator_eth_address kwarg. Remote peers'
ContentIndex will then carry the field via the sprint-244
_on_content_advertise wiring.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


def test_advertise_payload_includes_creator_eth_address():
    """Static check: every GOSSIP_CONTENT_ADVERTISE publish site
    in content_uploader.py mentions creator_eth_address within
    a small radius. Sprint 245 closes the gossip-side gap;
    without this, peers don't see the field even though
    ContentIndex (sprint 244) reads it."""
    src = open("prsm/node/content_uploader.py").read()
    lines = src.splitlines()
    publish_lines = [
        i for i, line in enumerate(lines)
        if "gossip.publish(GOSSIP_CONTENT_ADVERTISE" in line
    ]
    assert publish_lines, "no GOSSIP_CONTENT_ADVERTISE publish sites found"
    for i, ln in enumerate(publish_lines):
        # Search the next 30 lines for the field. ContentUploader
        # advertise payloads are always small enough to fit.
        block = "\n".join(lines[ln:ln + 30])
        assert "creator_eth_address" in block, (
            f"publish site at line {ln+1} missing "
            f"creator_eth_address within 30 lines"
        )
