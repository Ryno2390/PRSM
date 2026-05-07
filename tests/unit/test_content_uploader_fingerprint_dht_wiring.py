"""PRSM-PROV-1 Item 4 T4.9.next3 — ContentUploader fingerprint DHT wiring.

Pins the lockstep guarantee: when ContentUploader is constructed with
embedding_dht_client + content_index + embedding_model_id, BOTH the
text-vector lane (``_semantic_index``) AND the binary fingerprint
lane (``_fingerprint_index``) must report ``dht_enabled is True``.

If a future refactor wires the embedding lane but forgets the
fingerprint lane (or vice versa), this test fails and points at the
exact construction site.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prsm.node.content_uploader import ContentUploader
from prsm.node.identity import generate_node_identity


def _make_uploader(
    *,
    content_index=None,
    embedding_dht_client=None,
    embedding_model_id=None,
    fingerprint_index_path: Path = None,
):
    return ContentUploader(
        identity=generate_node_identity("test-node"),
        gossip=MagicMock(),
        ledger=MagicMock(),
        content_index=content_index,
        embedding_dht_client=embedding_dht_client,
        embedding_model_id=embedding_model_id,
        fingerprint_index_path=fingerprint_index_path,
    )


class TestLockstepWiring:
    def test_both_disabled_when_no_dht_wired(self):
        uploader = _make_uploader()
        assert uploader._semantic_index.dht_enabled is False
        assert uploader._fingerprint_index.dht_enabled is False

    def test_both_enabled_when_full_dht_kit_wired(self):
        ci = MagicMock()
        ci._records = {}
        uploader = _make_uploader(
            content_index=ci,
            embedding_dht_client=MagicMock(),
            embedding_model_id="test-model",
        )
        assert uploader._semantic_index.dht_enabled is True
        assert uploader._fingerprint_index.dht_enabled is True

    def test_both_disabled_when_only_client_wired(self):
        """No content_index → no peer_candidates_fn → both lanes off."""
        uploader = _make_uploader(
            embedding_dht_client=MagicMock(),
            embedding_model_id="test-model",
            # no content_index
        )
        assert uploader._semantic_index.dht_enabled is False
        assert uploader._fingerprint_index.dht_enabled is False

    def test_both_disabled_when_only_content_index_wired(self):
        """No client → no escalation. Both lanes must reflect this."""
        ci = MagicMock()
        ci._records = {}
        uploader = _make_uploader(
            content_index=ci,
            embedding_model_id="test-model",
            # no embedding_dht_client
        )
        assert uploader._semantic_index.dht_enabled is False
        assert uploader._fingerprint_index.dht_enabled is False


class TestFingerprintIndexPath:
    def test_persist_path_propagates(self, tmp_path: Path):
        path = tmp_path / "fp_index.json"
        uploader = _make_uploader(fingerprint_index_path=path)
        # Internal — but the assertion is load-bearing for production
        # disk persistence: a missing pass-through here means warm-cache
        # dedup doesn't survive node restarts.
        assert uploader._fingerprint_index._persist_path == path

    def test_no_path_means_no_persistence(self):
        uploader = _make_uploader()
        assert uploader._fingerprint_index._persist_path is None
