"""PRSM-PROV-1 Item 4 T4.9.next4 — late-bind EmbeddingDHTClient.

Pins the lockstep late-bind contract that lights up cross-node dedup
the moment ``Node.start()`` finishes constructing the running DHT.

Covers:
- Both lanes start ``dht_enabled=False`` when client is None at ctor
- ``set_embedding_dht_client(client)`` flips BOTH lanes to True
- ``set_embedding_dht_client(None)`` flips BOTH lanes back to False
- Re-binding the same client is idempotent
- The same ``peer_candidates_fn`` ends up on both lanes (lockstep
  guarantee — same supplier, same view of peer space)
- A late-bind without ``content_index`` keeps ``peer_candidates_fn``
  None on both lanes (no peers to walk → no escalation)
- ``_embedding_dht_client`` attribute is updated for
  ``_register_local_embedding`` to use
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.content_uploader import ContentUploader
from prsm.node.identity import generate_node_identity


def _make_uploader(*, content_index=None, embedding_model_id="test-model"):
    return ContentUploader(
        identity=generate_node_identity("test-node"),
        gossip=MagicMock(),
        ledger=MagicMock(),
        content_index=content_index,
        embedding_model_id=embedding_model_id,
    )


def _make_content_index():
    ci = MagicMock()
    ci._records = {}
    return ci


class TestLateBindLockstep:
    def test_both_lanes_disabled_at_ctor(self):
        uploader = _make_uploader(content_index=_make_content_index())
        assert uploader._semantic_index.dht_enabled is False
        assert uploader._fingerprint_index.dht_enabled is False

    def test_late_bind_flips_both_lanes(self):
        uploader = _make_uploader(content_index=_make_content_index())
        client = MagicMock()
        uploader.set_embedding_dht_client(client)
        assert uploader._semantic_index.dht_enabled is True
        assert uploader._fingerprint_index.dht_enabled is True

    def test_unbind_flips_both_lanes_off(self):
        uploader = _make_uploader(content_index=_make_content_index())
        uploader.set_embedding_dht_client(MagicMock())
        uploader.set_embedding_dht_client(None)
        assert uploader._semantic_index.dht_enabled is False
        assert uploader._fingerprint_index.dht_enabled is False

    def test_late_bind_idempotent(self):
        uploader = _make_uploader(content_index=_make_content_index())
        client = MagicMock()
        uploader.set_embedding_dht_client(client)
        first_fn = uploader._semantic_index._peer_candidates_fn
        uploader.set_embedding_dht_client(client)
        second_fn = uploader._semantic_index._peer_candidates_fn
        # The supplier is rebuilt each call (it's cheap; just a closure)
        # but both lanes still see the same callable shape and behavior.
        assert callable(second_fn)
        assert uploader._semantic_index.dht_enabled is True
        assert uploader._fingerprint_index.dht_enabled is True


class TestPeerCandidatesFnSharing:
    def test_same_supplier_on_both_lanes(self):
        uploader = _make_uploader(content_index=_make_content_index())
        uploader.set_embedding_dht_client(MagicMock())
        # Lockstep: BOTH lanes must hold the SAME callable instance —
        # if they diverge the two lanes could see different peer
        # views and produce inconsistent dedup results.
        assert (
            uploader._semantic_index._peer_candidates_fn
            is uploader._fingerprint_index._peer_candidates_fn
        )

    def test_no_content_index_keeps_fn_none(self):
        uploader = _make_uploader(content_index=None)
        uploader.set_embedding_dht_client(MagicMock())
        assert uploader._semantic_index._peer_candidates_fn is None
        assert uploader._fingerprint_index._peer_candidates_fn is None
        # Without peer_candidates_fn, both lanes stay disabled even
        # though the client is wired.
        assert uploader._semantic_index.dht_enabled is False
        assert uploader._fingerprint_index.dht_enabled is False


class TestUploaderClientField:
    def test_register_local_embedding_sees_late_bound_client(self):
        """The ``_embedding_dht_client`` attribute is what
        ``_register_local_embedding`` consults for cross-node gossip.
        Late-bind must update it too — not just the index references."""
        uploader = _make_uploader(content_index=_make_content_index())
        assert uploader._embedding_dht_client is None
        client = MagicMock()
        uploader.set_embedding_dht_client(client)
        assert uploader._embedding_dht_client is client


class TestPeerCandidatesSupplierBehavior:
    def test_supplier_yields_records_with_provenance_and_embedding(self):
        """The late-bound peer_candidates_fn walks ContentIndex._records
        the same way the ctor-time supplier would."""
        ci = _make_content_index()
        record_with_both = MagicMock()
        record_with_both.provenance_hash = "0xabc"
        record_with_both.embedding_id = "emb:cid-A"
        record_no_anchor = MagicMock()
        record_no_anchor.provenance_hash = None
        record_no_anchor.embedding_id = "emb:cid-B"
        record_no_embedding = MagicMock()
        record_no_embedding.provenance_hash = "0xdef"
        record_no_embedding.embedding_id = None
        ci._records = {
            "cid-A": record_with_both,
            "cid-B": record_no_anchor,
            "cid-C": record_no_embedding,
        }

        uploader = _make_uploader(content_index=ci)
        uploader.set_embedding_dht_client(MagicMock())
        out = list(uploader._semantic_index._peer_candidates_fn())
        # Only cid-A has both an on-chain anchor + a gossiped embedding.
        assert out == [("cid-A", "0xabc")]
