"""Sprint 919 — first-creator-wins on creator_eth_address (content data-plane review).

The content data-plane adversarial review found ContentUploader.upload()
UNCONDITIONALLY overwrites uploaded_content[cid] (content_uploader.py:1492). On a
multi-tenant node, two concurrent uploads of IDENTICAL content (→ same SHA-256 →
same cid) with DIFFERENT creator_eth_address race; last-writer-wins clobbers the
first uploader's payout address. The retrieve path reads
uploaded_content[cid].creator_eth_address to route on-chain royalties, so the
WRONG creator gets credited — violating sp441 first-creator-wins. (creator_id is
the node identity and unaffected; the per-REQUEST creator_eth_address is the
money-routing field.)

Fix: first-creator-wins on the royalty-routing address — once a cid's record
carries a creator_eth_address, a later identical-content upload cannot reassign
it. The check+set is synchronous (no await between) → atomic under asyncio's
single-thread model, so no lock is needed.
"""
from unittest.mock import MagicMock

from prsm.node.content_uploader import ContentUploader, UploadedContent

_A = "0x" + "aa" * 20
_B = "0x" + "bb" * 20


def _uploader():
    return ContentUploader(
        identity=MagicMock(), gossip=MagicMock(), ledger=MagicMock(),
    )


def _rec(eth):
    return UploadedContent(
        content_id="cid1", filename="f", size_bytes=1,
        content_hash="h", creator_id="node", creator_eth_address=eth,
    )


def test_first_eth_wins_over_later_different():
    u = _uploader()
    assert u._canonical_creator_eth_address(_rec(_A), _B) == _A


def test_no_existing_record_uses_new_eth():
    u = _uploader()
    assert u._canonical_creator_eth_address(None, _B) == _B


def test_existing_without_eth_uses_new():
    u = _uploader()
    assert u._canonical_creator_eth_address(_rec(None), _B) == _B


def test_same_eth_is_unchanged():
    u = _uploader()
    assert u._canonical_creator_eth_address(_rec(_A), _A) == _A


def test_insert_site_preserves_first_creator():
    # Simulate the dict-insert site: first upload, then a second identical one
    # with a DIFFERENT payout address must NOT clobber the first.
    u = _uploader()
    u.uploaded_content["cid1"] = _rec(_A)
    second = _rec(_B)
    second.creator_eth_address = u._canonical_creator_eth_address(
        u.uploaded_content.get("cid1"), second.creator_eth_address,
    )
    u.uploaded_content["cid1"] = second
    assert u.uploaded_content["cid1"].creator_eth_address == _A   # first wins
