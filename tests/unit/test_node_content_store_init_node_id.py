"""Sprint 168 — ContentStore initialization must pass node_id so
content manifests carry correct owner attribution.

Pre-fix prsm/node/node.py:2816 called:
  init_content_store()

— no arguments. The init helper defaults node_id="" (empty
string), so every content manifest written by this node claimed
ownership by "". That breaks:
  - Royalty routing (creator_address derived from manifest
    owner → "" maps to no on-chain address)
  - Provenance verification (manifests fail attribution checks)
  - Operator-side debugging (every owner is "")

Fix: thread self.identity.node_id through to init_content_store.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_init_content_store_call_threads_node_id():
    """Sprint 168 invariant — ContentStore initialization receives
    the node's identity.node_id, not an empty string."""
    # We can't easily run a full PRSMNode.start() — instead we
    # static-source-grep the node.py file to confirm the call
    # site passes node_id explicitly. The actual integration
    # would require mocking the entire startup graph.
    import inspect

    from prsm.node import node as node_module
    src = inspect.getsource(node_module)

    # Find the init_content_store call site and confirm it carries
    # a node_id kwarg (multi-line tolerant via whitespace-stripped
    # search).
    import re
    pattern = re.compile(r"init_content_store\(\s*node_id\s*=", re.S)
    assert pattern.search(src), (
        "Sprint 168: prsm/node/node.py must call "
        "init_content_store(node_id=self.identity.node_id), but "
        "the call site doesn't include the node_id kwarg. "
        "Without it, manifests claim ownership by empty string."
    )


def test_init_content_store_storage_provider_also_threads_node_id():
    """Sprint 168 — storage_provider.py's init_content_store()
    fallback should also pass node_id when reachable. Defensive
    grep similar to the node.py check above."""
    import inspect

    from prsm.node import storage_provider as sp_module
    src = inspect.getsource(sp_module)

    # The storage_provider's init path is more nuanced — it falls
    # back to default args when no node_id is reachable. The grep
    # still pins the explicit-passing pattern when self has one.
    # Acceptable forms:
    #   init_content_store(node_id=self.node_id)
    #   init_content_store(node_id=self.identity.node_id)
    has_explicit = (
        "init_content_store(node_id=" in src
        or "node_id=self.node_id" in src
        or "node_id=self.identity.node_id" in src
    )
    assert has_explicit, (
        "storage_provider.init_content_store() must thread node_id "
        "from self when available — empty default breaks manifest "
        "ownership attribution."
    )
