"""Sprint 793 — production wire-up of WalletApiServices + ReceiptStore adapter.

Sprint 792 shipped the /devices/earnings endpoint but its
`ReceiptLookup` dependency defaults to `_NoOpReceiptLookup` →
the endpoint returns empty earnings until a production daemon
injects a real adapter.

Sprint 793 closes the carveout. New module:

  prsm/node/wallet_api_wiring.py:
    class _ReceiptStoreAdapter:
        Implements ReceiptLookup; scans node._receipt_store
        + filters by node_ids.

    def wire_wallet_api_services(node):
        Idempotently builds + registers WalletApiServices via
        wallet_api.set_services. Called from PRSMNode.start().

Adapter design:
- Scans the in-memory ReceiptStore cache via `list(offset=0,
  limit=...)` (existing API — gets up to 1000 most-recent
  receipts; sufficient for the typical operator's roster).
- In-memory filter by settler_node_id in node_ids.
- Empty input or empty store → []. Defensive against None
  receipt_store (returns [] rather than raising).

wire_wallet_api_services:
- Idempotent: calls reset_services_for_tests + set_services so
  double-call doesn't raise.
- Fail-soft: any exception during wiring is logged + swallowed
  (daemon must not crash on this peripheral surface).

Pin tests:
- Adapter exists + has list_receipts_for_node_ids.
- Adapter filters by node_ids correctly.
- Adapter on None store returns [].
- wire_wallet_api_services exists + callable.
- wire_wallet_api_services registers services successfully.
- wire_wallet_api_services is idempotent (second call doesn't raise).
- Wired services' receipt_lookup is the adapter (not no-op).
- PRSMNode.start source-shape: calls wire_wallet_api_services.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock


# ---- Adapter ----------------------------------------------------


def test_adapter_class_exists():
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter
    assert hasattr(_ReceiptStoreAdapter, "list_receipts_for_node_ids")


def test_adapter_returns_empty_for_none_store():
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter
    adapter = _ReceiptStoreAdapter(receipt_store=None)
    assert adapter.list_receipts_for_node_ids(["a" * 32]) == []


def test_adapter_returns_empty_for_empty_store():
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter
    from prsm.node.receipt_store import ReceiptStore

    store = ReceiptStore(persist_dir=None)
    adapter = _ReceiptStoreAdapter(receipt_store=store)
    assert adapter.list_receipts_for_node_ids(["a" * 32]) == []


def test_adapter_filters_by_node_id():
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter
    from prsm.node.receipt_store import ReceiptStore

    store = ReceiptStore(persist_dir=None)
    # Three receipts: two for node-a, one for node-b
    store.put("j1", {
        "job_id": "j1",
        "settler_node_id": "a" * 32,
        "cost_ftns": "1.0",
    })
    store.put("j2", {
        "job_id": "j2",
        "settler_node_id": "b" * 32,
        "cost_ftns": "2.0",
    })
    store.put("j3", {
        "job_id": "j3",
        "settler_node_id": "a" * 32,
        "cost_ftns": "3.0",
    })

    adapter = _ReceiptStoreAdapter(receipt_store=store)

    # Filter to node-a only → 2 receipts
    out_a = adapter.list_receipts_for_node_ids(["a" * 32])
    assert len(out_a) == 2
    for r in out_a:
        assert r["settler_node_id"] == "a" * 32

    # Filter to both → all 3
    out_both = adapter.list_receipts_for_node_ids(
        ["a" * 32, "b" * 32],
    )
    assert len(out_both) == 3

    # Filter to a node_id with no receipts → empty
    out_none = adapter.list_receipts_for_node_ids(["c" * 32])
    assert out_none == []


def test_adapter_empty_node_ids_returns_empty():
    """No node_ids requested → no receipts returned (regardless
    of store contents). Defense-in-depth against unbounded scans."""
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter
    from prsm.node.receipt_store import ReceiptStore

    store = ReceiptStore(persist_dir=None)
    store.put("j1", {
        "job_id": "j1",
        "settler_node_id": "a" * 32,
        "cost_ftns": "1.0",
    })
    adapter = _ReceiptStoreAdapter(receipt_store=store)
    assert adapter.list_receipts_for_node_ids([]) == []


# ---- wire_wallet_api_services -----------------------------------


def test_wire_function_exists():
    from prsm.node.wallet_api_wiring import wire_wallet_api_services
    assert callable(wire_wallet_api_services)


def _build_fake_node():
    from prsm.node.receipt_store import ReceiptStore
    node = MagicMock()
    node._receipt_store = ReceiptStore(persist_dir=None)
    return node


def test_wire_registers_services():
    """After wiring, wallet_api has services with a non-no-op
    receipt_lookup pointing at our adapter."""
    from prsm.interface.api import wallet_api
    from prsm.node.wallet_api_wiring import (
        _ReceiptStoreAdapter, wire_wallet_api_services,
    )
    wallet_api.reset_services_for_tests()
    node = _build_fake_node()

    wire_wallet_api_services(node)

    services = wallet_api.get_services()
    assert isinstance(services.receipt_lookup, _ReceiptStoreAdapter)
    # And the adapter points at our node's receipt_store
    assert services.receipt_lookup._receipt_store is node._receipt_store

    wallet_api.reset_services_for_tests()  # cleanup


def test_wire_is_idempotent():
    """Calling wire twice MUST NOT raise — daemon restart paths
    + lifecycle tests rely on this."""
    from prsm.interface.api import wallet_api
    from prsm.node.wallet_api_wiring import wire_wallet_api_services

    wallet_api.reset_services_for_tests()
    node = _build_fake_node()

    wire_wallet_api_services(node)
    # Second call must succeed (existing _services slot replaced
    # via internal reset path inside the helper).
    wire_wallet_api_services(node)

    services = wallet_api.get_services()
    assert services is not None
    wallet_api.reset_services_for_tests()


# ---- PRSMNode.start integration --------------------------------


def test_node_start_source_calls_wire_helper():
    """PRSMNode.start must invoke wire_wallet_api_services so a
    daemon-startup path actually wires the production adapter.
    Source-shape pin (the function is long + closure-heavy)."""
    from prsm.node.node import PRSMNode
    src = inspect.getsource(PRSMNode.start)
    assert "wire_wallet_api_services" in src, (
        "Sprint 793: PRSMNode.start must call "
        "wire_wallet_api_services to register WalletApiServices "
        "with the production ReceiptStore adapter."
    )
