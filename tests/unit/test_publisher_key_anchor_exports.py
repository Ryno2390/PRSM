"""
Smoke test — Phase 3.x.3 Task 6 — public-surface contract.

Pins every name in ``prsm.security.publisher_key_anchor.__all__`` so
accidental drift breaks at PR time, not on first downstream user
report.

Mirrors ``test_model_registry_exports.py`` (Phase 3.x.2 Task 6) and
``test_privacy_budget_persistence_exports.py`` (Phase 3.x.4 Task 7) —
same five guarantees:
  1. ``__all__`` matches an explicit expected set
  2. No duplicates in ``__all__``
  3. Every advertised name resolves at import time
  4. Type expectations are met (classes are classes, callables callable)
  5. Re-export identity preserved across import paths
  6. Star-import surface = ``__all__`` (no internal helper leakage)
"""

from __future__ import annotations

import inspect

import prsm.security.publisher_key_anchor as anchor_pkg


# ──────────────────────────────────────────────────────────────────────────
# __all__ contract
# ──────────────────────────────────────────────────────────────────────────


EXPECTED_PUBLIC_NAMES = frozenset({
    # Client (Task 3)
    "PublisherKeyAnchorClient",
    "PUBLISHER_KEY_ANCHOR_ABI",
    # Verifier wrappers (Task 4)
    "verify_manifest_with_anchor",
    "verify_entry_with_anchor",
    "verify_receipt_with_anchor",
    # Exceptions
    "PublisherKeyAnchorError",
    "PublisherAlreadyRegisteredError",
    "PublisherNotRegisteredError",
    "AnchorRPCError",
})


class TestPublicSurfacePins:
    def test_all_matches_expected_set(self):
        # If this fails, either __all__ has drifted or the test needs
        # an explicit decision about whether to add/remove a name.
        assert set(anchor_pkg.__all__) == EXPECTED_PUBLIC_NAMES

    def test_no_duplicate_names_in_all(self):
        all_list = anchor_pkg.__all__
        assert len(all_list) == len(set(all_list)), (
            f"duplicates in __all__: {all_list}"
        )

    def test_every_name_resolves(self):
        for name in anchor_pkg.__all__:
            assert hasattr(anchor_pkg, name), (
                f"__all__ advertises {name!r} but it doesn't resolve "
                f"on the module"
            )
            assert getattr(anchor_pkg, name) is not None, (
                f"{name!r} resolved to None — likely a stale import"
            )


# ──────────────────────────────────────────────────────────────────────────
# Type expectations — catches accidental shadowing
# ──────────────────────────────────────────────────────────────────────────


class TestExportedTypes:
    def test_client_is_class(self):
        assert inspect.isclass(anchor_pkg.PublisherKeyAnchorClient)

    def test_abi_is_list(self):
        # ABI is the embedded list-of-dicts the client uses to bind
        # the contract. Sanity-check shape.
        assert isinstance(anchor_pkg.PUBLISHER_KEY_ANCHOR_ABI, list)
        assert len(anchor_pkg.PUBLISHER_KEY_ANCHOR_ABI) > 0
        # Every entry has a "type" key (function | error | event | ...)
        assert all(
            "type" in entry
            for entry in anchor_pkg.PUBLISHER_KEY_ANCHOR_ABI
        )
        # Confirm the public function names we depend on are present.
        function_names = {
            entry.get("name")
            for entry in anchor_pkg.PUBLISHER_KEY_ANCHOR_ABI
            if entry["type"] == "function"
        }
        assert {"register", "lookup", "admin"} <= function_names

    def test_verifier_wrappers_callable(self):
        for name in [
            "verify_manifest_with_anchor",
            "verify_entry_with_anchor",
            "verify_receipt_with_anchor",
        ]:
            fn = getattr(anchor_pkg, name)
            assert callable(fn), f"{name} must be callable"

    def test_exceptions_form_a_hierarchy(self):
        # Every exception advertised by __all__ MUST inherit from the
        # documented base. A typo here breaks
        # `except PublisherKeyAnchorError` callers.
        assert issubclass(anchor_pkg.PublisherKeyAnchorError, Exception)
        for name in [
            "PublisherAlreadyRegisteredError",
            "PublisherNotRegisteredError",
            "AnchorRPCError",
        ]:
            cls = getattr(anchor_pkg, name)
            assert issubclass(cls, anchor_pkg.PublisherKeyAnchorError), (
                f"{name} must subclass PublisherKeyAnchorError"
            )


# ──────────────────────────────────────────────────────────────────────────
# Re-export integrity — package surface and submodule attrs are identity-equal
# ──────────────────────────────────────────────────────────────────────────


class TestReExportIdentity:
    """Sneaky regression mode: someone re-defines a name at the
    package level (e.g., adds a wrapper function) and the direct-import
    path stops being identity-equal to the submodule-import path.
    Downstream code using `isinstance(x, PublisherKeyAnchorError)`
    would then start failing on objects produced by the submodule.
    These tests pin identity."""

    def test_client_re_exports_match_submodule(self):
        from prsm.security.publisher_key_anchor import client as _client_mod
        assert anchor_pkg.PublisherKeyAnchorClient is _client_mod.PublisherKeyAnchorClient
        assert anchor_pkg.PUBLISHER_KEY_ANCHOR_ABI is _client_mod.PUBLISHER_KEY_ANCHOR_ABI

    def test_exceptions_re_exports_match_submodule(self):
        from prsm.security.publisher_key_anchor import exceptions as _exc_mod
        assert anchor_pkg.PublisherKeyAnchorError is _exc_mod.PublisherKeyAnchorError
        assert anchor_pkg.PublisherAlreadyRegisteredError is _exc_mod.PublisherAlreadyRegisteredError
        assert anchor_pkg.PublisherNotRegisteredError is _exc_mod.PublisherNotRegisteredError
        assert anchor_pkg.AnchorRPCError is _exc_mod.AnchorRPCError

    def test_verifiers_re_exports_match_submodule(self):
        from prsm.security.publisher_key_anchor import verifiers as _ver_mod
        assert anchor_pkg.verify_manifest_with_anchor is _ver_mod.verify_manifest_with_anchor
        assert anchor_pkg.verify_entry_with_anchor is _ver_mod.verify_entry_with_anchor
        assert anchor_pkg.verify_receipt_with_anchor is _ver_mod.verify_receipt_with_anchor


# ──────────────────────────────────────────────────────────────────────────
# Star-import smoke
# ──────────────────────────────────────────────────────────────────────────


class TestStarImport:
    def test_star_import_yields_only_public_names(self):
        # `from prsm.security.publisher_key_anchor import *` should
        # bring in EXACTLY the public surface. Catches accidental
        # leakage of internal helpers (_node_id_to_bytes16,
        # _NEGATIVE_CACHE, etc.).
        ns: dict = {}
        exec("from prsm.security.publisher_key_anchor import *", ns)
        imported = {k for k in ns if not k.startswith("_")}
        assert imported == EXPECTED_PUBLIC_NAMES, (
            f"star-import surface drift:\n"
            f"  unexpected: {imported - EXPECTED_PUBLIC_NAMES}\n"
            f"  missing:    {EXPECTED_PUBLIC_NAMES - imported}"
        )
