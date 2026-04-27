"""
Smoke test — Phase 3.x.4 Task 7 — public-surface contract.

Pins every name in ``prsm.security.privacy_budget_persistence.__all__``
so accidental drift (typo, stale import, accidental re-bind, internal
helper leak) breaks at PR time, not on first downstream user report.

Mirrors ``test_model_registry_exports.py`` (Phase 3.x.2 Task 6) — same
four guarantees:
  1. ``__all__`` matches an explicit expected set
  2. Every advertised name resolves at import time
  3. Type expectations are met (classes are classes, callables callable)
  4. Re-export identity preserved across import paths
  5. Star-import surface = ``__all__`` (no internal helper leakage)
"""

from __future__ import annotations

import inspect

import prsm.security.privacy_budget_persistence as pbp_pkg


# ──────────────────────────────────────────────────────────────────────────
# __all__ contract
# ──────────────────────────────────────────────────────────────────────────


EXPECTED_PUBLIC_NAMES = frozenset({
    # Constants
    "ENTRY_SCHEMA_VERSION",
    "ENTRY_SIGNING_DOMAIN",
    "GENESIS_PREV_HASH",
    # Dataclass + enum
    "PrivacyBudgetEntry",
    "PrivacyBudgetEntryType",
    # Signing functions
    "is_signed",
    "sign_entry",
    "verify_entry",
    # Store classes + helper
    "PrivacyBudgetStore",
    "InMemoryPrivacyBudgetStore",
    "FilesystemPrivacyBudgetStore",
    "hash_entry_payload",
    # Exceptions
    "PrivacyBudgetStoreError",
    "OutOfOrderAppendError",
    "JournalCorruptionError",
    # Tracker
    "PersistentPrivacyBudgetTracker",
})


class TestPublicSurfacePins:
    def test_all_matches_expected_set(self):
        # If this fails, either __all__ has drifted or the test needs
        # an explicit decision about whether to add/remove a name.
        assert set(pbp_pkg.__all__) == EXPECTED_PUBLIC_NAMES

    def test_no_duplicate_names_in_all(self):
        all_list = pbp_pkg.__all__
        assert len(all_list) == len(set(all_list)), (
            f"duplicates in __all__: {all_list}"
        )

    def test_every_name_resolves(self):
        for name in pbp_pkg.__all__:
            assert hasattr(pbp_pkg, name), (
                f"__all__ advertises {name!r} but it doesn't resolve "
                f"on the module"
            )
            assert getattr(pbp_pkg, name) is not None, (
                f"{name!r} resolved to None — likely a stale import"
            )


# ──────────────────────────────────────────────────────────────────────────
# Type expectations — catches accidental shadowing
# ──────────────────────────────────────────────────────────────────────────


class TestExportedTypes:
    def test_constants_are_correct_types(self):
        assert isinstance(pbp_pkg.ENTRY_SCHEMA_VERSION, int)
        assert isinstance(pbp_pkg.ENTRY_SIGNING_DOMAIN, bytes)
        assert isinstance(pbp_pkg.GENESIS_PREV_HASH, bytes)
        assert len(pbp_pkg.GENESIS_PREV_HASH) == 32

    def test_dataclass_and_enum_are_classes(self):
        assert inspect.isclass(pbp_pkg.PrivacyBudgetEntry)
        assert inspect.isclass(pbp_pkg.PrivacyBudgetEntryType)
        # Enum membership round-trips
        assert pbp_pkg.PrivacyBudgetEntryType("spend") is not None
        assert pbp_pkg.PrivacyBudgetEntryType("reset") is not None

    def test_signing_helpers_are_callable(self):
        assert callable(pbp_pkg.sign_entry)
        assert callable(pbp_pkg.verify_entry)
        assert callable(pbp_pkg.is_signed)

    def test_store_types_are_classes(self):
        assert inspect.isclass(pbp_pkg.PrivacyBudgetStore)
        assert inspect.isclass(pbp_pkg.InMemoryPrivacyBudgetStore)
        assert inspect.isclass(pbp_pkg.FilesystemPrivacyBudgetStore)

    def test_concrete_stores_subclass_abc(self):
        assert issubclass(
            pbp_pkg.InMemoryPrivacyBudgetStore, pbp_pkg.PrivacyBudgetStore
        )
        assert issubclass(
            pbp_pkg.FilesystemPrivacyBudgetStore, pbp_pkg.PrivacyBudgetStore
        )

    def test_hash_entry_payload_is_callable(self):
        assert callable(pbp_pkg.hash_entry_payload)

    def test_tracker_inherits_from_parent(self):
        # PersistentPrivacyBudgetTracker MUST IS-A PrivacyBudgetTracker
        # so existing isinstance checks in observability + dashboards
        # keep working.
        from prsm.security.privacy_budget import PrivacyBudgetTracker
        assert issubclass(
            pbp_pkg.PersistentPrivacyBudgetTracker, PrivacyBudgetTracker
        )

    def test_exceptions_form_a_hierarchy(self):
        # Every exception advertised by __all__ MUST inherit from the
        # documented base. A typo here breaks
        # `except PrivacyBudgetStoreError` callers.
        assert issubclass(pbp_pkg.PrivacyBudgetStoreError, Exception)
        for name in ["OutOfOrderAppendError", "JournalCorruptionError"]:
            cls = getattr(pbp_pkg, name)
            assert issubclass(cls, pbp_pkg.PrivacyBudgetStoreError), (
                f"{name} must subclass PrivacyBudgetStoreError"
            )


# ──────────────────────────────────────────────────────────────────────────
# Re-export integrity — package surface and submodule attrs are identity-equal
# ──────────────────────────────────────────────────────────────────────────


class TestReExportIdentity:
    """Sneaky regression mode: someone re-defines a name at the package
    level (e.g., adds a wrapper function) and the direct-import path
    stops being identity-equal to the submodule-import path. Downstream
    code using `isinstance(x, PrivacyBudgetEntry)` would then start
    failing on objects produced by the submodule. These tests pin
    identity."""

    def test_models_re_exports_match_submodule(self):
        from prsm.security.privacy_budget_persistence import models as _models_mod
        assert pbp_pkg.PrivacyBudgetEntry is _models_mod.PrivacyBudgetEntry
        assert pbp_pkg.PrivacyBudgetEntryType is _models_mod.PrivacyBudgetEntryType
        assert pbp_pkg.ENTRY_SCHEMA_VERSION is _models_mod.ENTRY_SCHEMA_VERSION
        assert pbp_pkg.ENTRY_SIGNING_DOMAIN is _models_mod.ENTRY_SIGNING_DOMAIN
        assert pbp_pkg.GENESIS_PREV_HASH is _models_mod.GENESIS_PREV_HASH

    def test_signing_re_exports_match_submodule(self):
        from prsm.security.privacy_budget_persistence import signing as _signing_mod
        assert pbp_pkg.sign_entry is _signing_mod.sign_entry
        assert pbp_pkg.verify_entry is _signing_mod.verify_entry
        assert pbp_pkg.is_signed is _signing_mod.is_signed

    def test_store_re_exports_match_submodule(self):
        from prsm.security.privacy_budget_persistence import store as _store_mod
        assert pbp_pkg.PrivacyBudgetStore is _store_mod.PrivacyBudgetStore
        assert pbp_pkg.InMemoryPrivacyBudgetStore is _store_mod.InMemoryPrivacyBudgetStore
        assert pbp_pkg.FilesystemPrivacyBudgetStore is _store_mod.FilesystemPrivacyBudgetStore
        assert pbp_pkg.hash_entry_payload is _store_mod.hash_entry_payload
        assert pbp_pkg.PrivacyBudgetStoreError is _store_mod.PrivacyBudgetStoreError
        assert pbp_pkg.OutOfOrderAppendError is _store_mod.OutOfOrderAppendError
        assert pbp_pkg.JournalCorruptionError is _store_mod.JournalCorruptionError

    def test_tracker_re_exports_match_submodule(self):
        from prsm.security.privacy_budget_persistence import tracker as _tracker_mod
        assert pbp_pkg.PersistentPrivacyBudgetTracker is _tracker_mod.PersistentPrivacyBudgetTracker


# ──────────────────────────────────────────────────────────────────────────
# Star-import smoke
# ──────────────────────────────────────────────────────────────────────────


class TestStarImport:
    def test_star_import_yields_only_public_names(self):
        # `from prsm.security.privacy_budget_persistence import *` should
        # bring in EXACTLY the public surface. Catches accidental leakage
        # of internal helpers.
        ns: dict = {}
        exec("from prsm.security.privacy_budget_persistence import *", ns)
        imported = {k for k in ns if not k.startswith("_")}
        assert imported == EXPECTED_PUBLIC_NAMES, (
            f"star-import surface drift:\n"
            f"  unexpected: {imported - EXPECTED_PUBLIC_NAMES}\n"
            f"  missing:    {EXPECTED_PUBLIC_NAMES - imported}"
        )
