"""
Smoke test — Phase 3.x.2 Task 6 — public-surface contract.

Pins every name in ``prsm.compute.model_registry.__all__`` to ensure:

  1. Every advertised name actually resolves at import time (no
     ``__all__`` typos that import-* would silently swallow).
  2. The advertised name and the underlying object are in sync — i.e.,
     ``from prsm.compute.model_registry import X`` and
     ``prsm.compute.model_registry.X`` return the same object.
  3. Type expectations are met: classes are classes, callables are
     callable, constants are the right type. Catches accidental
     import re-binding (e.g., shadowing ``ModelManifest`` with a
     factory function).

This test is the cheap insurance against the kind of breakage that
costs a downstream caller "I imported X from your module and it
doesn't exist anymore" — fixed at PR time, not on first user report.
"""

from __future__ import annotations

import inspect

import prsm.compute.model_registry as registry_pkg


# ──────────────────────────────────────────────────────────────────────────
# __all__ contract
# ──────────────────────────────────────────────────────────────────────────


EXPECTED_PUBLIC_NAMES = frozenset({
    # Constants
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SIGNING_DOMAIN",
    # Dataclasses
    "ManifestShardEntry",
    "ModelManifest",
    # Signing functions
    "is_signed",
    "sign_manifest",
    "verify_manifest",
    # Registry classes
    "ModelRegistry",
    "InMemoryModelRegistry",
    "FilesystemModelRegistry",
    "manifest_from_model",
    # Exceptions
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelAlreadyRegisteredError",
    "ManifestVerificationError",
})


class TestPublicSurfacePins:
    def test_all_matches_expected_set(self):
        # If this fails, either __all__ has drifted or the test needs
        # an explicit decision about whether to add/remove a name.
        assert set(registry_pkg.__all__) == EXPECTED_PUBLIC_NAMES

    def test_no_duplicate_names_in_all(self):
        # __all__ as a list — duplicates would suggest a stale merge
        all_list = registry_pkg.__all__
        assert len(all_list) == len(set(all_list)), (
            f"duplicates in __all__: {all_list}"
        )

    def test_every_name_resolves(self):
        for name in registry_pkg.__all__:
            assert hasattr(registry_pkg, name), (
                f"__all__ advertises {name!r} but it doesn't resolve "
                f"on the module"
            )
            assert getattr(registry_pkg, name) is not None, (
                f"{name!r} resolved to None — likely a stale import"
            )


# ──────────────────────────────────────────────────────────────────────────
# Type expectations — catches accidental shadowing
# ──────────────────────────────────────────────────────────────────────────


class TestExportedTypes:
    def test_constants_are_correct_types(self):
        assert isinstance(registry_pkg.MANIFEST_SCHEMA_VERSION, int)
        assert isinstance(registry_pkg.MANIFEST_SIGNING_DOMAIN, bytes)

    def test_dataclasses_are_classes(self):
        assert inspect.isclass(registry_pkg.ManifestShardEntry)
        assert inspect.isclass(registry_pkg.ModelManifest)

    def test_signing_helpers_are_callable(self):
        assert callable(registry_pkg.sign_manifest)
        assert callable(registry_pkg.verify_manifest)
        assert callable(registry_pkg.is_signed)

    def test_registry_types_are_classes(self):
        assert inspect.isclass(registry_pkg.ModelRegistry)
        assert inspect.isclass(registry_pkg.InMemoryModelRegistry)
        assert inspect.isclass(registry_pkg.FilesystemModelRegistry)

    def test_concrete_registries_subclass_abc(self):
        assert issubclass(
            registry_pkg.InMemoryModelRegistry, registry_pkg.ModelRegistry
        )
        assert issubclass(
            registry_pkg.FilesystemModelRegistry, registry_pkg.ModelRegistry
        )

    def test_manifest_from_model_is_callable(self):
        assert callable(registry_pkg.manifest_from_model)

    def test_exceptions_form_a_hierarchy(self):
        # Every exception advertised by __all__ MUST inherit from the
        # documented base. A typo here breaks `except ModelRegistryError`
        # callers who'd then leak the underlying error type.
        assert issubclass(registry_pkg.ModelRegistryError, Exception)
        for name in [
            "ModelNotFoundError",
            "ModelAlreadyRegisteredError",
            "ManifestVerificationError",
        ]:
            cls = getattr(registry_pkg, name)
            assert issubclass(cls, registry_pkg.ModelRegistryError), (
                f"{name} must subclass ModelRegistryError"
            )


# ──────────────────────────────────────────────────────────────────────────
# Re-export integrity — the package surface and the underlying module
# attributes are the same objects.
# ──────────────────────────────────────────────────────────────────────────


class TestReExportIdentity:
    """A sneaky regression mode: someone re-defines a name at the
    package level (e.g., adds a local wrapper function) and the
    direct-import path stops being identity-equal to the
    submodule-import path. Downstream code using
    `isinstance(x, ModelManifest)` would then start failing on
    objects produced by the submodule. These tests pin identity."""

    def test_models_re_exports_match_submodule(self):
        from prsm.compute.model_registry import models as _models_mod
        assert registry_pkg.ManifestShardEntry is _models_mod.ManifestShardEntry
        assert registry_pkg.ModelManifest is _models_mod.ModelManifest
        assert registry_pkg.MANIFEST_SCHEMA_VERSION is _models_mod.MANIFEST_SCHEMA_VERSION
        assert registry_pkg.MANIFEST_SIGNING_DOMAIN is _models_mod.MANIFEST_SIGNING_DOMAIN

    def test_signing_re_exports_match_submodule(self):
        from prsm.compute.model_registry import signing as _signing_mod
        assert registry_pkg.sign_manifest is _signing_mod.sign_manifest
        assert registry_pkg.verify_manifest is _signing_mod.verify_manifest
        assert registry_pkg.is_signed is _signing_mod.is_signed

    def test_registry_re_exports_match_submodule(self):
        from prsm.compute.model_registry import registry as _registry_mod
        assert registry_pkg.ModelRegistry is _registry_mod.ModelRegistry
        assert registry_pkg.InMemoryModelRegistry is _registry_mod.InMemoryModelRegistry
        assert registry_pkg.FilesystemModelRegistry is _registry_mod.FilesystemModelRegistry
        assert registry_pkg.manifest_from_model is _registry_mod.manifest_from_model
        assert registry_pkg.ModelRegistryError is _registry_mod.ModelRegistryError
        assert registry_pkg.ModelNotFoundError is _registry_mod.ModelNotFoundError
        assert registry_pkg.ModelAlreadyRegisteredError is _registry_mod.ModelAlreadyRegisteredError
        assert registry_pkg.ManifestVerificationError is _registry_mod.ManifestVerificationError


# ──────────────────────────────────────────────────────────────────────────
# Star-import smoke
# ──────────────────────────────────────────────────────────────────────────


class TestStarImport:
    def test_star_import_yields_only_public_names(self):
        # `from prsm.compute.model_registry import *` should only
        # bring in names listed in __all__. Catches accidental leakage
        # of internal helpers (_validate_fs_id, _hash_shard, etc.).
        ns: dict = {}
        exec("from prsm.compute.model_registry import *", ns)
        # Drop builtins and module-level dunders
        imported = {k for k in ns if not k.startswith("_")}
        # The exec namespace also includes everything * brings in;
        # filter to module-defined names that are not dunder-private.
        assert imported == EXPECTED_PUBLIC_NAMES, (
            f"star-import surface drift:\n"
            f"  unexpected: {imported - EXPECTED_PUBLIC_NAMES}\n"
            f"  missing:    {EXPECTED_PUBLIC_NAMES - imported}"
        )
