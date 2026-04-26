"""
PRSM Model Registry — signed model manifests for verifiable inference.

Phase 3.x.2 closes the in-memory-registry caveat from
``phase3.x.1-merge-ready-20260426`` by introducing a ``ModelRegistry``
abstraction with two implementations:

- ``InMemoryModelRegistry`` — preserves the Phase 3.x.1 dict-backed
  behavior for tests and lightweight callers.
- ``FilesystemModelRegistry`` — persists signed manifests + shard bytes
  to disk so a node restart doesn't drop the model registry, and so
  remote verifiers can independently confirm shard bytes match the
  publisher-signed manifest.

Trust model: open publishing + signature-verify. Any node may publish;
the publisher's NodeIdentity Ed25519 key authenticates the manifest;
shard sha256 commitments authenticate the bytes. Foundation curation
is a discovery-layer concern, not a registry concern.

See ``docs/2026-04-26-phase3.x.2-persistent-model-registry-design-plan.md``.
"""

from prsm.compute.model_registry.models import (
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_SIGNING_DOMAIN,
    ManifestShardEntry,
    ModelManifest,
)
from prsm.compute.model_registry.signing import (
    is_signed,
    sign_manifest,
    verify_manifest,
)
from prsm.compute.model_registry.registry import (
    InMemoryModelRegistry,
    ManifestVerificationError,
    ModelAlreadyRegisteredError,
    ModelNotFoundError,
    ModelRegistry,
    ModelRegistryError,
    manifest_from_model,
)

__all__ = [
    # Models (Task 1)
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SIGNING_DOMAIN",
    "ManifestShardEntry",
    "ModelManifest",
    # Signing (Task 2)
    "is_signed",
    "sign_manifest",
    "verify_manifest",
    # Registry (Task 3)
    "ModelRegistry",
    "InMemoryModelRegistry",
    "manifest_from_model",
    # Exceptions (Task 3)
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelAlreadyRegisteredError",
    "ManifestVerificationError",
]
