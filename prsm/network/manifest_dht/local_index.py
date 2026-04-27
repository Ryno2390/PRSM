"""
Manifest DHT — local provider index.

Phase 3.x.5 Task 2.

Per-node index of manifests THIS node can serve over the DHT,
keyed by ``model_id``. Sits next to the ``FilesystemModelRegistry``
storage layout (Phase 3.x.2):

  <root>/
  ├── dht_index.json              ← this module's persistence
  ├── <model_id_a>/
  │   ├── manifest.json
  │   └── shards/...
  └── <model_id_b>/
      └── ...

Populated implicitly by ``FilesystemModelRegistry.register()`` when
DHT is wired in (Task 5). The index is a read-side optimization —
the source of truth is still the per-model directory tree on disk.
If ``dht_index.json`` is missing or corrupt, the index is rebuilt
by walking the root for directories containing ``manifest.json``.

Concurrency: same single-writer-per-node assumption as the prior
phases (Phase 3.x.2, 3.x.4). No cross-process locking.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union


logger = logging.getLogger(__name__)


# Pinned filename for the on-disk index. Chosen to be obviously
# non-conflicting with Phase 3.x.2's per-model dirs (registry never
# creates a dir named with leading "dht_").
_INDEX_FILENAME = "dht_index.json"

# The Phase 3.x.2 manifest filename — what we look for during a walk.
_MANIFEST_FILENAME = "manifest.json"

# Same identifier-safety regex Phase 3.x.2 uses (defense in depth).
_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_RESERVED_NAMES = frozenset({".", ".."})


class LocalManifestIndex:
    """In-memory + on-disk index of locally-servable manifests.

    Wire-format-agnostic: stores ``model_id → relative path`` from
    the index root. The relative-path representation makes the index
    portable if the operator relocates the root directory.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(
                f"LocalManifestIndex root {self._root} does not exist; "
                f"create the directory before constructing the index"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"LocalManifestIndex root {self._root} is not a directory"
            )

        # In-memory map: model_id → relative path string under root.
        self._entries: Dict[str, str] = {}

        index_path = self._root / _INDEX_FILENAME
        if index_path.exists():
            self._load_or_rebuild(index_path)
        else:
            # Fresh root or first-time index — rebuild from filesystem
            # walk in case this is being attached to an already-populated
            # registry tree (3.x.2 → 3.x.5 migration path).
            self._rebuild_from_walk()
            # Persist the freshly-built index so subsequent constructions
            # take the fast path.
            self._persist()

    # -- public read API ----------------------------------------------------

    def lookup(self, model_id: str) -> Optional[Path]:
        """Return the absolute manifest path for ``model_id``, or
        None if not in the index.

        Validates model_id against the safe-id regex defensively;
        callers passing an unsafe id get None rather than risking
        a path-traversal lookup against the in-memory dict.
        """
        if not isinstance(model_id, str) or not model_id:
            return None
        if not _SAFE_MODEL_ID.fullmatch(model_id) or model_id in _RESERVED_NAMES:
            return None
        rel = self._entries.get(model_id)
        if rel is None:
            return None
        return self._root / rel

    def list_models(self) -> List[str]:
        """Sorted list of all known model_ids."""
        return sorted(self._entries.keys())

    def __contains__(self, model_id: str) -> bool:
        return isinstance(model_id, str) and model_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- public write API ---------------------------------------------------

    def register(self, model_id: str, manifest_path: Union[str, Path]) -> None:
        """Add or replace a model_id → manifest path entry.

        Overwrites silently — the registry-level uniqueness check
        (Phase 3.x.2 ``ModelAlreadyRegisteredError``) fires before
        this index gets called, so a duplicate here would indicate
        a bug at the registry layer that this index is not the
        place to enforce.

        Validates:
          - model_id matches the safe-id regex (defense in depth)
          - manifest_path resolves under self._root (no escapes)

        Persists the index file after every mutation.
        """
        _validate_model_id(model_id)
        path = Path(manifest_path).resolve()
        root_resolved = self._root.resolve()
        try:
            rel = path.relative_to(root_resolved)
        except ValueError:
            raise ValueError(
                f"manifest_path {path} is not under index root "
                f"{root_resolved}; refusing to register"
            )
        # Path-traversal defense in depth: even after relative_to() the
        # rel path may include "..", which means the path was UNDER
        # root in resolution but the supplied path was constructed
        # weirdly. Reject defensively.
        if ".." in rel.parts:
            raise ValueError(
                f"manifest_path {path} produced a relative path "
                f"containing '..'; refusing to register"
            )
        self._entries[model_id] = str(rel)
        self._persist()

    def unregister(self, model_id: str) -> bool:
        """Remove an entry. Returns True if it was present, False otherwise.

        Does NOT touch the underlying manifest.json on disk — the
        registry owns those files. Use this only when the index has
        drifted from the registry (e.g., manual cleanup).

        Note: orphan reconciliation at construction (Phase 3.x.5
        round 1 review MEDIUM-2) will RE-ADD this model_id on next
        instance construction if the on-disk ``manifest.json`` still
        exists. To persistently unregister, also delete the model
        directory at ``<root>/<model_id>/``.
        """
        if model_id in self._entries:
            del self._entries[model_id]
            self._persist()
            return True
        return False

    def rebuild(self) -> None:
        """Discard the in-memory index and rebuild from filesystem walk.

        Useful after manual operator intervention (deleted a model
        directory, manually edited dht_index.json, etc.).
        """
        self._entries.clear()
        self._rebuild_from_walk()
        self._persist()

    # -- internals ---------------------------------------------------------

    def _load_or_rebuild(self, index_path: Path) -> None:
        """Try to load the on-disk index; rebuild from filesystem walk
        on parse failure (corrupt JSON, missing fields, etc.)."""
        try:
            data = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "dht_index.json at %s unreadable (%s); rebuilding from walk",
                index_path, exc,
            )
            self._rebuild_from_walk()
            self._persist()
            return

        if not isinstance(data, dict):
            logger.warning(
                "dht_index.json at %s has unexpected shape (%s); "
                "rebuilding from walk",
                index_path, type(data).__name__,
            )
            self._rebuild_from_walk()
            self._persist()
            return

        # Validate each entry. Drop any that fail the safe-id check or
        # whose target manifest is gone — log a warning so operators
        # can investigate.
        for model_id, rel_path in data.items():
            if not isinstance(model_id, str) or not isinstance(rel_path, str):
                logger.warning(
                    "dht_index.json: dropping non-string entry %r → %r",
                    model_id, rel_path,
                )
                continue
            if (
                not _SAFE_MODEL_ID.fullmatch(model_id)
                or model_id in _RESERVED_NAMES
            ):
                logger.warning(
                    "dht_index.json: dropping unsafe model_id %r",
                    model_id,
                )
                continue
            if ".." in Path(rel_path).parts:
                logger.warning(
                    "dht_index.json: dropping traversal path %r → %r",
                    model_id, rel_path,
                )
                continue
            target = (self._root / rel_path).resolve()
            root_resolved = self._root.resolve()
            try:
                target.relative_to(root_resolved)
            except ValueError:
                logger.warning(
                    "dht_index.json: dropping out-of-root path %r → %r",
                    model_id, rel_path,
                )
                continue
            if not target.exists():
                logger.warning(
                    "dht_index.json: dropping missing-target entry %r → %s",
                    model_id, target,
                )
                continue
            self._entries[model_id] = rel_path

        # MEDIUM-2 from Phase 3.x.5 round 1 review: reconcile orphans.
        # If a manifest.json exists on disk under a valid model_id
        # directory but isn't represented in the JSON-loaded index,
        # auto-add it. This recovers from the divergence that occurs
        # when a writer (e.g., FilesystemModelRegistry._fetch_manifest_via_dht)
        # writes the manifest to disk and then has its
        # subsequent dht.announce() fail — the cache is on disk but
        # the index doesn't know about it. Without reconciliation the
        # node stops serving the cached model to peers indefinitely.
        # Persist the recovered index if any orphan was found.
        orphans_added = self._reconcile_orphans()
        if orphans_added:
            self._persist()

    def _reconcile_orphans(self) -> int:
        """Scan the root for manifest.json files not represented in
        ``self._entries`` and add them. Returns the number of orphans
        recovered. Called after a JSON-based load to catch divergence
        between on-disk caches and the persisted index."""
        added = 0
        if not self._root.exists():
            return 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            if (
                not _SAFE_MODEL_ID.fullmatch(child.name)
                or child.name in _RESERVED_NAMES
            ):
                continue
            if child.name in self._entries:
                continue
            manifest = child / _MANIFEST_FILENAME
            if not manifest.exists():
                continue
            rel = manifest.relative_to(self._root)
            logger.warning(
                "dht_index.json: reconciling orphan model_id %r "
                "(manifest on disk but not indexed) → %s",
                child.name, rel,
            )
            self._entries[child.name] = str(rel)
            added += 1
        return added

    def _rebuild_from_walk(self) -> None:
        """Walk the root for directories containing manifest.json and
        rebuild the index. Implements the Phase 3.x.2 → 3.x.5
        migration path: an index attached to an already-populated
        registry tree picks up every model on first construction.
        """
        if not self._root.exists():
            return
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            if (
                not _SAFE_MODEL_ID.fullmatch(child.name)
                or child.name in _RESERVED_NAMES
            ):
                continue
            manifest = child / _MANIFEST_FILENAME
            if not manifest.exists():
                continue
            rel = manifest.relative_to(self._root)
            self._entries[child.name] = str(rel)

    def _persist(self) -> None:
        """Write the index atomically. Uses .tmp + os.replace —
        same idiom as Phase 3.x.2 / 3.x.4 filesystem stores."""
        index_path = self._root / _INDEX_FILENAME
        tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        # Sort keys so the on-disk file is deterministic across writes
        # — easier diffing + git-friendly if an operator commits the
        # index file (rare but possible).
        text = json.dumps(self._entries, sort_keys=True, indent=2)
        with open(tmp, "wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, index_path)


def _validate_model_id(model_id: str) -> None:
    """Defense in depth — the Phase 3.x.2 registry already validates,
    but we re-check at the index boundary to keep this module safe
    in isolation."""
    if not isinstance(model_id, str):
        raise ValueError(
            f"model_id must be a string, got {type(model_id).__name__}"
        )
    if not model_id or not _SAFE_MODEL_ID.fullmatch(model_id):
        raise ValueError(
            f"model_id={model_id!r} unsafe: must match {_SAFE_MODEL_ID.pattern}"
        )
    if model_id in _RESERVED_NAMES:
        raise ValueError(
            f"model_id={model_id!r} is a reserved filesystem name "
            f"(would resolve to current/parent dir)"
        )
