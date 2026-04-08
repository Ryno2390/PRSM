"""
Local content-addressed blob store.

Blobs are stored under:
    {data_dir}/{first 2 hex chars of ContentHash.hex()}/{remaining hex chars}

Writes are atomic: data is first written to a .tmp file in the same directory
and then renamed via os.replace(), which is atomic on POSIX and best-effort on
Windows.
"""

from __future__ import annotations

import os
import tempfile

import aiofiles

from prsm.storage.exceptions import ContentNotFoundError
from prsm.storage.models import AlgorithmID, ContentHash


class BlobStore:
    """
    Local, content-addressed file store.

    Parameters
    ----------
    data_dir:
        Root directory under which blobs are stored.  The directory (and any
        required parents) is created on first write.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, content_hash: ContentHash) -> str:
        """Return the filesystem path for *content_hash*.

        Layout: ``{data_dir}/{first 2 hex chars of digest}/{full content hash hex}``

        The bucket prefix uses the first 2 hex chars of the **digest** (not the
        algorithm-prefixed hex) so that SHA-256 blobs spread across 256 buckets
        instead of all landing in ``01/``.
        """
        digest_hex = content_hash.digest.hex()
        prefix = digest_hex[:2]
        return os.path.join(self.data_dir, prefix, content_hash.hex())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(self, data: bytes) -> ContentHash:
        """Hash *data*, write it to disk atomically, and return its ContentHash.

        If the blob already exists the write is skipped (deduplication).
        """
        content_hash = ContentHash.from_data(data, AlgorithmID.SHA256)
        dest_path = self._path_for(content_hash)

        # Fast path: already present — nothing to do.
        if os.path.exists(dest_path):
            return content_hash

        # Ensure the two-level directory tree exists.
        parent_dir = os.path.dirname(dest_path)
        os.makedirs(parent_dir, exist_ok=True)

        # Atomic write: write to a temp file in the same directory, then
        # os.replace() so readers never see a partial file.
        fd, tmp_path = tempfile.mkstemp(dir=parent_dir, suffix=".tmp")
        try:
            async with aiofiles.open(fd, mode="wb") as fh:
                await fh.write(data)
            os.replace(tmp_path, dest_path)
        except Exception:
            # Clean up the temp file on failure; ignore errors during cleanup.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return content_hash

    async def retrieve(self, content_hash: ContentHash) -> bytes:
        """Return the raw bytes stored under *content_hash*.

        Raises
        ------
        ContentNotFoundError
            If the blob is not present in the local store.
        """
        path = self._path_for(content_hash)
        if not os.path.exists(path):
            raise ContentNotFoundError(content_hash.hex())

        async with aiofiles.open(path, mode="rb") as fh:
            return await fh.read()

    async def exists(self, content_hash: ContentHash) -> bool:
        """Return ``True`` if the blob is present in the local store."""
        return os.path.exists(self._path_for(content_hash))

    async def delete(self, content_hash: ContentHash) -> None:
        """Remove the blob from the local store.

        No-op if the blob is not present.
        """
        path = self._path_for(content_hash)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
