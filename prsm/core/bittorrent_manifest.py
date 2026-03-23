"""
BitTorrent Manifest System
===========================

Provides a metadata layer for tracking torrents in PRSM.
Mirrors the structure of ipfs_sharding.py for consistency.

The manifest system allows PRSM to:
- Track torrent metadata independently of the BitTorrent protocol
- Optionally pin .torrent files to IPFS for durability
- Link torrents to the PRSM provenance system
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# Check for bencode support
try:
    import bencodepy
    BENCODE_AVAILABLE = True
except ImportError:
    BENCODE_AVAILABLE = False
    bencodepy = None


@dataclass
class PieceInfo:
    """Information about a single piece in a torrent."""
    index: int
    hash: str  # SHA-1 hash from BT spec (40 hex chars)
    size: int
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "hash": self.hash,
            "size": self.size,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PieceInfo":
        return cls(
            index=data["index"],
            hash=data["hash"],
            size=data["size"],
            verified=data.get("verified", False),
        )


@dataclass
class FileEntry:
    """File within a torrent manifest."""
    path: str
    size_bytes: int
    offset_in_torrent: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "offset_in_torrent": self.offset_in_torrent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileEntry":
        return cls(
            path=data["path"],
            size_bytes=data["size_bytes"],
            offset_in_torrent=data.get("offset_in_torrent", 0),
        )


@dataclass
class TorrentManifest:
    """
    Complete metadata for a torrent in the PRSM system.

    This is the canonical record that PRSM uses to track torrents,
    separate from the raw .torrent file bytes.
    """
    infohash: str  # SHA-1 hash (40 hex chars)
    name: str
    total_size: int
    piece_length: int
    pieces: List[PieceInfo] = field(default_factory=list)
    files: List[FileEntry] = field(default_factory=list)
    magnet_uri: str = ""
    torrent_bytes: bytes = b""  # Raw .torrent file
    ipfs_cid: Optional[str] = None  # Optional: .torrent pinned to IPFS
    created_at: float = field(default_factory=time.time)
    created_by_node_id: str = ""
    provenance_id: Optional[str] = None  # Links to PRSM provenance system
    private: bool = False
    comment: str = ""
    announce_list: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_pieces(self) -> int:
        """Number of pieces in the torrent."""
        return len(self.pieces) if self.pieces else max(1, self.total_size // self.piece_length)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        return {
            "infohash": self.infohash,
            "name": self.name,
            "total_size": self.total_size,
            "piece_length": self.piece_length,
            "pieces": [p.to_dict() for p in self.pieces],
            "files": [f.to_dict() for f in self.files],
            "magnet_uri": self.magnet_uri,
            "torrent_bytes": self.torrent_bytes.hex() if self.torrent_bytes else "",
            "ipfs_cid": self.ipfs_cid,
            "created_at": self.created_at,
            "created_by_node_id": self.created_by_node_id,
            "provenance_id": self.provenance_id,
            "private": self.private,
            "comment": self.comment,
            "announce_list": self.announce_list,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TorrentManifest":
        """Deserialize manifest from dictionary."""
        torrent_bytes = data.get("torrent_bytes", "")
        if isinstance(torrent_bytes, str) and torrent_bytes:
            torrent_bytes = bytes.fromhex(torrent_bytes)
        elif not isinstance(torrent_bytes, bytes):
            torrent_bytes = b""

        return cls(
            infohash=data["infohash"],
            name=data["name"],
            total_size=data["total_size"],
            piece_length=data["piece_length"],
            pieces=[PieceInfo.from_dict(p) for p in data.get("pieces", [])],
            files=[FileEntry.from_dict(f) for f in data.get("files", [])],
            magnet_uri=data.get("magnet_uri", ""),
            torrent_bytes=torrent_bytes,
            ipfs_cid=data.get("ipfs_cid"),
            created_at=data.get("created_at", time.time()),
            created_by_node_id=data.get("created_by_node_id", ""),
            provenance_id=data.get("provenance_id"),
            private=data.get("private", False),
            comment=data.get("comment", ""),
            announce_list=data.get("announce_list", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TorrentManifest":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def generate_magnet_uri(self, include_trackers: bool = True) -> str:
        """
        Generate a magnet URI for this torrent.

        Args:
            include_trackers: If True, include tracker URLs from announce_list

        Returns:
            Magnet URI string
        """
        if self.magnet_uri:
            return self.magnet_uri

        # Base magnet URI
        uri = f"magnet:?xt=urn:btih:{self.infohash}&dn={self.name}"

        # Add trackers
        if include_trackers and self.announce_list:
            for tracker in self.announce_list:
                uri += f"&tr={tracker}"

        return uri


class TorrentManifestIndex:
    """
    In-memory index for fast torrent lookups.

    Maintains indexes by infohash and name, with optional LRU eviction
    for memory management.
    """

    def __init__(self, max_size: int = 10000):
        self._by_infohash: Dict[str, TorrentManifest] = {}
        self._by_name: Dict[str, List[str]] = {}  # name -> list of infohashes
        self._access_times: Dict[str, float] = {}
        self._max_size = max_size

    def add(self, manifest: TorrentManifest) -> None:
        """Add a manifest to the index."""
        infohash = manifest.infohash

        # Check if we need to evict
        if len(self._by_infohash) >= self._max_size:
            self.evict_lru(self._max_size // 10)  # Evict 10%

        self._by_infohash[infohash] = manifest
        self._access_times[infohash] = time.time()

        # Index by name (lowercase for case-insensitive search)
        name_key = manifest.name.lower()
        if name_key not in self._by_name:
            self._by_name[name_key] = []
        if infohash not in self._by_name[name_key]:
            self._by_name[name_key].append(infohash)

    def get_by_infohash(self, infohash: str) -> Optional[TorrentManifest]:
        """Look up a manifest by its infohash."""
        manifest = self._by_infohash.get(infohash)
        if manifest:
            self._access_times[infohash] = time.time()
        return manifest

    def search(self, query: str, limit: int = 50) -> List[TorrentManifest]:
        """
        Fuzzy name search for manifests.

        Args:
            query: Search query (case-insensitive)
            limit: Maximum results to return

        Returns:
            List of matching manifests
        """
        query_lower = query.lower()
        results = []

        # Exact match first
        if query_lower in self._by_name:
            for ih in self._by_name[query_lower]:
                manifest = self._by_infohash.get(ih)
                if manifest:
                    results.append(manifest)

        # Then partial matches
        for name_key, infohashes in self._by_name.items():
            if query_lower in name_key and name_key != query_lower:
                for ih in infohashes:
                    manifest = self._by_infohash.get(ih)
                    if manifest and manifest not in results:
                        results.append(manifest)
                        if len(results) >= limit:
                            return results

        return results

    def evict_lru(self, count: int) -> int:
        """
        Evict the least recently used manifests.

        Args:
            count: Number of entries to evict

        Returns:
            Actual number evicted
        """
        if not self._access_times:
            return 0

        # Sort by access time
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        evicted = 0

        for infohash, _ in sorted_items[:count]:
            manifest = self._by_infohash.get(infohash)
            if manifest:
                # Remove from name index
                name_key = manifest.name.lower()
                if name_key in self._by_name:
                    self._by_name[name_key] = [
                        ih for ih in self._by_name[name_key] if ih != infohash
                    ]
                    if not self._by_name[name_key]:
                        del self._by_name[name_key]

            # Remove from main index
            self._by_infohash.pop(infohash, None)
            self._access_times.pop(infohash, None)
            evicted += 1

        return evicted

    def remove(self, infohash: str) -> bool:
        """Remove a manifest from the index."""
        manifest = self._by_infohash.get(infohash)
        if not manifest:
            return False

        # Remove from name index
        name_key = manifest.name.lower()
        if name_key in self._by_name:
            self._by_name[name_key] = [
                ih for ih in self._by_name[name_key] if ih != infohash
            ]
            if not self._by_name[name_key]:
                del self._by_name[name_key]

        # Remove from main index
        del self._by_infohash[infohash]
        self._access_times.pop(infohash, None)
        return True

    def list_all(self, limit: int = 1000) -> List[TorrentManifest]:
        """List all manifests in the index."""
        return list(self._by_infohash.values())[:limit]

    def count(self) -> int:
        """Return number of manifests in the index."""
        return len(self._by_infohash)


class TorrentManifestStore:
    """
    Persistent store for torrent manifests.

    Can be backed by PostgreSQL (via SQLAlchemy) or local SQLite.
    """

    def __init__(self, database_url: str = "sqlite:///~/.prsm/torrent_manifests.db"):
        self.database_url = database_url
        self._initialized = False
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> bool:
        """Initialize the database connection and create tables if needed."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import Column, String, BigInteger, Boolean, Text, LargeBinary, Float
            from sqlalchemy.ext.declarative import declarative_base
            from sqlalchemy import JSON

            Base = declarative_base()

            class TorrentManifestModel(Base):
                __tablename__ = "torrent_manifests"

                infohash = Column(String(40), primary_key=True)
                name = Column(Text, nullable=False)
                total_size = Column(BigInteger, nullable=False)
                piece_length = Column(BigInteger, nullable=False)
                magnet_uri = Column(Text, nullable=False)
                torrent_bytes = Column(LargeBinary, nullable=True)
                ipfs_cid = Column(String(64), nullable=True)
                created_at = Column(Float, nullable=False)
                created_by = Column(String(64), nullable=False)
                provenance_id = Column(String(36), nullable=True)
                private = Column(Boolean, default=False)
                comment = Column(Text, default="")
                announce_list_json = Column(Text, default="[]")
                metadata_json = Column(Text, default="{}")

            self._model = TorrentManifestModel

            # Expand path for SQLite
            db_url = self.database_url
            if db_url.startswith("sqlite:///"):
                path = db_url.replace("sqlite:///", "")
                path = Path(path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                db_url = f"sqlite:///{path}"

            self._engine = create_engine(db_url, echo=False)
            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True

            logger.info(f"TorrentManifestStore initialized: {db_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TorrentManifestStore: {e}")
            return False

    async def save(self, manifest: TorrentManifest) -> bool:
        """Save a manifest to the database."""
        if not self._initialized:
            logger.error("TorrentManifestStore not initialized")
            return False

        try:
            import json
            from sqlalchemy import orm

            session = self._session_factory()
            try:
                # Check if exists
                existing = session.query(self._model).filter_by(
                    infohash=manifest.infohash
                ).first()

                if existing:
                    # Update existing
                    existing.name = manifest.name
                    existing.total_size = manifest.total_size
                    existing.piece_length = manifest.piece_length
                    existing.magnet_uri = manifest.magnet_uri
                    existing.torrent_bytes = manifest.torrent_bytes
                    existing.ipfs_cid = manifest.ipfs_cid
                    existing.created_at = manifest.created_at
                    existing.created_by = manifest.created_by_node_id
                    existing.provenance_id = manifest.provenance_id
                    existing.private = manifest.private
                    existing.comment = manifest.comment
                    existing.announce_list_json = json.dumps(manifest.announce_list)
                    existing.metadata_json = json.dumps(manifest.metadata)
                else:
                    # Create new
                    record = self._model(
                        infohash=manifest.infohash,
                        name=manifest.name,
                        total_size=manifest.total_size,
                        piece_length=manifest.piece_length,
                        magnet_uri=manifest.magnet_uri,
                        torrent_bytes=manifest.torrent_bytes,
                        ipfs_cid=manifest.ipfs_cid,
                        created_at=manifest.created_at,
                        created_by=manifest.created_by_node_id,
                        provenance_id=manifest.provenance_id,
                        private=manifest.private,
                        comment=manifest.comment,
                        announce_list_json=json.dumps(manifest.announce_list),
                        metadata_json=json.dumps(manifest.metadata),
                    )
                    session.add(record)

                session.commit()
                return True

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            return False

    async def load(self, infohash: str) -> Optional[TorrentManifest]:
        """Load a manifest from the database."""
        if not self._initialized:
            return None

        try:
            import json

            session = self._session_factory()
            try:
                record = session.query(self._model).filter_by(infohash=infohash).first()
                if not record:
                    return None

                return TorrentManifest(
                    infohash=record.infohash,
                    name=record.name,
                    total_size=record.total_size,
                    piece_length=record.piece_length,
                    magnet_uri=record.magnet_uri,
                    torrent_bytes=record.torrent_bytes or b"",
                    ipfs_cid=record.ipfs_cid,
                    created_at=record.created_at,
                    created_by_node_id=record.created_by,
                    provenance_id=record.provenance_id,
                    private=record.private,
                    comment=record.comment,
                    announce_list=json.loads(record.announce_list_json or "[]"),
                    metadata=json.loads(record.metadata_json or "{}"),
                )
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return None

    async def list_all(self, limit: int = 1000, offset: int = 0) -> List[TorrentManifest]:
        """List all manifests in the database."""
        if not self._initialized:
            return []

        try:
            import json

            session = self._session_factory()
            try:
                records = session.query(self._model).offset(offset).limit(limit).all()

                manifests = []
                for record in records:
                    manifests.append(TorrentManifest(
                        infohash=record.infohash,
                        name=record.name,
                        total_size=record.total_size,
                        piece_length=record.piece_length,
                        magnet_uri=record.magnet_uri,
                        torrent_bytes=record.torrent_bytes or b"",
                        ipfs_cid=record.ipfs_cid,
                        created_at=record.created_at,
                        created_by_node_id=record.created_by,
                        provenance_id=record.provenance_id,
                        private=record.private,
                        comment=record.comment,
                        announce_list=json.loads(record.announce_list_json or "[]"),
                        metadata=json.loads(record.metadata_json or "{}"),
                    ))
                return manifests
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to list manifests: {e}")
            return []

    async def delete(self, infohash: str) -> bool:
        """Delete a manifest from the database."""
        if not self._initialized:
            return False

        try:
            session = self._session_factory()
            try:
                record = session.query(self._model).filter_by(infohash=infohash).first()
                if record:
                    session.delete(record)
                    session.commit()
                    return True
                return False
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to delete manifest: {e}")
            return False

    async def search(self, query: str, limit: int = 50) -> List[TorrentManifest]:
        """Search manifests by name."""
        if not self._initialized:
            return []

        try:
            import json

            session = self._session_factory()
            try:
                records = session.query(self._model).filter(
                    self._model.name.ilike(f"%{query}%")
                ).limit(limit).all()

                manifests = []
                for record in records:
                    manifests.append(TorrentManifest(
                        infohash=record.infohash,
                        name=record.name,
                        total_size=record.total_size,
                        piece_length=record.piece_length,
                        magnet_uri=record.magnet_uri,
                        torrent_bytes=record.torrent_bytes or b"",
                        ipfs_cid=record.ipfs_cid,
                        created_at=record.created_at,
                        created_by_node_id=record.created_by,
                        provenance_id=record.provenance_id,
                        private=record.private,
                        comment=record.comment,
                        announce_list=json.loads(record.announce_list_json or "[]"),
                        metadata=json.loads(record.metadata_json or "{}"),
                    ))
                return manifests
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to search manifests: {e}")
            return []


def parse_torrent_file(torrent_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse a .torrent file and extract metadata.

    Args:
        torrent_bytes: Raw bytes of a .torrent file

    Returns:
        Dictionary with parsed metadata, or None on error
    """
    if not BENCODE_AVAILABLE:
        logger.warning("bencodepy not available - cannot parse .torrent files")
        return None

    try:
        data = bencodepy.decode(torrent_bytes)

        info = data.get(b'info', {})
        if not info:
            return None

        # Get infohash
        info_bencoded = bencodepy.encode(info)
        infohash = hashlib.sha1(info_bencoded).hexdigest()

        # Get file list
        files = []
        total_size = 0

        if b'files' in info:
            # Multi-file torrent
            for f in info[b'files']:
                path = b'/'.join(f[b'path']).decode('utf-8', errors='replace')
                size = f[b'length']
                files.append({'path': path, 'size': size})
                total_size += size
        else:
            # Single-file torrent
            name = info.get(b'name', b'unknown').decode('utf-8', errors='replace')
            size = info.get(b'length', 0)
            files.append({'path': name, 'size': size})
            total_size = size

        # Get piece hashes
        pieces_raw = info.get(b'pieces', b'')
        piece_length = info.get(b'piece length', 262144)
        num_pieces = len(pieces_raw) // 20

        pieces = []
        for i in range(num_pieces):
            piece_hash = pieces_raw[i*20:(i+1)*20].hex()
            piece_size = min(piece_length, total_size - i * piece_length)
            pieces.append({
                'index': i,
                'hash': piece_hash,
                'size': piece_size,
            })

        # Get announce list
        announce_list = []
        if b'announce-list' in data:
            for tier in data[b'announce-list']:
                for tracker in tier:
                    announce_list.append(tracker.decode('utf-8', errors='replace'))
        elif b'announce' in data:
            announce_list.append(data[b'announce'].decode('utf-8', errors='replace'))

        return {
            'infohash': infohash,
            'name': info.get(b'name', b'unknown').decode('utf-8', errors='replace'),
            'total_size': total_size,
            'piece_length': piece_length,
            'num_pieces': num_pieces,
            'pieces': pieces,
            'files': files,
            'private': info.get(b'private', 0) == 1,
            'comment': data.get(b'comment', b'').decode('utf-8', errors='replace'),
            'announce_list': announce_list,
            'created_by': data.get(b'created by', b'').decode('utf-8', errors='replace'),
            'creation_date': data.get(b'creation date', 0),
        }

    except Exception as e:
        logger.error(f"Failed to parse torrent file: {e}")
        return None


def create_manifest_from_torrent(
    torrent_bytes: bytes,
    node_id: str,
    provenance_id: Optional[str] = None,
) -> Optional[TorrentManifest]:
    """
    Create a TorrentManifest from raw .torrent bytes.

    Args:
        torrent_bytes: Raw bytes of a .torrent file
        node_id: ID of the node creating the manifest
        provenance_id: Optional PRSM provenance ID

    Returns:
        TorrentManifest, or None on error
    """
    parsed = parse_torrent_file(torrent_bytes)
    if not parsed:
        return None

    # Generate magnet URI
    magnet_uri = f"magnet:?xt=urn:btih:{parsed['infohash']}&dn={parsed['name']}"
    for tracker in parsed['announce_list'][:5]:  # Limit to 5 trackers
        magnet_uri += f"&tr={tracker}"

    return TorrentManifest(
        infohash=parsed['infohash'],
        name=parsed['name'],
        total_size=parsed['total_size'],
        piece_length=parsed['piece_length'],
        pieces=[PieceInfo(**p) for p in parsed['pieces']],
        files=[FileEntry(path=f['path'], size_bytes=f['size']) for f in parsed['files']],
        magnet_uri=magnet_uri,
        torrent_bytes=torrent_bytes,
        created_by_node_id=node_id,
        provenance_id=provenance_id,
        private=parsed['private'],
        comment=parsed['comment'],
        announce_list=parsed['announce_list'],
    )
