"""
Enhanced IPFS Client for PRSM
Extended from Co-Lab's ipfs_client.py with model storage and provenance tracking
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    print("Warning: ipfshttpclient not available. IPFS functionality will be simulated.")

from prsm.core.config import settings
from prsm.core.models import ProvenanceRecord, TeacherModel, ModelShard
from prsm.economy.tokenomics.ftns_service import get_ftns_service, FTNSTransactionType

logger = logging.getLogger(__name__)

# === IPFS Configuration ===

# Get IPFS API multiaddress from settings
IPFS_API_ADDR = getattr(settings, "IPFS_API_MULTIADDR", "/ip4/127.0.0.1/tcp/5001")
DEFAULT_IPFS_TIMEOUT = float(getattr(settings, "IPFS_TIMEOUT", 60.0))

# PRSM-specific IPFS settings
ENABLE_PROVENANCE_TRACKING = getattr(settings, "PRSM_IPFS_PROVENANCE", True)
ENABLE_ACCESS_REWARDS = getattr(settings, "PRSM_IPFS_REWARDS", True)
MAX_MODEL_SIZE_MB = int(getattr(settings, "PRSM_MAX_MODEL_SIZE_MB", 1000))


class PRSMIPFSClient:
    """
    Enhanced IPFS client for PRSM with model storage and provenance tracking
    Extends Co-Lab's basic IPFS functionality with PRSM-specific features
    """
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.provenance_cache: Dict[str, ProvenanceRecord] = {}
        self.access_log: Dict[str, List[Dict[str, Any]]] = {}
        self.simulation_storage: Dict[str, bytes] = {}  # For simulation mode
        self.ftns_service = get_ftns_service()
        
        # Initialize IPFS connection (will be done lazily to avoid event loop issues)
        self._initialization_started = False
    
    async def _ensure_initialized(self) -> bool:
        """Ensure IPFS connection is initialized"""
        if not self._initialization_started:
            self._initialization_started = True
            await self._initialize_connection()
        return self.connected
    
    async def _initialize_connection(self) -> bool:
        """Initialize connection to IPFS node"""
        if not IPFS_AVAILABLE:
            print("IPFS client running in simulation mode (ipfshttpclient not available)")
            self.connected = False
            return False
        
        try:
            # Run blocking IPFS connection in executor
            loop = asyncio.get_running_loop()
            self.client = await loop.run_in_executor(
                None,
                lambda: ipfshttpclient.connect(
                    addr=IPFS_API_ADDR, 
                    timeout=DEFAULT_IPFS_TIMEOUT
                )
            )
            
            # Verify connection
            node_id = await loop.run_in_executor(None, self.client.id)
            print(f"✅ Connected to IPFS node: {IPFS_API_ADDR} (ID: ...{node_id['ID'][-6:]})")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to IPFS node at {IPFS_API_ADDR}: {e}")
            print("IPFS functionality will run in simulation mode.")
            self.connected = False
            return False
    
    # === Enhanced Model Storage ===
    
    async def store_model(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        """
        Store a machine learning model in IPFS with comprehensive metadata
        """
        await self._ensure_initialized()
        
        size_mb = len(model_data) / (1024 * 1024)
        if size_mb > MAX_MODEL_SIZE_MB:
            raise ValueError(f"Model size {size_mb:.1f}MB exceeds limit {MAX_MODEL_SIZE_MB}MB")
        
        model_package = {
            "model_data": model_data.hex(),
            "metadata": {
                **metadata,
                "prsm_version": "1.0.0",
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "size_bytes": len(model_data),
                "size_mb": round(size_mb, 2),
                "data_hash": hashlib.sha256(model_data).hexdigest(),
                "storage_type": "model"
            }
        }
        
        package_json = json.dumps(model_package, indent=2)
        package_bytes = package_json.encode('utf-8')
        cid = await self._store_content(package_bytes)
        
        if cid:
            await self._create_provenance_record(
                cid, 
                metadata.get('uploader_id', 'unknown'),
                'model',
                len(model_data)
            )
            
            if ENABLE_ACCESS_REWARDS and 'uploader_id' in metadata:
                self.ftns_service.award_tokens(
                    metadata['uploader_id'],
                    FTNSTransactionType.MODEL_IMPROVEMENT,
                    Decimal('1.0'),
                    description=f"Model storage reward for CID: {cid}"
                )
        
        return cid
    
    async def store_dataset(self, data: bytes, provenance: Dict[str, Any]) -> str:
        """
        Store a dataset in IPFS with detailed provenance information
        """
        dataset_package = {
            "data": data.hex(),
            "provenance": {
                **provenance,
                "prsm_version": "1.0.0",
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "size_bytes": len(data),
                "size_mb": round(len(data) / (1024 * 1024), 2),
                "data_hash": hashlib.sha256(data).hexdigest(),
                "storage_type": "dataset"
            }
        }
        
        package_json = json.dumps(dataset_package, indent=2)
        package_bytes = package_json.encode('utf-8')
        cid = await self._store_content(package_bytes)
        
        if cid:
            await self._create_provenance_record(
                cid,
                provenance.get('uploader_id', 'unknown'),
                'dataset',
                len(data)
            )
            
            if ENABLE_ACCESS_REWARDS and 'uploader_id' in provenance:
                size_mb = Decimal(str(round(len(data) / (1024 * 1024), 2)))
                self.ftns_service.award_tokens(
                    provenance['uploader_id'],
                    FTNSTransactionType.DATA_CONTRIBUTION,
                    size_mb,
                    description=f"Dataset storage reward for CID: {cid}"
                )
        
        return cid
    
    async def retrieve_with_provenance(self, cid: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Retrieve content from IPFS and return with provenance information
        """
        await self._ensure_initialized()
        package_bytes = await self._retrieve_content(cid)
        if not package_bytes:
            raise ValueError(f"Failed to retrieve content for CID: {cid}")
        
        try:
            package_json = package_bytes.decode('utf-8')
            package = json.loads(package_json)
            
            if 'model_data' in package:
                content = bytes.fromhex(package['model_data'])
                metadata = package['metadata']
            elif 'data' in package:
                content = bytes.fromhex(package['data'])
                metadata = package['provenance']
            else:
                content = package_bytes
                metadata = {"storage_type": "legacy", "size_bytes": len(package_bytes)}
            
            return content, metadata
            
        except (json.JSONDecodeError, KeyError) as e:
            return package_bytes, {"storage_type": "legacy", "error": str(e)}
    
    async def track_access(self, cid: str, accessor_id: str) -> None:
        """
        Track access to content for usage analytics and royalty calculation
        """
        if not ENABLE_PROVENANCE_TRACKING:
            return
        
        access_record = {
            "accessor_id": accessor_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "access_type": "retrieve"
        }
        
        if cid not in self.access_log:
            self.access_log[cid] = []
        self.access_log[cid].append(access_record)
        
        if cid in self.provenance_cache:
            record = self.provenance_cache[cid]
            record.access_count += 1
            record.last_accessed = datetime.now(timezone.utc)
            
            if ENABLE_ACCESS_REWARDS:
                # Distribution of royalties (simplified)
                self.ftns_service.award_tokens(
                    record.uploader_id,
                    FTNSTransactionType.SYSTEM_USAGE,
                    Decimal('0.1'),
                    description=f"Royalty for access to {cid}"
                )
    
    async def register_model_shard(self, model_cid: str, shard_index: int, 
                                 total_shards: int, hosted_by: List[str]) -> ModelShard:
        """Register a model shard"""
        shard_content, _ = await self.retrieve_with_provenance(model_cid)
        verification_hash = hashlib.sha256(shard_content).hexdigest()
        
        return ModelShard(
            model_cid=model_cid,
            shard_index=shard_index,
            total_shards=total_shards,
            hosted_by=hosted_by,
            verification_hash=verification_hash,
            size_bytes=len(shard_content)
        )
    
    async def _store_content(self, content: bytes) -> Optional[str]:
        """Store content in IPFS (or simulate if not connected)"""
        if not self.connected:
            content_hash = hashlib.sha256(content).hexdigest()
            fake_cid = f"bafybeig{content_hash[:50]}"
            self.simulation_storage[fake_cid] = content
            return fake_cid
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self.client.add_bytes(content)
        )
        return result if isinstance(result, str) else result.get('Hash')
    
    async def _retrieve_content(self, cid: str) -> Optional[bytes]:
        """Retrieve content from IPFS (or simulate if not connected)"""
        if not self.connected:
            return self.simulation_storage.get(cid, f"SIMULATED_{cid}".encode('utf-8'))
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.cat(cid))
    
    async def _create_provenance_record(self, cid: str, uploader_id: str, 
                                      content_type: str, size_bytes: int) -> None:
        """Create a provenance record"""
        record = ProvenanceRecord(
            content_cid=cid,
            uploader_id=uploader_id,
            access_count=0,
            total_rewards_paid=0.0
        )
        self.provenance_cache[cid] = record

    async def get_status(self) -> Dict[str, Any]:
        return {"connected": self.connected, "tracked_content": len(self.provenance_cache)}


# Global Client Instance
prsm_ipfs_client = None

def get_ipfs_client() -> PRSMIPFSClient:
    global prsm_ipfs_client
    if prsm_ipfs_client is None:
        prsm_ipfs_client = PRSMIPFSClient()
    return prsm_ipfs_client
