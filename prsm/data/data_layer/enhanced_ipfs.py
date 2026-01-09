"""
Enhanced IPFS Client for PRSM
Extended from Co-Lab's ipfs_client.py with model storage and provenance tracking
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    print("Warning: ipfshttpclient not available. IPFS functionality will be simulated.")

from ..core.config import settings
from ..core.models import ProvenanceRecord, TeacherModel, ModelShard
from ..tokenomics.ftns_service import ftns_service


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
        
        Args:
            model_data: Serialized model data (e.g., pickled, ONNX, etc.)
            metadata: Model metadata including type, version, performance metrics
            
        Returns:
            CID of stored model
        """
        await self._ensure_initialized()
        
        # Validate model size
        size_mb = len(model_data) / (1024 * 1024)
        if size_mb > MAX_MODEL_SIZE_MB:
            raise ValueError(f"Model size {size_mb:.1f}MB exceeds limit {MAX_MODEL_SIZE_MB}MB")
        
        # Create comprehensive model package
        model_package = {
            "model_data": model_data.hex(),  # Convert to hex for JSON serialization
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
        
        # Serialize package
        package_json = json.dumps(model_package, indent=2)
        package_bytes = package_json.encode('utf-8')
        
        # Store in IPFS
        cid = await self._store_content(package_bytes)
        
        if cid:
            # Create provenance record
            await self._create_provenance_record(
                cid, 
                metadata.get('uploader_id', 'unknown'),
                'model',
                len(model_data)
            )
            
            # Reward model contribution
            if ENABLE_ACCESS_REWARDS and 'uploader_id' in metadata:
                await ftns_service.reward_contribution(
                    metadata['uploader_id'],
                    'model',
                    1.0,
                    {'cid': cid, 'size_mb': size_mb, 'model_type': metadata.get('model_type')}
                )
        
        return cid
    
    async def store_dataset(self, data: bytes, provenance: Dict[str, Any]) -> str:
        """
        Store a dataset in IPFS with detailed provenance information
        
        Args:
            data: Dataset bytes (could be CSV, JSON, Parquet, etc.)
            provenance: Provenance metadata including source, processing history
            
        Returns:
            CID of stored dataset
        """
        # Create dataset package with provenance
        dataset_package = {
            "data": data.hex(),  # Convert to hex for JSON serialization
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
        
        # Serialize package
        package_json = json.dumps(dataset_package, indent=2)
        package_bytes = package_json.encode('utf-8')
        
        # Store in IPFS
        cid = await self._store_content(package_bytes)
        
        if cid:
            # Create provenance record
            await self._create_provenance_record(
                cid,
                provenance.get('uploader_id', 'unknown'),
                'dataset',
                len(data)
            )
            
            # Reward data contribution
            if ENABLE_ACCESS_REWARDS and 'uploader_id' in provenance:
                size_mb = len(data) / (1024 * 1024)
                await ftns_service.reward_contribution(
                    provenance['uploader_id'],
                    'data',
                    size_mb,
                    {'cid': cid, 'dataset_type': provenance.get('data_type')}
                )
        
        return cid
    
    async def retrieve_with_provenance(self, cid: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Retrieve content from IPFS and return with provenance information
        
        Args:
            cid: Content identifier to retrieve
            
        Returns:
            Tuple of (content_bytes, provenance_metadata)
        """
        await self._ensure_initialized()
        
        # Retrieve package from IPFS
        package_bytes = await self._retrieve_content(cid)
        if not package_bytes:
            raise ValueError(f"Failed to retrieve content for CID: {cid}")
        
        try:
            # Parse package
            package_json = package_bytes.decode('utf-8')
            package = json.loads(package_json)
            
            # Extract content and metadata
            if 'model_data' in package:
                content = bytes.fromhex(package['model_data'])
                metadata = package['metadata']
            elif 'data' in package:
                content = bytes.fromhex(package['data'])
                metadata = package['provenance']
            else:
                # Legacy content without PRSM packaging
                content = package_bytes
                metadata = {
                    "storage_type": "legacy",
                    "size_bytes": len(package_bytes),
                    "retrieved_at": datetime.now(timezone.utc).isoformat()
                }
            
            return content, metadata
            
        except (json.JSONDecodeError, KeyError) as e:
            # Handle legacy or non-PRSM content
            return package_bytes, {
                "storage_type": "legacy",
                "size_bytes": len(package_bytes),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "error": f"Package parsing failed: {e}"
            }
    
    async def track_access(self, cid: str, accessor_id: str) -> None:
        """
        Track access to content for usage analytics and royalty calculation
        
        Args:
            cid: Content identifier that was accessed
            accessor_id: ID of user/system accessing the content
        """
        if not ENABLE_PROVENANCE_TRACKING:
            return
        
        access_record = {
            "accessor_id": accessor_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "access_type": "retrieve"
        }
        
        # Update access log
        if cid not in self.access_log:
            self.access_log[cid] = []
        self.access_log[cid].append(access_record)
        
        # Update provenance record if exists
        if cid in self.provenance_cache:
            self.provenance_cache[cid].access_count += 1
            self.provenance_cache[cid].last_accessed = datetime.now(timezone.utc)
            
            # Calculate and distribute royalties
            if ENABLE_ACCESS_REWARDS:
                royalty = await ftns_service.calculate_royalties(cid, 1)
                if royalty > 0:
                    await ftns_service.reward_contribution(
                        self.provenance_cache[cid].uploader_id,
                        'data',
                        royalty,
                        {'cid': cid, 'access_type': 'royalty'}
                    )
    
    async def calculate_usage_metrics(self, cid: str) -> Dict[str, Any]:
        """
        Calculate comprehensive usage metrics for content
        
        Args:
            cid: Content identifier to analyze
            
        Returns:
            Dictionary with usage statistics
        """
        metrics = {
            "cid": cid,
            "total_accesses": 0,
            "unique_accessors": 0,
            "first_access": None,
            "last_access": None,
            "access_frequency": 0.0,
            "top_accessors": [],
            "geographic_distribution": {},  # Placeholder for future geo tracking
            "access_patterns": {}
        }
        
        # Get access logs for this CID
        access_logs = self.access_log.get(cid, [])
        
        if access_logs:
            metrics["total_accesses"] = len(access_logs)
            
            # Unique accessors
            unique_accessors = set(log["accessor_id"] for log in access_logs)
            metrics["unique_accessors"] = len(unique_accessors)
            
            # Time analysis
            timestamps = [datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) 
                         for log in access_logs]
            metrics["first_access"] = min(timestamps).isoformat()
            metrics["last_access"] = max(timestamps).isoformat()
            
            # Calculate frequency (accesses per day)
            if len(timestamps) > 1:
                time_span = max(timestamps) - min(timestamps)
                days = max(time_span.total_seconds() / 86400, 1)  # At least 1 day
                metrics["access_frequency"] = len(access_logs) / days
            
            # Top accessors
            accessor_counts = {}
            for log in access_logs:
                accessor_id = log["accessor_id"]
                accessor_counts[accessor_id] = accessor_counts.get(accessor_id, 0) + 1
            
            metrics["top_accessors"] = sorted(
                [{"accessor_id": k, "count": v} for k, v in accessor_counts.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:5]  # Top 5 accessors
        
        # Add provenance info if available
        if cid in self.provenance_cache:
            record = self.provenance_cache[cid]
            metrics["uploader_id"] = record.uploader_id
            metrics["total_rewards_paid"] = record.total_rewards_paid
            metrics["content_type"] = getattr(record, 'content_type', 'unknown')
        
        return metrics
    
    # === Model Registry Integration ===
    
    async def register_model_shard(self, model_cid: str, shard_index: int, 
                                 total_shards: int, hosted_by: List[str]) -> ModelShard:
        """
        Register a model shard for distributed storage
        
        Args:
            model_cid: CID of the complete model
            shard_index: Index of this shard (0-based)
            total_shards: Total number of shards
            hosted_by: List of node IDs hosting this shard
            
        Returns:
            ModelShard instance
        """
        # Calculate verification hash for shard
        shard_content, _ = await self.retrieve_with_provenance(model_cid)
        verification_hash = hashlib.sha256(shard_content).hexdigest()
        
        shard = ModelShard(
            model_cid=model_cid,
            shard_index=shard_index,
            total_shards=total_shards,
            hosted_by=hosted_by,
            verification_hash=verification_hash,
            size_bytes=len(shard_content)
        )
        
        return shard
    
    async def verify_model_integrity(self, cid: str) -> bool:
        """
        Verify the integrity of a stored model
        
        Args:
            cid: Model CID to verify
            
        Returns:
            True if model integrity is verified
        """
        await self._ensure_initialized()
        
        try:
            content, metadata = await self.retrieve_with_provenance(cid)
            
            # Check if metadata contains expected hash
            if 'data_hash' in metadata:
                calculated_hash = hashlib.sha256(content).hexdigest()
                expected_hash = metadata['data_hash']
                
                if calculated_hash == expected_hash:
                    return True
                else:
                    print(f"⚠️ Model integrity check failed for {cid}: hash mismatch")
                    return False
            else:
                # No hash in metadata, calculate and assume valid for legacy content
                print(f"ℹ️ No integrity hash found for {cid}, assuming valid")
                return True
                
        except Exception as e:
            print(f"❌ Model integrity verification failed for {cid}: {e}")
            return False
    
    # === Private Helper Methods ===
    
    async def _store_content(self, content: bytes) -> Optional[str]:
        """Store content in IPFS (or simulate if not connected)"""
        if not self.connected:
            # Simulation mode - generate deterministic fake CID and store content
            content_hash = hashlib.sha256(content).hexdigest()
            fake_cid = f"bafybeig{content_hash[:50]}"  # IPFS-like CID format
            self.simulation_storage[fake_cid] = content  # Store for later retrieval
            print(f"📦 [SIMULATED] Stored {len(content)} bytes with CID: {fake_cid}")
            return fake_cid
        
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.add_bytes(content, timeout=DEFAULT_IPFS_TIMEOUT)
            )
            
            cid = result if isinstance(result, str) else result.get('Hash')
            if cid:
                print(f"📦 Stored {len(content)} bytes to IPFS with CID: {cid}")
                return cid
            else:
                print(f"❌ Failed to store content, unexpected result: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Error storing content to IPFS: {e}")
            return None
    
    async def _retrieve_content(self, cid: str) -> Optional[bytes]:
        """Retrieve content from IPFS (or simulate if not connected)"""
        if not self.connected:
            # Simulation mode - return stored content if available
            if cid in self.simulation_storage:
                content = self.simulation_storage[cid]
                print(f"📥 [SIMULATED] Retrieved {len(content)} bytes for CID: {cid}")
                return content
            else:
                # Return fake content for unknown CIDs
                fake_content = f"SIMULATED_CONTENT_FOR_{cid}".encode('utf-8')
                print(f"📥 [SIMULATED] Retrieved {len(fake_content)} bytes for CID: {cid} (not found, using fake)")
                return fake_content
        
        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                None,
                lambda: self.client.cat(cid, timeout=DEFAULT_IPFS_TIMEOUT)
            )
            
            print(f"📥 Retrieved {len(content)} bytes from IPFS for CID: {cid}")
            return content
            
        except Exception as e:
            print(f"❌ Error retrieving content from IPFS: {e}")
            return None
    
    async def _create_provenance_record(self, cid: str, uploader_id: str, 
                                      content_type: str, size_bytes: int) -> None:
        """Create a provenance record for tracked content"""
        if not ENABLE_PROVENANCE_TRACKING:
            return
        
        record = ProvenanceRecord(
            content_cid=cid,
            uploader_id=uploader_id,
            access_count=0,
            total_rewards_paid=0.0
        )
        
        # Add to cache
        self.provenance_cache[cid] = record
        
        print(f"📋 Created provenance record for {content_type}: {cid}")
    
    # === Public Status Methods ===
    
    async def get_status(self) -> Dict[str, Any]:
        """Get IPFS client status and statistics"""
        status = {
            "connected": self.connected,
            "api_address": IPFS_API_ADDR,
            "provenance_tracking": ENABLE_PROVENANCE_TRACKING,
            "access_rewards": ENABLE_ACCESS_REWARDS,
            "tracked_content": len(self.provenance_cache),
            "total_accesses": sum(len(logs) for logs in self.access_log.values()),
            "max_model_size_mb": MAX_MODEL_SIZE_MB
        }
        
        if self.connected and self.client:
            try:
                loop = asyncio.get_running_loop()
                node_info = await loop.run_in_executor(None, self.client.id)
                status["node_id"] = node_info.get('ID', 'unknown')
                status["node_version"] = node_info.get('AgentVersion', 'unknown')
            except Exception as e:
                status["node_error"] = str(e)
        
        return status


# === Global Client Instance ===
# Note: Initialize when needed to avoid event loop issues
prsm_ipfs_client = None

def get_ipfs_client() -> PRSMIPFSClient:
    """Get or create global IPFS client instance"""
    global prsm_ipfs_client
    if prsm_ipfs_client is None:
        prsm_ipfs_client = PRSMIPFSClient()
    return prsm_ipfs_client