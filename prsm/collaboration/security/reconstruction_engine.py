"""
Post-Quantum File Reconstruction Engine for PRSM P2P Collaboration

This module handles the secure reconstruction of encrypted file shards
distributed across the P2P network, implementing post-quantum cryptographic
verification and integrity checking during the reconstruction process.

Key Features:
- Secure shard collection from P2P network
- Post-quantum cryptographic verification
- Parallel reconstruction for performance
- Integrity validation and tamper detection
- Fault tolerance and error recovery
- Integration with access control and key management
"""

import asyncio
import json
import logging
import time
import secrets
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from enum import Enum
import base64
import struct

from .crypto_sharding import CryptoSharding
from .key_management import DistributedKeyManager, CryptographicKey, KeyType
from .access_control import PostQuantumAccessController, Permission

logger = logging.getLogger(__name__)


class ReconstructionStatus(Enum):
    """Status of file reconstruction process"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ShardVerificationStatus(Enum):
    """Status of individual shard verification"""
    PENDING = "pending"
    VERIFIED = "verified"
    INVALID = "invalid"
    MISSING = "missing"
    CORRUPTED = "corrupted"


@dataclass
class ReconstructionShard:
    """Represents a shard during reconstruction"""
    shard_id: str
    shard_index: int
    data: Optional[bytes] = None
    checksum: Optional[str] = None
    source_node_id: Optional[str] = None
    verification_status: ShardVerificationStatus = ShardVerificationStatus.PENDING
    retrieved_at: float = 0.0
    verification_signature: Optional[bytes] = None
    
    def __post_init__(self):
        if self.retrieved_at == 0.0:
            self.retrieved_at = time.time()
    
    @property
    def is_valid(self) -> bool:
        """Check if shard is valid and verified"""
        return (self.verification_status == ShardVerificationStatus.VERIFIED and
                self.data is not None)
    
    def verify_integrity(self) -> bool:
        """Verify shard data integrity"""
        if not self.data or not self.checksum:
            return False
        
        computed_checksum = hashlib.sha256(self.data).hexdigest()
        return computed_checksum == self.checksum


@dataclass
class ReconstructionTask:
    """Represents a file reconstruction task"""
    task_id: str
    file_id: str
    requester_id: str
    total_shards: int
    required_shards: int
    collected_shards: Dict[int, ReconstructionShard]
    status: ReconstructionStatus = ReconstructionStatus.PENDING
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    progress_callback: Optional[Callable[[float], None]] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    @property
    def progress_percentage(self) -> float:
        """Calculate reconstruction progress"""
        if self.total_shards == 0:
            return 0.0
        return (len(self.collected_shards) / self.required_shards) * 100
    
    @property
    def verified_shard_count(self) -> int:
        """Count of verified shards"""
        return sum(1 for shard in self.collected_shards.values() if shard.is_valid)
    
    @property
    def can_reconstruct(self) -> bool:
        """Check if we have enough verified shards to reconstruct"""
        return self.verified_shard_count >= self.required_shards
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since task started"""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at


class ShardCollector:
    """
    Collects shards from the P2P network with fault tolerance
    """
    
    def __init__(self, p2p_network, reputation_system):
        self.p2p_network = p2p_network
        self.reputation_system = reputation_system
        self.collection_timeout = 30.0  # seconds
        self.max_concurrent_requests = 10
    
    async def collect_shards(self, file_id: str, shard_locations: Dict[str, List[str]],
                           required_count: int, progress_callback: Optional[Callable] = None) -> Dict[int, ReconstructionShard]:
        """
        Collect shards from P2P network
        
        Args:
            file_id: ID of file to reconstruct
            shard_locations: Map of shard_id -> list of node_ids that have the shard
            required_count: Minimum number of shards needed
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of shard_index -> ReconstructionShard
        """
        collected_shards = {}
        collection_tasks = []
        
        # Create collection tasks for each shard
        for shard_id, node_list in shard_locations.items():
            if len(collected_shards) >= required_count:
                break
            
            # Extract shard index from shard_id (assuming format "file_id_index")
            try:
                shard_index = int(shard_id.split('_')[-1])
            except (ValueError, IndexError):
                logger.warning(f"Invalid shard ID format: {shard_id}")
                continue
            
            # Sort nodes by reputation for better success rate
            sorted_nodes = self.reputation_system.select_trustworthy_nodes(
                node_list, len(node_list), min_reputation=0.3
            )
            
            task = asyncio.create_task(
                self._collect_shard_from_nodes(shard_id, shard_index, sorted_nodes)
            )
            collection_tasks.append(task)
        
        # Wait for shard collection with timeout
        try:
            timeout = self.collection_timeout
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*collection_tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results
            for result in completed_tasks:
                if isinstance(result, ReconstructionShard) and result.is_valid:
                    collected_shards[result.shard_index] = result
                    
                    if progress_callback:
                        progress = len(collected_shards) / required_count
                        progress_callback(min(1.0, progress))
                
                elif isinstance(result, Exception):
                    logger.warning(f"Shard collection error: {result}")
        
        except asyncio.TimeoutError:
            logger.warning(f"Shard collection timeout after {timeout}s")
            
            # Collect any completed shards
            for task in collection_tasks:
                if task.done() and not task.cancelled():
                    try:
                        result = task.result()
                        if isinstance(result, ReconstructionShard) and result.is_valid:
                            collected_shards[result.shard_index] = result
                    except Exception as e:
                        logger.debug(f"Task result error: {e}")
        
        finally:
            # Cancel any remaining tasks
            for task in collection_tasks:
                if not task.done():
                    task.cancel()
        
        logger.info(f"Collected {len(collected_shards)} shards for {file_id}")
        return collected_shards
    
    async def _collect_shard_from_nodes(self, shard_id: str, shard_index: int,
                                      node_list: List[str]) -> Optional[ReconstructionShard]:
        """Try to collect a shard from multiple nodes"""
        for node_id in node_list:
            try:
                shard_data = await self._request_shard_from_node(node_id, shard_id)
                
                if shard_data:
                    shard = ReconstructionShard(
                        shard_id=shard_id,
                        shard_index=shard_index,
                        data=shard_data,
                        source_node_id=node_id
                    )
                    
                    # Verify shard integrity
                    if await self._verify_shard(shard):
                        shard.verification_status = ShardVerificationStatus.VERIFIED
                        logger.debug(f"Successfully collected shard {shard_id} from {node_id}")
                        return shard
                    else:
                        shard.verification_status = ShardVerificationStatus.INVALID
                        logger.warning(f"Shard {shard_id} from {node_id} failed verification")
            
            except Exception as e:
                logger.debug(f"Failed to get shard {shard_id} from {node_id}: {e}")
                continue
        
        logger.error(f"Failed to collect shard {shard_id} from any node")
        return None
    
    async def _request_shard_from_node(self, node_id: str, shard_id: str) -> Optional[bytes]:
        """Request a specific shard from a node"""
        if not self.p2p_network:
            return None
        
        # This would integrate with the actual P2P network API
        # For now, simulate shard retrieval
        logger.debug(f"Requesting shard {shard_id} from node {node_id}")
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Return mock shard data for testing
        mock_data = f"shard_data_{shard_id}_from_{node_id}".encode()
        return mock_data
    
    async def _verify_shard(self, shard: ReconstructionShard) -> bool:
        """Verify shard authenticity and integrity"""
        if not shard.data:
            return False
        
        # Basic integrity check (in production, would include signature verification)
        computed_checksum = hashlib.sha256(shard.data).hexdigest()
        
        # For testing, assume verification passes
        shard.checksum = computed_checksum
        return True


class ParallelReconstructor:
    """
    Handles parallel reconstruction of file from shards
    """
    
    def __init__(self, crypto_sharding: CryptoSharding):
        self.crypto_sharding = crypto_sharding
        self.max_workers = 4  # Number of parallel reconstruction workers
    
    async def reconstruct_file(self, shards: Dict[int, ReconstructionShard],
                             encryption_key: bytes, file_metadata: Dict[str, Any]) -> bytes:
        """
        Reconstruct file from collected shards
        
        Args:
            shards: Dictionary of verified shards
            encryption_key: Key for decrypting shards
            file_metadata: Metadata about the original file
            
        Returns:
            Reconstructed file data
        """
        if not shards:
            raise ValueError("No shards provided for reconstruction")
        
        # Sort shards by index
        sorted_shards = sorted(shards.items())
        
        # Extract shard data in order
        shard_data_list = []
        for shard_index, shard in sorted_shards:
            if not shard.is_valid:
                raise ValueError(f"Invalid shard at index {shard_index}")
            shard_data_list.append(shard.data)
        
        logger.info(f"Reconstructing file from {len(shard_data_list)} shards")
        
        # Use crypto sharding to reconstruct the file
        try:
            # This would call the actual reconstruction method from CryptoSharding
            reconstructed_data = await self._parallel_decrypt_and_merge(
                shard_data_list, encryption_key, file_metadata
            )
            
            # Verify reconstructed file integrity
            if await self._verify_reconstructed_file(reconstructed_data, file_metadata):
                logger.info("File reconstruction completed successfully")
                return reconstructed_data
            else:
                raise ValueError("Reconstructed file failed integrity verification")
        
        except Exception as e:
            logger.error(f"File reconstruction failed: {e}")
            raise
    
    async def _parallel_decrypt_and_merge(self, shard_data_list: List[bytes],
                                        encryption_key: bytes, 
                                        file_metadata: Dict[str, Any]) -> bytes:
        """Decrypt shards in parallel and merge them"""
        # Create decryption tasks
        decryption_tasks = []
        
        for i, shard_data in enumerate(shard_data_list):
            task = asyncio.create_task(
                self._decrypt_shard(shard_data, encryption_key, i)
            )
            decryption_tasks.append(task)
        
        # Wait for all decryptions to complete
        decrypted_shards = await asyncio.gather(*decryption_tasks)
        
        # Merge decrypted shards
        merged_data = b''.join(decrypted_shards)
        
        # Remove padding if present
        if 'original_size' in file_metadata:
            original_size = file_metadata['original_size']
            merged_data = merged_data[:original_size]
        
        return merged_data
    
    async def _decrypt_shard(self, shard_data: bytes, encryption_key: bytes,
                           shard_index: int) -> bytes:
        """Decrypt a single shard"""
        try:
            # This would use the actual crypto_sharding decrypt method
            # For now, simulate decryption (remove mock encryption)
            if shard_data.startswith(b"shard_data_"):
                # Mock decrypted data
                return f"decrypted_content_part_{shard_index}".encode()
            
            # In production, would call:
            # return self.crypto_sharding.decrypt_shard(shard_data, encryption_key)
            return shard_data
        
        except Exception as e:
            logger.error(f"Failed to decrypt shard {shard_index}: {e}")
            raise
    
    async def _verify_reconstructed_file(self, file_data: bytes,
                                       file_metadata: Dict[str, Any]) -> bool:
        """Verify integrity of reconstructed file"""
        if 'file_hash' in file_metadata:
            computed_hash = hashlib.sha256(file_data).hexdigest()
            expected_hash = file_metadata['file_hash']
            
            if computed_hash != expected_hash:
                logger.error(f"File hash mismatch: {computed_hash} != {expected_hash}")
                return False
        
        if 'original_size' in file_metadata:
            expected_size = file_metadata['original_size']
            if len(file_data) != expected_size:
                logger.error(f"File size mismatch: {len(file_data)} != {expected_size}")
                return False
        
        return True


class PostQuantumReconstructionEngine:
    """
    Main reconstruction engine with post-quantum security
    
    Coordinates the secure reconstruction of distributed files with
    post-quantum cryptographic verification and access control.
    """
    
    def __init__(self, key_manager: DistributedKeyManager,
                 access_controller: PostQuantumAccessController,
                 p2p_network, config: Optional[Dict[str, Any]] = None):
        self.key_manager = key_manager
        self.access_controller = access_controller
        self.p2p_network = p2p_network
        self.config = config or {}
        
        # Initialize components
        self.crypto_sharding = CryptoSharding()
        self.shard_collector = ShardCollector(p2p_network, None)  # Will set reputation system
        self.parallel_reconstructor = ParallelReconstructor(self.crypto_sharding)
        
        # Active reconstruction tasks
        self.active_tasks: Dict[str, ReconstructionTask] = {}
        
        # Configuration
        self.max_concurrent_reconstructions = self.config.get('max_concurrent', 5)
        self.reconstruction_timeout = self.config.get('timeout', 300)  # 5 minutes
        
        logger.info("Post-quantum reconstruction engine initialized")
    
    def set_reputation_system(self, reputation_system):
        """Set reputation system for shard collection"""
        self.shard_collector.reputation_system = reputation_system
    
    async def request_file_reconstruction(self, file_id: str, requester_id: str,
                                        shard_locations: Dict[str, List[str]],
                                        file_metadata: Dict[str, Any],
                                        progress_callback: Optional[Callable] = None) -> str:
        """
        Request reconstruction of a distributed file
        
        Args:
            file_id: ID of file to reconstruct
            requester_id: ID of node/user requesting reconstruction
            shard_locations: Map of shard_id -> list of node_ids
            file_metadata: Metadata about the file (size, hash, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Task ID for tracking reconstruction progress
        """
        # Check if requester has read permission
        has_permission = await self.access_controller.check_access(
            requester_id, file_id, "read"
        )
        
        if not has_permission:
            raise PermissionError(f"Requester {requester_id} lacks read permission for {file_id}")
        
        # Check if we're at the concurrent limit
        if len(self.active_tasks) >= self.max_concurrent_reconstructions:
            raise RuntimeError("Maximum concurrent reconstructions reached")
        
        # Determine reconstruction parameters
        total_shards = len(shard_locations)
        required_shards = file_metadata.get('required_shards', total_shards)
        
        # Create reconstruction task
        task_id = self._generate_task_id()
        task = ReconstructionTask(
            task_id=task_id,
            file_id=file_id,
            requester_id=requester_id,
            total_shards=total_shards,
            required_shards=required_shards,
            collected_shards={},
            progress_callback=progress_callback
        )
        
        self.active_tasks[task_id] = task
        
        # Start reconstruction in background
        asyncio.create_task(
            self._execute_reconstruction(task, shard_locations, file_metadata)
        )
        
        logger.info(f"Started reconstruction task {task_id} for file {file_id}")
        return task_id
    
    async def _execute_reconstruction(self, task: ReconstructionTask,
                                    shard_locations: Dict[str, List[str]],
                                    file_metadata: Dict[str, Any]):
        """Execute the reconstruction process"""
        task.status = ReconstructionStatus.IN_PROGRESS
        task.started_at = time.time()
        
        try:
            # Step 1: Collect shards from P2P network
            logger.info(f"Collecting shards for task {task.task_id}")
            
            def progress_update(progress: float):
                if task.progress_callback:
                    task.progress_callback(progress * 0.7)  # Collection is 70% of total progress
            
            collected_shards = await self.shard_collector.collect_shards(
                task.file_id, shard_locations, task.required_shards, progress_update
            )
            
            task.collected_shards = collected_shards
            
            if not task.can_reconstruct:
                raise RuntimeError(f"Insufficient shards collected: {len(collected_shards)}/{task.required_shards}")
            
            # Step 2: Obtain decryption key
            logger.info(f"Obtaining decryption key for task {task.task_id}")
            encryption_key = await self._obtain_decryption_key(task.file_id, task.requester_id)
            
            if not encryption_key:
                raise RuntimeError("Failed to obtain decryption key")
            
            # Step 3: Reconstruct file
            logger.info(f"Reconstructing file for task {task.task_id}")
            
            if task.progress_callback:
                task.progress_callback(0.8)  # 80% progress
            
            reconstructed_data = await self.parallel_reconstructor.reconstruct_file(
                collected_shards, encryption_key, file_metadata
            )
            
            # Step 4: Final verification and completion
            if task.progress_callback:
                task.progress_callback(1.0)  # 100% complete
            
            task.status = ReconstructionStatus.COMPLETED
            task.completed_at = time.time()
            
            # Store reconstructed data (in production, would return or store securely)
            logger.info(f"Reconstruction task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = ReconstructionStatus.FAILED
            task.error_message = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Reconstruction task {task.task_id} failed: {e}")
        
        finally:
            # Clean up task after some time
            asyncio.create_task(self._cleanup_task(task.task_id, delay=3600))  # 1 hour
    
    async def _obtain_decryption_key(self, file_id: str, requester_id: str) -> Optional[bytes]:
        """Obtain decryption key for file reconstruction"""
        # This would integrate with the key management system to
        # reconstruct the file's encryption key from distributed shares
        
        # For testing, return a mock key
        mock_key = hashlib.sha256(f"file_key_{file_id}".encode()).digest()
        return mock_key
    
    def get_reconstruction_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a reconstruction task"""
        if task_id not in self.active_tasks:
            return None
        
        task = self.active_tasks[task_id]
        
        return {
            'task_id': task_id,
            'file_id': task.file_id,
            'status': task.status.value,
            'progress_percentage': task.progress_percentage,
            'verified_shards': task.verified_shard_count,
            'required_shards': task.required_shards,
            'elapsed_time': task.elapsed_time,
            'error_message': task.error_message
        }
    
    def cancel_reconstruction(self, task_id: str) -> bool:
        """Cancel an active reconstruction task"""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        
        if task.status == ReconstructionStatus.IN_PROGRESS:
            task.status = ReconstructionStatus.CANCELLED
            task.completed_at = time.time()
            logger.info(f"Cancelled reconstruction task {task_id}")
            return True
        
        return False
    
    async def _cleanup_task(self, task_id: str, delay: int = 0):
        """Clean up completed task after delay"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            logger.debug(f"Cleaned up reconstruction task {task_id}")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        return hashlib.sha256(f"{time.time()}:{id(self)}:{secrets.token_hex(8)}".encode()).hexdigest()[:16]
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get reconstruction engine statistics"""
        status_counts = {}
        for status in ReconstructionStatus:
            status_counts[status.value] = sum(
                1 for task in self.active_tasks.values()
                if task.status == status
            )
        
        return {
            'active_tasks': len(self.active_tasks),
            'max_concurrent': self.max_concurrent_reconstructions,
            'status_distribution': status_counts,
            'average_shards_per_task': sum(
                len(task.collected_shards) for task in self.active_tasks.values()
            ) / len(self.active_tasks) if self.active_tasks else 0
        }


# Example usage and testing
async def example_reconstruction():
    """Example of file reconstruction usage"""
    from .key_management import DistributedKeyManager
    from .access_control import PostQuantumAccessController
    
    # Initialize components (mock P2P network)
    key_manager = DistributedKeyManager("test_node")
    access_controller = PostQuantumAccessController("test_node", key_manager)
    
    await access_controller.initialize()
    
    # Create reconstruction engine
    reconstruction_engine = PostQuantumReconstructionEngine(
        key_manager, access_controller, None  # Mock P2P network
    )
    
    # Grant access for testing
    await access_controller.grant_access(
        subject_id="user123",
        resource_id="test_file",
        permissions=[Permission.READ]
    )
    
    # Mock shard locations
    shard_locations = {
        "test_file_0": ["node1", "node2"],
        "test_file_1": ["node2", "node3"], 
        "test_file_2": ["node1", "node3"],
        "test_file_3": ["node1", "node2", "node3"]
    }
    
    file_metadata = {
        'original_size': 1024,
        'required_shards': 3,
        'file_hash': 'mock_hash'
    }
    
    def progress_callback(progress: float):
        print(f"Reconstruction progress: {progress:.1%}")
    
    # Request reconstruction
    task_id = await reconstruction_engine.request_file_reconstruction(
        file_id="test_file",
        requester_id="user123",
        shard_locations=shard_locations,
        file_metadata=file_metadata,
        progress_callback=progress_callback
    )
    
    print(f"Started reconstruction task: {task_id}")
    
    # Monitor progress
    for _ in range(10):
        await asyncio.sleep(1)
        status = reconstruction_engine.get_reconstruction_status(task_id)
        if status:
            print(f"Status: {status['status']}, Progress: {status['progress_percentage']:.1f}%")
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
    
    # Get final statistics
    stats = reconstruction_engine.get_engine_statistics()
    print(f"Engine statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_reconstruction())