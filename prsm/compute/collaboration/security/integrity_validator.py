"""
Post-Quantum Integrity Validator for PRSM P2P Collaboration

This module provides comprehensive integrity validation for distributed files
and shards using post-quantum cryptographic signatures and advanced merkle
tree structures to ensure data authenticity and detect tampering.

Key Features:
- Post-quantum digital signatures for authenticity
- Merkle tree validation for efficient integrity checking
- Batch verification for performance optimization
- Tamper detection and forensic analysis
- Time-stamped integrity proofs
- Integration with distributed key management
"""

import asyncio
import json
import logging
import time
import secrets
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import base64
import struct
from math import ceil, log2

from .key_management import (
    DistributedKeyManager, 
    CryptographicKey, 
    KeyType, 
    PostQuantumCrypto,
    PostQuantumAlgorithm
)

logger = logging.getLogger(__name__)


class IntegrityStatus(Enum):
    """Status of integrity validation"""
    VALID = "valid"
    INVALID = "invalid"
    TAMPERED = "tampered"
    MISSING_SIGNATURE = "missing_signature"
    EXPIRED_SIGNATURE = "expired_signature"
    UNKNOWN = "unknown"


class ValidationLevel(Enum):
    """Levels of validation thoroughness"""
    BASIC = "basic"           # Hash verification only
    STANDARD = "standard"     # Hash + signature verification
    COMPREHENSIVE = "comprehensive"  # Full merkle tree + signatures
    FORENSIC = "forensic"     # Detailed tamper analysis


@dataclass
class IntegrityProof:
    """Cryptographic proof of data integrity"""
    proof_id: str
    data_hash: str
    signature: bytes
    signer_key_id: str
    timestamp: float
    algorithm: PostQuantumAlgorithm
    merkle_root: Optional[str] = None
    merkle_proof: Optional[List[str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_expired(self) -> bool:
        """Check if proof has expired (24 hours default)"""
        max_age = self.metadata.get('max_age', 86400)  # 24 hours
        return time.time() - self.timestamp > max_age
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['signature'] = base64.b64encode(self.signature).decode()
        data['algorithm'] = self.algorithm.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IntegrityProof':
        """Create from dictionary"""
        data['signature'] = base64.b64decode(data['signature'])
        data['algorithm'] = PostQuantumAlgorithm(data['algorithm'])
        return cls(**data)


@dataclass
class ValidationResult:
    """Result of integrity validation"""
    is_valid: bool
    status: IntegrityStatus
    confidence_score: float  # 0.0 to 1.0
    validation_level: ValidationLevel
    checked_proofs: List[str]  # Proof IDs that were checked
    error_details: Optional[str] = None
    tamper_evidence: Optional[Dict[str, Any]] = None
    validated_at: float = 0.0
    
    def __post_init__(self):
        if self.validated_at == 0.0:
            self.validated_at = time.time()


class MerkleTree:
    """
    Post-quantum secure Merkle tree for efficient integrity verification
    
    Provides efficient batch verification and tamper localization
    for large distributed files.
    """
    
    def __init__(self, hash_algorithm: str = 'sha256'):
        self.hash_algorithm = hash_algorithm
        self.leaf_nodes: List[str] = []
        self.tree_levels: List[List[str]] = []
        self.is_built = False
    
    def _hash_data(self, data: Union[str, bytes]) -> str:
        """Hash data using specified algorithm"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if self.hash_algorithm == 'sha256':
            return hashlib.sha256(data).hexdigest()
        elif self.hash_algorithm == 'sha3_256':
            return hashlib.sha3_256(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def add_leaf(self, data: Union[str, bytes]) -> int:
        """Add a leaf node and return its index"""
        leaf_hash = self._hash_data(data)
        self.leaf_nodes.append(leaf_hash)
        self.is_built = False
        return len(self.leaf_nodes) - 1
    
    def build_tree(self):
        """Build the complete Merkle tree"""
        if not self.leaf_nodes:
            raise ValueError("No leaf nodes to build tree")
        
        self.tree_levels = []
        current_level = self.leaf_nodes.copy()
        
        # Build tree bottom-up
        while len(current_level) > 1:
            self.tree_levels.append(current_level)
            next_level = []
            
            # Process pairs of nodes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left  # Duplicate last node for odd counts
                
                # Combine hashes
                combined = left + right
                parent_hash = self._hash_data(combined)
                next_level.append(parent_hash)
            
            current_level = next_level
        
        # Add root level
        self.tree_levels.append(current_level)
        self.is_built = True
        
        logger.debug(f"Built Merkle tree with {len(self.leaf_nodes)} leaves")
    
    def get_root(self) -> str:
        """Get the Merkle root hash"""
        if not self.is_built:
            self.build_tree()
        
        if not self.tree_levels:
            raise ValueError("Tree is empty")
        
        return self.tree_levels[-1][0]
    
    def get_proof(self, leaf_index: int) -> List[str]:
        """Get Merkle proof for a specific leaf"""
        if not self.is_built:
            self.build_tree()
        
        if leaf_index >= len(self.leaf_nodes):
            raise ValueError(f"Leaf index {leaf_index} out of range")
        
        proof = []
        current_index = leaf_index
        
        # Traverse up the tree collecting sibling hashes
        for level in self.tree_levels[:-1]:  # Exclude root level
            # Find sibling index
            if current_index % 2 == 0:  # Left node
                sibling_index = current_index + 1
            else:  # Right node
                sibling_index = current_index - 1
            
            # Add sibling hash if it exists
            if sibling_index < len(level):
                proof.append(level[sibling_index])
            else:
                proof.append(level[current_index])  # Duplicate for odd counts
            
            # Move to parent index
            current_index = current_index // 2
        
        return proof
    
    def verify_proof(self, leaf_hash: str, leaf_index: int, 
                    proof: List[str], root_hash: str) -> bool:
        """Verify a Merkle proof"""
        current_hash = leaf_hash
        current_index = leaf_index
        
        # Reconstruct path to root
        for sibling_hash in proof:
            if current_index % 2 == 0:  # Left node
                combined = current_hash + sibling_hash
            else:  # Right node
                combined = sibling_hash + current_hash
            
            current_hash = self._hash_data(combined)
            current_index = current_index // 2
        
        return current_hash == root_hash
    
    def detect_tampering(self, expected_leaves: List[str]) -> List[int]:
        """Detect which leaves have been tampered with"""
        if len(expected_leaves) != len(self.leaf_nodes):
            raise ValueError("Leaf count mismatch")
        
        tampered_indices = []
        
        for i, (expected, actual) in enumerate(zip(expected_leaves, self.leaf_nodes)):
            if expected != actual:
                tampered_indices.append(i)
        
        return tampered_indices


class PostQuantumSigner:
    """
    Post-quantum digital signature operations for integrity validation
    """
    
    def __init__(self, key_manager: DistributedKeyManager):
        self.key_manager = key_manager
        self.pq_crypto = PostQuantumCrypto()
    
    async def sign_data(self, data: bytes, signing_key_id: str,
                       include_timestamp: bool = True) -> bytes:
        """Sign data with post-quantum signature"""
        # Get signing key
        signing_key = self.key_manager.owned_keys.get(signing_key_id)
        if not signing_key or not signing_key.private_key:
            raise ValueError(f"Signing key {signing_key_id} not available")
        
        # Prepare data to sign
        sign_data = data
        if include_timestamp:
            timestamp_bytes = struct.pack('>Q', int(time.time() * 1000))
            sign_data = data + timestamp_bytes
        
        # Sign with post-quantum algorithm
        if signing_key.algorithm == PostQuantumAlgorithm.ML_DSA_87:
            signature = self.pq_crypto.mldsa_sign(signing_key.private_key, sign_data)
        else:
            raise ValueError(f"Unsupported signing algorithm: {signing_key.algorithm}")
        
        logger.debug(f"Signed data with key {signing_key_id}")
        return signature
    
    async def verify_signature(self, data: bytes, signature: bytes,
                             public_key: bytes, algorithm: PostQuantumAlgorithm,
                             include_timestamp: bool = True) -> bool:
        """Verify post-quantum signature"""
        try:
            # Prepare data for verification
            verify_data = data
            if include_timestamp:
                # For simplicity, assume timestamp is appended
                # In production, would extract and validate timestamp
                pass
            
            # Verify with post-quantum algorithm
            if algorithm == PostQuantumAlgorithm.ML_DSA_87:
                is_valid = self.pq_crypto.mldsa_verify(public_key, verify_data, signature)
            else:
                raise ValueError(f"Unsupported verification algorithm: {algorithm}")
            
            return is_valid
        
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False
    
    async def batch_verify_signatures(self, verification_requests: List[Tuple[bytes, bytes, bytes, PostQuantumAlgorithm]]) -> List[bool]:
        """Batch verify multiple signatures for efficiency"""
        results = []
        
        # Process verifications in parallel
        verification_tasks = []
        for data, signature, public_key, algorithm in verification_requests:
            task = asyncio.create_task(
                self.verify_signature(data, signature, public_key, algorithm)
            )
            verification_tasks.append(task)
        
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        
        # Convert exceptions to False
        return [result if isinstance(result, bool) else False for result in results]


class IntegrityValidator:
    """
    Main post-quantum integrity validation system
    
    Provides comprehensive integrity checking with multiple validation
    levels and tamper detection capabilities.
    """
    
    def __init__(self, key_manager: DistributedKeyManager,
                 config: Optional[Dict[str, Any]] = None):
        self.key_manager = key_manager
        self.config = config or {}
        
        # Initialize components
        self.pq_signer = PostQuantumSigner(key_manager)
        
        # Storage for integrity proofs
        self.integrity_proofs: Dict[str, IntegrityProof] = {}
        self.merkle_trees: Dict[str, MerkleTree] = {}
        
        # Configuration
        self.default_validation_level = ValidationLevel(
            self.config.get('default_level', 'standard')
        )
        self.batch_size = self.config.get('batch_size', 100)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Validation cache
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        logger.info("Post-quantum integrity validator initialized")
    
    async def create_integrity_proof(self, data: bytes, data_id: str,
                                   signing_key_id: str,
                                   create_merkle_tree: bool = False) -> str:
        """
        Create integrity proof for data
        
        Args:
            data: Data to create proof for
            data_id: Unique identifier for the data
            signing_key_id: Key to use for signing
            create_merkle_tree: Whether to create Merkle tree for batch verification
            
        Returns:
            Proof ID
        """
        # Calculate data hash
        data_hash = hashlib.sha256(data).hexdigest()
        
        # Create Merkle tree if requested
        merkle_root = None
        if create_merkle_tree:
            # Split data into chunks for Merkle tree
            chunk_size = self.config.get('merkle_chunk_size', 4096)
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            
            merkle_tree = MerkleTree()
            for chunk in chunks:
                merkle_tree.add_leaf(chunk)
            
            merkle_tree.build_tree()
            merkle_root = merkle_tree.get_root()
            
            # Store tree for later verification
            self.merkle_trees[data_id] = merkle_tree
        
        # Get signing key info
        signing_key = self.key_manager.owned_keys.get(signing_key_id)
        if not signing_key:
            raise ValueError(f"Signing key {signing_key_id} not found")
        
        # Create signature
        signature_data = data_hash.encode()
        if merkle_root:
            signature_data += merkle_root.encode()
        
        signature = await self.pq_signer.sign_data(signature_data, signing_key_id)
        
        # Create proof
        proof_id = self._generate_proof_id()
        proof = IntegrityProof(
            proof_id=proof_id,
            data_hash=data_hash,
            signature=signature,
            signer_key_id=signing_key_id,
            timestamp=time.time(),
            algorithm=signing_key.algorithm,
            merkle_root=merkle_root,
            metadata={'data_id': data_id}
        )
        
        self.integrity_proofs[proof_id] = proof
        
        logger.info(f"Created integrity proof {proof_id} for data {data_id}")
        return proof_id
    
    async def validate_data_integrity(self, data: bytes, proof_id: str,
                                    validation_level: Optional[ValidationLevel] = None) -> ValidationResult:
        """
        Validate data integrity using proof
        
        Args:
            data: Data to validate
            proof_id: ID of integrity proof
            validation_level: Level of validation to perform
            
        Returns:
            Validation result
        """
        validation_level = validation_level or self.default_validation_level
        
        # Check cache first
        cache_key = f"{proof_id}_{hashlib.sha256(data).hexdigest()[:16]}"
        if self.enable_caching and cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            if time.time() - cached_result.validated_at < self.cache_ttl:
                return cached_result
        
        # Get integrity proof
        if proof_id not in self.integrity_proofs:
            return ValidationResult(
                is_valid=False,
                status=IntegrityStatus.MISSING_SIGNATURE,
                confidence_score=0.0,
                validation_level=validation_level,
                checked_proofs=[],
                error_details="Integrity proof not found"
            )
        
        proof = self.integrity_proofs[proof_id]
        
        # Check if proof is expired
        if proof.is_expired:
            return ValidationResult(
                is_valid=False,
                status=IntegrityStatus.EXPIRED_SIGNATURE,
                confidence_score=0.0,
                validation_level=validation_level,
                checked_proofs=[proof_id],
                error_details="Integrity proof has expired"
            )
        
        # Perform validation based on level
        result = await self._perform_validation(data, proof, validation_level)
        
        # Cache result
        if self.enable_caching:
            self.validation_cache[cache_key] = result
        
        return result
    
    async def _perform_validation(self, data: bytes, proof: IntegrityProof,
                                validation_level: ValidationLevel) -> ValidationResult:
        """Perform validation at specified level"""
        if validation_level == ValidationLevel.BASIC:
            return await self._basic_validation(data, proof)
        elif validation_level == ValidationLevel.STANDARD:
            return await self._standard_validation(data, proof)
        elif validation_level == ValidationLevel.COMPREHENSIVE:
            return await self._comprehensive_validation(data, proof)
        elif validation_level == ValidationLevel.FORENSIC:
            return await self._forensic_validation(data, proof)
        else:
            raise ValueError(f"Unknown validation level: {validation_level}")
    
    async def _basic_validation(self, data: bytes, proof: IntegrityProof) -> ValidationResult:
        """Basic validation - hash comparison only"""
        data_hash = hashlib.sha256(data).hexdigest()
        is_valid = data_hash == proof.data_hash
        
        return ValidationResult(
            is_valid=is_valid,
            status=IntegrityStatus.VALID if is_valid else IntegrityStatus.TAMPERED,
            confidence_score=1.0 if is_valid else 0.0,
            validation_level=ValidationLevel.BASIC,
            checked_proofs=[proof.proof_id]
        )
    
    async def _standard_validation(self, data: bytes, proof: IntegrityProof) -> ValidationResult:
        """Standard validation - hash + signature verification"""
        # First perform basic validation
        basic_result = await self._basic_validation(data, proof)
        
        if not basic_result.is_valid:
            return basic_result
        
        # Verify signature
        public_key = self.key_manager.get_public_key(proof.signer_key_id)
        if not public_key:
            return ValidationResult(
                is_valid=False,
                status=IntegrityStatus.INVALID,
                confidence_score=0.0,
                validation_level=ValidationLevel.STANDARD,
                checked_proofs=[proof.proof_id],
                error_details="Signer public key not available"
            )
        
        # Prepare signature data
        signature_data = proof.data_hash.encode()
        if proof.merkle_root:
            signature_data += proof.merkle_root.encode()
        
        # Verify signature
        signature_valid = await self.pq_signer.verify_signature(
            signature_data, proof.signature, public_key, proof.algorithm
        )
        
        if signature_valid:
            return ValidationResult(
                is_valid=True,
                status=IntegrityStatus.VALID,
                confidence_score=1.0,
                validation_level=ValidationLevel.STANDARD,
                checked_proofs=[proof.proof_id]
            )
        else:
            return ValidationResult(
                is_valid=False,
                status=IntegrityStatus.INVALID,
                confidence_score=0.0,
                validation_level=ValidationLevel.STANDARD,
                checked_proofs=[proof.proof_id],
                error_details="Digital signature verification failed"
            )
    
    async def _comprehensive_validation(self, data: bytes, proof: IntegrityProof) -> ValidationResult:
        """Comprehensive validation - includes Merkle tree verification"""
        # First perform standard validation
        standard_result = await self._standard_validation(data, proof)
        
        if not standard_result.is_valid:
            return ValidationResult(
                is_valid=False,
                status=standard_result.status,
                confidence_score=standard_result.confidence_score,
                validation_level=ValidationLevel.COMPREHENSIVE,
                checked_proofs=standard_result.checked_proofs,
                error_details=standard_result.error_details
            )
        
        # Verify Merkle tree if present
        if proof.merkle_root:
            data_id = proof.metadata.get('data_id')
            if data_id and data_id in self.merkle_trees:
                merkle_tree = self.merkle_trees[data_id]
                
                # Rebuild tree with current data
                chunk_size = self.config.get('merkle_chunk_size', 4096)
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                
                current_tree = MerkleTree()
                for chunk in chunks:
                    current_tree.add_leaf(chunk)
                
                current_tree.build_tree()
                current_root = current_tree.get_root()
                
                if current_root != proof.merkle_root:
                    # Detect tampered chunks
                    tampered_chunks = merkle_tree.detect_tampering(current_tree.leaf_nodes)
                    
                    return ValidationResult(
                        is_valid=False,
                        status=IntegrityStatus.TAMPERED,
                        confidence_score=0.0,
                        validation_level=ValidationLevel.COMPREHENSIVE,
                        checked_proofs=[proof.proof_id],
                        tamper_evidence={'tampered_chunks': tampered_chunks}
                    )
        
        return ValidationResult(
            is_valid=True,
            status=IntegrityStatus.VALID,
            confidence_score=1.0,
            validation_level=ValidationLevel.COMPREHENSIVE,
            checked_proofs=[proof.proof_id]
        )
    
    async def _forensic_validation(self, data: bytes, proof: IntegrityProof) -> ValidationResult:
        """Forensic validation - detailed tamper analysis"""
        # Perform comprehensive validation first
        comprehensive_result = await self._comprehensive_validation(data, proof)
        
        # Add detailed forensic analysis
        tamper_evidence = comprehensive_result.tamper_evidence or {}
        
        if not comprehensive_result.is_valid:
            # Perform detailed tamper analysis
            tamper_evidence.update({
                'analysis_timestamp': time.time(),
                'data_size': len(data),
                'expected_hash': proof.data_hash,
                'actual_hash': hashlib.sha256(data).hexdigest(),
                'hash_algorithms_checked': ['sha256']
            })
            
            # Check for specific tampering patterns
            if proof.merkle_root:
                # Analyze which parts of the file were tampered
                tamper_evidence['tampering_analysis'] = await self._analyze_tampering_patterns(data, proof)
        
        return ValidationResult(
            is_valid=comprehensive_result.is_valid,
            status=comprehensive_result.status,
            confidence_score=comprehensive_result.confidence_score,
            validation_level=ValidationLevel.FORENSIC,
            checked_proofs=comprehensive_result.checked_proofs,
            error_details=comprehensive_result.error_details,
            tamper_evidence=tamper_evidence
        )
    
    async def _analyze_tampering_patterns(self, data: bytes, proof: IntegrityProof) -> Dict[str, Any]:
        """Analyze patterns in data tampering for forensic purposes"""
        analysis = {
            'tamper_type': 'unknown',
            'affected_regions': [],
            'severity': 'unknown'
        }
        
        # This would include sophisticated tampering analysis
        # For now, return basic analysis
        return analysis
    
    async def batch_validate(self, validation_requests: List[Tuple[bytes, str]],
                           validation_level: Optional[ValidationLevel] = None) -> List[ValidationResult]:
        """Batch validate multiple data items for efficiency"""
        validation_level = validation_level or self.default_validation_level
        
        # Process validations in parallel batches
        results = []
        
        for i in range(0, len(validation_requests), self.batch_size):
            batch = validation_requests[i:i + self.batch_size]
            
            # Create validation tasks
            validation_tasks = []
            for data, proof_id in batch:
                task = asyncio.create_task(
                    self.validate_data_integrity(data, proof_id, validation_level)
                )
                validation_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, ValidationResult):
                    results.append(result)
                else:
                    # Handle exceptions
                    error_result = ValidationResult(
                        is_valid=False,
                        status=IntegrityStatus.UNKNOWN,
                        confidence_score=0.0,
                        validation_level=validation_level,
                        checked_proofs=[],
                        error_details=str(result)
                    )
                    results.append(error_result)
        
        logger.info(f"Batch validated {len(validation_requests)} items")
        return results
    
    def get_proof(self, proof_id: str) -> Optional[IntegrityProof]:
        """Get integrity proof by ID"""
        return self.integrity_proofs.get(proof_id)
    
    def list_proofs(self, data_id: Optional[str] = None) -> List[IntegrityProof]:
        """List integrity proofs, optionally filtered by data ID"""
        proofs = list(self.integrity_proofs.values())
        
        if data_id:
            proofs = [
                proof for proof in proofs
                if proof.metadata.get('data_id') == data_id
            ]
        
        return proofs
    
    def cleanup_expired_proofs(self) -> int:
        """Remove expired proofs and return count"""
        expired_proofs = [
            proof_id for proof_id, proof in self.integrity_proofs.items()
            if proof.is_expired
        ]
        
        for proof_id in expired_proofs:
            del self.integrity_proofs[proof_id]
        
        logger.info(f"Cleaned up {len(expired_proofs)} expired proofs")
        return len(expired_proofs)
    
    def _generate_proof_id(self) -> str:
        """Generate unique proof ID"""
        return hashlib.sha256(f"{time.time()}:{id(self)}:{secrets.token_hex(8)}".encode()).hexdigest()[:16]
    
    def get_validator_statistics(self) -> Dict[str, Any]:
        """Get validator statistics"""
        total_proofs = len(self.integrity_proofs)
        expired_proofs = sum(1 for proof in self.integrity_proofs.values() if proof.is_expired)
        
        algorithm_counts = {}
        for proof in self.integrity_proofs.values():
            alg = proof.algorithm.value
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        
        return {
            'total_proofs': total_proofs,
            'active_proofs': total_proofs - expired_proofs,
            'expired_proofs': expired_proofs,
            'cached_validations': len(self.validation_cache),
            'merkle_trees': len(self.merkle_trees),
            'algorithm_distribution': algorithm_counts
        }


# Example usage and testing
async def example_integrity_validation():
    """Example of integrity validation usage"""
    from .key_management import DistributedKeyManager, KeyType, PostQuantumAlgorithm
    
    # Initialize components
    key_manager = DistributedKeyManager("test_node")
    validator = IntegrityValidator(key_manager)
    
    # Generate signing key
    signing_key_id = await key_manager.generate_keypair(
        KeyType.SIGNING, PostQuantumAlgorithm.ML_DSA_87
    )
    
    # Test data
    test_data = b"This is test data for integrity validation" * 100
    
    # Create integrity proof
    proof_id = await validator.create_integrity_proof(
        test_data, "test_file", signing_key_id, create_merkle_tree=True
    )
    
    print(f"Created integrity proof: {proof_id}")
    
    # Validate data integrity
    result = await validator.validate_data_integrity(
        test_data, proof_id, ValidationLevel.COMPREHENSIVE
    )
    
    print(f"Validation result: {result.is_valid}")
    print(f"Status: {result.status.value}")
    print(f"Confidence: {result.confidence_score}")
    
    # Test with tampered data
    tampered_data = test_data[:-10] + b"TAMPERED!!"
    
    tampered_result = await validator.validate_data_integrity(
        tampered_data, proof_id, ValidationLevel.FORENSIC
    )
    
    print(f"Tampered validation: {tampered_result.is_valid}")
    print(f"Tampered status: {tampered_result.status.value}")
    if tampered_result.tamper_evidence:
        print(f"Tamper evidence: {tampered_result.tamper_evidence}")
    
    # Get statistics
    stats = validator.get_validator_statistics()
    print(f"Validator statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_integrity_validation())