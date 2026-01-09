"""
PRSM Content Verification and Provenance System

Advanced content verification, integrity checking, and provenance tracking
for scientific content stored in IPFS. Ensures content authenticity and
maintains complete audit trails for research integrity.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import base64

# Cryptographic signing (optional dependency)
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.exceptions import InvalidSignature
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

from .ipfs_client import IPFSClient
from .content_addressing import AddressedContent, ContentProvenance

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Status of content verification"""
    VERIFIED = "verified"
    FAILED = "failed"
    PENDING = "pending"
    NOT_VERIFIED = "not_verified"
    CORRUPTED = "corrupted"


class ProvenanceEventType(str, Enum):
    """Types of provenance events"""
    CREATED = "created"
    MODIFIED = "modified"
    REVIEWED = "reviewed"
    CITED = "cited"
    REPLICATED = "replicated"
    FLAGGED = "flagged"
    VERIFIED = "verified"


@dataclass
class VerificationResult:
    """Result of content verification"""
    cid: str
    status: VerificationStatus
    checksum_match: bool
    signature_valid: bool
    ipfs_accessible: bool
    verification_time: float
    error_message: Optional[str] = None
    verifier_id: Optional[str] = None
    verified_at: Optional[datetime] = None


@dataclass
class ProvenanceEvent:
    """A single event in the content provenance chain"""
    event_id: str
    event_type: ProvenanceEventType
    actor_id: str
    actor_name: str
    timestamp: datetime
    description: str
    metadata: Dict[str, Any]
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ProvenanceChain:
    """Complete provenance chain for content"""
    content_cid: str
    events: List[ProvenanceEvent]
    created_at: datetime
    updated_at: datetime
    chain_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'content_cid': self.content_cid,
            'events': [event.to_dict() for event in self.events],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'chain_hash': self.chain_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceChain':
        """Create from dictionary"""
        return cls(
            content_cid=data['content_cid'],
            events=[ProvenanceEvent.from_dict(e) for e in data['events']],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            chain_hash=data['chain_hash']
        )


@dataclass
class IntegrityCheck:
    """Content integrity verification"""
    content_cid: str
    original_checksum: str
    current_checksum: str
    file_size: int
    check_timestamp: datetime
    integrity_valid: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['check_timestamp'] = self.check_timestamp.isoformat()
        return data


class ContentVerificationSystem:
    """
    Advanced content verification and provenance tracking system
    
    Features:
    - Cryptographic content verification
    - Digital signature validation
    - Complete provenance chain tracking
    - Integrity monitoring and alerts
    - Automated verification workflows
    - Audit trail generation
    """
    
    def __init__(self, ipfs_client: IPFSClient):
        self.ipfs_client = ipfs_client
        self.provenance_chains: Dict[str, ProvenanceChain] = {}
        self.verification_cache: Dict[str, VerificationResult] = {}
        
        # Cryptographic keys (would be loaded from secure storage in production)
        self.private_key = None
        self.public_key = None
        
        if HAS_CRYPTO:
            self._initialize_crypto()
        
        # Statistics
        self.stats = {
            'verifications_performed': 0,
            'signatures_verified': 0,
            'integrity_checks': 0,
            'provenance_events_recorded': 0,
            'failed_verifications': 0
        }
    
    def _initialize_crypto(self):
        """Initialize cryptographic components"""
        try:
            # Generate keys for demonstration (in production, load from secure storage)
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            logger.info("✅ Cryptographic verification system initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize crypto: {e}")
    
    async def verify_content(self, 
                           cid: str,
                           expected_checksum: str = None,
                           verifier_id: str = None) -> VerificationResult:
        """
        Perform comprehensive content verification
        
        Args:
            cid: Content Identifier to verify
            expected_checksum: Expected SHA256 checksum
            verifier_id: ID of the entity performing verification
            
        Returns:
            VerificationResult with detailed verification status
        """
        start_time = time.time()
        
        try:
            # Check if already verified recently
            if cid in self.verification_cache:
                cached_result = self.verification_cache[cid]
                if (datetime.now() - cached_result.verified_at).seconds < 3600:  # 1 hour cache
                    logger.debug(f"Using cached verification for {cid}")
                    return cached_result
            
            # Step 1: Verify IPFS accessibility
            try:
                content_bytes = await self.ipfs_client.get_content(cid)
                ipfs_accessible = True
            except Exception as e:
                logger.error(f"IPFS access failed for {cid}: {e}")
                return VerificationResult(
                    cid=cid,
                    status=VerificationStatus.FAILED,
                    checksum_match=False,
                    signature_valid=False,
                    ipfs_accessible=False,
                    verification_time=time.time() - start_time,
                    error_message=f"IPFS access failed: {e}",
                    verifier_id=verifier_id,
                    verified_at=datetime.now()
                )
            
            # Step 2: Verify content checksum
            current_checksum = hashlib.sha256(content_bytes).hexdigest()
            checksum_match = True
            
            if expected_checksum:
                checksum_match = (current_checksum == expected_checksum)
                if not checksum_match:
                    logger.warning(f"Checksum mismatch for {cid}: expected {expected_checksum}, got {current_checksum}")
            
            # Step 3: Verify digital signature (if available)
            signature_valid = await self._verify_signature(cid, content_bytes)
            
            # Step 4: Determine overall status
            if ipfs_accessible and checksum_match and signature_valid:
                status = VerificationStatus.VERIFIED
            elif not checksum_match:
                status = VerificationStatus.CORRUPTED
            else:
                status = VerificationStatus.FAILED
            
            verification_time = time.time() - start_time
            
            result = VerificationResult(
                cid=cid,
                status=status,
                checksum_match=checksum_match,
                signature_valid=signature_valid,
                ipfs_accessible=ipfs_accessible,
                verification_time=verification_time,
                verifier_id=verifier_id,
                verified_at=datetime.now()
            )
            
            # Cache result
            self.verification_cache[cid] = result
            
            # Record provenance event
            await self._record_provenance_event(
                cid=cid,
                event_type=ProvenanceEventType.VERIFIED,
                actor_id=verifier_id or "system",
                actor_name=verifier_id or "Automated Verification System",
                description=f"Content verification: {status.value}",
                metadata={
                    'verification_result': result.__dict__,
                    'checksum': current_checksum
                }
            )
            
            # Update statistics
            self.stats['verifications_performed'] += 1
            if status != VerificationStatus.VERIFIED:
                self.stats['failed_verifications'] += 1
            
            logger.info(f"✅ Verified content {cid}: {status.value} ({verification_time:.3f}s)")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Verification failed for {cid}: {e}")
            self.stats['failed_verifications'] += 1
            
            return VerificationResult(
                cid=cid,
                status=VerificationStatus.FAILED,
                checksum_match=False,
                signature_valid=False,
                ipfs_accessible=False,
                verification_time=time.time() - start_time,
                error_message=str(e),
                verifier_id=verifier_id,
                verified_at=datetime.now()
            )
    
    async def sign_content(self, 
                          cid: str,
                          signer_id: str,
                          signer_name: str) -> str:
        """
        Digitally sign content for authenticity verification
        
        Args:
            cid: Content Identifier to sign
            signer_id: ID of the signer
            signer_name: Name of the signer
            
        Returns:
            Base64-encoded signature
        """
        if not HAS_CRYPTO or not self.private_key:
            raise ValueError("Cryptographic signing not available")
        
        try:
            # Get content to sign
            content_bytes = await self.ipfs_client.get_content(cid)
            
            # Create signature
            signature = self.private_key.sign(
                content_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            # Record provenance event
            await self._record_provenance_event(
                cid=cid,
                event_type=ProvenanceEventType.VERIFIED,
                actor_id=signer_id,
                actor_name=signer_name,
                description="Content digitally signed",
                metadata={
                    'signature': signature_b64,
                    'signature_algorithm': 'RSA-PSS-SHA256'
                }
            )
            
            logger.info(f"✅ Signed content {cid} by {signer_name}")
            
            return signature_b64
        
        except Exception as e:
            logger.error(f"❌ Failed to sign content {cid}: {e}")
            raise
    
    async def _verify_signature(self, cid: str, content_bytes: bytes) -> bool:
        """Verify digital signature for content"""
        if not HAS_CRYPTO or not self.public_key:
            return True  # Skip signature verification if crypto not available
        
        try:
            # Look for signature in provenance chain
            if cid not in self.provenance_chains:
                return True  # No signature to verify
            
            chain = self.provenance_chains[cid]
            signature_events = [
                e for e in chain.events 
                if e.event_type == ProvenanceEventType.VERIFIED and 
                'signature' in e.metadata
            ]
            
            if not signature_events:
                return True  # No signature to verify
            
            # Verify most recent signature
            latest_signature_event = max(signature_events, key=lambda e: e.timestamp)
            signature_b64 = latest_signature_event.metadata['signature']
            signature = base64.b64decode(signature_b64)
            
            # Verify signature
            self.public_key.verify(
                signature,
                content_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            self.stats['signatures_verified'] += 1
            return True
        
        except InvalidSignature:
            logger.warning(f"Invalid signature for content {cid}")
            return False
        except Exception as e:
            logger.error(f"Signature verification error for {cid}: {e}")
            return False
    
    async def create_provenance_chain(self, 
                                    content_cid: str,
                                    creator_id: str,
                                    creator_name: str,
                                    creation_metadata: Dict[str, Any] = None) -> ProvenanceChain:
        """
        Create initial provenance chain for new content
        
        Args:
            content_cid: Content Identifier
            creator_id: Creator's unique identifier
            creator_name: Creator's name
            creation_metadata: Additional creation metadata
            
        Returns:
            New ProvenanceChain object
        """
        now = datetime.now()
        
        # Create initial creation event
        creation_event = ProvenanceEvent(
            event_id=f"create_{content_cid}_{int(now.timestamp())}",
            event_type=ProvenanceEventType.CREATED,
            actor_id=creator_id,
            actor_name=creator_name,
            timestamp=now,
            description="Content created",
            metadata=creation_metadata or {},
        )
        
        # Create provenance chain
        chain = ProvenanceChain(
            content_cid=content_cid,
            events=[creation_event],
            created_at=now,
            updated_at=now,
            chain_hash=self._compute_chain_hash([creation_event])
        )
        
        # Store chain
        self.provenance_chains[content_cid] = chain
        
        # Store chain in IPFS
        await self._store_provenance_chain(chain)
        
        self.stats['provenance_events_recorded'] += 1
        
        logger.info(f"✅ Created provenance chain for {content_cid}")
        
        return chain
    
    async def _record_provenance_event(self,
                                     cid: str,
                                     event_type: ProvenanceEventType,
                                     actor_id: str,
                                     actor_name: str,
                                     description: str,
                                     metadata: Dict[str, Any] = None) -> ProvenanceEvent:
        """Record a new event in the provenance chain"""
        now = datetime.now()
        
        event = ProvenanceEvent(
            event_id=f"{event_type.value}_{cid}_{int(now.timestamp())}",
            event_type=event_type,
            actor_id=actor_id,
            actor_name=actor_name,
            timestamp=now,
            description=description,
            metadata=metadata or {}
        )
        
        # Get or create provenance chain
        if cid not in self.provenance_chains:
            # Create minimal chain if it doesn't exist
            self.provenance_chains[cid] = ProvenanceChain(
                content_cid=cid,
                events=[],
                created_at=now,
                updated_at=now,
                chain_hash=""
            )
        
        chain = self.provenance_chains[cid]
        chain.events.append(event)
        chain.updated_at = now
        chain.chain_hash = self._compute_chain_hash(chain.events)
        
        # Store updated chain in IPFS
        await self._store_provenance_chain(chain)
        
        self.stats['provenance_events_recorded'] += 1
        
        logger.debug(f"Recorded provenance event for {cid}: {event_type.value}")
        
        return event
    
    async def get_provenance_chain(self, cid: str) -> Optional[ProvenanceChain]:
        """Get complete provenance chain for content"""
        if cid in self.provenance_chains:
            return self.provenance_chains[cid]
        
        # Try to load from IPFS
        try:
            chain = await self._load_provenance_chain(cid)
            if chain:
                self.provenance_chains[cid] = chain
                return chain
        except Exception as e:
            logger.debug(f"Could not load provenance chain for {cid}: {e}")
        
        return None
    
    async def verify_provenance_integrity(self, cid: str) -> bool:
        """Verify the integrity of a provenance chain"""
        chain = await self.get_provenance_chain(cid)
        if not chain:
            return False
        
        # Recompute chain hash
        computed_hash = self._compute_chain_hash(chain.events)
        
        return computed_hash == chain.chain_hash
    
    def _compute_chain_hash(self, events: List[ProvenanceEvent]) -> str:
        """Compute hash of provenance chain for integrity verification"""
        chain_data = json.dumps([event.to_dict() for event in events], sort_keys=True, default=str)
        return hashlib.sha256(chain_data.encode('utf-8')).hexdigest()
    
    async def _store_provenance_chain(self, chain: ProvenanceChain):
        """Store provenance chain in IPFS"""
        try:
            chain_json = json.dumps(chain.to_dict(), indent=2, ensure_ascii=False, default=str)
            
            await self.ipfs_client.add_content(
                content=chain_json,
                filename=f"provenance_{chain.content_cid}.json",
                metadata={
                    'type': 'prsm_provenance_chain',
                    'content_cid': chain.content_cid,
                    'event_count': len(chain.events)
                }
            )
        except Exception as e:
            logger.error(f"Failed to store provenance chain for {chain.content_cid}: {e}")
    
    async def _load_provenance_chain(self, cid: str) -> Optional[ProvenanceChain]:
        """Load provenance chain from IPFS"""
        # This would require a registry or search mechanism in a full implementation
        # For now, this is a placeholder
        return None
    
    async def check_content_integrity(self, cid: str, 
                                    original_checksum: str) -> IntegrityCheck:
        """Perform integrity check on content"""
        try:
            content_bytes = await self.ipfs_client.get_content(cid)
            current_checksum = hashlib.sha256(content_bytes).hexdigest()
            
            integrity_check = IntegrityCheck(
                content_cid=cid,
                original_checksum=original_checksum,
                current_checksum=current_checksum,
                file_size=len(content_bytes),
                check_timestamp=datetime.now(),
                integrity_valid=(current_checksum == original_checksum)
            )
            
            self.stats['integrity_checks'] += 1
            
            if not integrity_check.integrity_valid:
                logger.warning(f"Integrity check failed for {cid}: checksum mismatch")
                
                # Record integrity failure event
                await self._record_provenance_event(
                    cid=cid,
                    event_type=ProvenanceEventType.FLAGGED,
                    actor_id="system",
                    actor_name="Integrity Check System",
                    description="Content integrity check failed",
                    metadata=integrity_check.to_dict()
                )
            
            return integrity_check
        
        except Exception as e:
            logger.error(f"Integrity check failed for {cid}: {e}")
            raise
    
    async def batch_verify_content(self, 
                                 cids: List[str],
                                 max_concurrent: int = 10) -> List[VerificationResult]:
        """Verify multiple content items concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def verify_single(cid):
            async with semaphore:
                return await self.verify_content(cid)
        
        tasks = [verify_single(cid) for cid in cids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        verified_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                verified_results.append(VerificationResult(
                    cid=cids[i],
                    status=VerificationStatus.FAILED,
                    checksum_match=False,
                    signature_valid=False,
                    ipfs_accessible=False,
                    verification_time=0.0,
                    error_message=str(result),
                    verified_at=datetime.now()
                ))
            else:
                verified_results.append(result)
        
        return verified_results
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification system statistics"""
        return {
            'verification_stats': self.stats.copy(),
            'cached_verifications': len(self.verification_cache),
            'provenance_chains': len(self.provenance_chains),
            'crypto_available': HAS_CRYPTO,
            'total_provenance_events': sum(
                len(chain.events) for chain in self.provenance_chains.values()
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on verification system"""
        try:
            # Test IPFS connectivity
            ipfs_health = await self.ipfs_client.health_check()
            
            # Test cryptographic functions
            crypto_working = False
            if HAS_CRYPTO and self.private_key:
                try:
                    test_data = b"health check test"
                    signature = self.private_key.sign(
                        test_data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    
                    # Verify signature
                    self.public_key.verify(
                        signature,
                        test_data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    crypto_working = True
                except Exception as e:
                    logger.error(f"Crypto health check failed: {e}")
            
            return {
                'healthy': ipfs_health['healthy'] and crypto_working,
                'ipfs_health': ipfs_health,
                'crypto_working': crypto_working,
                'stats': self.get_verification_stats()
            }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'stats': self.get_verification_stats()
            }


# Utility functions

def create_verification_system(ipfs_client: IPFSClient) -> ContentVerificationSystem:
    """Create a new content verification system"""
    return ContentVerificationSystem(ipfs_client)


async def verify_content_batch(verification_system: ContentVerificationSystem,
                             cids: List[str]) -> List[VerificationResult]:
    """Batch verify multiple content items"""
    return await verification_system.batch_verify_content(cids)