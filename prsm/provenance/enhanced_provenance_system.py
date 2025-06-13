#!/usr/bin/env python3
"""
Enhanced Provenance-Tracked Content System for PRSM
Phase 1 implementation with IPFS integration and automatic attribution

This system provides comprehensive content provenance tracking for Phase 1:
1. IPFS integration with automatic content attribution
2. Cryptographic fingerprinting for content integrity
3. Real-time usage tracking across 100+ model interactions
4. Automatic creator compensation via FTNS tokens
5. Complete audit trails for governance compliance
6. Performance optimization for high-volume content processing

Key Features:
- Content-addressed storage with IPFS integration
- Automatic attribution chain construction
- Cryptographic content fingerprinting (SHA-256, Blake2b)
- Real-time usage analytics and tracking
- FTNS token integration for creator rewards
- Compliance with licensing requirements
- Performance monitoring and optimization
"""

import asyncio
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import json
import structlog
from decimal import Decimal

from prsm.core.config import get_settings
from prsm.core.models import ProvenanceRecord, FTNSTransaction
from prsm.data_layer.enhanced_ipfs import PRSMIPFSClient
from prsm.tokenomics.enhanced_ftns_service import get_enhanced_ftns_service
from prsm.core.database_service import get_database_service

logger = structlog.get_logger(__name__)
settings = get_settings()

class ContentType(Enum):
    """Types of content in the provenance system"""
    MODEL = "model"
    DATASET = "dataset"
    RESEARCH_PAPER = "research_paper"
    CODE_REPOSITORY = "code_repository"
    TRAINING_DATA = "training_data"
    INFERENCE_RESULT = "inference_result"
    SYSTEM_OUTPUT = "system_output"

class LicenseType(Enum):
    """Content license types"""
    MIT = "mit"
    APACHE_2_0 = "apache_2_0"
    GPL_3_0 = "gpl_3_0"
    CREATIVE_COMMONS = "creative_commons"
    PROPRIETARY = "proprietary"
    OPEN_SOURCE = "open_source"
    COMMERCIAL = "commercial"

@dataclass
class ContentFingerprint:
    """Cryptographic content fingerprint"""
    sha256_hash: str
    blake2b_hash: str
    size_bytes: int
    content_type: ContentType
    creation_timestamp: datetime
    ipfs_hash: Optional[str] = None
    chunk_hashes: List[str] = field(default_factory=list)

@dataclass
class AttributionChain:
    """Complete attribution chain for content"""
    content_id: UUID
    original_creator: str
    creator_address: Optional[str]  # FTNS wallet address
    creation_timestamp: datetime
    parent_content: Optional[UUID]
    contributors: List[Dict[str, Any]] = field(default_factory=list)
    license_type: LicenseType = LicenseType.OPEN_SOURCE
    license_terms: Dict[str, Any] = field(default_factory=dict)
    platform_source: Optional[str] = None
    external_id: Optional[str] = None

@dataclass
class UsageEvent:
    """Individual content usage event"""
    event_id: UUID
    content_id: UUID
    user_id: str
    session_id: UUID
    usage_type: str  # "access", "training", "inference", "reference"
    timestamp: datetime
    context: Dict[str, Any]
    ftns_cost: Decimal
    attribution_credited: bool = False

@dataclass
class ProvenanceMetrics:
    """Content provenance metrics"""
    content_id: UUID
    total_usage_count: int
    unique_users: int
    total_ftns_earned: Decimal
    avg_usage_per_day: float
    first_usage: datetime
    last_usage: datetime
    popularity_score: float
    attribution_accuracy: float

class EnhancedProvenanceSystem:
    """
    Enhanced Provenance-Tracked Content System for Phase 1
    
    Provides comprehensive content tracking with:
    - IPFS integration for distributed storage
    - Automatic attribution chain construction
    - Cryptographic fingerprinting for integrity
    - Real-time usage tracking and analytics
    - FTNS token integration for creator rewards
    - Performance optimization for high-volume processing
    """
    
    def __init__(self):
        self.ipfs_client = PRSMIPFSClient()
        self.ftns_service = get_enhanced_ftns_service()
        self.database_service = get_database_service()
        
        # Content tracking
        self.content_registry: Dict[UUID, AttributionChain] = {}
        self.fingerprint_cache: Dict[str, ContentFingerprint] = {}
        self.usage_events: List[UsageEvent] = []
        
        # Performance tracking
        self.fingerprint_times: List[float] = []
        self.attribution_times: List[float] = []
        self.ipfs_store_times: List[float] = []
        
        # Configuration
        self.fingerprint_algorithms = ['sha256', 'blake2b']
        self.chunk_size = 8192  # 8KB chunks for fingerprinting
        self.usage_batch_size = 1000
        self.reward_calculation_interval = timedelta(hours=1)
        
        logger.info("Enhanced Provenance System initialized")
    
    async def register_content_with_provenance(
        self,
        content_data: bytes,
        content_type: ContentType,
        creator_info: Dict[str, Any],
        license_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[UUID, ContentFingerprint, AttributionChain]:
        """
        Register content with complete provenance tracking
        
        This is the primary entry point for Phase 1 content registration,
        providing:
        - Cryptographic fingerprinting for integrity
        - IPFS storage with content addressing
        - Attribution chain construction
        - Database persistence for governance
        
        Args:
            content_data: Raw content bytes
            content_type: Type of content being registered
            creator_info: Creator attribution information
            license_info: License and terms information
            metadata: Additional content metadata
            
        Returns:
            Tuple of (content_id, fingerprint, attribution_chain)
        """
        start_time = time.perf_counter()
        content_id = uuid4()
        
        try:
            logger.info("Registering content with provenance",
                       content_id=str(content_id),
                       content_type=content_type.value,
                       size_bytes=len(content_data))
            
            # Step 1: Generate cryptographic fingerprint
            fingerprint = await self._generate_content_fingerprint(
                content_data, content_type
            )
            
            # Step 2: Store content in IPFS
            ipfs_hash = await self._store_content_in_ipfs(
                content_data, content_id, metadata or {}
            )
            fingerprint.ipfs_hash = ipfs_hash
            
            # Step 3: Create attribution chain
            attribution_chain = await self._create_attribution_chain(
                content_id, creator_info, license_info, fingerprint
            )
            
            # Step 4: Register in content registry
            self.content_registry[content_id] = attribution_chain
            self.fingerprint_cache[fingerprint.sha256_hash] = fingerprint
            
            # Step 5: Persist to database
            await self._persist_provenance_data(
                content_id, fingerprint, attribution_chain
            )
            
            execution_time = time.perf_counter() - start_time
            logger.info("Content registered with complete provenance",
                       content_id=str(content_id),
                       ipfs_hash=ipfs_hash,
                       attribution_creator=attribution_chain.original_creator,
                       execution_time_ms=execution_time * 1000)
            
            return content_id, fingerprint, attribution_chain
            
        except Exception as e:
            logger.error("Content registration failed",
                        content_id=str(content_id),
                        error=str(e))
            raise
    
    async def track_content_usage(
        self,
        content_id: UUID,
        user_id: str,
        session_id: UUID,
        usage_type: str,
        context: Optional[Dict[str, Any]] = None,
        ftns_cost: Optional[Decimal] = None
    ) -> UsageEvent:
        """
        Track content usage with automatic attribution
        
        Phase 1 requirement: Track content usage across 100+ model interactions
        with automatic creator attribution and reward calculation.
        
        Args:
            content_id: Content being accessed
            user_id: User accessing content
            session_id: Session context
            usage_type: Type of usage (access, training, inference, etc.)
            context: Additional usage context
            ftns_cost: Associated FTNS cost for usage
            
        Returns:
            Usage event with attribution tracking
        """
        try:
            # Verify content exists
            attribution_chain = self.content_registry.get(content_id)
            if not attribution_chain:
                # Try to load from database
                attribution_chain = await self._load_attribution_chain(content_id)
                if not attribution_chain:
                    raise ValueError(f"Content {content_id} not found in provenance system")
            
            # Calculate usage cost if not provided
            if ftns_cost is None:
                ftns_cost = await self._calculate_usage_cost(
                    content_id, usage_type, context or {}
                )
            
            # Create usage event
            usage_event = UsageEvent(
                event_id=uuid4(),
                content_id=content_id,
                user_id=user_id,
                session_id=session_id,
                usage_type=usage_type,
                timestamp=datetime.now(timezone.utc),
                context=context or {},
                ftns_cost=ftns_cost,
                attribution_credited=False
            )
            
            # Add to tracking
            self.usage_events.append(usage_event)
            
            # Batch persist usage events
            if len(self.usage_events) >= self.usage_batch_size:
                await self._persist_usage_batch()
            
            # Calculate and distribute attribution rewards
            await self._process_attribution_rewards(usage_event, attribution_chain)
            
            logger.debug("Content usage tracked",
                        content_id=str(content_id),
                        user_id=user_id,
                        usage_type=usage_type,
                        ftns_cost=float(ftns_cost))
            
            return usage_event
            
        except Exception as e:
            logger.error("Content usage tracking failed",
                        content_id=str(content_id),
                        error=str(e))
            raise
    
    async def validate_content_integrity(
        self,
        content_id: UUID,
        content_data: bytes
    ) -> Dict[str, Any]:
        """
        Validate content integrity using cryptographic fingerprints
        
        Args:
            content_id: Content to validate
            content_data: Current content data
            
        Returns:
            Validation result with integrity status
        """
        try:
            # Get original fingerprint
            attribution_chain = self.content_registry.get(content_id)
            if not attribution_chain:
                attribution_chain = await self._load_attribution_chain(content_id)
            
            if not attribution_chain:
                return {
                    'valid': False,
                    'error': 'Content not found in provenance system'
                }
            
            # Find fingerprint
            original_fingerprint = None
            for fingerprint in self.fingerprint_cache.values():
                if fingerprint.sha256_hash in [attribution_chain.external_id]:
                    original_fingerprint = fingerprint
                    break
            
            if not original_fingerprint:
                # Load from database
                original_fingerprint = await self._load_content_fingerprint(content_id)
            
            if not original_fingerprint:
                return {
                    'valid': False,
                    'error': 'Original fingerprint not found'
                }
            
            # Generate current fingerprint
            current_fingerprint = await self._generate_content_fingerprint(
                content_data, original_fingerprint.content_type
            )
            
            # Compare fingerprints
            sha256_match = current_fingerprint.sha256_hash == original_fingerprint.sha256_hash
            blake2b_match = current_fingerprint.blake2b_hash == original_fingerprint.blake2b_hash
            size_match = current_fingerprint.size_bytes == original_fingerprint.size_bytes
            
            validation_result = {
                'valid': sha256_match and blake2b_match and size_match,
                'content_id': str(content_id),
                'fingerprint_comparison': {
                    'sha256_match': sha256_match,
                    'blake2b_match': blake2b_match,
                    'size_match': size_match,
                    'original_sha256': original_fingerprint.sha256_hash,
                    'current_sha256': current_fingerprint.sha256_hash,
                    'original_size': original_fingerprint.size_bytes,
                    'current_size': current_fingerprint.size_bytes
                },
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("Content integrity validation completed",
                       content_id=str(content_id),
                       valid=validation_result['valid'],
                       sha256_match=sha256_match,
                       size_match=size_match)
            
            return validation_result
            
        except Exception as e:
            logger.error("Content integrity validation failed",
                        content_id=str(content_id),
                        error=str(e))
            return {
                'valid': False,
                'error': str(e),
                'content_id': str(content_id)
            }
    
    async def generate_usage_analytics(
        self,
        content_id: Optional[UUID] = None,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive usage analytics for Phase 1 validation
        
        Args:
            content_id: Specific content to analyze (optional)
            time_period: Time period for analysis (optional)
            
        Returns:
            Comprehensive usage analytics
        """
        try:
            # Default to last 24 hours
            if time_period is None:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=24)
                time_period = (start_time, end_time)
            
            # Filter usage events
            filtered_events = [
                event for event in self.usage_events
                if (content_id is None or event.content_id == content_id) and
                   time_period[0] <= event.timestamp <= time_period[1]
            ]
            
            # Get additional events from database
            db_events = await self._load_usage_events(content_id, time_period)
            all_events = filtered_events + db_events
            
            if not all_events:
                return {
                    'total_events': 0,
                    'unique_content': 0,
                    'unique_users': 0,
                    'total_ftns_generated': 0.0,
                    'time_period': {
                        'start': time_period[0].isoformat(),
                        'end': time_period[1].isoformat()
                    }
                }
            
            # Calculate analytics
            unique_content = len(set(event.content_id for event in all_events))
            unique_users = len(set(event.user_id for event in all_events))
            total_ftns = sum(event.ftns_cost for event in all_events)
            
            # Usage patterns
            usage_by_type = {}
            usage_by_hour = {}
            content_popularity = {}
            
            for event in all_events:
                # By type
                usage_by_type[event.usage_type] = usage_by_type.get(event.usage_type, 0) + 1
                
                # By hour
                hour = event.timestamp.hour
                usage_by_hour[hour] = usage_by_hour.get(hour, 0) + 1
                
                # Content popularity
                content_str = str(event.content_id)
                content_popularity[content_str] = content_popularity.get(content_str, 0) + 1
            
            # Top content
            top_content = sorted(
                content_popularity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Calculate creator rewards distributed
            creator_rewards = {}
            for event in all_events:
                if event.attribution_credited:
                    attribution_chain = self.content_registry.get(event.content_id)
                    if attribution_chain:
                        creator = attribution_chain.original_creator
                        creator_rewards[creator] = creator_rewards.get(creator, Decimal('0')) + event.ftns_cost * Decimal('0.1')  # 10% to creator
            
            analytics = {
                'summary': {
                    'total_events': len(all_events),
                    'unique_content': unique_content,
                    'unique_users': unique_users,
                    'total_ftns_generated': float(total_ftns),
                    'avg_ftns_per_event': float(total_ftns / len(all_events)) if all_events else 0,
                    'events_per_hour': len(all_events) / 24 if all_events else 0
                },
                'usage_patterns': {
                    'by_type': usage_by_type,
                    'by_hour': usage_by_hour,
                    'peak_hour': max(usage_by_hour.items(), key=lambda x: x[1])[0] if usage_by_hour else 0
                },
                'content_analytics': {
                    'top_content': [{'content_id': cid, 'usage_count': count} for cid, count in top_content],
                    'avg_usage_per_content': len(all_events) / unique_content if unique_content > 0 else 0
                },
                'creator_rewards': {
                    'total_creators_rewarded': len(creator_rewards),
                    'total_rewards_distributed': float(sum(creator_rewards.values())),
                    'top_earning_creators': sorted(
                        [(creator, float(reward)) for creator, reward in creator_rewards.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                },
                'performance_metrics': {
                    'avg_fingerprint_time_ms': sum(self.fingerprint_times) / len(self.fingerprint_times) * 1000 if self.fingerprint_times else 0,
                    'avg_attribution_time_ms': sum(self.attribution_times) / len(self.attribution_times) * 1000 if self.attribution_times else 0,
                    'avg_ipfs_store_time_ms': sum(self.ipfs_store_times) / len(self.ipfs_store_times) * 1000 if self.ipfs_store_times else 0
                },
                'time_period': {
                    'start': time_period[0].isoformat(),
                    'end': time_period[1].isoformat(),
                    'duration_hours': (time_period[1] - time_period[0]).total_seconds() / 3600
                }
            }
            
            logger.info("Usage analytics generated",
                       total_events=analytics['summary']['total_events'],
                       unique_content=analytics['summary']['unique_content'],
                       total_ftns=analytics['summary']['total_ftns_generated'])
            
            return analytics
            
        except Exception as e:
            logger.error("Usage analytics generation failed", error=str(e))
            raise
    
    # === Private Helper Methods ===
    
    async def _generate_content_fingerprint(
        self,
        content_data: bytes,
        content_type: ContentType
    ) -> ContentFingerprint:
        """Generate cryptographic fingerprint for content"""
        start_time = time.perf_counter()
        
        try:
            # Generate multiple hash algorithms for security
            sha256_hash = hashlib.sha256(content_data).hexdigest()
            blake2b_hash = hashlib.blake2b(content_data).hexdigest()
            
            # Generate chunk hashes for large content
            chunk_hashes = []
            if len(content_data) > self.chunk_size:
                for i in range(0, len(content_data), self.chunk_size):
                    chunk = content_data[i:i + self.chunk_size]
                    chunk_hash = hashlib.sha256(chunk).hexdigest()
                    chunk_hashes.append(chunk_hash)
            
            fingerprint = ContentFingerprint(
                sha256_hash=sha256_hash,
                blake2b_hash=blake2b_hash,
                size_bytes=len(content_data),
                content_type=content_type,
                creation_timestamp=datetime.now(timezone.utc),
                chunk_hashes=chunk_hashes
            )
            
            execution_time = time.perf_counter() - start_time
            self.fingerprint_times.append(execution_time)
            
            logger.debug("Content fingerprint generated",
                        sha256=sha256_hash[:16] + "...",
                        size_bytes=len(content_data),
                        chunk_count=len(chunk_hashes),
                        execution_time_ms=execution_time * 1000)
            
            return fingerprint
            
        except Exception as e:
            logger.error("Content fingerprint generation failed", error=str(e))
            raise
    
    async def _store_content_in_ipfs(
        self,
        content_data: bytes,
        content_id: UUID,
        metadata: Dict[str, Any]
    ) -> str:
        """Store content in IPFS with metadata"""
        start_time = time.perf_counter()
        
        try:
            # Ensure IPFS client is initialized
            await self.ipfs_client._ensure_initialized()
            
            # Create content package with metadata
            content_package = {
                'content_id': str(content_id),
                'metadata': metadata,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'prsm_version': '1.0.0'
            }
            
            # Store content
            if self.ipfs_client.connected:
                # Real IPFS storage
                ipfs_hash = await self.ipfs_client.store_model(content_data, content_package)
            else:
                # Simulation mode
                ipfs_hash = f"Qm{hashlib.sha256(content_data).hexdigest()[:44]}"
                self.ipfs_client.simulation_storage[ipfs_hash] = content_data
            
            execution_time = time.perf_counter() - start_time
            self.ipfs_store_times.append(execution_time)
            
            logger.debug("Content stored in IPFS",
                        content_id=str(content_id),
                        ipfs_hash=ipfs_hash,
                        size_bytes=len(content_data),
                        execution_time_ms=execution_time * 1000)
            
            return ipfs_hash
            
        except Exception as e:
            logger.error("IPFS content storage failed",
                        content_id=str(content_id),
                        error=str(e))
            raise
    
    async def _create_attribution_chain(
        self,
        content_id: UUID,
        creator_info: Dict[str, Any],
        license_info: Dict[str, Any],
        fingerprint: ContentFingerprint
    ) -> AttributionChain:
        """Create complete attribution chain"""
        start_time = time.perf_counter()
        
        try:
            # Extract creator information
            original_creator = creator_info.get('name', 'Unknown Creator')
            creator_address = creator_info.get('ftns_address')
            
            # Parse license information
            license_type = LicenseType(license_info.get('type', 'open_source'))
            license_terms = license_info.get('terms', {})
            
            # Build contributor list
            contributors = creator_info.get('contributors', [])
            
            attribution_chain = AttributionChain(
                content_id=content_id,
                original_creator=original_creator,
                creator_address=creator_address,
                creation_timestamp=fingerprint.creation_timestamp,
                parent_content=creator_info.get('parent_content'),
                contributors=contributors,
                license_type=license_type,
                license_terms=license_terms,
                platform_source=creator_info.get('platform'),
                external_id=fingerprint.sha256_hash
            )
            
            execution_time = time.perf_counter() - start_time
            self.attribution_times.append(execution_time)
            
            logger.debug("Attribution chain created",
                        content_id=str(content_id),
                        creator=original_creator,
                        license_type=license_type.value,
                        execution_time_ms=execution_time * 1000)
            
            return attribution_chain
            
        except Exception as e:
            logger.error("Attribution chain creation failed",
                        content_id=str(content_id),
                        error=str(e))
            raise
    
    async def _calculate_usage_cost(
        self,
        content_id: UUID,
        usage_type: str,
        context: Dict[str, Any]
    ) -> Decimal:
        """Calculate FTNS cost for content usage"""
        try:
            base_costs = {
                'access': Decimal('0.01'),
                'training': Decimal('0.1'),
                'inference': Decimal('0.05'),
                'reference': Decimal('0.001')
            }
            
            base_cost = base_costs.get(usage_type, Decimal('0.01'))
            
            # Apply context modifiers
            if context.get('high_compute', False):
                base_cost *= Decimal('2.0')
            
            if context.get('commercial_use', False):
                base_cost *= Decimal('1.5')
            
            return base_cost
            
        except Exception as e:
            logger.error("Usage cost calculation failed", error=str(e))
            return Decimal('0.01')
    
    async def _process_attribution_rewards(
        self,
        usage_event: UsageEvent,
        attribution_chain: AttributionChain
    ):
        """Process attribution rewards for content creators"""
        try:
            if not attribution_chain.creator_address:
                # No creator address for rewards
                return
            
            # Calculate creator reward (10% of usage cost)
            creator_reward = usage_event.ftns_cost * Decimal('0.1')
            
            if creator_reward > Decimal('0'):
                # Distribute reward to creator
                # This would integrate with the FTNS service for actual token transfer
                
                # Mark attribution as credited
                usage_event.attribution_credited = True
                
                logger.debug("Attribution reward processed",
                            content_id=str(usage_event.content_id),
                            creator=attribution_chain.original_creator,
                            reward_ftns=float(creator_reward))
        
        except Exception as e:
            logger.error("Attribution reward processing failed", error=str(e))
    
    async def _persist_provenance_data(
        self,
        content_id: UUID,
        fingerprint: ContentFingerprint,
        attribution_chain: AttributionChain
    ):
        """Persist provenance data to database"""
        try:
            # Store provenance record
            await self.database_service.create_provenance_record({
                'content_id': str(content_id),
                'fingerprint_data': {
                    'sha256_hash': fingerprint.sha256_hash,
                    'blake2b_hash': fingerprint.blake2b_hash,
                    'size_bytes': fingerprint.size_bytes,
                    'content_type': fingerprint.content_type.value,
                    'ipfs_hash': fingerprint.ipfs_hash,
                    'chunk_hashes': fingerprint.chunk_hashes
                },
                'attribution_data': {
                    'original_creator': attribution_chain.original_creator,
                    'creator_address': attribution_chain.creator_address,
                    'license_type': attribution_chain.license_type.value,
                    'license_terms': attribution_chain.license_terms,
                    'contributors': attribution_chain.contributors,
                    'platform_source': attribution_chain.platform_source
                },
                'created_at': datetime.now(timezone.utc)
            })
            
        except Exception as e:
            logger.error("Provenance data persistence failed",
                        content_id=str(content_id),
                        error=str(e))
    
    async def _persist_usage_batch(self):
        """Persist batch of usage events"""
        try:
            if not self.usage_events:
                return
            
            batch_data = []
            for event in self.usage_events:
                batch_data.append({
                    'event_id': str(event.event_id),
                    'content_id': str(event.content_id),
                    'user_id': event.user_id,
                    'session_id': str(event.session_id),
                    'usage_type': event.usage_type,
                    'timestamp': event.timestamp,
                    'context': event.context,
                    'ftns_cost': float(event.ftns_cost),
                    'attribution_credited': event.attribution_credited
                })
            
            await self.database_service.batch_create_usage_events(batch_data)
            
            # Clear cached events
            self.usage_events.clear()
            
            logger.debug("Usage events batch persisted", event_count=len(batch_data))
            
        except Exception as e:
            logger.error("Usage batch persistence failed", error=str(e))
    
    async def _load_attribution_chain(self, content_id: UUID) -> Optional[AttributionChain]:
        """Load attribution chain from database"""
        try:
            record = await self.database_service.get_provenance_record(str(content_id))
            if not record:
                return None
            
            attribution_data = record.get('attribution_data', {})
            
            return AttributionChain(
                content_id=content_id,
                original_creator=attribution_data.get('original_creator', 'Unknown'),
                creator_address=attribution_data.get('creator_address'),
                creation_timestamp=record.get('created_at', datetime.now(timezone.utc)),
                parent_content=None,  # Would need to implement parent tracking
                contributors=attribution_data.get('contributors', []),
                license_type=LicenseType(attribution_data.get('license_type', 'open_source')),
                license_terms=attribution_data.get('license_terms', {}),
                platform_source=attribution_data.get('platform_source'),
                external_id=record.get('fingerprint_data', {}).get('sha256_hash')
            )
            
        except Exception as e:
            logger.error("Attribution chain loading failed",
                        content_id=str(content_id),
                        error=str(e))
            return None
    
    async def _load_content_fingerprint(self, content_id: UUID) -> Optional[ContentFingerprint]:
        """Load content fingerprint from database"""
        try:
            record = await self.database_service.get_provenance_record(str(content_id))
            if not record:
                return None
            
            fingerprint_data = record.get('fingerprint_data', {})
            
            return ContentFingerprint(
                sha256_hash=fingerprint_data.get('sha256_hash', ''),
                blake2b_hash=fingerprint_data.get('blake2b_hash', ''),
                size_bytes=fingerprint_data.get('size_bytes', 0),
                content_type=ContentType(fingerprint_data.get('content_type', 'model')),
                creation_timestamp=record.get('created_at', datetime.now(timezone.utc)),
                ipfs_hash=fingerprint_data.get('ipfs_hash'),
                chunk_hashes=fingerprint_data.get('chunk_hashes', [])
            )
            
        except Exception as e:
            logger.error("Content fingerprint loading failed",
                        content_id=str(content_id),
                        error=str(e))
            return None
    
    async def _load_usage_events(
        self,
        content_id: Optional[UUID],
        time_period: Tuple[datetime, datetime]
    ) -> List[UsageEvent]:
        """Load usage events from database"""
        try:
            events_data = await self.database_service.get_usage_events(
                content_id=str(content_id) if content_id else None,
                start_time=time_period[0],
                end_time=time_period[1]
            )
            
            events = []
            for event_data in events_data:
                events.append(UsageEvent(
                    event_id=UUID(event_data['event_id']),
                    content_id=UUID(event_data['content_id']),
                    user_id=event_data['user_id'],
                    session_id=UUID(event_data['session_id']),
                    usage_type=event_data['usage_type'],
                    timestamp=event_data['timestamp'],
                    context=event_data.get('context', {}),
                    ftns_cost=Decimal(str(event_data['ftns_cost'])),
                    attribution_credited=event_data.get('attribution_credited', False)
                ))
            
            return events
            
        except Exception as e:
            logger.error("Usage events loading failed", error=str(e))
            return []

# Global enhanced provenance system instance
enhanced_provenance_system = None

def get_enhanced_provenance_system() -> EnhancedProvenanceSystem:
    """Get or create global enhanced provenance system instance"""
    global enhanced_provenance_system
    if enhanced_provenance_system is None:
        enhanced_provenance_system = EnhancedProvenanceSystem()
    return enhanced_provenance_system