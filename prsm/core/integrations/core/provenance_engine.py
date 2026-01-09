"""
Provenance Engine
================

FTNS-integrated provenance tracking engine for accurate creator attribution
and reward distribution in PRSM's integration layer.

Key Features:
- Chain of custody tracking for imported content
- Creator attribution and licensing metadata
- Usage analytics for reward calculation
- Integration with FTNS token economy
- Compliance with content licensing requirements
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from ..models.integration_models import (
    IntegrationPlatform, ProvenanceMetadata, IntegrationSource,
    ImportRequest, ImportResult
)
from prsm.core.models import ProvenanceRecord, FTNSTransaction, RoyaltyPayment
from ...data_layer.enhanced_ipfs import get_ipfs_client
from ...tokenomics.ftns_service import ftns_service


class ProvenanceEngine:
    """
    FTNS-integrated provenance tracking for creator rewards
    
    Maintains complete chain of custody for imported content and
    calculates appropriate creator compensation through FTNS tokens.
    """
    
    def __init__(self):
        """Initialize the provenance engine"""
        
        # Provenance tracking
        self.provenance_records: Dict[str, ProvenanceMetadata] = {}
        self.usage_analytics: Dict[str, Dict[str, int]] = {}
        self.reward_history: List[RoyaltyPayment] = []
        
        # Configuration
        self.base_reward_rates = {
            "model": 10.0,
            "dataset": 5.0,
            "repository": 3.0,
            "code": 1.0
        }
        
        self.platform_multipliers = {
            IntegrationPlatform.GITHUB: 1.0,
            IntegrationPlatform.HUGGINGFACE: 1.2,
            IntegrationPlatform.OLLAMA: 0.8,
            IntegrationPlatform.META_LLAMA: 1.5,
            IntegrationPlatform.DEEPSEEK: 1.3,
            IntegrationPlatform.QWEN: 1.3,
            IntegrationPlatform.MISTRAL: 1.4
        }
        
        print("📝 Provenance Engine initialized")
    
    async def initialize(self):
        """
        Initialize the provenance engine (async initialization)
        
        This method exists for compatibility with the SystemIntegrator
        and other components that expect an async initialize() method.
        """
        # ProvenanceEngine is currently stateless at initialization
        # Future enhancement: Could initialize IPFS connection, database connections, etc.
        print("📝 Provenance Engine async initialization completed")
        return True
    
    async def register_content_creator(self, creator_id: str, creator_name: str, 
                                     creator_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new content creator in the provenance system
        
        Args:
            creator_id: Unique identifier for the creator
            creator_name: Display name for the creator
            creator_metadata: Additional metadata about the creator
            
        Returns:
            str: Registration record ID
        """
        registration_id = str(uuid4())
        
        # Create creator record
        creator_record = {
            'registration_id': registration_id,
            'creator_id': creator_id,
            'creator_name': creator_name,
            'metadata': creator_metadata or {},
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'total_content_items': 0,
            'total_rewards_earned': 0.0
        }
        
        # Store in provenance records (using creator_id as key)
        self.provenance_records[creator_id] = creator_record
        
        return registration_id
    
    async def create_content_record(self, creator_id: str, content_id: str, 
                                  content_type: str, metadata: Dict[str, Any]) -> str:
        """
        Create a provenance record for content
        
        Args:
            creator_id: ID of the content creator
            content_id: Unique identifier for the content
            content_type: Type of content (e.g., 'academic_paper', 'model', 'dataset')
            metadata: Content metadata
            
        Returns:
            str: Provenance record ID
        """
        record_id = str(uuid4())
        
        # Create provenance record
        provenance_record = {
            'record_id': record_id,
            'creator_id': creator_id,
            'content_id': content_id,
            'content_type': content_type,
            'metadata': metadata,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'usage_count': 0,
            'total_rewards': 0.0
        }
        
        # Store record
        self.provenance_records[record_id] = provenance_record
        
        # Update creator's content count
        if creator_id in self.provenance_records:
            self.provenance_records[creator_id]['total_content_items'] += 1
        
        return record_id
    
    # === Provenance Tracking ===
    
    async def create_provenance_record(self, import_request: ImportRequest, 
                                     metadata: Dict[str, Any]) -> ProvenanceMetadata:
        """
        Create comprehensive provenance record for imported content
        
        Args:
            import_request: Original import request
            metadata: Content metadata from platform
            
        Returns:
            ProvenanceMetadata with complete attribution chain
        """
        try:
            content_id = f"{import_request.source.platform.value}:{import_request.source.external_id}"
            
            # Extract creator information
            original_creator = self._extract_creator_info(metadata)
            
            # Build attribution chain
            attribution_chain = await self._build_attribution_chain(import_request, metadata)
            
            # Extract license information
            license_info = await self._extract_license_info(import_request.source, metadata)
            
            # Create provenance record
            provenance = ProvenanceMetadata(
                content_id=content_id,
                original_creator=original_creator,
                platform_source=import_request.source.platform,
                external_id=import_request.source.external_id,
                attribution_chain=attribution_chain,
                license_info=license_info,
                usage_metrics={},
                reward_eligible=self._is_reward_eligible(license_info),
                total_rewards_paid=0.0
            )
            
            # Store provenance record
            self.provenance_records[content_id] = provenance
            
            # Initialize usage analytics
            self.usage_analytics[content_id] = {
                "imports": 1,
                "downloads": 0,
                "citations": 0,
                "derived_works": 0
            }
            
            # Create IPFS record for immutable provenance
            await self._store_provenance_ipfs(provenance)
            
            print(f"📝 Created provenance record for {content_id}")
            print(f"   - Creator: {original_creator}")
            print(f"   - License: {license_info.get('type', 'unknown')}")
            print(f"   - Reward eligible: {provenance.reward_eligible}")
            
            return provenance
            
        except Exception as e:
            print(f"❌ Failed to create provenance record: {e}")
            raise
    
    async def update_usage_metrics(self, content_id: str, usage_type: str, 
                                 user_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update usage metrics for content tracking
        
        Args:
            content_id: ID of the content being used
            usage_type: Type of usage (download, citation, derived_work, etc.)
            user_id: ID of the user performing the action
            metadata: Additional usage metadata
            
        Returns:
            True if update successful
        """
        try:
            if content_id not in self.usage_analytics:
                self.usage_analytics[content_id] = {}
            
            analytics = self.usage_analytics[content_id]
            
            # Update usage count
            if usage_type not in analytics:
                analytics[usage_type] = 0
            analytics[usage_type] += 1
            
            # Update provenance record
            if content_id in self.provenance_records:
                provenance = self.provenance_records[content_id]
                provenance.usage_metrics = analytics.copy()
                
                # Update last accessed time
                provenance.updated_at = datetime.now(timezone.utc)
            
            # Log usage event
            print(f"📊 Updated usage metrics for {content_id}: {usage_type} (+1)")
            
            # Trigger reward calculation if significant usage
            if analytics.get(usage_type, 0) % 10 == 0:  # Every 10 uses
                await self._calculate_usage_rewards(content_id)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update usage metrics: {e}")
            return False
    
    # === Reward Calculation and Distribution ===
    
    async def calculate_creator_rewards(self, content_id: str, 
                                      usage_period_days: int = 30) -> Optional[RoyaltyPayment]:
        """
        Calculate creator rewards based on usage metrics
        
        Args:
            content_id: ID of the content to calculate rewards for
            usage_period_days: Period to calculate rewards over
            
        Returns:
            RoyaltyPayment with reward details or None if not eligible
        """
        try:
            if content_id not in self.provenance_records:
                print(f"⚠️ No provenance record found for {content_id}")
                return None
            
            provenance = self.provenance_records[content_id]
            
            if not provenance.reward_eligible or not provenance.original_creator:
                print(f"⚠️ Content {content_id} not eligible for rewards")
                return None
            
            # Get usage metrics
            usage_metrics = self.usage_analytics.get(content_id, {})
            
            # Calculate base reward
            content_type = self._determine_content_type(content_id)
            base_rate = self.base_reward_rates.get(content_type, 1.0)
            
            # Platform multiplier
            platform_multiplier = self.platform_multipliers.get(provenance.platform_source, 1.0)
            
            # Usage multiplier based on activity
            usage_multiplier = self._calculate_usage_multiplier(usage_metrics)
            
            # Quality multiplier (could be based on ratings, reviews, etc.)
            quality_multiplier = 1.0  # Placeholder for future implementation
            
            # Calculate total reward
            base_amount = base_rate * platform_multiplier
            bonus_amount = base_amount * (usage_multiplier - 1.0) * quality_multiplier
            total_amount = base_amount + bonus_amount
            
            # Create royalty payment record
            royalty = RoyaltyPayment(
                content_id=content_id,
                creator_id=provenance.original_creator,
                usage_period_start=datetime.now(timezone.utc) - timedelta(days=usage_period_days),
                usage_period_end=datetime.now(timezone.utc),
                total_usage=sum(usage_metrics.values()),
                usage_type="integration_import",
                royalty_rate=base_rate,
                base_amount=base_amount,
                bonus_amount=bonus_amount,
                total_amount=total_amount,
                impact_multiplier=usage_multiplier,
                quality_score=quality_multiplier,
                status="pending"
            )
            
            print(f"💰 Calculated creator rewards for {content_id}")
            print(f"   - Creator: {provenance.original_creator}")
            print(f"   - Base amount: {base_amount:.2f} FTNS")
            print(f"   - Bonus amount: {bonus_amount:.2f} FTNS")
            print(f"   - Total amount: {total_amount:.2f} FTNS")
            
            return royalty
            
        except Exception as e:
            print(f"❌ Failed to calculate creator rewards: {e}")
            return None
    
    async def distribute_rewards(self, royalty: RoyaltyPayment) -> bool:
        """
        Distribute FTNS rewards to content creator
        
        Args:
            royalty: RoyaltyPayment with reward details
            
        Returns:
            True if distribution successful
        """
        try:
            # Use FTNS service to distribute rewards
            transaction = await ftns_service.reward_contribution(
                royalty.creator_id,
                "creator_royalty",
                royalty.total_amount,
                {
                    "content_id": royalty.content_id,
                    "usage_period": f"{royalty.usage_period_start} to {royalty.usage_period_end}",
                    "total_usage": royalty.total_usage,
                    "royalty_id": str(royalty.payment_id)
                }
            )
            
            if transaction:
                # Update royalty status
                royalty.status = "paid"
                royalty.payment_date = datetime.now(timezone.utc)
                
                # Update provenance record
                if royalty.content_id in self.provenance_records:
                    provenance = self.provenance_records[royalty.content_id]
                    provenance.total_rewards_paid += royalty.total_amount
                
                # Store in reward history
                self.reward_history.append(royalty)
                
                print(f"✅ Distributed {royalty.total_amount:.2f} FTNS to {royalty.creator_id}")
                return True
            else:
                print(f"❌ Failed to distribute rewards via FTNS service")
                return False
                
        except Exception as e:
            print(f"❌ Failed to distribute rewards: {e}")
            return False
    
    # === Analytics and Reporting ===
    
    async def get_provenance_report(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive provenance report for content
        
        Args:
            content_id: ID of the content
            
        Returns:
            Detailed provenance report or None if not found
        """
        if content_id not in self.provenance_records:
            return None
        
        provenance = self.provenance_records[content_id]
        usage_metrics = self.usage_analytics.get(content_id, {})
        
        # Calculate reward summary
        total_rewards = sum(r.total_amount for r in self.reward_history if r.content_id == content_id)
        
        return {
            "content_id": content_id,
            "provenance": provenance.model_dump(),
            "usage_metrics": usage_metrics,
            "reward_summary": {
                "total_rewards_paid": total_rewards,
                "reward_eligible": provenance.reward_eligible,
                "last_reward_date": max(
                    (r.payment_date for r in self.reward_history if r.content_id == content_id and r.payment_date),
                    default=None
                )
            },
            "attribution_verified": len(provenance.attribution_chain) > 0,
            "license_compliant": provenance.license_info.get("compliant", False)
        }
    
    async def get_creator_analytics(self, creator_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific creator
        
        Args:
            creator_id: ID of the creator
            
        Returns:
            Creator analytics summary
        """
        creator_content = [
            content_id for content_id, prov in self.provenance_records.items()
            if prov.original_creator == creator_id
        ]
        
        total_usage = sum(
            sum(self.usage_analytics.get(content_id, {}).values())
            for content_id in creator_content
        )
        
        total_rewards = sum(
            r.total_amount for r in self.reward_history
            if r.creator_id == creator_id
        )
        
        return {
            "creator_id": creator_id,
            "content_count": len(creator_content),
            "total_usage": total_usage,
            "total_rewards": total_rewards,
            "content_list": creator_content,
            "average_reward_per_content": total_rewards / max(len(creator_content), 1),
            "platforms": list(set(
                self.provenance_records[cid].platform_source
                for cid in creator_content
            ))
        }
    
    # === Private Helper Methods ===
    
    def _extract_creator_info(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract creator information from content metadata"""
        # Try various common fields for creator information
        creator_fields = ["author", "creator", "owner", "user", "username", "login"]
        
        for field in creator_fields:
            if field in metadata and metadata[field]:
                creator = metadata[field]
                if isinstance(creator, dict):
                    return creator.get("login") or creator.get("name") or creator.get("username")
                return str(creator)
        
        return None
    
    async def _build_attribution_chain(self, import_request: ImportRequest, 
                                     metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build complete attribution chain for content"""
        chain = []
        
        # Original creator entry
        creator = self._extract_creator_info(metadata)
        if creator:
            chain.append({
                "role": "original_creator",
                "identifier": creator,
                "platform": import_request.source.platform.value,
                "content_id": import_request.source.external_id,
                "timestamp": metadata.get("created_at", datetime.now(timezone.utc).isoformat())
            })
        
        # Importer entry
        chain.append({
            "role": "importer",
            "identifier": import_request.user_id,
            "platform": "prsm",
            "action": "import_to_prsm",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return chain
    
    async def _extract_license_info(self, source: IntegrationSource, 
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate license information"""
        license_info = {
            "type": "unknown",
            "compliant": False,
            "details": {},
            "source": metadata.get("license", {})
        }
        
        # Extract license from metadata
        license_data = metadata.get("license")
        if license_data:
            if isinstance(license_data, dict):
                license_info["type"] = license_data.get("key", "unknown")
                license_info["details"] = license_data
            else:
                license_info["type"] = str(license_data)
        
        # Check compliance (simplified)
        permissive_licenses = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "unlicense", "cc0-1.0"]
        license_info["compliant"] = license_info["type"].lower() in permissive_licenses
        
        return license_info
    
    def _is_reward_eligible(self, license_info: Dict[str, Any]) -> bool:
        """Determine if content is eligible for creator rewards"""
        return license_info.get("compliant", False)
    
    def _determine_content_type(self, content_id: str) -> str:
        """Determine content type from content ID"""
        # Simple heuristic based on content ID
        if "model" in content_id.lower():
            return "model"
        elif "dataset" in content_id.lower():
            return "dataset"
        elif "repo" in content_id.lower():
            return "repository"
        else:
            return "code"
    
    def _calculate_usage_multiplier(self, usage_metrics: Dict[str, int]) -> float:
        """Calculate usage-based reward multiplier"""
        total_usage = sum(usage_metrics.values())
        
        # Progressive multiplier based on usage
        if total_usage < 10:
            return 1.0
        elif total_usage < 50:
            return 1.2
        elif total_usage < 100:
            return 1.5
        elif total_usage < 500:
            return 2.0
        else:
            return 3.0
    
    async def _calculate_usage_rewards(self, content_id: str) -> None:
        """Calculate and potentially distribute usage-based rewards"""
        try:
            royalty = await self.calculate_creator_rewards(content_id)
            if royalty and royalty.total_amount > 1.0:  # Minimum threshold
                await self.distribute_rewards(royalty)
        except Exception as e:
            print(f"❌ Failed to calculate usage rewards for {content_id}: {e}")
    
    async def _store_provenance_ipfs(self, provenance: ProvenanceMetadata) -> Optional[str]:
        """Store provenance record in IPFS for immutability"""
        try:
            ipfs_client = get_ipfs_client()
            
            # Create provenance document
            provenance_doc = {
                "provenance_id": str(provenance.provenance_id),
                "content_id": provenance.content_id,
                "timestamp": provenance.created_at.isoformat(),
                "attribution_chain": provenance.attribution_chain,
                "license_info": provenance.license_info,
                "platform_source": provenance.platform_source.value,
                "checksum": self._calculate_provenance_checksum(provenance)
            }
            
            # Store in IPFS
            cid = await ipfs_client.store_json(provenance_doc)
            print(f"📦 Stored provenance record in IPFS: {cid}")
            return cid
            
        except Exception as e:
            print(f"⚠️ Failed to store provenance in IPFS: {e}")
            return None
    
    def _calculate_provenance_checksum(self, provenance: ProvenanceMetadata) -> str:
        """Calculate checksum for provenance integrity verification"""
        content = f"{provenance.content_id}:{provenance.original_creator}:{provenance.platform_source.value}"
        return hashlib.sha256(content.encode()).hexdigest()


# === Import needed for datetime operations ===
from datetime import timedelta