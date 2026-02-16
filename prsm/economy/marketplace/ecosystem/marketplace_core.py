#!/usr/bin/env python3
"""
Marketplace Core System
=======================

Core marketplace functionality for managing third-party integrations,
plugins, models, and services with enterprise-grade capabilities.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid
from pathlib import Path
import hashlib
try:
    import semver
except ImportError:
    semver = None

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of marketplace integrations"""
    AI_MODEL = "ai_model"
    PLUGIN = "plugin"
    EXTENSION = "extension"
    SERVICE = "service"
    CONNECTOR = "connector"
    WORKFLOW_TEMPLATE = "workflow_template"
    REASONING_CHAIN = "reasoning_chain"
    CUSTOM_TOOL = "custom_tool"
    DATA_SOURCE = "data_source"
    ANALYTICS_DASHBOARD = "analytics_dashboard"


class IntegrationStatus(Enum):
    """Integration status in marketplace"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"
    REJECTED = "rejected"


class PricingModel(Enum):
    """Pricing models for integrations"""
    FREE = "free"
    FREEMIUM = "freemium"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    REVENUE_SHARE = "revenue_share"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class LicenseType(Enum):
    """License types for integrations"""
    MIT = "mit"
    APACHE_2_0 = "apache_2_0"
    GPL_V3 = "gpl_v3"
    BSD_3_CLAUSE = "bsd_3_clause"
    PROPRIETARY = "proprietary"
    COMMERCIAL = "commercial"
    CREATIVE_COMMONS = "creative_commons"
    CUSTOM = "custom"


@dataclass
class IntegrationVersion:
    """Version information for integration"""
    version: str
    changelog: str = ""
    release_notes: str = ""
    
    # Compatibility
    min_prsm_version: str = "1.0.0"
    max_prsm_version: Optional[str] = None
    
    # Assets and files
    package_url: str = ""
    package_hash: str = ""
    package_size_bytes: int = 0
    
    # Status and metadata
    status: IntegrationStatus = IntegrationStatus.DRAFT
    published_at: Optional[datetime] = None
    download_count: int = 0
    
    # Security and compliance
    security_scan_passed: bool = False
    security_scan_report: Optional[str] = None
    
    # Performance metrics
    performance_score: float = 0.0
    quality_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "changelog": self.changelog,
            "release_notes": self.release_notes,
            "min_prsm_version": self.min_prsm_version,
            "max_prsm_version": self.max_prsm_version,
            "package_url": self.package_url,
            "package_hash": self.package_hash,
            "package_size_bytes": self.package_size_bytes,
            "status": self.status.value,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "download_count": self.download_count,
            "security_scan_passed": self.security_scan_passed,
            "security_scan_report": self.security_scan_report,
            "performance_score": self.performance_score,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Integration:
    """Marketplace integration definition"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    description: str = ""
    long_description: str = ""
    
    # Developer information
    developer_id: str = ""
    developer_name: str = ""
    organization: Optional[str] = None
    
    # Categorization
    category: str = "General"
    subcategory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Versions
    versions: Dict[str, IntegrationVersion] = field(default_factory=dict)
    latest_version: Optional[str] = None
    
    # Pricing and licensing
    pricing_model: PricingModel = PricingModel.FREE
    license_type: LicenseType = LicenseType.MIT
    price: float = 0.0
    currency: str = "USD"
    
    # Capabilities and requirements
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    system_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Media and documentation
    icon_url: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    demo_url: Optional[str] = None
    source_code_url: Optional[str] = None
    
    # Marketplace metrics
    install_count: int = 0
    active_installs: int = 0
    rating: float = 0.0
    review_count: int = 0
    
    # Status and moderation
    status: IntegrationStatus = IntegrationStatus.DRAFT
    featured: bool = False
    verified: bool = False
    
    # Support and maintenance
    support_email: Optional[str] = None
    support_url: Optional[str] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Enterprise features
    enterprise_ready: bool = False
    sla_available: bool = False
    white_label_available: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_version(self, version: IntegrationVersion):
        """Add a new version"""
        self.versions[version.version] = version
        
        # Update latest version if this is newer
        if not self.latest_version or semver.compare(version.version, self.latest_version) > 0:
            self.latest_version = version.version
        
        self.updated_at = datetime.now(timezone.utc)
    
    def get_version(self, version: str) -> Optional[IntegrationVersion]:
        """Get specific version"""
        return self.versions.get(version)
    
    def get_latest_version(self) -> Optional[IntegrationVersion]:
        """Get latest version object"""
        if self.latest_version:
            return self.versions.get(self.latest_version)
        return None
    
    def is_compatible_with_prsm(self, prsm_version: str) -> bool:
        """Check if integration is compatible with PRSM version"""
        latest = self.get_latest_version()
        if not latest:
            return False
        
        # Check minimum version
        if semver.compare(prsm_version, latest.min_prsm_version) < 0:
            return False
        
        # Check maximum version if specified
        if latest.max_prsm_version and semver.compare(prsm_version, latest.max_prsm_version) > 0:
            return False
        
        return True
    
    def calculate_popularity_score(self) -> float:
        """Calculate popularity score based on various metrics"""
        
        # Base score components
        install_score = min(self.install_count / 1000, 100)  # Cap at 100 for 1000+ installs
        rating_score = self.rating * 20  # Scale 0-5 rating to 0-100
        review_score = min(self.review_count / 50, 20)  # Cap at 20 for 50+ reviews
        
        # Recency bonus
        days_since_update = (datetime.now(timezone.utc) - self.last_updated).days
        recency_score = max(0, 20 - (days_since_update / 30))  # Decay over 30 days
        
        # Status bonuses
        status_bonus = 0
        if self.featured:
            status_bonus += 20
        if self.verified:
            status_bonus += 10
        if self.enterprise_ready:
            status_bonus += 10
        
        total_score = install_score + rating_score + review_score + recency_score + status_bonus
        
        return min(total_score, 200)  # Cap at 200
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "integration_type": self.integration_type.value,
            "description": self.description,
            "long_description": self.long_description,
            "developer_id": self.developer_id,
            "developer_name": self.developer_name,
            "organization": self.organization,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "versions": {v: version.to_dict() for v, version in self.versions.items()},
            "latest_version": self.latest_version,
            "pricing_model": self.pricing_model.value,
            "license_type": self.license_type.value,
            "price": self.price,
            "currency": self.currency,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "system_requirements": self.system_requirements,
            "icon_url": self.icon_url,
            "screenshots": self.screenshots,
            "documentation_url": self.documentation_url,
            "demo_url": self.demo_url,
            "source_code_url": self.source_code_url,
            "install_count": self.install_count,
            "active_installs": self.active_installs,
            "rating": self.rating,
            "review_count": self.review_count,
            "status": self.status.value,
            "featured": self.featured,
            "verified": self.verified,
            "support_email": self.support_email,
            "support_url": self.support_url,
            "last_updated": self.last_updated.isoformat(),
            "enterprise_ready": self.enterprise_ready,
            "sla_available": self.sla_available,
            "white_label_available": self.white_label_available,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "popularity_score": self.calculate_popularity_score()
        }


class IntegrationCategoryManager:
    """Manager for integration categories and organization"""
    
    def __init__(self):
        self.categories = {
            "AI Models": {
                "description": "AI models and language models",
                "subcategories": ["Language Models", "Vision Models", "Audio Models", "Multimodal Models", "Specialized Models"]
            },
            "Plugins": {
                "description": "System plugins and extensions",
                "subcategories": ["Productivity", "Development Tools", "Analytics", "Security", "Utilities"]
            },
            "Data Connectors": {
                "description": "Data source connectors and integrations",
                "subcategories": ["Databases", "APIs", "File Systems", "Cloud Storage", "Streaming"]
            },
            "Workflow Templates": {
                "description": "Pre-built workflow templates",
                "subcategories": ["Data Processing", "Analysis", "Automation", "Content Generation", "Research"]
            },
            "Custom Tools": {
                "description": "Custom tools and utilities",
                "subcategories": ["Analysis Tools", "Visualization", "Processing", "Conversion", "Validation"]
            },
            "Services": {
                "description": "External services and APIs",
                "subcategories": ["Cloud Services", "Third-party APIs", "Webhooks", "Notifications", "Storage"]
            }
        }
    
    def get_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get all categories"""
        return self.categories
    
    def get_subcategories(self, category: str) -> List[str]:
        """Get subcategories for a category"""
        return self.categories.get(category, {}).get("subcategories", [])
    
    def is_valid_category(self, category: str, subcategory: Optional[str] = None) -> bool:
        """Check if category/subcategory combination is valid"""
        if category not in self.categories:
            return False
        
        if subcategory:
            return subcategory in self.categories[category].get("subcategories", [])
        
        return True


class IntegrationSearchEngine:
    """Advanced search and discovery engine for integrations"""
    
    def __init__(self):
        self.search_index: Dict[str, Set[str]] = {}
        self.popularity_cache: Dict[str, float] = {}
        self.last_index_update = datetime.now(timezone.utc)
    
    def index_integration(self, integration: Integration):
        """Add integration to search index"""
        
        integration_id = integration.integration_id
        search_terms = set()
        
        # Add name and description terms
        search_terms.update(self._extract_terms(integration.name))
        search_terms.update(self._extract_terms(integration.description))
        search_terms.update(self._extract_terms(integration.long_description))
        
        # Add category terms
        search_terms.update(self._extract_terms(integration.category))
        if integration.subcategory:
            search_terms.update(self._extract_terms(integration.subcategory))
        
        # Add tags
        for tag in integration.tags:
            search_terms.update(self._extract_terms(tag))
        
        # Add capabilities
        for capability in integration.capabilities:
            search_terms.update(self._extract_terms(capability))
        
        # Add developer information
        search_terms.update(self._extract_terms(integration.developer_name))
        if integration.organization:
            search_terms.update(self._extract_terms(integration.organization))
        
        # Store in index
        for term in search_terms:
            if term not in self.search_index:
                self.search_index[term] = set()
            self.search_index[term].add(integration_id)
        
        # Update popularity cache
        self.popularity_cache[integration_id] = integration.calculate_popularity_score()
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None,
               sort_by: str = "relevance", limit: int = 20) -> List[str]:
        """Search integrations"""
        
        query_terms = self._extract_terms(query)
        
        if not query_terms:
            return []
        
        # Find matching integrations
        matching_integrations: Dict[str, float] = {}
        
        for term in query_terms:
            # Exact matches
            if term in self.search_index:
                for integration_id in self.search_index[term]:
                    matching_integrations[integration_id] = \
                        matching_integrations.get(integration_id, 0) + 1.0
            
            # Partial matches
            for indexed_term in self.search_index:
                if term in indexed_term or indexed_term in term:
                    similarity = self._calculate_similarity(term, indexed_term)
                    if similarity > 0.6:  # Threshold for partial matches
                        for integration_id in self.search_index[indexed_term]:
                            matching_integrations[integration_id] = \
                                matching_integrations.get(integration_id, 0) + similarity
        
        # Calculate final scores
        scored_results = []
        for integration_id, relevance_score in matching_integrations.items():
            
            # Normalize relevance score
            relevance_score = relevance_score / len(query_terms)
            
            # Get popularity score
            popularity_score = self.popularity_cache.get(integration_id, 0)
            
            # Combined score based on sort method
            if sort_by == "relevance":
                final_score = relevance_score * 0.7 + (popularity_score / 200) * 0.3
            elif sort_by == "popularity":
                final_score = (popularity_score / 200) * 0.7 + relevance_score * 0.3
            elif sort_by == "newest":
                # Would need creation date - using relevance for now
                final_score = relevance_score
            else:
                final_score = relevance_score
            
            scored_results.append((integration_id, final_score))
        
        # Sort and limit results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [integration_id for integration_id, _ in scored_results[:limit]]
    
    def get_trending_integrations(self, limit: int = 10) -> List[str]:
        """Get trending integrations based on recent activity"""
        
        # Sort by popularity score
        trending = sorted(
            self.popularity_cache.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [integration_id for integration_id, _ in trending[:limit]]
    
    def get_recommendations(self, integration_id: str, limit: int = 5) -> List[str]:
        """Get recommendations based on an integration"""
        
        # Simple collaborative filtering based on shared terms
        # In production, this would use more sophisticated ML algorithms
        
        if integration_id not in self.popularity_cache:
            return []
        
        # Find integrations with shared search terms
        shared_terms: Dict[str, int] = {}
        
        for term, integration_ids in self.search_index.items():
            if integration_id in integration_ids:
                for other_id in integration_ids:
                    if other_id != integration_id:
                        shared_terms[other_id] = shared_terms.get(other_id, 0) + 1
        
        # Score by shared terms and popularity
        recommendations = []
        for other_id, shared_count in shared_terms.items():
            popularity = self.popularity_cache.get(other_id, 0)
            score = shared_count * 0.6 + (popularity / 200) * 0.4
            recommendations.append((other_id, score))
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [rec_id for rec_id, _ in recommendations[:limit]]
    
    def _extract_terms(self, text: str) -> Set[str]:
        """Extract search terms from text"""
        if not text:
            return set()
        
        # Simple tokenization (would use more sophisticated NLP in production)
        terms = set()
        
        # Split on common delimiters and clean
        words = text.lower().replace('-', ' ').replace('_', ' ').split()
        
        for word in words:
            # Remove punctuation and filter short words
            cleaned = ''.join(c for c in word if c.isalnum())
            if len(cleaned) >= 2:
                terms.add(cleaned)
        
        return terms
    
    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity between two terms"""
        
        # Simple character-based similarity
        if term1 == term2:
            return 1.0
        
        # Jaccard similarity on character bigrams
        bigrams1 = set(term1[i:i+2] for i in range(len(term1)-1))
        bigrams2 = set(term2[i:i+2] for i in range(len(term2)-1))
        
        if not bigrams1 and not bigrams2:
            return 1.0
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return intersection / union if union > 0 else 0.0


class MarketplaceCore:
    """Core marketplace management system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./marketplace_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.integrations: Dict[str, Integration] = {}
        self.category_manager = IntegrationCategoryManager()
        self.search_engine = IntegrationSearchEngine()
        
        # Caching and performance
        self.integration_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Statistics and analytics
        self.stats = {
            "total_integrations": 0,
            "published_integrations": 0,
            "total_downloads": 0,
            "active_developers": 0,
            "categories_count": len(self.category_manager.categories),
            "average_rating": 0.0
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Any]] = {}
        
        logger.info("Marketplace Core initialized")
    
    async def initialize(self):
        """Initialize the marketplace system
        
        Performs any async initialization required for the marketplace.
        """
        try:
            # Load any persisted integrations
            if self.storage_path.exists():
                # Could load from database/files here
                pass
            
            # Initialize search engine
            for integration in self.integrations.values():
                self.search_engine.index_integration(integration)
            
            logger.info("Marketplace Core async initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing marketplace: {e}")
            return False
    
    def register_integration(self, integration: Integration) -> bool:
        """Register a new integration"""
        
        try:
            # Validate integration
            if not self._validate_integration(integration):
                return False
            
            # Store integration
            self.integrations[integration.integration_id] = integration
            
            # Update search index
            self.search_engine.index_integration(integration)
            
            # Clear cache
            self._clear_cache(integration.integration_id)
            
            # Update statistics
            self.stats["total_integrations"] += 1
            if integration.status == IntegrationStatus.PUBLISHED:
                self.stats["published_integrations"] += 1
            
            self._update_average_rating()
            
            logger.info(f"Registered integration: {integration.name}")
            
            # Emit event
            asyncio.create_task(self._emit_event("integration_registered", integration.to_dict()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register integration {integration.name}: {e}")
            return False
    
    def update_integration(self, integration_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing integration"""
        
        if integration_id not in self.integrations:
            logger.error(f"Integration not found: {integration_id}")
            return False
        
        try:
            integration = self.integrations[integration_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(integration, key):
                    setattr(integration, key, value)
            
            integration.updated_at = datetime.now(timezone.utc)
            
            # Re-index for search
            self.search_engine.index_integration(integration)
            
            # Clear cache
            self._clear_cache(integration_id)
            
            logger.info(f"Updated integration: {integration.name}")
            
            # Emit event
            asyncio.create_task(self._emit_event("integration_updated", integration.to_dict()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update integration {integration_id}: {e}")
            return False
    
    def get_integration(self, integration_id: str) -> Optional[Integration]:
        """Get integration by ID"""
        return self.integrations.get(integration_id)
    
    def get_integration_dict(self, integration_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get integration as dictionary with optional caching"""
        
        # Check cache first
        if use_cache and integration_id in self.integration_cache:
            cache_entry = self.integration_cache[integration_id]
            cache_age = (datetime.now(timezone.utc) - cache_entry["cached_at"]).total_seconds()
            
            if cache_age < self.cache_ttl_seconds:
                return cache_entry["data"]
        
        # Get from storage
        integration = self.get_integration(integration_id)
        if not integration:
            return None
        
        integration_dict = integration.to_dict()
        
        # Cache result
        if use_cache:
            self.integration_cache[integration_id] = {
                "data": integration_dict,
                "cached_at": datetime.now(timezone.utc)
            }
        
        return integration_dict
    
    def list_integrations(self, filters: Optional[Dict[str, Any]] = None,
                         sort_by: str = "popularity", limit: int = 50,
                         offset: int = 0) -> List[Dict[str, Any]]:
        """List integrations with filtering and sorting"""
        
        # Start with all integrations
        integrations = list(self.integrations.values())
        
        # Apply filters
        if filters:
            integrations = self._apply_filters(integrations, filters)
        
        # Sort integrations
        integrations = self._sort_integrations(integrations, sort_by)
        
        # Apply pagination
        total_count = len(integrations)
        integrations = integrations[offset:offset + limit]
        
        # Convert to dictionaries
        results = []
        for integration in integrations:
            integration_dict = self.get_integration_dict(integration.integration_id)
            if integration_dict:
                results.append(integration_dict)
        
        return results
    
    def search_integrations(self, query: str, filters: Optional[Dict[str, Any]] = None,
                           sort_by: str = "relevance", limit: int = 20) -> List[Dict[str, Any]]:
        """Search integrations"""
        
        # Perform search
        integration_ids = self.search_engine.search(query, filters, sort_by, limit)
        
        # Get integration details
        results = []
        for integration_id in integration_ids:
            integration_dict = self.get_integration_dict(integration_id)
            if integration_dict:
                # Apply additional filters if specified
                if not filters or self._matches_filters(integration_dict, filters):
                    results.append(integration_dict)
        
        return results
    
    def get_featured_integrations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured integrations"""
        
        featured = [
            integration for integration in self.integrations.values()
            if integration.featured and integration.status == IntegrationStatus.PUBLISHED
        ]
        
        # Sort by popularity
        featured.sort(key=lambda x: x.calculate_popularity_score(), reverse=True)
        
        return [self.get_integration_dict(integration.integration_id) 
                for integration in featured[:limit]]
    
    def get_trending_integrations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending integrations"""
        
        trending_ids = self.search_engine.get_trending_integrations(limit)
        
        return [self.get_integration_dict(integration_id) 
                for integration_id in trending_ids 
                if self.get_integration_dict(integration_id)]
    
    def get_recommendations(self, integration_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for an integration"""
        
        recommendation_ids = self.search_engine.get_recommendations(integration_id, limit)
        
        return [self.get_integration_dict(rec_id) 
                for rec_id in recommendation_ids 
                if self.get_integration_dict(rec_id)]
    
    def get_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get all categories with integration counts"""
        
        categories = self.category_manager.get_categories().copy()
        
        # Add integration counts
        for category_name, category_info in categories.items():
            category_count = len([
                integration for integration in self.integrations.values()
                if integration.category == category_name and 
                integration.status == IntegrationStatus.PUBLISHED
            ])
            category_info["integration_count"] = category_count
        
        return categories
    
    def increment_download_count(self, integration_id: str, version: Optional[str] = None):
        """Increment download count for integration"""
        
        integration = self.get_integration(integration_id)
        if not integration:
            return
        
        # Increment overall download count
        integration.install_count += 1
        
        # Increment version-specific download count
        if version and version in integration.versions:
            integration.versions[version].download_count += 1
        elif integration.latest_version and integration.latest_version in integration.versions:
            integration.versions[integration.latest_version].download_count += 1
        
        # Update statistics
        self.stats["total_downloads"] += 1
        
        # Clear cache
        self._clear_cache(integration_id)
        
        # Update search index (for popularity changes)
        self.search_engine.index_integration(integration)
    
    def _validate_integration(self, integration: Integration) -> bool:
        """Validate integration data"""
        
        # Basic validation
        if not integration.name or not integration.integration_id:
            logger.error("Integration missing required fields: name or integration_id")
            return False
        
        # Check for duplicate ID
        if integration.integration_id in self.integrations:
            logger.error(f"Integration ID already exists: {integration.integration_id}")
            return False
        
        # Validate category
        if not self.category_manager.is_valid_category(integration.category, integration.subcategory):
            logger.error(f"Invalid category/subcategory: {integration.category}/{integration.subcategory}")
            return False
        
        # Validate versions
        for version_str, version in integration.versions.items():
            try:
                # Validate semver format
                semver.VersionInfo.parse(version_str)
            except ValueError:
                logger.error(f"Invalid version format: {version_str}")
                return False
        
        return True
    
    def _apply_filters(self, integrations: List[Integration], 
                      filters: Dict[str, Any]) -> List[Integration]:
        """Apply filters to integration list"""
        
        filtered = integrations
        
        # Filter by status
        if "status" in filters:
            filtered = [i for i in filtered if i.status.value == filters["status"]]
        
        # Filter by type
        if "type" in filters:
            filtered = [i for i in filtered if i.integration_type.value == filters["type"]]
        
        # Filter by category
        if "category" in filters:
            filtered = [i for i in filtered if i.category == filters["category"]]
        
        # Filter by subcategory
        if "subcategory" in filters:
            filtered = [i for i in filtered if i.subcategory == filters["subcategory"]]
        
        # Filter by pricing model
        if "pricing_model" in filters:
            filtered = [i for i in filtered if i.pricing_model.value == filters["pricing_model"]]
        
        # Filter by tags
        if "tags" in filters:
            filter_tags = filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]
            filtered = [i for i in filtered if any(tag in i.tags for tag in filter_tags)]
        
        # Filter by rating
        if "min_rating" in filters:
            filtered = [i for i in filtered if i.rating >= filters["min_rating"]]
        
        # Filter by developer
        if "developer_id" in filters:
            filtered = [i for i in filtered if i.developer_id == filters["developer_id"]]
        
        # Filter by featured
        if "featured" in filters:
            filtered = [i for i in filtered if i.featured == filters["featured"]]
        
        # Filter by verified
        if "verified" in filters:
            filtered = [i for i in filtered if i.verified == filters["verified"]]
        
        # Filter by enterprise ready
        if "enterprise_ready" in filters:
            filtered = [i for i in filtered if i.enterprise_ready == filters["enterprise_ready"]]
        
        return filtered
    
    def _matches_filters(self, integration_dict: Dict[str, Any], 
                        filters: Dict[str, Any]) -> bool:
        """Check if integration matches filters"""
        
        # Status filter
        if "status" in filters and integration_dict["status"] != filters["status"]:
            return False
        
        # Type filter
        if "type" in filters and integration_dict["integration_type"] != filters["type"]:
            return False
        
        # Category filter
        if "category" in filters and integration_dict["category"] != filters["category"]:
            return False
        
        # Other filters...
        return True
    
    def _sort_integrations(self, integrations: List[Integration], sort_by: str) -> List[Integration]:
        """Sort integrations by specified criteria"""
        
        if sort_by == "popularity":
            return sorted(integrations, key=lambda x: x.calculate_popularity_score(), reverse=True)
        elif sort_by == "rating":
            return sorted(integrations, key=lambda x: x.rating, reverse=True)
        elif sort_by == "downloads":
            return sorted(integrations, key=lambda x: x.install_count, reverse=True)
        elif sort_by == "newest":
            return sorted(integrations, key=lambda x: x.created_at, reverse=True)
        elif sort_by == "updated":
            return sorted(integrations, key=lambda x: x.updated_at, reverse=True)
        elif sort_by == "name":
            return sorted(integrations, key=lambda x: x.name.lower())
        else:  # Default to popularity
            return sorted(integrations, key=lambda x: x.calculate_popularity_score(), reverse=True)
    
    def _clear_cache(self, integration_id: str):
        """Clear cache for specific integration"""
        if integration_id in self.integration_cache:
            del self.integration_cache[integration_id]
    
    def _update_average_rating(self):
        """Update average rating statistic"""
        published_integrations = [
            integration for integration in self.integrations.values()
            if integration.status == IntegrationStatus.PUBLISHED and integration.review_count > 0
        ]
        
        if published_integrations:
            total_rating = sum(i.rating * i.review_count for i in published_integrations)
            total_reviews = sum(i.review_count for i in published_integrations)
            self.stats["average_rating"] = total_rating / total_reviews if total_reviews > 0 else 0.0
        else:
            self.stats["average_rating"] = 0.0
    
    def add_event_handler(self, event_type: str, handler):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit marketplace event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get comprehensive marketplace statistics"""
        
        # Update dynamic stats
        self.stats["published_integrations"] = len([
            i for i in self.integrations.values() 
            if i.status == IntegrationStatus.PUBLISHED
        ])
        
        # Get developer count (would be calculated from actual developer data)
        unique_developers = len(set(i.developer_id for i in self.integrations.values() if i.developer_id))
        self.stats["active_developers"] = unique_developers
        
        return {
            "marketplace_statistics": self.stats,
            "top_categories": self._get_top_categories(),
            "recent_integrations": len([
                i for i in self.integrations.values()
                if (datetime.now(timezone.utc) - i.created_at).days <= 30
            ]),
            "integration_types": self._get_integration_type_counts(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_top_categories(self) -> List[Dict[str, Any]]:
        """Get top categories by integration count"""
        
        category_counts = {}
        for integration in self.integrations.values():
            if integration.status == IntegrationStatus.PUBLISHED:
                category_counts[integration.category] = category_counts.get(integration.category, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{"category": cat, "count": count} for cat, count in top_categories[:10]]
    
    def _get_integration_type_counts(self) -> Dict[str, int]:
        """Get count of integrations by type"""
        
        type_counts = {}
        for integration in self.integrations.values():
            if integration.status == IntegrationStatus.PUBLISHED:
                int_type = integration.integration_type.value
                type_counts[int_type] = type_counts.get(int_type, 0) + 1
        
        return type_counts


# Export main classes
__all__ = [
    'IntegrationType',
    'IntegrationStatus',
    'PricingModel',
    'LicenseType',
    'IntegrationVersion',
    'Integration',
    'IntegrationCategoryManager',
    'IntegrationSearchEngine',
    'MarketplaceCore'
]