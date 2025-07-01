"""
MCP Tool Marketplace
Decentralized marketplace for MCP tools with FTNS integration

This module implements a comprehensive marketplace for MCP (Model Context Protocol) tools,
enabling tool developers to monetize their tools while providing models with access to
enhanced capabilities. The marketplace integrates with PRSM's FTNS token economy and
provides discovery, deployment, and management capabilities for MCP tools.

Key Features:
- Tool discovery and search with advanced filtering
- FTNS-based tool pricing and revenue sharing
- Tool quality ratings and community reviews
- Automated tool deployment and configuration
- Security validation and sandboxing
- Performance monitoring and analytics

Architecture Integration:
- Extends PRSM's existing marketplace infrastructure
- Integrates with Tool Router for intelligent tool selection
- Connects to FTNS service for payments and rewards
- Provides API endpoints for tool management
- Supports both built-in and community-contributed tools
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.core.config import get_settings
from prsm.core.models import TimestampMixin
from prsm.agents.routers.tool_router import MCPToolSpec, ToolType, ToolCapability, ToolSecurityLevel
from prsm.tokenomics.ftns_service import FTNSService

logger = structlog.get_logger(__name__)
settings = get_settings()


class ToolListingStatus(str, Enum):
    """Status of tool listings in the marketplace"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"


class ToolPricingModel(str, Enum):
    """Pricing models for tool usage"""
    FREE = "free"
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    FREEMIUM = "freemium"
    ENTERPRISE = "enterprise"


class ToolQualityGrade(str, Enum):
    """Quality grades for tools based on community feedback"""
    EXPERIMENTAL = "experimental"    # New or unproven tools
    COMMUNITY = "community"         # Community-validated tools
    VERIFIED = "verified"           # Platform-verified tools
    PREMIUM = "premium"             # High-quality, well-maintained tools
    ENTERPRISE = "enterprise"       # Enterprise-grade tools with SLA


class ToolReview(BaseModel):
    """User review for a tool"""
    review_id: UUID = Field(default_factory=uuid4)
    tool_id: str
    user_id: str
    rating: int = Field(ge=1, le=5)
    title: str
    content: str
    
    # Usage context
    use_case: str
    model_used: Optional[str] = None
    performance_rating: int = Field(ge=1, le=5)
    ease_of_use: int = Field(ge=1, le=5)
    
    # Verification
    verified_purchase: bool = False
    verified_usage: bool = False
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolListing(TimestampMixin):
    """Complete tool listing in the marketplace"""
    listing_id: UUID = Field(default_factory=uuid4)
    tool_spec: MCPToolSpec
    
    # Marketplace metadata
    status: ToolListingStatus = ToolListingStatus.PENDING_REVIEW
    quality_grade: ToolQualityGrade = ToolQualityGrade.EXPERIMENTAL
    featured: bool = False
    
    # Developer information
    developer_id: str
    developer_name: str
    developer_verified: bool = False
    support_contact: Optional[str] = None
    documentation_url: Optional[str] = None
    source_code_url: Optional[str] = None
    
    # Pricing and economics
    pricing_model: ToolPricingModel = ToolPricingModel.FREE
    base_price_ftns: float = 0.0
    subscription_price_ftns: Optional[float] = None
    enterprise_price_ftns: Optional[float] = None
    revenue_share_percentage: float = 0.7  # 70% to developer, 30% to platform
    
    # Usage and performance metrics
    total_downloads: int = 0
    active_installations: int = 0
    total_executions: int = 0
    average_rating: float = 0.0
    review_count: int = 0
    
    # Performance metrics
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    uptime_percentage: float = 100.0
    
    # Compatibility and requirements
    compatible_models: List[str] = Field(default_factory=list)
    required_api_keys: List[str] = Field(default_factory=list)
    supported_platforms: List[str] = Field(default_factory=list)
    minimum_prsm_version: str = "1.0.0"
    
    # Content and discovery
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    screenshots: List[str] = Field(default_factory=list)
    demo_url: Optional[str] = None
    
    # Moderation and safety
    security_audited: bool = False
    audit_report_url: Optional[str] = None
    safety_warnings: List[str] = Field(default_factory=list)
    content_rating: str = "general"  # general, mature, enterprise
    
    # Release information
    version: str = "1.0.0"
    changelog: str = ""
    release_notes: str = ""
    
    def calculate_popularity_score(self) -> float:
        """Calculate overall popularity score"""
        # Weighted combination of metrics
        rating_score = self.average_rating / 5.0
        usage_score = min(self.total_executions / 10000.0, 1.0)  # Cap at 10k executions
        download_score = min(self.total_downloads / 1000.0, 1.0)  # Cap at 1k downloads
        performance_score = self.success_rate
        
        return (rating_score * 0.3 + usage_score * 0.3 + 
                download_score * 0.2 + performance_score * 0.2)


class ToolInstallation(BaseModel):
    """Record of tool installation by a user/model"""
    installation_id: UUID = Field(default_factory=uuid4)
    tool_id: str
    user_id: str
    model_id: Optional[str] = None
    
    # Installation details
    installation_type: str = "user"  # user, model, system
    configuration: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    
    # Usage tracking
    total_executions: int = 0
    last_execution: Optional[datetime] = None
    total_cost_ftns: float = 0.0
    
    # Performance data
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    
    installed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolUsageMetrics(BaseModel):
    """Detailed usage metrics for a tool"""
    tool_id: str
    date: datetime
    
    # Usage counts
    total_executions: int = 0
    unique_users: int = 0
    unique_models: int = 0
    
    # Performance metrics
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    
    # Financial metrics
    revenue_ftns: float = 0.0
    developer_earnings_ftns: float = 0.0
    platform_fees_ftns: float = 0.0
    
    # Geographic and temporal data
    peak_usage_hour: int = 12  # Hour of day (0-23)
    primary_regions: List[str] = Field(default_factory=list)


class ToolMarketplace:
    """
    MCP Tool Marketplace
    
    Comprehensive marketplace for MCP tools with integrated economics,
    discovery, and management capabilities. Enables tool developers to
    monetize their tools while providing models with enhanced capabilities.
    
    Key Features:
    - Tool listing and discovery with advanced search
    - FTNS-based economics with revenue sharing
    - Quality assurance and community ratings
    - Automated deployment and configuration
    - Performance monitoring and analytics
    - Security validation and sandboxing
    """
    
    def __init__(self):
        self.listings: Dict[str, ToolListing] = {}
        self.reviews: Dict[str, List[ToolReview]] = {}  # tool_id -> reviews
        self.installations: Dict[str, List[ToolInstallation]] = {}  # user_id -> installations
        self.usage_metrics: Dict[str, List[ToolUsageMetrics]] = {}  # tool_id -> daily metrics
        
        # Economic integration
        self.ftns_service = FTNSService()
        
        # Initialize with sample tools
        self._initialize_sample_tools()
        
        logger.info("Tool Marketplace initialized",
                   total_listings=len(self.listings))
    
    def _initialize_sample_tools(self):
        """Initialize marketplace with sample tool listings"""
        sample_tools = [
            ToolListing(
                tool_spec=MCPToolSpec(
                    tool_id="advanced_web_search",
                    name="Advanced Web Search",
                    description="Professional web search with AI-powered result analysis and summarization",
                    tool_type=ToolType.WEB_INTERACTION,
                    capabilities=[ToolCapability.READ, ToolCapability.ANALYZE],
                    security_level=ToolSecurityLevel.SAFE,
                    mcp_server_url="https://api.toolforge.com/search",
                    tool_schema={
                        "name": "advanced_search",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "num_results": {"type": "integer", "default": 10},
                                "include_analysis": {"type": "boolean", "default": True}
                            }
                        }
                    },
                    provider="ToolForge Inc.",
                    cost_per_use=2.0,
                    average_latency=1.5,
                    success_rate=0.98
                ),
                developer_id="toolforge_dev",
                developer_name="ToolForge Inc.",
                developer_verified=True,
                pricing_model=ToolPricingModel.PAY_PER_USE,
                base_price_ftns=2.0,
                quality_grade=ToolQualityGrade.VERIFIED,
                featured=True,
                tags=["search", "web", "analysis", "ai"],
                categories=["Research", "Data Collection"],
                total_downloads=2547,
                active_installations=1892,
                total_executions=15673,
                average_rating=4.6,
                review_count=89,
                security_audited=True
            ),
            ToolListing(
                tool_spec=MCPToolSpec(
                    tool_id="scientific_calculator",
                    name="Scientific Calculator Pro",
                    description="Advanced mathematical calculator with symbolic computation and graphing",
                    tool_type=ToolType.COMPUTATION,
                    capabilities=[ToolCapability.EXECUTE, ToolCapability.ANALYZE],
                    security_level=ToolSecurityLevel.SAFE,
                    mcp_server_url="https://api.mathtools.org/calculator",
                    tool_schema={
                        "name": "calculate",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                                "mode": {"type": "string", "default": "symbolic"},
                                "precision": {"type": "integer", "default": 15}
                            }
                        }
                    },
                    provider="MathTools",
                    cost_per_use=0.5,
                    average_latency=0.3,
                    success_rate=0.99
                ),
                developer_id="mathtools_team",
                developer_name="MathTools Research",
                developer_verified=True,
                pricing_model=ToolPricingModel.FREEMIUM,
                base_price_ftns=0.0,  # Free tier
                subscription_price_ftns=50.0,  # Pro features
                quality_grade=ToolQualityGrade.PREMIUM,
                featured=True,
                tags=["math", "calculation", "scientific", "research"],
                categories=["Mathematics", "Research", "Education"],
                total_downloads=5423,
                active_installations=3876,
                total_executions=45692,
                average_rating=4.8,
                review_count=156,
                security_audited=True
            ),
            ToolListing(
                tool_spec=MCPToolSpec(
                    tool_id="data_visualizer",
                    name="AI Data Visualizer",
                    description="Create intelligent data visualizations with automatic chart selection",
                    tool_type=ToolType.MULTIMEDIA,
                    capabilities=[ToolCapability.ANALYZE, ToolCapability.TRANSFORM],
                    security_level=ToolSecurityLevel.RESTRICTED,
                    mcp_server_url="https://viz.datatools.com/api",
                    tool_schema={
                        "name": "create_visualization",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data": {"type": "array"},
                                "chart_type": {"type": "string", "default": "auto"},
                                "style": {"type": "string", "default": "modern"}
                            }
                        }
                    },
                    provider="DataViz Solutions",
                    cost_per_use=3.0,
                    average_latency=2.1,
                    success_rate=0.94
                ),
                developer_id="dataviz_dev",
                developer_name="DataViz Solutions",
                developer_verified=False,
                pricing_model=ToolPricingModel.PAY_PER_USE,
                base_price_ftns=3.0,
                quality_grade=ToolQualityGrade.COMMUNITY,
                featured=False,
                tags=["visualization", "charts", "data", "analysis"],
                categories=["Data Analysis", "Visualization"],
                total_downloads=1234,
                active_installations=892,
                total_executions=7834,
                average_rating=4.2,
                review_count=34,
                security_audited=False
            ),
            ToolListing(
                tool_spec=MCPToolSpec(
                    tool_id="secure_file_manager",
                    name="Secure File Manager",
                    description="Enterprise-grade file operations with encryption and audit trails",
                    tool_type=ToolType.FILE_SYSTEM,
                    capabilities=[ToolCapability.READ, ToolCapability.WRITE, ToolCapability.VALIDATE],
                    security_level=ToolSecurityLevel.PRIVILEGED,
                    mcp_server_url="https://fileops.enterprise.com/api",
                    tool_schema={
                        "name": "file_operation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {"type": "string"},
                                "path": {"type": "string"},
                                "encryption": {"type": "boolean", "default": True}
                            }
                        }
                    },
                    provider="Enterprise Security Corp",
                    cost_per_use=5.0,
                    average_latency=0.8,
                    success_rate=0.97
                ),
                developer_id="enterprise_sec",
                developer_name="Enterprise Security Corp",
                developer_verified=True,
                pricing_model=ToolPricingModel.ENTERPRISE,
                base_price_ftns=5.0,
                enterprise_price_ftns=200.0,  # Monthly subscription
                quality_grade=ToolQualityGrade.ENTERPRISE,
                featured=True,
                tags=["security", "files", "encryption", "enterprise"],
                categories=["Security", "File Management", "Enterprise"],
                total_downloads=567,
                active_installations=234,
                total_executions=3421,
                average_rating=4.9,
                review_count=23,
                security_audited=True,
                required_api_keys=["enterprise_auth"],
                content_rating="enterprise"
            )
        ]
        
        for listing in sample_tools:
            self.listings[listing.tool_spec.tool_id] = listing
            
            # Add sample reviews
            self._add_sample_reviews(listing.tool_spec.tool_id, listing.review_count)
    
    def _add_sample_reviews(self, tool_id: str, count: int):
        """Add sample reviews for a tool"""
        if tool_id not in self.reviews:
            self.reviews[tool_id] = []
        
        sample_reviews = [
            {
                "rating": 5, "title": "Excellent tool!", 
                "content": "Works perfectly for my research needs. Highly recommended.",
                "performance_rating": 5, "ease_of_use": 4
            },
            {
                "rating": 4, "title": "Very useful", 
                "content": "Great functionality but could use better documentation.",
                "performance_rating": 4, "ease_of_use": 3
            },
            {
                "rating": 5, "title": "Game changer", 
                "content": "This tool has revolutionized my workflow. Worth every FTNS token.",
                "performance_rating": 5, "ease_of_use": 5
            },
            {
                "rating": 3, "title": "Good but pricey", 
                "content": "Works well but the cost adds up quickly for heavy usage.",
                "performance_rating": 4, "ease_of_use": 4
            }
        ]
        
        for i in range(min(count, 4)):  # Add up to 4 sample reviews
            review_data = sample_reviews[i % len(sample_reviews)]
            review = ToolReview(
                tool_id=tool_id,
                user_id=f"user_{i+1}",
                rating=review_data["rating"],
                title=review_data["title"],
                content=review_data["content"],
                use_case="Research and development",
                performance_rating=review_data["performance_rating"],
                ease_of_use=review_data["ease_of_use"],
                verified_purchase=True
            )
            self.reviews[tool_id].append(review)
    
    async def search_tools(self, query: str = None, 
                          tool_types: List[ToolType] = None,
                          capabilities: List[ToolCapability] = None,
                          security_level: ToolSecurityLevel = None,
                          pricing_model: ToolPricingModel = None,
                          quality_grade: ToolQualityGrade = None,
                          max_price: float = None,
                          featured_only: bool = False,
                          limit: int = 20,
                          offset: int = 0) -> Dict[str, Any]:
        """
        Search for tools in the marketplace with advanced filtering
        
        Args:
            query: Text search query
            tool_types: Filter by tool types
            capabilities: Filter by required capabilities
            security_level: Maximum security level allowed
            pricing_model: Filter by pricing model
            quality_grade: Minimum quality grade
            max_price: Maximum price per use
            featured_only: Show only featured tools
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            Search results with metadata
        """
        candidates = list(self.listings.values())
        
        # Apply filters
        if tool_types:
            candidates = [t for t in candidates if t.tool_spec.tool_type in tool_types]
        
        if capabilities:
            candidates = [t for t in candidates 
                         if any(cap in t.tool_spec.capabilities for cap in capabilities)]
        
        if security_level:
            candidates = [t for t in candidates 
                         if self._security_level_allows(t.tool_spec.security_level, security_level)]
        
        if pricing_model:
            candidates = [t for t in candidates if t.pricing_model == pricing_model]
        
        if quality_grade:
            quality_order = {
                ToolQualityGrade.EXPERIMENTAL: 0,
                ToolQualityGrade.COMMUNITY: 1,
                ToolQualityGrade.VERIFIED: 2,
                ToolQualityGrade.PREMIUM: 3,
                ToolQualityGrade.ENTERPRISE: 4
            }
            min_quality = quality_order[quality_grade]
            candidates = [t for t in candidates 
                         if quality_order[t.quality_grade] >= min_quality]
        
        if max_price is not None:
            candidates = [t for t in candidates if t.base_price_ftns <= max_price]
        
        if featured_only:
            candidates = [t for t in candidates if t.featured]
        
        # Text search
        if query:
            query_lower = query.lower()
            scored_candidates = []
            
            for tool in candidates:
                score = 0
                # Search in name (high weight)
                if query_lower in tool.tool_spec.name.lower():
                    score += 10
                # Search in description (medium weight)
                if query_lower in tool.tool_spec.description.lower():
                    score += 5
                # Search in tags (low weight)
                for tag in tool.tags:
                    if query_lower in tag.lower():
                        score += 2
                
                if score > 0:
                    scored_candidates.append((tool, score))
            
            # Sort by relevance score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [t[0] for t in scored_candidates]
        else:
            # Sort by popularity if no text search
            candidates.sort(key=lambda t: t.calculate_popularity_score(), reverse=True)
        
        # Apply pagination
        total_results = len(candidates)
        candidates = candidates[offset:offset + limit]
        
        return {
            "tools": [self._tool_listing_to_dict(tool) for tool in candidates],
            "total_results": total_results,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(candidates) < total_results,
            "search_metadata": {
                "query": query,
                "filters_applied": {
                    "tool_types": tool_types,
                    "capabilities": capabilities,
                    "security_level": security_level.value if security_level else None,
                    "pricing_model": pricing_model.value if pricing_model else None,
                    "quality_grade": quality_grade.value if quality_grade else None,
                    "max_price": max_price,
                    "featured_only": featured_only
                }
            }
        }
    
    async def get_tool_details(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool"""
        listing = self.listings.get(tool_id)
        if not listing:
            return None
        
        # Get reviews
        tool_reviews = self.reviews.get(tool_id, [])
        
        # Calculate additional metrics
        listing_dict = self._tool_listing_to_dict(listing)
        listing_dict.update({
            "reviews": [self._review_to_dict(review) for review in tool_reviews[-10:]],  # Last 10 reviews
            "review_summary": self._calculate_review_summary(tool_reviews),
            "compatibility_info": await self._get_compatibility_info(tool_id),
            "performance_metrics": await self._get_performance_metrics(tool_id),
            "similar_tools": await self._find_similar_tools(tool_id, limit=5)
        })
        
        return listing_dict
    
    async def install_tool(self, tool_id: str, user_id: str, 
                          model_id: str = None, 
                          configuration: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Install a tool for a user or model
        
        Args:
            tool_id: ID of the tool to install
            user_id: ID of the installing user
            model_id: Optional ID of the model to install for
            configuration: Tool configuration settings
            
        Returns:
            Installation result and details
        """
        listing = self.listings.get(tool_id)
        if not listing:
            return {
                "success": False,
                "error": "Tool not found",
                "error_code": "TOOL_NOT_FOUND"
            }
        
        # Check if already installed
        user_installations = self.installations.get(user_id, [])
        existing = next((i for i in user_installations 
                        if i.tool_id == tool_id and i.model_id == model_id), None)
        
        if existing:
            return {
                "success": False,
                "error": "Tool already installed",
                "error_code": "ALREADY_INSTALLED",
                "existing_installation": existing.installation_id
            }
        
        try:
            # Process payment if required
            cost = 0.0
            if listing.pricing_model in [ToolPricingModel.PAY_PER_USE, ToolPricingModel.SUBSCRIPTION]:
                cost = listing.subscription_price_ftns or listing.base_price_ftns or 0.0
                
                if cost > 0:
                    # Check user balance
                    balance = await self.ftns_service.get_balance(user_id)
                    if balance < cost:
                        return {
                            "success": False,
                            "error": f"Insufficient FTNS balance. Required: {cost}, Available: {balance}",
                            "error_code": "INSUFFICIENT_BALANCE"
                        }
                    
                    # Deduct payment
                    success = await self.ftns_service.transfer(
                        user_id, "marketplace", cost, 
                        f"Tool installation: {tool_id}"
                    )
                    
                    if not success:
                        return {
                            "success": False,
                            "error": "Payment processing failed",
                            "error_code": "PAYMENT_FAILED"
                        }
            
            # Create installation record
            installation = ToolInstallation(
                tool_id=tool_id,
                user_id=user_id,
                model_id=model_id,
                installation_type="model" if model_id else "user",
                configuration=configuration or {},
                total_cost_ftns=cost
            )
            
            # Add to user's installations
            if user_id not in self.installations:
                self.installations[user_id] = []
            self.installations[user_id].append(installation)
            
            # Update tool metrics
            listing.total_downloads += 1
            listing.active_installations += 1
            
            logger.info("Tool installed successfully",
                       tool_id=tool_id,
                       user_id=user_id,
                       model_id=model_id,
                       cost=cost)
            
            return {
                "success": True,
                "installation_id": installation.installation_id,
                "tool_id": tool_id,
                "cost_ftns": cost,
                "configuration": installation.configuration
            }
            
        except Exception as e:
            logger.error("Tool installation failed",
                        tool_id=tool_id,
                        user_id=user_id,
                        error=str(e))
            
            return {
                "success": False,
                "error": str(e),
                "error_code": "INSTALLATION_ERROR"
            }
    
    async def execute_tool(self, tool_id: str, user_id: str, 
                          parameters: Dict[str, Any],
                          model_id: str = None) -> Dict[str, Any]:
        """
        Execute a tool on behalf of a user/model
        
        This method handles tool execution, billing, and performance tracking
        """
        listing = self.listings.get(tool_id)
        if not listing:
            return {
                "success": False,
                "error": "Tool not found",
                "error_code": "TOOL_NOT_FOUND"
            }
        
        # Check installation
        user_installations = self.installations.get(user_id, [])
        installation = next((i for i in user_installations 
                           if i.tool_id == tool_id and 
                           (i.model_id == model_id or (model_id is None and i.model_id is None))), None)
        
        if not installation:
            return {
                "success": False,
                "error": "Tool not installed",
                "error_code": "NOT_INSTALLED"
            }
        
        if not installation.enabled:
            return {
                "success": False,
                "error": "Tool installation is disabled",
                "error_code": "INSTALLATION_DISABLED"
            }
        
        start_time = time.time()
        
        try:
            # Process payment for usage
            usage_cost = listing.base_price_ftns if listing.pricing_model == ToolPricingModel.PAY_PER_USE else 0.0
            
            if usage_cost > 0:
                balance = await self.ftns_service.get_balance(user_id)
                if balance < usage_cost:
                    return {
                        "success": False,
                        "error": f"Insufficient FTNS balance for tool usage. Required: {usage_cost}, Available: {balance}",
                        "error_code": "INSUFFICIENT_BALANCE"
                    }
                
                # Deduct usage fee
                success = await self.ftns_service.transfer(
                    user_id, "marketplace", usage_cost,
                    f"Tool usage: {tool_id}"
                )
                
                if not success:
                    return {
                        "success": False,
                        "error": "Payment processing failed",
                        "error_code": "PAYMENT_FAILED"
                    }
                
                # Revenue sharing
                developer_share = usage_cost * listing.revenue_share_percentage
                await self.ftns_service.transfer(
                    "marketplace", listing.developer_id, developer_share,
                    f"Tool revenue share: {tool_id}"
                )
            
            # Simulate tool execution (in production, this would call actual MCP server)
            await asyncio.sleep(listing.tool_spec.average_latency)
            
            execution_time = time.time() - start_time
            success = True  # Simulate success based on tool's success rate
            
            # Update metrics
            installation.total_executions += 1
            installation.last_execution = datetime.now(timezone.utc)
            installation.total_cost_ftns += usage_cost
            installation.average_execution_time = (
                (installation.average_execution_time * (installation.total_executions - 1) + execution_time) /
                installation.total_executions
            )
            
            listing.total_executions += 1
            listing.average_execution_time = (
                (listing.average_execution_time * (listing.total_executions - 1) + execution_time) /
                listing.total_executions
            )
            
            if success:
                result_data = {"message": f"Tool {tool_id} executed successfully", "parameters": parameters}
            else:
                installation.error_count += 1
                result_data = None
            
            return {
                "success": success,
                "tool_id": tool_id,
                "execution_time": execution_time,
                "cost_ftns": usage_cost,
                "result": result_data
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            installation.error_count += 1
            
            logger.error("Tool execution failed",
                        tool_id=tool_id,
                        user_id=user_id,
                        error=str(e))
            
            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
                "error_code": "EXECUTION_ERROR"
            }
    
    async def submit_review(self, tool_id: str, user_id: str, 
                           rating: int, title: str, content: str,
                           use_case: str = "", model_used: str = None,
                           performance_rating: int = 5,
                           ease_of_use: int = 5) -> Dict[str, Any]:
        """Submit a review for a tool"""
        listing = self.listings.get(tool_id)
        if not listing:
            return {
                "success": False,
                "error": "Tool not found",
                "error_code": "TOOL_NOT_FOUND"
            }
        
        # Check if user has used the tool
        user_installations = self.installations.get(user_id, [])
        has_installation = any(i.tool_id == tool_id for i in user_installations)
        
        review = ToolReview(
            tool_id=tool_id,
            user_id=user_id,
            rating=rating,
            title=title,
            content=content,
            use_case=use_case,
            model_used=model_used,
            performance_rating=performance_rating,
            ease_of_use=ease_of_use,
            verified_purchase=has_installation
        )
        
        # Add review
        if tool_id not in self.reviews:
            self.reviews[tool_id] = []
        self.reviews[tool_id].append(review)
        
        # Update tool metrics
        listing.review_count += 1
        all_ratings = [r.rating for r in self.reviews[tool_id]]
        listing.average_rating = sum(all_ratings) / len(all_ratings)
        
        return {
            "success": True,
            "review_id": review.review_id,
            "verified_purchase": review.verified_purchase
        }
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get comprehensive marketplace statistics"""
        return {
            "total_tools": len(self.listings),
            "tools_by_type": {
                tool_type.value: len([t for t in self.listings.values() 
                                    if t.tool_spec.tool_type == tool_type])
                for tool_type in ToolType
            },
            "tools_by_quality": {
                grade.value: len([t for t in self.listings.values() if t.quality_grade == grade])
                for grade in ToolQualityGrade
            },
            "tools_by_pricing": {
                model.value: len([t for t in self.listings.values() if t.pricing_model == model])
                for model in ToolPricingModel
            },
            "total_downloads": sum(t.total_downloads for t in self.listings.values()),
            "total_executions": sum(t.total_executions for t in self.listings.values()),
            "total_reviews": sum(t.review_count for t in self.listings.values()),
            "average_rating": sum(t.average_rating * t.review_count for t in self.listings.values()) / 
                            max(sum(t.review_count for t in self.listings.values()), 1),
            "featured_tools": len([t for t in self.listings.values() if t.featured]),
            "verified_developers": len(set(t.developer_id for t in self.listings.values() if t.developer_verified))
        }
    
    def _tool_listing_to_dict(self, listing: ToolListing) -> Dict[str, Any]:
        """Convert tool listing to dictionary for API responses"""
        return {
            "tool_id": listing.tool_spec.tool_id,
            "name": listing.tool_spec.name,
            "description": listing.tool_spec.description,
            "tool_type": listing.tool_spec.tool_type.value,
            "capabilities": [cap.value for cap in listing.tool_spec.capabilities],
            "security_level": listing.tool_spec.security_level.value,
            "pricing_model": listing.pricing_model.value,
            "base_price_ftns": listing.base_price_ftns,
            "subscription_price_ftns": listing.subscription_price_ftns,
            "quality_grade": listing.quality_grade.value,
            "featured": listing.featured,
            "developer_name": listing.developer_name,
            "developer_verified": listing.developer_verified,
            "average_rating": listing.average_rating,
            "review_count": listing.review_count,
            "total_downloads": listing.total_downloads,
            "total_executions": listing.total_executions,
            "tags": listing.tags,
            "categories": listing.categories,
            "version": listing.version,
            "security_audited": listing.security_audited,
            "popularity_score": listing.calculate_popularity_score()
        }
    
    def _review_to_dict(self, review: ToolReview) -> Dict[str, Any]:
        """Convert review to dictionary for API responses"""
        return {
            "review_id": str(review.review_id),
            "rating": review.rating,
            "title": review.title,
            "content": review.content,
            "use_case": review.use_case,
            "performance_rating": review.performance_rating,
            "ease_of_use": review.ease_of_use,
            "verified_purchase": review.verified_purchase,
            "created_at": review.created_at.isoformat()
        }
    
    def _calculate_review_summary(self, reviews: List[ToolReview]) -> Dict[str, Any]:
        """Calculate review summary statistics"""
        if not reviews:
            return {"total_reviews": 0}
        
        ratings = [r.rating for r in reviews]
        performance_ratings = [r.performance_rating for r in reviews]
        ease_ratings = [r.ease_of_use for r in reviews]
        
        rating_distribution = {i: ratings.count(i) for i in range(1, 6)}
        
        return {
            "total_reviews": len(reviews),
            "average_rating": sum(ratings) / len(ratings),
            "average_performance": sum(performance_ratings) / len(performance_ratings),
            "average_ease_of_use": sum(ease_ratings) / len(ease_ratings),
            "rating_distribution": rating_distribution,
            "verified_reviews": len([r for r in reviews if r.verified_purchase])
        }
    
    async def _get_compatibility_info(self, tool_id: str) -> Dict[str, Any]:
        """Get tool compatibility information"""
        listing = self.listings.get(tool_id)
        if not listing:
            return {}
        
        return {
            "compatible_models": listing.compatible_models,
            "required_permissions": listing.tool_spec.required_permissions,
            "supported_platforms": listing.supported_platforms,
            "minimum_prsm_version": listing.minimum_prsm_version,
            "required_api_keys": listing.required_api_keys
        }
    
    async def _get_performance_metrics(self, tool_id: str) -> Dict[str, Any]:
        """Get detailed performance metrics for a tool"""
        listing = self.listings.get(tool_id)
        if not listing:
            return {}
        
        return {
            "average_execution_time": listing.average_execution_time,
            "success_rate": listing.success_rate,
            "uptime_percentage": listing.uptime_percentage,
            "total_executions": listing.total_executions,
            "active_installations": listing.active_installations
        }
    
    async def _find_similar_tools(self, tool_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find tools similar to the given tool"""
        current_tool = self.listings.get(tool_id)
        if not current_tool:
            return []
        
        # Find tools with similar type, capabilities, or tags
        similar_tools = []
        for other_id, other_tool in self.listings.items():
            if other_id == tool_id:
                continue
            
            similarity_score = 0
            
            # Same type
            if other_tool.tool_spec.tool_type == current_tool.tool_spec.tool_type:
                similarity_score += 3
            
            # Overlapping capabilities
            capability_overlap = len(set(other_tool.tool_spec.capabilities) & 
                                   set(current_tool.tool_spec.capabilities))
            similarity_score += capability_overlap
            
            # Overlapping tags
            tag_overlap = len(set(other_tool.tags) & set(current_tool.tags))
            similarity_score += tag_overlap * 0.5
            
            if similarity_score > 1:
                similar_tools.append((other_tool, similarity_score))
        
        # Sort by similarity and return top results
        similar_tools.sort(key=lambda x: x[1], reverse=True)
        return [self._tool_listing_to_dict(tool[0]) for tool in similar_tools[:limit]]
    
    def _security_level_allows(self, tool_level: ToolSecurityLevel, 
                              max_level: ToolSecurityLevel) -> bool:
        """Check if tool security level is allowed"""
        level_hierarchy = {
            ToolSecurityLevel.SAFE: 0,
            ToolSecurityLevel.RESTRICTED: 1,
            ToolSecurityLevel.PRIVILEGED: 2,
            ToolSecurityLevel.DANGEROUS: 3
        }
        return level_hierarchy[tool_level] <= level_hierarchy[max_level]


# Global marketplace instance
tool_marketplace = ToolMarketplace()