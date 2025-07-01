"""
PRSM Expanded Marketplace Service
=================================

Comprehensive marketplace service managing all resource types:
- AI Models, MCP Tools, Curated Datasets
- Agentic Workflows, Compute Resources, Knowledge Resources
- Evaluation Services, Training Services, Safety Tools

This unified service provides discovery, transactions, quality assurance,
and ecosystem management for the complete PRSM AI marketplace.
"""

import asyncio
import structlog
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.orm import Session

from ..core.database import get_database_service
from ..integrations.security.audit_logger import audit_logger
from ..tokenomics.ftns_service import FTNSService
from .expanded_models import (
    # Resource types
    ResourceType, ResourceStatus, PricingModel, QualityGrade,
    
    # Specific resource models
    DatasetListing, AgentWorkflowListing, ComputeResourceListing,
    KnowledgeResourceListing, EvaluationServiceListing, 
    TrainingServiceListing, SafetyToolListing,
    
    # Database models
    UnifiedResourceListingDB, ResourceReviewDB, ResourceOrderDB,
    
    # Search and response models
    UnifiedSearchFilters, MarketplaceSearchResponse, MarketplaceStatsResponse
)

logger = structlog.get_logger(__name__)


class ExpandedMarketplaceService:
    """
    Unified marketplace service for all PRSM AI resources
    
    Features:
    - Multi-resource type management and discovery
    - Advanced search and filtering across all categories
    - Quality assurance and verification workflows
    - FTNS token integration for all transactions
    - Comprehensive analytics and insights
    - Automated recommendation engine
    """
    
    def __init__(self):
        self.db_service = get_database_service()
        self.ftns_service = FTNSService()
        
        # Marketplace configuration
        self.platform_fee_percentage = Decimal('0.025')  # 2.5% platform fee
        self.quality_boost_multipliers = {
            QualityGrade.EXPERIMENTAL: Decimal('1.0'),
            QualityGrade.COMMUNITY: Decimal('1.2'),
            QualityGrade.VERIFIED: Decimal('1.5'),
            QualityGrade.PREMIUM: Decimal('2.0'),
            QualityGrade.ENTERPRISE: Decimal('3.0')
        }
        
        # Cache configuration
        self._search_cache = {}
        self._stats_cache = {}
        self._cache_ttl = timedelta(minutes=15)
        self._last_cache_update = {}
        
        # Resource type handlers
        self._resource_handlers = {
            ResourceType.DATASET: self._handle_dataset,
            ResourceType.AGENT_WORKFLOW: self._handle_agent_workflow,
            ResourceType.COMPUTE_RESOURCE: self._handle_compute_resource,
            ResourceType.KNOWLEDGE_RESOURCE: self._handle_knowledge_resource,
            ResourceType.EVALUATION_SERVICE: self._handle_evaluation_service,
            ResourceType.TRAINING_SERVICE: self._handle_training_service,
            ResourceType.SAFETY_TOOL: self._handle_safety_tool
        }
    
    # ========================================================================
    # RESOURCE CREATION AND MANAGEMENT
    # ========================================================================
    
    async def create_resource_listing(
        self,
        resource_type: ResourceType,
        resource_data: Union[DatasetListing, AgentWorkflowListing, ComputeResourceListing,
                           KnowledgeResourceListing, EvaluationServiceListing,
                           TrainingServiceListing, SafetyToolListing],
        owner_user_id: UUID
    ) -> Dict[str, Any]:
        """
        Create a new resource listing in the marketplace
        
        Args:
            resource_type: Type of resource being listed
            resource_data: Resource-specific data model
            owner_user_id: ID of the user creating the listing
            
        Returns:
            Created resource listing with marketplace metadata
        """
        try:
            # Validate resource data using appropriate handler
            handler = self._resource_handlers.get(resource_type)
            if not handler:
                raise ValueError(f"Unsupported resource type: {resource_type}")
            
            # Process resource-specific data
            processed_data = await handler(resource_data, "create")
            
            # Create unified database record
            resource_id = uuid4()
            
            unified_data = {
                "id": resource_id,
                "resource_type": resource_type.value,
                "name": resource_data.name,
                "description": resource_data.description,
                "owner_user_id": owner_user_id,
                "status": ResourceStatus.PENDING_REVIEW.value,
                "quality_grade": resource_data.quality_grade.value,
                "pricing_model": resource_data.pricing_model.value,
                "base_price": resource_data.base_price,
                "resource_metadata": processed_data["metadata"],
                "technical_specs": processed_data["technical_specs"],
                "access_config": processed_data["access_config"],
                "tags": resource_data.tags,
                "provider_name": getattr(resource_data, 'provider_name', None),
                "documentation_url": getattr(resource_data, 'documentation_url', None)
            }
            
            # Store in database
            db_record = UnifiedResourceListingDB(**unified_data)
            
            with self.db_service.get_session() as session:
                session.add(db_record)
                session.commit()
                session.refresh(db_record)
            
            # Log creation
            await audit_logger.log_marketplace_action(
                user_id=str(owner_user_id),
                action="resource_created",
                resource_type=resource_type.value,
                resource_id=str(resource_id),
                metadata={"name": resource_data.name}
            )
            
            logger.info(
                "Resource listing created",
                resource_id=str(resource_id),
                resource_type=resource_type.value,
                owner_id=str(owner_user_id)
            )
            
            return await self._format_resource_response(db_record)
            
        except Exception as e:
            logger.error(
                "Failed to create resource listing",
                error=str(e),
                resource_type=resource_type.value,
                owner_id=str(owner_user_id)
            )
            raise
    
    async def update_resource_listing(
        self,
        resource_id: UUID,
        updates: Dict[str, Any],
        user_id: UUID
    ) -> Dict[str, Any]:
        """Update an existing resource listing"""
        try:
            with self.db_service.get_session() as session:
                # Get existing resource
                resource = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.id == resource_id
                ).first()
                
                if not resource:
                    raise ValueError(f"Resource {resource_id} not found")
                
                # Verify ownership or admin permissions
                if resource.owner_user_id != user_id:
                    # Check admin permissions for resource modification
                    has_admin_permission = await self._check_admin_resource_permission(user_id, resource_id, "update")
                    if not has_admin_permission:
                        await audit_logger.log_marketplace_action(
                            action="resource_update_denied",
                            user_id=user_id,
                            resource_id=resource_id,
                            metadata={"reason": "insufficient_permissions", "owner_id": resource.owner_user_id}
                        )
                        raise PermissionError("Insufficient permissions to update resource")
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(resource, key):
                        setattr(resource, key, value)
                
                resource.updated_at = datetime.now(timezone.utc)
                session.commit()
                session.refresh(resource)
                
                await audit_logger.log_marketplace_action(
                    user_id=str(user_id),
                    action="resource_updated",
                    resource_id=str(resource_id),
                    metadata={"updates": list(updates.keys())}
                )
                
                return await self._format_resource_response(resource)
                
        except Exception as e:
            logger.error(
                "Failed to update resource listing",
                error=str(e),
                resource_id=str(resource_id),
                user_id=str(user_id)
            )
            raise
    
    async def _check_admin_resource_permission(self, user_id: str, resource_id: str, action: str) -> bool:
        """Check if user has admin permissions for resource operations"""
        try:
            # Import auth manager to check admin permissions
            from prsm.auth.enhanced_authorization import get_enhanced_auth_manager
            from prsm.auth.models import UserRole
            
            auth_manager = get_enhanced_auth_manager()
            
            # Check if user has admin role and marketplace permissions
            has_admin_permission = await auth_manager.check_permission(
                user_id=user_id,
                user_role=UserRole.ADMIN,  # Would fetch actual role from database
                resource_type="marketplace_resources",
                action=action
            )
            
            if has_admin_permission:
                logger.info("Admin permission granted for resource operation",
                           user_id=user_id,
                           resource_id=resource_id,
                           action=action)
                return True
            
            # Check for specialized marketplace admin role
            try:
                has_marketplace_admin = await auth_manager.check_permission(
                    user_id=user_id,
                    user_role=UserRole.MARKETPLACE_ADMIN,  # Specialized role for marketplace
                    resource_type="marketplace_resources",
                    action=action
                )
                
                if has_marketplace_admin:
                    logger.info("Marketplace admin permission granted for resource operation",
                               user_id=user_id,
                               resource_id=resource_id,
                               action=action)
                    return True
            except:
                # UserRole.MARKETPLACE_ADMIN might not exist, continue with other checks
                pass
            
            # Check for resource-specific admin permissions
            has_resource_admin = await self._check_resource_specific_admin_permission(user_id, resource_id, action)
            if has_resource_admin:
                logger.info("Resource-specific admin permission granted",
                           user_id=user_id,
                           resource_id=resource_id,
                           action=action)
                return True
            
            logger.debug("Admin permission denied for resource operation",
                        user_id=user_id,
                        resource_id=resource_id,
                        action=action)
            return False
            
        except Exception as e:
            logger.error("Failed to check admin resource permission",
                        user_id=user_id,
                        resource_id=resource_id,
                        action=action,
                        error=str(e))
            return False  # Secure default: deny permission on error
    
    async def _check_resource_specific_admin_permission(self, user_id: str, resource_id: str, action: str) -> bool:
        """Check for resource-specific admin permissions (e.g., category moderators)"""
        try:
            with self.database_service.get_session() as session:
                # Check if user is a moderator for the resource's category
                resource = session.query(MarketplaceResource).filter(
                    MarketplaceResource.id == resource_id
                ).first()
                
                if not resource:
                    return False
                
                # Check category moderator permissions (simplified implementation)
                # In a full implementation, you would have a CategoryModerator table
                
                # For now, check if user has any special roles stored in user metadata
                # or a dedicated permissions table
                
                # Placeholder for category-based permissions
                # This would be expanded based on your specific authorization model
                
                logger.debug("Resource-specific admin check completed (placeholder)",
                           user_id=user_id,
                           resource_id=resource_id,
                           action=action)
                
                return False  # Conservative default
                
        except Exception as e:
            logger.error("Failed to check resource-specific admin permission",
                        user_id=user_id,
                        resource_id=resource_id,
                        action=action,
                        error=str(e))
            return False
    
    # ========================================================================
    # SEARCH AND DISCOVERY
    # ========================================================================
    
    async def search_resources(
        self,
        filters: UnifiedSearchFilters,
        user_id: Optional[UUID] = None
    ) -> MarketplaceSearchResponse:
        """
        Advanced search across all marketplace resources
        
        Args:
            filters: Search and filter criteria
            user_id: Optional user ID for personalized results
            
        Returns:
            Search results with faceted navigation and metadata
        """
        try:
            # Check cache first
            cache_key = self._generate_search_cache_key(filters, user_id)
            cached_result = self._get_cached_search(cache_key)
            if cached_result:
                return cached_result
            
            with self.db_service.get_session() as session:
                # Build base query
                query = session.query(UnifiedResourceListingDB)
                
                # Apply filters
                query = self._apply_search_filters(query, filters)
                
                # Get total count before pagination
                total_count = query.count()
                
                # Apply sorting
                query = self._apply_sorting(query, filters.sort_by, filters.sort_order)
                
                # Apply pagination
                query = query.offset(filters.offset).limit(filters.limit)
                
                # Execute query
                resources = query.all()
                
                # Format results
                formatted_resources = []
                for resource in resources:
                    formatted_resource = await self._format_resource_response(resource)
                    # Add personalization if user provided
                    if user_id:
                        formatted_resource = await self._add_personalization(
                            formatted_resource, user_id
                        )
                    formatted_resources.append(formatted_resource)
                
                # Generate facets
                facets = await self._generate_search_facets(session, filters)
                
                # Create response
                response = MarketplaceSearchResponse(
                    resources=formatted_resources,
                    total_count=total_count,
                    has_more=(filters.offset + filters.limit) < total_count,
                    facets=facets,
                    search_metadata={
                        "search_time_ms": 0,  # TODO: Implement timing
                        "filters_applied": len([f for f in filters.dict().values() if f]),
                        "cache_hit": False
                    }
                )
                
                # Cache the response
                self._cache_search_result(cache_key, response)
                
                return response
                
        except Exception as e:
            logger.error(
                "Search failed",
                error=str(e),
                filters=filters.dict(),
                user_id=str(user_id) if user_id else None
            )
            raise
    
    async def get_resource_details(
        self,
        resource_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a specific resource"""
        try:
            with self.db_service.get_session() as session:
                resource = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.id == resource_id
                ).first()
                
                if not resource:
                    raise ValueError(f"Resource {resource_id} not found")
                
                # Format detailed response
                detailed_resource = await self._format_resource_response(resource, detailed=True)
                
                # Add user-specific data if available
                if user_id:
                    detailed_resource = await self._add_user_specific_data(
                        detailed_resource, user_id
                    )
                
                # Update view count
                await self._update_resource_metrics(resource_id, "view")
                
                return detailed_resource
                
        except Exception as e:
            logger.error(
                "Failed to get resource details",
                error=str(e),
                resource_id=str(resource_id)
            )
            raise
    
    # ========================================================================
    # TRANSACTIONS AND ORDERS
    # ========================================================================
    
    async def create_purchase_order(
        self,
        buyer_user_id: UUID,
        resource_id: UUID,
        quantity: int = 1,
        pricing_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a purchase order for a marketplace resource"""
        try:
            with self.db_service.get_session() as session:
                # Get resource details
                resource = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.id == resource_id
                ).first()
                
                if not resource:
                    raise ValueError(f"Resource {resource_id} not found")
                
                if resource.status != ResourceStatus.ACTIVE.value:
                    raise ValueError(f"Resource is not available for purchase")
                
                # Calculate pricing
                pricing = await self._calculate_order_pricing(
                    resource, quantity, pricing_options
                )
                
                # Generate order number
                order_number = f"ORD-{uuid4().hex[:8].upper()}"
                
                # Create order record
                order_data = {
                    "id": uuid4(),
                    "order_number": order_number,
                    "buyer_user_id": buyer_user_id,
                    "resource_id": resource_id,
                    "quantity": quantity,
                    "unit_price": pricing["unit_price"],
                    "total_amount": pricing["total_amount"],
                    "platform_fee": pricing["platform_fee"],
                    "order_status": "pending",
                    "payment_status": "pending",
                    "order_metadata": {
                        "pricing_breakdown": pricing,
                        "resource_type": resource.resource_type,
                        "resource_name": resource.name
                    }
                }
                
                order = ResourceOrderDB(**order_data)
                session.add(order)
                session.commit()
                session.refresh(order)
                
                # Process payment with FTNS service
                payment_result = await self._process_ftns_payment(
                    buyer_user_id,
                    pricing["total_amount"],
                    f"Purchase of {resource.name}"
                )
                
                if payment_result["success"]:
                    order.payment_status = "completed"
                    order.order_status = "confirmed"
                    session.commit()
                    
                    # Update resource metrics
                    await self._update_resource_metrics(resource_id, "purchase", quantity)
                    
                    # Grant access to resource
                    access_details = await self._grant_resource_access(
                        buyer_user_id, resource, order
                    )
                    
                    logger.info(
                        "Purchase order completed",
                        order_id=str(order.id),
                        buyer_id=str(buyer_user_id),
                        resource_id=str(resource_id)
                    )
                    
                    return {
                        "order_id": str(order.id),
                        "order_number": order_number,
                        "status": "completed",
                        "total_amount": float(pricing["total_amount"]),
                        "access_details": access_details
                    }
                else:
                    order.payment_status = "failed"
                    session.commit()
                    raise ValueError(f"Payment failed: {payment_result['error']}")
                    
        except Exception as e:
            logger.error(
                "Failed to create purchase order",
                error=str(e),
                buyer_id=str(buyer_user_id),
                resource_id=str(resource_id)
            )
            raise
    
    # ========================================================================
    # REVIEWS AND RATINGS
    # ========================================================================
    
    async def create_review(
        self,
        reviewer_user_id: UUID,
        resource_id: UUID,
        rating: int,
        title: str,
        content: str,
        verified_purchase: bool = False
    ) -> Dict[str, Any]:
        """Create a review for a marketplace resource"""
        try:
            if not (1 <= rating <= 5):
                raise ValueError("Rating must be between 1 and 5")
            
            with self.db_service.get_session() as session:
                # Check if resource exists
                resource = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.id == resource_id
                ).first()
                
                if not resource:
                    raise ValueError(f"Resource {resource_id} not found")
                
                # Check for existing review
                existing_review = session.query(ResourceReviewDB).filter(
                    and_(
                        ResourceReviewDB.resource_id == resource_id,
                        ResourceReviewDB.reviewer_user_id == reviewer_user_id
                    )
                ).first()
                
                if existing_review:
                    raise ValueError("User has already reviewed this resource")
                
                # Create review
                review_data = {
                    "id": uuid4(),
                    "resource_id": resource_id,
                    "reviewer_user_id": reviewer_user_id,
                    "rating": rating,
                    "title": title,
                    "content": content,
                    "verified_purchase": verified_purchase
                }
                
                review = ResourceReviewDB(**review_data)
                session.add(review)
                session.commit()
                session.refresh(review)
                
                # Update resource rating
                await self._update_resource_rating(resource_id)
                
                logger.info(
                    "Review created",
                    review_id=str(review.id),
                    resource_id=str(resource_id),
                    rating=rating
                )
                
                return {
                    "review_id": str(review.id),
                    "rating": rating,
                    "title": title,
                    "created_at": review.created_at.isoformat()
                }
                
        except Exception as e:
            logger.error(
                "Failed to create review",
                error=str(e),
                reviewer_id=str(reviewer_user_id),
                resource_id=str(resource_id)
            )
            raise
    
    # ========================================================================
    # ANALYTICS AND INSIGHTS
    # ========================================================================
    
    async def get_marketplace_stats(self) -> MarketplaceStatsResponse:
        """Get comprehensive marketplace statistics"""
        try:
            # Check cache
            if self._is_stats_cache_valid():
                return self._stats_cache["stats"]
            
            with self.db_service.get_session() as session:
                # Basic counts
                total_resources = session.query(UnifiedResourceListingDB).count()
                
                # Resources by type
                resources_by_type = {}
                for resource_type in ResourceType:
                    count = session.query(UnifiedResourceListingDB).filter(
                        UnifiedResourceListingDB.resource_type == resource_type.value
                    ).count()
                    resources_by_type[resource_type.value] = count
                
                # Provider count
                total_providers = session.query(
                    func.count(func.distinct(UnifiedResourceListingDB.owner_user_id))
                ).scalar()
                
                # Revenue and transaction data
                revenue_data = session.query(
                    func.sum(ResourceOrderDB.total_amount),
                    func.count(ResourceOrderDB.id)
                ).filter(
                    ResourceOrderDB.order_status == "confirmed"
                ).first()
                
                total_revenue = revenue_data[0] or Decimal('0')
                total_transactions = revenue_data[1] or 0
                
                # Quality metrics
                avg_rating = session.query(
                    func.avg(UnifiedResourceListingDB.rating_average)
                ).scalar() or Decimal('0')
                
                verified_count = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.verified == True
                ).count()
                verification_rate = (verified_count / total_resources * 100) if total_resources > 0 else 0
                
                # Growth metrics
                one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
                new_resources_this_month = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.created_at >= one_month_ago
                ).count()
                
                # Top performers
                top_resources = session.query(UnifiedResourceListingDB).order_by(
                    desc(UnifiedResourceListingDB.usage_count)
                ).limit(10).all()
                
                top_providers_query = session.query(
                    UnifiedResourceListingDB.owner_user_id,
                    func.count(UnifiedResourceListingDB.id).label('resource_count'),
                    func.sum(UnifiedResourceListingDB.revenue_total).label('total_revenue')
                ).group_by(
                    UnifiedResourceListingDB.owner_user_id
                ).order_by(
                    desc('total_revenue')
                ).limit(10).all()
                
                # Format response
                stats = MarketplaceStatsResponse(
                    total_resources=total_resources,
                    resources_by_type=resources_by_type,
                    total_providers=total_providers,
                    total_revenue=total_revenue,
                    total_transactions=total_transactions,
                    active_users=0,  # TODO: Implement active user tracking
                    average_rating=avg_rating,
                    verification_rate=Decimal(str(verification_rate)),
                    new_resources_this_month=new_resources_this_month,
                    revenue_growth_rate=Decimal('0'),  # TODO: Calculate growth rate
                    top_resources=[
                        {
                            "id": str(r.id),
                            "name": r.name,
                            "type": r.resource_type,
                            "usage_count": r.usage_count,
                            "rating": float(r.rating_average)
                        }
                        for r in top_resources
                    ],
                    top_providers=[
                        {
                            "user_id": str(p.owner_user_id),
                            "resource_count": p.resource_count,
                            "total_revenue": float(p.total_revenue or 0)
                        }
                        for p in top_providers_query
                    ],
                    trending_categories=[]  # TODO: Implement trending analysis
                )
                
                # Cache the stats
                self._stats_cache = {
                    "stats": stats,
                    "timestamp": datetime.now(timezone.utc)
                }
                
                return stats
                
        except Exception as e:
            logger.error("Failed to get marketplace stats", error=str(e))
            raise
    
    # ========================================================================
    # RESOURCE-SPECIFIC HANDLERS
    # ========================================================================
    
    async def _handle_dataset(self, data: DatasetListing, operation: str) -> Dict[str, Any]:
        """Handle dataset-specific processing"""
        return {
            "metadata": {
                "category": data.category.value,
                "size_bytes": data.size_bytes,
                "record_count": data.record_count,
                "data_format": data.data_format.value,
                "license_type": data.license_type.value,
                "quality_scores": {
                    "completeness": float(data.completeness_score),
                    "accuracy": float(data.accuracy_score),
                    "consistency": float(data.consistency_score)
                }
            },
            "technical_specs": {
                "schema_definition": data.schema_definition,
                "feature_count": data.feature_count,
                "privacy_compliance": data.privacy_compliance,
                "ethical_review_status": data.ethical_review_status
            },
            "access_config": {
                "access_url": data.access_url,
                "sample_data_url": data.sample_data_url,
                "preprocessing_scripts": data.preprocessing_scripts
            }
        }
    
    async def _handle_agent_workflow(self, data: AgentWorkflowListing, operation: str) -> Dict[str, Any]:
        """Handle agent workflow-specific processing"""
        return {
            "metadata": {
                "agent_type": data.agent_type.value,
                "capabilities": [cap.value for cap in data.capabilities],
                "performance": {
                    "success_rate": float(data.success_rate),
                    "avg_execution_time": float(data.average_execution_time or 0),
                    "accuracy_score": float(data.accuracy_score or 0)
                }
            },
            "technical_specs": {
                "input_types": data.input_types,
                "output_types": data.output_types,
                "max_execution_time": data.max_execution_time,
                "memory_requirements": data.memory_requirements,
                "required_tools": data.required_tools,
                "required_models": data.required_models,
                "environment_requirements": data.environment_requirements
            },
            "access_config": {
                "deployment_url": data.deployment_url,
                "source_code_url": data.source_code_url,
                "workflow_config": data.workflow_config,
                "example_usage": data.example_usage
            }
        }
    
    async def _handle_compute_resource(self, data: ComputeResourceListing, operation: str) -> Dict[str, Any]:
        """Handle compute resource-specific processing"""
        return {
            "metadata": {
                "resource_type": data.resource_type.value,
                "capabilities": [cap.value for cap in data.capabilities],
                "performance": {
                    "uptime_percentage": float(data.uptime_percentage),
                    "average_latency_ms": float(data.average_latency_ms or 0),
                    "throughput_ops_per_sec": data.throughput_ops_per_sec or 0
                }
            },
            "technical_specs": {
                "hardware": {
                    "cpu_cores": data.cpu_cores,
                    "memory_gb": data.memory_gb,
                    "storage_gb": data.storage_gb,
                    "gpu_count": data.gpu_count,
                    "gpu_model": data.gpu_model,
                    "network_bandwidth_gbps": float(data.network_bandwidth_gbps or 0)
                },
                "supported_frameworks": data.supported_frameworks,
                "operating_systems": data.operating_systems,
                "geographic_regions": data.geographic_regions,
                "auto_scaling_enabled": data.auto_scaling_enabled
            },
            "access_config": {
                "access_url": data.access_url,
                "api_endpoint": data.api_endpoint,
                "configuration_template": data.configuration_template,
                "security_features": data.security_features,
                "availability_schedule": data.availability_schedule
            }
        }
    
    async def _handle_knowledge_resource(self, data: KnowledgeResourceListing, operation: str) -> Dict[str, Any]:
        """Handle knowledge resource-specific processing"""
        return {
            "metadata": {
                "resource_type": data.resource_type.value,
                "domain": data.domain.value,
                "quality_scores": {
                    "completeness": float(data.completeness_score),
                    "accuracy": float(data.accuracy_score),
                    "consistency": float(data.consistency_score),
                    "expert_validated": data.expert_validation
                },
                "content_stats": {
                    "entity_count": data.entity_count or 0,
                    "relation_count": data.relation_count or 0,
                    "fact_count": data.fact_count or 0
                }
            },
            "technical_specs": {
                "format_type": data.format_type,
                "query_languages": data.query_languages,
                "reasoning_capabilities": data.reasoning_capabilities,
                "update_frequency": data.update_frequency,
                "coverage_scope": data.coverage_scope
            },
            "access_config": {
                "access_url": data.access_url,
                "api_endpoint": data.api_endpoint,
                "sparql_endpoint": data.sparql_endpoint,
                "integration_examples": data.integration_examples
            }
        }
    
    async def _handle_evaluation_service(self, data: EvaluationServiceListing, operation: str) -> Dict[str, Any]:
        """Handle evaluation service-specific processing"""
        return {
            "metadata": {
                "service_type": data.service_type.value,
                "evaluation_metrics": [metric.value for metric in data.evaluation_metrics],
                "quality": {
                    "benchmark_validity": data.benchmark_validity,
                    "peer_reviewed": data.peer_reviewed,
                    "reproducibility_score": float(data.reproducibility_score),
                    "validation_count": data.validation_count
                }
            },
            "technical_specs": {
                "supported_models": data.supported_models,
                "test_datasets": data.test_datasets,
                "evaluation_protocols": data.evaluation_protocols,
                "supported_frameworks": data.supported_frameworks,
                "average_evaluation_time": data.average_evaluation_time,
                "max_model_size": data.max_model_size
            },
            "access_config": {
                "service_url": data.service_url,
                "api_endpoint": data.api_endpoint,
                "example_reports": data.example_reports
            }
        }
    
    async def _handle_training_service(self, data: TrainingServiceListing, operation: str) -> Dict[str, Any]:
        """Handle training service-specific processing"""
        return {
            "metadata": {
                "service_type": data.service_type.value,
                "performance": {
                    "success_rate": float(data.success_rate),
                    "average_improvement": float(data.average_improvement or 0),
                    "client_satisfaction": float(data.client_satisfaction)
                }
            },
            "technical_specs": {
                "supported_frameworks": [fw.value for fw in data.supported_frameworks],
                "supported_architectures": data.supported_architectures,
                "max_model_parameters": data.max_model_parameters,
                "supported_data_types": data.supported_data_types,
                "max_training_time": data.max_training_time,
                "available_compute": data.available_compute,
                "distributed_training": data.distributed_training,
                "automated_tuning": data.automated_tuning
            },
            "access_config": {
                "service_url": data.service_url,
                "api_endpoint": data.api_endpoint,
                "portfolio_examples": data.portfolio_examples
            }
        }
    
    async def _handle_safety_tool(self, data: SafetyToolListing, operation: str) -> Dict[str, Any]:
        """Handle safety tool-specific processing"""
        return {
            "metadata": {
                "tool_type": data.tool_type.value,
                "compliance_standards": [std.value for std in data.compliance_standards],
                "validation": {
                    "third_party_validated": data.third_party_validated,
                    "certification_bodies": data.certification_bodies,
                    "audit_trail_support": data.audit_trail_support
                },
                "performance": {
                    "detection_accuracy": float(data.detection_accuracy),
                    "false_positive_rate": float(data.false_positive_rate),
                    "processing_speed": data.processing_speed
                }
            },
            "technical_specs": {
                "supported_models": data.supported_models,
                "detection_capabilities": data.detection_capabilities,
                "reporting_formats": data.reporting_formats
            },
            "access_config": {
                "tool_url": data.tool_url,
                "api_endpoint": data.api_endpoint,
                "compliance_reports": data.compliance_reports
            }
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _apply_search_filters(self, query, filters: UnifiedSearchFilters):
        """Apply search filters to query"""
        # Resource type filtering
        if filters.resource_types:
            query = query.filter(
                UnifiedResourceListingDB.resource_type.in_([rt.value for rt in filters.resource_types])
            )
        
        # Status filtering (only show active resources by default)
        query = query.filter(UnifiedResourceListingDB.status == ResourceStatus.ACTIVE.value)
        
        # Pricing filtering
        if filters.pricing_models:
            query = query.filter(
                UnifiedResourceListingDB.pricing_model.in_([pm.value for pm in filters.pricing_models])
            )
        
        if filters.min_price is not None:
            query = query.filter(UnifiedResourceListingDB.base_price >= filters.min_price)
        
        if filters.max_price is not None:
            query = query.filter(UnifiedResourceListingDB.base_price <= filters.max_price)
        
        # Quality filtering
        if filters.quality_grades:
            query = query.filter(
                UnifiedResourceListingDB.quality_grade.in_([qg.value for qg in filters.quality_grades])
            )
        
        if filters.min_rating is not None:
            query = query.filter(UnifiedResourceListingDB.rating_average >= filters.min_rating)
        
        if filters.verified_only:
            query = query.filter(UnifiedResourceListingDB.verified == True)
        
        if filters.featured_only:
            query = query.filter(UnifiedResourceListingDB.featured == True)
        
        # Text search
        if filters.search_query:
            search_term = f"%{filters.search_query}%"
            query = query.filter(
                or_(
                    UnifiedResourceListingDB.name.ilike(search_term),
                    UnifiedResourceListingDB.description.ilike(search_term),
                    UnifiedResourceListingDB.tags.op('@>')([filters.search_query.lower()])
                )
            )
        
        # Provider filtering
        if filters.provider_name:
            query = query.filter(
                UnifiedResourceListingDB.provider_name.ilike(f"%{filters.provider_name}%")
            )
        
        # Tags filtering
        if filters.tags:
            for tag in filters.tags:
                query = query.filter(
                    UnifiedResourceListingDB.tags.op('@>')([tag.lower()])
                )
        
        return query
    
    def _apply_sorting(self, query, sort_by: str, sort_order: str):
        """Apply sorting to query"""
        sort_column = {
            "popularity": UnifiedResourceListingDB.usage_count,
            "price": UnifiedResourceListingDB.base_price,
            "created_at": UnifiedResourceListingDB.created_at,
            "rating": UnifiedResourceListingDB.rating_average,
            "name": UnifiedResourceListingDB.name,
            "usage_count": UnifiedResourceListingDB.usage_count
        }.get(sort_by, UnifiedResourceListingDB.usage_count)
        
        if sort_order == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        return query
    
    async def _format_resource_response(self, resource: UnifiedResourceListingDB, detailed: bool = False) -> Dict[str, Any]:
        """Format database resource for API response"""
        base_data = {
            "id": str(resource.id),
            "resource_type": resource.resource_type,
            "name": resource.name,
            "description": resource.description,
            "status": resource.status,
            "quality_grade": resource.quality_grade,
            "pricing_model": resource.pricing_model,
            "base_price": float(resource.base_price),
            "rating_average": float(resource.rating_average),
            "rating_count": resource.rating_count,
            "usage_count": resource.usage_count,
            "tags": resource.tags or [],
            "provider_name": resource.provider_name,
            "created_at": resource.created_at.isoformat(),
            "updated_at": resource.updated_at.isoformat()
        }
        
        if detailed:
            base_data.update({
                "resource_metadata": resource.resource_metadata or {},
                "technical_specs": resource.technical_specs or {},
                "access_config": resource.access_config or {},
                "documentation_url": resource.documentation_url,
                "revenue_total": float(resource.revenue_total),
                "featured": resource.featured,
                "verified": resource.verified
            })
        
        return base_data
    
    async def _calculate_order_pricing(
        self,
        resource: UnifiedResourceListingDB,
        quantity: int,
        pricing_options: Optional[Dict[str, Any]]
    ) -> Dict[str, Decimal]:
        """Calculate order pricing with platform fees"""
        unit_price = resource.base_price
        subtotal = unit_price * quantity
        platform_fee = subtotal * self.platform_fee_percentage
        total_amount = subtotal + platform_fee
        
        return {
            "unit_price": unit_price,
            "quantity": quantity,
            "subtotal": subtotal,
            "platform_fee": platform_fee,
            "total_amount": total_amount
        }
    
    async def _process_ftns_payment(
        self,
        user_id: UUID,
        amount: Decimal,
        description: str
    ) -> Dict[str, Any]:
        """Process payment through FTNS service"""
        try:
            # Check user balance
            balance = await self.ftns_service.get_user_balance(str(user_id))
            if balance.balance < amount:
                return {
                    "success": False,
                    "error": f"Insufficient balance. Required: {amount}, Available: {balance.balance}"
                }
            
            # Process payment
            transaction = await self.ftns_service.transfer_tokens(
                from_user_id=str(user_id),
                to_user_id="marketplace",  # Platform account
                amount=amount,
                transaction_type="marketplace_purchase",
                description=description
            )
            
            return {"success": True, "transaction_id": transaction.transaction_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _grant_resource_access(
        self,
        user_id: UUID,
        resource: UnifiedResourceListingDB,
        order: ResourceOrderDB
    ) -> Dict[str, Any]:
        """Grant user access to purchased resource"""
        # Implementation depends on resource type
        access_details = {
            "access_granted": True,
            "access_type": resource.pricing_model,
            "valid_until": None,  # Set based on pricing model
            "access_instructions": "Check documentation for usage details"
        }
        
        # Resource-specific access granting logic would go here
        # For now, return basic access details
        
        return access_details
    
    async def _update_resource_metrics(
        self,
        resource_id: UUID,
        metric_type: str,
        value: int = 1
    ):
        """Update resource usage metrics"""
        try:
            with self.db_service.get_session() as session:
                resource = session.query(UnifiedResourceListingDB).filter(
                    UnifiedResourceListingDB.id == resource_id
                ).first()
                
                if resource:
                    if metric_type == "view":
                        # Don't update view count in database, just track elsewhere if needed
                        pass
                    elif metric_type == "purchase":
                        resource.usage_count += value
                        resource.last_used_at = datetime.now(timezone.utc)
                    
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update resource metrics: {e}")
    
    async def _update_resource_rating(self, resource_id: UUID):
        """Update resource average rating after new review"""
        try:
            with self.db_service.get_session() as session:
                # Calculate new average rating
                rating_stats = session.query(
                    func.avg(ResourceReviewDB.rating),
                    func.count(ResourceReviewDB.id)
                ).filter(
                    ResourceReviewDB.resource_id == resource_id
                ).first()
                
                if rating_stats[0] is not None:
                    avg_rating = Decimal(str(rating_stats[0]))
                    rating_count = rating_stats[1]
                    
                    # Update resource
                    resource = session.query(UnifiedResourceListingDB).filter(
                        UnifiedResourceListingDB.id == resource_id
                    ).first()
                    
                    if resource:
                        resource.rating_average = avg_rating
                        resource.rating_count = rating_count
                        session.commit()
                        
        except Exception as e:
            logger.error(f"Failed to update resource rating: {e}")
    
    # Cache management methods
    def _generate_search_cache_key(self, filters: UnifiedSearchFilters, user_id: Optional[UUID]) -> str:
        """Generate cache key for search results"""
        import hashlib
        filter_str = str(sorted(filters.dict().items()))
        user_str = str(user_id) if user_id else "anonymous"
        return hashlib.sha256(f"{filter_str}-{user_str}".encode()).hexdigest()
    
    def _get_cached_search(self, cache_key: str) -> Optional[MarketplaceSearchResponse]:
        """Get cached search results if valid"""
        if cache_key in self._search_cache:
            cache_entry = self._search_cache[cache_key]
            if datetime.now(timezone.utc) - cache_entry["timestamp"] < self._cache_ttl:
                return cache_entry["result"]
        return None
    
    def _cache_search_result(self, cache_key: str, result: MarketplaceSearchResponse):
        """Cache search results"""
        self._search_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now(timezone.utc)
        }
    
    def _is_stats_cache_valid(self) -> bool:
        """Check if stats cache is still valid"""
        if "stats" not in self._stats_cache:
            return False
        
        cache_time = self._stats_cache.get("timestamp")
        if not cache_time:
            return False
        
        return datetime.now(timezone.utc) - cache_time < self._cache_ttl
    
    async def _generate_search_facets(self, session: Session, filters: UnifiedSearchFilters) -> Dict[str, Any]:
        """Generate faceted search navigation"""
        # Implementation for faceted search
        return {
            "resource_types": {},
            "pricing_models": {},
            "quality_grades": {},
            "price_ranges": {}
        }
    
    async def _add_personalization(self, resource: Dict[str, Any], user_id: UUID) -> Dict[str, Any]:
        """Add personalized data to resource"""
        # Add user-specific information like purchase history, bookmarks, etc.
        return resource
    
    async def _add_user_specific_data(self, resource: Dict[str, Any], user_id: UUID) -> Dict[str, Any]:
        """Add user-specific data for detailed view"""
        # Add information like access status, purchase history, etc.
        return resource


# Global service instance
expanded_marketplace_service = ExpandedMarketplaceService()


def get_expanded_marketplace_service() -> ExpandedMarketplaceService:
    """Get the expanded marketplace service instance"""
    return expanded_marketplace_service