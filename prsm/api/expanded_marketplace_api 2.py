"""
PRSM Expanded Marketplace API
=============================

Comprehensive REST API for all marketplace resource types:
- Datasets, Agent Workflows, Compute Resources
- Knowledge Resources, Evaluation Services
- Training Services, Safety Tools

Provides unified endpoints for discovery, transactions, reviews,
and marketplace management across the complete PRSM AI ecosystem.
"""

from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.auth import get_current_user
from ..marketplace.expanded_marketplace_service import get_expanded_marketplace_service, ExpandedMarketplaceService
from ..marketplace.expanded_models import (
    # Resource types and base models
    ResourceType, UnifiedSearchFilters, MarketplaceSearchResponse, MarketplaceStatsResponse,
    
    # Specific resource models
    DatasetListing, AgentWorkflowListing, ComputeResourceListing,
    KnowledgeResourceListing, EvaluationServiceListing,
    TrainingServiceListing, SafetyToolListing,
    
    # Request/response models
    DatasetCategory, AgentType, ComputeResourceType, KnowledgeResourceType,
    EvaluationServiceType, TrainingServiceType, SafetyToolType,
    PricingModel, QualityGrade
)

router = APIRouter(prefix="/api/v1/marketplace", tags=["Expanded Marketplace"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateResourceRequest(BaseModel):
    """Unified request model for creating any resource type"""
    resource_type: ResourceType
    resource_data: Dict[str, Any]


class UpdateResourceRequest(BaseModel):
    """Request model for updating resources"""
    updates: Dict[str, Any]


class PurchaseRequest(BaseModel):
    """Request model for purchasing resources"""
    resource_id: UUID
    quantity: int = Field(default=1, ge=1)
    pricing_options: Optional[Dict[str, Any]] = None


class ReviewRequest(BaseModel):
    """Request model for creating reviews"""
    rating: int = Field(ge=1, le=5)
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=10, max_length=2000)


class ResourceResponse(BaseModel):
    """Generic resource response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# ============================================================================
# DATASET MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/datasets", response_model=ResourceResponse)
async def create_dataset_listing(
    dataset_data: DatasetListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new dataset listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.DATASET,
            resource_data=dataset_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Dataset listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/categories")
async def get_dataset_categories():
    """Get all available dataset categories"""
    return {
        "categories": [
            {
                "value": category.value,
                "label": category.value.replace("_", " ").title(),
                "description": f"Dataset category: {category.value}"
            }
            for category in DatasetCategory
        ]
    }


@router.get("/datasets/search")
async def search_datasets(
    category: Optional[DatasetCategory] = Query(None),
    min_size: Optional[int] = Query(None, ge=0),
    max_size: Optional[int] = Query(None, ge=0),
    data_format: Optional[str] = Query(None),
    license_type: Optional[str] = Query(None),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity", pattern=r'^(popularity|price|created_at|rating|size|name)$'),
    sort_order: str = Query("desc", pattern=r'^(asc|desc)$'),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter datasets"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.DATASET],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# AGENT WORKFLOW MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/agent-workflows", response_model=ResourceResponse)
async def create_agent_workflow_listing(
    workflow_data: AgentWorkflowListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new agent workflow listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.AGENT_WORKFLOW,
            resource_data=workflow_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Agent workflow listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agent-workflows/types")
async def get_agent_types():
    """Get all available agent types"""
    return {
        "agent_types": [
            {
                "value": agent_type.value,
                "label": agent_type.value.replace("_", " ").title(),
                "description": f"Agent type: {agent_type.value}"
            }
            for agent_type in AgentType
        ]
    }


@router.get("/agent-workflows/search")
async def search_agent_workflows(
    agent_type: Optional[AgentType] = Query(None),
    capabilities: Optional[List[str]] = Query(None),
    min_success_rate: Optional[float] = Query(None, ge=0, le=1),
    max_execution_time: Optional[int] = Query(None, ge=1),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter agent workflows"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.AGENT_WORKFLOW],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# COMPUTE RESOURCE MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/compute-resources", response_model=ResourceResponse)
async def create_compute_resource_listing(
    compute_data: ComputeResourceListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new compute resource listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.COMPUTE_RESOURCE,
            resource_data=compute_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Compute resource listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compute-resources/types")
async def get_compute_resource_types():
    """Get all available compute resource types"""
    return {
        "resource_types": [
            {
                "value": resource_type.value,
                "label": resource_type.value.replace("_", " ").title(),
                "description": f"Compute resource: {resource_type.value}"
            }
            for resource_type in ComputeResourceType
        ]
    }


@router.get("/compute-resources/search")
async def search_compute_resources(
    resource_type: Optional[ComputeResourceType] = Query(None),
    min_cpu_cores: Optional[int] = Query(None, ge=1),
    min_memory_gb: Optional[int] = Query(None, ge=1),
    gpu_required: Optional[bool] = Query(None),
    geographic_region: Optional[str] = Query(None),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter compute resources"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.COMPUTE_RESOURCE],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# KNOWLEDGE RESOURCE MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/knowledge-resources", response_model=ResourceResponse)
async def create_knowledge_resource_listing(
    knowledge_data: KnowledgeResourceListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new knowledge resource listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.KNOWLEDGE_RESOURCE,
            resource_data=knowledge_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Knowledge resource listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/knowledge-resources/types")
async def get_knowledge_resource_types():
    """Get all available knowledge resource types"""
    return {
        "resource_types": [
            {
                "value": resource_type.value,
                "label": resource_type.value.replace("_", " ").title(),
                "description": f"Knowledge resource: {resource_type.value}"
            }
            for resource_type in KnowledgeResourceType
        ]
    }


@router.get("/knowledge-resources/search")
async def search_knowledge_resources(
    resource_type: Optional[KnowledgeResourceType] = Query(None),
    domain: Optional[str] = Query(None),
    format_type: Optional[str] = Query(None),
    expert_validated: Optional[bool] = Query(None),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter knowledge resources"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.KNOWLEDGE_RESOURCE],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# EVALUATION SERVICE MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/evaluation-services", response_model=ResourceResponse)
async def create_evaluation_service_listing(
    evaluation_data: EvaluationServiceListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new evaluation service listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.EVALUATION_SERVICE,
            resource_data=evaluation_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Evaluation service listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/evaluation-services/types")
async def get_evaluation_service_types():
    """Get all available evaluation service types"""
    return {
        "service_types": [
            {
                "value": service_type.value,
                "label": service_type.value.replace("_", " ").title(),
                "description": f"Evaluation service: {service_type.value}"
            }
            for service_type in EvaluationServiceType
        ]
    }


@router.get("/evaluation-services/search")
async def search_evaluation_services(
    service_type: Optional[EvaluationServiceType] = Query(None),
    peer_reviewed: Optional[bool] = Query(None),
    benchmark_validity: Optional[bool] = Query(None),
    supported_frameworks: Optional[List[str]] = Query(None),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter evaluation services"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.EVALUATION_SERVICE],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# TRAINING SERVICE MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/training-services", response_model=ResourceResponse)
async def create_training_service_listing(
    training_data: TrainingServiceListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new training service listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.TRAINING_SERVICE,
            resource_data=training_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Training service listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/training-services/types")
async def get_training_service_types():
    """Get all available training service types"""
    return {
        "service_types": [
            {
                "value": service_type.value,
                "label": service_type.value.replace("_", " ").title(),
                "description": f"Training service: {service_type.value}"
            }
            for service_type in TrainingServiceType
        ]
    }


@router.get("/training-services/search")
async def search_training_services(
    service_type: Optional[TrainingServiceType] = Query(None),
    supported_frameworks: Optional[List[str]] = Query(None),
    distributed_training: Optional[bool] = Query(None),
    automated_tuning: Optional[bool] = Query(None),
    min_success_rate: Optional[float] = Query(None, ge=0, le=1),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter training services"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.TRAINING_SERVICE],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# SAFETY TOOL MARKETPLACE ENDPOINTS
# ============================================================================

@router.post("/safety-tools", response_model=ResourceResponse)
async def create_safety_tool_listing(
    safety_data: SafetyToolListing,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a new safety tool listing"""
    try:
        result = await marketplace_service.create_resource_listing(
            resource_type=ResourceType.SAFETY_TOOL,
            resource_data=safety_data,
            owner_user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Safety tool listing created successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/safety-tools/types")
async def get_safety_tool_types():
    """Get all available safety tool types"""
    return {
        "tool_types": [
            {
                "value": tool_type.value,
                "label": tool_type.value.replace("_", " ").title(),
                "description": f"Safety tool: {tool_type.value}"
            }
            for tool_type in SafetyToolType
        ]
    }


@router.get("/safety-tools/search")
async def search_safety_tools(
    tool_type: Optional[SafetyToolType] = Query(None),
    compliance_standards: Optional[List[str]] = Query(None),
    third_party_validated: Optional[bool] = Query(None),
    audit_trail_support: Optional[bool] = Query(None),
    min_detection_accuracy: Optional[float] = Query(None, ge=0, le=1),
    search_query: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Search and filter safety tools"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=[ResourceType.SAFETY_TOOL],
            search_query=search_query,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# UNIFIED MARKETPLACE ENDPOINTS
# ============================================================================

@router.get("/search")
async def unified_marketplace_search(
    resource_types: Optional[List[ResourceType]] = Query(None),
    pricing_models: Optional[List[PricingModel]] = Query(None),
    quality_grades: Optional[List[QualityGrade]] = Query(None),
    min_price: Optional[Decimal] = Query(None, ge=0),
    max_price: Optional[Decimal] = Query(None, ge=0),
    min_rating: Optional[Decimal] = Query(None, ge=0, le=5),
    verified_only: bool = Query(False),
    featured_only: bool = Query(False),
    tags: Optional[List[str]] = Query(None),
    search_query: Optional[str] = Query(None),
    provider_name: Optional[str] = Query(None),
    sort_by: str = Query("popularity"),
    sort_order: str = Query("desc"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Unified search across all marketplace resources"""
    try:
        filters = UnifiedSearchFilters(
            resource_types=resource_types,
            pricing_models=pricing_models,
            quality_grades=quality_grades,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            verified_only=verified_only,
            featured_only=featured_only,
            tags=tags,
            search_query=search_query,
            provider_name=provider_name,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        user_id = UUID(current_user["user_id"]) if current_user else None
        results = await marketplace_service.search_resources(filters, user_id)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/resources/{resource_id}")
async def get_resource_details(
    resource_id: UUID = Path(..., description="Resource ID"),
    current_user: Optional[Dict] = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Get detailed information about a specific resource"""
    try:
        user_id = UUID(current_user["user_id"]) if current_user else None
        resource = await marketplace_service.get_resource_details(resource_id, user_id)
        
        return resource
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/resources/{resource_id}")
async def update_resource(
    resource_id: UUID = Path(..., description="Resource ID"),
    update_request: UpdateResourceRequest = Body(...),
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Update an existing resource listing"""
    try:
        result = await marketplace_service.update_resource_listing(
            resource_id=resource_id,
            updates=update_request.updates,
            user_id=UUID(current_user["user_id"])
        )
        
        return ResourceResponse(
            success=True,
            message="Resource updated successfully",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# TRANSACTION ENDPOINTS
# ============================================================================

@router.post("/purchase")
async def purchase_resource(
    purchase_request: PurchaseRequest,
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Purchase a marketplace resource"""
    try:
        result = await marketplace_service.create_purchase_order(
            buyer_user_id=UUID(current_user["user_id"]),
            resource_id=purchase_request.resource_id,
            quantity=purchase_request.quantity,
            pricing_options=purchase_request.pricing_options
        )
        
        return ResourceResponse(
            success=True,
            message="Purchase completed successfully",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REVIEW ENDPOINTS
# ============================================================================

@router.post("/resources/{resource_id}/reviews")
async def create_review(
    resource_id: UUID = Path(..., description="Resource ID"),
    review_request: ReviewRequest = Body(...),
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Create a review for a resource"""
    try:
        result = await marketplace_service.create_review(
            reviewer_user_id=UUID(current_user["user_id"]),
            resource_id=resource_id,
            rating=review_request.rating,
            title=review_request.title,
            content=review_request.content,
            verified_purchase=False  # TODO: Check purchase history
        )
        
        return ResourceResponse(
            success=True,
            message="Review created successfully",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@router.get("/stats", response_model=MarketplaceStatsResponse)
async def get_marketplace_stats(
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Get comprehensive marketplace statistics"""
    try:
        stats = await marketplace_service.get_marketplace_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_all_categories():
    """Get all available categories across all resource types"""
    return {
        "resource_types": [
            {
                "value": rt.value,
                "label": rt.value.replace("_", " ").title(),
                "description": f"Resource type: {rt.value}"
            }
            for rt in ResourceType
        ],
        "pricing_models": [
            {
                "value": pm.value,
                "label": pm.value.replace("_", " ").title(),
                "description": f"Pricing model: {pm.value}"
            }
            for pm in PricingModel
        ],
        "quality_grades": [
            {
                "value": qg.value,
                "label": qg.value.title(),
                "description": f"Quality grade: {qg.value}"
            }
            for qg in QualityGrade
        ]
    }


# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@router.get("/recommendations")
async def get_personalized_recommendations(
    resource_types: Optional[List[ResourceType]] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    current_user: Dict = Depends(get_current_user),
    marketplace_service: ExpandedMarketplaceService = Depends(get_expanded_marketplace_service)
):
    """Get personalized resource recommendations"""
    try:
        # TODO: Implement recommendation engine
        # For now, return popular resources
        filters = UnifiedSearchFilters(
            resource_types=resource_types,
            featured_only=True,
            sort_by="popularity",
            sort_order="desc",
            limit=limit,
            offset=0
        )
        
        user_id = UUID(current_user["user_id"])
        results = await marketplace_service.search_resources(filters, user_id)
        
        return {
            "recommendations": results.resources,
            "recommendation_reason": "Popular and featured resources",
            "personalization_score": 0.5  # TODO: Implement personalization
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MARKETPLACE HEALTH ENDPOINTS
# ============================================================================

@router.get("/health")
async def marketplace_health_check():
    """Check marketplace service health"""
    try:
        # TODO: Implement actual health checks
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "services": {
                "database": "healthy",
                "ftns_service": "healthy",
                "search_engine": "healthy"
            },
            "version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unavailable")


# Export the router
__all__ = ["router"]