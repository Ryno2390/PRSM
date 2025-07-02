"""
Example Standardized Router for PRSM API
Demonstrates how to implement API endpoints following PRSM standards
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, Path, Body, Depends, status
from pydantic import BaseModel, Field

from ..standards import APIConfig, BaseAPIResponse, PaginatedResponse, PaginationParams
from ..exceptions import (
    raise_not_found, raise_validation_error, raise_forbidden,
    NotFoundException, ValidationException
)
from ..dependencies import (
    StandardAuth, RateLimitedAuth, RequireModelRead, RequireModelCreate,
    get_pagination_params, create_endpoint_dependencies
)
from prsm.auth.models import User, Permission


# === Request/Response Models ===

class ModelCreateRequest(BaseModel):
    """Standardized request model for creating models"""
    name: str = Field(..., min_length=1, max_length=255, description="Model name")
    description: str = Field(..., min_length=1, max_length=1000, description="Model description")
    model_type: str = Field(..., regex="^(teacher|student|specialist|general)$", description="Model type")
    specialization: Optional[str] = Field(None, max_length=255, description="Model specialization")
    performance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Performance score")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ModelUpdateRequest(BaseModel):
    """Standardized request model for updating models"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Model name")
    description: Optional[str] = Field(None, min_length=1, max_length=1000, description="Model description")
    specialization: Optional[str] = Field(None, max_length=255, description="Model specialization")
    performance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Performance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelResponse(BaseAPIResponse):
    """Standardized response model for model data"""
    model_id: UUID = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    model_type: str = Field(..., description="Model type")
    specialization: Optional[str] = Field(None, description="Model specialization")
    performance_score: Optional[float] = Field(None, description="Performance score")
    owner_id: str = Field(..., description="Model owner identifier")
    is_active: bool = Field(..., description="Whether model is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelListResponse(PaginatedResponse):
    """Standardized paginated response for model lists"""
    data: List[ModelResponse] = Field(..., description="List of models")


class ModelSearchRequest(BaseModel):
    """Standardized request model for model search"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    model_type: Optional[str] = Field(None, regex="^(teacher|student|specialist|general)$", description="Filter by model type")
    specialization: Optional[str] = Field(None, max_length=255, description="Filter by specialization")
    min_performance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum performance score")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")


# === Router Configuration ===

router = APIRouter(
    prefix=f"{APIConfig.API_PREFIX}/models",
    tags=["Models"],
    responses={
        400: {"description": "Validation Error"},
        401: {"description": "Authentication Required"},
        403: {"description": "Insufficient Permissions"},
        404: {"description": "Resource Not Found"},
        429: {"description": "Rate Limit Exceeded"},
        500: {"description": "Internal Server Error"}
    }
)


# === Endpoints ===

@router.post(
    "/",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Model",
    description="Create a new AI model with validation and permissions",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_CREATE],
        rate_limited=True
    )
)
async def create_model(
    request: ModelCreateRequest = Body(..., description="Model creation data"),
    user: User = Depends(RequireModelCreate)
) -> ModelResponse:
    """
    Create a new AI model with comprehensive validation
    
    This endpoint demonstrates:
    - Standardized request/response models
    - Proper permission checking
    - Rate limiting
    - Input validation
    - Error handling
    """
    
    try:
        # Validate business rules
        if request.model_type == "teacher" and not request.specialization:
            raise_validation_error(
                "Teacher models must have a specialization",
                {"specialization": ["This field is required for teacher models"]}
            )
        
        # Simulate model creation (replace with actual implementation)
        model_id = uuid4()
        
        # In real implementation, save to database
        # model = await ModelQueries.create_model({
        #     "model_id": model_id,
        #     "name": request.name,
        #     "description": request.description,
        #     "model_type": request.model_type,
        #     "specialization": request.specialization,
        #     "performance_score": request.performance_score,
        #     "owner_id": user.id,
        #     "metadata": request.metadata
        # })
        
        # Simulate created model
        now = datetime.utcnow()
        model_response = ModelResponse(
            model_id=model_id,
            name=request.name,
            description=request.description,
            model_type=request.model_type,
            specialization=request.specialization,
            performance_score=request.performance_score,
            owner_id=str(user.id),
            is_active=True,
            created_at=now,
            updated_at=now,
            metadata=request.metadata
        )
        
        return model_response
        
    except ValidationException:
        raise
    except Exception as e:
        # Log error and raise generic exception
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create model: {str(e)}")
        raise


@router.get(
    "/",
    response_model=ModelListResponse,
    summary="List Models",
    description="Get paginated list of models with filtering options",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_READ],
        rate_limited=True
    )
)
async def list_models(
    model_type: Optional[str] = Query(None, regex="^(teacher|student|specialist|general)$", description="Filter by model type"),
    specialization: Optional[str] = Query(None, max_length=255, description="Filter by specialization"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    pagination: PaginationParams = Depends(get_pagination_params),
    user: User = Depends(RequireModelRead)
) -> ModelListResponse:
    """
    Get paginated list of models with filtering
    
    This endpoint demonstrates:
    - Pagination with standardized parameters
    - Query parameter validation
    - Filtering capabilities
    - Standardized response format
    """
    
    try:
        # Build filter criteria
        filters = {}
        if model_type:
            filters["model_type"] = model_type
        if specialization:
            filters["specialization"] = specialization
        if is_active is not None:
            filters["is_active"] = is_active
        
        # In real implementation, query database with filters and pagination
        # results = await ModelQueries.list_models(
        #     filters=filters,
        #     offset=pagination["offset"],
        #     limit=pagination["page_size"],
        #     sort_by=pagination["sort_by"],
        #     sort_order=pagination["sort_order"]
        # )
        
        # Simulate results
        total_items = 42  # Would come from database count
        sample_models = [
            ModelResponse(
                model_id=uuid4(),
                name=f"Sample Model {i}",
                description=f"Description for model {i}",
                model_type="teacher",
                specialization="nlp",
                performance_score=0.85 + (i * 0.01),
                owner_id=str(user.id),
                is_active=True,
                created_at=datetime.utcnow(),
                metadata={}
            )
            for i in range(1, min(pagination["page_size"] + 1, 6))
        ]
        
        # Calculate pagination metadata
        total_pages = (total_items + pagination["page_size"] - 1) // pagination["page_size"]
        
        pagination_metadata = {
            "page": pagination["page"],
            "page_size": pagination["page_size"],
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": pagination["page"] < total_pages,
            "has_previous": pagination["page"] > 1
        }
        
        return ModelListResponse(
            data=sample_models,
            pagination=pagination_metadata
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to list models: {str(e)}")
        raise


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get Model",
    description="Get detailed information about a specific model",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_READ],
        rate_limited=True
    )
)
async def get_model(
    model_id: UUID = Path(..., description="Unique model identifier"),
    user: User = Depends(RequireModelRead)
) -> ModelResponse:
    """
    Get detailed model information
    
    This endpoint demonstrates:
    - Path parameter validation
    - Proper 404 handling
    - Detailed response model
    """
    
    try:
        # In real implementation, query database
        # model = await ModelQueries.get_model_by_id(model_id)
        # if not model:
        #     raise_not_found("Model", str(model_id))
        
        # Simulate model retrieval
        if str(model_id) == "00000000-0000-0000-0000-000000000000":
            raise_not_found("Model", str(model_id))
        
        model_response = ModelResponse(
            model_id=model_id,
            name="Sample Model",
            description="A sample model for demonstration",
            model_type="teacher",
            specialization="nlp",
            performance_score=0.92,
            owner_id=str(user.id),
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={"version": "1.0.0", "framework": "pytorch"}
        )
        
        return model_response
        
    except NotFoundException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get model {model_id}: {str(e)}")
        raise


@router.put(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Update Model",
    description="Update an existing model with validation",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_UPDATE],
        rate_limited=True
    )
)
async def update_model(
    model_id: UUID = Path(..., description="Unique model identifier"),
    request: ModelUpdateRequest = Body(..., description="Model update data"),
    user: User = Depends(StandardAuth)
) -> ModelResponse:
    """
    Update an existing model
    
    This endpoint demonstrates:
    - Partial updates with validation
    - Ownership verification
    - Proper error handling
    """
    
    try:
        # In real implementation, get existing model and verify ownership
        # existing_model = await ModelQueries.get_model_by_id(model_id)
        # if not existing_model:
        #     raise_not_found("Model", str(model_id))
        # 
        # if existing_model.owner_id != user.id and not user.is_superuser:
        #     raise_forbidden("You can only update your own models")
        
        # Simulate ownership check
        if str(model_id) == "00000000-0000-0000-0000-000000000000":
            raise_not_found("Model", str(model_id))
        
        # Update only provided fields
        update_data = request.dict(exclude_unset=True)
        
        # In real implementation, update database
        # updated_model = await ModelQueries.update_model(model_id, update_data)
        
        # Simulate updated model
        model_response = ModelResponse(
            model_id=model_id,
            name=request.name or "Updated Model",
            description=request.description or "Updated description",
            model_type="teacher",
            specialization=request.specialization or "nlp",
            performance_score=request.performance_score or 0.95,
            owner_id=str(user.id),
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=request.metadata or {}
        )
        
        return model_response
        
    except (NotFoundException, ValidationException):
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to update model {model_id}: {str(e)}")
        raise


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Model",
    description="Delete a model (soft delete)",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_DELETE],
        rate_limited=True
    )
)
async def delete_model(
    model_id: UUID = Path(..., description="Unique model identifier"),
    user: User = Depends(StandardAuth)
):
    """
    Delete a model (soft delete)
    
    This endpoint demonstrates:
    - 204 No Content response
    - Ownership verification
    - Soft delete pattern
    """
    
    try:
        # In real implementation, verify model exists and ownership
        # existing_model = await ModelQueries.get_model_by_id(model_id)
        # if not existing_model:
        #     raise_not_found("Model", str(model_id))
        # 
        # if existing_model.owner_id != user.id and not user.is_superuser:
        #     raise_forbidden("You can only delete your own models")
        
        # Simulate ownership check
        if str(model_id) == "00000000-0000-0000-0000-000000000000":
            raise_not_found("Model", str(model_id))
        
        # In real implementation, perform soft delete
        # await ModelQueries.soft_delete_model(model_id)
        
        # Return 204 No Content (no response body)
        return
        
    except (NotFoundException, ValidationException):
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to delete model {model_id}: {str(e)}")
        raise


@router.post(
    "/search",
    response_model=ModelListResponse,
    summary="Search Models",
    description="Search models using semantic similarity",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_READ],
        rate_limited=True
    )
)
async def search_models(
    request: ModelSearchRequest = Body(..., description="Search criteria"),
    user: User = Depends(RequireModelRead)
) -> ModelListResponse:
    """
    Search models using semantic similarity
    
    This endpoint demonstrates:
    - Complex request body validation
    - Semantic search capabilities
    - Custom response formatting
    """
    
    try:
        # In real implementation, perform semantic search
        # results = await VectorDB.search_models(
        #     query=request.query,
        #     filters={
        #         "model_type": request.model_type,
        #         "specialization": request.specialization,
        #         "min_performance": request.min_performance
        #     },
        #     max_results=request.max_results
        # )
        
        # Simulate search results
        search_results = [
            ModelResponse(
                model_id=uuid4(),
                name=f"Search Result {i}",
                description=f"Model matching query: {request.query}",
                model_type=request.model_type or "teacher",
                specialization=request.specialization or "nlp",
                performance_score=(request.min_performance or 0.8) + (i * 0.02),
                owner_id=str(user.id),
                is_active=True,
                created_at=datetime.utcnow(),
                metadata={"similarity_score": 0.95 - (i * 0.1)}
            )
            for i in range(min(request.max_results, 3))
        ]
        
        return ModelListResponse(
            data=search_results,
            pagination={
                "page": 1,
                "page_size": len(search_results),
                "total_items": len(search_results),
                "total_pages": 1,
                "has_next": False,
                "has_previous": False
            }
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to search models: {str(e)}")
        raise


# === Additional utility endpoints ===

@router.get(
    "/{model_id}/stats",
    response_model=Dict[str, Any],
    summary="Get Model Statistics",
    description="Get usage statistics and performance metrics for a model",
    dependencies=create_endpoint_dependencies(
        require_auth=True,
        required_permissions=[Permission.MODEL_READ],
        rate_limited=True
    )
)
async def get_model_stats(
    model_id: UUID = Path(..., description="Unique model identifier"),
    user: User = Depends(RequireModelRead)
) -> Dict[str, Any]:
    """Get model statistics and metrics"""
    
    # Verify model exists
    if str(model_id) == "00000000-0000-0000-0000-000000000000":
        raise_not_found("Model", str(model_id))
    
    # Return sample statistics
    return {
        "model_id": str(model_id),
        "usage_stats": {
            "total_requests": 1250,
            "avg_response_time_ms": 245,
            "success_rate": 0.987,
            "last_24h_requests": 45
        },
        "performance_metrics": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91
        },
        "resource_usage": {
            "avg_cpu_percent": 15.2,
            "avg_memory_mb": 512,
            "total_compute_hours": 23.5
        }
    }