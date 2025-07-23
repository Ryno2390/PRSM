"""
PRSM API Response Schemas and Examples
Comprehensive schemas for API documentation and validation
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# === Common Response Models ===

class APIResponse(BaseModel):
    """Base API response model"""
    success: bool = Field(description="Whether the request was successful")
    message: str = Field(description="Human-readable message")
    timestamp: datetime = Field(description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

class ErrorResponse(APIResponse):
    """Error response model"""
    success: bool = False
    error_code: str = Field(description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any] = Field(description="Array of items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    per_page: int = Field(description="Items per page")
    has_next: bool = Field(description="Whether there are more pages")
    has_prev: bool = Field(description="Whether there are previous pages")

# === Authentication Schemas ===

class LoginRequest(BaseModel):
    """Login request model"""
    email: str = Field(example="researcher@university.edu", description="User email address")
    password: str = Field(example="secure_password_123", description="User password")
    remember_me: bool = Field(False, description="Whether to extend session duration")

class LoginResponse(APIResponse):
    """Login response model"""
    access_token: str = Field(description="JWT access token")
    refresh_token: str = Field(description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(3600, description="Token expiration time in seconds")
    user: Dict[str, Any] = Field(description="User information")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Login successful",
                "timestamp": "2024-01-15T10:00:00Z",
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "email": "researcher@university.edu",
                    "role": "researcher",
                    "ftns_balance": 1000.0
                }
            }
        }

class RegisterRequest(BaseModel):
    """User registration request"""
    email: str = Field(example="new.researcher@university.edu", description="User email")
    password: str = Field(example="secure_password_123", description="Password (min 8 characters)")
    full_name: str = Field(example="Dr. Jane Smith", description="Full name")
    organization: Optional[str] = Field("University of Science", description="Organization/institution")
    role: str = Field("researcher", description="Requested role")

# === Marketplace Schemas ===

class ResourceType(str, Enum):
    """Available resource types in marketplace"""
    AI_MODEL = "ai_model"
    DATASET = "dataset"
    TOOL = "tool"
    COMPUTE_TIME = "compute_time"
    STORAGE = "storage"
    API_ACCESS = "api_access"
    RESEARCH_PAPER = "research_paper"
    TEMPLATE = "template"
    PLUGIN = "plugin"

class MarketplaceResource(BaseModel):
    """Marketplace resource model"""
    id: str = Field(description="Unique resource identifier")
    title: str = Field(example="Advanced NLP Model", description="Resource title")
    description: str = Field(description="Detailed resource description")
    resource_type: ResourceType = Field(description="Type of resource")
    price: float = Field(example=50.0, description="Price in FTNS tokens")
    seller_id: str = Field(description="Seller user ID")
    seller_name: str = Field(example="AI Research Lab", description="Seller display name")
    rating: float = Field(example=4.8, description="Average rating (0-5)")
    reviews_count: int = Field(example=25, description="Number of reviews")
    created_at: datetime = Field(description="Resource creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    tags: List[str] = Field(example=["nlp", "transformer", "pytorch"], description="Resource tags")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "res_123456789",
                "title": "GPT-4 Fine-tuned Scientific Model",
                "description": "A specialized GPT-4 model fine-tuned on scientific literature for research assistance",
                "resource_type": "ai_model",
                "price": 150.0,
                "seller_id": "user_987654321",
                "seller_name": "AI Research Consortium",
                "rating": 4.9,
                "reviews_count": 42,
                "created_at": "2024-01-10T08:00:00Z",
                "updated_at": "2024-01-14T16:30:00Z",
                "tags": ["gpt-4", "scientific", "research", "fine-tuned"]
            }
        }

class MarketplaceSearchResponse(PaginatedResponse):
    """Marketplace search results"""
    items: List[MarketplaceResource] = Field(description="Array of marketplace resources")
    filters_applied: Dict[str, Any] = Field(description="Applied search filters")
    suggestions: List[str] = Field(description="Search suggestions")

# === FTNS Token Schemas ===

class FTNSBalance(BaseModel):
    """FTNS token balance information"""
    user_id: str = Field(description="User identifier")
    available_balance: float = Field(example=1250.75, description="Available FTNS tokens")
    locked_balance: float = Field(example=250.0, description="Locked FTNS tokens (in escrow)")
    total_balance: float = Field(example=1500.75, description="Total FTNS tokens")
    last_updated: datetime = Field(description="Last balance update timestamp")

class FTNSTransaction(BaseModel):
    """FTNS transaction record"""
    transaction_id: str = Field(description="Unique transaction identifier")
    from_user_id: Optional[str] = Field(None, description="Sender user ID")
    to_user_id: Optional[str] = Field(None, description="Recipient user ID")
    amount: float = Field(example=75.0, description="Transaction amount in FTNS")
    transaction_type: str = Field(example="purchase", description="Type of transaction")
    description: str = Field(example="AI Model Purchase", description="Transaction description")
    status: str = Field(example="completed", description="Transaction status")
    created_at: datetime = Field(description="Transaction creation timestamp")
    
# === Session and Task Schemas ===

class ResearchSession(BaseModel):
    """Research session model"""
    session_id: str = Field(description="Unique session identifier")
    title: str = Field(example="Climate Change ML Analysis", description="Session title")
    description: str = Field(description="Session description")
    owner_id: str = Field(description="Session owner user ID")
    collaborators: List[str] = Field(description="List of collaborator user IDs")
    status: str = Field(example="active", description="Session status")
    created_at: datetime = Field(description="Session creation timestamp")
    last_activity: datetime = Field(description="Last activity timestamp")
    ftns_budget: float = Field(example=500.0, description="Allocated FTNS budget")
    ftns_spent: float = Field(example=125.0, description="FTNS tokens spent")

class TaskHierarchy(BaseModel):
    """Hierarchical task structure"""
    task_id: str = Field(description="Unique task identifier")
    parent_task_id: Optional[str] = Field(None, description="Parent task ID")
    title: str = Field(example="Data Preprocessing", description="Task title")
    description: str = Field(description="Task description")
    status: str = Field(example="in_progress", description="Task status")
    priority: int = Field(example=3, description="Task priority (1-5)")
    estimated_cost: float = Field(example=25.0, description="Estimated FTNS cost")
    actual_cost: Optional[float] = Field(None, description="Actual FTNS cost")
    subtasks: List['TaskHierarchy'] = Field([], description="Subtasks")

# === WebSocket Message Schemas ===

class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(description="Message type")
    data: Dict[str, Any] = Field(description="Message payload")
    timestamp: datetime = Field(description="Message timestamp")
    user_id: str = Field(description="Target user ID")

class ConversationMessage(WebSocketMessage):
    """AI conversation message"""
    conversation_id: str = Field(description="Conversation identifier")
    message_id: str = Field(description="Unique message identifier")
    role: str = Field(example="assistant", description="Message role (user/assistant)")
    content: str = Field(description="Message content")
    model_used: Optional[str] = Field(None, description="AI model used for response")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    ftns_cost: Optional[float] = Field(None, description="FTNS cost for message")

# === Health and Monitoring Schemas ===

class SystemHealth(BaseModel):
    """System health status"""
    status: str = Field(example="healthy", description="Overall system status")
    timestamp: datetime = Field(description="Health check timestamp")
    components: Dict[str, Any] = Field(description="Individual component statuses")
    response_time_ms: float = Field(example=245.5, description="Health check response time")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:00:00Z",
                "components": {
                    "database": {"status": "healthy", "response_time_ms": 15.2},
                    "redis": {"status": "healthy", "response_time_ms": 8.1},
                    "ipfs": {"status": "healthy", "response_time_ms": 45.6},
                    "vector_db": {"status": "healthy", "response_time_ms": 23.4}
                },
                "response_time_ms": 245.5
            }
        }

class MetricsSnapshot(BaseModel):
    """System metrics snapshot"""
    timestamp: datetime = Field(description="Metrics collection timestamp")
    active_users: int = Field(example=1250, description="Currently active users")
    total_sessions: int = Field(example=3420, description="Total research sessions")
    marketplace_transactions: int = Field(example=856, description="Marketplace transactions today")
    ftns_volume: float = Field(example=12500.75, description="FTNS token volume today")
    system_load: Dict[str, float] = Field(description="System load metrics")

# === API Usage Examples ===

API_EXAMPLES = {
    "authentication": {
        "login": {
            "request": {
                "email": "researcher@university.edu",
                "password": "secure_password_123",
                "remember_me": True
            },
            "response": {
                "success": True,
                "message": "Login successful",
                "timestamp": "2024-01-15T10:00:00Z",
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "email": "researcher@university.edu",
                    "role": "researcher",
                    "ftns_balance": 1000.0
                }
            }
        }
    },
    "marketplace": {
        "search_resources": {
            "request": {
                "query": "machine learning model",
                "resource_type": "ai_model",
                "max_price": 200.0,
                "min_rating": 4.0,
                "page": 1,
                "per_page": 20
            },
            "response": {
                "items": [
                    {
                        "id": "res_123456789",
                        "title": "Advanced Computer Vision Model",
                        "description": "State-of-the-art CNN for image classification",
                        "resource_type": "ai_model",
                        "price": 120.0,
                        "seller_name": "Vision Lab",
                        "rating": 4.8,
                        "reviews_count": 15
                    }
                ],
                "total": 45,
                "page": 1,
                "per_page": 20,
                "has_next": True,
                "has_prev": False
            }
        }
    },
    "websocket": {
        "real_time_updates": {
            "connection": "ws://localhost:8000/ws/{user_id}?token={jwt_token}",
            "messages": [
                {
                    "type": "session_update",
                    "data": {
                        "session_id": "session_123",
                        "status": "completed",
                        "results_available": True
                    },
                    "timestamp": "2024-01-15T10:00:00Z"
                },
                {
                    "type": "ftns_transaction",
                    "data": {
                        "transaction_id": "tx_456789",
                        "amount": 25.0,
                        "description": "Task completion reward"
                    },
                    "timestamp": "2024-01-15T10:01:00Z"
                }
            ]
        }
    }
}

# Update forward references
TaskHierarchy.model_rebuild()