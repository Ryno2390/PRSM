# PRSM Examples Cookbook

## 🍳 Practical Code Examples for Every Use Case

This cookbook provides copy-paste examples for common PRSM development tasks, from basic API usage to advanced multi-agent orchestration and custom integrations.

## 📚 Table of Contents

1. [Basic Examples](#basic-examples)
2. [API Development](#api-development)
3. [Agent Framework](#agent-framework)
4. [Database Operations](#database-operations)
5. [Authentication & Security](#authentication--security)
6. [Testing Patterns](#testing-patterns)
7. [Integration Examples](#integration-examples)
8. [Performance Optimization](#performance-optimization)

## 🚀 Basic Examples

### Simple Query Processing

```python
# examples/basic_query.py
import asyncio
import os
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import PRSMSession, UserInput

async def basic_query_example():
    """Process a simple research query with PRSM."""
    
    # Initialize the orchestrator
    orchestrator = NWTNOrchestrator()
    
    # Create a user session
    session = PRSMSession(
        user_id="example_user_001",
        nwtn_context_allocation=500,  # FTNS tokens for processing
        preferences={
            "domain": "general_science",
            "detail_level": "intermediate",
            "include_citations": True
        }
    )
    
    # Process the query
    user_input = UserInput(
        user_id="example_user_001",
        prompt="Explain the potential of quantum computing for drug discovery",
        domain="computational_biology"
    )
    
    try:
        response = await orchestrator.process_query(user_input, session)
        
        print("🔬 Research Response Generated!")
        print(f"Content length: {len(response.content)} characters")
        print(f"Context used: {response.context_used} FTNS tokens")
        print(f"Processing time: {response.processing_time:.2f} seconds")
        print(f"Confidence score: {response.confidence_score:.2f}")
        
        # Display reasoning trace
        print("\n🧠 Reasoning Steps:")
        for i, step in enumerate(response.reasoning_trace, 1):
            print(f"  {i}. {step}")
        
        # Show first 200 characters of response
        print(f"\n📄 Response Preview:")
        print(response.content[:200] + "..." if len(response.content) > 200 else response.content)
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return None

if __name__ == "__main__":
    # Run the example
    response = asyncio.run(basic_query_example())
```

### FTNS Token Management

```python
# examples/token_management.py
import asyncio
from prsm.tokenomics.ftns_service import FTNSService
from prsm.tokenomics.marketplace import ModelMarketplace

async def token_management_example():
    """Demonstrate FTNS token operations."""
    
    ftns_service = FTNSService()
    marketplace = ModelMarketplace()
    user_id = "example_user_001"
    
    # Check initial balance
    initial_balance = await ftns_service.get_balance(user_id)
    print(f"💰 Initial FTNS Balance: {initial_balance}")
    
    # Earn tokens through contribution
    contribution_reward = await ftns_service.reward_contribution(
        user_id=user_id,
        contribution_type="data_upload",
        value=250.0,  # Value of contribution
        metadata={
            "dataset_name": "climate_research_data.csv",
            "size_mb": 15.3,
            "quality_score": 0.89
        }
    )
    print(f"🎁 Earned {contribution_reward} FTNS for data contribution")
    
    # Spend tokens on marketplace
    try:
        rental_cost = await marketplace.rent_model(
            user_id=user_id,
            model_id="specialized_chemistry_model_v2",
            duration_hours=2,
            max_cost=100  # FTNS token limit
        )
        print(f"🤖 Rented model for {rental_cost} FTNS tokens")
    except InsufficientFundsError:
        print("❌ Insufficient FTNS tokens for model rental")
    
    # Check final balance
    final_balance = await ftns_service.get_balance(user_id)
    print(f"💰 Final FTNS Balance: {final_balance}")
    
    # Get transaction history
    transactions = await ftns_service.get_transaction_history(
        user_id=user_id,
        limit=10
    )
    
    print(f"\n📊 Recent Transactions ({len(transactions)}):")
    for tx in transactions:
        print(f"  {tx.timestamp}: {tx.type} - {tx.amount:+.2f} FTNS")

if __name__ == "__main__":
    asyncio.run(token_management_example())
```

### Multi-Agent Coordination

```python
# examples/multi_agent_workflow.py
import asyncio
from prsm.agents.architects.hierarchical_architect import HierarchicalArchitect
from prsm.agents.routers.model_router import ModelRouter
from prsm.agents.compilers.hierarchical_compiler import HierarchicalCompiler
from prsm.core.models import ComplexTask

async def multi_agent_workflow_example():
    """Demonstrate complex multi-agent task coordination."""
    
    # Initialize the agent pipeline
    architect = HierarchicalArchitect()
    router = ModelRouter()
    compiler = HierarchicalCompiler()
    
    # Define a complex research problem
    complex_problem = ComplexTask(
        task_id="climate_materials_research",
        description="""
        Design a comprehensive research strategy for developing new materials 
        that can capture carbon dioxide directly from the atmosphere while being 
        economically viable and environmentally sustainable. Include:
        1. Material composition analysis
        2. Manufacturing process design
        3. Economic feasibility assessment
        4. Environmental impact evaluation
        5. Scalability analysis
        """,
        domain="materials_science",
        priority="high",
        max_subtasks=8,
        timeout_minutes=15
    )
    
    print("🏗️ Starting Multi-Agent Workflow")
    print(f"Task: {complex_problem.description[:100]}...")
    
    # Step 1: Architectural decomposition
    print("\n📋 Step 1: Task Decomposition")
    task_hierarchy = await architect.recursive_decompose(
        task=complex_problem,
        max_depth=3,
        decomposition_strategy="domain_expertise"
    )
    
    print(f"✅ Decomposed into {len(task_hierarchy.subtasks)} subtasks:")
    for i, subtask in enumerate(task_hierarchy.subtasks, 1):
        print(f"  {i}. {subtask.description[:80]}...")
    
    # Step 2: Agent routing and execution
    print("\n🤖 Step 2: Agent Routing & Execution")
    subtask_results = []
    
    for subtask in task_hierarchy.subtasks:
        # Find the best agent for this subtask
        agent_candidates = await router.match_to_specialist(
            task=subtask,
            selection_criteria={
                "domain_expertise": 0.4,
                "performance_history": 0.3,
                "availability": 0.2,
                "cost": 0.1
            }
        )
        
        if agent_candidates:
            best_agent = agent_candidates[0]  # Top candidate
            print(f"🎯 Assigned to {best_agent.agent_type}: {subtask.description[:50]}...")
            
            # Execute the subtask (simulated)
            result = await best_agent.execute_task(subtask)
            subtask_results.append(result)
    
    # Step 3: Result compilation
    print("\n📝 Step 3: Result Compilation")
    final_report = await compiler.compile_final(
        mid_results=subtask_results,
        compilation_strategy="hierarchical_synthesis",
        include_confidence_scores=True,
        generate_executive_summary=True
    )
    
    print("✅ Multi-Agent Workflow Complete!")
    print(f"📊 Final report sections: {len(final_report.sections)}")
    print(f"🎯 Overall confidence: {final_report.confidence_score:.2f}")
    print(f"⏱️ Total processing time: {final_report.total_processing_time:.2f}s")
    
    # Display executive summary
    if final_report.executive_summary:
        print(f"\n📄 Executive Summary:")
        print(final_report.executive_summary[:300] + "...")
    
    return final_report

if __name__ == "__main__":
    result = asyncio.run(multi_agent_workflow_example())
```

## 🔌 API Development

### Custom API Endpoint with Authentication

```python
# examples/custom_api_endpoint.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import structlog
from datetime import datetime

from prsm.auth import get_current_user, require_permission
from prsm.auth.models import User, Permission
from prsm.core.database import get_database

logger = structlog.get_logger(__name__)

# Request/Response models
class ResearchProjectRequest(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=20, max_length=2000)
    domain: str = Field(..., regex="^[a-z_]+$")
    collaboration_level: str = Field(default="open", regex="^(private|restricted|open)$")
    funding_source: Optional[str] = None
    expected_duration_months: int = Field(ge=1, le=60)

class ResearchProjectResponse(BaseModel):
    project_id: str
    title: str
    description: str
    domain: str
    owner: str
    status: str
    created_at: datetime
    collaboration_level: str
    funding_source: Optional[str]
    expected_duration_months: int
    ftns_budget: float

# Router setup
router = APIRouter(prefix="/api/v1/research-projects", tags=["Research Projects"])

@router.post("/", response_model=ResearchProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_research_project(
    project: ResearchProjectRequest,
    current_user: User = Depends(require_permission(Permission.CREATE_PROJECT)),
    db = Depends(get_database)
):
    """
    Create a new research project with proper validation and authorization.
    
    This endpoint demonstrates:
    - Input validation with Pydantic
    - Authentication and authorization
    - Database operations
    - Structured logging
    - Error handling
    """
    
    logger.info("Creating research project",
               user_id=current_user.id,
               title=project.title,
               domain=project.domain)
    
    try:
        # Validate domain exists
        valid_domains = await get_valid_research_domains(db)
        if project.domain not in valid_domains:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid domain. Must be one of: {', '.join(valid_domains)}"
            )
        
        # Calculate initial FTNS budget based on project scope
        initial_budget = calculate_project_budget(
            duration_months=project.expected_duration_months,
            domain=project.domain,
            collaboration_level=project.collaboration_level
        )
        
        # Create project in database
        project_data = {
            "title": project.title,
            "description": project.description,
            "domain": project.domain,
            "owner_id": current_user.id,
            "collaboration_level": project.collaboration_level,
            "funding_source": project.funding_source,
            "expected_duration_months": project.expected_duration_months,
            "ftns_budget": initial_budget,
            "status": "planning",
            "created_at": datetime.utcnow()
        }
        
        project_id = await create_project_in_db(db, project_data)
        
        # Log successful creation
        logger.info("Research project created successfully",
                   project_id=project_id,
                   user_id=current_user.id,
                   initial_budget=initial_budget)
        
        # Return response
        return ResearchProjectResponse(
            project_id=project_id,
            title=project.title,
            description=project.description,
            domain=project.domain,
            owner=current_user.username,
            status="planning",
            created_at=project_data["created_at"],
            collaboration_level=project.collaboration_level,
            funding_source=project.funding_source,
            expected_duration_months=project.expected_duration_months,
            ftns_budget=initial_budget
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create research project",
                    user_id=current_user.id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )

@router.get("/", response_model=List[ResearchProjectResponse])
async def list_research_projects(
    domain: Optional[str] = None,
    collaboration_level: Optional[str] = None,
    limit: int = Field(default=20, ge=1, le=100),
    offset: int = Field(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """List research projects with filtering and pagination."""
    
    try:
        filters = {}
        if domain:
            filters["domain"] = domain
        if collaboration_level:
            filters["collaboration_level"] = collaboration_level
        
        # Only show projects user has access to
        if current_user.role != "admin":
            filters["owner_id"] = current_user.id
        
        projects = await get_projects_from_db(
            db, filters=filters, limit=limit, offset=offset
        )
        
        return [ResearchProjectResponse(**project) for project in projects]
        
    except Exception as e:
        logger.error("Failed to list research projects", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve projects"
        )

# Helper functions
async def get_valid_research_domains(db) -> List[str]:
    """Get list of valid research domains."""
    return ["artificial_intelligence", "materials_science", "biotechnology", 
            "climate_science", "quantum_computing", "neuroscience"]

def calculate_project_budget(duration_months: int, domain: str, collaboration_level: str) -> float:
    """Calculate initial FTNS budget for a project."""
    base_budget = duration_months * 100  # 100 FTNS per month base
    
    # Domain multipliers
    domain_multipliers = {
        "quantum_computing": 1.5,
        "artificial_intelligence": 1.3,
        "biotechnology": 1.2,
        "materials_science": 1.1,
        "climate_science": 1.0,
        "neuroscience": 1.2
    }
    
    # Collaboration multipliers
    collab_multipliers = {
        "private": 0.8,  # Private projects get less initial budget
        "restricted": 1.0,
        "open": 1.2  # Open projects get bonus budget
    }
    
    multiplier = domain_multipliers.get(domain, 1.0) * collab_multipliers.get(collaboration_level, 1.0)
    return round(base_budget * multiplier, 2)

async def create_project_in_db(db, project_data: dict) -> str:
    """Create project in database and return project ID."""
    # Simulate database creation
    import uuid
    project_id = str(uuid.uuid4())
    
    # In real implementation, insert into database
    # query = "INSERT INTO research_projects (...) VALUES (...)"
    # await db.execute(query, project_data)
    
    return project_id

async def get_projects_from_db(db, filters: dict, limit: int, offset: int) -> List[dict]:
    """Get projects from database with filters."""
    # Simulate database query
    # In real implementation, build and execute SQL query
    return []

# Usage example for testing
if __name__ == "__main__":
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    app.include_router(router)
    
    client = TestClient(app)
    
    # Test the endpoint
    response = client.post("/api/v1/research-projects/", json={
        "title": "AI-Driven Climate Modeling",
        "description": "Developing machine learning models to predict climate change impacts",
        "domain": "climate_science",
        "collaboration_level": "open",
        "expected_duration_months": 12
    })
    
    print(f"Response status: {response.status_code}")
    print(f"Response data: {response.json()}")
```

### WebSocket Real-time Communication

```python
# examples/websocket_example.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Session-specific subscriptions
        self.session_subscriptions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        
        logger.info("WebSocket connection established",
                   user_id=user_id,
                   total_connections=self.get_connection_count())
        
        # Send welcome message
        await self.send_to_connection(websocket, {
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to PRSM real-time updates"
        })
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove WebSocket connection."""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from session subscriptions
        for session_id, connections in self.session_subscriptions.items():
            connections.discard(websocket)
        
        logger.info("WebSocket connection closed",
                   user_id=user_id,
                   total_connections=self.get_connection_count())
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to all connections for a specific user."""
        if user_id in self.active_connections:
            connections = self.active_connections[user_id].copy()
            for connection in connections:
                try:
                    await self.send_to_connection(connection, message)
                except:
                    # Remove broken connections
                    self.active_connections[user_id].discard(connection)
    
    async def send_to_connection(self, websocket: WebSocket, message: dict):
        """Send message to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error("Failed to send WebSocket message", error=str(e))
            raise
    
    async def subscribe_to_session(self, websocket: WebSocket, session_id: str):
        """Subscribe WebSocket to session updates."""
        if session_id not in self.session_subscriptions:
            self.session_subscriptions[session_id] = set()
        
        self.session_subscriptions[session_id].add(websocket)
        
        await self.send_to_connection(websocket, {
            "type": "session_subscribed",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def broadcast_session_update(self, session_id: str, update: dict):
        """Broadcast update to all subscribers of a session."""
        if session_id in self.session_subscriptions:
            connections = self.session_subscriptions[session_id].copy()
            
            message = {
                "type": "session_update",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                **update
            }
            
            for connection in connections:
                try:
                    await self.send_to_connection(connection, message)
                except:
                    # Remove broken connections
                    self.session_subscriptions[session_id].discard(connection)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(connections) for connections in self.active_connections.values())

# Global WebSocket manager
websocket_manager = WebSocketManager()

# WebSocket endpoint
from fastapi import APIRouter

websocket_router = APIRouter()

@websocket_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication."""
    
    await websocket_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe_session":
                session_id = message.get("session_id")
                if session_id:
                    await websocket_manager.subscribe_to_session(websocket, session_id)
            
            elif message.get("type") == "ping":
                # Respond to ping with pong
                await websocket_manager.send_to_connection(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                logger.warning("Unknown WebSocket message type", 
                             message_type=message.get("type"),
                             user_id=user_id)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error("WebSocket error", user_id=user_id, error=str(e))
        websocket_manager.disconnect(websocket, user_id)

# Example usage: Sending real-time updates
async def send_processing_updates():
    """Example of sending real-time processing updates."""
    
    session_id = "example_session_123"
    user_id = "user_456"
    
    # Simulate processing with progress updates
    for progress in range(0, 101, 10):
        update = {
            "progress": progress,
            "stage": f"Processing step {progress // 10 + 1}",
            "estimated_completion": "2 minutes remaining"
        }
        
        # Send to session subscribers
        await websocket_manager.broadcast_session_update(session_id, update)
        
        # Also send directly to user
        await websocket_manager.send_to_user(user_id, {
            "type": "user_notification",
            "message": f"Your session is {progress}% complete",
            "session_id": session_id
        })
        
        await asyncio.sleep(1)  # Simulate processing time
    
    # Send completion notification
    await websocket_manager.broadcast_session_update(session_id, {
        "progress": 100,
        "stage": "completed",
        "result_preview": "Analysis complete! Click to view full results."
    })

# Client-side JavaScript example
CLIENT_JS_EXAMPLE = """
// client-side WebSocket connection
class PRSMWebSocketClient {
    constructor(userId) {
        this.userId = userId;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        const wsUrl = `ws://localhost:8000/ws/${this.userId}`;
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = (event) => {
            console.log('Connected to PRSM WebSocket');
            this.reconnectAttempts = 0;
            
            // Start heartbeat
            this.startHeartbeat();
        };
        
        this.socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.socket.onclose = (event) => {
            console.log('WebSocket connection closed');
            this.stopHeartbeat();
            this.attemptReconnect();
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'session_update':
                this.updateSessionProgress(message);
                break;
            case 'user_notification':
                this.showNotification(message);
                break;
            case 'pong':
                console.log('Received pong');
                break;
            default:
                console.log('Unknown message type:', message);
        }
    }
    
    subscribeToSession(sessionId) {
        this.send({
            type: 'subscribe_session',
            session_id: sessionId
        });
    }
    
    send(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        }
    }
    
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000); // Ping every 30 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
            
            console.log(`Attempting to reconnect in ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
        }
    }
    
    updateSessionProgress(message) {
        const progressBar = document.getElementById(`progress-${message.session_id}`);
        if (progressBar) {
            progressBar.style.width = `${message.progress}%`;
            progressBar.textContent = `${message.progress}% - ${message.stage}`;
        }
    }
    
    showNotification(message) {
        // Show browser notification or in-app notification
        if (Notification.permission === 'granted') {
            new Notification('PRSM Update', {
                body: message.message,
                icon: '/static/prsm-icon.png'
            });
        }
    }
}

// Usage
const client = new PRSMWebSocketClient('user_123');
client.connect();

// Subscribe to session updates
client.subscribeToSession('session_456');
"""

if __name__ == "__main__":
    # Test the WebSocket functionality
    print("WebSocket Example Code")
    print("Client-side JavaScript:")
    print(CLIENT_JS_EXAMPLE)
```

## 🤖 Agent Framework

### Custom Agent Implementation

```python
# examples/custom_agent.py
import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import structlog
from datetime import datetime

from prsm.agents.base import BaseAgent
from prsm.core.models import Task, AgentCapability

logger = structlog.get_logger(__name__)

class SpecializedResearchAgent(BaseAgent):
    """
    Example of a custom specialized agent for scientific research tasks.
    
    This agent demonstrates:
    - Custom capability definition
    - Task validation and preprocessing
    - Structured result formatting
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, specialization: str, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.specialization = specialization
        self.agent_type = f"research_agent_{specialization}"
        
        # Define agent capabilities
        self.capabilities = [
            AgentCapability(
                name="literature_review",
                confidence=0.95,
                processing_time_estimate=30.0,
                resource_requirements={"memory_mb": 512, "cpu_cores": 2}
            ),
            AgentCapability(
                name="hypothesis_generation",
                confidence=0.85,
                processing_time_estimate=45.0,
                resource_requirements={"memory_mb": 1024, "cpu_cores": 2}
            ),
            AgentCapability(
                name="experimental_design",
                confidence=0.80,
                processing_time_estimate=60.0,
                resource_requirements={"memory_mb": 768, "cpu_cores": 1}
            )
        ]
        
        # Specialization-specific configuration
        self.knowledge_base = self._load_specialization_knowledge(specialization)
        self.research_methods = self._get_research_methods(specialization)
        
        logger.info("Specialized research agent initialized",
                   specialization=specialization,
                   capabilities=len(self.capabilities))
    
    async def can_handle_task(self, task: Task) -> float:
        """
        Determine if this agent can handle the given task.
        Returns confidence score (0.0 - 1.0).
        """
        
        # Check if task domain matches our specialization
        domain_match = task.domain == self.specialization
        if not domain_match:
            return 0.0
        
        # Check if we have the required capability
        required_capability = task.metadata.get("required_capability")
        if required_capability:
            capability = self._get_capability(required_capability)
            if capability:
                return capability.confidence
            else:
                return 0.0
        
        # General confidence for domain-matched tasks
        return 0.75
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a research task with comprehensive result structure."""
        
        start_time = datetime.utcnow()
        
        logger.info("Processing research task",
                   agent_type=self.agent_type,
                   task_id=task.task_id,
                   task_type=task.task_type)
        
        try:
            # Validate task requirements
            await self._validate_task(task)
            
            # Preprocess task based on type
            processed_task = await self._preprocess_task(task)
            
            # Execute the main research logic
            research_result = await self._execute_research(processed_task)
            
            # Post-process and format results
            formatted_result = await self._format_research_result(research_result, task)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create comprehensive response
            response = {
                "status": "completed",
                "task_id": task.task_id,
                "agent_type": self.agent_type,
                "processing_time": processing_time,
                "result": formatted_result,
                "metadata": {
                    "specialization": self.specialization,
                    "confidence_score": research_result.get("confidence", 0.8),
                    "research_methods_used": research_result.get("methods_used", []),
                    "source_count": research_result.get("source_count", 0),
                    "validation_status": "passed"
                },
                "performance_metrics": {
                    "memory_used_mb": await self._get_memory_usage(),
                    "api_calls_made": research_result.get("api_calls", 0),
                    "cache_hit_rate": research_result.get("cache_hit_rate", 0.0)
                }
            }
            
            logger.info("Research task completed successfully",
                       task_id=task.task_id,
                       processing_time=processing_time,
                       confidence=response["metadata"]["confidence_score"])
            
            return response
            
        except Exception as e:
            logger.error("Research task failed",
                        task_id=task.task_id,
                        agent_type=self.agent_type,
                        error=str(e))
            
            return {
                "status": "failed",
                "task_id": task.task_id,
                "agent_type": self.agent_type,
                "error": str(e),
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _validate_task(self, task: Task):
        """Validate that the task meets our requirements."""
        
        if not task.description or len(task.description.strip()) < 10:
            raise ValueError("Task description must be at least 10 characters")
        
        if task.domain != self.specialization:
            raise ValueError(f"Task domain '{task.domain}' doesn't match agent specialization '{self.specialization}'")
        
        # Check resource requirements
        if task.metadata.get("max_processing_time", 300) < 60:
            raise ValueError("Minimum processing time for research tasks is 60 seconds")
    
    async def _preprocess_task(self, task: Task) -> Dict[str, Any]:
        """Preprocess the task based on its type and our specialization."""
        
        processed = {
            "original_task": task,
            "research_question": task.description,
            "context": task.metadata.get("context", ""),
            "expected_output_format": task.metadata.get("output_format", "comprehensive_report"),
            "resource_constraints": task.metadata.get("resource_constraints", {}),
            "time_limit": task.metadata.get("max_processing_time", 300)
        }
        
        # Add specialization-specific preprocessing
        if self.specialization == "climate_science":
            processed["focus_areas"] = ["temperature_trends", "precipitation_patterns", "ecosystem_impacts"]
        elif self.specialization == "artificial_intelligence":
            processed["focus_areas"] = ["machine_learning", "neural_networks", "optimization"]
        elif self.specialization == "biotechnology":
            processed["focus_areas"] = ["genetic_engineering", "protein_analysis", "drug_discovery"]
        
        return processed
    
    async def _execute_research(self, processed_task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main research logic."""
        
        research_question = processed_task["research_question"]
        focus_areas = processed_task.get("focus_areas", [])
        
        # Simulate comprehensive research process
        logger.info("Executing research", 
                   question=research_question[:100],
                   focus_areas=focus_areas)
        
        # Step 1: Literature review
        literature_results = await self._conduct_literature_review(research_question)
        
        # Step 2: Data analysis (if applicable)
        data_analysis = await self._analyze_relevant_data(research_question, focus_areas)
        
        # Step 3: Synthesis and insights
        insights = await self._synthesize_insights(literature_results, data_analysis)
        
        # Step 4: Generate recommendations
        recommendations = await self._generate_recommendations(insights)
        
        return {
            "literature_review": literature_results,
            "data_analysis": data_analysis,
            "key_insights": insights,
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(literature_results, data_analysis),
            "methods_used": ["literature_review", "data_analysis", "synthesis"],
            "source_count": len(literature_results.get("sources", [])),
            "api_calls": literature_results.get("api_calls", 0) + data_analysis.get("api_calls", 0),
            "cache_hit_rate": 0.75  # Simulated cache performance
        }
    
    async def _conduct_literature_review(self, research_question: str) -> Dict[str, Any]:
        """Conduct literature review for the research question."""
        
        # Simulate literature search and analysis
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            "sources": [
                {
                    "title": f"Recent advances in {self.specialization}",
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "year": 2024,
                    "relevance_score": 0.95,
                    "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
                },
                {
                    "title": f"Future directions in {self.specialization} research",
                    "authors": ["Prof. Davis", "Dr. Wilson"],
                    "year": 2023,
                    "relevance_score": 0.87,
                    "key_findings": ["Future direction 1", "Future direction 2"]
                }
            ],
            "summary": f"Literature review reveals significant progress in {self.specialization}...",
            "gaps_identified": ["Gap 1", "Gap 2"],
            "api_calls": 15  # Simulated API usage
        }
    
    async def _analyze_relevant_data(self, research_question: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze relevant data for the research question."""
        
        await asyncio.sleep(1.5)  # Simulate processing time
        
        return {
            "datasets_analyzed": [f"{area}_dataset.csv" for area in focus_areas],
            "statistical_summary": {
                "sample_size": 10000,
                "significant_correlations": ["correlation_1", "correlation_2"],
                "p_values": [0.001, 0.005, 0.023]
            },
            "visualizations_created": ["trend_chart.png", "correlation_matrix.png"],
            "anomalies_detected": 3,
            "api_calls": 8
        }
    
    async def _synthesize_insights(self, literature: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
        """Synthesize insights from literature and data analysis."""
        
        insights = [
            f"Literature analysis reveals {len(literature.get('sources', []))} relevant studies",
            f"Data analysis of {data.get('statistical_summary', {}).get('sample_size', 0)} samples shows significant trends",
            f"Key gaps identified: {', '.join(literature.get('gaps_identified', []))}",
            f"Statistical significance found in {len(data.get('statistical_summary', {}).get('significant_correlations', []))} correlations"
        ]
        
        return insights
    
    async def _generate_recommendations(self, insights: List[str]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on insights."""
        
        recommendations = [
            {
                "recommendation": "Conduct follow-up study to address identified gaps",
                "priority": "high",
                "estimated_timeline": "6-12 months",
                "required_resources": ["funding", "research_team", "equipment"]
            },
            {
                "recommendation": "Develop predictive model based on identified correlations",
                "priority": "medium",
                "estimated_timeline": "3-6 months",
                "required_resources": ["computational_resources", "data_scientists"]
            }
        ]
        
        return recommendations
    
    def _calculate_confidence(self, literature: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Calculate confidence score based on research quality."""
        
        source_count = len(literature.get("sources", []))
        data_quality = data.get("statistical_summary", {}).get("sample_size", 0)
        
        # Simple confidence calculation
        confidence = min(0.95, 0.5 + (source_count * 0.1) + (min(data_quality / 10000, 1) * 0.35))
        
        return round(confidence, 2)
    
    async def _format_research_result(self, research_result: Dict[str, Any], original_task: Task) -> Dict[str, Any]:
        """Format the research result according to the requested output format."""
        
        output_format = original_task.metadata.get("output_format", "comprehensive_report")
        
        if output_format == "executive_summary":
            return {
                "summary": f"Research on '{original_task.description}' completed",
                "key_insights": research_result["key_insights"][:3],  # Top 3 insights
                "top_recommendation": research_result["recommendations"][0] if research_result["recommendations"] else None
            }
        
        elif output_format == "detailed_report":
            return {
                "executive_summary": f"Comprehensive analysis of {original_task.description}",
                "methodology": research_result["methods_used"],
                "literature_review": research_result["literature_review"],
                "data_analysis": research_result["data_analysis"],
                "insights": research_result["key_insights"],
                "recommendations": research_result["recommendations"],
                "confidence_assessment": research_result["confidence"]
            }
        
        else:  # comprehensive_report (default)
            return research_result
    
    def _load_specialization_knowledge(self, specialization: str) -> Dict[str, Any]:
        """Load knowledge base specific to the specialization."""
        
        # In a real implementation, this would load from a database or file
        knowledge_bases = {
            "climate_science": {
                "key_concepts": ["greenhouse_gases", "temperature_anomalies", "sea_level_rise"],
                "methodologies": ["climate_modeling", "statistical_analysis", "remote_sensing"],
                "data_sources": ["NOAA", "NASA_GISS", "IPCC_reports"]
            },
            "artificial_intelligence": {
                "key_concepts": ["machine_learning", "neural_networks", "optimization"],
                "methodologies": ["supervised_learning", "unsupervised_learning", "reinforcement_learning"],
                "data_sources": ["ArXiv", "IEEE_papers", "Google_Scholar"]
            },
            "biotechnology": {
                "key_concepts": ["genetic_engineering", "protein_folding", "drug_discovery"],
                "methodologies": ["CRISPR", "protein_modeling", "clinical_trials"],
                "data_sources": ["PubMed", "UniProt", "PDB"]
            }
        }
        
        return knowledge_bases.get(specialization, {})
    
    def _get_research_methods(self, specialization: str) -> List[str]:
        """Get research methods appropriate for the specialization."""
        
        method_mappings = {
            "climate_science": ["statistical_analysis", "climate_modeling", "trend_analysis"],
            "artificial_intelligence": ["algorithm_design", "performance_evaluation", "comparative_analysis"],
            "biotechnology": ["experimental_design", "molecular_analysis", "clinical_testing"]
        }
        
        return method_mappings.get(specialization, ["general_research", "literature_review"])
    
    def _get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get a specific capability by name."""
        
        for capability in self.capabilities:
            if capability.name == capability_name:
                return capability
        
        return None
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

# Usage example
async def agent_usage_example():
    """Demonstrate how to use the custom specialized agent."""
    
    # Create specialized agents for different domains
    climate_agent = SpecializedResearchAgent("climate_science")
    ai_agent = SpecializedResearchAgent("artificial_intelligence")
    
    # Create sample tasks
    climate_task = Task(
        task_id="climate_001",
        task_type="research",
        description="Analyze the impact of rising sea levels on coastal ecosystems",
        domain="climate_science",
        priority="high",
        metadata={
            "required_capability": "literature_review",
            "output_format": "detailed_report",
            "max_processing_time": 180,
            "context": "Focus on impacts in the next 50 years"
        }
    )
    
    ai_task = Task(
        task_id="ai_001",
        task_type="research",
        description="Evaluate the effectiveness of transformer architectures for scientific text analysis",
        domain="artificial_intelligence",
        priority="medium",
        metadata={
            "required_capability": "experimental_design",
            "output_format": "executive_summary",
            "max_processing_time": 120
        }
    )
    
    # Test agent capabilities
    print("🧪 Testing Agent Capabilities")
    
    climate_confidence = await climate_agent.can_handle_task(climate_task)
    ai_confidence = await ai_agent.can_handle_task(ai_task)
    
    print(f"Climate agent confidence for climate task: {climate_confidence}")
    print(f"AI agent confidence for AI task: {ai_confidence}")
    
    # Process tasks
    print("\n🔬 Processing Climate Science Task")
    climate_result = await climate_agent.process_task(climate_task)
    print(f"Status: {climate_result['status']}")
    print(f"Processing time: {climate_result.get('processing_time', 0):.2f}s")
    print(f"Confidence: {climate_result.get('metadata', {}).get('confidence_score', 0)}")
    
    print("\n🤖 Processing AI Task")
    ai_result = await ai_agent.process_task(ai_task)
    print(f"Status: {ai_result['status']}")
    print(f"Processing time: {ai_result.get('processing_time', 0):.2f}s")
    print(f"Confidence: {ai_result.get('metadata', {}).get('confidence_score', 0)}")

if __name__ == "__main__":
    asyncio.run(agent_usage_example())
```

This examples cookbook provides comprehensive, practical code examples that developers can copy and adapt for their specific use cases. Each example includes proper error handling, logging, documentation, and follows the coding standards established in the project.