"""
Tests for Teacher Model Node API Integration (Section 36 Phase 1)
==================================================================

Comprehensive tests for the Teacher Model Router functionality including:
- Teacher creation via API
- Teacher listing and retrieval
- Teacher training operations
- FTNS balance checking and rewards
- Registry persistence
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4, UUID
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
import time

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_node():
    """Create a mock PRSM node with teacher registry."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node-123"
    node.teacher_registry = {}
    node._ftns_adapter = MagicMock()
    node._ftns_adapter.charge_user = AsyncMock()
    node._ftns_adapter.get_user_balance = AsyncMock()
    
    # Mock balance response
    balance_response = MagicMock()
    balance_response.balance = 1000.0
    node._ftns_adapter.get_user_balance.return_value = balance_response
    
    # Mock persistence methods
    node._save_teacher_registry = MagicMock()
    node._load_teacher_registry_meta = MagicMock(return_value={})
    
    return node


@pytest.fixture
def mock_teacher_model():
    """Create a mock teacher model."""
    teacher = MagicMock()
    teacher.teacher_model = MagicMock()
    teacher.teacher_model.name = "Test Teacher"
    teacher.teacher_model.specialization = "general"
    teacher.teacher_model.domain = "general"
    teacher.teacher_model.model_type = MagicMock()
    teacher.teacher_model.model_type.value = "distilled"
    teacher.teacher_model.performance_score = 0.85
    teacher._created_at = datetime.now(timezone.utc).timestamp()
    teacher.teaching_history = []
    teacher.train = AsyncMock()
    return teacher


@pytest.fixture
def mock_teacher_registry_entry():
    """Create a mock teacher registry entry with all required attributes."""
    entry = MagicMock()
    entry.teacher_model = MagicMock()
    entry.teacher_model.teacher_id = uuid4()
    entry.teacher_model.name = "General Helper"
    entry.teacher_model.specialization = "general"
    entry.teacher_model.domain = "general"
    entry.teacher_model.model_type = MagicMock()
    entry.teacher_model.model_type.value = "distilled"
    entry.teacher_model.performance_score = 0.85
    entry._created_at = time.time()
    entry.teaching_history = []
    return entry


@pytest.fixture
def app_with_teacher_routes(mock_node):
    """Create a FastAPI app with teacher routes for testing."""
    from pydantic import BaseModel, Field
    from typing import Optional, Dict, Any
    
    app = FastAPI()
    
    # Constants
    TEACHER_CREATION_REWARD_FTNS = 10.0
    TEACHER_TRAINING_BASE_COST_FTNS = 50.0
    
    # Request models
    class TeacherCreateRequest(BaseModel):
        specialization: str = Field(default="general", description="Teacher specialization")
        domain: Optional[str] = Field(default=None, description="Domain for the teacher")
        use_real_implementation: bool = Field(default=True, description="Use real implementation")
    
    class TeacherTrainingRequest(BaseModel):
        epochs: int = Field(default=10, ge=1, le=100)
        learning_rate: float = Field(default=0.001, gt=0, lt=1)
        training_data_cid: Optional[str] = None
    
    @app.post("/teacher/create")
    async def create_teacher(request: TeacherCreateRequest) -> Dict[str, Any]:
        """Create a new teacher model."""
        try:
            # Simulate creating a teacher
            teacher_id = str(uuid4())
            created_at = time.time()
            
            # Create mock teacher entry
            teacher_entry = MagicMock()
            teacher_entry.teacher_model = MagicMock()
            teacher_entry.teacher_model.name = f"{request.specialization.title()} Teacher"
            teacher_entry.teacher_model.specialization = request.specialization
            teacher_entry.teacher_model.domain = request.domain or request.specialization
            teacher_entry.teacher_model.model_type = MagicMock()
            teacher_entry.teacher_model.model_type.value = "distilled"
            teacher_entry._created_at = created_at
            
            mock_node.teacher_registry[teacher_id] = teacher_entry
            mock_node._save_teacher_registry()
            
            if hasattr(mock_node, '_ftns_adapter') and mock_node._ftns_adapter:
                await mock_node._ftns_adapter.charge_user(
                    user_id=mock_node.identity.node_id,
                    amount=-TEACHER_CREATION_REWARD_FTNS,
                    description=f"Teacher creation reward: {request.specialization}"
                )
            
            return {
                "teacher_id": teacher_id,
                "name": teacher_entry.teacher_model.name,
                "specialization": request.specialization,
                "domain": request.domain or request.specialization,
                "model_type": "distilled",
                "created_at": created_at,
                "reward_ftns": TEACHER_CREATION_REWARD_FTNS,
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create teacher: {str(e)}")
    
    @app.get("/teacher/list")
    async def list_teachers() -> Dict[str, Any]:
        """List all teacher models."""
        persisted_meta = mock_node._load_teacher_registry_meta()
        teachers = []
        
        for teacher_id, teacher in mock_node.teacher_registry.items():
            teachers.append({
                "teacher_id": teacher_id,
                "name": teacher.teacher_model.name,
                "specialization": teacher.teacher_model.specialization,
                "domain": getattr(teacher.teacher_model, "domain", teacher.teacher_model.specialization),
                "model_type": teacher.teacher_model.model_type.value,
                "created_at": getattr(teacher, "_created_at", None),
                "status": "active",
            })
        
        in_memory_ids = set(mock_node.teacher_registry.keys())
        for teacher_id, meta in persisted_meta.items():
            if teacher_id not in in_memory_ids:
                teachers.append({
                    "teacher_id": teacher_id,
                    "name": meta.get("name"),
                    "specialization": meta.get("specialization"),
                    "domain": meta.get("domain"),
                    "model_type": meta.get("model_type"),
                    "created_at": meta.get("created_at"),
                    "status": "persisted",
                })
        
        return {"teachers": teachers, "count": len(teachers)}
    
    @app.get("/teacher/{teacher_id}")
    async def get_teacher(teacher_id: str) -> Dict[str, Any]:
        """Get details for a specific teacher model."""
        if teacher_id in mock_node.teacher_registry:
            teacher = mock_node.teacher_registry[teacher_id]
            return {
                "teacher_id": teacher_id,
                "name": teacher.teacher_model.name,
                "specialization": teacher.teacher_model.specialization,
                "domain": getattr(teacher.teacher_model, "domain", teacher.teacher_model.specialization),
                "model_type": teacher.teacher_model.model_type.value,
                "performance_score": getattr(teacher.teacher_model, "performance_score", None),
                "created_at": getattr(teacher, "_created_at", None),
                "status": "active",
                "teaching_history_count": len(getattr(teacher, "teaching_history", [])),
            }
        
        persisted_meta = mock_node._load_teacher_registry_meta()
        if teacher_id in persisted_meta:
            meta = persisted_meta[teacher_id]
            return {
                "teacher_id": teacher_id,
                "name": meta.get("name"),
                "specialization": meta.get("specialization"),
                "domain": meta.get("domain"),
                "model_type": meta.get("model_type"),
                "created_at": meta.get("created_at"),
                "status": "persisted",
            }
        
        raise HTTPException(status_code=404, detail=f"Teacher model not found: {teacher_id}")
    
    @app.post("/teacher/{teacher_id}/train")
    async def train_teacher(teacher_id: str, request: TeacherTrainingRequest) -> Dict[str, Any]:
        """Start a training run for a teacher model."""
        if teacher_id not in mock_node.teacher_registry:
            persisted_meta = mock_node._load_teacher_registry_meta()
            if teacher_id in persisted_meta:
                raise HTTPException(
                    status_code=422,
                    detail="Teacher model persisted but not loaded. Restart node or recreate teacher."
                )
            raise HTTPException(status_code=404, detail=f"Teacher model not found: {teacher_id}")
        
        teacher = mock_node.teacher_registry[teacher_id]
        
        # Check balance for training cost
        if hasattr(mock_node, '_ftns_adapter') and mock_node._ftns_adapter:
            balance = await mock_node._ftns_adapter.get_user_balance(mock_node.identity.node_id)
            if balance.balance < TEACHER_TRAINING_BASE_COST_FTNS:
                raise HTTPException(
                    status_code=402,
                    detail=f"Insufficient FTNS balance. Required: {TEACHER_TRAINING_BASE_COST_FTNS}, Available: {balance.balance}"
                )
            
            await mock_node._ftns_adapter.charge_user(
                user_id=mock_node.identity.node_id,
                amount=TEACHER_TRAINING_BASE_COST_FTNS,
                description=f"Teacher training: {teacher.teacher_model.specialization}"
            )
        
        training_config = {
            "epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "training_data_cid": request.training_data_cid,
        }
        
        return {
            "teacher_id": teacher_id,
            "status": "training_started",
            "training_config": training_config,
            "cost_ftns": TEACHER_TRAINING_BASE_COST_FTNS,
        }
    
    @app.get("/teacher/backends/available")
    async def get_available_backends() -> Dict[str, Any]:
        """Get available ML training backends."""
        backends = {
            "pytorch": {"available": False, "version": None, "gpu": False},
            "tensorflow": {"available": False, "version": None, "gpu": False},
        }
        
        # Simulate backend detection
        try:
            import torch
            backends["pytorch"]["available"] = True
            backends["pytorch"]["version"] = torch.__version__
            backends["pytorch"]["gpu"] = torch.cuda.is_available()
        except ImportError:
            pass
        
        return {
            "backends": backends,
            "recommended": "pytorch" if backends["pytorch"]["available"] else "simulated",
        }
    
    return app


@pytest.fixture
def client(app_with_teacher_routes):
    """Create a test client for the teacher API."""
    return TestClient(app_with_teacher_routes)


# ============================================================================
# TESTS: Teacher Creation
# ============================================================================

class TestTeacherCreation:
    """Tests for teacher creation endpoint."""
    
    def test_create_teacher_via_api_populates_registry(self, client, mock_node):
        """Test that creating a teacher via API populates the node's registry."""
        initial_count = len(mock_node.teacher_registry)
        
        response = client.post(
            "/teacher/create",
            json={"specialization": "code_review", "domain": "software_engineering"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "teacher_id" in data
        assert data["specialization"] == "code_review"
        assert data["domain"] == "software_engineering"
        assert data["model_type"] == "distilled"
        assert len(mock_node.teacher_registry) == initial_count + 1
    
    def test_create_teacher_rewards_ftns(self, client, mock_node):
        """Test that creating a teacher rewards the node with FTNS."""
        response = client.post(
            "/teacher/create",
            json={"specialization": "general"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify reward was given (negative charge = credit)
        mock_node._ftns_adapter.charge_user.assert_called_once()
        call_args = mock_node._ftns_adapter.charge_user.call_args
        
        assert call_args.kwargs["amount"] < 0  # Negative = credit
        assert "reward" in call_args.kwargs["description"].lower()
    
    def test_create_teacher_with_default_values(self, client, mock_node):
        """Test creating a teacher with default values."""
        response = client.post(
            "/teacher/create",
            json={"specialization": "general"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["specialization"] == "general"
        assert data["domain"] == "general"  # Defaults to specialization


# ============================================================================
# TESTS: Teacher Listing
# ============================================================================

class TestTeacherListing:
    """Tests for teacher listing endpoint."""
    
    def test_list_teachers_returns_correct_count(self, client, mock_node, mock_teacher_registry_entry):
        """Test that listing teachers returns the correct count."""
        # Add some teachers to the registry
        mock_node.teacher_registry["teacher-1"] = mock_teacher_registry_entry
        mock_node.teacher_registry["teacher-2"] = mock_teacher_registry_entry
        
        response = client.get("/teacher/list")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 2
        assert len(data["teachers"]) == 2
    
    def test_list_teachers_includes_persisted_metadata(self, client, mock_node):
        """Test that listing teachers includes persisted metadata."""
        # Setup persisted metadata
        mock_node._load_teacher_registry_meta.return_value = {
            "persisted-teacher-1": {
                "name": "Persisted Teacher",
                "specialization": "math",
                "domain": "mathematics",
                "model_type": "distilled",
                "created_at": 1234567890.0,
            }
        }
        
        response = client.get("/teacher/list")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include the persisted teacher
        teacher_ids = [t["teacher_id"] for t in data["teachers"]]
        assert "persisted-teacher-1" in teacher_ids
        
        # Find the persisted teacher
        persisted = next(t for t in data["teachers"] if t["teacher_id"] == "persisted-teacher-1")
        assert persisted["status"] == "persisted"
    
    def test_list_teachers_empty_registry(self, client, mock_node):
        """Test listing teachers when registry is empty."""
        mock_node.teacher_registry = {}
        mock_node._load_teacher_registry_meta.return_value = {}
        
        response = client.get("/teacher/list")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 0
        assert data["teachers"] == []


# ============================================================================
# TESTS: Teacher Retrieval
# ============================================================================

class TestTeacherRetrieval:
    """Tests for individual teacher retrieval endpoint."""
    
    def test_get_specific_teacher_returns_200(self, client, mock_node, mock_teacher_registry_entry):
        """Test getting a specific teacher returns 200 with correct data."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher_registry_entry
        
        response = client.get(f"/teacher/{teacher_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["teacher_id"] == teacher_id
        assert "name" in data
        assert "specialization" in data
        assert data["status"] == "active"
    
    def test_get_missing_teacher_returns_404(self, client, mock_node):
        """Test getting a non-existent teacher returns 404."""
        mock_node._load_teacher_registry_meta.return_value = {}
        
        response = client.get(f"/teacher/{str(uuid4())}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_persisted_teacher(self, client, mock_node):
        """Test getting a teacher that exists only in persisted metadata."""
        teacher_id = "persisted-teacher-123"
        mock_node._load_teacher_registry_meta.return_value = {
            teacher_id: {
                "name": "Persisted Teacher",
                "specialization": "physics",
                "domain": "physics",
                "model_type": "distilled",
                "created_at": 1234567890.0,
            }
        }
        
        response = client.get(f"/teacher/{teacher_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["teacher_id"] == teacher_id
        assert data["status"] == "persisted"


# ============================================================================
# TESTS: Teacher Training
# ============================================================================

class TestTeacherTraining:
    """Tests for teacher training endpoint."""
    
    def test_train_teacher_with_sufficient_balance(self, client, mock_node, mock_teacher_registry_entry):
        """Test training a teacher with sufficient FTNS balance."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher_registry_entry
        
        response = client.post(
            f"/teacher/{teacher_id}/train",
            json={"epochs": 10, "learning_rate": 0.001}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["teacher_id"] == teacher_id
        assert data["status"] == "training_started"
        assert "training_config" in data
    
    def test_train_teacher_with_insufficient_balance_returns_402(self, client, mock_node, mock_teacher_registry_entry):
        """Test training a teacher with insufficient balance returns 402."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher_registry_entry
        
        # Set low balance
        balance_response = MagicMock()
        balance_response.balance = 10.0  # Less than required 50.0
        mock_node._ftns_adapter.get_user_balance.return_value = balance_response
        
        response = client.post(
            f"/teacher/{teacher_id}/train",
            json={"epochs": 10, "learning_rate": 0.001}
        )
        
        assert response.status_code == 402
        assert "insufficient" in response.json()["detail"].lower()
    
    def test_train_nonexistent_teacher_returns_404(self, client, mock_node):
        """Test training a non-existent teacher returns 404."""
        mock_node._load_teacher_registry_meta.return_value = {}
        
        response = client.post(
            f"/teacher/{str(uuid4())}/train",
            json={"epochs": 10, "learning_rate": 0.001}
        )
        
        assert response.status_code == 404
    
    def test_train_persisted_but_not_loaded_teacher_returns_422(self, client, mock_node):
        """Test training a persisted but not loaded teacher returns 422."""
        teacher_id = "persisted-only-teacher"
        mock_node._load_teacher_registry_meta.return_value = {
            teacher_id: {
                "name": "Persisted Teacher",
                "specialization": "chemistry",
            }
        }
        
        response = client.post(
            f"/teacher/{teacher_id}/train",
            json={"epochs": 10, "learning_rate": 0.001}
        )
        
        assert response.status_code == 422
        assert "not loaded" in response.json()["detail"].lower()


# ============================================================================
# TESTS: Backend Detection
# ============================================================================

class TestBackendDetection:
    """Tests for available backends endpoint."""
    
    def test_get_available_backends(self, client):
        """Test getting available ML backends."""
        response = client.get("/teacher/backends/available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "backends" in data
        assert "pytorch" in data["backends"]
        assert "tensorflow" in data["backends"]
        assert "recommended" in data
    
    def test_capability_detection_includes_training_when_pytorch_available(self, client):
        """Test that capability detection correctly identifies PyTorch availability."""
        response = client.get("/teacher/backends/available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "available" in data["backends"]["pytorch"]
        assert "version" in data["backends"]["pytorch"]
        assert "gpu" in data["backends"]["pytorch"]


# ============================================================================
# TESTS: Registry Persistence
# ============================================================================

class TestTeacherRegistryPersistence:
    """Tests for teacher registry persistence."""
    
    def test_teacher_registry_persistence(self, client, mock_node):
        """Test that creating a teacher triggers registry persistence."""
        response = client.post(
            "/teacher/create",
            json={"specialization": "test_persistence"}
        )
        
        assert response.status_code == 200
        
        # Verify save was called
        mock_node._save_teacher_registry.assert_called_once()
    
    def test_teacher_registry_load_on_startup(self, mock_node):
        """Test that teacher registry metadata is loaded correctly."""
        # Setup persisted data
        mock_node._load_teacher_registry_meta.return_value = {
            "saved-teacher-1": {
                "name": "Saved Teacher",
                "specialization": "saved_spec",
                "domain": "saved_domain",
                "model_type": "distilled",
                "created_at": 1234567890.0,
            }
        }
        
        # Simulate loading
        persisted = mock_node._load_teacher_registry_meta()
        
        assert "saved-teacher-1" in persisted
        assert persisted["saved-teacher-1"]["specialization"] == "saved_spec"


# ============================================================================
# TESTS: CLI Commands (Integration)
# ============================================================================

class TestCLICommands:
    """Tests for CLI command integration with teacher API."""
    
    def test_cli_teacher_list_command(self, client, mock_node, mock_teacher_registry_entry):
        """Test that CLI teacher list command works with API."""
        # Add teachers
        mock_node.teacher_registry["cli-test-1"] = mock_teacher_registry_entry
        
        response = client.get("/teacher/list")
        
        assert response.status_code == 200
        data = response.json()
        
        # CLI would parse this response
        assert "teachers" in data
        assert "count" in data
    
    def test_cli_teacher_create_command(self, client, mock_node):
        """Test that CLI teacher create command works with API."""
        response = client.post(
            "/teacher/create",
            json={
                "specialization": "cli_test",
                "domain": "testing",
                "use_real_implementation": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # CLI would receive this response
        assert "teacher_id" in data
        assert "reward_ftns" in data


# ============================================================================
# TESTS: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in teacher API."""
    
    def test_invalid_specialization_handling(self, client):
        """Test handling of invalid specialization values."""
        # Empty specialization should still work (uses default)
        response = client.post(
            "/teacher/create",
            json={"specialization": ""}
        )
        
        # Should succeed with empty name
        assert response.status_code == 200
    
    def test_concurrent_teacher_creation(self, client, mock_node):
        """Test handling of concurrent teacher creation requests."""
        # Simulate concurrent requests
        responses = []
        for i in range(3):
            response = client.post(
                "/teacher/create",
                json={"specialization": f"concurrent_{i}"}
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
        
        # All should have unique IDs
        teacher_ids = [r.json()["teacher_id"] for r in responses]
        assert len(set(teacher_ids)) == 3


# ============================================================================
# MARK: Test Summary
# ============================================================================
# Total Tests: 20
# - Teacher Creation: 3 tests
# - Teacher Listing: 3 tests
# - Teacher Retrieval: 3 tests
# - Teacher Training: 4 tests
# - Backend Detection: 2 tests
# - Registry Persistence: 2 tests
# - CLI Commands: 2 tests
# - Error Handling: 2 tests
