"""
Tests for Training Job Status Tracking (Step 6)

Tests the training job status tracking feature including:
- TrainingJob dataclass serialization
- POST /teacher/{teacher_id}/train endpoint
- GET /teacher/{teacher_id}/training/{run_id} endpoint
- GET /teacher/{teacher_id}/training endpoint (list runs)
- DELETE /teacher/{teacher_id}/training/{run_id} endpoint (cancel)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from prsm.node.node import TrainingJob, TrainingJobStatus
from prsm.node.api import create_api_app, TeacherTrainingRequest


# === Fixtures ===

@pytest.fixture
def mock_node():
    """Create a mock PRSMNode with teacher_registry and training_jobs."""
    node = MagicMock()
    node.teacher_registry = {}
    node.training_jobs = {}
    node.identity = MagicMock()
    node.identity.node_id = "test-node-id"
    node._ftns_adapter = None
    node._load_teacher_registry_meta = MagicMock(return_value={})
    node._save_training_runs = MagicMock()
    return node


@pytest.fixture
def mock_teacher():
    """Create a mock teacher model."""
    teacher = MagicMock()
    teacher.teacher_model = MagicMock()
    teacher.teacher_model.specialization = "test-domain"
    teacher.training_config = MagicMock()
    teacher.training_config.hyperparameters = MagicMock()
    teacher.training_config.hyperparameters.epochs = 10
    teacher.train = AsyncMock(return_value=MagicMock())
    teacher.trainer = None
    return teacher


@pytest.fixture
def client(mock_node):
    """Create a TestClient with a mocked node."""
    app = create_api_app(mock_node, enable_security=False)
    return TestClient(app)


# === TestTrainingJobDataclass ===

class TestTrainingJobDataclass:
    """Tests for the TrainingJob dataclass."""

    def test_to_dict_excludes_task(self):
        """to_dict() should exclude _task field from serialization."""
        job = TrainingJob(
            run_id="test-run-id",
            teacher_id="teacher-1",
            status=TrainingJobStatus.RUNNING,
            started_at=time.time(),
        )
        # Simulate an asyncio.Task — we need an event loop for this
        async def dummy():
            await asyncio.sleep(0)
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        job._task = loop.create_task(dummy())
        
        result = job.to_dict()
        
        assert "_task" not in result
        assert result["run_id"] == "test-run-id"
        assert result["status"] == "running"
        
        # Clean up the task
        if job._task and not job._task.done():
            job._task.cancel()

    def test_terminal_states_serialize_correctly(self):
        """Terminal states (COMPLETED, FAILED, CANCELLED) serialize correctly."""
        base_time = time.time()
        
        # Test COMPLETED state
        completed_job = TrainingJob(
            run_id="completed-run",
            teacher_id="teacher-1",
            status=TrainingJobStatus.COMPLETED,
            started_at=base_time - 100,
            completed_at=base_time,
            result={"accuracy": 0.95, "loss": 0.05},
        )
        completed_dict = completed_job.to_dict()
        assert completed_dict["status"] == "completed"
        assert completed_dict["completed_at"] == base_time
        assert completed_dict["result"] == {"accuracy": 0.95, "loss": 0.05}
        
        # Test FAILED state
        failed_job = TrainingJob(
            run_id="failed-run",
            teacher_id="teacher-1",
            status=TrainingJobStatus.FAILED,
            started_at=base_time - 50,
            completed_at=base_time,
            error="CUDA out of memory",
        )
        failed_dict = failed_job.to_dict()
        assert failed_dict["status"] == "failed"
        assert failed_dict["error"] == "CUDA out of memory"
        
        # Test CANCELLED state
        cancelled_job = TrainingJob(
            run_id="cancelled-run",
            teacher_id="teacher-1",
            status=TrainingJobStatus.CANCELLED,
            started_at=base_time - 25,
            completed_at=base_time,
        )
        cancelled_dict = cancelled_job.to_dict()
        assert cancelled_dict["status"] == "cancelled"

    def test_status_values(self):
        """All status values should work correctly."""
        assert TrainingJobStatus.PENDING.value == "pending"
        assert TrainingJobStatus.RUNNING.value == "running"
        assert TrainingJobStatus.COMPLETED.value == "completed"
        assert TrainingJobStatus.FAILED.value == "failed"
        assert TrainingJobStatus.CANCELLED.value == "cancelled"

    def test_to_dict_includes_total_epochs(self):
        """to_dict() should include total_epochs when set."""
        job = TrainingJob(
            run_id="test-run-id",
            teacher_id="teacher-1",
            status=TrainingJobStatus.RUNNING,
            started_at=time.time(),
            total_epochs=10,
        )
        
        result = job.to_dict()
        
        assert result["total_epochs"] == 10

    def test_to_dict_handles_result_with_to_dict_method(self):
        """to_dict() should call result.to_dict() if available."""
        mock_result = MagicMock()
        mock_result.to_dict = MagicMock(return_value={"final_loss": 0.01})
        
        job = TrainingJob(
            run_id="test-run-id",
            teacher_id="teacher-1",
            status=TrainingJobStatus.COMPLETED,
            started_at=time.time(),
            completed_at=time.time(),
            result=mock_result,
        )
        
        result = job.to_dict()
        
        mock_result.to_dict.assert_called_once()
        assert result["result"] == {"final_loss": 0.01}


# === TestTrainEndpointReturnsRunId ===

class TestTrainEndpointReturnsRunId:
    """Tests for the POST /teacher/{teacher_id}/train endpoint."""

    def test_response_contains_run_id(self, client, mock_node, mock_teacher):
        """Response should contain run_id."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher
        
        response = client.post(f"/teacher/{teacher_id}/train", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["run_id"]  # Should be a non-empty string

    def test_response_contains_poll_url(self, client, mock_node, mock_teacher):
        """Response should contain poll_url."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher
        
        response = client.post(f"/teacher/{teacher_id}/train", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "poll_url" in data
        assert teacher_id in data["poll_url"]
        assert data["run_id"] in data["poll_url"]

    def test_response_contains_total_epochs(self, client, mock_node, mock_teacher):
        """Response should contain total_epochs when available."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher
        
        response = client.post(f"/teacher/{teacher_id}/train", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "total_epochs" in data
        assert data["total_epochs"] == 10  # From mock_teacher config

    def test_job_enters_registry_immediately_in_pending_state(self, client, mock_node, mock_teacher):
        """Job enters registry immediately in PENDING state."""
        teacher_id = str(uuid4())
        mock_node.teacher_registry[teacher_id] = mock_teacher
        
        response = client.post(f"/teacher/{teacher_id}/train", json={})
        
        assert response.status_code == 200
        data = response.json()
        run_id = data["run_id"]
        
        # The response should indicate pending status
        assert data["status"] == "pending"
        
        # Check that the job is in the registry
        assert run_id in mock_node.training_jobs
        job = mock_node.training_jobs[run_id]
        assert job.teacher_id == teacher_id
        
        # The job was registered in PENDING state - it may have already completed
        # due to async execution, but it was definitely registered
        assert job.status in (TrainingJobStatus.PENDING, TrainingJobStatus.RUNNING, TrainingJobStatus.COMPLETED)


# === TestGetTrainingStatus ===

class TestGetTrainingStatus:
    """Tests for GET /teacher/{teacher_id}/training/{run_id}."""

    def test_returns_404_for_unknown_run_id(self, client, mock_node):
        """Returns 404 for unknown run_id."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        response = client.get(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_returns_404_when_teacher_id_mismatches(self, client, mock_node):
        """Returns 404 when teacher_id mismatches."""
        teacher_id = str(uuid4())
        other_teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a job with a different teacher_id
        job = TrainingJob(
            run_id=run_id,
            teacher_id=other_teacher_id,
            status=TrainingJobStatus.RUNNING,
            started_at=time.time(),
        )
        mock_node.training_jobs[run_id] = job
        
        response = client.get(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 404
        assert "does not belong" in response.json()["detail"].lower()

    def test_running_state_augments_response_with_progress(self, client, mock_node, mock_teacher):
        """RUNNING state augments response with progress block."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a running job
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.RUNNING,
            started_at=time.time(),
            total_epochs=10,
        )
        mock_node.training_jobs[run_id] = job
        
        # Add teacher with trainer to registry
        mock_teacher.trainer = MagicMock()
        mock_teacher.trainer.current_epoch = 5
        mock_teacher.trainer.current_step = 100
        mock_teacher.trainer.global_step = 500
        mock_teacher.trainer.training_start_time = time.time() - 60
        mock_teacher.trainer.config = MagicMock()
        mock_teacher.trainer.config.hyperparameters = MagicMock()
        mock_teacher.trainer.config.hyperparameters.epochs = 10
        mock_node.teacher_registry[teacher_id] = mock_teacher
        
        response = client.get(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "progress" in data
        assert data["progress"]["current_epoch"] == 5
        assert data["progress"]["total_epochs"] == 10

    def test_completed_state_includes_result(self, client, mock_node):
        """COMPLETED state includes result."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a completed job with result
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.COMPLETED,
            started_at=time.time() - 100,
            completed_at=time.time(),
            result={"accuracy": 0.95, "final_loss": 0.02},
        )
        mock_node.training_jobs[run_id] = job
        
        response = client.get(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "result" in data
        assert data["result"]["accuracy"] == 0.95

    def test_failed_state_includes_error(self, client, mock_node):
        """FAILED state includes error."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a failed job with error
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.FAILED,
            started_at=time.time() - 50,
            completed_at=time.time(),
            error="RuntimeError: CUDA out of memory",
        )
        mock_node.training_jobs[run_id] = job
        
        response = client.get(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "error" in data
        assert "CUDA out of memory" in data["error"]


# === TestListTrainingRuns ===

class TestListTrainingRuns:
    """Tests for GET /teacher/{teacher_id}/training."""

    def test_returns_empty_list_when_no_runs_exist(self, client, mock_node):
        """Returns empty list when no runs exist."""
        teacher_id = str(uuid4())
        
        response = client.get(f"/teacher/{teacher_id}/training")
        
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["count"] == 0

    def test_returns_all_runs_for_specific_teacher(self, client, mock_node):
        """Returns all runs for a specific teacher."""
        teacher_id = str(uuid4())
        
        # Create multiple jobs for this teacher
        for i in range(3):
            run_id = str(uuid4())
            job = TrainingJob(
                run_id=run_id,
                teacher_id=teacher_id,
                status=TrainingJobStatus.COMPLETED,
                started_at=time.time() - (i * 100),
                completed_at=time.time() - (i * 100) + 50,
            )
            mock_node.training_jobs[run_id] = job
        
        response = client.get(f"/teacher/{teacher_id}/training")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["runs"]) == 3

    def test_excludes_runs_belonging_to_other_teachers(self, client, mock_node):
        """Excludes runs belonging to other teachers."""
        teacher_id = str(uuid4())
        other_teacher_id = str(uuid4())
        
        # Create job for target teacher
        job1 = TrainingJob(
            run_id=str(uuid4()),
            teacher_id=teacher_id,
            status=TrainingJobStatus.COMPLETED,
            started_at=time.time(),
        )
        mock_node.training_jobs[job1.run_id] = job1
        
        # Create job for other teacher
        job2 = TrainingJob(
            run_id=str(uuid4()),
            teacher_id=other_teacher_id,
            status=TrainingJobStatus.COMPLETED,
            started_at=time.time(),
        )
        mock_node.training_jobs[job2.run_id] = job2
        
        response = client.get(f"/teacher/{teacher_id}/training")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["runs"][0]["teacher_id"] == teacher_id

    def test_runs_sorted_newest_first(self, client, mock_node):
        """Runs are sorted newest-first (by started_at)."""
        teacher_id = str(uuid4())
        
        # Create jobs with different start times
        base_time = time.time()
        run_ids = []
        for i in range(3):
            run_id = str(uuid4())
            run_ids.append(run_id)
            job = TrainingJob(
                run_id=run_id,
                teacher_id=teacher_id,
                status=TrainingJobStatus.COMPLETED,
                started_at=base_time - (i * 100),  # Decreasing times
            )
            mock_node.training_jobs[run_id] = job
        
        response = client.get(f"/teacher/{teacher_id}/training")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify sorted by started_at descending (newest first)
        started_times = [run["started_at"] for run in data["runs"]]
        assert started_times == sorted(started_times, reverse=True)


# === TestCancelTrainingRun ===

class TestCancelTrainingRun:
    """Tests for DELETE /teacher/{teacher_id}/training/{run_id}."""

    def test_returns_404_for_unknown_run(self, client, mock_node):
        """Returns 404 for unknown run."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        response = client.delete(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_returns_409_if_already_completed(self, client, mock_node):
        """Returns 409 if already in COMPLETED state."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a completed job
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.COMPLETED,
            started_at=time.time() - 100,
            completed_at=time.time(),
        )
        mock_node.training_jobs[run_id] = job
        
        response = client.delete(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 409
        assert "cannot cancel" in response.json()["detail"].lower()

    def test_calls_task_cancel_for_running_job(self, client, mock_node):
        """Calls task.cancel() and returns cancelled: True for RUNNING job."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a running job with a mock task
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.RUNNING,
            started_at=time.time(),
        )
        
        # Create a mock task that's not done
        mock_task = MagicMock()
        mock_task.done = MagicMock(return_value=False)
        mock_task.cancel = MagicMock()
        job._task = mock_task
        
        mock_node.training_jobs[run_id] = job
        
        response = client.delete(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is True
        mock_task.cancel.assert_called_once()

    def test_pending_job_is_cancellable(self, client, mock_node):
        """PENDING job is also cancellable."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a pending job with a mock task
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.PENDING,
            started_at=time.time(),
        )
        
        # Create a mock task that's not done
        mock_task = MagicMock()
        mock_task.done = MagicMock(return_value=False)
        mock_task.cancel = MagicMock()
        job._task = mock_task
        
        mock_node.training_jobs[run_id] = job
        
        response = client.delete(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is True
        mock_task.cancel.assert_called_once()

    def test_returns_409_if_already_failed(self, client, mock_node):
        """Returns 409 if already in FAILED state."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a failed job
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.FAILED,
            started_at=time.time() - 100,
            completed_at=time.time(),
            error="Training failed",
        )
        mock_node.training_jobs[run_id] = job
        
        response = client.delete(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 409

    def test_returns_409_if_already_cancelled(self, client, mock_node):
        """Returns 409 if already in CANCELLED state."""
        teacher_id = str(uuid4())
        run_id = str(uuid4())
        
        # Create a cancelled job
        job = TrainingJob(
            run_id=run_id,
            teacher_id=teacher_id,
            status=TrainingJobStatus.CANCELLED,
            started_at=time.time() - 100,
            completed_at=time.time(),
        )
        mock_node.training_jobs[run_id] = job
        
        response = client.delete(f"/teacher/{teacher_id}/training/{run_id}")
        
        assert response.status_code == 409
