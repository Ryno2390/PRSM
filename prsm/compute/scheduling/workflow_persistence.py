"""
Workflow Persistence and Parameter Storage

💾 WORKFLOW PERSISTENCE SYSTEM:
- Save and resume complex multi-step workflows
- Secure parameter storage with encryption
- Checkpoint-based recovery for fault tolerance
- Workflow templates and sharing capabilities
- State management across workflow execution lifecycle
- Version control for workflow definitions

This module implements robust persistence that enables:
1. Save workflows in progress and resume later
2. Create reusable workflow templates
3. Recover from failures at specific checkpoints
4. Share workflow configurations between users
5. Track workflow evolution and optimization
6. Secure storage of sensitive parameters
"""

import asyncio
import json
import hashlib
import pickle
import zlib
import base64
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4
from pathlib import Path
import os

import structlog
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

from prsm.core.models import PRSMBaseModel, TimestampMixin, AgentType
from prsm.compute.scheduling.workflow_scheduler import (
    ScheduledWorkflow, WorkflowStep, WorkflowStatus, SchedulingPriority, 
    ExecutionWindow, ResourceRequirement
)

logger = structlog.get_logger(__name__)


class PersistenceStatus(str, Enum):
    """Workflow persistence status"""
    DRAFT = "draft"                    # Workflow being designed
    SAVED = "saved"                    # Workflow saved but not scheduled
    ACTIVE = "active"                  # Workflow currently running
    PAUSED = "paused"                  # Workflow paused mid-execution
    CHECKPOINTED = "checkpointed"      # Workflow saved at checkpoint
    COMPLETED = "completed"            # Workflow finished successfully
    FAILED = "failed"                  # Workflow failed and saved for analysis
    ARCHIVED = "archived"              # Workflow archived for long-term storage


class TemplateCategory(str, Enum):
    """Workflow template categories"""
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    RESEARCH = "research"
    EDUCATION = "education"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class SharePermission(str, Enum):
    """Workflow sharing permissions"""
    PRIVATE = "private"                # Only owner can access
    SHARED_READ = "shared_read"        # Specific users can view
    SHARED_EXECUTE = "shared_execute"  # Specific users can run
    PUBLIC_READ = "public_read"        # Anyone can view
    PUBLIC_TEMPLATE = "public_template" # Anyone can use as template
    MARKETPLACE = "marketplace"        # Available for purchase


class WorkflowCheckpoint(PRSMBaseModel):
    """Checkpoint for workflow recovery"""
    checkpoint_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    
    # Checkpoint metadata
    checkpoint_name: str
    checkpoint_description: str = Field(default="")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Execution state
    completed_steps: List[UUID] = Field(default_factory=list)
    current_step_id: Optional[UUID] = None
    step_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Resource state
    allocated_resources: Dict[str, float] = Field(default_factory=dict)
    resource_usage_stats: Dict[str, Any] = Field(default_factory=dict)
    
    # Context and variables
    workflow_variables: Dict[str, Any] = Field(default_factory=dict)
    agent_states: Dict[str, Any] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    execution_time_elapsed: float = Field(default=0.0)
    steps_completed: int = Field(default=0)
    total_steps: int = Field(default=0)
    completion_percentage: float = Field(default=0.0)
    
    # Recovery information
    recovery_instructions: List[str] = Field(default_factory=list)
    prerequisite_checks: List[Dict[str, Any]] = Field(default_factory=list)
    rollback_points: List[UUID] = Field(default_factory=list)
    
    def calculate_completion_percentage(self) -> float:
        """Calculate workflow completion percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.steps_completed / self.total_steps) * 100


class WorkflowTemplate(PRSMBaseModel):
    """Reusable workflow template"""
    template_id: UUID = Field(default_factory=uuid4)
    template_name: str
    description: str
    category: TemplateCategory
    
    # Template metadata
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0.0")
    tags: List[str] = Field(default_factory=list)
    
    # Template definition
    step_templates: List[Dict[str, Any]] = Field(default_factory=list)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    parameter_schema: Dict[str, Any] = Field(default_factory=dict)
    
    # Usage and sharing
    usage_count: int = Field(default=0)
    share_permission: SharePermission = Field(default=SharePermission.PRIVATE)
    shared_with: List[str] = Field(default_factory=list)
    
    # Ratings and reviews
    rating: float = Field(ge=0.0, le=5.0, default=0.0)
    review_count: int = Field(default=0)
    reviews: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Pricing (for marketplace templates)
    price: Optional[float] = None
    license_type: str = Field(default="standard")
    
    # Performance statistics
    average_execution_time: Optional[float] = None
    success_rate: Optional[float] = None
    resource_requirements_estimate: Dict[str, float] = Field(default_factory=dict)
    
    def create_workflow_from_template(
        self, 
        user_id: str, 
        parameters: Dict[str, Any] = None
    ) -> ScheduledWorkflow:
        """Create a workflow instance from this template"""
        parameters = parameters or {}
        
        # Merge default parameters with provided parameters
        merged_params = {**self.default_parameters, **parameters}
        
        # Create workflow steps from template
        steps = []
        for step_template in self.step_templates:
            step = WorkflowStep(
                step_name=step_template.get("name", "Untitled Step"),
                step_description=step_template.get("description", ""),
                agent_type=AgentType(step_template.get("agent_type", "architect")),
                prompt_template=step_template.get("prompt_template", ""),
                parameters=merged_params
            )
            steps.append(step)
        
        # Create execution window (24 hours from now by default)
        execution_window = ExecutionWindow(
            earliest_start=datetime.now(timezone.utc),
            latest_start=datetime.now(timezone.utc) + timedelta(hours=24),
            max_duration=timedelta(hours=12)
        )
        
        # Create workflow
        workflow = ScheduledWorkflow(
            user_id=user_id,
            workflow_name=f"{self.template_name} - {datetime.now().strftime('%Y%m%d_%H%M')}",
            description=f"Created from template: {self.template_name}",
            steps=steps,
            execution_window=execution_window,
            created_by=user_id
        )
        
        # Update usage statistics
        self.usage_count += 1
        
        return workflow


class PersistedWorkflow(PRSMBaseModel):
    """Persisted workflow with full state information"""
    persistence_id: UUID = Field(default_factory=uuid4)
    workflow: ScheduledWorkflow
    
    # Persistence metadata
    persistence_status: PersistenceStatus = Field(default=PersistenceStatus.DRAFT)
    saved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Checkpoints and recovery
    checkpoints: List[WorkflowCheckpoint] = Field(default_factory=list)
    latest_checkpoint_id: Optional[UUID] = None
    auto_checkpoint_enabled: bool = Field(default=True)
    checkpoint_interval_minutes: int = Field(default=30)
    
    # Version control
    version_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_version: str = Field(default="1.0.0")
    
    # Security and encryption
    is_encrypted: bool = Field(default=False)
    encryption_key_hash: Optional[str] = None
    sensitive_parameters: List[str] = Field(default_factory=list)
    
    # Collaboration
    collaborators: List[str] = Field(default_factory=list)
    share_permission: SharePermission = Field(default=SharePermission.PRIVATE)
    comments: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Storage optimization
    is_compressed: bool = Field(default=False)
    original_size_bytes: int = Field(default=0)
    compressed_size_bytes: int = Field(default=0)
    compression_ratio: float = Field(default=1.0)
    
    # Template information
    template_id: Optional[UUID] = None
    template_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    def add_checkpoint(self, checkpoint: WorkflowCheckpoint):
        """Add a new checkpoint"""
        self.checkpoints.append(checkpoint)
        self.latest_checkpoint_id = checkpoint.checkpoint_id
        
        # Keep only last 10 checkpoints to save space
        if len(self.checkpoints) > 10:
            self.checkpoints = self.checkpoints[-10:]
    
    def get_latest_checkpoint(self) -> Optional[WorkflowCheckpoint]:
        """Get the most recent checkpoint"""
        if self.latest_checkpoint_id:
            return next(
                (cp for cp in self.checkpoints if cp.checkpoint_id == self.latest_checkpoint_id),
                None
            )
        return self.checkpoints[-1] if self.checkpoints else None


class WorkflowStorage(PRSMBaseModel):
    """Storage configuration for workflows"""
    storage_id: UUID = Field(default_factory=uuid4)
    user_id: str
    
    # Storage quotas
    max_workflows: int = Field(default=100)
    max_storage_mb: float = Field(default=1000.0)  # 1GB default
    max_checkpoints_per_workflow: int = Field(default=10)
    
    # Current usage
    current_workflows: int = Field(default=0)
    current_storage_mb: float = Field(default=0.0)
    current_checkpoints: int = Field(default=0)
    
    # Auto-cleanup settings
    auto_cleanup_enabled: bool = Field(default=True)
    cleanup_after_days: int = Field(default=90)
    archive_old_workflows: bool = Field(default=True)
    
    # Backup settings
    backup_enabled: bool = Field(default=True)
    backup_frequency_hours: int = Field(default=24)
    last_backup: Optional[datetime] = None
    
    def check_quota(self, additional_size_mb: float = 0) -> Dict[str, Any]:
        """Check if user is within storage quotas"""
        return {
            "workflows_ok": self.current_workflows < self.max_workflows,
            "storage_ok": (self.current_storage_mb + additional_size_mb) <= self.max_storage_mb,
            "workflows_used": self.current_workflows,
            "workflows_limit": self.max_workflows,
            "storage_used_mb": self.current_storage_mb,
            "storage_limit_mb": self.max_storage_mb,
            "storage_available_mb": self.max_storage_mb - self.current_storage_mb
        }


class WorkflowPersistenceEngine(TimestampMixin):
    """
    Workflow Persistence and Parameter Storage Engine
    
    Comprehensive system for saving, resuming, and managing workflows
    with secure parameter storage and fault tolerance.
    """
    
    def __init__(self, storage_base_path: str = "./workflow_storage"):
        super().__init__()
        
        # Storage configuration
        self.storage_base_path = Path(storage_base_path)
        self.storage_base_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.persisted_workflows: Dict[UUID, PersistedWorkflow] = {}
        self.workflow_templates: Dict[UUID, WorkflowTemplate] = {}
        self.user_storage: Dict[str, WorkflowStorage] = {}
        
        # Security
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Performance tracking
        self.persistence_statistics: Dict[str, Any] = {
            "workflows_saved": 0,
            "workflows_loaded": 0,
            "checkpoints_created": 0,
            "templates_created": 0,
            "recoveries_performed": 0,
            "total_storage_mb": 0.0
        }
        
        # Configuration
        self.auto_save_interval = timedelta(minutes=5)
        self.max_versions_per_workflow = 20
        self.compression_threshold_kb = 100  # Compress workflows > 100KB
        
        self._initialize_storage()
        self._start_auto_save_task()
        
        logger.info("WorkflowPersistenceEngine initialized", storage_path=str(self.storage_base_path))
    
    def _initialize_storage(self):
        """Initialize storage directories and load existing data"""
        # Create directory structure
        (self.storage_base_path / "workflows").mkdir(exist_ok=True)
        (self.storage_base_path / "templates").mkdir(exist_ok=True)
        (self.storage_base_path / "checkpoints").mkdir(exist_ok=True)
        (self.storage_base_path / "backups").mkdir(exist_ok=True)
        
        # Load existing data
        self._load_existing_workflows()
        self._load_existing_templates()
    
    def _load_existing_workflows(self):
        """Load existing workflows from storage"""
        try:
            workflows_dir = self.storage_base_path / "workflows"
            for workflow_file in workflows_dir.glob("*.json"):
                try:
                    with open(workflow_file, 'r') as f:
                        data = json.load(f)
                    
                    persisted_workflow = PersistedWorkflow(**data)
                    self.persisted_workflows[persisted_workflow.persistence_id] = persisted_workflow
                    
                    # Update user storage statistics
                    user_id = persisted_workflow.workflow.user_id
                    if user_id not in self.user_storage:
                        self.user_storage[user_id] = WorkflowStorage(user_id=user_id)
                    
                    self.user_storage[user_id].current_workflows += 1
                    
                except Exception as e:
                    logger.warning("Failed to load workflow", file=str(workflow_file), error=str(e))
            
            logger.info("Loaded existing workflows", count=len(self.persisted_workflows))
            
        except Exception as e:
            logger.error("Error loading existing workflows", error=str(e))
    
    def _load_existing_templates(self):
        """Load existing templates from storage"""
        try:
            templates_dir = self.storage_base_path / "templates"
            for template_file in templates_dir.glob("*.json"):
                try:
                    with open(template_file, 'r') as f:
                        data = json.load(f)
                    
                    template = WorkflowTemplate(**data)
                    self.workflow_templates[template.template_id] = template
                    
                except Exception as e:
                    logger.warning("Failed to load template", file=str(template_file), error=str(e))
            
            logger.info("Loaded existing templates", count=len(self.workflow_templates))
            
        except Exception as e:
            logger.error("Error loading existing templates", error=str(e))
    
    def _start_auto_save_task(self):
        """Start auto-save background task"""
        # In production, this would run as an async background task
        pass
    
    async def save_workflow(
        self,
        workflow: ScheduledWorkflow,
        encryption_enabled: bool = False,
        compression_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Save a workflow with optional encryption and compression
        
        Args:
            workflow: Workflow to save
            encryption_enabled: Whether to encrypt sensitive data
            compression_enabled: Whether to compress the workflow
            
        Returns:
            Save operation result
        """
        try:
            # Check user storage quota
            user_storage = self.user_storage.get(workflow.user_id)
            if not user_storage:
                user_storage = WorkflowStorage(user_id=workflow.user_id)
                self.user_storage[workflow.user_id] = user_storage
            
            quota_check = user_storage.check_quota(additional_size_mb=1.0)  # Estimate 1MB
            if not quota_check["workflows_ok"] or not quota_check["storage_ok"]:
                return {
                    "success": False,
                    "error": "Storage quota exceeded",
                    "quota_status": quota_check
                }
            
            # Create or update persisted workflow
            persistence_id = getattr(workflow, 'persistence_id', None)
            if persistence_id and persistence_id in self.persisted_workflows:
                persisted_workflow = self.persisted_workflows[persistence_id]
                persisted_workflow.workflow = workflow
                persisted_workflow.saved_at = datetime.now(timezone.utc)
            else:
                persisted_workflow = PersistedWorkflow(
                    workflow=workflow,
                    persistence_status=PersistenceStatus.SAVED
                )
                workflow.persistence_id = persisted_workflow.persistence_id
            
            # Apply encryption if requested
            if encryption_enabled:
                await self._encrypt_sensitive_data(persisted_workflow)
            
            # Apply compression if the workflow is large
            workflow_data = persisted_workflow.dict()
            original_size = len(json.dumps(workflow_data))
            
            if compression_enabled and original_size > (self.compression_threshold_kb * 1024):
                compressed_data = await self._compress_workflow_data(workflow_data)
                persisted_workflow.is_compressed = True
                persisted_workflow.original_size_bytes = original_size
                persisted_workflow.compressed_size_bytes = len(compressed_data)
                persisted_workflow.compression_ratio = len(compressed_data) / original_size
                
                # Save compressed data
                file_path = self.storage_base_path / "workflows" / f"{persisted_workflow.persistence_id}.json.gz"
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
            else:
                # Save uncompressed data
                file_path = self.storage_base_path / "workflows" / f"{persisted_workflow.persistence_id}.json"
                with open(file_path, 'w') as f:
                    json.dump(workflow_data, f, indent=2, default=str)
                
                persisted_workflow.original_size_bytes = original_size
                persisted_workflow.compressed_size_bytes = original_size
            
            # Store in memory
            self.persisted_workflows[persisted_workflow.persistence_id] = persisted_workflow
            
            # Update storage statistics
            user_storage.current_workflows += 1
            user_storage.current_storage_mb += persisted_workflow.compressed_size_bytes / (1024 * 1024)
            
            # Update global statistics
            self.persistence_statistics["workflows_saved"] += 1
            self.persistence_statistics["total_storage_mb"] += persisted_workflow.compressed_size_bytes / (1024 * 1024)
            
            logger.info(
                "Workflow saved successfully",
                persistence_id=str(persisted_workflow.persistence_id),
                workflow_id=str(workflow.workflow_id),
                user_id=workflow.user_id,
                compressed=persisted_workflow.is_compressed,
                encrypted=encryption_enabled
            )
            
            return {
                "success": True,
                "persistence_id": str(persisted_workflow.persistence_id),
                "file_size_bytes": persisted_workflow.compressed_size_bytes,
                "compression_ratio": persisted_workflow.compression_ratio,
                "encrypted": encryption_enabled,
                "storage_used": user_storage.current_storage_mb,
                "storage_limit": user_storage.max_storage_mb
            }
            
        except Exception as e:
            logger.error("Error saving workflow", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def load_workflow(
        self,
        persistence_id: UUID,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Load a persisted workflow
        
        Args:
            persistence_id: ID of persisted workflow
            user_id: ID of user requesting the workflow
            
        Returns:
            Load operation result with workflow data
        """
        try:
            # Check if workflow exists in memory
            if persistence_id in self.persisted_workflows:
                persisted_workflow = self.persisted_workflows[persistence_id]
            else:
                # Try to load from disk
                persisted_workflow = await self._load_workflow_from_disk(persistence_id)
                if not persisted_workflow:
                    return {"success": False, "error": "Workflow not found"}
            
            # Check access permissions
            if not await self._check_workflow_access(persisted_workflow, user_id):
                return {"success": False, "error": "Access denied"}
            
            # Decrypt if necessary
            if persisted_workflow.is_encrypted:
                await self._decrypt_sensitive_data(persisted_workflow)
            
            # Update access time
            persisted_workflow.last_accessed = datetime.now(timezone.utc)
            
            # Update statistics
            self.persistence_statistics["workflows_loaded"] += 1
            
            logger.info(
                "Workflow loaded successfully",
                persistence_id=str(persistence_id),
                user_id=user_id,
                workflow_name=persisted_workflow.workflow.workflow_name
            )
            
            return {
                "success": True,
                "workflow": persisted_workflow.workflow,
                "persistence_info": {
                    "persistence_id": str(persisted_workflow.persistence_id),
                    "saved_at": persisted_workflow.saved_at.isoformat(),
                    "last_accessed": persisted_workflow.last_accessed.isoformat(),
                    "status": persisted_workflow.persistence_status,
                    "checkpoints_count": len(persisted_workflow.checkpoints),
                    "version": persisted_workflow.current_version
                }
            }
            
        except Exception as e:
            logger.error("Error loading workflow", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _load_workflow_from_disk(self, persistence_id: UUID) -> Optional[PersistedWorkflow]:
        """Load workflow from disk storage"""
        try:
            # Try compressed file first
            compressed_path = self.storage_base_path / "workflows" / f"{persistence_id}.json.gz"
            uncompressed_path = self.storage_base_path / "workflows" / f"{persistence_id}.json"
            
            if compressed_path.exists():
                with open(compressed_path, 'rb') as f:
                    compressed_data = f.read()
                
                workflow_data = await self._decompress_workflow_data(compressed_data)
                persisted_workflow = PersistedWorkflow(**workflow_data)
                
            elif uncompressed_path.exists():
                with open(uncompressed_path, 'r') as f:
                    workflow_data = json.load(f)
                
                persisted_workflow = PersistedWorkflow(**workflow_data)
            else:
                return None
            
            # Store in memory for faster access
            self.persisted_workflows[persistence_id] = persisted_workflow
            return persisted_workflow
            
        except Exception as e:
            logger.error("Error loading workflow from disk", persistence_id=str(persistence_id), error=str(e))
            return None
    
    async def create_checkpoint(
        self,
        workflow_id: UUID,
        checkpoint_name: str,
        completed_steps: List[UUID],
        current_step_id: Optional[UUID] = None,
        step_results: Dict[str, Any] = None,
        workflow_variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a checkpoint for workflow recovery
        
        Args:
            workflow_id: ID of workflow to checkpoint
            checkpoint_name: Name for the checkpoint
            completed_steps: List of completed step IDs
            current_step_id: ID of currently executing step
            step_results: Results from completed steps
            workflow_variables: Current workflow variables
            
        Returns:
            Checkpoint creation result
        """
        try:
            # Find the persisted workflow
            persisted_workflow = None
            for pw in self.persisted_workflows.values():
                if pw.workflow.workflow_id == workflow_id:
                    persisted_workflow = pw
                    break
            
            if not persisted_workflow:
                return {"success": False, "error": "Workflow not found"}
            
            # Create checkpoint
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_id,
                checkpoint_name=checkpoint_name,
                completed_steps=completed_steps,
                current_step_id=current_step_id,
                step_results=step_results or {},
                workflow_variables=workflow_variables or {},
                steps_completed=len(completed_steps),
                total_steps=len(persisted_workflow.workflow.steps)
            )
            
            checkpoint.completion_percentage = checkpoint.calculate_completion_percentage()
            
            # Add to persisted workflow
            persisted_workflow.add_checkpoint(checkpoint)
            persisted_workflow.persistence_status = PersistenceStatus.CHECKPOINTED
            
            # Save checkpoint to disk
            await self._save_checkpoint_to_disk(checkpoint)
            
            # Update statistics
            self.persistence_statistics["checkpoints_created"] += 1
            
            logger.info(
                "Checkpoint created",
                checkpoint_id=str(checkpoint.checkpoint_id),
                workflow_id=str(workflow_id),
                checkpoint_name=checkpoint_name,
                completion_percentage=checkpoint.completion_percentage
            )
            
            return {
                "success": True,
                "checkpoint_id": str(checkpoint.checkpoint_id),
                "completion_percentage": checkpoint.completion_percentage,
                "steps_completed": checkpoint.steps_completed,
                "total_steps": checkpoint.total_steps
            }
            
        except Exception as e:
            logger.error("Error creating checkpoint", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _save_checkpoint_to_disk(self, checkpoint: WorkflowCheckpoint):
        """Save checkpoint to disk for persistence"""
        try:
            checkpoint_path = self.storage_base_path / "checkpoints" / f"{checkpoint.checkpoint_id}.json"
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint.dict(), f, indent=2, default=str)
                
        except Exception as e:
            logger.error("Error saving checkpoint to disk", error=str(e))
    
    async def recover_from_checkpoint(
        self,
        checkpoint_id: UUID,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Recover workflow execution from a specific checkpoint
        
        Args:
            checkpoint_id: ID of checkpoint to recover from
            user_id: ID of user requesting recovery
            
        Returns:
            Recovery operation result
        """
        try:
            # Find checkpoint
            checkpoint = None
            persisted_workflow = None
            
            for pw in self.persisted_workflows.values():
                for cp in pw.checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        checkpoint = cp
                        persisted_workflow = pw
                        break
                if checkpoint:
                    break
            
            if not checkpoint or not persisted_workflow:
                return {"success": False, "error": "Checkpoint not found"}
            
            # Check access permissions
            if not await self._check_workflow_access(persisted_workflow, user_id):
                return {"success": False, "error": "Access denied"}
            
            # Create recovery plan
            recovery_plan = {
                "workflow_id": str(persisted_workflow.workflow.workflow_id),
                "checkpoint_id": str(checkpoint_id),
                "completed_steps": checkpoint.completed_steps,
                "current_step_id": str(checkpoint.current_step_id) if checkpoint.current_step_id else None,
                "remaining_steps": [
                    str(step.step_id) for step in persisted_workflow.workflow.steps
                    if step.step_id not in checkpoint.completed_steps
                ],
                "step_results": checkpoint.step_results,
                "workflow_variables": checkpoint.workflow_variables,
                "completion_percentage": checkpoint.completion_percentage
            }
            
            # Update workflow status
            persisted_workflow.workflow.status = WorkflowStatus.QUEUED  # Ready to resume
            persisted_workflow.persistence_status = PersistenceStatus.ACTIVE
            
            # Update statistics
            self.persistence_statistics["recoveries_performed"] += 1
            
            logger.info(
                "Workflow recovery prepared",
                checkpoint_id=str(checkpoint_id),
                workflow_id=str(persisted_workflow.workflow.workflow_id),
                user_id=user_id,
                completion_percentage=checkpoint.completion_percentage
            )
            
            return {
                "success": True,
                "recovery_plan": recovery_plan,
                "workflow": persisted_workflow.workflow,
                "message": f"Workflow can be resumed from {checkpoint.completion_percentage:.1f}% completion"
            }
            
        except Exception as e:
            logger.error("Error recovering from checkpoint", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def create_template(
        self,
        workflow: ScheduledWorkflow,
        template_name: str,
        description: str,
        category: TemplateCategory,
        created_by: str,
        share_permission: SharePermission = SharePermission.PRIVATE,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create a workflow template from an existing workflow
        
        Args:
            workflow: Source workflow to create template from
            template_name: Name for the template
            description: Template description
            category: Template category
            created_by: ID of user creating template
            share_permission: Sharing permission level
            price: Price for marketplace templates
            
        Returns:
            Template creation result
        """
        try:
            # Extract step templates from workflow
            step_templates = []
            default_parameters = {}
            
            for step in workflow.steps:
                step_template = {
                    "name": step.step_name,
                    "description": step.step_description,
                    "agent_type": step.agent_type.value,
                    "prompt_template": step.prompt_template,
                    "estimated_duration_minutes": step.estimated_duration.total_seconds() / 60,
                    "max_retries": step.max_retries,
                    "timeout_minutes": step.timeout.total_seconds() / 60 if step.timeout else None
                }
                step_templates.append(step_template)
                
                # Extract parameters
                for key, value in step.parameters.items():
                    if key not in default_parameters:
                        default_parameters[key] = value
            
            # Create template
            template = WorkflowTemplate(
                template_name=template_name,
                description=description,
                category=category,
                created_by=created_by,
                step_templates=step_templates,
                default_parameters=default_parameters,
                share_permission=share_permission,
                price=price
            )
            
            # Generate parameter schema for validation
            template.parameter_schema = self._generate_parameter_schema(default_parameters)
            
            # Save template
            template_path = self.storage_base_path / "templates" / f"{template.template_id}.json"
            with open(template_path, 'w') as f:
                json.dump(template.dict(), f, indent=2, default=str)
            
            # Store in memory
            self.workflow_templates[template.template_id] = template
            
            # Update statistics
            self.persistence_statistics["templates_created"] += 1
            
            logger.info(
                "Template created",
                template_id=str(template.template_id),
                template_name=template_name,
                created_by=created_by,
                category=category.value
            )
            
            return {
                "success": True,
                "template_id": str(template.template_id),
                "template_name": template_name,
                "step_count": len(step_templates),
                "parameter_count": len(default_parameters),
                "share_permission": share_permission.value
            }
            
        except Exception as e:
            logger.error("Error creating template", error=str(e))
            return {"success": False, "error": str(e)}
    
    def _generate_parameter_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON schema for template parameters"""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for key, value in parameters.items():
            param_type = "string"  # Default
            
            if isinstance(value, bool):
                param_type = "boolean"
            elif isinstance(value, int):
                param_type = "integer"
            elif isinstance(value, float):
                param_type = "number"
            elif isinstance(value, list):
                param_type = "array"
            elif isinstance(value, dict):
                param_type = "object"
            
            schema["properties"][key] = {
                "type": param_type,
                "description": f"Parameter {key}",
                "default": value
            }
        
        return schema
    
    async def _encrypt_sensitive_data(self, persisted_workflow: PersistedWorkflow):
        """Encrypt sensitive parameters in workflow"""
        try:
            # Identify sensitive parameters
            sensitive_keys = ["password", "api_key", "secret", "token", "credential"]
            
            for step in persisted_workflow.workflow.steps:
                for key, value in step.parameters.items():
                    if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                        if isinstance(value, str):
                            encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
                            step.parameters[key] = encrypted_value
                            
                            if key not in persisted_workflow.sensitive_parameters:
                                persisted_workflow.sensitive_parameters.append(key)
            
            persisted_workflow.is_encrypted = True
            persisted_workflow.encryption_key_hash = hashlib.sha256(self.encryption_key).hexdigest()
            
        except Exception as e:
            logger.error("Error encrypting sensitive data", error=str(e))
    
    async def _decrypt_sensitive_data(self, persisted_workflow: PersistedWorkflow):
        """Decrypt sensitive parameters in workflow"""
        try:
            if not persisted_workflow.is_encrypted:
                return
            
            for step in persisted_workflow.workflow.steps:
                for key in persisted_workflow.sensitive_parameters:
                    if key in step.parameters:
                        encrypted_value = step.parameters[key]
                        if isinstance(encrypted_value, str):
                            try:
                                decrypted_value = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
                                step.parameters[key] = decrypted_value
                            except Exception:
                                logger.warning("Failed to decrypt parameter", key=key)
            
        except Exception as e:
            logger.error("Error decrypting sensitive data", error=str(e))
    
    async def _compress_workflow_data(self, workflow_data: Dict[str, Any]) -> bytes:
        """Compress workflow data"""
        json_data = json.dumps(workflow_data, default=str)
        return zlib.compress(json_data.encode('utf-8'))
    
    async def _decompress_workflow_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress workflow data"""
        json_data = zlib.decompress(compressed_data).decode('utf-8')
        return json.loads(json_data)
    
    async def _check_workflow_access(self, persisted_workflow: PersistedWorkflow, user_id: str) -> bool:
        """Check if user has access to workflow"""
        # Owner always has access
        if persisted_workflow.workflow.user_id == user_id:
            return True
        
        # Check collaborators
        if user_id in persisted_workflow.collaborators:
            return True
        
        # Check sharing permissions
        if persisted_workflow.share_permission in [SharePermission.PUBLIC_READ, SharePermission.PUBLIC_TEMPLATE]:
            return True
        
        return False
    
    def get_user_workflows(
        self,
        user_id: str,
        status_filter: Optional[List[PersistenceStatus]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get workflows for a specific user"""
        user_workflows = []
        
        for persisted_workflow in self.persisted_workflows.values():
            # Check access
            if not asyncio.run(self._check_workflow_access(persisted_workflow, user_id)):
                continue
            
            # Apply status filter
            if status_filter and persisted_workflow.persistence_status not in status_filter:
                continue
            
            workflow_info = {
                "persistence_id": str(persisted_workflow.persistence_id),
                "workflow_id": str(persisted_workflow.workflow.workflow_id),
                "workflow_name": persisted_workflow.workflow.workflow_name,
                "description": persisted_workflow.workflow.description,
                "status": persisted_workflow.persistence_status.value,
                "saved_at": persisted_workflow.saved_at.isoformat(),
                "last_accessed": persisted_workflow.last_accessed.isoformat(),
                "checkpoints_count": len(persisted_workflow.checkpoints),
                "version": persisted_workflow.current_version,
                "step_count": len(persisted_workflow.workflow.steps),
                "is_encrypted": persisted_workflow.is_encrypted,
                "is_compressed": persisted_workflow.is_compressed,
                "file_size_bytes": persisted_workflow.compressed_size_bytes
            }
            
            user_workflows.append(workflow_info)
        
        # Sort by last accessed (most recent first)
        user_workflows.sort(key=lambda x: x["last_accessed"], reverse=True)
        
        return user_workflows[:limit]
    
    def get_available_templates(
        self,
        user_id: str,
        category_filter: Optional[TemplateCategory] = None,
        include_private: bool = False
    ) -> List[Dict[str, Any]]:
        """Get available workflow templates"""
        available_templates = []
        
        for template in self.workflow_templates.values():
            # Check access permissions
            can_access = (
                template.created_by == user_id or  # Owner
                template.share_permission in [
                    SharePermission.PUBLIC_READ, 
                    SharePermission.PUBLIC_TEMPLATE, 
                    SharePermission.MARKETPLACE
                ] or
                (include_private and user_id in template.shared_with)
            )
            
            if not can_access:
                continue
            
            # Apply category filter
            if category_filter and template.category != category_filter:
                continue
            
            template_info = {
                "template_id": str(template.template_id),
                "template_name": template.template_name,
                "description": template.description,
                "category": template.category.value,
                "created_by": template.created_by,
                "created_at": template.created_at.isoformat(),
                "version": template.version,
                "usage_count": template.usage_count,
                "rating": template.rating,
                "review_count": template.review_count,
                "price": template.price,
                "step_count": len(template.step_templates),
                "parameter_count": len(template.default_parameters),
                "share_permission": template.share_permission.value,
                "tags": template.tags
            }
            
            available_templates.append(template_info)
        
        # Sort by rating and usage
        available_templates.sort(
            key=lambda x: (x["rating"] * x["review_count"] + x["usage_count"]), 
            reverse=True
        )
        
        return available_templates
    
    def get_persistence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive persistence statistics"""
        total_workflows = len(self.persisted_workflows)
        total_templates = len(self.workflow_templates)
        total_users = len(self.user_storage)
        
        # Calculate storage usage by user
        storage_by_user = {}
        for user_id, storage in self.user_storage.items():
            storage_by_user[user_id] = {
                "workflows": storage.current_workflows,
                "storage_mb": storage.current_storage_mb,
                "quota_mb": storage.max_storage_mb,
                "utilization_percentage": (storage.current_storage_mb / storage.max_storage_mb) * 100
            }
        
        # Calculate status distribution
        status_distribution = {}
        for status in PersistenceStatus:
            count = sum(1 for pw in self.persisted_workflows.values() if pw.persistence_status == status)
            status_distribution[status.value] = count
        
        return {
            "persistence_statistics": dict(self.persistence_statistics),
            "total_workflows": total_workflows,
            "total_templates": total_templates,
            "total_users": total_users,
            "storage_by_user": storage_by_user,
            "status_distribution": status_distribution,
            "compression_stats": {
                "compressed_workflows": sum(1 for pw in self.persisted_workflows.values() if pw.is_compressed),
                "average_compression_ratio": sum(
                    pw.compression_ratio for pw in self.persisted_workflows.values() if pw.is_compressed
                ) / max(1, sum(1 for pw in self.persisted_workflows.values() if pw.is_compressed))
            },
            "security_stats": {
                "encrypted_workflows": sum(1 for pw in self.persisted_workflows.values() if pw.is_encrypted),
                "workflows_with_sensitive_data": sum(
                    1 for pw in self.persisted_workflows.values() if pw.sensitive_parameters
                )
            }
        }


# Global instance for easy access
_workflow_persistence_engine = None

def get_workflow_persistence_engine() -> WorkflowPersistenceEngine:
    """Get global workflow persistence engine instance"""
    global _workflow_persistence_engine
    if _workflow_persistence_engine is None:
        _workflow_persistence_engine = WorkflowPersistenceEngine()
    return _workflow_persistence_engine