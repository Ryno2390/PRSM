"""
PRSM Database Service Layer
===========================

Comprehensive database operations service providing CRUD functionality
for all PRSM entities with proper async/await patterns and error handling.

This service layer bridges the gap between the application logic and
SQLAlchemy models, providing clean interfaces for database operations.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from sqlalchemy import select, update, delete, and_, or_, desc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from .database import (
    get_async_session, Base,
    PRSMSessionModel, ReasoningStepModel, SafetyFlagModel,
    ArchitectTaskModel, FTNSTransactionModel, FTNSBalanceModel,
    TeacherModelModel, CircuitBreakerEventModel, PeerNodeModel,
    ModelRegistryModel, TeamModel, TeamMemberModel, TeamWalletModel,
    TeamTaskModel, TeamGovernanceModel
)
from .models import (
    PRSMSession, ReasoningStep, SafetyFlag, ArchitectTask,
    FTNSTransaction, FTNSBalance, TeacherModel
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Centralized database operations service for PRSM
    
    Provides CRUD operations for all PRSM entities with:
    - Proper async/await patterns
    - Transaction management
    - Error handling and logging
    - Type safety with Pydantic models
    - Performance optimization
    """
    
    def __init__(self):
        self.logger = logger
    
    # === Session Management CRUD ===
    
    async def get_session(self, session_id: UUID) -> Optional[PRSMSession]:
        """Get PRSM session by ID"""
        async with get_async_session() as db:
            try:
                stmt = select(PRSMSessionModel).where(PRSMSessionModel.session_id == session_id)
                result = await db.execute(stmt)
                session_model = result.scalar_one_or_none()
                
                if session_model:
                    return PRSMSession(
                        session_id=session_model.session_id,
                        user_id=session_model.user_id,
                        nwtn_context_allocation=session_model.nwtn_context_allocation,
                        context_used=session_model.context_used,
                        status=session_model.status,
                        metadata=session_model.model_metadata,
                        created_at=session_model.created_at,
                        updated_at=session_model.updated_at
                    )
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to get session {session_id}: {e}")
                raise
    
    async def list_user_sessions(self, user_id: str, limit: int = 100, offset: int = 0) -> List[PRSMSession]:
        """List sessions for a user with pagination"""
        async with get_async_session() as db:
            try:
                stmt = (
                    select(PRSMSessionModel)
                    .where(PRSMSessionModel.user_id == user_id)
                    .order_by(desc(PRSMSessionModel.created_at))
                    .limit(limit)
                    .offset(offset)
                )
                result = await db.execute(stmt)
                session_models = result.scalars().all()
                
                return [
                    PRSMSession(
                        session_id=session.session_id,
                        user_id=session.user_id,
                        nwtn_context_allocation=session.nwtn_context_allocation,
                        context_used=session.context_used,
                        status=session.status,
                        metadata=session.model_metadata,
                        created_at=session.created_at,
                        updated_at=session.updated_at
                    )
                    for session in session_models
                ]
                
            except Exception as e:
                self.logger.error(f"Failed to list sessions for user {user_id}: {e}")
                raise
    
    # === Helper Methods ===
    
    def _serialize_for_json(self, data: Any) -> Any:
        """Convert data to JSON-serializable format, handling UUIDs, datetime, and complex objects"""
        if isinstance(data, UUID):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif hasattr(data, '__dict__'):
            # Handle Pydantic models and other complex objects
            if hasattr(data, 'model_dump'):
                # Pydantic v2 style
                return self._serialize_for_json(data.model_dump())
            elif hasattr(data, 'dict'):
                # Pydantic v1 style
                return self._serialize_for_json(data.dict())
            else:
                # Convert to string representation for complex objects
                return str(data)
        elif isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_json(item) for item in data]
        else:
            return data
    
    # === Reasoning Steps CRUD ===
    
    async def create_reasoning_step(
        self,
        session_id: UUID,
        step_data: Dict[str, Any]
    ) -> str:
        """Create a new reasoning step"""
        async with get_async_session() as db:
            try:
                step_id = uuid4()
                
                # Serialize data to handle UUIDs
                input_data = self._serialize_for_json(step_data.get("input_data", {}))
                output_data = self._serialize_for_json(step_data.get("output_data", {}))
                
                reasoning_step = ReasoningStepModel(
                    step_id=step_id,
                    session_id=session_id,
                    agent_type=step_data.get("agent_type", "default"),
                    agent_id=step_data.get("agent_id", "unknown"),
                    input_data=input_data,
                    output_data=output_data,
                    execution_time=step_data.get("execution_time", 0.0),
                    confidence_score=step_data.get("confidence_score", 0.0)
                )
                
                db.add(reasoning_step)
                await db.commit()
                
                self.logger.info(f"Created reasoning step {step_id} for session {session_id}")
                return str(step_id)
                
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to create reasoning step: {e}")
                raise
    
    async def get_reasoning_steps_by_session(
        self,
        session_id: UUID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all reasoning steps for a session"""
        async with get_async_session() as db:
            try:
                stmt = (
                    select(ReasoningStepModel)
                    .where(ReasoningStepModel.session_id == session_id)
                    .order_by(ReasoningStepModel.timestamp)
                    .limit(limit)
                )
                result = await db.execute(stmt)
                step_models = result.scalars().all()
                
                # Note: Converting database model to simplified dict for now
                return [
                    {
                        "step_id": str(step.step_id),
                        "session_id": str(step.session_id),
                        "agent_type": step.agent_type,
                        "agent_id": step.agent_id,
                        "input_data": step.input_data,
                        "output_data": step.output_data,
                        "execution_time": step.execution_time,
                        "confidence_score": step.confidence_score,
                        "timestamp": step.timestamp
                    }
                    for step in step_models
                ]
                
            except Exception as e:
                self.logger.error(f"Failed to get reasoning steps for session {session_id}: {e}")
                raise
    
    async def update_reasoning_step(
        self,
        step_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a reasoning step"""
        async with get_async_session() as db:
            try:
                stmt = (
                    update(ReasoningStepModel)
                    .where(ReasoningStepModel.step_id == step_id)
                    .values(**updates)
                )
                result = await db.execute(stmt)
                await db.commit()
                
                success = result.rowcount > 0
                if success:
                    self.logger.info(f"Updated reasoning step {step_id}")
                else:
                    self.logger.warning(f"Reasoning step {step_id} not found for update")
                
                return success
                
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to update reasoning step {step_id}: {e}")
                raise
    
    # === Safety Flags CRUD ===
    
    async def create_safety_flag(
        self,
        session_id: UUID,
        flag_data: Dict[str, Any]
    ) -> str:
        """Create a new safety flag"""
        async with get_async_session() as db:
            try:
                flag_id = uuid4()
                safety_flag = SafetyFlagModel(
                    flag_id=flag_id,
                    session_id=session_id,
                    level=flag_data.get("level", "medium"),
                    category=flag_data.get("category", "unknown"),
                    description=flag_data.get("description", ""),
                    triggered_by=flag_data.get("triggered_by", "system"),
                    resolved=flag_data.get("resolved", False)
                )
                
                db.add(safety_flag)
                await db.commit()
                
                self.logger.warning(f"Created safety flag {flag_id} for session {session_id}: {flag_data.get('category')}")
                return str(flag_id)
                
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to create safety flag: {e}")
                raise
    
    async def get_safety_flags_by_session(
        self,
        session_id: UUID,
        include_resolved: bool = True
    ) -> List[SafetyFlag]:
        """Get safety flags for a session"""
        async with get_async_session() as db:
            try:
                stmt = select(SafetyFlagModel).where(SafetyFlagModel.session_id == session_id)
                
                if not include_resolved:
                    stmt = stmt.where(SafetyFlagModel.resolution_status != "resolved")
                
                stmt = stmt.order_by(desc(SafetyFlagModel.created_at))
                
                result = await db.execute(stmt)
                flag_models = result.scalars().all()
                
                return [
                    SafetyFlag(
                        flag_id=flag.flag_id,
                        session_id=flag.session_id,
                        step_id=flag.step_id,
                        flag_type=flag.flag_type,
                        severity=flag.severity,
                        description=flag.description,
                        flagged_content=flag.flagged_content,
                        confidence=flag.confidence,
                        resolution_status=flag.resolution_status,
                        metadata=flag.model_metadata,
                        created_at=flag.created_at,
                        resolved_at=flag.resolved_at
                    )
                    for flag in flag_models
                ]
                
            except Exception as e:
                self.logger.error(f"Failed to get safety flags for session {session_id}: {e}")
                raise
    
    async def resolve_safety_flag(
        self,
        flag_id: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """Resolve a safety flag"""
        async with get_async_session() as db:
            try:
                updates = {
                    "resolution_status": "resolved",
                    "resolved_at": datetime.now(timezone.utc)
                }
                if resolution_notes:
                    updates["resolution_notes"] = resolution_notes
                
                stmt = (
                    update(SafetyFlagModel)
                    .where(SafetyFlagModel.flag_id == flag_id)
                    .values(**updates)
                )
                result = await db.execute(stmt)
                await db.commit()
                
                success = result.rowcount > 0
                if success:
                    self.logger.info(f"Resolved safety flag {flag_id}")
                else:
                    self.logger.warning(f"Safety flag {flag_id} not found for resolution")
                
                return success
                
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to resolve safety flag {flag_id}: {e}")
                raise
    
    # === Architect Tasks CRUD ===
    
    async def create_architect_task(
        self,
        session_id: UUID,
        task_data: Dict[str, Any]
    ) -> str:
        """Create a new architect task"""
        async with get_async_session() as db:
            try:
                task_id = uuid4()
                architect_task = ArchitectTaskModel(
                    task_id=task_id,
                    session_id=session_id,
                    parent_task_id=task_data.get("parent_task_id"),
                    level=task_data.get("level", 0),
                    instruction=task_data.get("instruction", ""),
                    complexity_score=task_data.get("complexity_score", 0.0),
                    dependencies=task_data.get("dependencies", []),
                    status=task_data.get("status", "pending"),
                    assigned_agent=task_data.get("assigned_agent"),
                    result=task_data.get("result"),
                    execution_time=task_data.get("execution_time"),
                    model_metadata=task_data.get("metadata", {})
                )
                
                db.add(architect_task)
                await db.commit()
                
                self.logger.info(f"Created architect task {task_id} for session {session_id}")
                return str(task_id)
                
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to create architect task: {e}")
                raise
    
    async def get_task_hierarchy(
        self,
        session_id: UUID,
        root_only: bool = False
    ) -> List[ArchitectTask]:
        """Get task hierarchy for a session"""
        async with get_async_session() as db:
            try:
                stmt = select(ArchitectTaskModel).where(ArchitectTaskModel.session_id == session_id)
                
                if root_only:
                    stmt = stmt.where(ArchitectTaskModel.parent_task_id.is_(None))
                
                stmt = stmt.order_by(ArchitectTaskModel.created_at)
                
                result = await db.execute(stmt)
                task_models = result.scalars().all()
                
                return [
                    ArchitectTask(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        parent_task_id=task.parent_task_id,
                        task_type=task.task_type,
                        description=task.description,
                        requirements=task.requirements,
                        dependencies=task.dependencies,
                        estimated_complexity=task.estimated_complexity,
                        status=task.status,
                        assigned_agent=task.assigned_agent,
                        result=task.result,
                        execution_time=task.execution_time,
                        metadata=task.model_metadata,
                        created_at=task.created_at
                    )
                    for task in task_models
                ]
                
            except Exception as e:
                self.logger.error(f"Failed to get task hierarchy for session {session_id}: {e}")
                raise
    
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> bool:
        """Update task status and result"""
        async with get_async_session() as db:
            try:
                updates = {"status": status}
                if result is not None:
                    updates["result"] = result
                if execution_time is not None:
                    updates["execution_time"] = execution_time
                
                stmt = (
                    update(ArchitectTaskModel)
                    .where(ArchitectTaskModel.task_id == task_id)
                    .values(**updates)
                )
                result = await db.execute(stmt)
                await db.commit()
                
                success = result.rowcount > 0
                if success:
                    self.logger.info(f"Updated task {task_id} status to {status}")
                else:
                    self.logger.warning(f"Task {task_id} not found for status update")
                
                return success
                
            except Exception as e:
                await db.rollback()
                self.logger.error(f"Failed to update task {task_id} status: {e}")
                raise
    
    async def get_tasks_by_parent(
        self,
        parent_task_id: str
    ) -> List[ArchitectTask]:
        """Get subtasks by parent task ID"""
        async with get_async_session() as db:
            try:
                stmt = (
                    select(ArchitectTaskModel)
                    .where(ArchitectTaskModel.parent_task_id == parent_task_id)
                    .order_by(ArchitectTaskModel.created_at)
                )
                result = await db.execute(stmt)
                task_models = result.scalars().all()
                
                return [
                    ArchitectTask(
                        task_id=task.task_id,
                        session_id=task.session_id,
                        parent_task_id=task.parent_task_id,
                        task_type=task.task_type,
                        description=task.description,
                        requirements=task.requirements,
                        dependencies=task.dependencies,
                        estimated_complexity=task.estimated_complexity,
                        status=task.status,
                        assigned_agent=task.assigned_agent,
                        result=task.result,
                        execution_time=task.execution_time,
                        metadata=task.model_metadata,
                        created_at=task.created_at
                    )
                    for task in task_models
                ]
                
            except Exception as e:
                self.logger.error(f"Failed to get subtasks for parent {parent_task_id}: {e}")
                raise
    
    # === Statistics and Analytics ===
    
    async def get_session_statistics(self, session_id: UUID) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        async with get_async_session() as db:
            try:
                # Count reasoning steps
                steps_stmt = select(func.count(ReasoningStepModel.step_id)).where(
                    ReasoningStepModel.session_id == session_id
                )
                steps_result = await db.execute(steps_stmt)
                total_steps = steps_result.scalar() or 0
                
                # Count safety flags
                flags_stmt = select(func.count(SafetyFlagModel.flag_id)).where(
                    SafetyFlagModel.session_id == session_id
                )
                flags_result = await db.execute(flags_stmt)
                total_flags = flags_result.scalar() or 0
                
                # Count tasks
                tasks_stmt = select(func.count(ArchitectTaskModel.task_id)).where(
                    ArchitectTaskModel.session_id == session_id
                )
                tasks_result = await db.execute(tasks_stmt)
                total_tasks = tasks_result.scalar() or 0
                
                # Calculate total execution time (substitute for tokens)
                time_stmt = select(func.sum(ReasoningStepModel.execution_time)).where(
                    ReasoningStepModel.session_id == session_id
                )
                time_result = await db.execute(time_stmt)
                total_execution_time = time_result.scalar() or 0.0
                
                return {
                    "session_id": str(session_id),
                    "total_reasoning_steps": total_steps,
                    "total_safety_flags": total_flags,
                    "total_tasks": total_tasks,
                    "total_execution_time": total_execution_time,
                    "has_safety_issues": total_flags > 0
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get session statistics for {session_id}: {e}")
                raise
    
    # === Provenance Record Management ===
    
    async def create_provenance_record(self, record_data: Dict[str, Any]) -> bool:
        """Create a new provenance record in the database"""
        async with get_async_session() as db:
            try:
                from .models import ProvenanceRecord
                
                # Create record entry for the provenance system
                provenance_record = {
                    'content_id': record_data['content_id'],
                    'fingerprint_data': record_data['fingerprint_data'],
                    'attribution_data': record_data['attribution_data'],
                    'usage_data': record_data.get('usage_data', {}),
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc)
                }
                
                # For now, store in a simple in-memory registry
                # In production, this would be a proper database table
                if not hasattr(self, '_provenance_records'):
                    self._provenance_records = {}
                
                self._provenance_records[record_data['content_id']] = provenance_record
                
                self.logger.info(f"Created provenance record for content {record_data['content_id']}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to create provenance record: {e}")
                return False
    
    async def get_provenance_record(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get provenance record by content ID"""
        try:
            # Check in-memory registry first
            if hasattr(self, '_provenance_records') and content_id in self._provenance_records:
                return self._provenance_records[content_id]
            
            # In production, this would query a proper database table
            # For now, return None if not found in memory
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get provenance record for {content_id}: {e}")
            return None
    
    async def update_provenance_record(self, content_id: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing provenance record"""
        try:
            if hasattr(self, '_provenance_records') and content_id in self._provenance_records:
                self._provenance_records[content_id].update(update_data)
                self._provenance_records[content_id]['updated_at'] = datetime.now(timezone.utc)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update provenance record for {content_id}: {e}")
            return False
    
    async def store_royalty_distribution_record(self, record: Dict[str, Any]) -> bool:
        """Store royalty distribution record for audit trail"""
        try:
            # For now, store in a simple in-memory registry
            # In production, this would be a proper database table
            if not hasattr(self, '_royalty_distribution_records'):
                self._royalty_distribution_records = []
            
            self._royalty_distribution_records.append(record)
            
            self.logger.info(f"Stored royalty distribution record for session {record['session_id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store royalty distribution record: {e}")
            return False
    
    async def get_creator_earnings(self, creator_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get creator earnings for a specific time period"""
        try:
            # For now, return empty list as this is a placeholder
            # In production, this would query earnings from royalty distribution records
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get creator earnings for {creator_id}: {e}")
            return []
    
    async def record_content_ingestion(self, content_id: str, user_id: str, quality_assessment: Any, reward_amount: float) -> bool:
        """Record successful content ingestion for audit trail"""
        try:
            # For now, store in a simple in-memory registry
            # In production, this would be a proper database table
            if not hasattr(self, '_content_ingestion_records'):
                self._content_ingestion_records = []
            
            ingestion_record = {
                'content_id': content_id,
                'user_id': user_id,
                'quality_level': quality_assessment.quality_level.value if hasattr(quality_assessment, 'quality_level') else 'standard',
                'reward_amount': reward_amount,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self._content_ingestion_records.append(ingestion_record)
            
            self.logger.info(f"Recorded content ingestion for {content_id} by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record content ingestion: {e}")
            return False
    
    # === Health Check ===
    
    async def store_user_api_config(self, user_id: str, provider: str, config: Dict[str, Any]) -> bool:
        """
        Store user API configuration for LLM providers
        
        Args:
            user_id: User identifier
            provider: LLM provider name (e.g., 'claude', 'openai')
            config: Configuration data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # For testing purposes, store in memory
            if not hasattr(self, '_user_api_configs'):
                self._user_api_configs = {}
            
            if user_id not in self._user_api_configs:
                self._user_api_configs[user_id] = {}
                
            self._user_api_configs[user_id][provider] = config
            self.logger.info(f"Stored API config for user {user_id}, provider {provider}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store user API config: {e}")
            return False
    
    async def get_user_api_config(self, user_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user API configuration for a specific provider
        
        Args:
            user_id: User identifier
            provider: LLM provider name
            
        Returns:
            Configuration data or None if not found
        """
        try:
            if hasattr(self, '_user_api_configs'):
                return self._user_api_configs.get(user_id, {}).get(provider)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get user API config: {e}")
            return None

    # === Voicebox Integration Methods ===
    
    async def store_voicebox_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> bool:
        """Store voicebox interaction in database"""
        try:
            # For now, we'll store in a simple dict structure
            # In production, this would use a proper database table
            if not hasattr(self, '_voicebox_interactions'):
                self._voicebox_interactions = {}
            
            if user_id not in self._voicebox_interactions:
                self._voicebox_interactions[user_id] = []
            
            interaction_data['stored_at'] = datetime.now(timezone.utc).isoformat()
            self._voicebox_interactions[user_id].append(interaction_data)
            
            # Keep only last 100 interactions per user
            if len(self._voicebox_interactions[user_id]) > 100:
                self._voicebox_interactions[user_id] = self._voicebox_interactions[user_id][-100:]
            
            self.logger.info(f"Stored voicebox interaction for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store voicebox interaction: {e}")
            return False
    
    async def store_user_api_config(self, user_id: str, provider: str, config_data: Dict[str, Any]) -> bool:
        """Store user API configuration"""
        try:
            # For now, we'll store in a simple dict structure
            # In production, this would use a proper database table with encryption
            if not hasattr(self, '_user_api_configs'):
                self._user_api_configs = {}
            
            if user_id not in self._user_api_configs:
                self._user_api_configs[user_id] = {}
            
            config_data['stored_at'] = datetime.now(timezone.utc).isoformat()
            self._user_api_configs[user_id][provider] = config_data
            
            self.logger.info(f"Stored API config for user {user_id}, provider {provider}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store user API config: {e}")
            return False
    
    async def get_user_voicebox_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's voicebox interaction history"""
        try:
            if not hasattr(self, '_voicebox_interactions'):
                self._voicebox_interactions = {}
            
            user_interactions = self._voicebox_interactions.get(user_id, [])
            
            # Return the most recent interactions
            return user_interactions[-limit:] if user_interactions else []
            
        except Exception as e:
            self.logger.error(f"Failed to get voicebox history for user {user_id}: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            async with get_async_session() as db:
                # Test basic connectivity
                await db.execute(text("SELECT 1"))
                
                # Check table existence
                tables_exist = True
                for model in [PRSMSessionModel, ReasoningStepModel, SafetyFlagModel, ArchitectTaskModel]:
                    try:
                        await db.execute(select(func.count()).select_from(model))
                    except Exception:
                        tables_exist = False
                        break
                
                return {
                    "status": "healthy" if tables_exist else "degraded",
                    "database_connection": "ok",
                    "tables_accessible": tables_exist,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_connection": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# === Global Service Instance ===

_database_service_instance: Optional[DatabaseService] = None

def get_database_service() -> DatabaseService:
    """Get or create the global database service instance"""
    global _database_service_instance
    if _database_service_instance is None:
        _database_service_instance = DatabaseService()
    return _database_service_instance