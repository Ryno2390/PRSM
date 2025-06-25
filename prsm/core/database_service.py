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
                reasoning_step = ReasoningStepModel(
                    step_id=step_id,
                    session_id=session_id,
                    agent_type=step_data.get("agent_type", "default"),
                    agent_id=step_data.get("agent_id", "unknown"),
                    input_data=step_data.get("input_data", {}),
                    output_data=step_data.get("output_data", {}),
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
    
    # === Health Check ===
    
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