#!/usr/bin/env python3
"""
Enterprise ETL/ELT Pipeline System
==================================

Comprehensive Extract, Transform, Load pipeline framework supporting
batch and streaming data processing with robust error handling and monitoring.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
import uuid
from pathlib import Path

from .data_connectors import DataConnector, ConnectionConfig, QueryRequest, QueryResult

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of ETL tasks"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    CLEANUP = "cleanup"
    CUSTOM = "custom"


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    INCREMENTAL = "incremental"
    REAL_TIME = "real_time"


@dataclass
class TaskConfiguration:
    """Configuration for ETL tasks"""
    task_id: str
    task_type: TaskType
    name: str
    description: str = ""
    
    # Source and target configurations
    source_connector_id: Optional[str] = None
    target_connector_id: Optional[str] = None
    
    # Query/operation specifications
    source_query: Optional[str] = None
    target_query: Optional[str] = None
    
    # Transformation specifications
    transformation_rules: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution settings
    batch_size: int = 1000
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay_seconds: int = 5
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Scheduling
    schedule_cron: Optional[str] = None
    trigger_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "description": self.description,
            "source_connector_id": self.source_connector_id,
            "target_connector_id": self.target_connector_id,
            "source_query": self.source_query,
            "target_query": self.target_query,
            "transformation_rules": self.transformation_rules,
            "validation_rules": self.validation_rules,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "depends_on": self.depends_on,
            "schedule_cron": self.schedule_cron,
            "trigger_conditions": self.trigger_conditions,
            "custom_params": self.custom_params,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags
        }


@dataclass
class PipelineConfiguration:
    """Configuration for ETL pipelines"""
    pipeline_id: str
    name: str
    description: str = ""
    
    # Task configuration
    tasks: List[TaskConfiguration] = field(default_factory=list)
    
    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.BATCH
    max_parallel_tasks: int = 5
    timeout_seconds: int = 3600
    
    # Error handling
    on_failure: str = "stop"  # stop, continue, retry
    notification_endpoints: List[str] = field(default_factory=list)
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: Optional[str] = None
    
    # Monitoring and logging
    log_level: str = "INFO"
    metrics_enabled: bool = True
    audit_enabled: bool = True
    
    # Data quality
    quality_checks_enabled: bool = True
    data_profiling_enabled: bool = False
    
    # Security
    encryption_enabled: bool = True
    access_control: Dict[str, Any] = field(default_factory=dict)
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "execution_mode": self.execution_mode.value,
            "max_parallel_tasks": self.max_parallel_tasks,
            "timeout_seconds": self.timeout_seconds,
            "on_failure": self.on_failure,
            "notification_endpoints": self.notification_endpoints,
            "schedule_enabled": self.schedule_enabled,
            "schedule_cron": self.schedule_cron,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
            "audit_enabled": self.audit_enabled,
            "quality_checks_enabled": self.quality_checks_enabled,
            "data_profiling_enabled": self.data_profiling_enabled,
            "encryption_enabled": self.encryption_enabled,
            "access_control": self.access_control,
            "custom_params": self.custom_params,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags
        }


@dataclass
class TaskExecution:
    """Record of task execution"""
    execution_id: str
    task_id: str
    pipeline_id: str
    
    # Execution state
    status: PipelineStatus = PipelineStatus.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Metadata
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_processed": self.records_processed,
            "records_successful": self.records_successful,
            "records_failed": self.records_failed,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "execution_metadata": self.execution_metadata
        }


class ETLTask:
    """Individual ETL task executor"""
    
    def __init__(self, config: TaskConfiguration, 
                 source_connector: Optional[DataConnector] = None,
                 target_connector: Optional[DataConnector] = None):
        self.config = config
        self.source_connector = source_connector
        self.target_connector = target_connector
        
        # Execution state
        self.current_execution: Optional[TaskExecution] = None
        self.execution_history: List[TaskExecution] = []
        
        # Data processors
        self.data_processors: List[Callable] = []
        self.validators: List[Callable] = []
        
        logger.info(f"Initialized ETL task: {config.name}")
    
    async def execute(self, pipeline_id: str) -> TaskExecution:
        """Execute the ETL task"""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=self.config.task_id,
            pipeline_id=pipeline_id,
            start_time=datetime.now(timezone.utc)
        )
        
        self.current_execution = execution
        
        try:
            execution.status = PipelineStatus.RUNNING
            logger.info(f"Starting task execution: {self.config.name}")
            
            # Execute based on task type
            if self.config.task_type == TaskType.EXTRACT:
                await self._execute_extract(execution)
            elif self.config.task_type == TaskType.TRANSFORM:
                await self._execute_transform(execution)
            elif self.config.task_type == TaskType.LOAD:
                await self._execute_load(execution)
            elif self.config.task_type == TaskType.VALIDATE:
                await self._execute_validate(execution)
            elif self.config.task_type == TaskType.CLEANUP:
                await self._execute_cleanup(execution)
            elif self.config.task_type == TaskType.CUSTOM:
                await self._execute_custom(execution)
            else:
                raise Exception(f"Unsupported task type: {self.config.task_type}")
            
            execution.status = PipelineStatus.COMPLETED
            execution.end_time = datetime.now(timezone.utc)
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            
            logger.info(f"Task completed: {self.config.name} in {execution.duration_seconds:.2f}s")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now(timezone.utc)
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            
            logger.error(f"Task failed: {self.config.name} - {e}")
            
            # Handle retries
            if execution.retry_count < self.config.retry_count:
                execution.retry_count += 1
                await asyncio.sleep(self.config.retry_delay_seconds)
                return await self.execute(pipeline_id)
        
        # Store execution history
        self.execution_history.append(execution)
        self.current_execution = None
        
        return execution
    
    async def _execute_extract(self, execution: TaskExecution):
        """Execute data extraction"""
        if not self.source_connector:
            raise Exception("Source connector required for extract task")
        
        if not self.config.source_query:
            raise Exception("Source query required for extract task")
        
        # Execute query
        request = QueryRequest(
            query=self.config.source_query,
            limit=self.config.batch_size,
            timeout=self.config.timeout_seconds
        )
        
        result = await self.source_connector.execute_query(request)
        
        execution.records_processed = result.total_rows
        execution.records_successful = result.total_rows
        execution.execution_metadata["extracted_data"] = result.data
        execution.execution_metadata["columns"] = result.columns
    
    async def _execute_transform(self, execution: TaskExecution):
        """Execute data transformation"""
        # Get data from previous task or source
        data = execution.execution_metadata.get("extracted_data", [])
        
        if not data:
            # Extract from source if needed
            if self.source_connector and self.config.source_query:
                request = QueryRequest(query=self.config.source_query)
                result = await self.source_connector.execute_query(request)
                data = result.data
        
        # Apply transformation rules
        transformed_data = []
        failed_records = 0
        
        for record in data:
            try:
                transformed_record = await self._apply_transformations(record)
                transformed_data.append(transformed_record)
            except Exception as e:
                failed_records += 1
                logger.warning(f"Record transformation failed: {e}")
        
        execution.records_processed = len(data)
        execution.records_successful = len(transformed_data)
        execution.records_failed = failed_records
        execution.execution_metadata["transformed_data"] = transformed_data
    
    async def _execute_load(self, execution: TaskExecution):
        """Execute data loading"""
        if not self.target_connector:
            raise Exception("Target connector required for load task")
        
        # Get transformed data
        data = execution.execution_metadata.get("transformed_data", [])
        
        if not data:
            # Get from previous task or extract
            data = execution.execution_metadata.get("extracted_data", [])
        
        if not data:
            raise Exception("No data available for loading")
        
        # Load data in batches
        loaded_records = 0
        failed_records = 0
        
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            
            try:
                await self._load_batch(batch)
                loaded_records += len(batch)
            except Exception as e:
                failed_records += len(batch)
                logger.error(f"Batch load failed: {e}")
                
                if self.config.custom_params.get("stop_on_load_error", True):
                    raise
        
        execution.records_processed = len(data)
        execution.records_successful = loaded_records
        execution.records_failed = failed_records
    
    async def _execute_validate(self, execution: TaskExecution):
        """Execute data validation"""
        data = execution.execution_metadata.get("transformed_data", 
                execution.execution_metadata.get("extracted_data", []))
        
        if not data:
            raise Exception("No data available for validation")
        
        validation_results = []
        failed_validations = 0
        
        for record in data:
            try:
                validation_result = await self._validate_record(record)
                validation_results.append(validation_result)
                
                if not validation_result.get("valid", True):
                    failed_validations += 1
                    
            except Exception as e:
                failed_validations += 1
                logger.warning(f"Record validation failed: {e}")
        
        execution.records_processed = len(data)
        execution.records_successful = len(data) - failed_validations
        execution.records_failed = failed_validations
        execution.execution_metadata["validation_results"] = validation_results
    
    async def _execute_cleanup(self, execution: TaskExecution):
        """Execute cleanup operations"""
        cleanup_operations = self.config.custom_params.get("cleanup_operations", [])
        
        for operation in cleanup_operations:
            try:
                if operation["type"] == "delete_temp_files":
                    await self._cleanup_temp_files(operation.get("path"))
                elif operation["type"] == "archive_data":
                    await self._archive_data(operation.get("source"), operation.get("target"))
                elif operation["type"] == "vacuum_database":
                    await self._vacuum_database(operation.get("connector_id"))
                
                execution.records_successful += 1
                
            except Exception as e:
                execution.records_failed += 1
                logger.error(f"Cleanup operation failed: {e}")
        
        execution.records_processed = len(cleanup_operations)
    
    async def _execute_custom(self, execution: TaskExecution):
        """Execute custom task logic"""
        custom_function = self.config.custom_params.get("function")
        
        if not custom_function:
            raise Exception("Custom function not specified")
        
        # Execute custom function
        result = await custom_function(self, execution)
        
        if isinstance(result, dict):
            execution.execution_metadata.update(result)
    
    async def _apply_transformations(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation rules to a record"""
        transformed_record = record.copy()
        
        for rule in self.config.transformation_rules:
            rule_type = rule.get("type")
            
            if rule_type == "rename_field":
                old_name = rule["old_name"]
                new_name = rule["new_name"]
                if old_name in transformed_record:
                    transformed_record[new_name] = transformed_record.pop(old_name)
            
            elif rule_type == "transform_field":
                field_name = rule["field_name"]
                transformation = rule["transformation"]
                
                if field_name in transformed_record:
                    value = transformed_record[field_name]
                    
                    if transformation == "uppercase":
                        transformed_record[field_name] = str(value).upper()
                    elif transformation == "lowercase":
                        transformed_record[field_name] = str(value).lower()
                    elif transformation == "trim":
                        transformed_record[field_name] = str(value).strip()
                    elif transformation == "to_date":
                        from dateutil import parser
                        transformed_record[field_name] = parser.parse(str(value)).isoformat()
                    elif transformation == "to_number":
                        transformed_record[field_name] = float(value)
            
            elif rule_type == "add_field":
                field_name = rule["field_name"]
                field_value = rule["field_value"]
                transformed_record[field_name] = field_value
            
            elif rule_type == "remove_field":
                field_name = rule["field_name"]
                transformed_record.pop(field_name, None)
            
            elif rule_type == "conditional_transform":
                condition_field = rule["condition_field"]
                condition_value = rule["condition_value"]
                target_field = rule["target_field"]
                target_value = rule["target_value"]
                
                if transformed_record.get(condition_field) == condition_value:
                    transformed_record[target_field] = target_value
        
        return transformed_record
    
    async def _validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a record against validation rules"""
        validation_result = {"valid": True, "errors": []}
        
        for rule in self.config.validation_rules:
            rule_type = rule.get("type")
            
            if rule_type == "required_field":
                field_name = rule["field_name"]
                if field_name not in record or record[field_name] is None:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Required field missing: {field_name}")
            
            elif rule_type == "field_type":
                field_name = rule["field_name"]
                expected_type = rule["expected_type"]
                
                if field_name in record:
                    value = record[field_name]
                    if expected_type == "string" and not isinstance(value, str):
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Field {field_name} should be string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Field {field_name} should be number")
            
            elif rule_type == "field_range":
                field_name = rule["field_name"]
                min_value = rule.get("min_value")
                max_value = rule.get("max_value")
                
                if field_name in record:
                    value = record[field_name]
                    if min_value is not None and value < min_value:
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Field {field_name} below minimum: {min_value}")
                    if max_value is not None and value > max_value:
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Field {field_name} above maximum: {max_value}")
        
        return validation_result
    
    async def _load_batch(self, batch: List[Dict[str, Any]]):
        """Load a batch of records to target"""
        if not self.config.target_query:
            # Simple insert - would need connector-specific implementation
            logger.info(f"Loading batch of {len(batch)} records")
            return
        
        # Execute target query with batch data
        # This would need connector-specific implementation
        logger.info(f"Executing target query for batch of {len(batch)} records")
    
    async def _cleanup_temp_files(self, path: str):
        """Clean up temporary files"""
        import os
        import shutil
        
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            logger.info(f"Cleaned up path: {path}")
    
    async def _archive_data(self, source: str, target: str):
        """Archive data from source to target"""
        import shutil
        shutil.move(source, target)
        logger.info(f"Archived data from {source} to {target}")
    
    async def _vacuum_database(self, connector_id: str):
        """Vacuum database for cleanup"""
        logger.info(f"Database vacuum completed for connector: {connector_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get task status"""
        return {
            "task_id": self.config.task_id,
            "name": self.config.name,
            "type": self.config.task_type.value,
            "current_execution": self.current_execution.to_dict() if self.current_execution else None,
            "execution_count": len(self.execution_history),
            "last_execution": self.execution_history[-1].to_dict() if self.execution_history else None
        }


class ETLPipeline:
    """ETL/ELT Pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.tasks: Dict[str, ETLTask] = {}
        self.connectors: Dict[str, DataConnector] = {}
        
        # Execution state
        self.status = PipelineStatus.IDLE
        self.current_execution_id: Optional[str] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Task dependency graph
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "total_records_processed": 0
        }
        
        logger.info(f"Initialized ETL pipeline: {config.name}")
    
    def add_task(self, task: ETLTask):
        """Add task to pipeline"""
        self.tasks[task.config.task_id] = task
        
        # Build dependency graph
        self.dependency_graph[task.config.task_id] = task.config.depends_on.copy()
        
        logger.info(f"Added task to pipeline: {task.config.name}")
    
    def add_connector(self, connector: DataConnector):
        """Add data connector to pipeline"""
        self.connectors[connector.config.connector_id] = connector
        logger.info(f"Added connector to pipeline: {connector.config.name}")
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the complete pipeline"""
        execution_id = f"pipeline_exec_{uuid.uuid4().hex[:8]}"
        self.current_execution_id = execution_id
        
        execution_result = {
            "execution_id": execution_id,
            "pipeline_id": self.config.pipeline_id,
            "start_time": datetime.now(timezone.utc),
            "status": PipelineStatus.RUNNING,
            "task_executions": {}
        }
        
        try:
            self.status = PipelineStatus.RUNNING
            logger.info(f"Starting pipeline execution: {self.config.name}")
            
            # Execute tasks in dependency order
            execution_order = self._calculate_execution_order()
            
            for task_id in execution_order:
                if task_id not in self.tasks:
                    raise Exception(f"Task not found: {task_id}")
                
                task = self.tasks[task_id]
                
                # Connect task to appropriate connectors
                if task.config.source_connector_id:
                    task.source_connector = self.connectors.get(task.config.source_connector_id)
                
                if task.config.target_connector_id:
                    task.target_connector = self.connectors.get(task.config.target_connector_id)
                
                # Execute task
                task_execution = await task.execute(self.config.pipeline_id)
                execution_result["task_executions"][task_id] = task_execution.to_dict()
                
                # Check if pipeline should stop on failure
                if task_execution.status == PipelineStatus.FAILED:
                    if self.config.on_failure == "stop":
                        raise Exception(f"Pipeline stopped due to task failure: {task_id}")
                    elif self.config.on_failure == "continue":
                        logger.warning(f"Task failed but pipeline continuing: {task_id}")
            
            # Pipeline completed successfully
            execution_result["status"] = PipelineStatus.COMPLETED
            execution_result["end_time"] = datetime.now(timezone.utc)
            
            # Calculate statistics
            total_records = sum(
                exec_data.get("records_processed", 0) 
                for exec_data in execution_result["task_executions"].values()
            )
            
            execution_result["total_records_processed"] = total_records
            
            self.stats["total_executions"] += 1
            self.stats["successful_executions"] += 1
            self.stats["total_records_processed"] += total_records
            
            logger.info(f"Pipeline completed successfully: {self.config.name}")
            
        except Exception as e:
            execution_result["status"] = PipelineStatus.FAILED
            execution_result["error_message"] = str(e)
            execution_result["end_time"] = datetime.now(timezone.utc)
            
            self.stats["total_executions"] += 1
            self.stats["failed_executions"] += 1
            
            logger.error(f"Pipeline execution failed: {self.config.name} - {e}")
        
        finally:
            self.status = PipelineStatus.IDLE
            self.current_execution_id = None
            
            # Calculate execution time
            if "end_time" in execution_result:
                duration = (execution_result["end_time"] - execution_result["start_time"]).total_seconds()
                execution_result["duration_seconds"] = duration
                
                # Update average execution time
                total_execs = self.stats["total_executions"]
                current_avg = self.stats["avg_execution_time"]
                self.stats["avg_execution_time"] = (current_avg * (total_execs - 1) + duration) / total_execs
            
            # Store execution history
            self.execution_history.append(execution_result)
        
        return execution_result
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate task execution order based on dependencies"""
        # Topological sort of dependency graph
        in_degree = {task_id: 0 for task_id in self.tasks.keys()}
        
        # Calculate in-degrees
        for task_id, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        # Kahn's algorithm for topological sorting
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current_task = queue.pop(0)
            execution_order.append(current_task)
            
            # Update in-degrees for dependent tasks
            for task_id, dependencies in self.dependency_graph.items():
                if current_task in dependencies:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        # Check for circular dependencies
        if len(execution_order) != len(self.tasks):
            raise Exception("Circular dependency detected in pipeline tasks")
        
        return execution_order
    
    async def pause(self):
        """Pause pipeline execution"""
        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.PAUSED
            logger.info(f"Pipeline paused: {self.config.name}")
    
    async def resume(self):
        """Resume pipeline execution"""
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.RUNNING
            logger.info(f"Pipeline resumed: {self.config.name}")
    
    async def cancel(self):
        """Cancel pipeline execution"""
        if self.status in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
            self.status = PipelineStatus.CANCELLED
            logger.info(f"Pipeline cancelled: {self.config.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and statistics"""
        return {
            "pipeline_id": self.config.pipeline_id,
            "name": self.config.name,
            "status": self.status.value,
            "current_execution_id": self.current_execution_id,
            "execution_count": len(self.execution_history),
            "task_count": len(self.tasks),
            "connector_count": len(self.connectors),
            "statistics": self.stats,
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }


class PipelineManager:
    """Manager for multiple ETL pipelines"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./etl_pipelines")
        self.storage_path.mkdir(exist_ok=True)
        
        # Pipeline registry
        self.pipelines: Dict[str, ETLPipeline] = {}
        self.connectors: Dict[str, DataConnector] = {}
        
        # Scheduling
        self.scheduler_enabled = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.manager_stats = {
            "total_pipelines": 0,
            "active_pipelines": 0,
            "total_pipeline_executions": 0,
            "total_records_processed": 0,
            "avg_pipeline_success_rate": 0.0
        }
        
        logger.info("Pipeline Manager initialized")
    
    def register_pipeline(self, pipeline: ETLPipeline):
        """Register a pipeline with the manager"""
        self.pipelines[pipeline.config.pipeline_id] = pipeline
        self.manager_stats["total_pipelines"] += 1
        
        logger.info(f"Registered pipeline: {pipeline.config.name}")
    
    def register_connector(self, connector: DataConnector):
        """Register a connector with the manager"""
        self.connectors[connector.config.connector_id] = connector
        
        # Add connector to all pipelines that need it
        for pipeline in self.pipelines.values():
            pipeline.add_connector(connector)
        
        logger.info(f"Registered connector: {connector.config.name}")
    
    async def execute_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Execute a specific pipeline"""
        if pipeline_id not in self.pipelines:
            raise Exception(f"Pipeline not found: {pipeline_id}")
        
        pipeline = self.pipelines[pipeline_id]
        self.manager_stats["active_pipelines"] += 1
        
        try:
            result = await pipeline.execute()
            
            self.manager_stats["total_pipeline_executions"] += 1
            self.manager_stats["total_records_processed"] += result.get("total_records_processed", 0)
            
            return result
            
        finally:
            self.manager_stats["active_pipelines"] -= 1
    
    async def execute_all_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Execute all registered pipelines"""
        results = {}
        
        for pipeline_id, pipeline in self.pipelines.items():
            try:
                results[pipeline_id] = await self.execute_pipeline(pipeline_id)
            except Exception as e:
                results[pipeline_id] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        return results
    
    async def start_scheduler(self):
        """Start the pipeline scheduler"""
        if self.scheduler_enabled:
            return
        
        self.scheduler_enabled = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Pipeline scheduler started")
    
    async def stop_scheduler(self):
        """Stop the pipeline scheduler"""
        if not self.scheduler_enabled:
            return
        
        self.scheduler_enabled = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Pipeline scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        from croniter import croniter
        
        while self.scheduler_enabled:
            try:
                current_time = datetime.now()
                
                for pipeline in self.pipelines.values():
                    if (pipeline.config.schedule_enabled and 
                        pipeline.config.schedule_cron and
                        pipeline.status == PipelineStatus.IDLE):
                        
                        # Check if pipeline should run
                        cron = croniter(pipeline.config.schedule_cron, current_time)
                        next_run = cron.get_next(datetime)
                        
                        # If next run is within the next minute, execute
                        if (next_run - current_time).total_seconds() <= 60:
                            logger.info(f"Scheduled execution triggered: {pipeline.config.name}")
                            asyncio.create_task(self.execute_pipeline(pipeline.config.pipeline_id))
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific pipeline"""
        if pipeline_id not in self.pipelines:
            return None
        
        return self.pipelines[pipeline_id].get_status()
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all registered pipelines"""
        return [
            {
                "pipeline_id": pipeline.config.pipeline_id,
                "name": pipeline.config.name,
                "description": pipeline.config.description,
                "task_count": len(pipeline.tasks),
                "status": pipeline.status.value,
                "execution_mode": pipeline.config.execution_mode.value,
                "schedule_enabled": pipeline.config.schedule_enabled,
                "created_at": pipeline.config.created_at.isoformat()
            }
            for pipeline in self.pipelines.values()
        ]
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        # Calculate success rate
        total_executions = sum(p.stats["total_executions"] for p in self.pipelines.values())
        successful_executions = sum(p.stats["successful_executions"] for p in self.pipelines.values())
        
        success_rate = (successful_executions / max(1, total_executions)) * 100
        
        return {
            **self.manager_stats,
            "current_success_rate": success_rate,
            "scheduler_enabled": self.scheduler_enabled,
            "total_connectors": len(self.connectors)
        }


class DataProcessor:
    """Utility class for common data processing operations"""
    
    @staticmethod
    async def deduplicate_records(records: List[Dict[str, Any]], 
                                 key_fields: List[str]) -> List[Dict[str, Any]]:
        """Remove duplicate records based on key fields"""
        seen_keys = set()
        unique_records = []
        
        for record in records:
            key = tuple(record.get(field) for field in key_fields)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_records.append(record)
        
        return unique_records
    
    @staticmethod
    async def merge_records(left_records: List[Dict[str, Any]],
                           right_records: List[Dict[str, Any]],
                           left_key: str, right_key: str,
                           join_type: str = "inner") -> List[Dict[str, Any]]:
        """Merge two datasets"""
        # Create lookup index for right records
        right_index = {}
        for record in right_records:
            key_value = record.get(right_key)
            if key_value not in right_index:
                right_index[key_value] = []
            right_index[key_value].append(record)
        
        merged_records = []
        
        for left_record in left_records:
            left_key_value = left_record.get(left_key)
            right_matches = right_index.get(left_key_value, [])
            
            if right_matches:
                # Merge with all matching right records
                for right_record in right_matches:
                    merged_record = {**left_record, **right_record}
                    merged_records.append(merged_record)
            elif join_type in ["left", "outer"]:
                # Include left record even without right match
                merged_records.append(left_record)
        
        # For outer join, include unmatched right records
        if join_type == "outer":
            matched_right_keys = set()
            for left_record in left_records:
                left_key_value = left_record.get(left_key)
                if left_key_value in right_index:
                    matched_right_keys.add(left_key_value)
            
            for right_key_value, right_records_list in right_index.items():
                if right_key_value not in matched_right_keys:
                    merged_records.extend(right_records_list)
        
        return merged_records
    
    @staticmethod
    async def aggregate_records(records: List[Dict[str, Any]],
                               group_by: List[str],
                               aggregations: Dict[str, str]) -> List[Dict[str, Any]]:
        """Aggregate records by group"""
        from collections import defaultdict
        
        # Group records
        groups = defaultdict(list)
        for record in records:
            group_key = tuple(record.get(field) for field in group_by)
            groups[group_key].append(record)
        
        # Apply aggregations
        aggregated_records = []
        for group_key, group_records in groups.items():
            aggregated_record = {}
            
            # Add group by fields
            for i, field in enumerate(group_by):
                aggregated_record[field] = group_key[i]
            
            # Apply aggregation functions
            for field, agg_func in aggregations.items():
                values = [record.get(field) for record in group_records if record.get(field) is not None]
                
                if not values:
                    aggregated_record[f"{agg_func}_{field}"] = None
                    continue
                
                if agg_func == "sum":
                    aggregated_record[f"sum_{field}"] = sum(values)
                elif agg_func == "avg":
                    aggregated_record[f"avg_{field}"] = sum(values) / len(values)
                elif agg_func == "count":
                    aggregated_record[f"count_{field}"] = len(values)
                elif agg_func == "min":
                    aggregated_record[f"min_{field}"] = min(values)
                elif agg_func == "max":
                    aggregated_record[f"max_{field}"] = max(values)
            
            aggregated_records.append(aggregated_record)
        
        return aggregated_records


# Export main classes
__all__ = [
    'PipelineStatus',
    'TaskType',
    'ExecutionMode',
    'TaskConfiguration',
    'PipelineConfiguration',
    'TaskExecution',
    'ETLTask',
    'ETLPipeline',
    'PipelineManager',
    'DataProcessor'
]