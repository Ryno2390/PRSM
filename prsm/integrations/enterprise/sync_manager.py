#!/usr/bin/env python3
"""
Real-Time Data Synchronization Manager
======================================

Advanced data synchronization system supporting real-time, batch, and hybrid
synchronization patterns with conflict resolution and change tracking.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Set
import uuid
import hashlib
from pathlib import Path

from .data_connectors import DataConnector, QueryRequest, QueryResult
from .transformation_engine import TransformationEngine, TransformationRule

logger = logging.getLogger(__name__)


class SyncType(Enum):
    """Types of data synchronization"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    INCREMENTAL = "incremental"
    BIDIRECTIONAL = "bidirectional"
    MASTER_SLAVE = "master_slave"
    PEER_TO_PEER = "peer_to_peer"


class SyncStatus(Enum):
    """Synchronization status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    NEWEST_WINS = "newest_wins"
    CUSTOM_LOGIC = "custom_logic"
    MANUAL_RESOLUTION = "manual_resolution"
    MERGE_FIELDS = "merge_fields"


class ChangeType(Enum):
    """Types of data changes"""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SCHEMA_CHANGE = "schema_change"


@dataclass
class SyncConfiguration:
    """Configuration for data synchronization"""
    sync_id: str
    name: str
    description: str = ""
    
    # Source and target
    source_connector_id: str
    target_connector_id: str
    
    # Sync settings
    sync_type: SyncType = SyncType.BATCH
    sync_direction: str = "source_to_target"  # source_to_target, target_to_source, bidirectional
    
    # Data selection
    source_query: Optional[str] = None
    target_query: Optional[str] = None
    filter_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Key fields for matching records
    key_fields: List[str] = field(default_factory=list)
    
    # Transformation rules
    transformation_rules: List[str] = field(default_factory=list)
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: Optional[str] = None
    sync_interval_seconds: int = 300
    
    # Batch settings
    batch_size: int = 1000
    max_parallel_batches: int = 5
    
    # Real-time settings
    change_detection_method: str = "timestamp"  # timestamp, checksum, trigger
    change_tracking_field: Optional[str] = None
    
    # Conflict resolution
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.SOURCE_WINS
    custom_conflict_resolver: Optional[Callable] = None
    
    # Error handling
    on_error: str = "stop"  # stop, continue, retry
    max_retries: int = 3
    retry_delay_seconds: int = 30
    
    # Data quality
    validate_before_sync: bool = True
    checksum_validation: bool = True
    
    # Performance
    use_bulk_operations: bool = True
    connection_pooling: bool = True
    
    # Monitoring
    log_level: str = "INFO"
    metrics_enabled: bool = True
    
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
            "sync_id": self.sync_id,
            "name": self.name,
            "description": self.description,
            "source_connector_id": self.source_connector_id,
            "target_connector_id": self.target_connector_id,
            "sync_type": self.sync_type.value,
            "sync_direction": self.sync_direction,
            "source_query": self.source_query,
            "target_query": self.target_query,
            "filter_conditions": self.filter_conditions,
            "key_fields": self.key_fields,
            "transformation_rules": self.transformation_rules,
            "schedule_enabled": self.schedule_enabled,
            "schedule_cron": self.schedule_cron,
            "sync_interval_seconds": self.sync_interval_seconds,
            "batch_size": self.batch_size,
            "max_parallel_batches": self.max_parallel_batches,
            "change_detection_method": self.change_detection_method,
            "change_tracking_field": self.change_tracking_field,
            "conflict_resolution": self.conflict_resolution.value,
            "on_error": self.on_error,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "validate_before_sync": self.validate_before_sync,
            "checksum_validation": self.checksum_validation,
            "use_bulk_operations": self.use_bulk_operations,
            "connection_pooling": self.connection_pooling,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
            "custom_params": self.custom_params,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags
        }


@dataclass
class ChangeRecord:
    """Record of a data change"""
    change_id: str
    change_type: ChangeType
    table_name: str
    record_key: str
    
    # Change data
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_system: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "table_name": self.table_name,
            "record_key": self.record_key,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "timestamp": self.timestamp.isoformat(),
            "source_system": self.source_system,
            "checksum": self.checksum
        }


@dataclass
class SyncExecution:
    """Record of sync execution"""
    execution_id: str
    sync_id: str
    
    # Execution state
    status: SyncStatus = SyncStatus.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    records_failed: int = 0
    
    # Conflicts
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Metadata
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution_id": self.execution_id,
            "sync_id": self.sync_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_processed": self.records_processed,
            "records_inserted": self.records_inserted,
            "records_updated": self.records_updated,
            "records_deleted": self.records_deleted,
            "records_failed": self.records_failed,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "execution_metadata": self.execution_metadata
        }


class ChangeTracker:
    """Change tracking system"""
    
    def __init__(self):
        self.change_log: List[ChangeRecord] = []
        self.change_index: Dict[str, ChangeRecord] = {}
        self.checksum_cache: Dict[str, str] = {}
        self.last_sync_timestamps: Dict[str, datetime] = {}
    
    def track_change(self, change: ChangeRecord):
        """Track a data change"""
        self.change_log.append(change)
        self.change_index[change.change_id] = change
        
        # Limit change log size
        if len(self.change_log) > 100000:
            # Remove oldest changes
            removed_changes = self.change_log[:10000]
            self.change_log = self.change_log[10000:]
            
            for removed_change in removed_changes:
                self.change_index.pop(removed_change.change_id, None)
    
    async def detect_changes(self, connector: DataConnector, config: SyncConfiguration,
                           last_sync_time: Optional[datetime] = None) -> List[ChangeRecord]:
        """Detect changes in data source"""
        changes = []
        
        if config.change_detection_method == "timestamp":
            changes = await self._detect_timestamp_changes(connector, config, last_sync_time)
        elif config.change_detection_method == "checksum":
            changes = await self._detect_checksum_changes(connector, config)
        elif config.change_detection_method == "trigger":
            changes = await self._detect_trigger_changes(connector, config)
        
        # Track detected changes
        for change in changes:
            self.track_change(change)
        
        return changes
    
    async def _detect_timestamp_changes(self, connector: DataConnector, 
                                       config: SyncConfiguration,
                                       last_sync_time: Optional[datetime]) -> List[ChangeRecord]:
        """Detect changes using timestamp comparison"""
        changes = []
        
        if not config.change_tracking_field:
            logger.warning("Change tracking field not specified for timestamp-based detection")
            return changes
        
        # Build query to get changed records
        base_query = config.source_query or f"SELECT * FROM {config.custom_params.get('table_name', 'data')}"
        
        if last_sync_time:
            timestamp_filter = f" WHERE {config.change_tracking_field} > '{last_sync_time.isoformat()}'"
            query = base_query + timestamp_filter
        else:
            query = base_query
        
        try:
            request = QueryRequest(query=query, limit=config.batch_size)
            result = await connector.execute_query(request)
            
            for record in result.data:
                change_id = f"change_{uuid.uuid4().hex[:8]}"
                record_key = self._generate_record_key(record, config.key_fields)
                
                change = ChangeRecord(
                    change_id=change_id,
                    change_type=ChangeType.UPDATE,  # Assume update for timestamp-based
                    table_name=config.custom_params.get('table_name', 'data'),
                    record_key=record_key,
                    new_values=record,
                    source_system=config.source_connector_id
                )
                
                changes.append(change)
                
        except Exception as e:
            logger.error(f"Error detecting timestamp changes: {e}")
        
        return changes
    
    async def _detect_checksum_changes(self, connector: DataConnector, 
                                      config: SyncConfiguration) -> List[ChangeRecord]:
        """Detect changes using checksum comparison"""
        changes = []
        
        try:
            # Get current data
            request = QueryRequest(query=config.source_query, limit=config.batch_size)
            result = await connector.execute_query(request)
            
            for record in result.data:
                record_key = self._generate_record_key(record, config.key_fields)
                current_checksum = self._calculate_checksum(record)
                
                # Compare with cached checksum
                cached_checksum = self.checksum_cache.get(record_key)
                
                if cached_checksum != current_checksum:
                    change_type = ChangeType.UPDATE if cached_checksum else ChangeType.INSERT
                    
                    change = ChangeRecord(
                        change_id=f"change_{uuid.uuid4().hex[:8]}",
                        change_type=change_type,
                        table_name=config.custom_params.get('table_name', 'data'),
                        record_key=record_key,
                        new_values=record,
                        checksum=current_checksum,
                        source_system=config.source_connector_id
                    )
                    
                    changes.append(change)
                    
                    # Update checksum cache
                    self.checksum_cache[record_key] = current_checksum
                    
        except Exception as e:
            logger.error(f"Error detecting checksum changes: {e}")
        
        return changes
    
    async def _detect_trigger_changes(self, connector: DataConnector, 
                                     config: SyncConfiguration) -> List[ChangeRecord]:
        """Detect changes using database triggers (change log table)"""
        changes = []
        
        # Query change log table
        change_log_table = config.custom_params.get('change_log_table', 'change_log')
        last_sync_time = self.last_sync_timestamps.get(config.sync_id)
        
        if last_sync_time:
            query = f"""
            SELECT * FROM {change_log_table} 
            WHERE timestamp > '{last_sync_time.isoformat()}'
            ORDER BY timestamp ASC
            """
        else:
            query = f"SELECT * FROM {change_log_table} ORDER BY timestamp ASC"
        
        try:
            request = QueryRequest(query=query, limit=config.batch_size)
            result = await connector.execute_query(request)
            
            for record in result.data:
                change_type_str = record.get('change_type', 'UPDATE').upper()
                change_type = ChangeType[change_type_str] if change_type_str in ChangeType.__members__ else ChangeType.UPDATE
                
                change = ChangeRecord(
                    change_id=record.get('change_id', f"change_{uuid.uuid4().hex[:8]}"),
                    change_type=change_type,
                    table_name=record.get('table_name', 'data'),
                    record_key=record.get('record_key', ''),
                    old_values=json.loads(record.get('old_values', '{}')) if record.get('old_values') else None,
                    new_values=json.loads(record.get('new_values', '{}')) if record.get('new_values') else None,
                    timestamp=datetime.fromisoformat(record.get('timestamp', datetime.now().isoformat())),
                    source_system=config.source_connector_id
                )
                
                changes.append(change)
                
        except Exception as e:
            logger.error(f"Error detecting trigger changes: {e}")
        
        return changes
    
    def _generate_record_key(self, record: Dict[str, Any], key_fields: List[str]) -> str:
        """Generate unique key for record"""
        if not key_fields:
            # Use all fields to generate key
            key_data = json.dumps(record, sort_keys=True)
        else:
            # Use specified key fields
            key_values = [str(record.get(field, '')) for field in key_fields]
            key_data = '|'.join(key_values)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_checksum(self, record: Dict[str, Any]) -> str:
        """Calculate checksum for record"""
        record_json = json.dumps(record, sort_keys=True)
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    def get_changes_since(self, timestamp: datetime) -> List[ChangeRecord]:
        """Get changes since specified timestamp"""
        return [change for change in self.change_log if change.timestamp >= timestamp]
    
    def clear_old_changes(self, older_than: datetime):
        """Clear changes older than specified timestamp"""
        self.change_log = [change for change in self.change_log if change.timestamp >= older_than]
        
        # Update change index
        self.change_index = {change.change_id: change for change in self.change_log}


class ConflictResolver:
    """Conflict resolution system"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictResolutionStrategy.SOURCE_WINS: self._source_wins,
            ConflictResolutionStrategy.TARGET_WINS: self._target_wins,
            ConflictResolutionStrategy.NEWEST_WINS: self._newest_wins,
            ConflictResolutionStrategy.MERGE_FIELDS: self._merge_fields
        }
    
    async def resolve_conflict(self, source_record: Dict[str, Any], 
                              target_record: Dict[str, Any],
                              strategy: ConflictResolutionStrategy,
                              custom_resolver: Optional[Callable] = None) -> Dict[str, Any]:
        """Resolve conflict between source and target records"""
        
        if strategy == ConflictResolutionStrategy.CUSTOM_LOGIC and custom_resolver:
            return await custom_resolver(source_record, target_record)
        elif strategy in self.resolution_strategies:
            return await self.resolution_strategies[strategy](source_record, target_record)
        else:
            logger.warning(f"Unknown conflict resolution strategy: {strategy}")
            return source_record  # Default to source wins
    
    async def _source_wins(self, source_record: Dict[str, Any], 
                          target_record: Dict[str, Any]) -> Dict[str, Any]:
        """Source record takes precedence"""
        return source_record
    
    async def _target_wins(self, source_record: Dict[str, Any], 
                          target_record: Dict[str, Any]) -> Dict[str, Any]:
        """Target record takes precedence"""
        return target_record
    
    async def _newest_wins(self, source_record: Dict[str, Any], 
                          target_record: Dict[str, Any]) -> Dict[str, Any]:
        """Most recently modified record wins"""
        source_timestamp = self._extract_timestamp(source_record)
        target_timestamp = self._extract_timestamp(target_record)
        
        if source_timestamp and target_timestamp:
            return source_record if source_timestamp > target_timestamp else target_record
        elif source_timestamp:
            return source_record
        elif target_timestamp:
            return target_record
        else:
            return source_record  # Default to source if no timestamps
    
    async def _merge_fields(self, source_record: Dict[str, Any], 
                           target_record: Dict[str, Any]) -> Dict[str, Any]:
        """Merge non-conflicting fields"""
        merged_record = target_record.copy()
        
        for field, value in source_record.items():
            if field not in merged_record or merged_record[field] is None:
                merged_record[field] = value
            elif value is not None and merged_record[field] != value:
                # Field conflict - could implement field-specific resolution
                merged_record[field] = value  # Default to source value
        
        return merged_record
    
    def _extract_timestamp(self, record: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from record"""
        timestamp_fields = ['updated_at', 'modified_date', 'timestamp', 'last_modified']
        
        for field in timestamp_fields:
            if field in record:
                try:
                    if isinstance(record[field], datetime):
                        return record[field]
                    elif isinstance(record[field], str):
                        return datetime.fromisoformat(record[field])
                except:
                    continue
        
        return None


class RealtimeSync:
    """Real-time synchronization implementation"""
    
    def __init__(self, config: SyncConfiguration, 
                 source_connector: DataConnector,
                 target_connector: DataConnector,
                 change_tracker: ChangeTracker,
                 transformation_engine: Optional[TransformationEngine] = None):
        self.config = config
        self.source_connector = source_connector
        self.target_connector = target_connector
        self.change_tracker = change_tracker
        self.transformation_engine = transformation_engine
        self.conflict_resolver = ConflictResolver()
        
        # State
        self.status = SyncStatus.IDLE
        self.sync_task: Optional[asyncio.Task] = None
        self.last_sync_time: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            "total_changes_processed": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "conflicts_resolved": 0,
            "avg_sync_latency_ms": 0.0
        }
    
    async def start(self):
        """Start real-time synchronization"""
        if self.status == SyncStatus.RUNNING:
            logger.warning("Real-time sync already running")
            return
        
        self.status = SyncStatus.RUNNING
        self.sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info(f"Started real-time sync: {self.config.name}")
    
    async def stop(self):
        """Stop real-time synchronization"""
        if self.status != SyncStatus.RUNNING:
            return
        
        self.status = SyncStatus.CANCELLED
        
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped real-time sync: {self.config.name}")
    
    async def _sync_loop(self):
        """Main synchronization loop"""
        while self.status == SyncStatus.RUNNING:
            try:
                sync_start_time = datetime.now()
                
                # Detect changes
                changes = await self.change_tracker.detect_changes(
                    self.source_connector, self.config, self.last_sync_time
                )
                
                if changes:
                    logger.info(f"Processing {len(changes)} changes for sync {self.config.name}")
                    
                    # Process changes
                    await self._process_changes(changes)
                    
                    # Update statistics
                    sync_latency = (datetime.now() - sync_start_time).total_seconds() * 1000
                    self._update_sync_stats(sync_latency, True, len(changes))
                
                # Update last sync time
                self.last_sync_time = sync_start_time
                
                # Wait before next check
                await asyncio.sleep(self.config.sync_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Real-time sync error: {e}")
                self._update_sync_stats(0, False, 0)
                
                # Wait before retrying
                await asyncio.sleep(self.config.retry_delay_seconds)
    
    async def _process_changes(self, changes: List[ChangeRecord]):
        """Process detected changes"""
        processed_count = 0
        
        for change in changes:
            try:
                await self._process_single_change(change)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process change {change.change_id}: {e}")
                
                if self.config.on_error == "stop":
                    raise
                elif self.config.on_error == "continue":
                    continue
                # For retry, would implement retry logic here
        
        self.stats["total_changes_processed"] += processed_count
    
    async def _process_single_change(self, change: ChangeRecord):
        """Process a single change"""
        if change.change_type == ChangeType.INSERT:
            await self._handle_insert(change)
        elif change.change_type == ChangeType.UPDATE:
            await self._handle_update(change)
        elif change.change_type == ChangeType.DELETE:
            await self._handle_delete(change)
    
    async def _handle_insert(self, change: ChangeRecord):
        """Handle record insertion"""
        record = change.new_values
        if not record:
            return
        
        # Apply transformations
        if self.transformation_engine and self.config.transformation_rules:
            transform_result = await self.transformation_engine.transform_record(
                record, self.config.transformation_rules
            )
            
            if transform_result.success:
                record = transform_result.transformed_data
            else:
                logger.error(f"Transformation failed for insert: {transform_result.errors}")
                return
        
        # Insert into target
        # This would need connector-specific implementation
        logger.info(f"Inserting record with key {change.record_key}")
    
    async def _handle_update(self, change: ChangeRecord):
        """Handle record update"""
        record = change.new_values
        if not record:
            return
        
        # Check for conflicts in target
        target_record = await self._get_target_record(change.record_key)
        
        if target_record:
            # Resolve conflict
            resolved_record = await self.conflict_resolver.resolve_conflict(
                record, target_record, self.config.conflict_resolution,
                self.config.custom_conflict_resolver
            )
            
            self.stats["conflicts_resolved"] += 1
            record = resolved_record
        
        # Apply transformations
        if self.transformation_engine and self.config.transformation_rules:
            transform_result = await self.transformation_engine.transform_record(
                record, self.config.transformation_rules
            )
            
            if transform_result.success:
                record = transform_result.transformed_data
            else:
                logger.error(f"Transformation failed for update: {transform_result.errors}")
                return
        
        # Update target
        logger.info(f"Updating record with key {change.record_key}")
    
    async def _handle_delete(self, change: ChangeRecord):
        """Handle record deletion"""
        # Delete from target
        logger.info(f"Deleting record with key {change.record_key}")
    
    async def _get_target_record(self, record_key: str) -> Optional[Dict[str, Any]]:
        """Get record from target system"""
        # This would need connector-specific implementation
        # For now, return None (no existing record)
        return None
    
    def _update_sync_stats(self, latency_ms: float, success: bool, changes_count: int):
        """Update synchronization statistics"""
        if success:
            self.stats["successful_syncs"] += 1
        else:
            self.stats["failed_syncs"] += 1
        
        # Update average latency
        total_syncs = self.stats["successful_syncs"] + self.stats["failed_syncs"]
        current_avg = self.stats["avg_sync_latency_ms"]
        self.stats["avg_sync_latency_ms"] = \
            (current_avg * (total_syncs - 1) + latency_ms) / total_syncs
    
    def get_status(self) -> Dict[str, Any]:
        """Get sync status"""
        return {
            "sync_id": self.config.sync_id,
            "name": self.config.name,
            "status": self.status.value,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "statistics": self.stats
        }


class BatchSync:
    """Batch synchronization implementation"""
    
    def __init__(self, config: SyncConfiguration,
                 source_connector: DataConnector,
                 target_connector: DataConnector,
                 transformation_engine: Optional[TransformationEngine] = None):
        self.config = config
        self.source_connector = source_connector
        self.target_connector = target_connector
        self.transformation_engine = transformation_engine
        self.conflict_resolver = ConflictResolver()
        
        # State
        self.status = SyncStatus.IDLE
        
        # Statistics
        self.stats = {
            "total_batches_processed": 0,
            "total_records_synced": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "avg_batch_time_ms": 0.0
        }
    
    async def execute(self) -> SyncExecution:
        """Execute batch synchronization"""
        execution_id = f"batch_exec_{uuid.uuid4().hex[:8]}"
        
        execution = SyncExecution(
            execution_id=execution_id,
            sync_id=self.config.sync_id,
            start_time=datetime.now(timezone.utc)
        )
        
        try:
            self.status = SyncStatus.RUNNING
            execution.status = SyncStatus.RUNNING
            
            logger.info(f"Starting batch sync: {self.config.name}")
            
            # Get source data
            source_data = await self._get_source_data()
            execution.records_processed = len(source_data)
            
            # Process in batches
            batch_size = self.config.batch_size
            total_batches = (len(source_data) + batch_size - 1) // batch_size
            
            for i in range(0, len(source_data), batch_size):
                batch = source_data[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")
                
                batch_start_time = datetime.now()
                
                try:
                    batch_result = await self._process_batch(batch)
                    
                    execution.records_inserted += batch_result.get("inserted", 0)
                    execution.records_updated += batch_result.get("updated", 0)
                    execution.records_deleted += batch_result.get("deleted", 0)
                    execution.conflicts_resolved += batch_result.get("conflicts_resolved", 0)
                    
                    # Update batch statistics
                    batch_time = (datetime.now() - batch_start_time).total_seconds() * 1000
                    self._update_batch_stats(batch_time, True)
                    
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    execution.records_failed += len(batch)
                    self._update_batch_stats(0, False)
                    
                    if self.config.on_error == "stop":
                        raise
            
            execution.status = SyncStatus.COMPLETED
            execution.end_time = datetime.now(timezone.utc)
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            
            logger.info(f"Batch sync completed: {self.config.name} - {execution.records_processed} records processed")
            
        except Exception as e:
            execution.status = SyncStatus.ERROR
            execution.error_message = str(e)
            execution.end_time = datetime.now(timezone.utc)
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            
            logger.error(f"Batch sync failed: {self.config.name} - {e}")
        
        finally:
            self.status = SyncStatus.IDLE
        
        return execution
    
    async def _get_source_data(self) -> List[Dict[str, Any]]:
        """Get data from source"""
        request = QueryRequest(query=self.config.source_query)
        result = await self.source_connector.execute_query(request)
        return result.data
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process a batch of records"""
        results = {
            "inserted": 0,
            "updated": 0,
            "deleted": 0,
            "conflicts_resolved": 0
        }
        
        for record in batch:
            try:
                # Apply transformations
                if self.transformation_engine and self.config.transformation_rules:
                    transform_result = await self.transformation_engine.transform_record(
                        record, self.config.transformation_rules
                    )
                    
                    if transform_result.success:
                        record = transform_result.transformed_data
                    else:
                        logger.error(f"Transformation failed: {transform_result.errors}")
                        continue
                
                # Check if record exists in target
                record_key = self._generate_record_key(record)
                target_record = await self._get_target_record(record_key)
                
                if target_record:
                    # Record exists - update
                    if self._records_differ(record, target_record):
                        # Resolve conflict
                        resolved_record = await self.conflict_resolver.resolve_conflict(
                            record, target_record, self.config.conflict_resolution,
                            self.config.custom_conflict_resolver
                        )
                        
                        await self._update_target_record(record_key, resolved_record)
                        results["updated"] += 1
                        results["conflicts_resolved"] += 1
                else:
                    # Record doesn't exist - insert
                    await self._insert_target_record(record)
                    results["inserted"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process record: {e}")
        
        return results
    
    def _generate_record_key(self, record: Dict[str, Any]) -> str:
        """Generate unique key for record"""
        if self.config.key_fields:
            key_values = [str(record.get(field, '')) for field in self.config.key_fields]
            key_data = '|'.join(key_values)
        else:
            key_data = json.dumps(record, sort_keys=True)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_target_record(self, record_key: str) -> Optional[Dict[str, Any]]:
        """Get record from target system"""
        # This would need connector-specific implementation
        return None
    
    async def _insert_target_record(self, record: Dict[str, Any]):
        """Insert record into target system"""
        # This would need connector-specific implementation
        logger.debug(f"Inserting record: {record}")
    
    async def _update_target_record(self, record_key: str, record: Dict[str, Any]):
        """Update record in target system"""
        # This would need connector-specific implementation
        logger.debug(f"Updating record {record_key}: {record}")
    
    def _records_differ(self, record1: Dict[str, Any], record2: Dict[str, Any]) -> bool:
        """Check if two records differ"""
        return json.dumps(record1, sort_keys=True) != json.dumps(record2, sort_keys=True)
    
    def _update_batch_stats(self, batch_time_ms: float, success: bool):
        """Update batch statistics"""
        self.stats["total_batches_processed"] += 1
        
        if success:
            self.stats["successful_batches"] += 1
        else:
            self.stats["failed_batches"] += 1
        
        # Update average batch time
        total_batches = self.stats["total_batches_processed"]
        current_avg = self.stats["avg_batch_time_ms"]
        self.stats["avg_batch_time_ms"] = \
            (current_avg * (total_batches - 1) + batch_time_ms) / total_batches
    
    def get_status(self) -> Dict[str, Any]:
        """Get batch sync status"""
        return {
            "sync_id": self.config.sync_id,
            "name": self.config.name,
            "status": self.status.value,
            "statistics": self.stats
        }


class DataSyncManager:
    """Main data synchronization manager"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./sync_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Components
        self.sync_configurations: Dict[str, SyncConfiguration] = {}
        self.connectors: Dict[str, DataConnector] = {}
        self.transformation_engine = TransformationEngine()
        self.change_tracker = ChangeTracker()
        
        # Active synchronizations
        self.realtime_syncs: Dict[str, RealtimeSync] = {}
        self.batch_syncs: Dict[str, BatchSync] = {}
        
        # Scheduling
        self.scheduler_enabled = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.manager_stats = {
            "total_sync_configurations": 0,
            "active_realtime_syncs": 0,
            "total_sync_executions": 0,
            "successful_sync_executions": 0,
            "failed_sync_executions": 0
        }
        
        logger.info("Data Sync Manager initialized")
    
    def register_sync_configuration(self, config: SyncConfiguration):
        """Register synchronization configuration"""
        self.sync_configurations[config.sync_id] = config
        self.manager_stats["total_sync_configurations"] += 1
        
        logger.info(f"Registered sync configuration: {config.name}")
    
    def register_connector(self, connector: DataConnector):
        """Register data connector"""
        self.connectors[connector.config.connector_id] = connector
        logger.info(f"Registered connector: {connector.config.name}")
    
    async def start_realtime_sync(self, sync_id: str) -> bool:
        """Start real-time synchronization"""
        if sync_id not in self.sync_configurations:
            logger.error(f"Sync configuration not found: {sync_id}")
            return False
        
        if sync_id in self.realtime_syncs:
            logger.warning(f"Real-time sync already running: {sync_id}")
            return True
        
        config = self.sync_configurations[sync_id]
        
        # Get connectors
        source_connector = self.connectors.get(config.source_connector_id)
        target_connector = self.connectors.get(config.target_connector_id)
        
        if not source_connector or not target_connector:
            logger.error(f"Required connectors not found for sync: {sync_id}")
            return False
        
        # Create and start real-time sync
        realtime_sync = RealtimeSync(
            config, source_connector, target_connector,
            self.change_tracker, self.transformation_engine
        )
        
        await realtime_sync.start()
        self.realtime_syncs[sync_id] = realtime_sync
        self.manager_stats["active_realtime_syncs"] += 1
        
        return True
    
    async def stop_realtime_sync(self, sync_id: str) -> bool:
        """Stop real-time synchronization"""
        if sync_id not in self.realtime_syncs:
            return False
        
        realtime_sync = self.realtime_syncs[sync_id]
        await realtime_sync.stop()
        
        del self.realtime_syncs[sync_id]
        self.manager_stats["active_realtime_syncs"] -= 1
        
        return True
    
    async def execute_batch_sync(self, sync_id: str) -> Optional[SyncExecution]:
        """Execute batch synchronization"""
        if sync_id not in self.sync_configurations:
            logger.error(f"Sync configuration not found: {sync_id}")
            return None
        
        config = self.sync_configurations[sync_id]
        
        # Get connectors
        source_connector = self.connectors.get(config.source_connector_id)
        target_connector = self.connectors.get(config.target_connector_id)
        
        if not source_connector or not target_connector:
            logger.error(f"Required connectors not found for sync: {sync_id}")
            return None
        
        # Create and execute batch sync
        batch_sync = BatchSync(config, source_connector, target_connector, self.transformation_engine)
        self.batch_syncs[sync_id] = batch_sync
        
        try:
            execution = await batch_sync.execute()
            
            self.manager_stats["total_sync_executions"] += 1
            if execution.status == SyncStatus.COMPLETED:
                self.manager_stats["successful_sync_executions"] += 1
            else:
                self.manager_stats["failed_sync_executions"] += 1
            
            return execution
            
        finally:
            # Remove from active batch syncs
            self.batch_syncs.pop(sync_id, None)
    
    async def start_scheduler(self):
        """Start sync scheduler"""
        if self.scheduler_enabled:
            return
        
        self.scheduler_enabled = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Sync scheduler started")
    
    async def stop_scheduler(self):
        """Stop sync scheduler"""
        if not self.scheduler_enabled:
            return
        
        self.scheduler_enabled = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Sync scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_enabled:
            try:
                current_time = datetime.now()
                
                for config in self.sync_configurations.values():
                    if (config.schedule_enabled and 
                        config.schedule_cron and
                        config.sync_id not in self.realtime_syncs and
                        config.sync_id not in self.batch_syncs):
                        
                        # Check if sync should run
                        from croniter import croniter
                        cron = croniter(config.schedule_cron, current_time)
                        next_run = cron.get_next(datetime)
                        
                        # If next run is within the next minute, execute
                        if (next_run - current_time).total_seconds() <= 60:
                            logger.info(f"Scheduled sync triggered: {config.name}")
                            
                            if config.sync_type == SyncType.REAL_TIME:
                                asyncio.create_task(self.start_realtime_sync(config.sync_id))
                            else:
                                asyncio.create_task(self.execute_batch_sync(config.sync_id))
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    def get_sync_status(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific sync"""
        if sync_id in self.realtime_syncs:
            return self.realtime_syncs[sync_id].get_status()
        elif sync_id in self.batch_syncs:
            return self.batch_syncs[sync_id].get_status()
        elif sync_id in self.sync_configurations:
            return {
                "sync_id": sync_id,
                "name": self.sync_configurations[sync_id].name,
                "status": "idle"
            }
        else:
            return None
    
    def list_sync_configurations(self) -> List[Dict[str, Any]]:
        """List all sync configurations"""
        return [
            {
                "sync_id": config.sync_id,
                "name": config.name,
                "description": config.description,
                "sync_type": config.sync_type.value,
                "sync_direction": config.sync_direction,
                "schedule_enabled": config.schedule_enabled,
                "created_at": config.created_at.isoformat(),
                "is_active": (config.sync_id in self.realtime_syncs or 
                            config.sync_id in self.batch_syncs)
            }
            for config in self.sync_configurations.values()
        ]
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            **self.manager_stats,
            "scheduler_enabled": self.scheduler_enabled,
            "total_connectors": len(self.connectors),
            "change_tracker_changes": len(self.change_tracker.change_log)
        }


# Export main classes
__all__ = [
    'SyncType',
    'SyncStatus', 
    'ConflictResolutionStrategy',
    'ChangeType',
    'SyncConfiguration',
    'ChangeRecord',
    'SyncExecution',
    'ChangeTracker',
    'ConflictResolver',
    'RealtimeSync',
    'BatchSync',
    'DataSyncManager'
]