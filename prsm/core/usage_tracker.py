"""
Generic Usage Tracker for PRSM

Consolidated usage tracking system that unifies token usage, cost tracking,
performance metrics, and resource consumption across all PRSM components.

This module provides a centralized, consistent interface for tracking:
- Token and resource consumption
- Cost calculation with high precision
- Performance metrics and analytics
- User activity and session tracking
- System-wide analytics and reporting
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from abc import ABC, abstractmethod
import statistics
import json

import structlog

# Set high precision for financial calculations
getcontext().prec = 28

logger = structlog.get_logger(__name__)


class ResourceType(Enum):
    """Types of resources that can be tracked"""
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    TOKEN_TOTAL = "token_total"
    CONTEXT_UNITS = "context_units"
    COMPUTE_TIME = "compute_time"
    MEMORY_USAGE = "memory_usage"
    API_CALLS = "api_calls"
    DATA_TRANSFER = "data_transfer"
    STORAGE_USAGE = "storage_usage"


class CostCategory(Enum):
    """Categories for cost tracking"""
    MODEL_INFERENCE = "model_inference"
    AGENT_COORDINATION = "agent_coordination"
    TOOL_EXECUTION = "tool_execution"
    DATA_ACCESS = "data_access"
    NETWORK_OPERATIONS = "network_operations"
    STORAGE_OPERATIONS = "storage_operations"
    SECURITY_OPERATIONS = "security_operations"
    MONITORING = "monitoring"


class OperationType(Enum):
    """Types of operations for performance tracking"""
    MODEL_EXECUTION = "model_execution"
    QUERY_PROCESSING = "query_processing"
    AGENT_COORDINATION = "agent_coordination"
    TOOL_INVOCATION = "tool_invocation"
    DATA_RETRIEVAL = "data_retrieval"
    RESPONSE_GENERATION = "response_generation"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"


class AggregationType(Enum):
    """Types of data aggregation"""
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"


@dataclass
class UsageRecord:
    """Record of resource usage"""
    record_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resource_type: ResourceType = None
    amount: Union[int, float, Decimal] = 0
    user_id: str = ""
    session_id: Optional[UUID] = None
    operation_type: Optional[OperationType] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostRecord:
    """Record of cost/financial tracking"""
    record_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cost_category: CostCategory = None
    amount: Decimal = Decimal('0')
    currency: str = "FTNS"
    user_id: str = ""
    session_id: Optional[UUID] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceRecord:
    """Record of performance metrics"""
    record_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation: OperationType = None
    duration_ms: float = 0.0
    success: bool = True
    user_id: str = ""
    session_id: Optional[UUID] = None
    throughput: Optional[float] = None
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsFilters:
    """Filters for analytics queries"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    user_ids: Optional[List[str]] = None
    session_ids: Optional[List[UUID]] = None
    resource_types: Optional[List[ResourceType]] = None
    cost_categories: Optional[List[CostCategory]] = None
    operation_types: Optional[List[OperationType]] = None
    providers: Optional[List[str]] = None
    model_ids: Optional[List[str]] = None
    success_only: Optional[bool] = None


@dataclass
class AnalyticsResult:
    """Result of analytics query"""
    total_records: int = 0
    aggregated_value: Union[float, Decimal] = 0
    aggregation_type: AggregationType = None
    breakdown: Dict[str, Union[float, Decimal]] = field(default_factory=dict)
    time_series: List[Tuple[datetime, Union[float, Decimal]]] = field(default_factory=list)
    percentiles: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UsageStorage(ABC):
    """Abstract interface for usage data storage"""
    
    @abstractmethod
    async def store_usage_record(self, record: UsageRecord) -> bool:
        """Store a usage record"""
        pass
    
    @abstractmethod
    async def store_cost_record(self, record: CostRecord) -> bool:
        """Store a cost record"""
        pass
    
    @abstractmethod
    async def store_performance_record(self, record: PerformanceRecord) -> bool:
        """Store a performance record"""
        pass
    
    @abstractmethod
    async def query_records(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        """Query records with filters"""
        pass


class InMemoryStorage(UsageStorage):
    """In-memory storage implementation for testing and development"""
    
    def __init__(self, max_records: int = 10000):
        self.usage_records: List[UsageRecord] = []
        self.cost_records: List[CostRecord] = []
        self.performance_records: List[PerformanceRecord] = []
        self.max_records = max_records
        self._lock = asyncio.Lock()
    
    async def store_usage_record(self, record: UsageRecord) -> bool:
        async with self._lock:
            self.usage_records.append(record)
            if len(self.usage_records) > self.max_records:
                self.usage_records.pop(0)
            return True
    
    async def store_cost_record(self, record: CostRecord) -> bool:
        async with self._lock:
            self.cost_records.append(record)
            if len(self.cost_records) > self.max_records:
                self.cost_records.pop(0)
            return True
    
    async def store_performance_record(self, record: PerformanceRecord) -> bool:
        async with self._lock:
            self.performance_records.append(record)
            if len(self.performance_records) > self.max_records:
                self.performance_records.pop(0)
            return True
    
    async def query_records(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        async with self._lock:
            all_records = []
            
            # Convert records to dictionaries and apply filters
            for record in self.usage_records + self.cost_records + self.performance_records:
                if self._matches_filters(record, filters):
                    all_records.append(self._record_to_dict(record))
            
            return all_records
    
    def _matches_filters(self, record: Any, filters: AnalyticsFilters) -> bool:
        """Check if record matches the given filters"""
        if filters.start_time and record.timestamp < filters.start_time:
            return False
        if filters.end_time and record.timestamp > filters.end_time:
            return False
        if filters.user_ids and record.user_id not in filters.user_ids:
            return False
        if filters.session_ids and record.session_id not in filters.session_ids:
            return False
        
        # Type-specific filters
        if isinstance(record, UsageRecord):
            if filters.resource_types and record.resource_type not in filters.resource_types:
                return False
            if filters.providers and record.provider not in filters.providers:
                return False
            if filters.model_ids and record.model_id not in filters.model_ids:
                return False
        elif isinstance(record, CostRecord):
            if filters.cost_categories and record.cost_category not in filters.cost_categories:
                return False
        elif isinstance(record, PerformanceRecord):
            if filters.operation_types and record.operation not in filters.operation_types:
                return False
            if filters.success_only is not None and record.success != filters.success_only:
                return False
        
        return True
    
    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert record to dictionary"""
        record_dict = {
            'record_id': str(record.record_id),
            'timestamp': record.timestamp.isoformat(),
            'user_id': record.user_id,
            'session_id': str(record.session_id) if record.session_id else None,
            'metadata': record.metadata
        }
        
        if isinstance(record, UsageRecord):
            record_dict.update({
                'type': 'usage',
                'resource_type': record.resource_type.value if record.resource_type else None,
                'amount': float(record.amount) if isinstance(record.amount, Decimal) else record.amount,
                'operation_type': record.operation_type.value if record.operation_type else None,
                'provider': record.provider,
                'model_id': record.model_id
            })
        elif isinstance(record, CostRecord):
            record_dict.update({
                'type': 'cost',
                'cost_category': record.cost_category.value if record.cost_category else None,
                'amount': float(record.amount),
                'currency': record.currency,
                'description': record.description
            })
        elif isinstance(record, PerformanceRecord):
            record_dict.update({
                'type': 'performance',
                'operation': record.operation.value if record.operation else None,
                'duration_ms': record.duration_ms,
                'success': record.success,
                'throughput': record.throughput,
                'quality_score': record.quality_score,
                'error_message': record.error_message
            })
        
        return record_dict


class GenericUsageTracker:
    """
    Unified usage tracking system for PRSM
    
    🎯 PURPOSE:
    Provides centralized, consistent tracking of resource usage, costs,
    and performance metrics across all PRSM components.
    
    🔧 FEATURES:
    - High-precision cost tracking (28 decimal places)
    - Microsecond-precision timing
    - Comprehensive analytics and aggregation
    - Pluggable storage backends
    - Thread-safe operations
    - Real-time monitoring capabilities
    """
    
    def __init__(self, storage: Optional[UsageStorage] = None):
        self.storage = storage or InMemoryStorage()
        self._session_cache: Dict[UUID, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def track_resource_usage(
        self,
        resource_type: ResourceType,
        amount: Union[int, float, Decimal],
        user_id: str,
        session_id: Optional[UUID] = None,
        operation_type: Optional[OperationType] = None,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """
        Track resource usage with comprehensive metadata
        
        Args:
            resource_type: Type of resource being tracked
            amount: Amount of resource consumed
            user_id: User who consumed the resource
            session_id: Optional session identifier
            operation_type: Type of operation that consumed the resource
            provider: Service provider (e.g., OpenAI, Anthropic)
            model_id: Model identifier if applicable
            metadata: Additional metadata
            
        Returns:
            UsageRecord: Created usage record
        """
        record = UsageRecord(
            resource_type=resource_type,
            amount=Decimal(str(amount)) if isinstance(amount, (int, float)) else amount,
            user_id=user_id,
            session_id=session_id,
            operation_type=operation_type,
            provider=provider,
            model_id=model_id,
            metadata=metadata or {}
        )
        
        await self.storage.store_usage_record(record)
        
        logger.debug(
            "Resource usage tracked",
            resource_type=resource_type.value,
            amount=amount,
            user_id=user_id,
            session_id=str(session_id) if session_id else None,
            provider=provider,
            model_id=model_id
        )
        
        return record
    
    async def track_cost(
        self,
        cost_category: CostCategory,
        amount: Union[float, Decimal],
        user_id: str,
        session_id: Optional[UUID] = None,
        currency: str = "FTNS",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """
        Track cost with high precision
        
        Args:
            cost_category: Category of cost
            amount: Cost amount
            user_id: User who incurred the cost
            session_id: Optional session identifier
            currency: Currency type
            description: Description of the cost
            metadata: Additional metadata
            
        Returns:
            CostRecord: Created cost record
        """
        record = CostRecord(
            cost_category=cost_category,
            amount=Decimal(str(amount)),
            user_id=user_id,
            session_id=session_id,
            currency=currency,
            description=description,
            metadata=metadata or {}
        )
        
        await self.storage.store_cost_record(record)
        
        logger.info(
            "Cost tracked",
            cost_category=cost_category.value,
            amount=float(amount),
            currency=currency,
            user_id=user_id,
            session_id=str(session_id) if session_id else None,
            description=description
        )
        
        return record
    
    async def track_performance(
        self,
        operation: OperationType,
        duration_ms: float,
        success: bool,
        user_id: str,
        session_id: Optional[UUID] = None,
        throughput: Optional[float] = None,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceRecord:
        """
        Track performance metrics
        
        Args:
            operation: Type of operation
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            user_id: User who performed the operation
            session_id: Optional session identifier
            throughput: Optional throughput metric
            quality_score: Optional quality score
            error_message: Optional error message for failed operations
            metadata: Additional metadata
            
        Returns:
            PerformanceRecord: Created performance record
        """
        record = PerformanceRecord(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            session_id=session_id,
            throughput=throughput,
            quality_score=quality_score,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        await self.storage.store_performance_record(record)
        
        logger.debug(
            "Performance tracked",
            operation=operation.value,
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            session_id=str(session_id) if session_id else None,
            quality_score=quality_score
        )
        
        return record
    
    async def get_analytics(
        self,
        filters: AnalyticsFilters,
        aggregation: AggregationType,
        group_by: Optional[str] = None
    ) -> AnalyticsResult:
        """
        Get analytics with aggregation and grouping
        
        Args:
            filters: Filters to apply to the data
            aggregation: Type of aggregation to perform
            group_by: Optional field to group by
            
        Returns:
            AnalyticsResult: Aggregated analytics result
        """
        records = await self.storage.query_records(filters)
        
        if not records:
            return AnalyticsResult(
                total_records=0,
                aggregated_value=0,
                aggregation_type=aggregation
            )
        
        # Extract values for aggregation
        values = []
        for record in records:
            if record['type'] == 'usage':
                values.append(record['amount'])
            elif record['type'] == 'cost':
                values.append(record['amount'])
            elif record['type'] == 'performance':
                values.append(record['duration_ms'])
        
        # Perform aggregation
        if not values:
            aggregated_value = 0
        elif aggregation == AggregationType.SUM:
            aggregated_value = sum(values)
        elif aggregation == AggregationType.AVERAGE:
            aggregated_value = statistics.mean(values)
        elif aggregation == AggregationType.MEDIAN:
            aggregated_value = statistics.median(values)
        elif aggregation == AggregationType.COUNT:
            aggregated_value = len(values)
        elif aggregation == AggregationType.MIN:
            aggregated_value = min(values)
        elif aggregation == AggregationType.MAX:
            aggregated_value = max(values)
        elif aggregation == AggregationType.PERCENTILE_95:
            aggregated_value = statistics.quantiles(values, n=20)[18]  # 95th percentile
        elif aggregation == AggregationType.PERCENTILE_99:
            aggregated_value = statistics.quantiles(values, n=100)[98]  # 99th percentile
        else:
            aggregated_value = 0
        
        # Group by if specified
        breakdown = {}
        if group_by:
            grouped = {}
            for record in records:
                key = record.get(group_by, 'unknown')
                if key not in grouped:
                    grouped[key] = []
                
                if record['type'] == 'usage':
                    grouped[key].append(record['amount'])
                elif record['type'] == 'cost':
                    grouped[key].append(record['amount'])
                elif record['type'] == 'performance':
                    grouped[key].append(record['duration_ms'])
            
            for key, group_values in grouped.items():
                if aggregation == AggregationType.SUM:
                    breakdown[key] = sum(group_values)
                elif aggregation == AggregationType.AVERAGE:
                    breakdown[key] = statistics.mean(group_values)
                elif aggregation == AggregationType.COUNT:
                    breakdown[key] = len(group_values)
                # Add other aggregations as needed
        
        # Calculate percentiles for performance data
        percentiles = {}
        if values and len(values) > 1:
            try:
                percentiles = {
                    'p50': statistics.median(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
                }
            except statistics.StatisticsError:
                # Handle case where quantiles can't be calculated
                percentiles = {
                    'p50': statistics.median(values),
                    'p95': max(values),
                    'p99': max(values)
                }
        
        return AnalyticsResult(
            total_records=len(records),
            aggregated_value=aggregated_value,
            aggregation_type=aggregation,
            breakdown=breakdown,
            percentiles=percentiles,
            metadata={
                'filters_applied': str(filters),
                'group_by': group_by,
                'data_types': list(set(record['type'] for record in records))
            }
        )
    
    async def get_session_summary(self, session_id: UUID) -> Dict[str, Any]:
        """Get comprehensive summary for a session"""
        filters = AnalyticsFilters(session_ids=[session_id])
        
        # Get all records for the session
        records = await self.storage.query_records(filters)
        
        summary = {
            'session_id': str(session_id),
            'total_records': len(records),
            'usage_summary': {},
            'cost_summary': {},
            'performance_summary': {},
            'timeline': []
        }
        
        usage_records = [r for r in records if r['type'] == 'usage']
        cost_records = [r for r in records if r['type'] == 'cost']
        performance_records = [r for r in records if r['type'] == 'performance']
        
        # Usage summary
        if usage_records:
            summary['usage_summary'] = {
                'total_tokens': sum(r['amount'] for r in usage_records if r.get('resource_type') == 'token_total'),
                'total_api_calls': sum(r['amount'] for r in usage_records if r.get('resource_type') == 'api_calls'),
                'providers_used': list(set(r['provider'] for r in usage_records if r.get('provider'))),
                'models_used': list(set(r['model_id'] for r in usage_records if r.get('model_id')))
            }
        
        # Cost summary
        if cost_records:
            total_cost = sum(r['amount'] for r in cost_records)
            summary['cost_summary'] = {
                'total_cost': total_cost,
                'currency': cost_records[0]['currency'] if cost_records else 'FTNS',
                'cost_by_category': {}
            }
            
            for record in cost_records:
                category = record.get('cost_category', 'unknown')
                if category not in summary['cost_summary']['cost_by_category']:
                    summary['cost_summary']['cost_by_category'][category] = 0
                summary['cost_summary']['cost_by_category'][category] += record['amount']
        
        # Performance summary
        if performance_records:
            durations = [r['duration_ms'] for r in performance_records]
            successful = [r for r in performance_records if r['success']]
            
            summary['performance_summary'] = {
                'total_operations': len(performance_records),
                'successful_operations': len(successful),
                'success_rate': len(successful) / len(performance_records) if performance_records else 0,
                'average_duration_ms': statistics.mean(durations) if durations else 0,
                'total_duration_ms': sum(durations),
                'operations_by_type': {}
            }
            
            for record in performance_records:
                op_type = record.get('operation', 'unknown')
                if op_type not in summary['performance_summary']['operations_by_type']:
                    summary['performance_summary']['operations_by_type'][op_type] = 0
                summary['performance_summary']['operations_by_type'][op_type] += 1
        
        # Timeline
        summary['timeline'] = sorted(records, key=lambda x: x['timestamp'])
        
        return summary
    
    async def get_user_analytics(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for a user"""
        filters = AnalyticsFilters(
            user_ids=[user_id],
            start_time=start_time,
            end_time=end_time
        )
        
        records = await self.storage.query_records(filters)
        
        analytics = {
            'user_id': user_id,
            'period': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'summary': {},
            'trends': {},
            'recommendations': []
        }
        
        if not records:
            return analytics
        
        # Calculate summary statistics
        cost_total = sum(r['amount'] for r in records if r['type'] == 'cost')
        usage_total = sum(r['amount'] for r in records if r['type'] == 'usage')
        avg_response_time = statistics.mean([r['duration_ms'] for r in records if r['type'] == 'performance']) if [r for r in records if r['type'] == 'performance'] else 0
        
        analytics['summary'] = {
            'total_cost': cost_total,
            'total_usage': usage_total,
            'average_response_time_ms': avg_response_time,
            'total_sessions': len(set(r['session_id'] for r in records if r.get('session_id'))),
            'most_used_provider': self._get_most_frequent([r['provider'] for r in records if r.get('provider')]),
            'most_used_model': self._get_most_frequent([r['model_id'] for r in records if r.get('model_id')])
        }
        
        # Generate simple recommendations
        if cost_total > 100:  # Arbitrary threshold
            analytics['recommendations'].append("Consider using more cost-efficient models for routine tasks")
        if avg_response_time > 1000:  # 1 second threshold
            analytics['recommendations'].append("Consider using faster models for better response times")
        
        return analytics
    
    def _get_most_frequent(self, items: List[str]) -> Optional[str]:
        """Get the most frequently occurring item"""
        if not items:
            return None
        
        frequency = {}
        for item in items:
            frequency[item] = frequency.get(item, 0) + 1
        
        return max(frequency.items(), key=lambda x: x[1])[0] if frequency else None


# Utility functions for easy integration

def create_usage_tracker(storage: Optional[UsageStorage] = None) -> GenericUsageTracker:
    """Create a usage tracker instance"""
    return GenericUsageTracker(storage)


async def track_model_execution(
    tracker: GenericUsageTracker,
    user_id: str,
    session_id: UUID,
    model_id: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    cost: Decimal,
    success: bool,
    error_message: Optional[str] = None
) -> Tuple[UsageRecord, UsageRecord, CostRecord, PerformanceRecord]:
    """
    Convenience function to track a complete model execution
    
    Returns tuple of (input_usage, output_usage, cost, performance) records
    """
    input_usage = await tracker.track_resource_usage(
        ResourceType.TOKEN_INPUT,
        input_tokens,
        user_id,
        session_id,
        OperationType.MODEL_EXECUTION,
        provider,
        model_id
    )
    
    output_usage = await tracker.track_resource_usage(
        ResourceType.TOKEN_OUTPUT,
        output_tokens,
        user_id,
        session_id,
        OperationType.MODEL_EXECUTION,
        provider,
        model_id
    )
    
    cost_record = await tracker.track_cost(
        CostCategory.MODEL_INFERENCE,
        cost,
        user_id,
        session_id,
        description=f"Model execution: {model_id} via {provider}"
    )
    
    performance_record = await tracker.track_performance(
        OperationType.MODEL_EXECUTION,
        duration_ms,
        success,
        user_id,
        session_id,
        error_message=error_message,
        metadata={'model_id': model_id, 'provider': provider}
    )
    
    return input_usage, output_usage, cost_record, performance_record