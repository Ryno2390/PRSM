#!/usr/bin/env python3
"""
Business Intelligence Query Engine
==================================

Advanced query engine for business intelligence analytics,
supporting complex aggregations, filtering, and data analysis.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import re
from pathlib import Path

from prsm.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of business intelligence queries"""
    AGGREGATION = "aggregation"
    FILTER = "filter"
    JOIN = "join"
    GROUPBY = "groupby"
    PIVOT = "pivot"
    TIMESERIES = "timeseries"
    STATISTICAL = "statistical"
    PREDICTIVE = "predictive"
    CUSTOM = "custom"


class AggregationFunction(Enum):
    """Aggregation functions for BI queries"""
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MODE = "mode"
    STDDEV = "stddev"
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    DISTINCT_COUNT = "distinct_count"
    FIRST = "first"
    LAST = "last"


class FilterOperator(Enum):
    """Filter operators for conditions"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"


@dataclass
class QueryFilter:
    """Filter condition for BI queries"""
    field: str
    operator: FilterOperator
    value: Any
    logical_operator: str = "AND"  # AND, OR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "logical_operator": self.logical_operator
        }


@dataclass
class QueryAggregation:
    """Aggregation specification for BI queries"""
    field: str
    function: AggregationFunction
    alias: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "function": self.function.value,
            "alias": self.alias or f"{self.function.value}_{self.field}",
            "parameters": self.parameters
        }


@dataclass
class QueryGroupBy:
    """Group by specification for BI queries"""
    fields: List[str]
    time_bucket: Optional[str] = None  # hour, day, week, month, quarter, year
    time_field: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": self.fields,
            "time_bucket": self.time_bucket,
            "time_field": self.time_field
        }


@dataclass
class QuerySort:
    """Sort specification for BI queries"""
    field: str
    direction: str = "ASC"  # ASC, DESC
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "direction": self.direction
        }


@dataclass
class BIQuery:
    """Business Intelligence Query specification"""
    query_id: str
    query_type: QueryType
    data_source: str
    
    # Core query components
    select_fields: List[str] = field(default_factory=list)
    filters: List[QueryFilter] = field(default_factory=list)
    aggregations: List[QueryAggregation] = field(default_factory=list)
    group_by: Optional[QueryGroupBy] = None
    sort_by: List[QuerySort] = field(default_factory=list)
    
    # Query constraints
    limit: Optional[int] = None
    offset: int = 0
    
    # Time range
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    
    # Advanced options
    distinct: bool = False
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary"""
        return {
            "query_id": self.query_id,
            "query_type": self.query_type.value,
            "data_source": self.data_source,
            "select_fields": self.select_fields,
            "filters": [f.to_dict() for f in self.filters],
            "aggregations": [a.to_dict() for a in self.aggregations],
            "group_by": self.group_by.to_dict() if self.group_by else None,
            "sort_by": [s.to_dict() for s in self.sort_by],
            "limit": self.limit,
            "offset": self.offset,
            "time_range_start": self.time_range_start.isoformat() if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
            "distinct": self.distinct,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags
        }


@dataclass
class QueryResult:
    """Result of a BI query execution"""
    query_id: str
    data: List[Dict[str, Any]]
    total_rows: int
    execution_time_ms: float
    cached: bool = False
    
    # Metadata
    columns: List[str] = field(default_factory=list)
    data_types: Dict[str, str] = field(default_factory=dict)
    aggregation_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Query info
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_plan: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "query_id": self.query_id,
            "data": self.data,
            "total_rows": self.total_rows,
            "execution_time_ms": self.execution_time_ms,
            "cached": self.cached,
            "columns": self.columns,
            "data_types": self.data_types,
            "aggregation_summary": self.aggregation_summary,
            "executed_at": self.executed_at.isoformat(),
            "query_plan": self.query_plan
        }


class QueryBuilder:
    """Builder for constructing BI queries programmatically"""
    
    def __init__(self, data_source: str):
        self.query = BIQuery(
            query_id=self._generate_query_id(),
            query_type=QueryType.CUSTOM,
            data_source=data_source
        )
    
    def _generate_query_id(self) -> str:
        """Generate unique query ID"""
        import uuid
        return f"query_{uuid.uuid4().hex[:8]}"
    
    def select(self, *fields: str) -> 'QueryBuilder':
        """Add fields to select"""
        self.query.select_fields.extend(fields)
        return self
    
    def where(self, field: str, operator: FilterOperator, value: Any, 
              logical_op: str = "AND") -> 'QueryBuilder':
        """Add filter condition"""
        filter_condition = QueryFilter(
            field=field,
            operator=operator,
            value=value,
            logical_operator=logical_op
        )
        self.query.filters.append(filter_condition)
        return self
    
    def aggregate(self, field: str, function: AggregationFunction, 
                  alias: Optional[str] = None, **parameters) -> 'QueryBuilder':
        """Add aggregation"""
        aggregation = QueryAggregation(
            field=field,
            function=function,
            alias=alias,
            parameters=parameters
        )
        self.query.aggregations.append(aggregation)
        self.query.query_type = QueryType.AGGREGATION
        return self
    
    def group_by(self, *fields: str, time_bucket: Optional[str] = None,
                 time_field: Optional[str] = None) -> 'QueryBuilder':
        """Add group by clause"""
        self.query.group_by = QueryGroupBy(
            fields=list(fields),
            time_bucket=time_bucket,
            time_field=time_field
        )
        if self.query.query_type == QueryType.CUSTOM:
            self.query.query_type = QueryType.GROUPBY
        return self
    
    def order_by(self, field: str, direction: str = "ASC") -> 'QueryBuilder':
        """Add sort clause"""
        self.query.sort_by.append(QuerySort(field=field, direction=direction))
        return self
    
    def limit(self, count: int, offset: int = 0) -> 'QueryBuilder':
        """Set result limit and offset"""
        self.query.limit = count
        self.query.offset = offset
        return self
    
    def time_range(self, start: datetime, end: datetime) -> 'QueryBuilder':
        """Set time range filter"""
        self.query.time_range_start = start
        self.query.time_range_end = end
        if self.query.query_type == QueryType.CUSTOM:
            self.query.query_type = QueryType.TIMESERIES
        return self
    
    def distinct(self, enable: bool = True) -> 'QueryBuilder':
        """Enable distinct results"""
        self.query.distinct = enable
        return self
    
    def cache(self, enabled: bool = True, ttl: int = 300) -> 'QueryBuilder':
        """Configure caching"""
        self.query.cache_enabled = enabled
        self.query.cache_ttl = ttl
        return self
    
    def metadata(self, description: str = "", tags: List[str] = None,
                 created_by: str = None) -> 'QueryBuilder':
        """Set query metadata"""
        self.query.description = description
        self.query.tags = tags or []
        self.query.created_by = created_by
        return self
    
    def build(self) -> BIQuery:
        """Build the final query"""
        return self.query


class DataProcessor:
    """Process and transform data for BI queries"""
    
    def __init__(self):
        self.pandas = require_optional("pandas")
        self.numpy = require_optional("numpy")
        
    def apply_filters(self, data: List[Dict[str, Any]], 
                     filters: List[QueryFilter]) -> List[Dict[str, Any]]:
        """Apply filter conditions to data"""
        if not filters:
            return data
        
        filtered_data = []
        
        for record in data:
            should_include = True
            current_logical_op = "AND"
            condition_result = True
            
            for filter_condition in filters:
                field_value = record.get(filter_condition.field)
                filter_result = self._evaluate_filter(field_value, filter_condition)
                
                # Apply logical operator
                if current_logical_op == "AND":
                    condition_result = condition_result and filter_result
                elif current_logical_op == "OR":
                    condition_result = condition_result or filter_result
                
                current_logical_op = filter_condition.logical_operator
            
            if condition_result:
                filtered_data.append(record)
        
        return filtered_data
    
    def _evaluate_filter(self, field_value: Any, filter_condition: QueryFilter) -> bool:
        """Evaluate a single filter condition"""
        operator = filter_condition.operator
        filter_value = filter_condition.value
        
        if operator == FilterOperator.EQUALS:
            return field_value == filter_value
        elif operator == FilterOperator.NOT_EQUALS:
            return field_value != filter_value
        elif operator == FilterOperator.GREATER_THAN:
            return field_value > filter_value
        elif operator == FilterOperator.GREATER_EQUAL:
            return field_value >= filter_value
        elif operator == FilterOperator.LESS_THAN:
            return field_value < filter_value
        elif operator == FilterOperator.LESS_EQUAL:
            return field_value <= filter_value
        elif operator == FilterOperator.IN:
            return field_value in filter_value
        elif operator == FilterOperator.NOT_IN:
            return field_value not in filter_value
        elif operator == FilterOperator.CONTAINS:
            return str(filter_value) in str(field_value)
        elif operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(filter_value))
        elif operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(filter_value))
        elif operator == FilterOperator.REGEX:
            return bool(re.search(str(filter_value), str(field_value)))
        elif operator == FilterOperator.IS_NULL:
            return field_value is None
        elif operator == FilterOperator.IS_NOT_NULL:
            return field_value is not None
        elif operator == FilterOperator.BETWEEN:
            return filter_value[0] <= field_value <= filter_value[1]
        
        return False
    
    def apply_aggregations(self, data: List[Dict[str, Any]],
                          aggregations: List[QueryAggregation],
                          group_by: Optional[QueryGroupBy] = None) -> List[Dict[str, Any]]:
        """Apply aggregation functions to data"""
        if not aggregations:
            return data
        
        if group_by:
            return self._apply_grouped_aggregations(data, aggregations, group_by)
        else:
            return self._apply_simple_aggregations(data, aggregations)
    
    def _apply_simple_aggregations(self, data: List[Dict[str, Any]],
                                  aggregations: List[QueryAggregation]) -> List[Dict[str, Any]]:
        """Apply aggregations without grouping"""
        if not data:
            return []
        
        result = {}
        
        for aggregation in aggregations:
            field_name = aggregation.field
            function = aggregation.function
            alias = aggregation.alias or f"{function.value}_{field_name}"
            
            # Extract field values
            field_values = [record.get(field_name) for record in data if record.get(field_name) is not None]
            
            if not field_values:
                result[alias] = None
                continue
            
            # Apply aggregation function
            if function == AggregationFunction.SUM:
                result[alias] = sum(field_values)
            elif function == AggregationFunction.COUNT:
                result[alias] = len(field_values)
            elif function == AggregationFunction.AVERAGE:
                result[alias] = sum(field_values) / len(field_values)
            elif function == AggregationFunction.MIN:
                result[alias] = min(field_values)
            elif function == AggregationFunction.MAX:
                result[alias] = max(field_values)
            elif function == AggregationFunction.MEDIAN:
                sorted_values = sorted(field_values)
                n = len(sorted_values)
                result[alias] = sorted_values[n // 2] if n % 2 == 1 else \
                    (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            elif function == AggregationFunction.DISTINCT_COUNT:
                result[alias] = len(set(field_values))
            elif function == AggregationFunction.FIRST:
                result[alias] = field_values[0]
            elif function == AggregationFunction.LAST:
                result[alias] = field_values[-1]
            elif function == AggregationFunction.STDDEV and self.numpy:
                result[alias] = float(self.numpy.std(field_values))
            elif function == AggregationFunction.VARIANCE and self.numpy:
                result[alias] = float(self.numpy.var(field_values))
        
        return [result]
    
    def _apply_grouped_aggregations(self, data: List[Dict[str, Any]],
                                   aggregations: List[QueryAggregation],
                                   group_by: QueryGroupBy) -> List[Dict[str, Any]]:
        """Apply aggregations with grouping"""
        if not self.pandas:
            # Fallback to manual grouping
            return self._manual_grouped_aggregations(data, aggregations, group_by)
        
        try:
            # Use pandas for efficient grouping
            df = self.pandas.DataFrame(data)
            
            # Handle time bucketing
            if group_by.time_bucket and group_by.time_field:
                df[group_by.time_field] = self.pandas.to_datetime(df[group_by.time_field])
                if group_by.time_bucket == "hour":
                    df['time_bucket'] = df[group_by.time_field].dt.floor('H')
                elif group_by.time_bucket == "day":
                    df['time_bucket'] = df[group_by.time_field].dt.floor('D')
                elif group_by.time_bucket == "week":
                    df['time_bucket'] = df[group_by.time_field].dt.to_period('W').astype(str)
                elif group_by.time_bucket == "month":
                    df['time_bucket'] = df[group_by.time_field].dt.to_period('M').astype(str)
                elif group_by.time_bucket == "quarter":
                    df['time_bucket'] = df[group_by.time_field].dt.to_period('Q').astype(str)
                elif group_by.time_bucket == "year":
                    df['time_bucket'] = df[group_by.time_field].dt.to_period('Y').astype(str)
                
                group_fields = group_by.fields + ['time_bucket']
            else:
                group_fields = group_by.fields
            
            # Group by specified fields
            grouped = df.groupby(group_fields)
            
            # Apply aggregations
            agg_dict = {}
            for aggregation in aggregations:
                field_name = aggregation.field
                function = aggregation.function
                alias = aggregation.alias or f"{function.value}_{field_name}"
                
                if function == AggregationFunction.SUM:
                    agg_dict[alias] = (field_name, 'sum')
                elif function == AggregationFunction.COUNT:
                    agg_dict[alias] = (field_name, 'count')
                elif function == AggregationFunction.AVERAGE:
                    agg_dict[alias] = (field_name, 'mean')
                elif function == AggregationFunction.MIN:
                    agg_dict[alias] = (field_name, 'min')
                elif function == AggregationFunction.MAX:
                    agg_dict[alias] = (field_name, 'max')
                elif function == AggregationFunction.MEDIAN:
                    agg_dict[alias] = (field_name, 'median')
                elif function == AggregationFunction.STDDEV:
                    agg_dict[alias] = (field_name, 'std')
                elif function == AggregationFunction.VARIANCE:
                    agg_dict[alias] = (field_name, 'var')
                elif function == AggregationFunction.DISTINCT_COUNT:
                    agg_dict[alias] = (field_name, 'nunique')
                elif function == AggregationFunction.FIRST:
                    agg_dict[alias] = (field_name, 'first')
                elif function == AggregationFunction.LAST:
                    agg_dict[alias] = (field_name, 'last')
            
            # Perform aggregation
            result_df = grouped.agg(**agg_dict).reset_index()
            
            # Convert back to list of dictionaries
            return result_df.to_dict('records')
            
        except Exception as e:
            logger.warning(f"Pandas aggregation failed, falling back to manual: {e}")
            return self._manual_grouped_aggregations(data, aggregations, group_by)
    
    def _manual_grouped_aggregations(self, data: List[Dict[str, Any]],
                                    aggregations: List[QueryAggregation],
                                    group_by: QueryGroupBy) -> List[Dict[str, Any]]:
        """Manual implementation of grouped aggregations"""
        from collections import defaultdict
        
        # Group data manually
        groups = defaultdict(list)
        
        for record in data:
            # Create group key
            group_key = tuple(record.get(field, '') for field in group_by.fields)
            groups[group_key].append(record)
        
        # Apply aggregations to each group
        results = []
        for group_key, group_data in groups.items():
            result = {}
            
            # Add group by fields
            for i, field in enumerate(group_by.fields):
                result[field] = group_key[i]
            
            # Apply aggregations
            aggregated = self._apply_simple_aggregations(group_data, aggregations)
            if aggregated:
                result.update(aggregated[0])
            
            results.append(result)
        
        return results
    
    def apply_sorting(self, data: List[Dict[str, Any]], 
                     sort_specs: List[QuerySort]) -> List[Dict[str, Any]]:
        """Apply sorting to data"""
        if not sort_specs:
            return data
        
        def sort_key(record):
            key_values = []
            for sort_spec in sort_specs:
                value = record.get(sort_spec.field, 0)
                # Handle None values
                if value is None:
                    value = '' if sort_spec.direction == "ASC" else 'zzz'
                key_values.append(value)
            return key_values
        
        reverse = any(spec.direction == "DESC" for spec in sort_specs)
        return sorted(data, key=sort_key, reverse=reverse)
    
    def apply_limit(self, data: List[Dict[str, Any]], 
                   limit: Optional[int], offset: int = 0) -> List[Dict[str, Any]]:
        """Apply limit and offset to data"""
        if offset > 0:
            data = data[offset:]
        
        if limit is not None:
            data = data[:limit]
        
        return data


class BusinessIntelligenceEngine:
    """Main BI query engine for processing complex analytics queries"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./bi_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.data_processor = DataProcessor()
        self.data_sources: Dict[str, Any] = {}
        self.query_cache: Dict[str, QueryResult] = {}
        self.query_history: List[BIQuery] = []
        
        # Performance metrics
        self.execution_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time": 0.0,
            "failed_queries": 0
        }
        
        # Initialize data sources
        self._initialize_data_sources()
        
        logger.info("Business Intelligence Engine initialized")
    
    def _initialize_data_sources(self):
        """Initialize available data sources"""
        # Register built-in data sources
        self.register_data_source("metrics", self._get_metrics_data)
        self.register_data_source("system", self._get_system_data)
        self.register_data_source("reasoning", self._get_reasoning_data)
    
    def register_data_source(self, name: str, data_provider: Callable):
        """Register a data source"""
        self.data_sources[name] = data_provider
        logger.info(f"Registered data source: {name}")
    
    def _get_metrics_data(self) -> List[Dict[str, Any]]:
        """Get metrics data (mock implementation)"""
        # In real implementation, this would fetch from MetricsCollector
        return [
            {"timestamp": datetime.now(), "metric": "cpu_usage", "value": 45.2, "host": "server1"},
            {"timestamp": datetime.now(), "metric": "memory_usage", "value": 78.5, "host": "server1"},
            {"timestamp": datetime.now(), "metric": "cpu_usage", "value": 52.1, "host": "server2"},
        ]
    
    def _get_system_data(self) -> List[Dict[str, Any]]:
        """Get system data (mock implementation)"""
        return [
            {"timestamp": datetime.now(), "component": "reasoning_engine", "status": "healthy", "uptime": 99.9},
            {"timestamp": datetime.now(), "component": "api_server", "status": "healthy", "uptime": 99.8},
        ]
    
    def _get_reasoning_data(self) -> List[Dict[str, Any]]:
        """Get reasoning engine data (mock implementation)"""
        return [
            {"timestamp": datetime.now(), "engine": "deductive", "executions": 150, "avg_time": 2.3},
            {"timestamp": datetime.now(), "engine": "inductive", "executions": 89, "avg_time": 3.1},
        ]
    
    async def execute_query(self, query: BIQuery) -> QueryResult:
        """Execute a BI query and return results"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if query.cache_enabled:
                cached_result = self._get_cached_result(query)
                if cached_result:
                    self.execution_stats["cache_hits"] += 1
                    return cached_result
            
            self.execution_stats["cache_misses"] += 1
            
            # Get data from source
            if query.data_source not in self.data_sources:
                raise ValueError(f"Unknown data source: {query.data_source}")
            
            raw_data = self.data_sources[query.data_source]()
            
            # Process query
            processed_data = await self._process_query(query, raw_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = QueryResult(
                query_id=query.query_id,
                data=processed_data,
                total_rows=len(processed_data),
                execution_time_ms=execution_time,
                cached=False,
                columns=list(processed_data[0].keys()) if processed_data else [],
                executed_at=datetime.now(timezone.utc)
            )
            
            # Cache result
            if query.cache_enabled:
                self._cache_result(query, result)
            
            # Update statistics
            self._update_execution_stats(execution_time)
            
            # Store query in history
            self.query_history.append(query)
            
            return result
            
        except Exception as e:
            self.execution_stats["failed_queries"] += 1
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def _process_query(self, query: BIQuery, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process query against raw data"""
        data = raw_data.copy()
        
        # Apply time range filter
        if query.time_range_start or query.time_range_end:
            data = self._apply_time_range_filter(data, query.time_range_start, query.time_range_end)
        
        # Apply filters
        if query.filters:
            data = self.data_processor.apply_filters(data, query.filters)
        
        # Apply aggregations
        if query.aggregations:
            data = self.data_processor.apply_aggregations(data, query.aggregations, query.group_by)
        
        # Apply sorting
        if query.sort_by:
            data = self.data_processor.apply_sorting(data, query.sort_by)
        
        # Apply distinct
        if query.distinct:
            seen = set()
            unique_data = []
            for record in data:
                key = tuple(sorted(record.items()))
                if key not in seen:
                    seen.add(key)
                    unique_data.append(record)
            data = unique_data
        
        # Apply select fields
        if query.select_fields:
            data = [{field: record.get(field) for field in query.select_fields} for record in data]
        
        # Apply limit and offset
        if query.limit is not None or query.offset > 0:
            data = self.data_processor.apply_limit(data, query.limit, query.offset)
        
        return data
    
    def _apply_time_range_filter(self, data: List[Dict[str, Any]], 
                                start: Optional[datetime], 
                                end: Optional[datetime]) -> List[Dict[str, Any]]:
        """Apply time range filter to data"""
        if not (start or end):
            return data
        
        filtered_data = []
        for record in data:
            # Find timestamp field (common names)
            timestamp_value = None
            for field in ['timestamp', 'created_at', 'time', 'date']:
                if field in record:
                    timestamp_value = record[field]
                    break
            
            if timestamp_value is None:
                continue
            
            # Convert to datetime if needed
            if isinstance(timestamp_value, str):
                try:
                    timestamp_value = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                except:
                    continue
            
            # Apply range filter
            if start and timestamp_value < start:
                continue
            if end and timestamp_value > end:
                continue
            
            filtered_data.append(record)
        
        return filtered_data
    
    def _get_cached_result(self, query: BIQuery) -> Optional[QueryResult]:
        """Get cached query result if available and valid"""
        query_hash = self._generate_query_hash(query)
        
        if query_hash not in self.query_cache:
            return None
        
        cached_result = self.query_cache[query_hash]
        
        # Check if cache is still valid
        cache_age = (datetime.now(timezone.utc) - cached_result.executed_at).total_seconds()
        if cache_age > query.cache_ttl:
            del self.query_cache[query_hash]
            return None
        
        # Mark as cached
        cached_result.cached = True
        return cached_result
    
    def _cache_result(self, query: BIQuery, result: QueryResult):
        """Cache query result"""
        query_hash = self._generate_query_hash(query)
        self.query_cache[query_hash] = result
        
        # Limit cache size (keep last 100 results)
        if len(self.query_cache) > 100:
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k].executed_at)
            del self.query_cache[oldest_key]
    
    def _generate_query_hash(self, query: BIQuery) -> str:
        """Generate hash for query caching"""
        import hashlib
        
        # Create deterministic string representation
        query_str = json.dumps(query.to_dict(), sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _update_execution_stats(self, execution_time_ms: float):
        """Update execution statistics"""
        self.execution_stats["total_queries"] += 1
        
        # Update average execution time
        total_queries = self.execution_stats["total_queries"]
        current_avg = self.execution_stats["avg_execution_time"]
        self.execution_stats["avg_execution_time"] = \
            (current_avg * (total_queries - 1) + execution_time_ms) / total_queries
    
    def create_query_builder(self, data_source: str) -> QueryBuilder:
        """Create a new query builder"""
        return QueryBuilder(data_source)
    
    def get_data_sources(self) -> List[str]:
        """Get list of available data sources"""
        return list(self.data_sources.keys())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get query execution statistics"""
        cache_hit_rate = 0.0
        if self.execution_stats["total_queries"] > 0:
            cache_hit_rate = self.execution_stats["cache_hits"] / self.execution_stats["total_queries"]
        
        return {
            **self.execution_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache),
            "query_history_size": len(self.query_history),
            "data_sources": len(self.data_sources)
        }
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query history"""
        recent_queries = self.query_history[-limit:] if limit else self.query_history
        return [query.to_dict() for query in recent_queries]
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")


# Export main classes
__all__ = [
    'QueryType',
    'AggregationFunction',
    'FilterOperator',
    'QueryFilter',
    'QueryAggregation', 
    'QueryGroupBy',
    'QuerySort',
    'BIQuery',
    'QueryResult',
    'QueryBuilder',
    'DataProcessor',
    'BusinessIntelligenceEngine'
]