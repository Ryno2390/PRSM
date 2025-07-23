"""
PRSM Database Indexing Strategies
Intelligent index analysis, recommendations, and automated optimization
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
import re
from collections import defaultdict, Counter
import redis.asyncio as aioredis
from .database_optimization import get_connection_pool, QueryType

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Database index types"""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"        # Generalized Inverted Index
    GIST = "gist"      # Generalized Search Tree
    BRIN = "brin"      # Block Range Index
    PARTIAL = "partial"
    UNIQUE = "unique"
    COMPOSITE = "composite"


class IndexUsagePattern(Enum):
    """Index usage patterns"""
    FREQUENT = "frequent"        # Used frequently
    OCCASIONAL = "occasional"    # Used occasionally
    RARE = "rare"               # Rarely used
    UNUSED = "unused"           # Never used
    DUPLICATE = "duplicate"     # Duplicate of another index


@dataclass
class IndexStatistics:
    """Database index statistics"""
    index_name: str
    table_name: str
    index_type: IndexType
    columns: List[str]
    size_bytes: int
    index_scans: int
    tuples_read: int
    tuples_fetched: int
    blocks_read: int
    blocks_hit: int
    last_used: Optional[datetime]
    created_at: datetime
    usage_pattern: IndexUsagePattern = IndexUsagePattern.OCCASIONAL
    selectivity: float = 0.0  # 0-1, higher is more selective
    maintenance_cost: float = 0.0  # Cost of maintaining index


@dataclass
class IndexRecommendation:
    """Index creation/modification recommendation"""
    recommendation_id: str
    recommendation_type: str  # "create", "drop", "modify"
    table_name: str
    columns: List[str]
    index_type: IndexType
    estimated_benefit: float  # 0-100 score
    estimated_cost: float    # 0-100 score
    reason: str
    priority: str  # "high", "medium", "low"
    sql_statement: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryIndexAnalysis:
    """Analysis of query index usage"""
    query_hash: str
    query_type: QueryType
    tables_accessed: List[str]
    columns_in_where: List[str]
    columns_in_join: List[str]
    columns_in_order: List[str]
    columns_in_group: List[str]
    existing_indexes_used: List[str]
    suggested_indexes: List[IndexRecommendation]
    performance_impact: str  # "high", "medium", "low"


class IndexAnalyzer:
    """Intelligent database index analyzer"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
        # Index tracking
        self.current_indexes: Dict[str, IndexStatistics] = {}
        self.index_usage_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.query_patterns: Dict[str, QueryIndexAnalysis] = {}
        
        # Analysis configuration
        self.analysis_interval = 3600  # 1 hour
        self.analysis_task: Optional[asyncio.Task] = None
        self.min_selectivity = 0.01  # Minimum selectivity for useful index
        self.unused_index_threshold_days = 30
        
        # Recommendation engine
        self.recommendations: Dict[str, IndexRecommendation] = {}
        self.recommendation_handlers: List[callable] = []
        
        # Statistics
        self.stats = {
            "indexes_analyzed": 0,
            "recommendations_generated": 0,
            "queries_analyzed": 0,
            "unused_indexes_detected": 0
        }
    
    async def start_analysis(self):
        """Start continuous index analysis"""
        if self.analysis_task is None or self.analysis_task.done():
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            logger.info("✅ Index analysis started")
    
    async def stop_analysis(self):
        """Stop index analysis"""
        if self.analysis_task and not self.analysis_task.done():
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Index analysis stopped")
    
    async def _analysis_loop(self):
        """Main analysis loop"""
        while True:
            try:
                # Collect current index statistics
                await self._collect_index_statistics()
                
                # Analyze query patterns
                await self._analyze_query_patterns()
                
                # Generate recommendations
                await self._generate_index_recommendations()
                
                # Detect unused indexes
                await self._detect_unused_indexes()
                
                # Store analysis results
                await self._store_analysis_results()
                
                await asyncio.sleep(self.analysis_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in index analysis loop: {e}")
                await asyncio.sleep(self.analysis_interval)
    
    async def _collect_index_statistics(self):
        """Collect current database index statistics"""
        try:
            connection_pool = get_connection_pool()
            
            # Query to get index statistics (PostgreSQL specific)
            index_stats_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch,
                pg_relation_size(indexrelname::regclass) as size_bytes,
                pg_stat_get_blocks_read(indexrelid) as blocks_read,
                pg_stat_get_blocks_hit(indexrelid) as blocks_hit
            FROM pg_stat_user_indexes 
            JOIN pg_indexes ON indexname = pg_indexes.indexname
            WHERE schemaname = 'public';
            """
            
            async with connection_pool.get_connection(QueryType.SELECT) as (conn, pool_name):
                result = await conn.fetch(index_stats_query)
                
                for row in result:
                    await self._process_index_statistics(dict(row))
            
            self.stats["indexes_analyzed"] = len(self.current_indexes)
            
        except Exception as e:
            logger.error(f"Error collecting index statistics: {e}")
    
    async def _process_index_statistics(self, stats_row: Dict[str, Any]):
        """Process individual index statistics"""
        index_name = stats_row["indexname"]
        table_name = stats_row["tablename"]
        
        # Get index definition
        index_definition = await self._get_index_definition(index_name)
        
        if not index_definition:
            return
        
        # Create or update index statistics
        index_stats = IndexStatistics(
            index_name=index_name,
            table_name=table_name,
            index_type=self._determine_index_type(index_definition),
            columns=self._extract_index_columns(index_definition),
            size_bytes=stats_row.get("size_bytes", 0),
            index_scans=stats_row.get("idx_scan", 0),
            tuples_read=stats_row.get("idx_tup_read", 0),
            tuples_fetched=stats_row.get("idx_tup_fetch", 0),
            blocks_read=stats_row.get("blocks_read", 0),
            blocks_hit=stats_row.get("blocks_hit", 0),
            last_used=await self._get_last_used_time(index_name),
            created_at=await self._get_index_created_time(index_name)
        )
        
        # Calculate selectivity and usage pattern
        index_stats.selectivity = await self._calculate_index_selectivity(
            table_name, index_stats.columns
        )
        index_stats.usage_pattern = self._determine_usage_pattern(index_stats)
        index_stats.maintenance_cost = self._calculate_maintenance_cost(index_stats)
        
        self.current_indexes[index_name] = index_stats
        
        # Store usage history
        self.index_usage_history[index_name].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scans": index_stats.index_scans,
            "tuples_read": index_stats.tuples_read,
            "size_bytes": index_stats.size_bytes
        })
        
        # Keep only last 100 entries
        self.index_usage_history[index_name] = self.index_usage_history[index_name][-100:]
    
    async def _get_index_definition(self, index_name: str) -> Optional[str]:
        """Get index definition SQL"""
        try:
            connection_pool = get_connection_pool()
            
            query = """
            SELECT pg_get_indexdef(indexrelid) as definition
            FROM pg_stat_user_indexes 
            WHERE indexname = $1;
            """
            
            async with connection_pool.get_connection(QueryType.SELECT) as (conn, pool_name):
                result = await conn.fetchval(query, index_name)
                return result
                
        except Exception as e:
            logger.error(f"Error getting index definition for {index_name}: {e}")
            return None
    
    def _determine_index_type(self, definition: str) -> IndexType:
        """Determine index type from definition"""
        definition_lower = definition.lower()
        
        if "using gin" in definition_lower:
            return IndexType.GIN
        elif "using gist" in definition_lower:
            return IndexType.GIST
        elif "using hash" in definition_lower:
            return IndexType.HASH
        elif "using brin" in definition_lower:
            return IndexType.BRIN
        elif "unique" in definition_lower:
            return IndexType.UNIQUE
        elif "," in definition_lower:  # Multiple columns
            return IndexType.COMPOSITE
        else:
            return IndexType.BTREE
    
    def _extract_index_columns(self, definition: str) -> List[str]:
        """Extract column names from index definition"""
        try:
            # Simple regex to extract column names
            # This is a simplified version - production would need more robust parsing
            match = re.search(r'\((.*?)\)', definition)
            if match:
                columns_str = match.group(1)
                # Split by comma and clean up
                columns = [col.strip().strip('"') for col in columns_str.split(',')]
                return [col for col in columns if col]
            return []
        except Exception as e:
            logger.error(f"Error extracting columns from definition: {e}")
            return []
    
    async def _get_last_used_time(self, index_name: str) -> Optional[datetime]:
        """Get last time index was used (placeholder implementation)"""
        # This would require additional tracking or pg_stat_statements
        return None
    
    async def _get_index_created_time(self, index_name: str) -> datetime:
        """Get index creation time (placeholder implementation)"""
        # This would require system catalog queries
        return datetime.now(timezone.utc)
    
    async def _calculate_index_selectivity(self, table_name: str, columns: List[str]) -> float:
        """Calculate index selectivity"""
        try:
            if not columns:
                return 0.0
            
            connection_pool = get_connection_pool()
            
            # Simple selectivity calculation for first column
            # Production implementation would handle composite indexes better
            column = columns[0]
            
            query = f"""
            SELECT 
                COUNT(DISTINCT {column})::float / COUNT({column})::float as selectivity
            FROM {table_name}
            WHERE {column} IS NOT NULL;
            """
            
            async with connection_pool.get_connection(QueryType.SELECT) as (conn, pool_name):
                result = await conn.fetchval(query)
                return float(result) if result else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating selectivity for {table_name}.{columns}: {e}")
            return 0.0
    
    def _determine_usage_pattern(self, index_stats: IndexStatistics) -> IndexUsagePattern:
        """Determine index usage pattern"""
        scans = index_stats.index_scans
        
        if scans == 0:
            return IndexUsagePattern.UNUSED
        elif scans < 10:
            return IndexUsagePattern.RARE
        elif scans < 100:
            return IndexUsagePattern.OCCASIONAL
        else:
            return IndexUsagePattern.FREQUENT
    
    def _calculate_maintenance_cost(self, index_stats: IndexStatistics) -> float:
        """Calculate index maintenance cost (0-100 scale)"""
        # Factors: size, type, table update frequency
        size_cost = min(index_stats.size_bytes / (100 * 1024 * 1024), 1.0) * 30  # Size up to 100MB
        
        type_cost = {
            IndexType.BTREE: 10,
            IndexType.HASH: 5,
            IndexType.GIN: 25,
            IndexType.GIST: 20,
            IndexType.BRIN: 5,
            IndexType.PARTIAL: 8,
            IndexType.UNIQUE: 15,
            IndexType.COMPOSITE: 20
        }.get(index_stats.index_type, 10)
        
        return size_cost + type_cost
    
    async def _analyze_query_patterns(self):
        """Analyze query patterns for index optimization opportunities"""
        try:
            # Get recent slow queries from Redis
            slow_queries_data = await self.redis.lrange("slow_queries", 0, 999)
            
            for query_data in slow_queries_data:
                try:
                    query_info = json.loads(query_data)
                    await self._analyze_individual_query(query_info)
                except json.JSONDecodeError:
                    continue
            
            self.stats["queries_analyzed"] += len(slow_queries_data)
            
        except Exception as e:
            logger.error(f"Error analyzing query patterns: {e}")
    
    async def _analyze_individual_query(self, query_info: Dict[str, Any]):
        """Analyze individual query for index opportunities"""
        query_hash = query_info.get("query_hash")
        if not query_hash:
            return
        
        # This is a simplified analysis - production would use actual query parsing
        tables = query_info.get("tables", [])
        
        # Create query analysis
        analysis = QueryIndexAnalysis(
            query_hash=query_hash,
            query_type=QueryType.SELECT,  # Would be determined from actual query
            tables_accessed=tables,
            columns_in_where=[],  # Would be extracted from query
            columns_in_join=[],   # Would be extracted from query
            columns_in_order=[],  # Would be extracted from query
            columns_in_group=[],  # Would be extracted from query
            existing_indexes_used=[],  # Would come from query plan
            suggested_indexes=[],
            performance_impact="medium"
        )
        
        # Generate index suggestions for this query
        suggestions = await self._generate_query_index_suggestions(analysis)
        analysis.suggested_indexes = suggestions
        
        self.query_patterns[query_hash] = analysis
    
    async def _generate_query_index_suggestions(self, analysis: QueryIndexAnalysis) -> List[IndexRecommendation]:
        """Generate index suggestions for specific query"""
        suggestions = []
        
        # Example suggestions based on query analysis
        for table in analysis.tables_accessed:
            for column in analysis.columns_in_where:
                # Check if index already exists
                existing_index = self._find_existing_index(table, [column])
                
                if not existing_index:
                    suggestion = IndexRecommendation(
                        recommendation_id=f"idx_{table}_{column}_{int(datetime.now().timestamp())}",
                        recommendation_type="create",
                        table_name=table,
                        columns=[column],
                        index_type=IndexType.BTREE,
                        estimated_benefit=70.0,  # Would be calculated based on query frequency
                        estimated_cost=20.0,
                        reason=f"Frequently filtered column in WHERE clause",
                        priority="medium",
                        sql_statement=f"CREATE INDEX idx_{table}_{column} ON {table} ({column});",
                        metadata={"query_hash": analysis.query_hash}
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _find_existing_index(self, table_name: str, columns: List[str]) -> Optional[IndexStatistics]:
        """Find existing index for table and columns"""
        for index_stats in self.current_indexes.values():
            if (index_stats.table_name == table_name and 
                set(index_stats.columns) == set(columns)):
                return index_stats
        return None
    
    async def _generate_index_recommendations(self):
        """Generate comprehensive index recommendations"""
        try:
            self.recommendations.clear()
            
            # Analyze unused indexes for removal
            await self._recommend_unused_index_removal()
            
            # Analyze missing indexes
            await self._recommend_missing_indexes()
            
            # Analyze composite index opportunities
            await self._recommend_composite_indexes()
            
            # Analyze partial index opportunities
            await self._recommend_partial_indexes()
            
            self.stats["recommendations_generated"] = len(self.recommendations)
            
        except Exception as e:
            logger.error(f"Error generating index recommendations: {e}")
    
    async def _recommend_unused_index_removal(self):
        """Recommend removal of unused indexes"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.unused_index_threshold_days)
        
        for index_name, index_stats in self.current_indexes.items():
            if (index_stats.usage_pattern == IndexUsagePattern.UNUSED and 
                index_stats.created_at < cutoff_date):
                
                recommendation = IndexRecommendation(
                    recommendation_id=f"drop_{index_name}",
                    recommendation_type="drop",
                    table_name=index_stats.table_name,
                    columns=index_stats.columns,
                    index_type=index_stats.index_type,
                    estimated_benefit=index_stats.maintenance_cost,  # Benefit = saved maintenance cost
                    estimated_cost=5.0,  # Low cost to drop
                    reason=f"Index unused for {self.unused_index_threshold_days} days",
                    priority="low",
                    sql_statement=f"DROP INDEX {index_name};",
                    metadata={"size_bytes": index_stats.size_bytes}
                )
                
                self.recommendations[recommendation.recommendation_id] = recommendation
                self.stats["unused_indexes_detected"] += 1
    
    async def _recommend_missing_indexes(self):
        """Recommend creation of missing indexes based on query patterns"""
        # Collect frequently accessed columns without indexes
        column_usage = defaultdict(int)
        
        for analysis in self.query_patterns.values():
            for table in analysis.tables_accessed:
                for column in analysis.columns_in_where:
                    column_usage[f"{table}.{column}"] += 1
        
        # Recommend indexes for frequently used columns
        for table_column, usage_count in column_usage.items():
            if usage_count >= 5:  # Used in at least 5 queries
                table, column = table_column.split(".", 1)
                
                if not self._find_existing_index(table, [column]):
                    recommendation = IndexRecommendation(
                        recommendation_id=f"create_{table}_{column}",
                        recommendation_type="create",
                        table_name=table,
                        columns=[column],
                        index_type=IndexType.BTREE,
                        estimated_benefit=min(usage_count * 10, 100),
                        estimated_cost=30.0,
                        reason=f"Column used in {usage_count} queries",
                        priority="high" if usage_count >= 20 else "medium",
                        sql_statement=f"CREATE INDEX idx_{table}_{column} ON {table} ({column});",
                        metadata={"usage_count": usage_count}
                    )
                    
                    self.recommendations[recommendation.recommendation_id] = recommendation
    
    async def _recommend_composite_indexes(self):
        """Recommend composite indexes for multi-column queries"""
        # Analyze queries that use multiple columns together
        multi_column_patterns = defaultdict(int)
        
        for analysis in self.query_patterns.values():
            if len(analysis.columns_in_where) > 1:
                for table in analysis.tables_accessed:
                    columns_key = f"{table}:{','.join(sorted(analysis.columns_in_where))}"
                    multi_column_patterns[columns_key] += 1
        
        # Recommend composite indexes for frequent patterns
        for pattern, count in multi_column_patterns.items():
            if count >= 3:  # Pattern appears in at least 3 queries
                table, columns_str = pattern.split(":", 1)
                columns = columns_str.split(",")
                
                if not self._find_existing_index(table, columns):
                    recommendation = IndexRecommendation(
                        recommendation_id=f"composite_{table}_{'_'.join(columns)}",
                        recommendation_type="create",
                        table_name=table,
                        columns=columns,
                        index_type=IndexType.COMPOSITE,
                        estimated_benefit=count * 15,
                        estimated_cost=50.0,
                        reason=f"Multi-column pattern used in {count} queries",
                        priority="high" if count >= 10 else "medium",
                        sql_statement=f"CREATE INDEX idx_{table}_{'_'.join(columns)} ON {table} ({', '.join(columns)});",
                        metadata={"pattern_count": count}
                    )
                    
                    self.recommendations[recommendation.recommendation_id] = recommendation
    
    async def _recommend_partial_indexes(self):
        """Recommend partial indexes for filtered queries"""
        # This would analyze queries with consistent WHERE conditions
        # and recommend partial indexes for frequently filtered data
        pass
    
    async def _detect_unused_indexes(self):
        """Detect and flag unused indexes"""
        unused_count = 0
        
        for index_stats in self.current_indexes.values():
            if index_stats.usage_pattern == IndexUsagePattern.UNUSED:
                unused_count += 1
        
        self.stats["unused_indexes_detected"] = unused_count
    
    async def _store_analysis_results(self):
        """Store analysis results in Redis"""
        try:
            # Store recommendations
            recommendations_data = {
                rec_id: {
                    "type": rec.recommendation_type,
                    "table": rec.table_name,
                    "columns": rec.columns,
                    "benefit": rec.estimated_benefit,
                    "cost": rec.estimated_cost,
                    "priority": rec.priority,
                    "reason": rec.reason,
                    "sql": rec.sql_statement
                }
                for rec_id, rec in self.recommendations.items()
            }
            
            await self.redis.setex(
                "index_recommendations",
                3600,  # 1 hour TTL
                json.dumps(recommendations_data)
            )
            
            # Store statistics
            await self.redis.setex(
                "index_analysis_stats",
                3600,
                json.dumps(self.stats)
            )
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    def add_recommendation_handler(self, handler: callable):
        """Add handler for new recommendations"""
        self.recommendation_handlers.append(handler)
    
    async def get_index_health_report(self) -> Dict[str, Any]:
        """Get comprehensive index health report"""
        total_indexes = len(self.current_indexes)
        unused_indexes = len([idx for idx in self.current_indexes.values() 
                            if idx.usage_pattern == IndexUsagePattern.UNUSED])
        
        total_size = sum(idx.size_bytes for idx in self.current_indexes.values())
        unused_size = sum(idx.size_bytes for idx in self.current_indexes.values() 
                         if idx.usage_pattern == IndexUsagePattern.UNUSED)
        
        return {
            "summary": {
                "total_indexes": total_indexes,
                "unused_indexes": unused_indexes,
                "unused_percentage": (unused_indexes / total_indexes * 100) if total_indexes > 0 else 0,
                "total_size_mb": total_size / (1024 * 1024),
                "unused_size_mb": unused_size / (1024 * 1024),
                "wasted_space_percentage": (unused_size / total_size * 100) if total_size > 0 else 0
            },
            "usage_distribution": {
                pattern.value: len([idx for idx in self.current_indexes.values() 
                                 if idx.usage_pattern == pattern])
                for pattern in IndexUsagePattern
            },
            "type_distribution": {
                idx_type.value: len([idx for idx in self.current_indexes.values() 
                                   if idx.index_type == idx_type])
                for idx_type in IndexType
            },
            "recommendations": {
                "total": len(self.recommendations),
                "high_priority": len([r for r in self.recommendations.values() if r.priority == "high"]),
                "medium_priority": len([r for r in self.recommendations.values() if r.priority == "medium"]),
                "low_priority": len([r for r in self.recommendations.values() if r.priority == "low"])
            },
            "statistics": self.stats
        }
    
    async def get_table_index_analysis(self, table_name: str) -> Dict[str, Any]:
        """Get detailed index analysis for specific table"""
        table_indexes = {
            name: stats for name, stats in self.current_indexes.items()
            if stats.table_name == table_name
        }
        
        table_recommendations = {
            rec_id: rec for rec_id, rec in self.recommendations.items()
            if rec.table_name == table_name
        }
        
        return {
            "table_name": table_name,
            "indexes": {
                name: {
                    "type": stats.index_type.value,
                    "columns": stats.columns,
                    "size_mb": stats.size_bytes / (1024 * 1024),
                    "usage_pattern": stats.usage_pattern.value,
                    "selectivity": stats.selectivity,
                    "scans": stats.index_scans,
                    "maintenance_cost": stats.maintenance_cost
                }
                for name, stats in table_indexes.items()
            },
            "recommendations": {
                rec_id: {
                    "type": rec.recommendation_type,
                    "columns": rec.columns,
                    "priority": rec.priority,
                    "benefit": rec.estimated_benefit,
                    "reason": rec.reason,
                    "sql": rec.sql_statement
                }
                for rec_id, rec in table_recommendations.items()
            }
        }


# Global index analyzer instance
index_analyzer: Optional[IndexAnalyzer] = None


async def initialize_index_analyzer(redis_client: aioredis.Redis):
    """Initialize the global index analyzer"""
    global index_analyzer
    
    index_analyzer = IndexAnalyzer(redis_client)
    await index_analyzer.start_analysis()
    
    logger.info("✅ Index analyzer initialized")


def get_index_analyzer() -> IndexAnalyzer:
    """Get the global index analyzer instance"""
    if index_analyzer is None:
        raise RuntimeError("Index analyzer not initialized.")
    return index_analyzer


async def shutdown_index_analyzer():
    """Shutdown the index analyzer"""
    if index_analyzer:
        await index_analyzer.stop_analysis()