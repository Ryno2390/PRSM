#!/usr/bin/env python3
"""
Enterprise Data Connectors
===========================

Comprehensive data connectivity suite supporting SQL databases, NoSQL stores,
REST APIs, file systems, and streaming platforms.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
import urllib.parse
import hashlib

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class ConnectorType(Enum):
    """Types of data connectors"""
    SQL_DATABASE = "sql_database"
    NOSQL_DATABASE = "nosql_database"
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    FILE_SYSTEM = "file_system"
    CLOUD_STORAGE = "cloud_storage"
    MESSAGE_QUEUE = "message_queue"
    STREAM_PLATFORM = "stream_platform"
    WEBHOOK = "webhook"
    FTP_SFTP = "ftp_sftp"


class ConnectionStatus(Enum):
    """Connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    AUTHENTICATION_FAILED = "authentication_failed"
    TIMEOUT = "timeout"


@dataclass
class ConnectionConfig:
    """Configuration for data source connections"""
    connector_id: str
    connector_type: ConnectorType
    name: str
    description: str = ""
    
    # Connection parameters
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    
    # API specific
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # File system specific
    path: Optional[str] = None
    
    # Advanced options
    ssl_enabled: bool = True
    timeout_seconds: int = 30
    retry_count: int = 3
    pool_size: int = 10
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding sensitive data"""
        data = {
            "connector_id": self.connector_id,
            "connector_type": self.connector_type.value,
            "name": self.name,
            "description": self.description,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "base_url": self.base_url,
            "path": self.path,
            "ssl_enabled": self.ssl_enabled,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "pool_size": self.pool_size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags
        }
        
        # Exclude sensitive information
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class QueryRequest:
    """Request for data query operations"""
    query: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    offset: int = 0
    timeout: Optional[int] = None
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    def get_cache_key(self) -> str:
        """Generate cache key for the query"""
        key_data = {
            "query": self.query,
            "parameters": self.parameters,
            "limit": self.limit,
            "offset": self.offset
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class QueryResult:
    """Result of a data query operation"""
    data: List[Dict[str, Any]]
    columns: List[str]
    total_rows: int
    execution_time_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "data": self.data,
            "columns": self.columns,
            "total_rows": self.total_rows,
            "execution_time_ms": self.execution_time_ms,
            "cached": self.cached,
            "metadata": self.metadata
        }


class DataConnector(ABC):
    """Abstract base class for all data connectors"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.connection = None
        self.last_error = None
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_transferred": 0
        }
        
        # Query cache
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized {self.__class__.__name__} connector: {config.name}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the data source"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if connection is healthy"""
        pass
    
    @abstractmethod
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute a query against the data source"""
        pass
    
    async def execute_batch(self, requests: List[QueryRequest]) -> List[QueryResult]:
        """Execute multiple queries in batch"""
        results = []
        for request in requests:
            try:
                result = await self.execute_query(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch query failed: {e}")
                results.append(QueryResult(
                    data=[],
                    columns=[],
                    total_rows=0,
                    execution_time_ms=0.0,
                    metadata={"error": str(e)}
                ))
        return results
    
    def get_cached_result(self, cache_key: str, ttl: int) -> Optional[QueryResult]:
        """Get cached query result if valid"""
        if cache_key not in self.query_cache:
            return None
        
        cached_entry = self.query_cache[cache_key]
        cache_age = (datetime.now(timezone.utc) - cached_entry["cached_at"]).total_seconds()
        
        if cache_age > ttl:
            del self.query_cache[cache_key]
            return None
        
        self.stats["cache_hits"] += 1
        cached_entry["result"].cached = True
        return cached_entry["result"]
    
    def cache_result(self, cache_key: str, result: QueryResult):
        """Cache query result"""
        self.query_cache[cache_key] = {
            "result": result,
            "cached_at": datetime.now(timezone.utc)
        }
        
        # Limit cache size
        if len(self.query_cache) > 1000:
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k]["cached_at"])
            del self.query_cache[oldest_key]
    
    def update_stats(self, execution_time: float, success: bool, bytes_count: int = 0):
        """Update connector statistics"""
        self.stats["total_queries"] += 1
        self.stats["total_execution_time"] += execution_time
        self.stats["bytes_transferred"] += bytes_count
        
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
        
        # Update average
        self.stats["avg_execution_time"] = \
            self.stats["total_execution_time"] / self.stats["total_queries"]
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status and statistics"""
        return {
            "connector_id": self.config.connector_id,
            "name": self.config.name,
            "type": self.config.connector_type.value,
            "status": self.status.value,
            "last_error": self.last_error,
            "statistics": self.stats,
            "cache_size": len(self.query_cache),
            "config": self.config.to_dict()
        }


class SQLConnector(DataConnector):
    """SQL Database connector supporting multiple database engines"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # Try to import SQL dependencies
        self.asyncpg = require_optional("asyncpg")  # PostgreSQL
        self.aiomysql = require_optional("aiomysql")  # MySQL
        self.aiosqlite = require_optional("aiosqlite")  # SQLite
        self.sqlalchemy = require_optional("sqlalchemy")
        
        # Determine database engine
        self.engine_type = self._detect_engine_type()
        
    def _detect_engine_type(self) -> str:
        """Detect database engine from connection parameters"""
        if self.config.connection_string:
            conn_str = self.config.connection_string.lower()
            if conn_str.startswith("postgresql://") or conn_str.startswith("postgres://"):
                return "postgresql"
            elif conn_str.startswith("mysql://"):
                return "mysql"
            elif conn_str.startswith("sqlite://"):
                return "sqlite"
        
        # Default based on port
        if self.config.port == 5432:
            return "postgresql"
        elif self.config.port == 3306:
            return "mysql"
        
        return "postgresql"  # Default
    
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            if self.engine_type == "postgresql" and self.asyncpg:
                self.connection = await self.asyncpg.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    command_timeout=self.config.timeout_seconds
                )
            elif self.engine_type == "mysql" and self.aiomysql:
                self.connection = await self.aiomysql.connect(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.timeout_seconds
                )
            elif self.engine_type == "sqlite" and self.aiosqlite:
                self.connection = await self.aiosqlite.connect(self.config.database)
            else:
                raise Exception(f"Unsupported database engine: {self.engine_type}")
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to {self.engine_type} database: {self.config.name}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to database {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close database connection"""
        try:
            if self.connection:
                await self.connection.close()
                self.connection = None
            
            self.status = ConnectionStatus.DISCONNECTED
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from database {self.config.name}: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test database connection health"""
        try:
            if not self.connection:
                return False
            
            # Simple test query
            if self.engine_type == "postgresql":
                await self.connection.fetchval("SELECT 1")
            elif self.engine_type == "mysql":
                async with self.connection.cursor() as cursor:
                    await cursor.execute("SELECT 1")
            elif self.engine_type == "sqlite":
                await self.connection.execute("SELECT 1")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed for {self.config.name}: {e}")
            return False
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute SQL query"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if request.cache_enabled:
                cache_key = request.get_cache_key()
                cached_result = self.get_cached_result(cache_key, request.cache_ttl)
                if cached_result:
                    return cached_result
            
            if not self.connection:
                await self.connect()
            
            # Execute query based on engine type
            if self.engine_type == "postgresql":
                result = await self._execute_postgresql_query(request)
            elif self.engine_type == "mysql":
                result = await self._execute_mysql_query(request)
            elif self.engine_type == "sqlite":
                result = await self._execute_sqlite_query(request)
            else:
                raise Exception(f"Unsupported engine: {self.engine_type}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            query_result = QueryResult(
                data=result["data"],
                columns=result["columns"],
                total_rows=len(result["data"]),
                execution_time_ms=execution_time,
                metadata={"engine": self.engine_type}
            )
            
            # Cache result
            if request.cache_enabled:
                self.cache_result(cache_key, query_result)
                self.stats["cache_misses"] += 1
            
            self.update_stats(execution_time, True, len(str(result["data"])))
            return query_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.update_stats(execution_time, False)
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def _execute_postgresql_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Execute PostgreSQL query"""
        rows = await self.connection.fetch(request.query)
        
        if not rows:
            return {"data": [], "columns": []}
        
        columns = list(rows[0].keys())
        data = [dict(row) for row in rows]
        
        # Apply limit if specified
        if request.limit:
            data = data[request.offset:request.offset + request.limit]
        
        return {"data": data, "columns": columns}
    
    async def _execute_mysql_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Execute MySQL query"""
        async with self.connection.cursor() as cursor:
            await cursor.execute(request.query)
            rows = await cursor.fetchall()
            
            if not rows:
                return {"data": [], "columns": []}
            
            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            
            # Apply limit if specified
            if request.limit:
                data = data[request.offset:request.offset + request.limit]
            
            return {"data": data, "columns": columns}
    
    async def _execute_sqlite_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Execute SQLite query"""
        cursor = await self.connection.execute(request.query)
        rows = await cursor.fetchall()
        
        if not rows:
            return {"data": [], "columns": []}
        
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]
        
        # Apply limit if specified
        if request.limit:
            data = data[request.offset:request.offset + request.limit]
        
        return {"data": data, "columns": columns}


class NoSQLConnector(DataConnector):
    """NoSQL Database connector supporting MongoDB, Redis, etc."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # Try to import NoSQL dependencies
        self.motor = require_optional("motor")  # MongoDB async driver
        self.redis = require_optional("redis")  # Redis
        
        # Determine NoSQL engine
        self.engine_type = self._detect_engine_type()
    
    def _detect_engine_type(self) -> str:
        """Detect NoSQL engine type"""
        if self.config.port == 27017:
            return "mongodb"
        elif self.config.port == 6379:
            return "redis"
        
        # Check connection string
        if self.config.connection_string:
            conn_str = self.config.connection_string.lower()
            if "mongodb://" in conn_str:
                return "mongodb"
            elif "redis://" in conn_str:
                return "redis"
        
        return "mongodb"  # Default
    
    async def connect(self) -> bool:
        """Establish NoSQL connection"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            if self.engine_type == "mongodb" and self.motor:
                client = self.motor.motor_asyncio.AsyncIOMotorClient(
                    host=self.config.host,
                    port=self.config.port,
                    username=self.config.username,
                    password=self.config.password,
                    serverSelectionTimeoutMS=self.config.timeout_seconds * 1000
                )
                self.connection = client[self.config.database]
                
                # Test connection
                await client.admin.command('ping')
                
            elif self.engine_type == "redis" and self.redis:
                self.connection = self.redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    decode_responses=True,
                    socket_timeout=self.config.timeout_seconds
                )
                
                # Test connection
                await self.connection.ping()
            
            else:
                raise Exception(f"Unsupported NoSQL engine: {self.engine_type}")
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to {self.engine_type}: {self.config.name}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to {self.engine_type} {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close NoSQL connection"""
        try:
            if self.connection:
                if self.engine_type == "mongodb":
                    self.connection.client.close()
                elif self.engine_type == "redis":
                    await self.connection.close()
                
                self.connection = None
            
            self.status = ConnectionStatus.DISCONNECTED
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.engine_type} {self.config.name}: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test NoSQL connection health"""
        try:
            if not self.connection:
                return False
            
            if self.engine_type == "mongodb":
                await self.connection.client.admin.command('ping')
            elif self.engine_type == "redis":
                await self.connection.ping()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed for {self.config.name}: {e}")
            return False
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute NoSQL query"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if request.cache_enabled:
                cache_key = request.get_cache_key()
                cached_result = self.get_cached_result(cache_key, request.cache_ttl)
                if cached_result:
                    return cached_result
            
            if not self.connection:
                await self.connect()
            
            # Execute query based on engine type
            if self.engine_type == "mongodb":
                result = await self._execute_mongodb_query(request)
            elif self.engine_type == "redis":
                result = await self._execute_redis_query(request)
            else:
                raise Exception(f"Unsupported engine: {self.engine_type}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            query_result = QueryResult(
                data=result["data"],
                columns=result.get("columns", []),
                total_rows=len(result["data"]),
                execution_time_ms=execution_time,
                metadata={"engine": self.engine_type}
            )
            
            # Cache result
            if request.cache_enabled:
                self.cache_result(cache_key, query_result)
                self.stats["cache_misses"] += 1
            
            self.update_stats(execution_time, True, len(str(result["data"])))
            return query_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.update_stats(execution_time, False)
            logger.error(f"NoSQL query execution failed: {e}")
            raise
    
    async def _execute_mongodb_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Execute MongoDB query"""
        # Parse query as JSON
        query_data = json.loads(request.query)
        collection_name = query_data.get("collection")
        operation = query_data.get("operation", "find")
        filter_query = query_data.get("filter", {})
        
        collection = self.connection[collection_name]
        
        if operation == "find":
            cursor = collection.find(filter_query)
            
            if request.limit:
                cursor = cursor.skip(request.offset).limit(request.limit)
            
            documents = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            # Extract column names
            columns = list(documents[0].keys()) if documents else []
            
            return {"data": documents, "columns": columns}
        
        elif operation == "aggregate":
            pipeline = query_data.get("pipeline", [])
            cursor = collection.aggregate(pipeline)
            documents = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            columns = list(documents[0].keys()) if documents else []
            return {"data": documents, "columns": columns}
        
        else:
            raise Exception(f"Unsupported MongoDB operation: {operation}")
    
    async def _execute_redis_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Execute Redis query"""
        # Parse query as JSON
        query_data = json.loads(request.query)
        operation = query_data.get("operation", "get")
        key = query_data.get("key")
        
        if operation == "get":
            value = await self.connection.get(key)
            data = [{"key": key, "value": value}] if value else []
            return {"data": data, "columns": ["key", "value"]}
        
        elif operation == "keys":
            pattern = query_data.get("pattern", "*")
            keys = await self.connection.keys(pattern)
            data = [{"key": k} for k in keys]
            return {"data": data, "columns": ["key"]}
        
        elif operation == "hgetall":
            hash_data = await self.connection.hgetall(key)
            data = [{"field": k, "value": v} for k, v in hash_data.items()]
            return {"data": data, "columns": ["field", "value"]}
        
        else:
            raise Exception(f"Unsupported Redis operation: {operation}")


class RestAPIConnector(DataConnector):
    """REST API connector for HTTP-based data sources"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # Try to import HTTP client
        self.aiohttp = require_optional("aiohttp")
        self.session = None
    
    async def connect(self) -> bool:
        """Establish HTTP session"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            if not self.aiohttp:
                raise Exception("aiohttp not available for REST API connector")
            
            # Create session with custom headers
            headers = self.config.headers.copy()
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            timeout = self.aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = self.aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to REST API: {self.config.name}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to REST API {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close HTTP session"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.status = ConnectionStatus.DISCONNECTED
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from REST API {self.config.name}: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test API connection health"""
        try:
            if not self.session:
                return False
            
            # Make a simple HEAD request to base URL
            async with self.session.head(self.config.base_url) as response:
                return response.status < 400
            
        except Exception as e:
            logger.error(f"Connection test failed for {self.config.name}: {e}")
            return False
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute REST API request"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if request.cache_enabled:
                cache_key = request.get_cache_key()
                cached_result = self.get_cached_result(cache_key, request.cache_ttl)
                if cached_result:
                    return cached_result
            
            if not self.session:
                await self.connect()
            
            # Parse query as API request
            api_request = json.loads(request.query)
            method = api_request.get("method", "GET").upper()
            endpoint = api_request.get("endpoint", "")
            params = api_request.get("params", {})
            data = api_request.get("data", {})
            
            # Merge with request parameters
            params.update(request.parameters)
            if request.limit:
                params["limit"] = request.limit
            if request.offset:
                params["offset"] = request.offset
            
            # Build URL
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if method in ["POST", "PUT", "PATCH"] else None
            ) as response:
                
                if response.status >= 400:
                    raise Exception(f"API request failed with status {response.status}")
                
                response_data = await response.json()
                
                # Normalize response data
                if isinstance(response_data, list):
                    data_list = response_data
                elif isinstance(response_data, dict):
                    # Try common pagination patterns
                    data_list = (
                        response_data.get("data", []) or
                        response_data.get("items", []) or
                        response_data.get("results", []) or
                        [response_data]
                    )
                else:
                    data_list = [{"value": response_data}]
                
                # Extract columns
                columns = list(data_list[0].keys()) if data_list else []
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Create result
                query_result = QueryResult(
                    data=data_list,
                    columns=columns,
                    total_rows=len(data_list),
                    execution_time_ms=execution_time,
                    metadata={
                        "status_code": response.status,
                        "url": str(response.url)
                    }
                )
                
                # Cache result
                if request.cache_enabled:
                    self.cache_result(cache_key, query_result)
                    self.stats["cache_misses"] += 1
                
                self.update_stats(execution_time, True, len(str(data_list)))
                return query_result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.update_stats(execution_time, False)
            logger.error(f"REST API query execution failed: {e}")
            raise


class FileConnector(DataConnector):
    """File system connector for local and network file access"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # Try to import file processing libraries
        self.pandas = require_optional("pandas")
        self.aiofiles = require_optional("aiofiles")
        
        # Supported file formats
        self.supported_formats = [".csv", ".json", ".jsonl", ".parquet", ".xlsx", ".xml"]
    
    async def connect(self) -> bool:
        """Verify file path access"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            import os
            if not os.path.exists(self.config.path):
                raise Exception(f"Path does not exist: {self.config.path}")
            
            if not os.access(self.config.path, os.R_OK):
                raise Exception(f"No read access to path: {self.config.path}")
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to file system: {self.config.name}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to file system {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """No-op for file system"""
        self.status = ConnectionStatus.DISCONNECTED
        return True
    
    async def test_connection(self) -> bool:
        """Test file system access"""
        try:
            import os
            return os.path.exists(self.config.path) and os.access(self.config.path, os.R_OK)
        except Exception:
            return False
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute file query"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if request.cache_enabled:
                cache_key = request.get_cache_key()
                cached_result = self.get_cached_result(cache_key, request.cache_ttl)
                if cached_result:
                    return cached_result
            
            # Parse query as file operation
            file_query = json.loads(request.query)
            operation = file_query.get("operation", "read")
            file_path = file_query.get("file_path")
            
            if not file_path:
                raise Exception("file_path required in query")
            
            # Make path absolute relative to config path
            import os
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.config.path, file_path)
            
            if operation == "read":
                result = await self._read_file(file_path, request)
            elif operation == "list":
                result = await self._list_files(file_path, request)
            else:
                raise Exception(f"Unsupported file operation: {operation}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            query_result = QueryResult(
                data=result["data"],
                columns=result["columns"],
                total_rows=len(result["data"]),
                execution_time_ms=execution_time,
                metadata={"file_path": file_path}
            )
            
            # Cache result
            if request.cache_enabled:
                self.cache_result(cache_key, query_result)
                self.stats["cache_misses"] += 1
            
            self.update_stats(execution_time, True, len(str(result["data"])))
            return query_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.update_stats(execution_time, False)
            logger.error(f"File query execution failed: {e}")
            raise
    
    async def _read_file(self, file_path: str, request: QueryRequest) -> Dict[str, Any]:
        """Read file content"""
        import os
        
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".csv" and self.pandas:
            df = self.pandas.read_csv(file_path)
            data = df.to_dict('records')
            columns = list(df.columns)
            
        elif file_ext == ".json":
            if self.aiofiles:
                async with self.aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            
            json_data = json.loads(content)
            
            if isinstance(json_data, list):
                data = json_data
                columns = list(json_data[0].keys()) if json_data else []
            else:
                data = [json_data]
                columns = list(json_data.keys())
        
        elif file_ext == ".jsonl":
            data = []
            columns = set()
            
            if self.aiofiles:
                async with self.aiofiles.open(file_path, 'r') as f:
                    async for line in f:
                        if line.strip():
                            record = json.loads(line)
                            data.append(record)
                            columns.update(record.keys())
            else:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            data.append(record)
                            columns.update(record.keys())
            
            columns = list(columns)
        
        elif file_ext == ".parquet" and self.pandas:
            df = self.pandas.read_parquet(file_path)
            data = df.to_dict('records')
            columns = list(df.columns)
        
        else:
            # Plain text file
            if self.aiofiles:
                async with self.aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            
            lines = content.split('\n')
            data = [{"line_number": i+1, "content": line} for i, line in enumerate(lines)]
            columns = ["line_number", "content"]
        
        # Apply pagination
        if request.limit:
            data = data[request.offset:request.offset + request.limit]
        
        return {"data": data, "columns": columns}
    
    async def _list_files(self, dir_path: str, request: QueryRequest) -> Dict[str, Any]:
        """List files in directory"""
        import os
        
        if not os.path.isdir(dir_path):
            raise Exception(f"Directory not found: {dir_path}")
        
        files_data = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            stat = os.stat(item_path)
            
            files_data.append({
                "name": item,
                "path": item_path,
                "size": stat.st_size,
                "is_file": os.path.isfile(item_path),
                "is_directory": os.path.isdir(item_path),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        columns = ["name", "path", "size", "is_file", "is_directory", "modified_time"]
        
        # Apply pagination
        if request.limit:
            files_data = files_data[request.offset:request.offset + request.limit]
        
        return {"data": files_data, "columns": columns}


class StreamConnector(DataConnector):
    """Stream connector for message queues and event streams"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # Try to import streaming dependencies
        self.kafka = require_optional("aiokafka")  # Kafka
        self.pika = require_optional("pika")  # RabbitMQ
        
        # Determine stream engine
        self.engine_type = self._detect_engine_type()
        self.consumer = None
        self.producer = None
    
    def _detect_engine_type(self) -> str:
        """Detect streaming engine type"""
        if self.config.port == 9092:
            return "kafka"
        elif self.config.port == 5672:
            return "rabbitmq"
        
        # Check connection string
        if self.config.connection_string:
            conn_str = self.config.connection_string.lower()
            if "kafka" in conn_str:
                return "kafka"
            elif "rabbitmq" in conn_str or "amqp" in conn_str:
                return "rabbitmq"
        
        return "kafka"  # Default
    
    async def connect(self) -> bool:
        """Establish stream connection"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            if self.engine_type == "kafka" and self.kafka:
                # Create Kafka consumer
                self.consumer = self.kafka.AIOKafkaConsumer(
                    bootstrap_servers=[f"{self.config.host}:{self.config.port}"],
                    auto_offset_reset='earliest'
                )
                await self.consumer.start()
                
                # Create Kafka producer
                self.producer = self.kafka.AIOKafkaProducer(
                    bootstrap_servers=[f"{self.config.host}:{self.config.port}"]
                )
                await self.producer.start()
                
            elif self.engine_type == "rabbitmq" and self.pika:
                # RabbitMQ connection (simplified - would need proper async implementation)
                connection_params = self.pika.ConnectionParameters(
                    host=self.config.host,
                    port=self.config.port,
                    credentials=self.pika.PlainCredentials(
                        self.config.username, self.config.password
                    ) if self.config.username else None
                )
                self.connection = self.pika.BlockingConnection(connection_params)
                
            else:
                raise Exception(f"Unsupported stream engine: {self.engine_type}")
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to {self.engine_type}: {self.config.name}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to {self.engine_type} {self.config.name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close stream connection"""
        try:
            if self.engine_type == "kafka":
                if self.consumer:
                    await self.consumer.stop()
                if self.producer:
                    await self.producer.stop()
            elif self.engine_type == "rabbitmq":
                if self.connection:
                    self.connection.close()
            
            self.status = ConnectionStatus.DISCONNECTED
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.engine_type} {self.config.name}: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test stream connection health"""
        try:
            if self.engine_type == "kafka":
                return self.consumer is not None and self.producer is not None
            elif self.engine_type == "rabbitmq":
                return self.connection is not None and not self.connection.is_closed
            
            return False
            
        except Exception as e:
            logger.error(f"Connection test failed for {self.config.name}: {e}")
            return False
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute stream query (consume messages)"""
        start_time = datetime.now()
        
        try:
            if not await self.test_connection():
                await self.connect()
            
            # Parse query as stream operation
            stream_query = json.loads(request.query)
            operation = stream_query.get("operation", "consume")
            topic = stream_query.get("topic")
            
            if operation == "consume":
                result = await self._consume_messages(topic, request)
            elif operation == "produce":
                result = await self._produce_message(topic, stream_query.get("message"), request)
            else:
                raise Exception(f"Unsupported stream operation: {operation}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            query_result = QueryResult(
                data=result["data"],
                columns=result["columns"],
                total_rows=len(result["data"]),
                execution_time_ms=execution_time,
                metadata={"engine": self.engine_type, "topic": topic}
            )
            
            self.update_stats(execution_time, True, len(str(result["data"])))
            return query_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.update_stats(execution_time, False)
            logger.error(f"Stream query execution failed: {e}")
            raise
    
    async def _consume_messages(self, topic: str, request: QueryRequest) -> Dict[str, Any]:
        """Consume messages from stream"""
        messages = []
        
        if self.engine_type == "kafka":
            # Subscribe to topic
            self.consumer.subscribe([topic])
            
            # Consume messages with timeout
            timeout_ms = (request.timeout or 30) * 1000
            
            try:
                # Get messages
                msg_count = 0
                async for msg in self.consumer:
                    if request.limit and msg_count >= request.limit:
                        break
                    
                    messages.append({
                        "offset": msg.offset,
                        "partition": msg.partition,
                        "timestamp": msg.timestamp,
                        "key": msg.key.decode() if msg.key else None,
                        "value": msg.value.decode() if msg.value else None,
                        "topic": msg.topic
                    })
                    
                    msg_count += 1
                    
                    # Break after timeout
                    if (datetime.now() - datetime.now()).total_seconds() * 1000 > timeout_ms:
                        break
                        
            except Exception as e:
                logger.warning(f"Kafka consume timeout or error: {e}")
        
        elif self.engine_type == "rabbitmq":
            # Simplified RabbitMQ consume (would need proper async implementation)
            channel = self.connection.channel()
            
            for i in range(request.limit or 10):
                method, properties, body = channel.basic_get(queue=topic, auto_ack=True)
                if method:
                    messages.append({
                        "delivery_tag": method.delivery_tag,
                        "exchange": method.exchange,
                        "routing_key": method.routing_key,
                        "message": body.decode() if body else None,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    break  # No more messages
        
        columns = list(messages[0].keys()) if messages else []
        return {"data": messages, "columns": columns}
    
    async def _produce_message(self, topic: str, message: Any, request: QueryRequest) -> Dict[str, Any]:
        """Produce message to stream"""
        if self.engine_type == "kafka":
            await self.producer.send_and_wait(topic, json.dumps(message).encode())
            
            return {
                "data": [{"status": "sent", "topic": topic, "message": message}],
                "columns": ["status", "topic", "message"]
            }
        
        elif self.engine_type == "rabbitmq":
            channel = self.connection.channel()
            channel.queue_declare(queue=topic, durable=True)
            channel.basic_publish(
                exchange='',
                routing_key=topic,
                body=json.dumps(message)
            )
            
            return {
                "data": [{"status": "sent", "queue": topic, "message": message}],
                "columns": ["status", "queue", "message"]
            }
        
        else:
            raise Exception(f"Unsupported engine: {self.engine_type}")


# Export main classes
__all__ = [
    'ConnectorType',
    'ConnectionStatus',
    'ConnectionConfig',
    'QueryRequest',
    'QueryResult',
    'DataConnector',
    'SQLConnector',
    'NoSQLConnector',
    'RestAPIConnector',
    'FileConnector',
    'StreamConnector'
]