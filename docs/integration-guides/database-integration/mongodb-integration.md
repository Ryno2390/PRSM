# MongoDB Integration Guide

Integrate PRSM with MongoDB for flexible, document-based data storage and advanced NoSQL features.

## 🎯 Overview

This guide covers integrating PRSM with MongoDB, including setup, connection management, schema design, performance optimization, and production best practices.

## 📋 Prerequisites

- MongoDB 5.0+ installed
- PRSM instance configured
- Basic knowledge of MongoDB and NoSQL concepts
- Python development environment

## 🚀 Quick Start

### 1. MongoDB Setup

```bash
# Install MongoDB (Ubuntu/Debian)
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Install MongoDB (macOS with Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Install MongoDB (Docker)
docker run --name mongodb -d -p 27017:27017 -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=password mongo:6.0
```

### 2. Database and User Setup

```javascript
// Connect to MongoDB shell
mongosh

// Create database and user
use prsm_production

db.createUser({
  user: "prsm_user",
  pwd: "secure_password_here",
  roles: [
    { role: "readWrite", db: "prsm_production" },
    { role: "dbAdmin", db: "prsm_production" }
  ]
})

// Create collections with validation
db.createCollection("users", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["username", "email", "created_at"],
      properties: {
        username: {
          bsonType: "string",
          maxLength: 50
        },
        email: {
          bsonType: "string",
          pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        },
        created_at: {
          bsonType: "date"
        }
      }
    }
  }
})

// Exit shell
exit
```

### 3. Basic Connection Test

```python
# test_connection.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def test_connection():
    """Test MongoDB connection."""
    try:
        # Connect to MongoDB
        client = MongoClient(
            host="localhost",
            port=27017,
            username="prsm_user",
            password="secure_password_here",
            authSource="prsm_production"
        )
        
        # Test connection
        client.admin.command('ping')
        
        # Get database
        db = client.prsm_production
        
        # Test collection access
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
        
        print("MongoDB connection successful!")
        return True
        
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    test_connection()
```

## 🔧 PRSM MongoDB Configuration

### MongoDB Manager

```python
# prsm/core/mongodb.py
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, OperationFailure
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from contextlib import asynccontextmanager

class MongoDBConfig:
    def __init__(self):
        self.host = os.environ.get('MONGODB_HOST', 'localhost')
        self.port = int(os.environ.get('MONGODB_PORT', '27017'))
        self.database = os.environ.get('MONGODB_DATABASE', 'prsm_production')
        self.username = os.environ.get('MONGODB_USERNAME', 'prsm_user')
        self.password = os.environ.get('MONGODB_PASSWORD', '')
        self.auth_source = os.environ.get('MONGODB_AUTH_SOURCE', 'prsm_production')
        self.replica_set = os.environ.get('MONGODB_REPLICA_SET')
        self.ssl = os.environ.get('MONGODB_SSL', 'false').lower() == 'true'
        self.max_pool_size = int(os.environ.get('MONGODB_MAX_POOL_SIZE', '100'))
        self.min_pool_size = int(os.environ.get('MONGODB_MIN_POOL_SIZE', '10'))
        self.max_idle_time_ms = int(os.environ.get('MONGODB_MAX_IDLE_TIME_MS', '30000'))

class MongoDBManager:
    def __init__(self, config: MongoDBConfig = None):
        self.config = config or MongoDBConfig()
        self.client = None
        self.async_client = None
        self.database = None
        self.async_database = None
        self.logger = logging.getLogger(__name__)

    def _build_connection_string(self) -> str:
        """Build MongoDB connection string."""
        auth_part = ""
        if self.config.username and self.config.password:
            auth_part = f"{self.config.username}:{self.config.password}@"
        
        options = []
        if self.config.auth_source:
            options.append(f"authSource={self.config.auth_source}")
        if self.config.replica_set:
            options.append(f"replicaSet={self.config.replica_set}")
        if self.config.ssl:
            options.append("ssl=true")
        
        options.extend([
            f"maxPoolSize={self.config.max_pool_size}",
            f"minPoolSize={self.config.min_pool_size}",
            f"maxIdleTimeMS={self.config.max_idle_time_ms}"
        ])
        
        options_str = "&".join(options)
        
        return f"mongodb://{auth_part}{self.config.host}:{self.config.port}/{self.config.database}?{options_str}"

    def connect(self):
        """Create synchronous MongoDB connection."""
        try:
            connection_string = self._build_connection_string()
            self.client = MongoClient(connection_string)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.database = self.client[self.config.database]
            self.logger.info(f"Connected to MongoDB: {self.config.host}:{self.config.port}")
            
            return self.database
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def connect_async(self):
        """Create asynchronous MongoDB connection."""
        try:
            connection_string = self._build_connection_string()
            self.async_client = AsyncIOMotorClient(connection_string)
            
            # Test connection
            await self.async_client.admin.command('ping')
            
            self.async_database = self.async_client[self.config.database]
            self.logger.info(f"Connected to MongoDB (async): {self.config.host}:{self.config.port}")
            
            return self.async_database
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB (async): {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Check MongoDB connectivity and status."""
        try:
            if not self.client:
                self.connect()
            
            # Test connection
            ping_result = self.client.admin.command('ping')
            
            # Get server info
            server_info = self.client.server_info()
            
            # Get database stats
            db_stats = self.database.command('dbStats')
            
            return {
                "status": "healthy",
                "server_version": server_info.get('version'),
                "database": self.config.database,
                "collections": db_stats.get('collections', 0),
                "data_size": db_stats.get('dataSize', 0),
                "storage_size": db_stats.get('storageSize', 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check."""
        try:
            if not self.async_client:
                await self.connect_async()
            
            # Test connection
            await self.async_client.admin.command('ping')
            
            # Get server info
            server_info = await self.async_client.server_info()
            
            # Get database stats
            db_stats = await self.async_database.command('dbStats')
            
            return {
                "status": "healthy",
                "server_version": server_info.get('version'),
                "database": self.config.database,
                "collections": db_stats.get('collections', 0),
                "data_size": db_stats.get('dataSize', 0),
                "storage_size": db_stats.get('storageSize', 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def close(self):
        """Close connections."""
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()

# Global MongoDB manager instance
mongodb_manager = MongoDBManager()
```

### Document Models

```python
# prsm/models/mongodb_models.py
from datetime import datetime
from typing import Dict, List, Optional, Any
from bson import ObjectId
from pydantic import BaseModel, Field
from enum import Enum

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class QueryStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING = "pending"

class User(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    username: str = Field(..., max_length=50)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    profile: Optional[Dict[str, Any]] = {}
    preferences: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Conversation(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_archived: bool = False
    tags: List[str] = []
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Message(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    conversation_id: PyObjectId
    content: str
    role: MessageRole
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # AI-specific fields
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    
    # Additional context
    context: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class QueryLog(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[PyObjectId] = None
    conversation_id: Optional[PyObjectId] = None
    prompt: str
    response: Optional[str] = None
    status: QueryStatus
    error_message: Optional[str] = None
    
    # Performance metrics
    response_time: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    confidence: Optional[float] = None
    
    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class SystemMetrics(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    metric_name: str
    metric_value: float
    metric_type: str  # 'counter', 'gauge', 'histogram'
    labels: Optional[Dict[str, str]] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
```

### Repository Pattern

```python
# prsm/repositories/mongodb_repository.py
from typing import List, Optional, Dict, Any
from pymongo.collection import Collection
from pymongo import ASCENDING, DESCENDING, TEXT, IndexModel
from bson import ObjectId
from datetime import datetime, timedelta
from prsm.core.mongodb import mongodb_manager
from prsm.models.mongodb_models import User, Conversation, Message, QueryLog

class BaseMongoRepository:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None

    def get_collection(self) -> Collection:
        """Get MongoDB collection."""
        if not self.collection:
            if not mongodb_manager.database:
                mongodb_manager.connect()
            self.collection = mongodb_manager.database[self.collection_name]
        return self.collection

    def create_indexes(self):
        """Create collection indexes - override in subclasses."""
        pass

class UserRepository(BaseMongoRepository):
    def __init__(self):
        super().__init__("users")
        self.create_indexes()

    def create_indexes(self):
        """Create indexes for users collection."""
        collection = self.get_collection()
        indexes = [
            IndexModel([("username", ASCENDING)], unique=True),
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("is_active", ASCENDING)]),
            IndexModel([("username", TEXT), ("email", TEXT)])
        ]
        collection.create_indexes(indexes)

    def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user."""
        collection = self.get_collection()
        user = User(**user_data)
        result = collection.insert_one(user.dict(by_alias=True))
        user.id = result.inserted_id
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        collection = self.get_collection()
        doc = collection.find_one({"_id": ObjectId(user_id)})
        return User(**doc) if doc else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        collection = self.get_collection()
        doc = collection.find_one({"email": email})
        return User(**doc) if doc else None

    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user."""
        collection = self.get_collection()
        update_data["updated_at"] = datetime.utcnow()
        result = collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0

    def search_users(self, query: str, limit: int = 20) -> List[User]:
        """Search users by text."""
        collection = self.get_collection()
        docs = collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        return [User(**doc) for doc in docs]

class ConversationRepository(BaseMongoRepository):
    def __init__(self):
        super().__init__("conversations")
        self.create_indexes()

    def create_indexes(self):
        """Create indexes for conversations collection."""
        collection = self.get_collection()
        indexes = [
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
            IndexModel([("is_archived", ASCENDING)]),
            IndexModel([("tags", ASCENDING)]),
            IndexModel([("title", TEXT), ("summary", TEXT)])
        ]
        collection.create_indexes(indexes)

    def create_conversation(self, conversation_data: Dict[str, Any]) -> Conversation:
        """Create a new conversation."""
        collection = self.get_collection()
        conversation = Conversation(**conversation_data)
        result = collection.insert_one(conversation.dict(by_alias=True))
        conversation.id = result.inserted_id
        return conversation

    def get_user_conversations(
        self, 
        user_id: str, 
        limit: int = 50,
        include_archived: bool = False
    ) -> List[Conversation]:
        """Get conversations for a user."""
        collection = self.get_collection()
        
        filter_query = {"user_id": ObjectId(user_id)}
        if not include_archived:
            filter_query["is_archived"] = False
        
        docs = collection.find(filter_query).sort([
            ("updated_at", DESCENDING)
        ]).limit(limit)
        
        return [Conversation(**doc) for doc in docs]

    def search_conversations(
        self,
        user_id: str,
        search_term: str,
        limit: int = 20
    ) -> List[Conversation]:
        """Search conversations by title or summary."""
        collection = self.get_collection()
        
        docs = collection.find({
            "user_id": ObjectId(user_id),
            "$text": {"$search": search_term}
        }, {
            "score": {"$meta": "textScore"}
        }).sort([
            ("score", {"$meta": "textScore"})
        ]).limit(limit)
        
        return [Conversation(**doc) for doc in docs]

    def update_conversation(self, conversation_id: str, update_data: Dict[str, Any]) -> bool:
        """Update conversation."""
        collection = self.get_collection()
        update_data["updated_at"] = datetime.utcnow()
        result = collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0

    def archive_conversation(self, conversation_id: str) -> bool:
        """Archive a conversation."""
        return self.update_conversation(conversation_id, {"is_archived": True})

class MessageRepository(BaseMongoRepository):
    def __init__(self):
        super().__init__("messages")
        self.create_indexes()

    def create_indexes(self):
        """Create indexes for messages collection."""
        collection = self.get_collection()
        indexes = [
            IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)]),
            IndexModel([("role", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("content", TEXT)])
        ]
        collection.create_indexes(indexes)

    def add_message(self, message_data: Dict[str, Any]) -> Message:
        """Add a message to a conversation."""
        collection = self.get_collection()
        message = Message(**message_data)
        result = collection.insert_one(message.dict(by_alias=True))
        message.id = result.inserted_id
        
        # Update conversation timestamp
        conversation_repo = ConversationRepository()
        conversation_repo.update_conversation(
            str(message.conversation_id),
            {"updated_at": datetime.utcnow()}
        )
        
        return message

    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation."""
        collection = self.get_collection()
        
        docs = collection.find({
            "conversation_id": ObjectId(conversation_id)
        }).sort([
            ("created_at", ASCENDING)
        ]).skip(offset).limit(limit)
        
        return [Message(**doc) for doc in docs]

    def search_messages(
        self,
        conversation_id: str,
        search_term: str,
        limit: int = 20
    ) -> List[Message]:
        """Search messages in a conversation."""
        collection = self.get_collection()
        
        docs = collection.find({
            "conversation_id": ObjectId(conversation_id),
            "$text": {"$search": search_term}
        }, {
            "score": {"$meta": "textScore"}
        }).sort([
            ("score", {"$meta": "textScore"})
        ]).limit(limit)
        
        return [Message(**doc) for doc in docs]

class QueryLogRepository(BaseMongoRepository):
    def __init__(self):
        super().__init__("query_logs")
        self.create_indexes()

    def create_indexes(self):
        """Create indexes for query logs collection."""
        collection = self.get_collection()
        indexes = [
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("request_id", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)], expireAfterSeconds=7776000)  # 90 days TTL
        ]
        collection.create_indexes(indexes)

    def log_query(self, log_data: Dict[str, Any]) -> QueryLog:
        """Log a PRSM query."""
        collection = self.get_collection()
        query_log = QueryLog(**log_data)
        result = collection.insert_one(query_log.dict(by_alias=True))
        query_log.id = result.inserted_id
        return query_log

    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user analytics data."""
        collection = self.get_collection()
        
        since_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "user_id": ObjectId(user_id),
                    "created_at": {"$gte": since_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_queries": {"$sum": 1},
                    "successful_queries": {
                        "$sum": {"$cond": [{"$eq": ["$status", "success"]}, 1, 0]}
                    },
                    "avg_response_time": {"$avg": "$response_time"},
                    "total_tokens": {"$sum": "$tokens_used"},
                    "total_cost": {"$sum": "$cost"},
                    "avg_confidence": {"$avg": "$confidence"}
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        if not result:
            return {
                "total_queries": 0,
                "success_rate": 0,
                "avg_response_time": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "avg_confidence": 0
            }
        
        stats = result[0]
        success_rate = (stats["successful_queries"] / stats["total_queries"]) * 100 if stats["total_queries"] > 0 else 0
        
        return {
            "total_queries": stats["total_queries"],
            "success_rate": success_rate,
            "avg_response_time": stats["avg_response_time"] or 0,
            "total_tokens": stats["total_tokens"] or 0,
            "total_cost": stats["total_cost"] or 0,
            "avg_confidence": stats["avg_confidence"] or 0
        }

# Initialize repositories
user_repo = UserRepository()
conversation_repo = ConversationRepository()
message_repo = MessageRepository()
query_log_repo = QueryLogRepository()
```

### Database Service

```python
# prsm/services/mongodb_service.py
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, OperationFailure
from prsm.repositories.mongodb_repository import (
    user_repo, conversation_repo, message_repo, query_log_repo
)
from prsm.models.mongodb_models import User, Conversation, Message, QueryLog

class MongoDBService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_user(self, username: str, email: str, **metadata) -> Optional[User]:
        """Create a new user."""
        try:
            user_data = {
                "username": username,
                "email": email,
                "metadata": metadata
            }
            return user_repo.create_user(user_data)
        except DuplicateKeyError:
            self.logger.error(f"User with email {email} or username {username} already exists")
            return None
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            return user_repo.get_user_by_email(email)
        except Exception as e:
            self.logger.error(f"Failed to get user by email: {e}")
            return None

    def create_conversation(
        self, 
        user_id: str, 
        title: str = None,
        tags: List[str] = None,
        **metadata
    ) -> Optional[Conversation]:
        """Create a new conversation."""
        try:
            conversation_data = {
                "user_id": ObjectId(user_id),
                "title": title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                "tags": tags or [],
                "metadata": metadata
            }
            return conversation_repo.create_conversation(conversation_data)
        except Exception as e:
            self.logger.error(f"Failed to create conversation: {e}")
            return None

    def add_message(
        self,
        conversation_id: str,
        content: str,
        role: str,
        confidence: float = None,
        tokens_used: int = None,
        processing_time: float = None,
        **metadata
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        try:
            message_data = {
                "conversation_id": ObjectId(conversation_id),
                "content": content,
                "role": role,
                "confidence": confidence,
                "tokens_used": tokens_used,
                "processing_time": processing_time,
                "metadata": metadata
            }
            return message_repo.add_message(message_data)
        except Exception as e:
            self.logger.error(f"Failed to add message: {e}")
            return None

    def log_query(
        self,
        user_id: str,
        prompt: str,
        response: str = None,
        status: str = "success",
        error_message: str = None,
        response_time: float = None,
        tokens_used: int = None,
        cost: float = None,
        confidence: float = None,
        **context
    ) -> Optional[QueryLog]:
        """Log a PRSM query."""
        try:
            log_data = {
                "user_id": ObjectId(user_id) if user_id else None,
                "prompt": prompt,
                "response": response,
                "status": status,
                "error_message": error_message,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "cost": cost,
                "confidence": confidence,
                "ip_address": context.get('ip_address'),
                "user_agent": context.get('user_agent'),
                "request_id": context.get('request_id'),
                "metadata": context
            }
            return query_log_repo.log_query(log_data)
        except Exception as e:
            self.logger.error(f"Failed to log query: {e}")
            return None

    def get_user_conversations(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[Conversation]:
        """Get conversations for a user."""
        try:
            return conversation_repo.get_user_conversations(user_id, limit)
        except Exception as e:
            self.logger.error(f"Failed to get user conversations: {e}")
            return []

    def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: int = 100
    ) -> List[Message]:
        """Get conversation message history."""
        try:
            return message_repo.get_conversation_messages(conversation_id, limit)
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []

    def search_conversations(
        self,
        user_id: str,
        search_term: str,
        limit: int = 20
    ) -> List[Conversation]:
        """Search user's conversations."""
        try:
            return conversation_repo.search_conversations(user_id, search_term, limit)
        except Exception as e:
            self.logger.error(f"Failed to search conversations: {e}")
            return []

    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user analytics data."""
        try:
            return query_log_repo.get_user_analytics(user_id, days)
        except Exception as e:
            self.logger.error(f"Failed to get user analytics: {e}")
            return {}

    def update_conversation_summary(self, conversation_id: str) -> bool:
        """Update conversation summary based on messages."""
        try:
            # Get recent messages
            messages = message_repo.get_conversation_messages(conversation_id, limit=10)
            
            if not messages:
                return False
            
            # Generate summary (simplified - you might want to use AI for this)
            user_messages = [msg for msg in messages if msg.role == "user"]
            if user_messages:
                # Use first user message as summary
                summary = user_messages[0].content[:200] + "..." if len(user_messages[0].content) > 200 else user_messages[0].content
                
                return conversation_repo.update_conversation(
                    conversation_id,
                    {"summary": summary}
                )
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to update conversation summary: {e}")
            return False

# Initialize service
mongodb_service = MongoDBService()
```

## 🚀 Performance Optimization

### Aggregation Pipelines

```python
# prsm/analytics/mongodb_analytics.py
from typing import List, Dict, Any
from datetime import datetime, timedelta
from pymongo import MongoClient
from prsm.core.mongodb import mongodb_manager

class MongoDBAnalytics:
    def __init__(self):
        self.db = mongodb_manager.get_collection()

    def get_user_activity_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get user activity trends over time."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": since_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                        "status": "$status"
                    },
                    "count": {"$sum": 1},
                    "avg_response_time": {"$avg": "$response_time"},
                    "total_tokens": {"$sum": "$tokens_used"}
                }
            },
            {
                "$sort": {"_id.date": 1}
            }
        ]
        
        return list(mongodb_manager.database.query_logs.aggregate(pipeline))

    def get_top_users_by_usage(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by query volume."""
        pipeline = [
            {
                "$match": {
                    "user_id": {"$ne": None},
                    "created_at": {"$gte": datetime.utcnow() - timedelta(days=30)}
                }
            },
            {
                "$group": {
                    "_id": "$user_id",
                    "total_queries": {"$sum": 1},
                    "successful_queries": {
                        "$sum": {"$cond": [{"$eq": ["$status", "success"]}, 1, 0]}
                    },
                    "avg_response_time": {"$avg": "$response_time"},
                    "total_tokens": {"$sum": "$tokens_used"},
                    "total_cost": {"$sum": "$cost"}
                }
            },
            {
                "$lookup": {
                    "from": "users",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "user_info"
                }
            },
            {
                "$unwind": "$user_info"
            },
            {
                "$project": {
                    "username": "$user_info.username",
                    "email": "$user_info.email",
                    "total_queries": 1,
                    "success_rate": {
                        "$multiply": [
                            {"$divide": ["$successful_queries", "$total_queries"]},
                            100
                        ]
                    },
                    "avg_response_time": 1,
                    "total_tokens": 1,
                    "total_cost": 1
                }
            },
            {
                "$sort": {"total_queries": -1}
            },
            {
                "$limit": limit
            }
        ]
        
        return list(mongodb_manager.database.query_logs.aggregate(pipeline))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_queries": {"$sum": 1},
                    "successful_queries": {
                        "$sum": {"$cond": [{"$eq": ["$status", "success"]}, 1, 0]}
                    },
                    "avg_response_time": {"$avg": "$response_time"},
                    "p95_response_time": {"$push": "$response_time"},
                    "total_tokens": {"$sum": "$tokens_used"},
                    "avg_confidence": {"$avg": "$confidence"}
                }
            },
            {
                "$project": {
                    "total_queries": 1,
                    "success_rate": {
                        "$multiply": [
                            {"$divide": ["$successful_queries", "$total_queries"]},
                            100
                        ]
                    },
                    "avg_response_time": 1,
                    "p95_response_time": {
                        "$arrayElemAt": [
                            {"$slice": [
                                {"$sortArray": {
                                    "input": "$p95_response_time",
                                    "sortBy": 1
                                }},
                                {"$floor": {"$multiply": [{"$size": "$p95_response_time"}, 0.95]}},
                                1
                            ]},
                            0
                        ]
                    },
                    "total_tokens": 1,
                    "avg_confidence": 1
                }
            }
        ]
        
        result = list(mongodb_manager.database.query_logs.aggregate(pipeline))
        return result[0] if result else {}

# Initialize analytics
mongodb_analytics = MongoDBAnalytics()
```

### Indexing Strategy

```python
# prsm/optimization/mongodb_indexes.py
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT, HASHED
from prsm.core.mongodb import mongodb_manager

def create_performance_indexes():
    """Create performance-optimized indexes."""
    db = mongodb_manager.database
    
    # Users collection indexes
    users_indexes = [
        IndexModel([("username", ASCENDING)], unique=True),
        IndexModel([("email", ASCENDING)], unique=True),
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("is_active", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("username", TEXT), ("email", TEXT)]),
        IndexModel([("preferences.language", ASCENDING)]),
        IndexModel([("metadata.tier", ASCENDING)])
    ]
    db.users.create_indexes(users_indexes)
    
    # Conversations collection indexes
    conversations_indexes = [
        IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        IndexModel([("user_id", ASCENDING), ("is_archived", ASCENDING), ("updated_at", DESCENDING)]),
        IndexModel([("tags", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("title", TEXT), ("summary", TEXT)]),
        IndexModel([("user_id", HASHED)])  # For sharding
    ]
    db.conversations.create_indexes(conversations_indexes)
    
    # Messages collection indexes
    messages_indexes = [
        IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)]),
        IndexModel([("role", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("content", TEXT)]),
        IndexModel([("tokens_used", DESCENDING)]),
        IndexModel([("confidence", DESCENDING)]),
        IndexModel([("processing_time", DESCENDING)])
    ]
    db.messages.create_indexes(messages_indexes)
    
    # Query logs collection indexes
    query_logs_indexes = [
        IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("status", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("request_id", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("response_time", DESCENDING)]),
        IndexModel([("tokens_used", DESCENDING)]),
        IndexModel([("cost", DESCENDING)]),
        # TTL index for automatic cleanup
        IndexModel([("created_at", ASCENDING)], expireAfterSeconds=7776000)  # 90 days
    ]
    db.query_logs.create_indexes(query_logs_indexes)

def create_sharding_configuration():
    """Configure sharding for horizontal scaling."""
    # This would be run on a MongoDB cluster with sharding enabled
    commands = [
        # Enable sharding on database
        {"enableSharding": "prsm_production"},
        
        # Shard collections
        {
            "shardCollection": "prsm_production.users",
            "key": {"_id": "hashed"}
        },
        {
            "shardCollection": "prsm_production.conversations",
            "key": {"user_id": "hashed"}
        },
        {
            "shardCollection": "prsm_production.messages",
            "key": {"conversation_id": "hashed"}
        },
        {
            "shardCollection": "prsm_production.query_logs",
            "key": {"user_id": "hashed", "created_at": 1}
        }
    ]
    
    return commands

if __name__ == "__main__":
    create_performance_indexes()
    print("Performance indexes created successfully")
```

## 📊 Monitoring and Administration

### MongoDB Monitoring

```python
# prsm/monitoring/mongodb_monitor.py
from typing import Dict, List, Any
from datetime import datetime, timedelta
from pymongo.errors import OperationFailure
from prsm.core.mongodb import mongodb_manager

class MongoDBMonitor:
    def __init__(self):
        self.client = mongodb_manager.client
        self.db = mongodb_manager.database

    def get_server_status(self) -> Dict[str, Any]:
        """Get MongoDB server status."""
        try:
            status = self.db.command("serverStatus")
            return {
                "version": status["version"],
                "uptime": status["uptime"],
                "connections": status["connections"],
                "memory": status["mem"],
                "locks": status.get("locks", {}),
                "opcounters": status["opcounters"],
                "network": status["network"]
            }
        except OperationFailure as e:
            return {"error": str(e)}

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = self.db.command("dbStats")
            return {
                "collections": stats["collections"],
                "views": stats.get("views", 0),
                "objects": stats["objects"],
                "avgObjSize": stats["avgObjSize"],
                "dataSize": stats["dataSize"],
                "storageSize": stats["storageSize"],
                "indexes": stats["indexes"],
                "indexSize": stats["indexSize"]
            }
        except OperationFailure as e:
            return {"error": str(e)}

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        collections = self.db.list_collection_names()
        stats = {}
        
        for collection_name in collections:
            try:
                coll_stats = self.db.command("collStats", collection_name)
                stats[collection_name] = {
                    "count": coll_stats["count"],
                    "size": coll_stats["size"],
                    "storageSize": coll_stats["storageSize"],
                    "avgObjSize": coll_stats.get("avgObjSize", 0),
                    "indexSizes": coll_stats.get("indexSizes", {})
                }
            except OperationFailure:
                continue
        
        return stats

    def get_slow_operations(self, threshold_ms: int = 1000) -> List[Dict[str, Any]]:
        """Get currently running slow operations."""
        try:
            ops = self.db.command("currentOp", {"active": True})
            slow_ops = []
            
            for op in ops.get("inprog", []):
                if op.get("microsecs_running", 0) > threshold_ms * 1000:
                    slow_ops.append({
                        "opid": op.get("opid"),
                        "op": op.get("op"),
                        "ns": op.get("ns"),
                        "command": op.get("command", {}),
                        "duration_ms": op.get("microsecs_running", 0) / 1000,
                        "client": op.get("client")
                    })
            
            return slow_ops
        except OperationFailure as e:
            return [{"error": str(e)}]

    def get_index_usage_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get index usage statistics."""
        collections = self.db.list_collection_names()
        index_stats = {}
        
        for collection_name in collections:
            try:
                pipeline = [{"$indexStats": {}}]
                stats = list(self.db[collection_name].aggregate(pipeline))
                index_stats[collection_name] = stats
            except OperationFailure:
                continue
        
        return index_stats

    def check_replica_set_status(self) -> Dict[str, Any]:
        """Check replica set status (if applicable)."""
        try:
            status = self.db.command("replSetGetStatus")
            return {
                "set": status["set"],
                "members": len(status["members"]),
                "primary": next((m["name"] for m in status["members"] if m["stateStr"] == "PRIMARY"), None),
                "health": all(m["health"] == 1 for m in status["members"])
            }
        except OperationFailure:
            return {"error": "Not a replica set or insufficient privileges"}

    def analyze_query_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze query performance from logs."""
        since_date = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {"$match": {"created_at": {"$gte": since_date}}},
            {"$group": {
                "_id": None,
                "total_queries": {"$sum": 1},
                "avg_response_time": {"$avg": "$response_time"},
                "max_response_time": {"$max": "$response_time"},
                "min_response_time": {"$min": "$response_time"},
                "slow_queries": {
                    "$sum": {"$cond": [{"$gt": ["$response_time", 5.0]}, 1, 0]}
                }
            }}
        ]
        
        result = list(self.db.query_logs.aggregate(pipeline))
        return result[0] if result else {}

# Initialize monitor
mongodb_monitor = MongoDBMonitor()
```

## 🔒 Security and Best Practices

### Security Configuration

```python
# prsm/security/mongodb_security.py
from typing import Dict, List, Any
from pymongo.errors import OperationFailure
from prsm.core.mongodb import mongodb_manager

class MongoDBSecurity:
    def __init__(self):
        self.db = mongodb_manager.database
        self.admin_db = mongodb_manager.client.admin

    def create_user_roles(self):
        """Create custom roles for different access levels."""
        roles = [
            {
                "role": "prsmReadOnly",
                "privileges": [
                    {
                        "resource": {"db": "prsm_production", "collection": ""},
                        "actions": ["find", "listCollections", "listIndexes"]
                    }
                ],
                "roles": []
            },
            {
                "role": "prsmAppUser",
                "privileges": [
                    {
                        "resource": {"db": "prsm_production", "collection": ""},
                        "actions": [
                            "find", "insert", "update", "remove",
                            "createIndex", "listCollections", "listIndexes"
                        ]
                    }
                ],
                "roles": []
            },
            {
                "role": "prsmAnalytics",
                "privileges": [
                    {
                        "resource": {"db": "prsm_production", "collection": ""},
                        "actions": ["find", "listCollections", "listIndexes"]
                    },
                    {
                        "resource": {"db": "prsm_production", "collection": "query_logs"},
                        "actions": ["find", "aggregate"]
                    }
                ],
                "roles": []
            }
        ]
        
        for role in roles:
            try:
                self.admin_db.command("createRole", **role)
                print(f"Created role: {role['role']}")
            except OperationFailure as e:
                if "already exists" not in str(e):
                    print(f"Failed to create role {role['role']}: {e}")

    def enable_auditing(self):
        """Enable MongoDB auditing."""
        audit_config = {
            "auditLog": {
                "destination": "file",
                "format": "JSON",
                "path": "/var/log/mongodb/audit.log",
                "filter": {
                    "atype": {
                        "$in": [
                            "authenticate", "authCheck", "createUser",
                            "dropUser", "createRole", "dropRole",
                            "createCollection", "dropCollection"
                        ]
                    }
                }
            }
        }
        return audit_config

    def configure_ssl_tls(self):
        """SSL/TLS configuration."""
        ssl_config = {
            "net": {
                "ssl": {
                    "mode": "requireSSL",
                    "PEMKeyFile": "/etc/ssl/mongodb.pem",
                    "CAFile": "/etc/ssl/ca.pem",
                    "allowConnectionsWithoutCertificates": False,
                    "allowInvalidHostnames": False
                }
            }
        }
        return ssl_config

    def setup_field_level_encryption(self):
        """Configure field-level encryption for sensitive data."""
        # This is a simplified example - actual implementation would require
        # proper key management service integration
        encryption_schema = {
            "prsm_production.users": {
                "bsonType": "object",
                "properties": {
                    "email": {
                        "encrypt": {
                            "bsonType": "string",
                            "algorithm": "AEAD_AES_256_CBC_HMAC_SHA_512-Deterministic"
                        }
                    },
                    "profile.ssn": {
                        "encrypt": {
                            "bsonType": "string",
                            "algorithm": "AEAD_AES_256_CBC_HMAC_SHA_512-Random"
                        }
                    }
                }
            }
        }
        return encryption_schema

    def validate_security_settings(self) -> Dict[str, Any]:
        """Validate current security configuration."""
        try:
            # Check authentication
            users = self.admin_db.command("usersInfo")
            auth_enabled = len(users.get("users", [])) > 0
            
            # Check SSL/TLS
            server_status = self.admin_db.command("serverStatus")
            ssl_enabled = server_status.get("security", {}).get("SSLServerSubjectName") is not None
            
            # Check role-based access
            roles = self.admin_db.command("rolesInfo")
            custom_roles = [r for r in roles.get("roles", []) if r["role"].startswith("prsm")]
            
            return {
                "authentication_enabled": auth_enabled,
                "ssl_enabled": ssl_enabled,
                "custom_roles_count": len(custom_roles),
                "users_count": len(users.get("users", [])),
                "security_status": "configured" if auth_enabled and ssl_enabled else "needs_attention"
            }
            
        except OperationFailure as e:
            return {"error": str(e)}

# Initialize security manager
mongodb_security = MongoDBSecurity()
```

## 📋 Best Practices

### Configuration Management

```python
# config/mongodb.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class MongoDBConfig:
    """MongoDB configuration settings."""
    
    # Connection settings
    host: str = os.environ.get('MONGODB_HOST', 'localhost')
    port: int = int(os.environ.get('MONGODB_PORT', '27017'))
    database: str = os.environ.get('MONGODB_DATABASE', 'prsm_production')
    username: Optional[str] = os.environ.get('MONGODB_USERNAME')
    password: Optional[str] = os.environ.get('MONGODB_PASSWORD')
    auth_source: str = os.environ.get('MONGODB_AUTH_SOURCE', 'admin')
    
    # Replica set settings
    replica_set: Optional[str] = os.environ.get('MONGODB_REPLICA_SET')
    
    # Connection pool settings
    max_pool_size: int = int(os.environ.get('MONGODB_MAX_POOL_SIZE', '100'))
    min_pool_size: int = int(os.environ.get('MONGODB_MIN_POOL_SIZE', '10'))
    max_idle_time_ms: int = int(os.environ.get('MONGODB_MAX_IDLE_TIME_MS', '30000'))
    
    # SSL settings
    ssl: bool = os.environ.get('MONGODB_SSL', 'false').lower() == 'true'
    ssl_cert_file: Optional[str] = os.environ.get('MONGODB_SSL_CERT_FILE')
    ssl_key_file: Optional[str] = os.environ.get('MONGODB_SSL_KEY_FILE')
    ssl_ca_file: Optional[str] = os.environ.get('MONGODB_SSL_CA_FILE')
    
    # Performance settings
    read_preference: str = os.environ.get('MONGODB_READ_PREFERENCE', 'primary')
    write_concern: int = int(os.environ.get('MONGODB_WRITE_CONCERN', '1'))
    read_concern: str = os.environ.get('MONGODB_READ_CONCERN', 'local')
```

### Maintenance Scripts

```python
# scripts/mongodb_maintenance.py
#!/usr/bin/env python3
"""MongoDB maintenance scripts."""

import argparse
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
from prsm.core.mongodb import mongodb_manager

def cleanup_old_logs(days_to_keep: int = 90):
    """Clean up old query logs."""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    db = mongodb_manager.database
    result = db.query_logs.delete_many({
        "created_at": {"$lt": cutoff_date}
    })
    
    print(f"Deleted {result.deleted_count} old query logs")

def optimize_collections():
    """Run collection optimization operations."""
    db = mongodb_manager.database
    collections = db.list_collection_names()
    
    for collection_name in collections:
        print(f"Optimizing collection: {collection_name}")
        
        # Compact collection
        try:
            db.command("compact", collection_name)
            print(f"Compacted {collection_name}")
        except Exception as e:
            print(f"Failed to compact {collection_name}: {e}")
        
        # Reindex collection
        try:
            db[collection_name].reindex()
            print(f"Reindexed {collection_name}")
        except Exception as e:
            print(f"Failed to reindex {collection_name}: {e}")

def analyze_performance():
    """Analyze database performance."""
    db = mongodb_manager.database
    
    # Get slow operations
    slow_ops = db.command("currentOp", {"active": True, "microsecs_running": {"$gt": 1000000}})
    print(f"Slow operations: {len(slow_ops.get('inprog', []))}")
    
    # Get index usage
    for collection_name in db.list_collection_names():
        try:
            index_stats = list(db[collection_name].aggregate([{"$indexStats": {}}]))
            unused_indexes = [idx for idx in index_stats if idx.get("accesses", {}).get("ops", 0) == 0]
            
            if unused_indexes:
                print(f"Unused indexes in {collection_name}: {len(unused_indexes)}")
                for idx in unused_indexes:
                    print(f"  - {idx['name']}")
        except Exception as e:
            continue

def backup_database(backup_path: str):
    """Create database backup."""
    import subprocess
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{backup_path}/prsm_backup_{timestamp}"
    
    # Use mongodump for backup
    cmd = [
        "mongodump",
        "--host", f"{mongodb_manager.config.host}:{mongodb_manager.config.port}",
        "--db", mongodb_manager.config.database,
        "--out", backup_file
    ]
    
    if mongodb_manager.config.username:
        cmd.extend(["--username", mongodb_manager.config.username])
        cmd.extend(["--password", mongodb_manager.config.password])
        cmd.extend(["--authenticationDatabase", mongodb_manager.config.auth_source])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Backup created: {backup_file}")
        
        # Compress backup
        subprocess.run(["tar", "-czf", f"{backup_file}.tar.gz", "-C", backup_path, f"prsm_backup_{timestamp}"], check=True)
        subprocess.run(["rm", "-rf", backup_file], check=True)
        
        print(f"Backup compressed: {backup_file}.tar.gz")
        
    except subprocess.CalledProcessError as e:
        print(f"Backup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='MongoDB maintenance')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old logs')
    parser.add_argument('--optimize', action='store_true', help='Optimize collections')
    parser.add_argument('--analyze', action='store_true', help='Analyze performance')
    parser.add_argument('--backup', type=str, help='Create backup to specified path')
    parser.add_argument('--days', type=int, default=90, help='Days to keep logs')
    
    args = parser.parse_args()
    
    # Connect to MongoDB
    mongodb_manager.connect()
    
    if args.cleanup:
        cleanup_old_logs(args.days)
    
    if args.optimize:
        optimize_collections()
    
    if args.analyze:
        analyze_performance()
    
    if args.backup:
        backup_database(args.backup)

if __name__ == '__main__':
    main()
```

---

**Need help with MongoDB integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).