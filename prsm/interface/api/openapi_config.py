"""
PRSM API OpenAPI Configuration
Enhanced API documentation and schema definitions
"""

from typing import Dict, Any, List, Optional
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI

# API Tags with descriptions for better organization
API_TAGS_METADATA = [
    {
        "name": "Authentication",
        "description": "User authentication, authorization, and session management. "
                      "Includes JWT tokens, API keys, role-based access control, and two-factor authentication.",
        "externalDocs": {
            "description": "Authentication Guide",
            "url": "https://docs.prsm.org/authentication"
        }
    },
    {
        "name": "Core API",
        "description": "Main PRSM functionality including NWTN orchestration, model management, "
                      "and system health monitoring.",
        "externalDocs": {
            "description": "Core API Guide",
            "url": "https://docs.prsm.org/core-api"
        }
    },
    {
        "name": "Marketplace",
        "description": "Universal resource marketplace with FTNS token economy. "
                      "Trade AI models, datasets, tools, and other resources with real token transactions.",
        "externalDocs": {
            "description": "Marketplace Guide",
            "url": "https://docs.prsm.org/marketplace"
        }
    },
    {
        "name": "Teams",
        "description": "Team collaboration and management system for distributed research projects.",
    },
    {
        "name": "Sessions",
        "description": "Research session tracking and management with real-time progress updates.",
    },
    {
        "name": "Tasks",
        "description": "Task execution, monitoring, and hierarchical decomposition system.",
    },
    {
        "name": "Payments",
        "description": "FTNS token transactions, balance management, and payment processing.",
        "externalDocs": {
            "description": "FTNS Token Guide",
            "url": "https://docs.prsm.org/ftns-tokens"
        }
    },
    {
        "name": "Governance",
        "description": "Decentralized governance system with proposals, voting, and community decision-making.",
    },
    {
        "name": "Security",
        "description": "Security monitoring, logging, cryptography, and access control systems.",
    },
    {
        "name": "Budget Management",
        "description": "Resource allocation, cost tracking, and budget management for FTNS tokens.",
    },
    {
        "name": "Recommendations",
        "description": "AI-powered recommendation system for models, resources, and collaborations.",
    },
    {
        "name": "Reputation",
        "description": "User and resource reputation system based on contribution quality and community feedback.",
    },
    {
        "name": "Distillation",
        "description": "Model distillation and knowledge transfer between AI systems.",
    },
    {
        "name": "Monitoring",
        "description": "System performance monitoring, metrics collection, and analytics.",
    },
    {
        "name": "Compliance",
        "description": "Regulatory compliance, audit trails, and policy enforcement.",
    },
    {
        "name": "Contributors",
        "description": "Contributor management, rewards, and recognition system.",
    },
    {
        "name": "Web3",
        "description": "Blockchain integration, smart contracts, and decentralized features.",
    },
    {
        "name": "CHRONOS",
        "description": "Time-series analysis, scheduling, and temporal data management.",
    },
    {
        "name": "Health",
        "description": "System health monitoring, diagnostics, and operational status.",
    },
    {
        "name": "UI",
        "description": "Frontend user interface endpoints and web application integration.",
    },
    {
        "name": "IPFS",
        "description": "Distributed storage system integration and file management.",
        "externalDocs": {
            "description": "IPFS Integration Guide",
            "url": "https://docs.prsm.org/ipfs"
        }
    },
    {
        "name": "WebSocket",
        "description": "Real-time communication, live updates, and streaming data endpoints.",
    }
]

# Security scheme definitions
SECURITY_SCHEMES = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT Bearer token authentication. Include the token in the Authorization header as 'Bearer <token>'"
    },
    "ApiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API Key authentication for service-to-service communication"
    },
    "RefreshToken": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT Refresh token for obtaining new access tokens"
    }
}

# Common response schemas
COMMON_RESPONSES = {
    "400": {
        "description": "Bad Request - Invalid input parameters",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Invalid request parameters"},
                        "error_code": {"type": "string", "example": "INVALID_INPUT"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                }
            }
        }
    },
    "401": {
        "description": "Unauthorized - Authentication required",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Authentication required"},
                        "error_code": {"type": "string", "example": "UNAUTHORIZED"}
                    }
                }
            }
        }
    },
    "403": {
        "description": "Forbidden - Insufficient permissions",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Insufficient permissions"},
                        "error_code": {"type": "string", "example": "FORBIDDEN"}
                    }
                }
            }
        }
    },
    "404": {
        "description": "Not Found - Resource does not exist",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Resource not found"},
                        "error_code": {"type": "string", "example": "NOT_FOUND"}
                    }
                }
            }
        }
    },
    "429": {
        "description": "Too Many Requests - Rate limit exceeded",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Rate limit exceeded"},
                        "error_code": {"type": "string", "example": "RATE_LIMITED"},
                        "retry_after": {"type": "integer", "example": 60}
                    }
                }
            }
        }
    },
    "500": {
        "description": "Internal Server Error - Unexpected server error",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Internal server error"},
                        "error_code": {"type": "string", "example": "INTERNAL_ERROR"},
                        "request_id": {"type": "string", "format": "uuid"}
                    }
                }
            }
        }
    }
}

# Custom OpenAPI schema
def custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate enhanced OpenAPI schema with comprehensive documentation
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="PRSM API",
        version="0.1.0",
        description="""
# Protocol for Recursive Scientific Modeling (PRSM) API

## Overview

PRSM is a comprehensive decentralized AI framework designed for scientific discovery and collaboration. 
This API provides access to advanced AI orchestration, distributed computing, and tokenized research economics.

## Key Features

### 🧠 NWTN (Neural Work Token Network)
- **AI Orchestration**: Advanced multi-agent reasoning and task decomposition
- **Model Integration**: Support for OpenAI, Anthropic, Hugging Face, and custom models
- **Distributed Computing**: P2P network for collaborative AI processing

### 💰 FTNS Token Economy
- **Real Token Transactions**: Native cryptocurrency integration with marketplace
- **Resource Trading**: Buy/sell AI models, datasets, compute time, and research tools
- **Stake-based Governance**: Community-driven platform development

### 🔬 Research Tools
- **Session Management**: Track research projects with real-time collaboration
- **Task Hierarchies**: Break down complex problems into manageable components
- **Quality Assurance**: Peer review and reputation systems

### 🏪 Universal Marketplace
- **9 Resource Types**: Models, datasets, tools, compute, storage, APIs, research papers, templates, plugins
- **Escrow System**: Secure transactions with automatic dispute resolution
- **Discovery Engine**: AI-powered recommendations and search

### 🔐 Enterprise Security
- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Role-based Access**: Granular permissions for different user types
- **Rate Limiting**: Configurable limits for different subscription tiers
- **Audit Trails**: Comprehensive logging for compliance and security

## Getting Started

### Authentication
All API endpoints require authentication. Obtain an access token by:

1. **Register/Login**: Use `/api/v1/auth/login` or `/api/v1/auth/register`
2. **Include Token**: Add `Authorization: Bearer <token>` header to requests
3. **Refresh Tokens**: Use refresh tokens to obtain new access tokens

### Rate Limits
- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: Custom limits

### WebSocket Real-time Features
- **Live Updates**: Connect to `/ws/{user_id}` for real-time notifications
- **Conversation Streaming**: Use `/ws/conversation/{user_id}/{conversation_id}` for AI responses

## Support

- **Documentation**: [https://docs.prsm.org](https://docs.prsm.org)
- **API Support**: [api-support@prsm.org](mailto:api-support@prsm.org)
- **Community**: [https://community.prsm.org](https://community.prsm.org)
- **GitHub**: [https://github.com/prsm-org/prsm](https://github.com/prsm-org/prsm)
        """,
        routes=app.routes,
        tags=API_TAGS_METADATA,
        servers=[
            {"url": "https://api.prsm.org", "description": "Production server"},
            {"url": "https://staging-api.prsm.org", "description": "Staging server"},
            {"url": "http://localhost:8000", "description": "Development server"}
        ],
        contact={
            "name": "PRSM API Support",
            "email": "api-support@prsm.org",
            "url": "https://developers.prsm.org"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES

    # Add common response schemas
    if "responses" not in openapi_schema["components"]:
        openapi_schema["components"]["responses"] = {}
    openapi_schema["components"]["responses"].update(COMMON_RESPONSES)

    # Add custom extensions
    openapi_schema["x-logo"] = {
        "url": "https://assets.prsm.org/logo.png",
        "altText": "PRSM Logo"
    }

    openapi_schema["x-api-id"] = "prsm-api-v1"
    openapi_schema["x-preferred-language"] = "python"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Example usage configurations for different environments
ENVIRONMENT_CONFIGS = {
    "development": {
        "servers": [{"url": "http://localhost:8000", "description": "Local development server"}],
        "security_requirements": [],  # Optional auth for development
    },
    "staging": {
        "servers": [{"url": "https://staging-api.prsm.org", "description": "Staging server"}],
        "security_requirements": [{"BearerAuth": []}],
    },
    "production": {
        "servers": [{"url": "https://api.prsm.org", "description": "Production server"}],
        "security_requirements": [{"BearerAuth": []}],
    }
}


def apply_environment_config(app: FastAPI, environment: str) -> None:
    """Apply environment-specific OpenAPI configuration"""
    if environment in ENVIRONMENT_CONFIGS:
        config = ENVIRONMENT_CONFIGS[environment]
        
        # Update servers
        if hasattr(app, 'servers'):
            app.servers = config["servers"]
        
        # Update security requirements for all routes
        if config.get("security_requirements"):
            for route in app.routes:
                if hasattr(route, 'dependencies'):
                    # Add security requirements to route
                    pass


# API Examples for different programming languages
API_EXAMPLES = {
    "python": {
        "install": "pip install httpx prsm-sdk",
        "auth": '''
import httpx

# Authenticate
response = httpx.post("https://api.prsm.org/api/v1/auth/login", 
                     json={"email": "user@example.com", "password": "password"})
token = response.json()["access_token"]

# Make authenticated request
headers = {"Authorization": f"Bearer {token}"}
response = httpx.get("https://api.prsm.org/api/v1/marketplace/resources", headers=headers)
        ''',
        "websocket": '''
import asyncio
import websockets
import json

async def connect_websocket():
    uri = "ws://localhost:8000/ws/user_123?token=your_jwt_token"
    async with websockets.connect(uri) as websocket:
        # Listen for real-time updates
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
        '''
    },
    "javascript": {
        "install": "npm install axios prsm-js-sdk",
        "auth": '''
const axios = require('axios');

// Authenticate
const authResponse = await axios.post('https://api.prsm.org/api/v1/auth/login', {
  email: 'user@example.com',
  password: 'password'
});
const token = authResponse.data.access_token;

// Make authenticated request
const response = await axios.get('https://api.prsm.org/api/v1/marketplace/resources', {
  headers: { Authorization: `Bearer ${token}` }
});
        ''',
        "websocket": '''
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws/user_123?token=your_jwt_token');

ws.on('message', (data) => {
  const message = JSON.parse(data);
  console.log('Received:', message);
});
        '''
    },
    "curl": {
        "auth": '''
# Authenticate
curl -X POST "https://api.prsm.org/api/v1/auth/login" \\
     -H "Content-Type: application/json" \\
     -d '{"email": "user@example.com", "password": "password"}'

# Make authenticated request  
curl -X GET "https://api.prsm.org/api/v1/marketplace/resources" \\
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
        '''
    }
}