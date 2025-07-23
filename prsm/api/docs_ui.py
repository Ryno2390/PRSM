"""
PRSM API Documentation UI Enhancements
Custom documentation interface with interactive features
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import os
from pathlib import Path

# Template configuration
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def create_enhanced_docs_ui(app: FastAPI) -> None:
    """Add enhanced documentation UI endpoints"""
    
    @app.get("/docs/playground", response_class=HTMLResponse, include_in_schema=False)
    async def api_playground(request: Request):
        """Interactive API playground with real-time testing"""
        return templates.TemplateResponse("api_playground.html", {
            "request": request,
            "api_base_url": str(request.base_url),
            "title": "PRSM API Playground"
        })
    
    @app.get("/docs/examples", response_class=HTMLResponse, include_in_schema=False)
    async def code_examples(request: Request):
        """Code examples in multiple programming languages"""
        return templates.TemplateResponse("code_examples.html", {
            "request": request,
            "title": "PRSM API Code Examples"
        })
    
    @app.get("/docs/guides", response_class=HTMLResponse, include_in_schema=False)
    async def integration_guides(request: Request):
        """Integration guides and tutorials"""
        return templates.TemplateResponse("integration_guides.html", {
            "request": request,
            "title": "PRSM API Integration Guides"
        })
    
    @app.get("/docs/sdk", response_class=HTMLResponse, include_in_schema=False)
    async def sdk_documentation(request: Request):
        """SDK documentation and downloads"""
        return templates.TemplateResponse("sdk_docs.html", {
            "request": request,
            "title": "PRSM SDKs and Libraries"
        })
    
    @app.get("/docs/postman", response_class=JSONResponse, include_in_schema=False)
    async def postman_collection():
        """Generate Postman collection for API testing"""
        
        # This would generate a complete Postman collection
        # For now, providing a sample structure
        collection = {
            "info": {
                "name": "PRSM API Collection",
                "description": "Complete PRSM API collection for testing",
                "version": "1.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{access_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "https://api.prsm.org",
                    "type": "string"
                },
                {
                    "key": "access_token",
                    "value": "",
                    "type": "string"
                }
            ],
            "item": [
                {
                    "name": "Authentication",
                    "item": [
                        {
                            "name": "Login",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": json.dumps({
                                        "email": "researcher@university.edu",
                                        "password": "secure_password_123"
                                    })
                                },
                                "url": {
                                    "raw": "{{base_url}}/api/v1/auth/login",
                                    "host": ["{{base_url}}"],
                                    "path": ["api", "v1", "auth", "login"]
                                }
                            },
                            "response": []
                        }
                    ]
                },
                {
                    "name": "Marketplace",
                    "item": [
                        {
                            "name": "Search Resources",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/api/v1/marketplace/resources?query=machine learning&resource_type=ai_model",
                                    "host": ["{{base_url}}"],
                                    "path": ["api", "v1", "marketplace", "resources"],
                                    "query": [
                                        {
                                            "key": "query",
                                            "value": "machine learning"
                                        },
                                        {
                                            "key": "resource_type",
                                            "value": "ai_model"
                                        }
                                    ]
                                }
                            },
                            "response": []
                        }
                    ]
                }
            ]
        }
        
        return JSONResponse(
            content=collection,
            headers={
                "Content-Disposition": "attachment; filename=PRSM_API_Collection.json"
            }
        )
    
    @app.get("/docs/openapi-spec", response_class=JSONResponse, include_in_schema=False)
    async def enhanced_openapi_spec():
        """Enhanced OpenAPI specification with examples"""
        
        # Get the base OpenAPI spec
        openapi_spec = app.openapi()
        
        # Add comprehensive examples
        enhanced_examples = {
            "/api/v1/auth/login": {
                "post": {
                    "examples": {
                        "successful_login": {
                            "summary": "Successful login",
                            "description": "Example of a successful login request",
                            "value": {
                                "email": "researcher@university.edu",
                                "password": "secure_password_123"
                            }
                        }
                    }
                }
            },
            "/api/v1/marketplace/resources": {
                "get": {
                    "examples": {
                        "search_ai_models": {
                            "summary": "Search for AI models",
                            "description": "Search for machine learning models under $200",
                            "parameters": {
                                "query": "machine learning",
                                "resource_type": "ai_model",
                                "max_price": 200.0,
                                "min_rating": 4.0
                            }
                        }
                    }
                }
            }
        }
        
        # Merge examples into the spec
        for path, methods in enhanced_examples.items():
            if path in openapi_spec.get("paths", {}):
                for method, example_data in methods.items():
                    if method in openapi_spec["paths"][path]:
                        openapi_spec["paths"][path][method].update(example_data)
        
        return JSONResponse(content=openapi_spec)
    
    @app.get("/docs/status", response_class=JSONResponse, include_in_schema=False)
    async def api_status():
        """API status and health information for documentation"""
        return JSONResponse(content={
            "api_version": "0.1.0",
            "documentation_version": "1.0.0",
            "last_updated": "2024-01-15T10:00:00Z",
            "features": {
                "interactive_docs": True,
                "code_examples": True,
                "sdk_support": True,
                "postman_collection": True,
                "websocket_docs": True
            },
            "supported_formats": ["JSON", "WebSocket", "Server-Sent Events"],
            "authentication_methods": ["JWT Bearer", "API Key", "OAuth2"],
            "rate_limits": {
                "free_tier": "100 requests/hour",
                "pro_tier": "1000 requests/hour",
                "enterprise_tier": "Custom limits"
            }
        })


def generate_curl_examples() -> Dict[str, Dict[str, str]]:
    """Generate cURL examples for all major endpoints"""
    
    return {
        "authentication": {
            "register": '''curl -X POST "https://api.prsm.org/api/v1/auth/register" \\
  -H "Content-Type: application/json" \\
  -d '{
    "email": "researcher@university.edu",
    "password": "secure_password_123",
    "full_name": "Dr. Jane Smith",
    "organization": "University of Science",
    "role": "researcher"
  }'
''',
            "login": '''curl -X POST "https://api.prsm.org/api/v1/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{
    "email": "researcher@university.edu",
    "password": "secure_password_123"
  }'
'''
        },
        "marketplace": {
            "search": '''curl -X GET "https://api.prsm.org/api/v1/marketplace/resources?query=machine%20learning&resource_type=ai_model&max_price=200" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
''',
            "purchase": '''curl -X POST "https://api.prsm.org/api/v1/marketplace/resources/res_123456789/purchase" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"payment_method": "ftns_balance"}'
'''
        },
        "sessions": {
            "create": '''curl -X POST "https://api.prsm.org/api/v1/sessions" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "title": "Climate Change Analysis",
    "description": "ML analysis of climate data",
    "collaborators": ["user_id_1", "user_id_2"],
    "ftns_budget": 500.0,
    "tags": ["climate", "machine-learning"]
  }'
''',
            "list": '''curl -X GET "https://api.prsm.org/api/v1/sessions" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
'''
        },
        "tasks": {
            "create": '''curl -X POST "https://api.prsm.org/api/v1/tasks" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "title": "Data Preprocessing",
    "description": "Clean and prepare dataset for analysis",
    "session_id": "session_123",
    "estimated_cost": 25.0,
    "priority": 3
  }'
'''
        },
        "tokens": {
            "balance": '''curl -X GET "https://api.prsm.org/api/v1/users/YOUR_USER_ID/balance" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
''',
            "transfer": '''curl -X POST "https://api.prsm.org/api/v1/transactions/transfer" \\
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "to_user_id": "recipient_user_id",
    "amount": 100.0,
    "description": "Payment for consultation"
  }'
'''
        }
    }


def generate_python_examples() -> Dict[str, str]:
    """Generate Python SDK examples"""
    
    return {
        "installation": '''# Install the PRSM Python SDK
pip install prsm-sdk

# Or install from source
pip install git+https://github.com/prsm-org/prsm-python-sdk.git
''',
        "basic_usage": '''from prsm_sdk import PRSMClient
import asyncio

# Initialize client
client = PRSMClient(
    api_key="your_api_key_here",
    base_url="https://api.prsm.org"
)

# Authenticate
await client.auth.login("researcher@university.edu", "password")

# Search marketplace
resources = await client.marketplace.search(
    query="machine learning",
    resource_type="ai_model",
    max_price=200.0,
    min_rating=4.0
)

print(f"Found {len(resources)} AI models")
for resource in resources[:3]:
    print(f"- {resource.title}: {resource.price} FTNS")
''',
        "session_management": '''# Create a research session
session = await client.sessions.create(
    title="Climate Change Analysis",
    description="Machine learning analysis of climate data",
    ftns_budget=500.0,
    collaborators=["collaborator_user_id"]
)

# Create tasks within the session
task = await client.tasks.create(
    session_id=session.id,
    title="Data Preprocessing",
    description="Clean and prepare climate dataset",
    estimated_cost=25.0,
    priority=3
)

# Monitor task progress
while task.status != "completed":
    await asyncio.sleep(5)
    task = await client.tasks.get(task.id)
    print(f"Task status: {task.status}")
''',
        "websocket_integration": '''import asyncio
import websockets
import json

async def handle_real_time_updates():
    # Connect to WebSocket
    uri = f"wss://api.prsm.org/ws/{user_id}?token={access_token}"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to PRSM real-time updates")
        
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "session_update":
                print(f"Session {data['session_id']} updated: {data['status']}")
            elif data["type"] == "ftns_transaction":
                print(f"FTNS transaction: {data['amount']} tokens")
            elif data["type"] == "task_completed":
                print(f"Task {data['task_id']} completed!")

# Run the WebSocket handler
asyncio.run(handle_real_time_updates())
'''
    }


def generate_javascript_examples() -> Dict[str, str]:
    """Generate JavaScript SDK examples"""
    
    return {
        "installation": '''// Install via npm
npm install @prsm/js-sdk

// Or via yarn
yarn add @prsm/js-sdk

// Browser CDN (for quick testing)
<script src="https://cdn.prsm.org/js-sdk/v1.0.0/prsm-sdk.min.js"></script>
''',
        "basic_usage": '''import { PRSMClient } from '@prsm/js-sdk';

// Initialize client
const client = new PRSMClient({
  apiKey: 'your_api_key_here',
  baseURL: 'https://api.prsm.org'
});

// Authenticate
await client.auth.login('researcher@university.edu', 'password');

// Search marketplace
const resources = await client.marketplace.search({
  query: 'machine learning',
  resourceType: 'ai_model',
  maxPrice: 200.0,
  minRating: 4.0
});

console.log(`Found ${resources.length} AI models`);
resources.slice(0, 3).forEach(resource => {
  console.log(`- ${resource.title}: ${resource.price} FTNS`);
});
''',
        "react_integration": '''import React, { useState, useEffect } from 'react';
import { PRSMClient } from '@prsm/js-sdk';

const PRSMDashboard = () => {
  const [client, setClient] = useState(null);
  const [balance, setBalance] = useState(null);
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    const initializeClient = async () => {
      const prsmClient = new PRSMClient({
        apiKey: process.env.REACT_APP_PRSM_API_KEY,
        baseURL: 'https://api.prsm.org'
      });
      
      await prsmClient.auth.login(
        process.env.REACT_APP_USER_EMAIL,
        process.env.REACT_APP_USER_PASSWORD
      );
      
      setClient(prsmClient);
      
      // Load user data
      const userBalance = await prsmClient.tokens.getBalance();
      setBalance(userBalance);
      
      const userSessions = await prsmClient.sessions.list();
      setSessions(userSessions);
    };

    initializeClient();
  }, []);

  return (
    <div>
      <h1>PRSM Dashboard</h1>
      {balance && (
        <div>
          <h2>FTNS Balance: {balance.available_balance}</h2>
        </div>
      )}
      <div>
        <h2>Research Sessions ({sessions.length})</h2>
        {sessions.map(session => (
          <div key={session.id}>
            <h3>{session.title}</h3>
            <p>Status: {session.status}</p>
            <p>Budget: {session.ftns_budget} FTNS</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PRSMDashboard;
''',
        "websocket_integration": '''// WebSocket connection for real-time updates
const ws = new WebSocket('wss://api.prsm.org/ws/user_123?token=your_jwt_token');

ws.onopen = () => {
  console.log('Connected to PRSM real-time updates');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'session_update':
      console.log(`Session ${data.session_id} updated:`, data.status);
      break;
    case 'ftns_transaction':
      console.log(`FTNS transaction: ${data.amount} tokens`);
      break;
    case 'task_completed':
      console.log(`Task ${data.task_id} completed!`);
      break;
  }
};

ws.onclose = () => {
  console.log('Disconnected from PRSM updates');
  // Implement reconnection logic
  setTimeout(() => {
    connectWebSocket();
  }, 5000);
};
'''
    }


# Export functions for use in other modules
__all__ = [
    'create_enhanced_docs_ui',
    'generate_curl_examples',
    'generate_python_examples', 
    'generate_javascript_examples'
]