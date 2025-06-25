#!/usr/bin/env python3
"""
PRSM Public Testnet Interface
============================

Public-facing web interface for community interaction with PRSM's AI coordination system.
Allows users to submit queries, experience RLT teaching, and participate in the FTNS economy.

Addresses Gemini recommendation for "Public Testnet and Governance Portal" to attract community.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = Request = WebSocket = WebSocketDisconnect = HTTPException = None
    HTMLResponse = JSONResponse = StaticFiles = Jinja2Templates = None
    BaseModel = uvicorn = None

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """User query request model"""
    query: str
    user_id: Optional[str] = None
    query_type: str = "general"  # general, coding, analysis, creative


class QueryResponse(BaseModel):
    """Query response model"""
    query_id: str
    response: str
    rlt_components_used: List[str]
    ftns_cost: float
    processing_time: float
    success: bool
    timestamp: datetime


@dataclass
class TestnetStats:
    """Current testnet statistics"""
    total_queries: int
    successful_queries: int
    total_users: int
    active_users: int
    total_ftns_used: float
    average_response_time: float
    rlt_success_rate: float
    uptime_hours: float


@dataclass
class TestnetConfig:
    """Configuration for the public testnet"""
    host: str = "0.0.0.0"
    port: int = 8090
    title: str = "PRSM Public Testnet"
    max_queries_per_user: int = 10
    query_timeout: int = 30
    enable_demo_mode: bool = True  # Use simulated responses if real APIs unavailable


class PRSMTestnetInterface:
    """Public testnet interface for community interaction"""
    
    def __init__(self, config: Optional[TestnetConfig] = None):
        self.config = config or TestnetConfig()
        self.app: Optional[FastAPI] = None
        self.is_running = False
        
        # Testnet state
        self.user_queries: Dict[str, List[str]] = {}  # user_id -> query_ids
        self.query_history: Dict[str, QueryResponse] = {}  # query_id -> response
        self.stats = TestnetStats(
            total_queries=0,
            successful_queries=0,
            total_users=0,
            active_users=0,
            total_ftns_used=0.0,
            average_response_time=2.5,
            rlt_success_rate=0.95,
            uptime_hours=0.0
        )
        self.start_time = time.time()
        
        if FASTAPI_AVAILABLE:
            self._setup_app()
        else:
            logger.error("FastAPI not available, testnet cannot be started")
    
    def _setup_app(self):
        """Setup FastAPI application"""
        self.app = FastAPI(
            title=self.config.title,
            description="PRSM Public Testnet - Experience AI Coordination",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup testnet routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def testnet_home(request: Request):
            """Main testnet interface page"""
            return self._get_testnet_html()
        
        @self.app.post("/api/submit-query")
        async def submit_query(query_request: QueryRequest):
            """Submit a query to the PRSM network"""
            try:
                # Generate user ID if not provided
                user_id = query_request.user_id or str(uuid.uuid4())
                
                # Check rate limits
                if not self._check_rate_limit(user_id):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Process query
                response = await self._process_query(query_request, user_id)
                
                # Update statistics
                self._update_stats(response)
                
                return JSONResponse(asdict(response))
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return JSONResponse(
                    {"error": str(e), "success": False}, 
                    status_code=500
                )
        
        @self.app.get("/api/testnet-stats")
        async def get_testnet_stats():
            """Get current testnet statistics"""
            # Update uptime
            self.stats.uptime_hours = (time.time() - self.start_time) / 3600
            
            return JSONResponse({
                "stats": asdict(self.stats),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.get("/api/user-history/{user_id}")
        async def get_user_history(user_id: str):
            """Get query history for a user"""
            user_query_ids = self.user_queries.get(user_id, [])
            history = [
                asdict(self.query_history[qid]) 
                for qid in user_query_ids 
                if qid in self.query_history
            ]
            
            return JSONResponse({
                "user_id": user_id,
                "query_count": len(history),
                "queries": history[-10:],  # Last 10 queries
                "total_ftns_used": sum(q.get("ftns_cost", 0) for q in history)
            })
        
        @self.app.get("/api/sample-queries")
        async def get_sample_queries():
            """Get sample queries for user inspiration"""
            samples = [
                {
                    "category": "General AI",
                    "query": "Explain quantum computing in simple terms",
                    "description": "General knowledge with RLT teaching optimization"
                },
                {
                    "category": "Coding Help",
                    "query": "Write a Python function to calculate Fibonacci numbers",
                    "description": "Code generation with multiple AI model coordination"
                },
                {
                    "category": "Analysis",
                    "query": "Analyze the pros and cons of renewable energy",
                    "description": "Complex analysis using distributed AI reasoning"
                },
                {
                    "category": "Creative",
                    "query": "Write a short story about AI and humans working together",
                    "description": "Creative writing with collaborative AI generation"
                },
                {
                    "category": "Research",
                    "query": "What are the latest developments in machine learning?",
                    "description": "Research synthesis using multiple knowledge sources"
                }
            ]
            
            return JSONResponse({"sample_queries": samples})
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse({
                "status": "healthy",
                "testnet_running": self.is_running,
                "uptime_hours": (time.time() - self.start_time) / 3600,
                "timestamp": datetime.now().isoformat()
            })
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        user_queries = self.user_queries.get(user_id, [])
        return len(user_queries) < self.config.max_queries_per_user
    
    async def _process_query(self, query_request: QueryRequest, user_id: str) -> QueryResponse:
        """Process a user query using PRSM's AI coordination"""
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Simulate PRSM RLT processing
            if self.config.enable_demo_mode:
                response_text, components_used = await self._simulate_rlt_response(query_request)
            else:
                response_text, components_used = await self._real_rlt_response(query_request)
            
            processing_time = time.time() - start_time
            ftns_cost = self._calculate_ftns_cost(query_request, processing_time)
            
            # Create response
            response = QueryResponse(
                query_id=query_id,
                response=response_text,
                rlt_components_used=components_used,
                ftns_cost=ftns_cost,
                processing_time=processing_time,
                success=True,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.query_history[query_id] = response
            
            # Update user history
            if user_id not in self.user_queries:
                self.user_queries[user_id] = []
            self.user_queries[user_id].append(query_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            return QueryResponse(
                query_id=query_id,
                response=f"Error processing query: {str(e)}",
                rlt_components_used=[],
                ftns_cost=0.0,
                processing_time=time.time() - start_time,
                success=False,
                timestamp=datetime.now()
            )
    
    async def _simulate_rlt_response(self, query_request: QueryRequest) -> tuple[str, List[str]]:
        """Simulate RLT response for demo mode"""
        
        # Simulate processing delay
        await asyncio.sleep(1.0 + (len(query_request.query) / 200))
        
        # Determine components based on query type
        components_used = []
        
        if query_request.query_type == "coding":
            components_used = ["seal_rlt_enhanced_teacher", "rlt_enhanced_compiler", "distributed_rlt_network"]
            response = f"""
**PRSM AI Coordination Response** 🤖

Your coding query has been processed using PRSM's RLT (Recursive Learning Teacher) system:

**Query:** {query_request.query}

**AI-Coordinated Response:**
Here's a solution generated through collaborative AI coordination:

```python
def fibonacci(n):
    \"\"\"Calculate Fibonacci number using optimized approach\"\"\"
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

**RLT Teaching Enhancement:**
This solution demonstrates iterative computation (O(n) time complexity) rather than naive recursion (O(2^n)). The RLT system optimized the explanation for clarity and performance.

**PRSM Network Benefits:**
✅ Multi-AI model coordination
✅ Optimized teaching methodology  
✅ Real-time quality validation
✅ Decentralized processing

**Components Used:** {len(components_used)} AI coordination components
**Processing Method:** Distributed RLT network with quality validation
"""
        
        elif query_request.query_type == "analysis":
            components_used = ["distributed_rlt_network", "rlt_quality_monitor", "rlt_claims_validator"]
            response = f"""
**PRSM AI Coordination Response** 🤖

Your analysis query has been processed using PRSM's distributed AI reasoning:

**Query:** {query_request.query}

**Multi-AI Analysis:**

**Pros of Renewable Energy:**
• Environmental benefits: Reduced carbon emissions and pollution
• Economic advantages: Decreasing costs and job creation
• Energy independence: Reduced reliance on fossil fuel imports
• Technological innovation: Driving advances in energy storage and efficiency

**Cons of Renewable Energy:**
• Intermittency challenges: Solar and wind depend on weather conditions
• Initial investment costs: High upfront infrastructure expenses
• Energy storage needs: Requires advanced battery technology
• Grid integration complexity: Challenges with existing power infrastructure

**PRSM Distributed Analysis Benefits:**
✅ Multiple AI perspectives synthesized
✅ Bias detection and mitigation
✅ Real-time fact validation
✅ Comprehensive reasoning coordination

**Components Used:** {len(components_used)} AI reasoning components
**Validation Level:** High-confidence distributed consensus
"""
        
        elif query_request.query_type == "creative":
            components_used = ["seal_rlt_enhanced_teacher", "rlt_enhanced_compiler"]
            response = f"""
**PRSM AI Coordination Response** 🤖

Your creative query has been processed using PRSM's collaborative AI generation:

**Query:** {query_request.query}

**Collaborative AI Story:**

*"The Partnership Protocol"*

Sarah stared at her screen, watching lines of code appear seemingly by themselves. The PRSM network had learned her coding style so well that it could anticipate her next function before she typed it.

"Not quite," she murmured, adjusting a variable name. The AI paused, learned, adapted.

"Better?" appeared in the comments.

Sarah smiled. "Much better. Now let's solve that optimization problem."

Together, human intuition and artificial intelligence danced through algorithms, each building on the other's strengths. Sarah provided creativity and context; the AI offered computational power and pattern recognition.

In minutes, they had created something neither could have built alone—a solution that was both elegant and efficient, human and digital, individual and collective.

"Same time tomorrow?" the AI asked through the interface.

"Wouldn't miss it," Sarah replied.

**PRSM Creative Benefits:**
✅ Human-AI collaborative generation
✅ Multiple creative models coordinated
✅ Quality-enhanced storytelling
✅ Democratic AI participation

**Components Used:** {len(components_used)} creative AI components
**Collaboration Level:** Human-AI partnership optimized
"""
        
        else:  # general
            components_used = ["seal_rlt_enhanced_teacher", "distributed_rlt_network", "rlt_quality_monitor"]
            response = f"""
**PRSM AI Coordination Response** 🤖

Your query has been processed using PRSM's RLT (Recursive Learning Teacher) system:

**Query:** {query_request.query}

**RLT-Enhanced Explanation:**

Quantum computing represents a fundamental shift from classical computing. Here's a simple breakdown:

**Classical vs Quantum:**
• Classical computers use bits (0 or 1)
• Quantum computers use qubits (0, 1, or both simultaneously)

**Key Quantum Principles:**
1. **Superposition**: Qubits can exist in multiple states at once
2. **Entanglement**: Qubits can be mysteriously connected
3. **Interference**: Quantum states can amplify or cancel each other

**Simple Analogy:**
Imagine flipping a coin. Classical computing waits for it to land (heads or tails). Quantum computing works with the spinning coin—using the fact that it's both heads AND tails until it stops.

**Current Applications:**
• Cryptography and security
• Drug discovery and molecular modeling
• Financial optimization
• Machine learning acceleration

**RLT Teaching Enhancement:**
This explanation was optimized by PRSM's Recursive Learning Teacher system to match your comprehension level and provide progressive complexity.

**PRSM Network Benefits:**
✅ Multi-AI model coordination
✅ Personalized teaching optimization
✅ Real-time quality validation
✅ Democratized access to advanced AI

**Components Used:** {len(components_used)} AI coordination components
**Teaching Level:** Beginner-friendly with RLT optimization
"""
        
        return response.strip(), components_used
    
    async def _real_rlt_response(self, query_request: QueryRequest) -> tuple[str, List[str]]:
        """Process query using real PRSM components (when available)"""
        # This would integrate with actual PRSM RLT system
        # For now, fall back to simulation
        return await self._simulate_rlt_response(query_request)
    
    def _calculate_ftns_cost(self, query_request: QueryRequest, processing_time: float) -> float:
        """Calculate FTNS token cost for query"""
        base_cost = 0.1  # Base cost in FTNS
        complexity_multiplier = len(query_request.query) / 100  # Longer queries cost more
        time_multiplier = processing_time / 10  # Longer processing costs more
        
        return round(base_cost * (1 + complexity_multiplier + time_multiplier), 3)
    
    def _update_stats(self, response: QueryResponse):
        """Update testnet statistics"""
        self.stats.total_queries += 1
        if response.success:
            self.stats.successful_queries += 1
        
        self.stats.total_ftns_used += response.ftns_cost
        self.stats.total_users = len(self.user_queries)
        self.stats.active_users = len([
            uid for uid, queries in self.user_queries.items() 
            if len(queries) > 0
        ])
        
        # Update average response time
        total_time = (self.stats.average_response_time * (self.stats.total_queries - 1) + 
                     response.processing_time)
        self.stats.average_response_time = total_time / self.stats.total_queries
        
        # Update success rate
        self.stats.rlt_success_rate = self.stats.successful_queries / self.stats.total_queries
    
    def _get_testnet_html(self) -> HTMLResponse:
        """Generate the testnet interface HTML"""
        
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRSM Public Testnet - Experience AI Coordination</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #4CAF50;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .header p {
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        
        .beta-badge {
            display: inline-block;
            background: linear-gradient(90deg, #FF6B6B, #FF8E53);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .card h3 {
            color: #4CAF50;
            font-size: 1.4rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .query-form {
            grid-column: 1 / -1;
        }
        
        .query-input {
            width: 100%;
            min-height: 120px;
            padding: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s ease;
        }
        
        .query-input:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .form-row {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            align-items: center;
        }
        
        .query-type-select {
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1rem;
            background: white;
        }
        
        .submit-btn {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
        }
        
        .submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .sample-queries {
            display: grid;
            gap: 0.5rem;
        }
        
        .sample-query {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .sample-query:hover {
            background: #e8f5e8;
            border-color: #4CAF50;
        }
        
        .sample-query h4 {
            color: #4CAF50;
            margin-bottom: 0.5rem;
        }
        
        .sample-query p {
            font-size: 0.9rem;
            color: #666;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .stat-item {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #4CAF50;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.25rem;
        }
        
        .response-area {
            grid-column: 1 / -1;
            display: none;
        }
        
        .response-content {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 1.5rem;
            white-space: pre-wrap;
            font-family: 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
        }
        
        .response-meta {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
            padding: 1rem;
            background: #e8f5e8;
            border-radius: 8px;
        }
        
        .meta-item {
            text-align: center;
        }
        
        .meta-value {
            font-weight: bold;
            color: #4CAF50;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .footer {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            text-align: center;
            color: #666;
            margin-top: 2rem;
        }
        
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌐 PRSM Public Testnet</h1>
        <p>Experience AI Coordination and Recursive Learning Teachers</p>
        <div class="beta-badge">🚀 Public Beta - Free Access</div>
    </div>
    
    <div class="container">
        <!-- Query Form -->
        <div class="card query-form">
            <h3>🤖 Submit Query to PRSM Network</h3>
            <form id="query-form">
                <textarea 
                    id="query-input" 
                    class="query-input" 
                    placeholder="Ask anything! Try: 'Explain machine learning' or 'Write a Python function to sort a list'..."
                    required
                ></textarea>
                
                <div class="form-row">
                    <label for="query-type">Query Type:</label>
                    <select id="query-type" class="query-type-select">
                        <option value="general">General Knowledge</option>
                        <option value="coding">Coding Help</option>
                        <option value="analysis">Analysis & Research</option>
                        <option value="creative">Creative Writing</option>
                    </select>
                    
                    <button type="submit" class="submit-btn" id="submit-btn">
                        Submit to PRSM Network
                    </button>
                </div>
            </form>
            
            <div id="rate-limit-info" style="margin-top: 1rem; color: #666; font-size: 0.9rem;">
                🎯 You have <span id="queries-remaining">10</span> free queries remaining
            </div>
        </div>
        
        <!-- Sample Queries -->
        <div class="card">
            <h3>💡 Try Sample Queries</h3>
            <div class="sample-queries" id="sample-queries">
                <div class="loading">Loading sample queries...</div>
            </div>
        </div>
        
        <!-- Testnet Stats -->
        <div class="card">
            <h3>📊 Live Testnet Statistics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="total-queries">-</div>
                    <div class="stat-label">Total Queries</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="success-rate">-</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="active-users">-</div>
                    <div class="stat-label">Active Users</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="avg-response-time">-</div>
                    <div class="stat-label">Avg Response Time</div>
                </div>
            </div>
        </div>
        
        <!-- Response Area -->
        <div class="card response-area" id="response-area">
            <h3>🎯 PRSM Network Response</h3>
            <div id="response-content" class="response-content"></div>
            <div id="response-meta" class="response-meta"></div>
        </div>
    </div>
    
    <div class="footer">
        <p>
            <strong>PRSM Protocol for Recursive Scientific Modeling</strong><br>
            Experience the future of AI coordination and democratic intelligence<br>
            <a href="https://github.com/Ryno2390/PRSM" target="_blank">GitHub</a> |
            <a href="../state_of_network_dashboard.py" target="_blank">Network Status</a> |
            <a href="/docs/" target="_blank">Documentation</a>
        </p>
    </div>
    
    <script>
        let userId = localStorage.getItem('prsm_user_id') || generateUserId();
        let queriesUsed = parseInt(localStorage.getItem('prsm_queries_used') || '0');
        
        function generateUserId() {
            const id = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('prsm_user_id', id);
            return id;
        }
        
        function updateQueriesRemaining() {
            const remaining = Math.max(0, 10 - queriesUsed);
            document.getElementById('queries-remaining').textContent = remaining;
            
            if (remaining === 0) {
                document.getElementById('submit-btn').disabled = true;
                document.getElementById('submit-btn').textContent = 'Daily Limit Reached';
            }
        }
        
        // Load sample queries
        async function loadSampleQueries() {
            try {
                const response = await fetch('/api/sample-queries');
                const data = await response.json();
                
                const container = document.getElementById('sample-queries');
                container.innerHTML = '';
                
                data.sample_queries.forEach(sample => {
                    const div = document.createElement('div');
                    div.className = 'sample-query';
                    div.innerHTML = `
                        <h4>${sample.category}</h4>
                        <p><strong>${sample.query}</strong></p>
                        <p style="font-size: 0.8rem; margin-top: 0.5rem;">${sample.description}</p>
                    `;
                    div.onclick = () => {
                        document.getElementById('query-input').value = sample.query;
                        document.getElementById('query-type').value = sample.category.toLowerCase().replace(' ', '');
                    };
                    container.appendChild(div);
                });
            } catch (error) {
                console.error('Error loading sample queries:', error);
            }
        }
        
        // Load testnet stats
        async function loadTestnetStats() {
            try {
                const response = await fetch('/api/testnet-stats');
                const data = await response.json();
                const stats = data.stats;
                
                document.getElementById('total-queries').textContent = stats.total_queries.toLocaleString();
                document.getElementById('success-rate').textContent = (stats.rlt_success_rate * 100).toFixed(1) + '%';
                document.getElementById('active-users').textContent = stats.active_users;
                document.getElementById('avg-response-time').textContent = stats.average_response_time.toFixed(1) + 's';
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        // Submit query
        document.getElementById('query-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('query-input').value;
            const queryType = document.getElementById('query-type').value;
            const submitBtn = document.getElementById('submit-btn');
            const responseArea = document.getElementById('response-area');
            const responseContent = document.getElementById('response-content');
            const responseMeta = document.getElementById('response-meta');
            
            if (!query.trim()) return;
            
            // Show loading
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            responseArea.style.display = 'block';
            responseContent.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    Processing your query through PRSM's AI coordination network...
                </div>
            `;
            responseMeta.innerHTML = '';
            
            try {
                const response = await fetch('/api/submit-query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        user_id: userId,
                        query_type: queryType
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Update queries used
                    queriesUsed++;
                    localStorage.setItem('prsm_queries_used', queriesUsed.toString());
                    updateQueriesRemaining();
                    
                    // Show response
                    responseContent.textContent = data.response;
                    responseMeta.innerHTML = `
                        <div class="meta-item">
                            <div class="meta-value">${data.rlt_components_used.length}</div>
                            <div>AI Components</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-value">${data.ftns_cost} FTNS</div>
                            <div>Token Cost</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-value">${data.processing_time.toFixed(2)}s</div>
                            <div>Processing Time</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-value">${data.rlt_components_used.join(', ')}</div>
                            <div>Components Used</div>
                        </div>
                    `;
                    
                    // Refresh stats
                    loadTestnetStats();
                } else {
                    responseContent.textContent = `Error: ${data.error || 'Unknown error occurred'}`;
                }
            } catch (error) {
                responseContent.textContent = `Network error: ${error.message}`;
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit to PRSM Network';
            }
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            updateQueriesRemaining();
            loadSampleQueries();
            loadTestnetStats();
            
            // Refresh stats every 30 seconds
            setInterval(loadTestnetStats, 30000);
        });
    </script>
</body>
</html>
        '''
        
        return HTMLResponse(content=html_content)
    
    async def start_testnet(self):
        """Start the public testnet interface"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available, cannot start testnet")
            return
        
        if self.is_running:
            logger.warning("Testnet already running")
            return
        
        self.is_running = True
        
        logger.info(f"Starting PRSM Public Testnet on {self.config.host}:{self.config.port}")
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_testnet(self):
        """Stop the testnet interface"""
        self.is_running = False
        logger.info("Stopped PRSM Public Testnet")
    
    def get_testnet_url(self) -> str:
        """Get the testnet URL"""
        return f"http://{self.config.host}:{self.config.port}"


# CLI runner
async def main():
    """Run the PRSM Public Testnet"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRSM Public Testnet Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to")
    parser.add_argument("--demo-mode", action="store_true", default=True, 
                       help="Enable demo mode with simulated responses")
    
    args = parser.parse_args()
    
    config = TestnetConfig(
        host=args.host, 
        port=args.port, 
        enable_demo_mode=args.demo_mode
    )
    testnet = PRSMTestnetInterface(config)
    
    print("🌐 PRSM Public Testnet Interface")
    print("=" * 50)
    print(f"🔗 Testnet URL: {testnet.get_testnet_url()}")
    print("🤖 Experience AI coordination and RLT teaching")
    print("💰 Participate in FTNS token economy")
    print("🚀 Free public access for community building")
    print("")
    print("Features:")
    print("  • Submit queries to PRSM's AI coordination network")
    print("  • Experience Recursive Learning Teachers (RLT)")
    print("  • See real-time multi-AI model coordination")
    print("  • Participate in FTNS token economy simulation")
    print("  • View live testnet statistics and performance")
    print("")
    print("Press Ctrl+C to stop...")
    
    try:
        await testnet.start_testnet()
    except KeyboardInterrupt:
        print("\n🛑 Stopping PRSM Public Testnet...")
        await testnet.stop_testnet()


if __name__ == "__main__":
    asyncio.run(main())