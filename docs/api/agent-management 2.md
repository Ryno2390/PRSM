# Agent Management API

**Create, configure, and orchestrate AI agents across distributed networks**

## 🎯 Overview

The Agent Management API allows you to create sophisticated AI agents, configure their capabilities, and orchestrate complex multi-agent workflows. Agents in PRSM can leverage multiple LLM providers, execute code, perform research, and collaborate with other agents.

## 📋 Base Information

- **Base URL**: `https://api.prsm.network/v1/agents`
- **Authentication**: Bearer token or FTNS token required
- **Rate Limits**: 1000 requests/hour (Free), 10,000/hour (Pro)
- **WebSocket**: `wss://ws.prsm.network/agents` (real-time updates)

## 🤖 Agent Types

| Type | Description | Use Cases |
|------|-------------|-----------|
| **Researcher** | Scientific analysis and discovery | Literature review, hypothesis generation |
| **Coder** | Code generation and debugging | Software development, automation |
| **Analyst** | Data analysis and visualization | Business intelligence, research insights |
| **Orchestrator** | Multi-agent coordination | Complex workflows, task delegation |
| **Specialist** | Domain-specific expertise | Legal, medical, financial analysis |

## 🚀 Quick Start

### Create Your First Agent

```python
import prsm

client = prsm.Client(api_key="your_api_key")

# Create a research agent
agent = client.agents.create(
    name="genomics_researcher",
    type="researcher",
    model_provider="openai",
    model_name="gpt-4",
    capabilities=[
        "literature_search",
        "data_analysis", 
        "hypothesis_generation"
    ],
    specialized_knowledge="genomics,bioinformatics,protein_folding"
)

print(f"Agent created: {agent.id}")
```

## 📚 API Reference

### Create Agent

**`POST /v1/agents`**

Creates a new AI agent with specified capabilities and configuration.

#### Request Body

```json
{
  "name": "string",
  "description": "string",
  "type": "researcher|coder|analyst|orchestrator|specialist",
  "model_provider": "openai|anthropic|huggingface|local",
  "model_name": "string",
  "capabilities": ["string"],
  "specialized_knowledge": "string",
  "configuration": {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "context_window": 8192
  },
  "tools": ["code_execution", "web_search", "file_upload"],
  "memory_type": "conversation|persistent|shared",
  "collaboration_mode": "solo|team|swarm"
}
```

#### Response

```json
{
  "id": "agent_abc123",
  "name": "genomics_researcher",
  "type": "researcher",
  "status": "active",
  "model_provider": "openai",
  "model_name": "gpt-4",
  "capabilities": [
    "literature_search",
    "data_analysis",
    "hypothesis_generation"
  ],
  "created_at": "2024-12-21T16:59:23Z",
  "updated_at": "2024-12-21T16:59:23Z",
  "configuration": {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "context_window": 8192
  },
  "performance_metrics": {
    "success_rate": 0.0,
    "avg_response_time": 0.0,
    "total_executions": 0
  }
}
```

#### Code Examples

<details>
<summary><strong>Python Example</strong></summary>

```python
import prsm

client = prsm.Client(api_key="your_api_key")

# Create a sophisticated coding agent
coding_agent = client.agents.create(
    name="senior_python_developer",
    description="Expert Python developer with focus on AI/ML",
    type="coder",
    model_provider="openai",
    model_name="gpt-4",
    capabilities=[
        "code_generation",
        "debugging",
        "performance_optimization",
        "testing",
        "documentation"
    ],
    specialized_knowledge="python,machine_learning,data_science,web_development",
    configuration={
        "temperature": 0.3,  # Lower temperature for more consistent code
        "max_tokens": 4096,
        "top_p": 0.95
    },
    tools=["code_execution", "file_upload", "git_integration"],
    memory_type="persistent",
    collaboration_mode="team"
)

print(f"Coding agent created: {coding_agent.id}")
print(f"Capabilities: {coding_agent.capabilities}")
```
</details>

<details>
<summary><strong>JavaScript Example</strong></summary>

```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({ apiKey: 'your_api_key' });

// Create a data analysis agent
const analystAgent = await client.agents.create({
    name: "financial_analyst",
    description: "Expert in financial data analysis and modeling",
    type: "analyst",
    modelProvider: "anthropic",
    modelName: "claude-3-sonnet",
    capabilities: [
        "data_analysis",
        "statistical_modeling",
        "visualization",
        "report_generation"
    ],
    specializedKnowledge: "finance,economics,statistics,risk_management",
    configuration: {
        temperature: 0.5,
        maxTokens: 3072,
        topP: 0.9
    },
    tools: ["code_execution", "data_upload", "chart_generation"],
    memoryType: "conversation",
    collaborationMode: "solo"
});

console.log(`Analyst agent created: ${analystAgent.id}`);
```
</details>

<details>
<summary><strong>cURL Example</strong></summary>

```bash
curl -X POST https://api.prsm.network/v1/agents \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "medical_researcher",
    "description": "Specialized in medical literature review and clinical research",
    "type": "researcher",
    "model_provider": "openai",
    "model_name": "gpt-4",
    "capabilities": [
      "literature_search",
      "clinical_analysis",
      "protocol_design",
      "statistical_analysis"
    ],
    "specialized_knowledge": "medicine,clinical_trials,pharmacology,epidemiology",
    "configuration": {
      "temperature": 0.6,
      "max_tokens": 2048,
      "top_p": 0.9
    },
    "tools": ["web_search", "file_upload", "data_analysis"],
    "memory_type": "persistent",
    "collaboration_mode": "team"
  }'
```
</details>

### List Agents

**`GET /v1/agents`**

Retrieves a list of all agents in your workspace.

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Number of agents to return (default: 20, max: 100) |
| `offset` | integer | Number of agents to skip (default: 0) |
| `type` | string | Filter by agent type |
| `status` | string | Filter by status (active, inactive, error) |
| `sort_by` | string | Sort field (created_at, name, type) |
| `sort_order` | string | Sort order (asc, desc) |

#### Response

```json
{
  "agents": [
    {
      "id": "agent_abc123",
      "name": "genomics_researcher",
      "type": "researcher",
      "status": "active",
      "model_provider": "openai",
      "created_at": "2024-12-21T16:59:23Z",
      "performance_metrics": {
        "success_rate": 0.95,
        "avg_response_time": 1.2,
        "total_executions": 150
      }
    }
  ],
  "total": 25,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

### Get Agent

**`GET /v1/agents/{agent_id}`**

Retrieves detailed information about a specific agent.

#### Response

```json
{
  "id": "agent_abc123",
  "name": "genomics_researcher",
  "description": "Expert in genomics and bioinformatics research",
  "type": "researcher",
  "status": "active",
  "model_provider": "openai",
  "model_name": "gpt-4",
  "capabilities": [
    "literature_search",
    "data_analysis",
    "hypothesis_generation"
  ],
  "specialized_knowledge": "genomics,bioinformatics,protein_folding",
  "configuration": {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "context_window": 8192
  },
  "tools": ["code_execution", "web_search", "file_upload"],
  "memory_type": "persistent",
  "collaboration_mode": "team",
  "created_at": "2024-12-21T16:59:23Z",
  "updated_at": "2024-12-21T17:30:45Z",
  "performance_metrics": {
    "success_rate": 0.95,
    "avg_response_time": 1.2,
    "total_executions": 150,
    "error_rate": 0.02,
    "cost_per_execution": 0.15
  },
  "recent_activity": [
    {
      "timestamp": "2024-12-21T17:25:00Z",
      "action": "task_execution",
      "task_id": "task_xyz789",
      "duration": 2.1,
      "success": true
    }
  ]
}
```

### Update Agent

**`PATCH /v1/agents/{agent_id}`**

Updates an existing agent's configuration or capabilities.

#### Request Body

```json
{
  "name": "string",
  "description": "string",
  "capabilities": ["string"],
  "configuration": {
    "temperature": 0.8,
    "max_tokens": 3072
  },
  "tools": ["string"],
  "status": "active|inactive"
}
```

### Delete Agent

**`DELETE /v1/agents/{agent_id}`**

Permanently deletes an agent and all associated data.

#### Response

```json
{
  "message": "Agent successfully deleted",
  "agent_id": "agent_abc123",
  "deleted_at": "2024-12-21T17:45:00Z"
}
```

## ⚡ Agent Execution

### Execute Task

**`POST /v1/agents/{agent_id}/execute`**

Executes a task using the specified agent.

#### Request Body

```json
{
  "prompt": "string",
  "context": {
    "domain": "string",
    "priority": "low|medium|high",
    "deadline": "2024-12-22T10:00:00Z"
  },
  "data": {
    "files": ["file_id_1", "file_id_2"],
    "urls": ["https://example.com/data.csv"],
    "parameters": {}
  },
  "execution_options": {
    "timeout": 300,
    "max_iterations": 5,
    "stream_response": true,
    "save_to_memory": true
  }
}
```

#### Response

```json
{
  "execution_id": "exec_def456",
  "status": "completed",
  "output": {
    "text": "Based on my analysis of the genomic data...",
    "code": "import pandas as pd\n# Analysis code here",
    "files": ["result_chart.png", "analysis_report.pdf"],
    "data": {
      "confidence_score": 0.92,
      "key_findings": ["Finding 1", "Finding 2"]
    }
  },
  "metadata": {
    "execution_time": 4.2,
    "tokens_used": 1250,
    "cost": 0.18,
    "model_calls": 3
  },
  "started_at": "2024-12-21T17:50:00Z",
  "completed_at": "2024-12-21T17:50:04Z"
}
```

#### Code Examples

<details>
<summary><strong>Python: Complex Research Task</strong></summary>

```python
# Execute a complex research task
execution = client.agents.execute(
    agent_id="agent_abc123",
    prompt="""
    Analyze the latest research on CRISPR-Cas9 gene editing efficiency.
    Focus on:
    1. Recent improvements in targeting accuracy
    2. Off-target effects and mitigation strategies
    3. Clinical trial outcomes from 2024
    
    Provide a comprehensive summary with citations.
    """,
    context={
        "domain": "biomedical_research",
        "priority": "high",
        "deadline": "2024-12-22T18:00:00Z"
    },
    data={
        "urls": [
            "https://pubmed.ncbi.nlm.nih.gov/search?term=CRISPR+2024",
            "https://clinicaltrials.gov/search?term=CRISPR"
        ]
    },
    execution_options={
        "timeout": 600,  # 10 minutes
        "max_iterations": 10,
        "stream_response": True,
        "save_to_memory": True
    }
)

print(f"Execution ID: {execution.execution_id}")
print(f"Status: {execution.status}")

# Stream real-time updates
for update in execution.stream():
    print(f"Progress: {update.progress}")
    if update.partial_output:
        print(f"Partial result: {update.partial_output}")

# Get final results
final_result = execution.get_result()
print(f"Final analysis: {final_result.output.text}")
print(f"Generated files: {final_result.output.files}")
print(f"Confidence: {final_result.output.data.confidence_score}")
```
</details>

<details>
<summary><strong>JavaScript: Code Generation Task</strong></summary>

```javascript
// Execute code generation task
const execution = await client.agents.execute({
    agentId: "agent_def456",
    prompt: `
        Create a Python class for analyzing financial time series data.
        Requirements:
        - Load data from CSV files
        - Calculate moving averages (SMA, EMA)
        - Detect trend patterns
        - Generate buy/sell signals
        - Include proper error handling and documentation
    `,
    context: {
        domain: "financial_analysis",
        priority: "medium"
    },
    executionOptions: {
        timeout: 180,
        maxIterations: 3,
        streamResponse: false,
        saveToMemory: true
    }
});

console.log(`Generated code:\n${execution.output.code}`);
console.log(`Documentation: ${execution.output.text}`);

// Test the generated code
if (execution.output.code) {
    const testExecution = await client.agents.execute({
        agentId: "agent_def456",
        prompt: "Test the generated financial analysis class with sample data",
        data: {
            code: execution.output.code,
            testData: "sample_stock_data.csv"
        }
    });
    
    console.log(`Test results: ${testExecution.output.text}`);
}
```
</details>

### Stream Execution

**`GET /v1/agents/{agent_id}/executions/{execution_id}/stream`**

Stream real-time updates for a running execution.

#### WebSocket Connection

```javascript
const ws = new WebSocket('wss://ws.prsm.network/agents/agent_abc123/executions/exec_def456/stream');

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log(`Progress: ${update.progress}%`);
    console.log(`Status: ${update.status}`);
    
    if (update.partial_output) {
        console.log(`Partial result: ${update.partial_output}`);
    }
    
    if (update.status === 'completed') {
        console.log('Execution completed!');
        console.log(`Final result: ${update.final_output}`);
    }
};
```

## 🤝 Multi-Agent Collaboration

### Create Agent Team

**`POST /v1/agent-teams`**

Creates a team of agents that can collaborate on complex tasks.

#### Request Body

```json
{
  "name": "research_team",
  "description": "Collaborative team for scientific research",
  "agents": [
    {
      "agent_id": "agent_abc123",
      "role": "lead_researcher",
      "permissions": ["task_assignment", "result_review"]
    },
    {
      "agent_id": "agent_def456", 
      "role": "data_analyst",
      "permissions": ["data_processing", "visualization"]
    },
    {
      "agent_id": "agent_ghi789",
      "role": "technical_writer",
      "permissions": ["documentation", "report_generation"]
    }
  ],
  "collaboration_settings": {
    "communication_style": "structured",
    "decision_making": "consensus",
    "task_distribution": "automatic",
    "quality_assurance": "peer_review"
  }
}
```

### Execute Team Task

**`POST /v1/agent-teams/{team_id}/execute`**

Executes a complex task using the entire agent team.

#### Code Example

```python
# Create a research team
team = client.agent_teams.create(
    name="covid_research_team",
    description="Multidisciplinary team for COVID-19 research analysis",
    agents=[
        {
            "agent_id": "virologist_agent",
            "role": "domain_expert",
            "permissions": ["scientific_analysis", "hypothesis_validation"]
        },
        {
            "agent_id": "statistician_agent", 
            "role": "data_analyst",
            "permissions": ["statistical_analysis", "model_validation"]
        },
        {
            "agent_id": "writer_agent",
            "role": "communicator",
            "permissions": ["report_writing", "visualization"]
        }
    ],
    collaboration_settings={
        "communication_style": "structured",
        "decision_making": "expert_weighted",
        "task_distribution": "skill_based"
    }
)

# Execute team research project
team_execution = team.execute(
    prompt="""
    Analyze the effectiveness of different COVID-19 vaccination strategies
    across various demographic groups. Produce a comprehensive report with:
    
    1. Statistical analysis of vaccination efficacy data
    2. Identification of factors affecting vaccine effectiveness
    3. Recommendations for targeted vaccination strategies
    4. Clear visualizations and executive summary
    """,
    data={
        "datasets": [
            "vaccination_efficacy_data.csv",
            "demographic_data.csv",
            "clinical_trial_results.json"
        ]
    },
    collaboration_options={
        "parallel_processing": True,
        "cross_validation": True,
        "iterative_refinement": True
    }
)

# Monitor team collaboration
for update in team_execution.stream():
    print(f"Team progress: {update.overall_progress}%")
    for agent_update in update.agent_updates:
        print(f"  {agent_update.role}: {agent_update.current_task}")
    
    if update.collaboration_events:
        for event in update.collaboration_events:
            print(f"  Collaboration: {event.type} - {event.description}")
```

## 📊 Performance Monitoring

### Get Agent Metrics

**`GET /v1/agents/{agent_id}/metrics`**

Retrieves detailed performance metrics for an agent.

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | string | Start date for metrics (ISO 8601) |
| `end_date` | string | End date for metrics (ISO 8601) |
| `granularity` | string | Metrics granularity (hour, day, week) |
| `metrics` | array | Specific metrics to retrieve |

#### Response

```json
{
  "agent_id": "agent_abc123",
  "period": {
    "start": "2024-12-14T00:00:00Z",
    "end": "2024-12-21T23:59:59Z"
  },
  "metrics": {
    "execution_count": 247,
    "success_rate": 0.954,
    "avg_response_time": 1.34,
    "p95_response_time": 3.2,
    "error_rate": 0.024,
    "total_cost": 45.67,
    "avg_cost_per_execution": 0.185,
    "tokens_used": 125040,
    "unique_users": 18
  },
  "time_series": [
    {
      "timestamp": "2024-12-21T16:00:00Z",
      "executions": 12,
      "success_rate": 1.0,
      "avg_response_time": 1.1,
      "cost": 2.34
    }
  ],
  "error_breakdown": {
    "timeout_errors": 3,
    "model_errors": 2,
    "resource_errors": 1,
    "user_errors": 0
  }
}
```

## 🔧 Advanced Configuration

### Custom Model Integration

```python
# Create agent with custom model
custom_agent = client.agents.create(
    name="specialized_coder",
    type="coder",
    model_provider="custom",
    model_configuration={
        "endpoint": "https://your-model.example.com/v1/chat/completions",
        "api_key": "your_custom_model_key",
        "model_name": "your-fine-tuned-model",
        "headers": {
            "X-Custom-Header": "value"
        }
    },
    capabilities=["code_generation", "debugging"],
    specialized_knowledge="domain_specific_coding"
)
```

### Agent Memory Management

```python
# Configure persistent memory
agent = client.agents.create(
    name="memory_agent",
    memory_configuration={
        "type": "persistent",
        "storage": "vector_database",
        "retention_policy": {
            "max_conversations": 1000,
            "max_age_days": 90,
            "auto_summarization": True
        },
        "indexing": {
            "semantic_search": True,
            "keyword_search": True,
            "temporal_indexing": True
        }
    }
)

# Query agent memory
memory_results = agent.memory.search(
    query="previous discussions about protein folding",
    limit=10,
    similarity_threshold=0.8
)

for result in memory_results:
    print(f"Relevance: {result.score}")
    print(f"Content: {result.content}")
    print(f"Timestamp: {result.timestamp}")
```

## 🚨 Error Handling

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `AGENT_NOT_FOUND` | Agent ID doesn't exist | Verify agent ID |
| `INSUFFICIENT_PERMISSIONS` | Lack required permissions | Check API key permissions |
| `MODEL_UNAVAILABLE` | Requested model not accessible | Use alternative model |
| `EXECUTION_TIMEOUT` | Task exceeded time limit | Increase timeout or simplify task |
| `RESOURCE_EXHAUSTED` | Insufficient resources | Upgrade plan or retry later |
| `INVALID_CONFIGURATION` | Invalid agent configuration | Review configuration parameters |

### Error Handling Example

```python
try:
    agent = client.agents.create(
        name="test_agent",
        type="researcher",
        model_provider="openai",
        model_name="gpt-4"
    )
    
    result = agent.execute(
        prompt="Analyze this complex dataset",
        data={"file": "large_dataset.csv"},
        execution_options={"timeout": 300}
    )
    
except prsm.exceptions.AgentCreationError as e:
    print(f"Failed to create agent: {e.message}")
    print(f"Suggestions: {e.suggestions}")
    
except prsm.exceptions.ExecutionTimeoutError as e:
    print(f"Execution timed out after {e.timeout} seconds")
    print("Consider increasing timeout or breaking down the task")
    
except prsm.exceptions.ResourceExhaustedError as e:
    print(f"Insufficient resources: {e.resource_type}")
    print(f"Current usage: {e.current_usage}")
    print(f"Limit: {e.limit}")
    
except prsm.exceptions.PRSMAPIError as e:
    print(f"API Error: {e.error_code}")
    print(f"Message: {e.message}")
    print(f"Request ID: {e.request_id}")
```

## 🎯 Best Practices

### 1. Agent Design

- **Single Responsibility**: Create agents with focused, specific capabilities
- **Clear Naming**: Use descriptive names that indicate purpose and domain
- **Appropriate Models**: Match model capabilities to task requirements
- **Resource Limits**: Set reasonable timeouts and token limits

### 2. Performance Optimization

- **Batch Operations**: Group similar tasks for efficient processing
- **Caching**: Enable memory for frequently accessed information
- **Model Selection**: Use smaller models for simple tasks
- **Parallel Execution**: Leverage multi-agent teams for complex workflows

### 3. Cost Management

- **Monitor Usage**: Track token consumption and costs regularly
- **Right-size Models**: Don't use GPT-4 for simple classification tasks
- **Implement Caching**: Reduce redundant API calls
- **Use P2P Network**: Leverage community resources for cost savings

### 4. Security

- **Validate Inputs**: Sanitize all user inputs before processing
- **Limit Permissions**: Grant minimum required permissions
- **Monitor Activity**: Track agent executions for anomalies
- **Secure Storage**: Encrypt sensitive data in agent memory

---

**Next Steps:**
- [Model Inference API](./model-inference.md) - Execute inference across providers
- [P2P Network API](./p2p-network.md) - Join distributed AI networks
- [Interactive Examples](./examples/) - Hands-on tutorials and code samples