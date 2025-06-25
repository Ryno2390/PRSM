# Workflow Management API

Design and execute complex scientific workflows with PRSM's distributed workflow orchestration system.

## 🎯 Overview

The Workflow Management API enables creation, execution, and monitoring of complex multi-step scientific workflows that can span multiple models, data sources, and computational resources across the PRSM network.

## 📋 Base URL

```
https://api.prsm.ai/v1/workflows
```

## 🔐 Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/workflows
```

## 🚀 Quick Start

### Create a Simple Workflow

```python
import prsm

client = prsm.Client(api_key="your-api-key")

# Define a simple workflow
workflow = client.workflows.create(
    name="Text Analysis Pipeline",
    description="Analyze text sentiment and extract entities",
    steps=[
        {
            "id": "sentiment",
            "type": "inference",
            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "input": "${workflow.input.text}"
        },
        {
            "id": "entities",
            "type": "inference", 
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "input": "${workflow.input.text}",
            "depends_on": ["sentiment"]
        },
        {
            "id": "summary",
            "type": "inference",
            "model": "gpt-3.5-turbo",
            "input": "Summarize: Sentiment=${sentiment.result}, Entities=${entities.result}",
            "depends_on": ["sentiment", "entities"]
        }
    ]
)

print(f"Created workflow: {workflow.id}")
```

### Execute Workflow

```python
# Execute the workflow
execution = client.workflows.execute(
    workflow_id=workflow.id,
    inputs={
        "text": "I love using PRSM for AI research! It's revolutionary."
    }
)

print(f"Execution ID: {execution.id}")
```

## 📊 Endpoints

### POST /workflows
Create a new workflow definition.

**Request Body:**
```json
{
  "name": "Scientific Paper Analysis",
  "description": "Comprehensive analysis of scientific papers",
  "version": "1.0",
  "steps": [
    {
      "id": "extract_text",
      "type": "data_processing",
      "function": "pdf_to_text",
      "input": "${workflow.input.pdf_url}",
      "timeout": 300
    },
    {
      "id": "summarize",
      "type": "inference",
      "model": "gpt-4",
      "input": "Summarize this paper: ${extract_text.result}",
      "depends_on": ["extract_text"],
      "parameters": {
        "max_tokens": 500,
        "temperature": 0.3
      }
    },
    {
      "id": "extract_methods",
      "type": "inference",
      "model": "scibert",
      "input": "Extract methodology from: ${extract_text.result}",
      "depends_on": ["extract_text"]
    },
    {
      "id": "final_analysis",
      "type": "aggregation",
      "function": "combine_results",
      "inputs": {
        "summary": "${summarize.result}",
        "methods": "${extract_methods.result}"
      },
      "depends_on": ["summarize", "extract_methods"]
    }
  ],
  "error_handling": {
    "retry_failed_steps": true,
    "max_retries": 3,
    "fallback_models": {
      "gpt-4": "gpt-3.5-turbo",
      "scibert": "bert-base-uncased"
    }
  },
  "resource_requirements": {
    "cpu_cores": 4,
    "ram_gb": 8,
    "gpu_required": false
  }
}
```

**Response:**
```json
{
  "id": "wf_abc123",
  "name": "Scientific Paper Analysis",
  "status": "created",
  "version": "1.0",
  "created_at": "2024-01-15T10:30:00Z",
  "step_count": 4,
  "estimated_cost": 0.25,
  "estimated_duration": 180
}
```

### POST /workflows/{workflow_id}/execute
Execute a workflow with specific inputs.

**Request Body:**
```json
{
  "inputs": {
    "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf"
  },
  "execution_config": {
    "priority": "high",
    "max_parallel_steps": 3,
    "timeout": 1800,
    "resource_allocation": "auto"
  },
  "notification_config": {
    "on_completion": ["email", "webhook"],
    "on_failure": ["email"],
    "webhook_url": "https://your-app.com/webhook"
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_xyz789",
  "workflow_id": "wf_abc123",
  "status": "running",
  "started_at": "2024-01-15T10:35:00Z",
  "current_step": "extract_text",
  "progress": 0.25,
  "estimated_completion": "2024-01-15T10:38:00Z"
}
```

### GET /workflows/{workflow_id}/executions/{execution_id}
Get execution status and results.

**Response:**
```json
{
  "execution_id": "exec_xyz789",
  "workflow_id": "wf_abc123",
  "status": "completed",
  "started_at": "2024-01-15T10:35:00Z",
  "completed_at": "2024-01-15T10:37:45Z",
  "duration_seconds": 165,
  "steps": [
    {
      "id": "extract_text",
      "status": "completed",
      "started_at": "2024-01-15T10:35:00Z",
      "completed_at": "2024-01-15T10:35:30Z",
      "result": "This paper presents a novel approach...",
      "cost": 0.05
    },
    {
      "id": "summarize",
      "status": "completed", 
      "started_at": "2024-01-15T10:35:31Z",
      "completed_at": "2024-01-15T10:36:15Z",
      "result": "The paper introduces a new method for...",
      "cost": 0.12
    }
  ],
  "final_result": {
    "summary": "The paper introduces a new method for...",
    "methodology": "The authors used a quantitative approach...",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
  },
  "total_cost": 0.23,
  "resource_usage": {
    "cpu_hours": 0.5,
    "ram_gb_hours": 2.1,
    "gpu_hours": 0.0
  }
}
```

### GET /workflows
List all workflows.

**Query Parameters:**
- `status`: Filter by status (created, running, completed, failed)
- `limit`: Maximum number of workflows to return
- `offset`: Number of workflows to skip
- `sort`: Sort order (created_at, name, status)

### PUT /workflows/{workflow_id}
Update a workflow definition.

### DELETE /workflows/{workflow_id}
Delete a workflow.

## 🔧 Workflow Components

### Step Types

**Inference Steps:**
```python
{
    "id": "analyze_sentiment",
    "type": "inference",
    "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "input": "${previous_step.result}",
    "parameters": {
        "temperature": 0.1,
        "max_tokens": 100
    }
}
```

**Data Processing Steps:**
```python
{
    "id": "preprocess_data",
    "type": "data_processing",
    "function": "clean_text",
    "input": "${workflow.input.raw_text}",
    "parameters": {
        "remove_stopwords": true,
        "lowercase": true
    }
}
```

**Aggregation Steps:**
```python
{
    "id": "combine_results",
    "type": "aggregation",
    "function": "merge_json",
    "inputs": {
        "sentiment": "${sentiment_analysis.result}",
        "entities": "${entity_extraction.result}"
    }
}
```

**Conditional Steps:**
```python
{
    "id": "conditional_analysis",
    "type": "conditional",
    "condition": "${sentiment.result.label} == 'POSITIVE'",
    "if_true": {
        "type": "inference",
        "model": "gpt-3.5-turbo",
        "input": "Analyze positive sentiment: ${workflow.input.text}"
    },
    "if_false": {
        "type": "inference", 
        "model": "gpt-3.5-turbo",
        "input": "Analyze negative sentiment: ${workflow.input.text}"
    }
}
```

**Loop Steps:**
```python
{
    "id": "process_documents",
    "type": "loop",
    "iterate_over": "${workflow.input.document_list}",
    "step": {
        "type": "inference",
        "model": "gpt-3.5-turbo",
        "input": "Analyze document: ${loop.current_item}"
    },
    "max_iterations": 100,
    "parallel": true
}
```

## 🎛️ Advanced Workflow Features

### Parallel Execution

```python
workflow = client.workflows.create(
    name="Parallel Analysis",
    steps=[
        {
            "id": "sentiment_analysis",
            "type": "inference",
            "model": "sentiment-model",
            "input": "${workflow.input.text}",
            "parallel_group": "analysis"
        },
        {
            "id": "entity_extraction", 
            "type": "inference",
            "model": "ner-model",
            "input": "${workflow.input.text}",
            "parallel_group": "analysis"
        },
        {
            "id": "topic_modeling",
            "type": "inference", 
            "model": "topic-model",
            "input": "${workflow.input.text}",
            "parallel_group": "analysis"
        },
        {
            "id": "combine_results",
            "type": "aggregation",
            "depends_on": ["sentiment_analysis", "entity_extraction", "topic_modeling"]
        }
    ]
)
```

### Dynamic Workflows

```python
# Create workflow with dynamic step generation
dynamic_workflow = client.workflows.create_dynamic(
    name="Dynamic Document Processing",
    generator_function="generate_steps_for_documents",
    input_schema={
        "documents": {"type": "array", "items": {"type": "string"}},
        "analysis_types": {"type": "array", "items": {"type": "string"}}
    }
)
```

### Workflow Templates

```python
# Create reusable workflow template
template = client.workflows.create_template(
    name="Text Analysis Template",
    parameters=[
        {"name": "input_model", "type": "string", "default": "gpt-3.5-turbo"},
        {"name": "max_tokens", "type": "integer", "default": 150},
        {"name": "analysis_depth", "type": "string", "enum": ["basic", "detailed"]}
    ],
    steps_template=[
        {
            "id": "analyze",
            "type": "inference",
            "model": "${template.input_model}",
            "parameters": {
                "max_tokens": "${template.max_tokens}"
            }
        }
    ]
)
```

## 📊 Monitoring and Observability

### Real-time Monitoring

```python
# Monitor workflow execution in real-time
monitor = client.workflows.monitor(execution_id="exec_xyz789")

@monitor.on_step_completed
def step_completed(step_id, result):
    print(f"Step {step_id} completed: {result}")

@monitor.on_step_failed  
def step_failed(step_id, error):
    print(f"Step {step_id} failed: {error}")

@monitor.on_workflow_completed
def workflow_completed(final_result):
    print(f"Workflow completed: {final_result}")
```

### Performance Analytics

```python
# Get workflow performance analytics
analytics = client.workflows.analytics(
    workflow_id="wf_abc123",
    timeframe="last_30_days"
)

print(f"Average execution time: {analytics.avg_duration_seconds}s")
print(f"Success rate: {analytics.success_rate}%")
print(f"Average cost: ${analytics.avg_cost}")
```

### Resource Usage Tracking

```python
# Track resource usage across workflows
usage = client.workflows.resource_usage(
    timeframe="last_week",
    group_by="workflow_id"
)

for workflow_id, metrics in usage.items():
    print(f"Workflow {workflow_id}:")
    print(f"  CPU hours: {metrics.cpu_hours}")
    print(f"  RAM GB-hours: {metrics.ram_gb_hours}")
    print(f"  Total cost: ${metrics.total_cost}")
```

## 🔄 Error Handling and Recovery

### Automatic Retry Configuration

```python
workflow = client.workflows.create(
    name="Resilient Workflow",
    steps=[...],
    error_handling={
        "retry_policy": {
            "max_retries": 3,
            "retry_delay_seconds": 30,
            "backoff_multiplier": 2.0,
            "retry_on_errors": ["timeout", "rate_limit", "temporary_failure"]
        },
        "fallback_strategy": {
            "enabled": True,
            "fallback_models": {
                "gpt-4": "gpt-3.5-turbo",
                "claude-3-opus": "claude-3-sonnet"
            }
        }
    }
)
```

### Custom Error Handlers

```python
# Define custom error handling
def custom_error_handler(step_id, error_type, error_message):
    if error_type == "model_unavailable":
        return {"action": "use_fallback", "fallback_model": "gpt-3.5-turbo"}
    elif error_type == "rate_limit":
        return {"action": "wait_and_retry", "wait_seconds": 60}
    else:
        return {"action": "fail_workflow"}

workflow.set_error_handler(custom_error_handler)
```

## 🔐 Security and Access Control

### Workflow Permissions

```python
# Set workflow access permissions
permissions = client.workflows.set_permissions(
    workflow_id="wf_abc123",
    permissions={
        "read": ["user1", "user2", "team:researchers"],
        "execute": ["user1", "team:researchers"],
        "modify": ["user1"],
        "delete": ["user1"]
    }
)
```

### Secure Input Handling

```python
# Handle sensitive inputs securely
execution = client.workflows.execute(
    workflow_id="wf_abc123",
    inputs={
        "api_key": {"type": "secret", "value": "sk-..."},
        "user_data": {"type": "encrypted", "value": "encrypted_data"}
    },
    security_config={
        "encrypt_intermediate_results": True,
        "audit_logging": True,
        "secure_deletion": True
    }
)
```

## 🧪 Testing and Validation

### Workflow Testing

```python
# Test workflow with mock data
test_result = client.workflows.test(
    workflow_id="wf_abc123",
    test_inputs={
        "text": "This is a test input for validation"
    },
    mock_models={
        "gpt-3.5-turbo": {
            "response": "Mocked response for testing",
            "latency_ms": 100
        }
    }
)
```

### Validation Framework

```python
# Validate workflow definition
validation = client.workflows.validate(
    workflow_definition=workflow_dict,
    checks=[
        "syntax_validation",
        "dependency_check", 
        "resource_requirements",
        "cost_estimation"
    ]
)

if not validation.is_valid:
    print("Validation errors:", validation.errors)
```

## 📈 Optimization

### Performance Optimization

```python
# Optimize workflow for performance
optimized = client.workflows.optimize(
    workflow_id="wf_abc123",
    optimization_goals=["speed", "cost", "accuracy"],
    constraints={
        "max_cost": 1.00,
        "max_duration_seconds": 300
    }
)
```

### Resource Allocation

```python
# Configure optimal resource allocation
resource_config = {
    "cpu_allocation_strategy": "balanced",
    "memory_allocation_strategy": "conservative", 
    "gpu_allocation_strategy": "on_demand",
    "network_optimization": True
}

execution = client.workflows.execute(
    workflow_id="wf_abc123",
    inputs=inputs,
    resource_config=resource_config
)
```

## 📚 Integration Examples

### Scientific Research Pipeline

```python
# Complete scientific research workflow
research_workflow = client.workflows.create(
    name="Literature Review and Analysis",
    steps=[
        {
            "id": "search_papers",
            "type": "data_collection",
            "function": "academic_search",
            "input": "${workflow.input.research_query}"
        },
        {
            "id": "filter_papers",
            "type": "data_processing", 
            "function": "relevance_filter",
            "input": "${search_papers.result}",
            "parameters": {"min_relevance": 0.8}
        },
        {
            "id": "extract_abstracts",
            "type": "parallel_processing",
            "function": "extract_text",
            "input": "${filter_papers.result}",
            "parallel": True
        },
        {
            "id": "analyze_trends",
            "type": "inference",
            "model": "gpt-4",
            "input": "Analyze research trends in: ${extract_abstracts.result}"
        },
        {
            "id": "generate_summary",
            "type": "inference",
            "model": "gpt-4",
            "input": "Generate comprehensive summary: ${analyze_trends.result}"
        }
    ]
)
```

## 📞 Support

- **Workflow Issues**: workflows@prsm.ai
- **Execution Problems**: execution-support@prsm.ai
- **Performance**: performance@prsm.ai
- **Documentation**: docs@prsm.ai