Workflow Scheduling API
=======================

The Workflow Scheduling API provides advanced workflow scheduling and orchestration capabilities with critical path optimization, resource management, and cost optimization. This enterprise-grade scheduling system enables complex multi-step workflow execution with dependency management and intelligent resource allocation.

**Key Features:**

* Dependency-aware workflow scheduling with critical path calculation
* Resource conflict detection and intelligent resolution
* Cost optimization through strategic scheduling and timing
* Real-time execution monitoring and adaptive adjustment
* Parallel execution opportunity analysis and optimization
* Advanced critical path analysis and bottleneck identification

.. automodule:: prsm.api.workflow_scheduling_api
   :members:
   :undoc-members:
   :show-inheritance:

API Endpoints
-------------

Schedule Workflow
~~~~~~~~~~~~~~~~~

.. http:post:: /api/v1/scheduling/schedule-workflow

   Schedule a workflow for execution with advanced dependency analysis and critical path optimization.

   **Request Body:**

   .. code-block:: json

      {
        "workflow_name": "Data Processing Pipeline",
        "description": "Multi-step data analysis workflow with ML inference",
        "steps": [
          {
            "step_name": "data_ingestion",
            "step_description": "Load and validate raw data",
            "agent_type": "data_processor",
            "prompt_template": "Process data from {source} with validation rules: {rules}",
            "parameters": {
              "source": "s3://data-bucket/raw/",
              "rules": ["non_null", "schema_validation"]
            },
            "depends_on": [],
            "blocks": ["data_cleaning"],
            "estimated_duration_minutes": 15,
            "resource_requirements": [
              {
                "resource_type": "cpu_cores",
                "amount": 4.0,
                "min_amount": 2.0,
                "max_amount": 8.0
              },
              {
                "resource_type": "memory_gb",
                "amount": 8.0,
                "min_amount": 4.0
              }
            ],
            "max_retries": 3,
            "timeout_minutes": 30
          },
          {
            "step_name": "ml_inference",
            "step_description": "Run ML model inference on processed data",
            "agent_type": "ml_processor",
            "prompt_template": "Run inference using model {model_name} on dataset {dataset}",
            "parameters": {
              "model_name": "bert-large-classifier",
              "dataset": "processed_data"
            },
            "depends_on": ["data_cleaning"],
            "blocks": [],
            "estimated_duration_minutes": 45,
            "resource_requirements": [
              {
                "resource_type": "gpu_memory_gb",
                "amount": 16.0,
                "min_amount": 8.0
              }
            ]
          }
        ],
        "execution_window": {
          "earliest_start_time": "2025-07-02T14:00:00Z",
          "latest_start_time": "2025-07-02T20:00:00Z",
          "preferred_start_time": "2025-07-02T16:00:00Z",
          "max_duration_hours": 4.0,
          "allow_split_execution": false,
          "allow_preemption": true
        },
        "scheduling_priority": "high",
        "max_ftns_cost": 150.0,
        "cost_optimization_enabled": true,
        "preemption_allowed": false,
        "tags": ["data_science", "production", "daily_batch"]
      }

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "workflow_id": "wf_abc123def456",
        "scheduled_start": "2025-07-02T16:15:00Z",
        "estimated_completion": "2025-07-02T18:30:00Z",
        "critical_path_duration_seconds": 3600,
        "estimated_cost": 125.50,
        "cost_savings": 24.50,
        "optimization_analysis": {
          "parallelization_opportunities": 2,
          "resource_conflicts_resolved": 1,
          "scheduling_efficiency": 0.92,
          "recommendations": [
            "Step 'data_validation' can run in parallel with 'data_cleaning'",
            "Resource allocation optimized to reduce idle time by 15%"
          ]
        },
        "message": "Workflow scheduled successfully with critical path optimization"
      }

Critical Path Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /api/v1/scheduling/optimize-critical-path

   Perform advanced critical path analysis and optimization for a scheduled workflow.

   **Request Body:**

   .. code-block:: json

      {
        "workflow_id": "wf_abc123def456",
        "optimization_goals": {
          "minimize_duration": 0.5,
          "minimize_cost": 0.3,
          "balance_resources": 0.2
        },
        "apply_optimizations": true
      }

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "workflow_id": "wf_abc123def456",
        "original_duration_seconds": 4200,
        "optimized_duration_seconds": 3600,
        "time_savings_seconds": 600,
        "original_cost": 150.0,
        "optimized_cost": 125.50,
        "cost_savings": 24.50,
        "critical_path": [
          {
            "step_name": "data_ingestion",
            "duration_seconds": 900,
            "is_critical": true,
            "slack_time_seconds": 0
          },
          {
            "step_name": "ml_inference",
            "duration_seconds": 2700,
            "is_critical": true,
            "slack_time_seconds": 0
          }
        ],
        "bottlenecks": [
          {
            "step_name": "ml_inference",
            "issue": "GPU memory constraint",
            "impact": "Limits parallel execution",
            "recommendation": "Consider step splitting or resource upgrade"
          }
        ],
        "parallelization_opportunities": [
          {
            "parallel_steps": ["data_validation", "data_cleaning"],
            "time_savings_seconds": 600,
            "resource_requirements": "Additional 2 CPU cores"
          }
        ],
        "optimizations_applied": [
          "Reordered non-critical steps for better resource utilization",
          "Optimized resource allocation to reduce wait times",
          "Identified parallel execution opportunities"
        ]
      }

Workflow Status
~~~~~~~~~~~~~~~

.. http:get:: /api/v1/scheduling/workflow/{workflow_id}/status

   Get comprehensive status and metrics for a scheduled workflow.

   **Path Parameters:**
   
   * ``workflow_id`` - Unique identifier for the workflow

   **Response:**

   .. code-block:: json

      {
        "workflow_id": "wf_abc123def456",
        "workflow_name": "Data Processing Pipeline",
        "status": "running",
        "scheduled_start": "2025-07-02T16:15:00Z",
        "actual_start": "2025-07-02T16:16:23Z",
        "actual_end": null,
        "execution_attempts": 1,
        "critical_path_duration_seconds": 3600,
        "estimated_cost": 125.50,
        "actual_cost": 67.20,
        "cost_savings": 24.50,
        "current_step": {
          "step_name": "ml_inference",
          "status": "running",
          "progress": 0.65,
          "estimated_completion": "2025-07-02T17:45:00Z"
        },
        "completed_steps": ["data_ingestion", "data_cleaning"],
        "resource_utilization": {
          "cpu_cores": 6.0,
          "memory_gb": 12.0,
          "gpu_memory_gb": 16.0
        }
      }

   **Workflow Status Values:**
   
   * ``scheduled`` - Workflow is scheduled but not yet started
   * ``running`` - Workflow is currently executing
   * ``completed`` - Workflow completed successfully
   * ``failed`` - Workflow failed during execution
   * ``cancelled`` - Workflow was cancelled by user
   * ``paused`` - Workflow execution is paused

List User Workflows
~~~~~~~~~~~~~~~~~~~

.. http:get:: /api/v1/scheduling/workflows

   List workflows owned by the current user with filtering and pagination.

   **Query Parameters:**
   
   * ``status_filter`` (optional) - Filter by workflow status
   * ``limit`` (optional, default: 50) - Maximum number of workflows (1-100)
   * ``offset`` (optional, default: 0) - Number of workflows to skip

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "workflows": [
          {
            "workflow_id": "wf_abc123def456",
            "workflow_name": "Data Processing Pipeline",
            "description": "Multi-step data analysis workflow",
            "status": "completed",
            "scheduling_priority": "high",
            "created_at": "2025-07-02T14:30:00Z",
            "scheduled_start": "2025-07-02T16:15:00Z",
            "step_count": 5,
            "critical_path_duration_seconds": 3600,
            "estimated_cost": 125.50,
            "actual_cost": 118.75,
            "cost_savings": 24.50,
            "tags": ["data_science", "production"]
          }
        ],
        "pagination": {
          "total_count": 25,
          "limit": 50,
          "offset": 0,
          "has_more": false
        },
        "summary": {
          "total_workflows": 25,
          "active_workflows": 3,
          "completed_workflows": 20,
          "total_cost_savings": 456.75
        }
      }

Execute Workflow
~~~~~~~~~~~~~~~~

.. http:post:: /api/v1/scheduling/workflow/{workflow_id}/execute

   Execute a scheduled workflow immediately, overriding the scheduled start time.

   **Path Parameters:**
   
   * ``workflow_id`` - Unique identifier for the workflow

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "workflow_id": "wf_abc123def456",
        "message": "Workflow execution started",
        "execution_id": "exec_789xyz123",
        "started_at": "2025-07-02T11:30:00Z",
        "estimated_completion": "2025-07-02T12:30:00Z",
        "resource_allocation": {
          "cpu_cores": 6.0,
          "memory_gb": 12.0,
          "gpu_memory_gb": 16.0
        },
        "critical_path_optimized": true
      }

System Statistics
~~~~~~~~~~~~~~~~~

.. http:get:: /api/v1/scheduling/system/statistics

   Get comprehensive scheduling system statistics and performance metrics.

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "statistics": {
          "total_workflows_scheduled": 1250,
          "active_workflows": 15,
          "completed_workflows_today": 45,
          "average_execution_time_seconds": 1800,
          "system_utilization": {
            "cpu_utilization": 0.72,
            "memory_utilization": 0.68,
            "gpu_utilization": 0.85
          },
          "scheduling_efficiency": 0.94,
          "cost_optimization": {
            "total_savings_today": 234.50,
            "average_savings_per_workflow": 15.60,
            "optimization_success_rate": 0.89
          },
          "queue_statistics": {
            "pending_workflows": 8,
            "average_wait_time_seconds": 120,
            "priority_queue_depth": {
              "critical": 2,
              "high": 3,
              "normal": 8,
              "low": 5
            }
          },
          "resource_statistics": {
            "peak_resource_usage": "14:30-16:00 UTC",
            "resource_conflicts_resolved": 12,
            "parallel_execution_efficiency": 0.87
          }
        },
        "timestamp": "2025-07-02T11:30:00Z",
        "user_id": "user_123"
      }

Authentication & Authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All workflow scheduling endpoints require authentication. Users can only access and manage their own workflows.

**Headers:**

.. code-block:: http

   Authorization: Bearer <jwt_token>
   Content-Type: application/json

**Permission Model:**

* **Workflow Management** - Users can schedule, execute, and manage their own workflows
* **System Statistics** - Available to all authenticated users
* **Cross-User Access** - Restricted (users cannot access other users' workflows)

Resource Management
~~~~~~~~~~~~~~~~~~~

The scheduling system supports various resource types and intelligent allocation:

**Resource Types:**

* ``cpu_cores`` - CPU core allocation
* ``memory_gb`` - Memory allocation in GB
* ``gpu_memory_gb`` - GPU memory allocation in GB
* ``storage_gb`` - Temporary storage allocation
* ``network_bandwidth_mbps`` - Network bandwidth allocation

**Resource Allocation Features:**

* **Burst Capability** - Allow temporary resource overallocation
* **Priority Multipliers** - Weight resource allocation by step priority
* **Conflict Resolution** - Automatic scheduling adjustments for resource conflicts
* **Dynamic Scaling** - Resource allocation adjustment during execution

Cost Optimization
~~~~~~~~~~~~~~~~~~

The system provides comprehensive cost optimization features:

**Optimization Strategies:**

* **Time-based Pricing** - Schedule workflows during lower-cost periods
* **Resource Efficiency** - Optimize resource allocation to minimize waste
* **Parallel Execution** - Maximize parallelization to reduce total runtime
* **Critical Path Focus** - Prioritize optimization efforts on critical path steps

**Cost Tracking:**

* **Estimated Costs** - Pre-execution cost estimates
* **Real-time Tracking** - Actual cost monitoring during execution
* **Savings Analysis** - Detailed breakdown of optimization savings
* **Budget Management** - Workflow cost limits and budget enforcement

Error Responses
--------------

Workflow Scheduling API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "invalid_workflow_status",
     "message": "Workflow cannot be executed in completed state",
     "code": 400,
     "timestamp": "2025-07-02T11:30:00Z",
     "workflow_id": "wf_abc123def456"
   }

**Common Error Codes:**

* ``400`` - Bad Request (invalid parameters, malformed workflow definition)
* ``401`` - Unauthorized (missing or invalid authentication token)
* ``403`` - Forbidden (access denied to workflow belonging to another user)
* ``404`` - Not Found (workflow not found)
* ``409`` - Conflict (resource conflicts that cannot be automatically resolved)
* ``429`` - Rate Limited (too many scheduling requests)
* ``500`` - Internal Server Error (scheduling system error)

**Workflow-Specific Errors:**

* ``invalid_workflow_status`` - Workflow cannot be executed in current state
* ``resource_conflict`` - Unable to resolve resource allocation conflicts
* ``dependency_cycle`` - Circular dependency detected in workflow steps
* ``cost_limit_exceeded`` - Workflow cost exceeds specified budget
* ``scheduling_window_expired`` - Execution window has passed
