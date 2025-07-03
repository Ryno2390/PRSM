Task API
========

The Task API provides distributed task queue management for the PRSM platform, enabling asynchronous task processing across the P2P network with priority ordering, load balancing, and comprehensive monitoring capabilities.

**Key Features:**

* Redis-based distributed task queue with high availability
* Priority-based task scheduling and execution
* Real-time task status monitoring and progress tracking
* Load balancing across P2P network nodes
* Retry mechanisms and failure handling
* Comprehensive task analytics and performance metrics

.. automodule:: prsm.api.task_api
   :members:
   :undoc-members:
   :show-inheritance:

API Endpoints
-------------

Enqueue Task
~~~~~~~~~~~~

.. http:post:: /enqueue

   Enqueue a task in the distributed task queue for asynchronous processing.

   **Request Body:**

   .. code-block:: json

      {
        "queue_name": "ml_inference_queue",
        "task_data": {
          "model_id": "bert-large-classifier",
          "input_text": "Analyze this document for sentiment",
          "parameters": {
            "max_length": 512,
            "return_confidence": true
          }
        },
        "priority": 7,
        "retry_count": 3,
        "timeout_seconds": 300
      }

   **Priority Levels:**
   
   * ``1-3`` - Low priority (batch processing, non-urgent tasks)
   * ``4-6`` - Normal priority (standard operations)
   * ``7-9`` - High priority (user-facing requests)
   * ``10`` - Critical priority (system-critical operations)

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "task_id": "task_abc123def456",
        "queue_name": "ml_inference_queue",
        "priority": 7,
        "status": "enqueued",
        "estimated_wait_time_seconds": 45,
        "position_in_queue": 3
      }

Get Task Status
~~~~~~~~~~~~~~~

.. http:get:: /status/{task_id}

   Get the current status and progress of a queued or executing task.

   **Path Parameters:**
   
   * ``task_id`` - Unique identifier for the task

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "task_id": "task_abc123def456",
        "status": "processing",
        "progress": 0.65,
        "queue_name": "ml_inference_queue",
        "priority": 7,
        "created_at": "2025-07-02T14:30:00Z",
        "started_at": "2025-07-02T14:31:15Z",
        "estimated_completion": "2025-07-02T14:35:00Z",
        "worker_node": "node_worker_007",
        "retry_count": 0,
        "max_retries": 3,
        "result": null,
        "error": null
      }

   **Task Status Values:**
   
   * ``enqueued`` - Task is waiting in queue
   * ``processing`` - Task is currently being executed
   * ``completed`` - Task completed successfully
   * ``failed`` - Task failed after all retry attempts
   * ``cancelled`` - Task was cancelled by user or system
   * ``timeout`` - Task exceeded timeout limit

Get Task Result
~~~~~~~~~~~~~~~

.. http:get:: /result/{task_id}

   Retrieve the result of a completed task.

   **Path Parameters:**
   
   * ``task_id`` - Unique identifier for the task

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "task_id": "task_abc123def456",
        "status": "completed",
        "result": {
          "sentiment": "positive",
          "confidence": 0.89,
          "processing_time_seconds": 2.34,
          "model_version": "bert-large-v1.2"
        },
        "execution_time_seconds": 2.34,
        "completed_at": "2025-07-02T14:33:34Z",
        "worker_node": "node_worker_007"
      }

Cancel Task
~~~~~~~~~~~

.. http:delete:: /cancel/{task_id}

   Cancel a queued or processing task.

   **Path Parameters:**
   
   * ``task_id`` - Unique identifier for the task

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "task_id": "task_abc123def456",
        "status": "cancelled",
        "message": "Task cancelled successfully",
        "was_processing": false,
        "cancelled_at": "2025-07-02T14:32:00Z"
      }

Queue Statistics
~~~~~~~~~~~~~~~~

.. http:get:: /queue/{queue_name}/stats

   Get comprehensive statistics for a specific task queue.

   **Path Parameters:**
   
   * ``queue_name`` - Name of the task queue

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "queue_name": "ml_inference_queue",
        "statistics": {
          "total_tasks": 1250,
          "pending_tasks": 15,
          "processing_tasks": 8,
          "completed_tasks_today": 89,
          "failed_tasks_today": 2,
          "average_processing_time_seconds": 45.2,
          "average_wait_time_seconds": 12.5,
          "throughput_tasks_per_hour": 125,
          "worker_nodes_active": 12,
          "queue_health_score": 0.96
        },
        "priority_distribution": {
          "low": 5,
          "normal": 8,
          "high": 2,
          "critical": 0
        },
        "performance_metrics": {
          "success_rate": 0.98,
          "retry_rate": 0.05,
          "timeout_rate": 0.01,
          "avg_cpu_usage": 0.67,
          "avg_memory_usage": 0.54
        }
      }

List User Tasks
~~~~~~~~~~~~~~~

.. http:get:: /tasks

   List tasks submitted by the current user with filtering and pagination.

   **Query Parameters:**
   
   * ``status`` (optional) - Filter by task status
   * ``queue_name`` (optional) - Filter by queue name
   * ``limit`` (optional, default: 50) - Maximum number of tasks (1-100)
   * ``offset`` (optional, default: 0) - Number of tasks to skip

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "tasks": [
          {
            "task_id": "task_abc123def456",
            "queue_name": "ml_inference_queue",
            "status": "completed",
            "priority": 7,
            "created_at": "2025-07-02T14:30:00Z",
            "completed_at": "2025-07-02T14:33:34Z",
            "execution_time_seconds": 2.34,
            "retry_count": 0
          }
        ],
        "pagination": {
          "total_count": 125,
          "limit": 50,
          "offset": 0,
          "has_more": true
        },
        "summary": {
          "total_tasks": 125,
          "completed_tasks": 118,
          "failed_tasks": 3,
          "pending_tasks": 4,
          "success_rate": 0.976
        }
      }

Bulk Operations
~~~~~~~~~~~~~~~

.. http:post:: /bulk/enqueue

   Enqueue multiple tasks in a single request for batch processing.

   **Request Body:**

   .. code-block:: json

      {
        "queue_name": "batch_processing_queue",
        "tasks": [
          {
            "task_data": {"input": "data1"},
            "priority": 5
          },
          {
            "task_data": {"input": "data2"},
            "priority": 5
          }
        ],
        "batch_options": {
          "parallel_execution": true,
          "max_concurrent": 5,
          "fail_fast": false
        }
      }

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "batch_id": "batch_xyz789",
        "task_ids": [
          "task_abc123def456",
          "task_def456ghi789"
        ],
        "total_tasks": 2,
        "estimated_completion": "2025-07-02T14:40:00Z"
      }

System Health
~~~~~~~~~~~~~

.. http:get:: /health

   Get task system health and performance metrics.

   **Response:**

   .. code-block:: json

      {
        "status": "healthy",
        "timestamp": "2025-07-02T14:30:00Z",
        "system_metrics": {
          "total_queues": 8,
          "active_workers": 45,
          "total_tasks_processed_today": 15420,
          "current_queue_depth": 125,
          "average_processing_time_seconds": 23.4,
          "system_throughput_tasks_per_minute": 125,
          "error_rate": 0.02
        },
        "resource_utilization": {
          "cpu_usage": 0.68,
          "memory_usage": 0.54,
          "network_throughput_mbps": 120,
          "storage_usage": 0.45
        },
        "queue_health": {
          "healthy_queues": 8,
          "degraded_queues": 0,
          "unhealthy_queues": 0
        }
      }

Authentication & Authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All task API endpoints require authentication. Users can only access and manage their own tasks.

**Headers:**

.. code-block:: http

   Authorization: Bearer <jwt_token>
   Content-Type: application/json

Task Queue Types
~~~~~~~~~~~~~~~~

The system supports various specialized queues for different types of operations:

**Queue Categories:**

* **ML Processing** - ``ml_inference_queue``, ``training_queue``, ``model_optimization_queue``
* **Data Processing** - ``data_ingestion_queue``, ``etl_processing_queue``, ``batch_processing_queue``
* **System Operations** - ``system_maintenance_queue``, ``monitoring_queue``, ``backup_queue``
* **User Operations** - ``user_requests_queue``, ``notification_queue``, ``analytics_queue``

Error Responses
--------------

Task API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "task_not_found",
     "message": "Task with ID task_abc123def456 not found",
     "code": 404,
     "timestamp": "2025-07-02T14:30:00Z",
     "task_id": "task_abc123def456"
   }

**Common Error Codes:**

* ``400`` - Bad Request (invalid task data, missing required fields)
* ``401`` - Unauthorized (missing or invalid authentication token)
* ``403`` - Forbidden (access denied to task belonging to another user)
* ``404`` - Not Found (task not found)
* ``429`` - Rate Limited (too many task submissions)
* ``503`` - Service Unavailable (task queue system unavailable)
* ``500`` - Internal Server Error (task processing system error)

**Task-Specific Errors:**

* ``task_not_found`` - Specified task ID does not exist
* ``queue_not_available`` - Specified queue is not available or disabled
* ``task_timeout`` - Task exceeded maximum execution time
* ``worker_unavailable`` - No workers available to process the task
* ``invalid_task_data`` - Task data format is invalid or incomplete
