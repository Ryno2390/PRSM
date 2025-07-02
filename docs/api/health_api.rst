Health API
==========

The Health API provides system health monitoring and diagnostics endpoints for the PRSM platform.

.. automodule:: prsm.api.health_api
   :members:
   :undoc-members:
   :show-inheritance:

System Health
-----------

Health Check
^^^^^^^^^^^

.. http:get:: /health

   Basic health check endpoint for load balancers and monitoring systems.

   **Response JSON Object:**
   
   * **status** (*string*) -- Overall system status ("healthy", "degraded", "unhealthy")
   * **timestamp** (*string*) -- Health check timestamp (ISO 8601)
   * **version** (*string*) -- Application version
   * **uptime** (*integer*) -- System uptime in seconds

   **Status Codes:**
   
   * **200** -- System is healthy
   * **503** -- System is unhealthy

Detailed Health
^^^^^^^^^^^^^^

.. http:get:: /health/detailed

   Comprehensive health check with component-level status.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (optional, provides more details if authenticated)

   **Response JSON Object:**
   
   * **status** (*string*) -- Overall system status
   * **components** (*object*) -- Component-level health status
   * **metrics** (*object*) -- Key performance metrics
   * **dependencies** (*object*) -- External dependency status
   * **timestamp** (*string*) -- Check timestamp
   * **checks_performed** (*integer*) -- Number of health checks performed

   **Component Status Object:**
   
   * **database** (*object*) -- Database connection and performance
   * **redis** (*object*) -- Redis cache status
   * **ipfs** (*object*) -- IPFS network connectivity
   * **ai_models** (*object*) -- AI model availability
   * **blockchain** (*object*) -- Blockchain network status
   * **storage** (*object*) -- Storage system health

   **Status Codes:**
   
   * **200** -- Health information retrieved successfully
   * **503** -- System has critical issues

Component Health
^^^^^^^^^^^^^^^

.. http:get:: /health/components/{component}

   Get health status for a specific system component.

   **Path Parameters:**
   
   * **component** (*string*) -- Component name ("database", "redis", "ipfs", "models", "blockchain")

   **Response JSON Object:**
   
   * **component** (*string*) -- Component name
   * **status** (*string*) -- Component status ("healthy", "degraded", "unhealthy")
   * **response_time** (*float*) -- Response time in milliseconds
   * **last_check** (*string*) -- Last health check timestamp
   * **details** (*object*) -- Component-specific health details
   * **metrics** (*object*) -- Component performance metrics

   **Status Codes:**
   
   * **200** -- Component health retrieved successfully
   * **404** -- Component not found
   * **503** -- Component is unhealthy

Performance Metrics
------------------

System Metrics
^^^^^^^^^^^^^

.. http:get:: /health/metrics

   Get current system performance metrics.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (optional)

   **Response JSON Object:**
   
   * **cpu** (*object*) -- CPU usage statistics
   * **memory** (*object*) -- Memory usage statistics
   * **disk** (*object*) -- Disk usage and I/O metrics
   * **network** (*object*) -- Network traffic metrics
   * **requests** (*object*) -- API request statistics
   * **errors** (*object*) -- Error rate statistics
   * **response_times** (*object*) -- Response time percentiles

   **CPU Object:**
   
   * **usage_percent** (*float*) -- Current CPU usage percentage
   * **load_average** (*array*) -- Load average (1, 5, 15 minutes)
   * **cores** (*integer*) -- Number of CPU cores

   **Memory Object:**
   
   * **used_bytes** (*integer*) -- Used memory in bytes
   * **total_bytes** (*integer*) -- Total memory in bytes
   * **usage_percent** (*float*) -- Memory usage percentage
   * **swap_used** (*integer*) -- Swap usage in bytes

   **Status Codes:**
   
   * **200** -- Metrics retrieved successfully
   * **401** -- Unauthorized (for detailed metrics)

Database Health
^^^^^^^^^^^^^^

.. http:get:: /health/database

   Get detailed database health and performance metrics.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **status** (*string*) -- Database status
   * **connection_pool** (*object*) -- Connection pool metrics
   * **query_performance** (*object*) -- Query performance statistics
   * **storage** (*object*) -- Database storage metrics
   * **replication** (*object*) -- Replication status (if applicable)

   **Connection Pool Object:**
   
   * **active_connections** (*integer*) -- Currently active connections
   * **idle_connections** (*integer*) -- Idle connections in pool
   * **max_connections** (*integer*) -- Maximum allowed connections
   * **pool_utilization** (*float*) -- Pool utilization percentage

   **Status Codes:**
   
   * **200** -- Database health retrieved successfully
   * **401** -- Unauthorized
   * **503** -- Database is unhealthy

AI Model Health
^^^^^^^^^^^^^^

.. http:get:: /health/models

   Get health status of AI model services.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **total_models** (*integer*) -- Total number of models
   * **healthy_models** (*integer*) -- Number of healthy models
   * **providers** (*object*) -- Provider-specific health status
   * **average_response_time** (*float*) -- Average model response time
   * **error_rate** (*float*) -- Model error rate percentage

   **Provider Status:**
   
   * **openai** (*object*) -- OpenAI service status
   * **anthropic** (*object*) -- Anthropic service status
   * **local** (*object*) -- Local model status
   * **openrouter** (*object*) -- OpenRouter service status

   **Status Codes:**
   
   * **200** -- Model health retrieved successfully
   * **401** -- Unauthorized

Dependency Health
----------------

External Dependencies
^^^^^^^^^^^^^^^^^^^^

.. http:get:: /health/dependencies

   Check status of external service dependencies.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **dependencies** (*object*) -- Status of each dependency
   * **overall_status** (*string*) -- Overall dependency health
   * **critical_failures** (*integer*) -- Number of critical dependency failures

   **Dependency Object:**
   
   * **service_name** (*string*) -- Name of the external service
   * **status** (*string*) -- Service status
   * **response_time** (*float*) -- Service response time
   * **last_check** (*string*) -- Last check timestamp
   * **error_message** (*string*, optional) -- Error details if unhealthy

   **External Services:**
   
   * **blockchain_rpc** -- Blockchain network RPC
   * **ipfs_gateway** -- IPFS gateway service
   * **ai_providers** -- External AI model providers
   * **payment_gateway** -- Payment processing service

   **Status Codes:**
   
   * **200** -- Dependency status retrieved successfully
   * **401** -- Unauthorized

Blockchain Health
^^^^^^^^^^^^^^^

.. http:get:: /health/blockchain

   Get blockchain network health and connectivity status.

   **Response JSON Object:**
   
   * **network** (*string*) -- Blockchain network name
   * **connected** (*boolean*) -- Network connectivity status
   * **block_height** (*integer*) -- Current block height
   * **sync_status** (*string*) -- Synchronization status
   * **peer_count** (*integer*) -- Number of connected peers
   * **gas_price** (*object*) -- Current gas price information

   **Status Codes:**
   
   * **200** -- Blockchain health retrieved successfully
   * **503** -- Blockchain connectivity issues

Health Monitoring
---------------

Health History
^^^^^^^^^^^^^

.. http:get:: /health/history

   Get historical health data and trends.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **period** (*string*, optional) -- Time period ("hour", "day", "week", default: "day")
   * **component** (*string*, optional) -- Specific component to analyze

   **Response JSON Object:**
   
   * **period** (*string*) -- Requested time period
   * **data_points** (*array*) -- Historical health data points
   * **trends** (*object*) -- Health trend analysis
   * **incidents** (*array*) -- Health incidents during period

   **Status Codes:**
   
   * **200** -- Health history retrieved successfully
   * **401** -- Unauthorized

Alerts and Incidents
^^^^^^^^^^^^^^^^^^^

.. http:get:: /health/alerts

   Get current system alerts and incidents.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **severity** (*string*, optional) -- Filter by severity ("critical", "high", "medium", "low")
   * **status** (*string*, optional) -- Filter by status ("active", "resolved", "acknowledged")

   **Response JSON Object:**
   
   * **active_alerts** (*array*) -- Currently active alerts
   * **recent_incidents** (*array*) -- Recent incident history
   * **alert_summary** (*object*) -- Alert count by severity

   **Alert Object:**
   
   * **alert_id** (*string*) -- Unique alert identifier
   * **severity** (*string*) -- Alert severity level
   * **component** (*string*) -- Affected component
   * **message** (*string*) -- Alert description
   * **triggered_at** (*string*) -- Alert trigger timestamp
   * **status** (*string*) -- Alert status

   **Status Codes:**
   
   * **200** -- Alerts retrieved successfully
   * **401** -- Unauthorized

Diagnostic Tools
--------------

System Diagnostics
^^^^^^^^^^^^^^^^^

.. http:post:: /health/diagnostics

   Run comprehensive system diagnostics.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (admin role required)

   **Request JSON Object:**
   
   * **components** (*array*, optional) -- Specific components to test
   * **deep_check** (*boolean*, optional) -- Perform deep diagnostic tests
   * **include_logs** (*boolean*, optional) -- Include recent error logs

   **Response JSON Object:**
   
   * **diagnostic_id** (*string*) -- Diagnostic session identifier
   * **results** (*object*) -- Diagnostic test results
   * **recommendations** (*array*) -- System improvement recommendations
   * **error_logs** (*array*, optional) -- Recent error entries

   **Status Codes:**
   
   * **200** -- Diagnostics completed successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Performance Test
^^^^^^^^^^^^^^^

.. http:post:: /health/performance-test

   Execute a controlled performance test.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (admin role required)

   **Request JSON Object:**
   
   * **test_type** (*string*) -- Type of test ("load", "stress", "endurance")
   * **duration** (*integer*, optional) -- Test duration in seconds
   * **concurrency** (*integer*, optional) -- Number of concurrent requests

   **Response JSON Object:**
   
   * **test_id** (*string*) -- Performance test identifier
   * **status** (*string*) -- Test execution status
   * **results** (*object*, optional) -- Test results (if completed)

   **Status Codes:**
   
   * **202** -- Performance test started
   * **200** -- Test results available
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Configuration
-----------

Health Check Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /health/config

   Get current health check configuration.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (admin role required)

   **Response JSON Object:**
   
   * **check_intervals** (*object*) -- Health check intervals for each component
   * **thresholds** (*object*) -- Health threshold configurations
   * **alerting** (*object*) -- Alert configuration settings
   * **retention** (*object*) -- Data retention policies

   **Status Codes:**
   
   * **200** -- Configuration retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Update Configuration
^^^^^^^^^^^^^^^^^^^

.. http:put:: /health/config

   Update health monitoring configuration.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (admin role required)

   **Request JSON Object:**
   
   * **check_intervals** (*object*, optional) -- Updated check intervals
   * **thresholds** (*object*, optional) -- Updated health thresholds
   * **alerting** (*object*, optional) -- Updated alert settings

   **Status Codes:**
   
   * **200** -- Configuration updated successfully
   * **400** -- Invalid configuration
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Error Responses
--------------

Health API error responses:

.. code-block:: json

   {
     "error": "component_unhealthy",
     "message": "Database connection failed",
     "code": 503,
     "component": "database",
     "timestamp": "2025-07-02T10:30:00Z",
     "details": {
       "connection_error": "Connection timeout after 5 seconds"
     }
   }

Common error types:

* ``component_unhealthy`` - System component is not functioning
* ``dependency_failure`` - External dependency is unavailable
* ``threshold_exceeded`` - Performance threshold exceeded
* ``diagnostic_failed`` - Diagnostic test failure
* ``insufficient_resources`` - System resource constraints

Rate Limits
-----------

Health API rate limits:

* Basic health checks: No rate limit (designed for frequent monitoring)
* Detailed health: 100 requests per minute
* Diagnostics: 10 requests per hour per user
* Performance tests: 5 requests per day per user
* Configuration changes: 20 requests per hour per admin

Security and Access
------------------

* Basic health endpoints are publicly accessible
* Detailed metrics require authentication
* Diagnostic tools require admin privileges
* All configuration changes are logged and audited
* Sensitive information is filtered from public endpoints