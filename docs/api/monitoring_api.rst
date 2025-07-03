Monitoring API
==============

The Monitoring API provides comprehensive system monitoring, observability, and business intelligence for the PRSM platform. This enterprise-grade monitoring system delivers real-time insights, performance analytics, and business metrics tracking with advanced alerting capabilities.

**Key Features:**

* Real-time system health monitoring and alerting
* Advanced performance analytics and optimization insights  
* Business metrics tracking and KPI dashboards
* Distributed tracing and request correlation
* Compliance logging and audit trail management

.. automodule:: prsm.api.monitoring_api
   :members:
   :undoc-members:
   :show-inheritance:

API Endpoints
-------------

Metrics Management
~~~~~~~~~~~~~~~~~~

.. http:post:: /monitoring/metrics

   Record custom metrics for monitoring and analytics.

   **Request Body:**

   .. code-block:: json

      {
        "name": "api_response_time",
        "value": 245.3,
        "metric_type": "timer",
        "component": "api_gateway",
        "tags": {"endpoint": "/recommendations", "method": "GET"},
        "labels": {"version": "v1", "region": "us-east-1"}
      }

   **Supported Metric Types:**
   
   * ``counter`` - Cumulative metrics that only increase
   * ``gauge`` - Point-in-time values that can go up or down
   * ``histogram`` - Distribution of values over time
   * ``timer`` - Duration measurements
   * ``business`` - Business KPIs and outcome metrics

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "message": "Metric recorded successfully",
        "metric_name": "api_response_time",
        "timestamp": "2025-07-02T10:30:00Z"
      }

.. http:post:: /monitoring/business-metrics

   Record business KPI and outcome metrics.

   **Request Body:**

   .. code-block:: json

      {
        "metric_name": "monthly_recurring_revenue",
        "value": 125000.0,
        "dimension": "monthly",
        "target": 150000.0,
        "metadata": {
          "currency": "USD",
          "period": "2025-07"
        }
      }

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "message": "Business metric recorded successfully",
        "metric_name": "monthly_recurring_revenue",
        "value": 125000.0,
        "dimension": "monthly",
        "timestamp": "2025-07-02T10:30:00Z"
      }

Alert Management
~~~~~~~~~~~~~~~~

.. http:post:: /monitoring/alerts

   Create a new monitoring alert with configurable rules and thresholds.

   **Request Body:**

   .. code-block:: json

      {
        "name": "High API Response Time",
        "description": "Alert when API response time exceeds acceptable threshold",
        "severity": "warning",
        "component": "api_gateway",
        "condition": "avg(api_response_time) > threshold",
        "threshold": 500.0,
        "duration_seconds": 300,
        "notification_channels": ["slack", "email"]
      }

   **Alert Severities:**
   
   * ``info`` - Informational alerts
   * ``warning`` - Warning level issues
   * ``error`` - Error conditions requiring attention
   * ``critical`` - Critical issues requiring immediate response

   **Response:**

   .. code-block:: json

      {
        "success": true,
        "message": "Alert created successfully",
        "alert_id": "alert_12345",
        "name": "High API Response Time",
        "severity": "warning",
        "component": "api_gateway"
      }

System Health
~~~~~~~~~~~~~

.. http:get:: /monitoring/health

   Get comprehensive system health status and metrics.

   **Response:**

   .. code-block:: json

      {
        "overall_health_score": 0.95,
        "system_metrics": {
          "cpu_usage": 45.2,
          "memory_usage": 67.8,
          "disk_usage": 23.1,
          "network_throughput": 1250
        },
        "recent_performance": {
          "avg_response_time": 234.5,
          "requests_per_second": 45.2,
          "error_rate": 0.001
        },
        "active_alerts": [
          {
            "id": "alert_001",
            "name": "Database Connection Pool",
            "severity": "warning",
            "component": "database"
          }
        ],
        "monitoring_status": {
          "metrics_collected_last_minute": 1250,
          "traces_collected_last_minute": 89,
          "alerts_evaluated": 45
        },
        "timestamp": "2025-07-02T10:30:00Z"
      }

Performance Analytics
~~~~~~~~~~~~~~~~~~~~~

.. http:get:: /monitoring/analytics

   Get comprehensive performance analytics and insights.

   **Query Parameters:**
   
   * ``component`` (optional) - Filter analytics by specific component

   **Response:**

   .. code-block:: json

      {
        "summary": {
          "total_requests": 125000,
          "avg_response_time": 245.3,
          "error_rate": 0.002,
          "throughput": 42.3
        },
        "trends": {
          "response_time_trend": "improving",
          "error_rate_trend": "stable",
          "throughput_trend": "increasing"
        },
        "bottlenecks": [
          "Database connection pool saturation",
          "External API rate limiting"
        ],
        "recommendations": [
          "Increase database connection pool size",
          "Implement response caching for frequently accessed data",
          "Add circuit breaker for external API calls"
        ],
        "top_operations": [
          {
            "name": "GET /recommendations",
            "avg_duration_ms": 189.4,
            "request_count": 45200
          }
        ]
      }

Business Dashboard
~~~~~~~~~~~~~~~~~~

.. http:get:: /monitoring/business-dashboard

   Get comprehensive business metrics dashboard. Requires enterprise role or higher.

   **Response:**

   .. code-block:: json

      {
        "kpis": {
          "monthly_recurring_revenue": 125000.0,
          "customer_acquisition_cost": 45.50,
          "customer_lifetime_value": 2350.0,
          "churn_rate": 0.02
        },
        "trends": {
          "revenue": "increasing",
          "user_engagement": "stable",
          "platform_adoption": "increasing"
        },
        "targets": {
          "revenue": {"current": 125000.0, "target": 150000.0, "achievement": 0.83},
          "users": {"current": 5200, "target": 6000, "achievement": 0.87}
        },
        "user_metrics": {
          "daily_active_users": 1250,
          "monthly_active_users": 5200,
          "user_retention_rate": 0.89
        },
        "revenue_metrics": {
          "monthly_revenue": 125000.0,
          "revenue_per_user": 24.04,
          "revenue_growth_rate": 0.12
        },
        "growth_metrics": {
          "user_growth_rate": 0.08,
          "feature_adoption_rate": 0.65,
          "api_usage_growth": 0.15
        },
        "quality_metrics": {
          "system_uptime": 0.999,
          "response_time_sla": 0.98,
          "customer_satisfaction": 4.6
        },
        "generated_at": "2025-07-02T10:30:00Z"
      }

Distributed Tracing
~~~~~~~~~~~~~~~~~~~

.. http:get:: /monitoring/traces/{trace_id}

   Get detailed information about a specific distributed trace.

   **Path Parameters:**
   
   * ``trace_id`` - Unique identifier for the trace

   **Response:**

   .. code-block:: json

      {
        "trace_id": "trace_abc123",
        "operation": "marketplace_recommendation",
        "component": "recommendation",
        "total_duration_ms": 245.3,
        "spans": [
          {
            "name": "database_query",
            "duration_ms": 45.2,
            "status": "success"
          },
          {
            "name": "ml_inference",
            "duration_ms": 180.1,
            "status": "success"
          },
          {
            "name": "response_formatting",
            "duration_ms": 20.0,
            "status": "success"
          }
        ],
        "success": true,
        "timestamp": "2025-07-02T10:30:00Z",
        "metadata": {
          "user_id": "user_123",
          "request_size": "2.3KB",
          "response_size": "15.7KB"
        }
      }

Reference Endpoints
~~~~~~~~~~~~~~~~~~~

.. http:get:: /monitoring/components

   Get available monitoring components and their capabilities.

   **Response:**

   .. code-block:: json

      {
        "components": {
          "api_gateway": {
            "name": "API Gateway",
            "description": "Monitoring for API gateway component",
            "metrics_available": ["requests", "errors", "duration", "throughput"],
            "alert_types": ["performance", "availability", "errors", "capacity"]
          }
        },
        "total_count": 8,
        "metric_types": ["counter", "gauge", "histogram", "timer", "business"],
        "alert_severities": ["info", "warning", "error", "critical"]
      }

.. http:get:: /monitoring/status

   Health check for monitoring system.

   **Response:**

   .. code-block:: json

      {
        "status": "healthy",
        "timestamp": "2025-07-02T10:30:00Z",
        "monitoring_active": true,
        "components_monitored": 8,
        "metric_types_supported": 5,
        "alert_severities": 4,
        "system_capacity": {
          "max_metrics_per_minute": 10000,
          "max_traces_per_minute": 1000,
          "alert_evaluation_interval": "60 seconds",
          "data_retention_days": 30
        },
        "version": "1.0.0"
      }

Authentication & Authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All monitoring endpoints require authentication. Business dashboard endpoint requires enterprise role or higher. Alert creation requires developer role or higher.

**Headers:**

.. code-block:: http

   Authorization: Bearer <jwt_token>
   Content-Type: application/json

Error Responses
--------------

Monitoring API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "invalid_metric_type",
     "message": "Invalid metric type. Must be one of: counter, gauge, histogram, timer, business",
     "code": 400,
     "timestamp": "2025-07-02T10:30:00Z"
   }

**Common Error Codes:**

* ``400`` - Bad Request (invalid parameters)
* ``401`` - Unauthorized (missing or invalid token)
* ``403`` - Forbidden (insufficient permissions)
* ``404`` - Not Found (resource not found)
* ``429`` - Rate Limited (too many requests)
* ``500`` - Internal Server Error (system error)
