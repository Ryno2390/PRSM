Budget API
==========

The Budget API provides comprehensive budget management and cost tracking for PRSM FTNS tokens and AI model usage.

.. automodule:: prsm.api.budget_api
   :members:
   :undoc-members:
   :show-inheritance:

Budget Overview
--------------

Current Budget
^^^^^^^^^^^^^

.. http:get:: /budget

   Get current budget status and allocation for authenticated user.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **total_budget** (*object*) -- Total budget allocation
   * **current_usage** (*object*) -- Current spending
   * **remaining_budget** (*object*) -- Remaining budget
   * **budget_period** (*object*) -- Budget period information
   * **categories** (*array*) -- Budget by category
   * **alerts** (*array*) -- Active budget alerts

   **Budget Object:**
   
   * **ftns_tokens** (*integer*) -- FTNS token amount
   * **usd_equivalent** (*float*) -- USD equivalent value
   * **last_updated** (*string*) -- Last update timestamp

   **Status Codes:**
   
   * **200** -- Budget information retrieved successfully
   * **401** -- Unauthorized

Budget Details
^^^^^^^^^^^^^

.. http:get:: /budget/detailed

   Get detailed budget breakdown with usage analytics.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **period** (*string*, optional) -- Time period ("day", "week", "month", default: "month")
   * **category** (*string*, optional) -- Filter by category

   **Response JSON Object:**
   
   * **period_summary** (*object*) -- Summary for requested period
   * **daily_usage** (*array*) -- Daily usage breakdown
   * **category_breakdown** (*object*) -- Spending by category
   * **model_usage** (*object*) -- Usage by AI model
   * **trends** (*object*) -- Spending trend analysis
   * **projections** (*object*) -- Budget projection based on current usage

   **Status Codes:**
   
   * **200** -- Detailed budget retrieved successfully
   * **401** -- Unauthorized

Budget Management
----------------

Set Budget
^^^^^^^^^

.. http:post:: /budget

   Create or update budget allocation.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **total_amount** (*integer*) -- Total budget in FTNS tokens
   * **period_type** (*string*) -- Budget period ("daily", "weekly", "monthly")
   * **categories** (*object*, optional) -- Budget allocation by category
   * **alerts** (*object*, optional) -- Budget alert configuration
   * **auto_renewal** (*boolean*, optional) -- Auto-renew budget (default: false)

   **Category Allocation:**
   
   * **ai_models** (*integer*) -- Budget for AI model usage
   * **storage** (*integer*) -- Budget for IPFS storage
   * **governance** (*integer*) -- Budget for governance participation
   * **marketplace** (*integer*) -- Budget for marketplace transactions

   **Alert Configuration:**
   
   * **threshold_50** (*boolean*) -- Alert at 50% usage
   * **threshold_80** (*boolean*) -- Alert at 80% usage
   * **threshold_95** (*boolean*) -- Alert at 95% usage
   * **daily_limit** (*integer*, optional) -- Daily spending limit

   **Response JSON Object:**
   
   * **budget_id** (*string*) -- Budget configuration identifier
   * **status** (*string*) -- Budget status
   * **effective_date** (*string*) -- When budget takes effect
   * **next_renewal** (*string*, optional) -- Next renewal date

   **Status Codes:**
   
   * **201** -- Budget created successfully
   * **200** -- Budget updated successfully
   * **400** -- Invalid budget configuration
   * **401** -- Unauthorized

Update Budget
^^^^^^^^^^^^

.. http:put:: /budget/{budget_id}

   Update existing budget configuration.

   **Path Parameters:**
   
   * **budget_id** (*string*) -- Budget identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   Same as Set Budget endpoint

   **Status Codes:**
   
   * **200** -- Budget updated successfully
   * **400** -- Invalid budget data
   * **401** -- Unauthorized
   * **404** -- Budget not found

Delete Budget
^^^^^^^^^^^^

.. http:delete:: /budget/{budget_id}

   Delete a budget configuration.

   **Path Parameters:**
   
   * **budget_id** (*string*) -- Budget identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Status Codes:**
   
   * **200** -- Budget deleted successfully
   * **401** -- Unauthorized
   * **404** -- Budget not found

Usage Tracking
-------------

Current Usage
^^^^^^^^^^^^

.. http:get:: /budget/usage

   Get current period usage statistics.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **current_period** (*object*) -- Current period usage
   * **today_usage** (*object*) -- Today's usage
   * **recent_transactions** (*array*) -- Recent spending transactions
   * **usage_by_hour** (*array*) -- Hourly usage breakdown for today

   **Usage Object:**
   
   * **total_spent** (*integer*) -- Total FTNS tokens spent
   * **transaction_count** (*integer*) -- Number of transactions
   * **average_cost** (*float*) -- Average cost per transaction
   * **peak_usage_hour** (*integer*, optional) -- Hour with highest usage

   **Status Codes:**
   
   * **200** -- Usage data retrieved successfully
   * **401** -- Unauthorized

Usage History
^^^^^^^^^^^^

.. http:get:: /budget/usage/history

   Get historical usage data.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **start_date** (*string*, optional) -- Start date (ISO 8601)
   * **end_date** (*string*, optional) -- End date (ISO 8601)
   * **granularity** (*string*, optional) -- Data granularity ("hour", "day", "week", default: "day")
   * **category** (*string*, optional) -- Filter by category

   **Response JSON Object:**
   
   * **period** (*object*) -- Queried period information
   * **usage_data** (*array*) -- Historical usage data points
   * **summary** (*object*) -- Period summary statistics
   * **trends** (*object*) -- Usage trend analysis

   **Status Codes:**
   
   * **200** -- Usage history retrieved successfully
   * **400** -- Invalid date range
   * **401** -- Unauthorized

Transaction Details
^^^^^^^^^^^^^^^^^^

.. http:get:: /budget/transactions

   Get detailed transaction history.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **page** (*integer*, optional) -- Page number (default: 1)
   * **limit** (*integer*, optional) -- Results per page (default: 50, max: 200)
   * **category** (*string*, optional) -- Filter by category
   * **model** (*string*, optional) -- Filter by AI model
   * **start_date** (*string*, optional) -- Start date filter
   * **end_date** (*string*, optional) -- End date filter

   **Response JSON Object:**
   
   * **transactions** (*array*) -- Transaction list
   * **total** (*integer*) -- Total number of transactions
   * **page** (*integer*) -- Current page
   * **pages** (*integer*) -- Total pages
   * **summary** (*object*) -- Period summary

   **Transaction Object:**
   
   * **transaction_id** (*string*) -- Unique transaction identifier
   * **timestamp** (*string*) -- Transaction timestamp
   * **category** (*string*) -- Transaction category
   * **description** (*string*) -- Transaction description
   * **cost** (*integer*) -- Cost in FTNS tokens
   * **model_used** (*string*, optional) -- AI model identifier
   * **tokens_processed** (*integer*, optional) -- Number of tokens processed
   * **request_id** (*string*, optional) -- Related request identifier

   **Status Codes:**
   
   * **200** -- Transactions retrieved successfully
   * **401** -- Unauthorized

Cost Estimation
--------------

Estimate Cost
^^^^^^^^^^^^

.. http:post:: /budget/estimate

   Estimate cost for a planned operation.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **operation_type** (*string*) -- Type of operation ("ai_query", "storage", "governance")
   * **parameters** (*object*) -- Operation-specific parameters

   **AI Query Parameters:**
   
   * **model_id** (*string*) -- AI model identifier
   * **prompt_length** (*integer*) -- Estimated prompt length in tokens
   * **max_tokens** (*integer*) -- Maximum response tokens
   * **quantity** (*integer*, optional) -- Number of queries (default: 1)

   **Storage Parameters:**
   
   * **file_size** (*integer*) -- File size in bytes
   * **storage_duration** (*integer*) -- Storage duration in days

   **Response JSON Object:**
   
   * **estimated_cost** (*object*) -- Cost estimation
   * **breakdown** (*object*) -- Cost breakdown by component
   * **comparison** (*object*, optional) -- Cost comparison with alternatives

   **Cost Estimation:**
   
   * **ftns_tokens** (*integer*) -- Estimated cost in FTNS tokens
   * **usd_equivalent** (*float*) -- USD equivalent
   * **confidence** (*float*) -- Estimation confidence (0.0-1.0)

   **Status Codes:**
   
   * **200** -- Cost estimated successfully
   * **400** -- Invalid estimation parameters
   * **401** -- Unauthorized

Bulk Estimation
^^^^^^^^^^^^^^

.. http:post:: /budget/estimate/bulk

   Estimate costs for multiple operations.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **operations** (*array*) -- List of operations to estimate

   **Response JSON Object:**
   
   * **estimations** (*array*) -- Individual cost estimations
   * **total_cost** (*object*) -- Total estimated cost
   * **bulk_discounts** (*object*, optional) -- Available bulk discounts

   **Status Codes:**
   
   * **200** -- Bulk estimation completed successfully
   * **400** -- Invalid operations
   * **401** -- Unauthorized

Budget Alerts
------------

Alert Configuration
^^^^^^^^^^^^^^^^^^

.. http:get:: /budget/alerts

   Get current budget alert configuration.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **active_alerts** (*array*) -- Currently active alerts
   * **alert_rules** (*array*) -- Configured alert rules
   * **notification_preferences** (*object*) -- Notification settings

   **Alert Rule Object:**
   
   * **rule_id** (*string*) -- Alert rule identifier
   * **trigger_type** (*string*) -- Trigger condition ("threshold", "daily_limit", "anomaly")
   * **threshold** (*float*) -- Threshold value
   * **enabled** (*boolean*) -- Rule status
   * **notification_methods** (*array*) -- Notification methods

   **Status Codes:**
   
   * **200** -- Alert configuration retrieved successfully
   * **401** -- Unauthorized

Create Alert
^^^^^^^^^^^

.. http:post:: /budget/alerts

   Create a new budget alert rule.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **name** (*string*) -- Alert rule name
   * **trigger_type** (*string*) -- Trigger condition
   * **threshold** (*float*) -- Threshold value (percentage for budget alerts)
   * **category** (*string*, optional) -- Specific category to monitor
   * **notification_methods** (*array*) -- Notification methods ("email", "webhook", "in_app")
   * **enabled** (*boolean*, optional) -- Enable rule (default: true)

   **Response JSON Object:**
   
   * **rule_id** (*string*) -- Created alert rule identifier
   * **status** (*string*) -- Rule creation status

   **Status Codes:**
   
   * **201** -- Alert rule created successfully
   * **400** -- Invalid alert configuration
   * **401** -- Unauthorized

Update Alert
^^^^^^^^^^^

.. http:put:: /budget/alerts/{rule_id}

   Update an existing alert rule.

   **Path Parameters:**
   
   * **rule_id** (*string*) -- Alert rule identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   Same as Create Alert endpoint

   **Status Codes:**
   
   * **200** -- Alert rule updated successfully
   * **400** -- Invalid alert data
   * **401** -- Unauthorized
   * **404** -- Alert rule not found

Budget Analytics
---------------

Spending Analysis
^^^^^^^^^^^^^^^^

.. http:get:: /budget/analytics/spending

   Get comprehensive spending analysis.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **period** (*string*, optional) -- Analysis period ("week", "month", "quarter", default: "month")

   **Response JSON Object:**
   
   * **spending_trends** (*object*) -- Spending trend analysis
   * **category_distribution** (*object*) -- Spending distribution by category
   * **model_efficiency** (*object*) -- Cost efficiency by model
   * **peak_usage_times** (*array*) -- Times of highest usage
   * **recommendations** (*array*) -- Cost optimization recommendations

   **Status Codes:**
   
   * **200** -- Spending analysis retrieved successfully
   * **401** -- Unauthorized

Cost Optimization
^^^^^^^^^^^^^^^^

.. http:get:: /budget/analytics/optimization

   Get cost optimization recommendations.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **potential_savings** (*object*) -- Potential cost savings
   * **optimization_opportunities** (*array*) -- Specific optimization recommendations
   * **alternative_models** (*object*) -- Suggestions for more cost-effective models
   * **usage_patterns** (*object*) -- Analysis of usage patterns

   **Optimization Opportunity:**
   
   * **type** (*string*) -- Optimization type
   * **description** (*string*) -- Detailed description
   * **potential_savings** (*integer*) -- Potential FTNS token savings
   * **implementation_effort** (*string*) -- Implementation difficulty

   **Status Codes:**
   
   * **200** -- Optimization recommendations retrieved successfully
   * **401** -- Unauthorized

Team Budget Management
---------------------

Team Budgets
^^^^^^^^^^^

.. http:get:: /budget/team

   Get team budget information (team admin only).

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **team_budget** (*object*) -- Total team budget
   * **member_allocations** (*array*) -- Budget per team member
   * **team_usage** (*object*) -- Team-wide usage statistics
   * **shared_resources** (*object*) -- Shared resource usage

   **Status Codes:**
   
   * **200** -- Team budget retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Allocate Budget
^^^^^^^^^^^^^^

.. http:post:: /budget/team/allocate

   Allocate budget to team members.

   **Request Headers:**
   
   * **Authorization** -- Bearer token (team admin role required)

   **Request JSON Object:**
   
   * **allocations** (*array*) -- Budget allocations for team members

   **Allocation Object:**
   
   * **user_id** (*string*) -- Team member identifier
   * **amount** (*integer*) -- Allocated FTNS tokens
   * **restrictions** (*object*, optional) -- Usage restrictions

   **Status Codes:**
   
   * **200** -- Budget allocated successfully
   * **400** -- Invalid allocation
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Error Handling
-------------

Budget API error responses:

.. code-block:: json

   {
     "error": "insufficient_budget",
     "message": "Insufficient budget for requested operation",
     "code": 402,
     "required_amount": 100,
     "available_amount": 50,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error types:

* ``insufficient_budget`` - Not enough budget for operation
* ``budget_not_found`` - Budget configuration doesn't exist
* ``invalid_budget_period`` - Invalid budget period specification
* ``alert_limit_exceeded`` - Too many alert rules configured
* ``budget_locked`` - Budget is locked and cannot be modified

Rate Limits
-----------

Budget API rate limits:

* Budget queries: 200 requests per minute
* Budget updates: 50 requests per hour
* Usage tracking: 1000 requests per hour
* Cost estimation: 100 requests per minute
* Analytics: 50 requests per minute

Security Features
----------------

* All budget operations require authentication
* Team budget management requires admin privileges
* Budget modifications are logged and audited
* Real-time fraud detection for unusual spending patterns
* Automatic budget locks for suspected unauthorized usage