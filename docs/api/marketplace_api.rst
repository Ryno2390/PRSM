Marketplace API
===============

The Marketplace API provides endpoints for accessing and interacting with the PRSM AI model marketplace.

.. automodule:: prsm.api.marketplace_api
   :members:
   :undoc-members:
   :show-inheritance:

Model Discovery
--------------

List Models
^^^^^^^^^^^

.. http:get:: /marketplace/models

   Retrieve a list of available AI models in the marketplace.

   **Query Parameters:**
   
   * **category** (*string*, optional) -- Filter by model category (e.g., "nlp", "vision", "code")
   * **provider** (*string*, optional) -- Filter by model provider (e.g., "openai", "anthropic")
   * **tier** (*string*, optional) -- Filter by pricing tier ("free", "basic", "premium", "ultra")
   * **max_cost** (*float*, optional) -- Maximum cost per 1K tokens
   * **capabilities** (*string*, optional) -- Comma-separated list of required capabilities
   * **page** (*integer*, optional) -- Page number for pagination (default: 1)
   * **limit** (*integer*, optional) -- Number of results per page (default: 20, max: 100)

   **Response JSON Object:**
   
   * **models** (*array*) -- List of model objects
   * **total** (*integer*) -- Total number of models matching criteria
   * **page** (*integer*) -- Current page number
   * **pages** (*integer*) -- Total number of pages

   **Model Object:**
   
   * **id** (*string*) -- Unique model identifier
   * **name** (*string*) -- Human-readable model name
   * **provider** (*string*) -- Model provider name
   * **category** (*string*) -- Model category
   * **description** (*string*) -- Model description
   * **pricing** (*object*) -- Pricing information
   * **capabilities** (*array*) -- List of model capabilities
   * **performance** (*object*) -- Performance metrics
   * **availability** (*string*) -- Model availability status

   **Status Codes:**
   
   * **200** -- Models retrieved successfully
   * **400** -- Invalid query parameters

Model Details
^^^^^^^^^^^^

.. http:get:: /marketplace/models/{model_id}

   Get detailed information about a specific model.

   **Path Parameters:**
   
   * **model_id** (*string*) -- Unique model identifier

   **Response JSON Object:**
   
   * **id** (*string*) -- Model identifier
   * **name** (*string*) -- Model name
   * **provider** (*string*) -- Provider name
   * **description** (*string*) -- Detailed description
   * **pricing** (*object*) -- Detailed pricing structure
   * **capabilities** (*object*) -- Detailed capabilities
   * **documentation** (*string*) -- Documentation URL
   * **examples** (*array*) -- Usage examples
   * **performance_metrics** (*object*) -- Performance benchmarks
   * **terms_of_use** (*string*) -- Terms and conditions

   **Status Codes:**
   
   * **200** -- Model details retrieved successfully
   * **404** -- Model not found

Model Execution
--------------

Execute Model
^^^^^^^^^^^^

.. http:post:: /marketplace/models/{model_id}/execute

   Execute a model with given input parameters.

   **Path Parameters:**
   
   * **model_id** (*string*) -- Model identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token
   * **Content-Type** -- application/json

   **Request JSON Object:**
   
   * **prompt** (*string*) -- Input prompt for the model
   * **max_tokens** (*integer*, optional) -- Maximum tokens to generate (default: 150)
   * **temperature** (*float*, optional) -- Sampling temperature (0.0-2.0, default: 0.7)
   * **top_p** (*float*, optional) -- Nucleus sampling parameter (default: 1.0)
   * **system_prompt** (*string*, optional) -- System prompt for instruction
   * **tools** (*array*, optional) -- Available tools for the model
   * **stream** (*boolean*, optional) -- Enable streaming response (default: false)

   **Response JSON Object:**
   
   * **response** (*string*) -- Model's generated response
   * **model_id** (*string*) -- Model used for execution
   * **usage** (*object*) -- Token usage statistics
   * **cost** (*object*) -- Cost breakdown
   * **execution_time** (*float*) -- Execution time in seconds
   * **request_id** (*string*) -- Unique request identifier

   **Status Codes:**
   
   * **200** -- Model executed successfully
   * **400** -- Invalid request parameters
   * **401** -- Unauthorized
   * **402** -- Insufficient credits
   * **429** -- Rate limit exceeded

Streaming Execution
^^^^^^^^^^^^^^^^^^

.. http:post:: /marketplace/models/{model_id}/stream

   Execute a model with streaming response.

   **Path Parameters:**
   
   * **model_id** (*string*) -- Model identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token
   * **Content-Type** -- application/json

   **Request JSON Object:**
   Same as execute endpoint with ``stream: true``

   **Response:**
   Server-Sent Events (SSE) stream with incremental response chunks.

   **Event Types:**
   
   * **data** -- Response chunk
   * **usage** -- Token usage update
   * **done** -- Execution complete

   **Status Codes:**
   
   * **200** -- Stream established successfully
   * **400** -- Invalid request parameters
   * **401** -- Unauthorized

Model Comparison
---------------

Compare Models
^^^^^^^^^^^^^

.. http:post:: /marketplace/compare

   Compare multiple models for the same input.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **models** (*array*) -- List of model IDs to compare
   * **prompt** (*string*) -- Input prompt for comparison
   * **parameters** (*object*, optional) -- Common parameters for all models

   **Response JSON Object:**
   
   * **comparisons** (*array*) -- Array of model results
   * **summary** (*object*) -- Comparison summary
   * **recommendations** (*array*) -- Recommended models

   **Status Codes:**
   
   * **200** -- Comparison completed successfully
   * **400** -- Invalid request parameters

Model Recommendations
-------------------

Get Recommendations
^^^^^^^^^^^^^^^^^^

.. http:get:: /marketplace/recommendations

   Get personalized model recommendations based on usage patterns.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **task_type** (*string*, optional) -- Type of task (e.g., "text_generation", "code", "analysis")
   * **budget** (*float*, optional) -- Budget constraint
   * **quality_preference** (*string*, optional) -- Quality preference ("speed", "balanced", "quality")

   **Response JSON Object:**
   
   * **recommendations** (*array*) -- Recommended models
   * **reasoning** (*string*) -- Explanation for recommendations
   * **alternatives** (*array*) -- Alternative options

   **Status Codes:**
   
   * **200** -- Recommendations generated successfully
   * **401** -- Unauthorized

Model Analytics
--------------

Usage Statistics
^^^^^^^^^^^^^^^

.. http:get:: /marketplace/analytics/usage

   Get usage analytics for marketplace models.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Query Parameters:**
   
   * **period** (*string*, optional) -- Time period ("day", "week", "month", default: "week")
   * **model_id** (*string*, optional) -- Specific model ID

   **Response JSON Object:**
   
   * **total_requests** (*integer*) -- Total number of requests
   * **total_cost** (*float*) -- Total cost in FTNS tokens
   * **average_response_time** (*float*) -- Average response time
   * **popular_models** (*array*) -- Most used models
   * **cost_breakdown** (*object*) -- Cost by model/provider

   **Status Codes:**
   
   * **200** -- Analytics retrieved successfully
   * **401** -- Unauthorized

Performance Metrics
^^^^^^^^^^^^^^^^^^

.. http:get:: /marketplace/analytics/performance

   Get performance metrics for marketplace models.

   **Query Parameters:**
   
   * **model_id** (*string*, optional) -- Specific model ID
   * **metric** (*string*, optional) -- Specific metric ("latency", "quality", "cost")

   **Response JSON Object:**
   
   * **metrics** (*object*) -- Performance metrics by model
   * **benchmarks** (*array*) -- Benchmark results
   * **trends** (*object*) -- Performance trends over time

   **Status Codes:**
   
   * **200** -- Performance metrics retrieved successfully

Model Management
---------------

Add Custom Model
^^^^^^^^^^^^^^^

.. http:post:: /marketplace/models

   Add a custom model to the marketplace (provider only).

   **Request Headers:**
   
   * **Authorization** -- Bearer token (provider role required)

   **Request JSON Object:**
   
   * **name** (*string*) -- Model name
   * **description** (*string*) -- Model description
   * **category** (*string*) -- Model category
   * **pricing** (*object*) -- Pricing structure
   * **capabilities** (*array*) -- Model capabilities
   * **endpoint_url** (*string*) -- Model API endpoint
   * **documentation_url** (*string*, optional) -- Documentation link

   **Status Codes:**
   
   * **201** -- Model added successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Update Model
^^^^^^^^^^^

.. http:put:: /marketplace/models/{model_id}

   Update model information (provider only).

   **Path Parameters:**
   
   * **model_id** (*string*) -- Model identifier

   **Status Codes:**
   
   * **200** -- Model updated successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions
   * **404** -- Model not found

Error Handling
-------------

The Marketplace API uses standard HTTP status codes and returns detailed error information:

.. code-block:: json

   {
     "error": "model_not_found",
     "message": "The specified model does not exist or is not available",
     "code": 404,
     "model_id": "gpt-4-turbo",
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error types:

* ``model_not_found`` - Requested model doesn't exist
* ``insufficient_credits`` - Not enough FTNS tokens for execution
* ``model_unavailable`` - Model temporarily unavailable
* ``rate_limit_exceeded`` - Too many requests
* ``invalid_parameters`` - Invalid request parameters
* ``execution_failed`` - Model execution error

Rate Limits
-----------

Marketplace API endpoints have the following rate limits:

* Model listing: 100 requests per minute
* Model execution: 1000 requests per hour per user
* Streaming: 10 concurrent streams per user
* Analytics: 50 requests per minute

FTNS Token Integration
---------------------

All model executions are paid for using FTNS tokens:

* Costs are calculated based on token usage and model pricing
* Prepaid credits system with automatic deduction
* Real-time balance checking before execution
* Detailed cost breakdowns in responses