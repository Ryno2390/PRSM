Session API
===========

The Session API provides comprehensive session management for PRSM research workflows, including session creation, monitoring, and result retrieval.

.. automodule:: prsm.api.session_api
   :members:
   :undoc-members:
   :show-inheritance:

Session Management Endpoints
----------------------------

Create Session
^^^^^^^^^^^^^^

.. http:post:: /api/v1/sessions

   Create a new PRSM research session.

   **Request JSON Object:**
   
   * **prompt** (*string*) -- Research query or prompt (required)
   * **session_type** (*string*) -- Type of session (research, analysis, synthesis)
   * **budget** (*object*, optional) -- Session budget configuration
   * **preferences** (*object*, optional) -- User preferences and settings
   * **metadata** (*object*, optional) -- Additional session metadata

   **Response JSON Object:**
   
   * **session_id** (*string*) -- Unique session identifier
   * **status** (*string*) -- Session status
   * **estimated_duration** (*integer*) -- Estimated completion time in minutes
   * **cost_estimate** (*object*) -- Estimated session costs
   * **created_at** (*string*) -- Session creation timestamp

   **Status Codes:**
   
   * **201** -- Session created successfully
   * **400** -- Invalid session data
   * **401** -- Unauthorized
   * **402** -- Insufficient balance for session

Get Session Status
^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/sessions/{session_id}

   Retrieve detailed session status and progress information.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Response JSON Object:**
   
   * **session_id** (*string*) -- Session identifier
   * **status** (*string*) -- Current session status
   * **progress** (*integer*) -- Completion percentage (0-100)
   * **current_stage** (*string*) -- Current processing stage
   * **results** (*object*, optional) -- Session results (if completed)
   * **error_details** (*object*, optional) -- Error information (if failed)
   * **resource_usage** (*object*) -- Resource consumption metrics

   **Status Codes:**
   
   * **200** -- Session status retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Session not found

List User Sessions
^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/sessions

   Retrieve paginated list of user's sessions with filtering options.

   **Query Parameters:**
   
   * **status** (*string*, optional) -- Filter by session status
   * **session_type** (*string*, optional) -- Filter by session type
   * **date_from** (*string*, optional) -- Start date filter
   * **date_to** (*string*, optional) -- End date filter
   * **page** (*integer*, optional) -- Page number
   * **limit** (*integer*, optional) -- Items per page

   **Response JSON Object:**
   
   * **sessions** (*array*) -- List of session objects
   * **total_count** (*integer*) -- Total number of sessions
   * **page** (*integer*) -- Current page number
   * **summary** (*object*) -- Session summary statistics

   **Status Codes:**
   
   * **200** -- Sessions retrieved successfully
   * **401** -- Unauthorized

Session Control Endpoints
-------------------------

Cancel Session
^^^^^^^^^^^^^^

.. http:post:: /api/v1/sessions/{session_id}/cancel

   Cancel a running or queued session.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Request JSON Object:**
   
   * **reason** (*string*, optional) -- Cancellation reason

   **Response JSON Object:**
   
   * **session_id** (*string*) -- Session identifier
   * **status** (*string*) -- Updated session status
   * **refund_amount** (*decimal*, optional) -- Refunded amount for unused resources

   **Status Codes:**
   
   * **200** -- Session cancelled successfully
   * **400** -- Session cannot be cancelled
   * **401** -- Unauthorized
   * **404** -- Session not found

Pause Session
^^^^^^^^^^^^^

.. http:post:: /api/v1/sessions/{session_id}/pause

   Pause a running session (if supported by session type).

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Status Codes:**
   
   * **200** -- Session paused successfully
   * **400** -- Session cannot be paused
   * **401** -- Unauthorized
   * **404** -- Session not found

Resume Session
^^^^^^^^^^^^^^

.. http:post:: /api/v1/sessions/{session_id}/resume

   Resume a paused session.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Status Codes:**
   
   * **200** -- Session resumed successfully
   * **400** -- Session cannot be resumed
   * **401** -- Unauthorized
   * **404** -- Session not found

Session Results and Output
--------------------------

Get Session Results
^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/sessions/{session_id}/results

   Retrieve comprehensive session results and outputs.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Query Parameters:**
   
   * **format** (*string*, optional) -- Response format (json, markdown, pdf)
   * **include_metadata** (*boolean*, optional) -- Include execution metadata

   **Response JSON Object:**
   
   * **session_id** (*string*) -- Session identifier
   * **results** (*object*) -- Structured session results
   * **summary** (*string*) -- Executive summary of findings
   * **citations** (*array*) -- Source citations and references
   * **confidence_score** (*float*) -- Result confidence score
   * **execution_metadata** (*object*) -- Execution details and metrics

   **Status Codes:**
   
   * **200** -- Results retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Session not found or no results available

Export Session Results
^^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/sessions/{session_id}/export

   Export session results in various formats.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Query Parameters:**
   
   * **format** (*string*) -- Export format (pdf, docx, markdown, json)
   * **include_raw_data** (*boolean*, optional) -- Include raw processing data

   **Response:**
   
   File download in specified format

   **Status Codes:**
   
   * **200** -- Export downloaded successfully
   * **400** -- Invalid export format
   * **401** -- Unauthorized
   * **404** -- Session not found

Session Analytics
-----------------

Get Session Metrics
^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/sessions/{session_id}/metrics

   Retrieve detailed session performance and resource usage metrics.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Response JSON Object:**
   
   * **execution_time** (*integer*) -- Total execution time in seconds
   * **resource_usage** (*object*) -- Detailed resource consumption
   * **cost_breakdown** (*object*) -- Itemized cost analysis
   * **performance_metrics** (*object*) -- Performance indicators
   * **efficiency_score** (*float*) -- Session efficiency rating

   **Status Codes:**
   
   * **200** -- Metrics retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Session not found

Session History
^^^^^^^^^^^^^^^

.. http:get:: /api/v1/sessions/{session_id}/history

   Retrieve detailed session execution history and timeline.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Response JSON Object:**
   
   * **session_id** (*string*) -- Session identifier
   * **timeline** (*array*) -- Chronological session events
   * **stage_transitions** (*array*) -- Processing stage changes
   * **resource_allocations** (*array*) -- Resource allocation events
   * **decision_points** (*array*) -- Key decision points in processing

   **Status Codes:**
   
   * **200** -- History retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Session not found

Session Collaboration
---------------------

Share Session
^^^^^^^^^^^^^

.. http:post:: /api/v1/sessions/{session_id}/share

   Share session with other users or teams.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Request JSON Object:**
   
   * **recipients** (*array*) -- List of user IDs or email addresses
   * **permissions** (*string*) -- Shared access level (view, edit, collaborate)
   * **message** (*string*, optional) -- Share message
   * **expiration** (*string*, optional) -- Share expiration date

   **Response JSON Object:**
   
   * **share_id** (*string*) -- Unique share identifier
   * **share_url** (*string*) -- Shareable URL
   * **recipients** (*array*) -- Confirmed recipients
   * **expires_at** (*string*) -- Share expiration timestamp

   **Status Codes:**
   
   * **201** -- Session shared successfully
   * **400** -- Invalid share configuration
   * **401** -- Unauthorized
   * **404** -- Session not found

Add Session Comment
^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/sessions/{session_id}/comments

   Add comment or annotation to session.

   **Path Parameters:**
   
   * **session_id** (*string*) -- Unique session identifier

   **Request JSON Object:**
   
   * **content** (*string*) -- Comment content (required)
   * **comment_type** (*string*, optional) -- Comment type (note, question, insight)
   * **reference_point** (*string*, optional) -- Specific reference within session

   **Response JSON Object:**
   
   * **comment_id** (*string*) -- Unique comment identifier
   * **content** (*string*) -- Comment content
   * **author** (*object*) -- Comment author information
   * **created_at** (*string*) -- Comment creation timestamp

   **Status Codes:**
   
   * **201** -- Comment added successfully
   * **400** -- Invalid comment data
   * **401** -- Unauthorized
   * **404** -- Session not found

Session Types
------------

Research Sessions
^^^^^^^^^^^^^^^^^

Comprehensive research and analysis sessions:

* Literature review and synthesis
* Data analysis and visualization
* Hypothesis generation and testing
* Multi-source information integration

Analysis Sessions
^^^^^^^^^^^^^^^^^

Focused analytical processing:

* Statistical analysis and modeling
* Pattern recognition and insights
* Comparative analysis
* Trend identification

Synthesis Sessions
^^^^^^^^^^^^^^^^^^

Knowledge synthesis and consolidation:

* Multi-session result integration
* Cross-domain knowledge bridging
* Executive summary generation
* Report compilation and formatting

Session Status Values
--------------------

* **queued** - Session waiting to start
* **initializing** - Session setup in progress
* **running** - Session actively processing
* **paused** - Session temporarily paused
* **completed** - Session finished successfully
* **failed** - Session encountered error
* **cancelled** - Session cancelled by user
* **timeout** - Session exceeded time limits

Error Responses
--------------

Session API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "session_not_found",
     "message": "The specified session does not exist",
     "code": 404,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error codes:

* ``session_not_found`` - Session does not exist
* ``session_not_accessible`` - User lacks session access
* ``session_cannot_be_cancelled`` - Session in non-cancellable state
* ``insufficient_balance`` - Insufficient balance for session
* ``session_quota_exceeded`` - User session limit reached