Security Status API
===================

The Security Status API provides real-time monitoring and reporting of security events, threat detection, and system security posture across the PRSM platform.

.. automodule:: prsm.api.security_status_api
   :members:
   :undoc-members:
   :show-inheritance:

Security Monitoring Endpoints
-----------------------------

Get Security Status
^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/status

   Retrieve comprehensive security status and threat assessment.

   **Response JSON Object:**
   
   * **security_level** (*string*) -- Current security level (low, medium, high, critical)
   * **threats_detected** (*integer*) -- Number of active threats
   * **last_scan** (*string*) -- Last security scan timestamp
   * **system_health** (*object*) -- Security system component health
   * **recommendations** (*array*) -- Security improvement recommendations

   **Status Codes:**
   
   * **200** -- Security status retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Get Threat Dashboard
^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/threats

   Retrieve active threat dashboard with detailed threat intelligence.

   **Query Parameters:**
   
   * **severity** (*string*, optional) -- Filter by threat severity
   * **category** (*string*, optional) -- Filter by threat category
   * **time_range** (*string*, optional) -- Time range filter (1h, 24h, 7d, 30d)
   * **limit** (*integer*, optional) -- Maximum threats to return

   **Response JSON Object:**
   
   * **threats** (*array*) -- List of active threats
   * **summary** (*object*) -- Threat summary statistics
   * **trends** (*object*) -- Threat trend analysis
   * **mitigation_status** (*object*) -- Mitigation progress

   **Status Codes:**
   
   * **200** -- Threat dashboard retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Security Incident Endpoints
---------------------------

Report Security Incident
^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/security/incidents

   Report a security incident for investigation.

   **Request JSON Object:**
   
   * **title** (*string*) -- Incident title (required)
   * **description** (*string*) -- Detailed incident description
   * **severity** (*string*) -- Incident severity (low, medium, high, critical)
   * **category** (*string*) -- Incident category
   * **affected_systems** (*array*) -- List of affected system components
   * **evidence** (*array*, optional) -- Supporting evidence or logs

   **Response JSON Object:**
   
   * **incident_id** (*string*) -- Unique incident identifier
   * **status** (*string*) -- Incident status
   * **assigned_to** (*string*) -- Assigned security analyst
   * **created_at** (*string*) -- Incident creation timestamp

   **Status Codes:**
   
   * **201** -- Incident reported successfully
   * **400** -- Invalid request data
   * **401** -- Unauthorized

Get Security Incidents
^^^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/incidents

   Retrieve security incidents with filtering and pagination.

   **Query Parameters:**
   
   * **status** (*string*, optional) -- Filter by incident status
   * **severity** (*string*, optional) -- Filter by severity level
   * **assigned_to** (*string*, optional) -- Filter by assigned analyst
   * **date_from** (*string*, optional) -- Start date filter
   * **date_to** (*string*, optional) -- End date filter
   * **page** (*integer*, optional) -- Page number
   * **limit** (*integer*, optional) -- Items per page

   **Response JSON Object:**
   
   * **incidents** (*array*) -- List of security incidents
   * **total_count** (*integer*) -- Total number of incidents
   * **page** (*integer*) -- Current page number
   * **summary** (*object*) -- Incident summary statistics

   **Status Codes:**
   
   * **200** -- Incidents retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Update Security Incident
^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:put:: /api/v1/security/incidents/{incident_id}

   Update security incident status and investigation details.

   **Path Parameters:**
   
   * **incident_id** (*string*) -- Unique incident identifier

   **Request JSON Object:**
   
   * **status** (*string*, optional) -- Updated incident status
   * **severity** (*string*, optional) -- Updated severity level
   * **notes** (*string*, optional) -- Investigation notes
   * **resolution** (*string*, optional) -- Incident resolution details
   * **assigned_to** (*string*, optional) -- Reassign to analyst

   **Status Codes:**
   
   * **200** -- Incident updated successfully
   * **400** -- Invalid request data
   * **401** -- Unauthorized
   * **404** -- Incident not found

Vulnerability Management
------------------------

Get Vulnerability Scan Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/vulnerabilities

   Retrieve latest vulnerability scan results and assessments.

   **Query Parameters:**
   
   * **severity** (*string*, optional) -- Filter by vulnerability severity
   * **component** (*string*, optional) -- Filter by system component
   * **status** (*string*, optional) -- Filter by remediation status
   * **scan_type** (*string*, optional) -- Filter by scan type

   **Response JSON Object:**
   
   * **vulnerabilities** (*array*) -- List of identified vulnerabilities
   * **scan_summary** (*object*) -- Scan execution summary
   * **remediation_plan** (*object*) -- Recommended remediation actions
   * **compliance_status** (*object*) -- Security compliance assessment

   **Status Codes:**
   
   * **200** -- Vulnerability data retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Trigger Security Scan
^^^^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/security/scan

   Initiate a security scan of specified system components.

   **Request JSON Object:**
   
   * **scan_type** (*string*) -- Type of scan (full, quick, targeted)
   * **components** (*array*, optional) -- Specific components to scan
   * **scan_config** (*object*, optional) -- Custom scan configuration

   **Response JSON Object:**
   
   * **scan_id** (*string*) -- Unique scan identifier
   * **status** (*string*) -- Scan status
   * **estimated_duration** (*integer*) -- Estimated completion time in minutes
   * **started_at** (*string*) -- Scan start timestamp

   **Status Codes:**
   
   * **202** -- Scan initiated successfully
   * **400** -- Invalid scan configuration
   * **401** -- Unauthorized
   * **429** -- Scan rate limit exceeded

Access Control Monitoring
-------------------------

Get Access Patterns
^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/access-patterns

   Analyze user access patterns and detect anomalies.

   **Query Parameters:**
   
   * **user_id** (*string*, optional) -- Filter by specific user
   * **time_range** (*string*, optional) -- Analysis time range
   * **anomaly_threshold** (*float*, optional) -- Anomaly detection threshold

   **Response JSON Object:**
   
   * **access_summary** (*object*) -- Access pattern summary
   * **anomalies** (*array*) -- Detected access anomalies
   * **risk_score** (*float*) -- Overall access risk score
   * **recommendations** (*array*) -- Security recommendations

   **Status Codes:**
   
   * **200** -- Access patterns retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Get Failed Authentication Attempts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/failed-auth

   Retrieve failed authentication attempts and potential brute force attacks.

   **Query Parameters:**
   
   * **ip_address** (*string*, optional) -- Filter by source IP
   * **username** (*string*, optional) -- Filter by attempted username
   * **time_range** (*string*, optional) -- Time range filter
   * **threshold** (*integer*, optional) -- Minimum attempts threshold

   **Response JSON Object:**
   
   * **failed_attempts** (*array*) -- List of failed authentication attempts
   * **suspicious_ips** (*array*) -- IPs with suspicious activity
   * **attack_patterns** (*object*) -- Identified attack patterns
   * **mitigation_actions** (*array*) -- Recommended security actions

   **Status Codes:**
   
   * **200** -- Failed auth data retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Security Metrics and Reporting
------------------------------

Get Security Metrics
^^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/security/metrics

   Retrieve comprehensive security metrics and KPIs.

   **Query Parameters:**
   
   * **metric_type** (*string*, optional) -- Filter by metric category
   * **time_range** (*string*, optional) -- Metrics time range
   * **granularity** (*string*, optional) -- Data granularity (hour, day, week)

   **Response JSON Object:**
   
   * **security_score** (*float*) -- Overall security score
   * **threat_trends** (*object*) -- Threat trend analysis
   * **incident_metrics** (*object*) -- Incident response metrics
   * **compliance_metrics** (*object*) -- Compliance status metrics
   * **performance_metrics** (*object*) -- Security system performance

   **Status Codes:**
   
   * **200** -- Security metrics retrieved successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Generate Security Report
^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/security/reports

   Generate comprehensive security report for specified time period.

   **Request JSON Object:**
   
   * **report_type** (*string*) -- Type of report (executive, technical, compliance)
   * **time_range** (*object*) -- Report time range specification
   * **include_sections** (*array*) -- Sections to include in report
   * **format** (*string*, optional) -- Report format (pdf, html, json)

   **Response JSON Object:**
   
   * **report_id** (*string*) -- Unique report identifier
   * **status** (*string*) -- Report generation status
   * **download_url** (*string*) -- Report download URL (when ready)
   * **estimated_completion** (*string*) -- Estimated completion time

   **Status Codes:**
   
   * **202** -- Report generation initiated
   * **400** -- Invalid report configuration
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Security Alert Configuration
---------------------------

Configure Security Alerts
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/security/alerts/config

   Configure security alert rules and notification preferences.

   **Request JSON Object:**
   
   * **alert_rules** (*array*) -- List of alert rule configurations
   * **notification_channels** (*array*) -- Notification delivery channels
   * **escalation_policy** (*object*) -- Alert escalation configuration
   * **suppression_rules** (*array*, optional) -- Alert suppression rules

   **Status Codes:**
   
   * **200** -- Alert configuration updated successfully
   * **400** -- Invalid configuration
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Error Responses
--------------

Security API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "insufficient_permissions",
     "message": "User lacks required security permissions",
     "code": 403,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error codes:

* ``insufficient_permissions`` - User lacks security monitoring access
* ``scan_in_progress`` - Another security scan is currently running
* ``invalid_scan_config`` - Scan configuration is invalid
* ``threat_data_unavailable`` - Threat intelligence data temporarily unavailable
* ``incident_not_found`` - Specified security incident does not exist