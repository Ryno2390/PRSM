Credential API
==============

The Credential API provides secure management of API keys, authentication credentials, and access tokens for external service integrations.

.. automodule:: prsm.api.credential_api
   :members:
   :undoc-members:
   :show-inheritance:

Credential Management Endpoints
-------------------------------

Create Credential
^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/credentials

   Create a new credential for external service integration.

   **Request JSON Object:**
   
   * **name** (*string*) -- Credential name (required)
   * **description** (*string*) -- Credential description
   * **credential_type** (*string*) -- Type of credential (api_key, oauth, certificate)
   * **service_provider** (*string*) -- Target service provider
   * **credential_data** (*object*) -- Encrypted credential data
   * **scope** (*array*, optional) -- Access scope permissions

   **Response JSON Object:**
   
   * **credential_id** (*string*) -- Unique credential identifier
   * **name** (*string*) -- Credential name
   * **status** (*string*) -- Credential status
   * **created_at** (*string*) -- Creation timestamp

   **Status Codes:**
   
   * **201** -- Credential created successfully
   * **400** -- Invalid request data
   * **401** -- Unauthorized
   * **409** -- Credential name already exists

List Credentials
^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/credentials

   Retrieve paginated list of user's credentials.

   **Query Parameters:**
   
   * **page** (*integer*, optional) -- Page number (default: 1)
   * **limit** (*integer*, optional) -- Items per page (default: 20)
   * **credential_type** (*string*, optional) -- Filter by credential type
   * **service_provider** (*string*, optional) -- Filter by service provider

   **Response JSON Object:**
   
   * **credentials** (*array*) -- List of credential objects (data masked)
   * **total_count** (*integer*) -- Total number of credentials
   * **page** (*integer*) -- Current page number

   **Status Codes:**
   
   * **200** -- Credentials retrieved successfully
   * **401** -- Unauthorized

Get Credential
^^^^^^^^^^^^^^

.. http:get:: /api/v1/credentials/{credential_id}

   Retrieve specific credential details (sensitive data masked).

   **Path Parameters:**
   
   * **credential_id** (*string*) -- Unique credential identifier

   **Response JSON Object:**
   
   * **credential_id** (*string*) -- Credential identifier
   * **name** (*string*) -- Credential name
   * **credential_type** (*string*) -- Credential type
   * **service_provider** (*string*) -- Service provider
   * **status** (*string*) -- Credential status
   * **last_used** (*string*) -- Last usage timestamp

   **Status Codes:**
   
   * **200** -- Credential retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Credential not found

Update Credential
^^^^^^^^^^^^^^^^^

.. http:put:: /api/v1/credentials/{credential_id}

   Update credential configuration or rotate credential data.

   **Path Parameters:**
   
   * **credential_id** (*string*) -- Unique credential identifier

   **Request JSON Object:**
   
   * **name** (*string*, optional) -- Updated credential name
   * **description** (*string*, optional) -- Updated description
   * **credential_data** (*object*, optional) -- Updated credential data
   * **scope** (*array*, optional) -- Updated access scope

   **Status Codes:**
   
   * **200** -- Credential updated successfully
   * **400** -- Invalid request data
   * **401** -- Unauthorized
   * **404** -- Credential not found

Delete Credential
^^^^^^^^^^^^^^^^^

.. http:delete:: /api/v1/credentials/{credential_id}

   Delete a credential (irreversible).

   **Path Parameters:**
   
   * **credential_id** (*string*) -- Unique credential identifier

   **Status Codes:**
   
   * **204** -- Credential deleted successfully
   * **401** -- Unauthorized
   * **404** -- Credential not found

Credential Validation Endpoints
-------------------------------

Test Credential
^^^^^^^^^^^^^^^

.. http:post:: /api/v1/credentials/{credential_id}/test

   Test credential validity with target service.

   **Path Parameters:**
   
   * **credential_id** (*string*) -- Unique credential identifier

   **Response JSON Object:**
   
   * **valid** (*boolean*) -- Whether credential is valid
   * **test_timestamp** (*string*) -- Test execution timestamp
   * **error_message** (*string*, optional) -- Error details if test failed

   **Status Codes:**
   
   * **200** -- Test completed
   * **401** -- Unauthorized
   * **404** -- Credential not found

Refresh Credential
^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/credentials/{credential_id}/refresh

   Refresh OAuth tokens or renew time-limited credentials.

   **Path Parameters:**
   
   * **credential_id** (*string*) -- Unique credential identifier

   **Status Codes:**
   
   * **200** -- Credential refreshed successfully
   * **400** -- Credential type doesn't support refresh
   * **401** -- Unauthorized
   * **404** -- Credential not found

Credential Types
---------------

API Key Credentials
^^^^^^^^^^^^^^^^^^^

For services that use API key authentication:

* **Structure**: Simple key-value pairs
* **Security**: Encrypted at rest, masked in responses
* **Usage**: Injected into HTTP headers or query parameters
* **Rotation**: Manual rotation supported

OAuth Credentials
^^^^^^^^^^^^^^^^^

For OAuth 2.0 authenticated services:

* **Structure**: Access token, refresh token, expiration
* **Security**: Automatic token refresh, encrypted storage
* **Usage**: Bearer token in Authorization header
* **Rotation**: Automatic refresh before expiration

Certificate Credentials
^^^^^^^^^^^^^^^^^^^^^^^

For certificate-based authentication:

* **Structure**: Certificate, private key, chain
* **Security**: PKI-based encryption, secure key storage
* **Usage**: Mutual TLS authentication
* **Rotation**: Certificate renewal workflows

Database Credentials
^^^^^^^^^^^^^^^^^^^^

For database connection authentication:

* **Structure**: Username, password, connection string
* **Security**: Connection pooling, encrypted storage
* **Usage**: Database connection establishment
* **Rotation**: Coordinated credential rotation

Security Features
----------------

Encryption at Rest
^^^^^^^^^^^^^^^^^^

All credential data is encrypted using AES-256-GCM:

* Individual credential encryption keys
* Master key rotation capabilities
* Hardware security module (HSM) integration
* Zero-knowledge architecture

Access Control
^^^^^^^^^^^^^^

Comprehensive access control mechanisms:

* Role-based access control (RBAC)
* Resource-level permissions
* Audit logging for all operations
* Multi-factor authentication for sensitive operations

Credential Monitoring
^^^^^^^^^^^^^^^^^^^^^

Real-time monitoring and alerting:

* Usage tracking and analytics
* Anomaly detection for unusual access patterns
* Credential expiration notifications
* Security breach detection and response

Error Responses
--------------

Credential API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "credential_not_found",
     "message": "The specified credential does not exist",
     "code": 404,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error codes:

* ``credential_not_found`` - Credential does not exist
* ``invalid_credential_type`` - Unsupported credential type
* ``credential_test_failed`` - Credential validation failed
* ``insufficient_permissions`` - User lacks credential access
* ``credential_expired`` - Credential has expired