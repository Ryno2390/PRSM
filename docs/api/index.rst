.. PRSM API Documentation documentation master file, created by
   sphinx-quickstart on Wed Jul  2 10:57:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PRSM API Documentation
======================

Complete API documentation for the Protocol for Recursive Scientific Modeling (PRSM).
This documentation covers all API endpoints, authentication, and integration patterns.

Overview
--------

The PRSM API provides a comprehensive RESTful interface for:

* User authentication and authorization
* Research session management
* Marketplace interactions
* Team collaboration
* Payment processing and FTNS token management
* Security monitoring and compliance
* System administration and monitoring

All API endpoints follow REST conventions and return JSON responses. The API uses
JWT-based authentication and implements comprehensive rate limiting and security measures.

Base URL: ``https://api.prsm.org/api/v1/``

.. toctree::
   :maxdepth: 2
   :caption: Core API Modules:

   main
   auth_api
   session_api
   task_api
   teams_api

.. toctree::
   :maxdepth: 2
   :caption: Marketplace & Payments:

   marketplace_api
   real_marketplace_api
   marketplace_launch_api
   payment_api
   budget_api

.. toctree::
   :maxdepth: 2
   :caption: Security & Compliance:

   security_status_api
   security_logging_api
   credential_api
   compliance_api
   cryptography_api

.. toctree::
   :maxdepth: 2
   :caption: System Administration:

   governance_api
   health_api
   monitoring_api
   mainnet_deployment_api
   workflow_scheduling_api

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features:

   recommendation_api
   reputation_api
   distillation_api
   resource_management_api
   alpha_api

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure APIs:

   ipfs_api
   cdn_api
   ui_api

API Authentication
------------------

All API endpoints require authentication using JWT tokens. Obtain tokens through the
:doc:`auth_api` endpoints.

Example authenticated request:

.. code-block:: bash

   curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
        https://api.prsm.org/api/v1/sessions

Rate Limiting
-------------

API endpoints implement rate limiting to ensure fair usage:

* Authentication endpoints: 5 requests per minute
* Standard API endpoints: 1000 requests per hour
* Resource-intensive endpoints: 100 requests per hour

Rate limit headers are included in all responses:

.. code-block:: http

   X-RateLimit-Limit: 1000
   X-RateLimit-Remaining: 999
   X-RateLimit-Reset: 1641024000

Error Handling
--------------

All API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "validation_error",
     "message": "Invalid request data",
     "code": 400,
     "timestamp": "2025-07-02T10:30:00Z",
     "details": {
       "field": "prompt",
       "issue": "Prompt cannot be empty"
     }
   }

Common HTTP status codes:

* **200** - Success
* **201** - Created
* **400** - Bad Request
* **401** - Unauthorized
* **403** - Forbidden
* **404** - Not Found
* **422** - Validation Error
* **429** - Rate Limited
* **500** - Internal Server Error

Quick Start
-----------

1. **Authentication**: Register and obtain JWT tokens via :doc:`auth_api`
2. **Create Session**: Start a research session via :doc:`session_api`
3. **Monitor Progress**: Track session status and retrieve results
4. **Manage Resources**: Use :doc:`payment_api` for FTNS token management

For detailed examples and integration guides, see the individual API module documentation.

