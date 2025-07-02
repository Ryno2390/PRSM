Teams API
=========

The Teams API provides comprehensive team management capabilities for collaborative research and development within the PRSM ecosystem.

.. automodule:: prsm.api.teams_api
   :members:
   :undoc-members:
   :show-inheritance:

Team Management Endpoints
-------------------------

Create Team
^^^^^^^^^^^

.. http:post:: /api/v1/teams

   Create a new research team with specified governance model and reward policies.

   **Request JSON Object:**
   
   * **name** (*string*) -- Team name (required)
   * **description** (*string*) -- Team description
   * **team_type** (*string*) -- Type of team (research, development, etc.)
   * **governance_model** (*string*) -- Governance structure
   * **reward_policy** (*object*) -- Team reward distribution policy

   **Response JSON Object:**
   
   * **team_id** (*string*) -- Unique team identifier
   * **name** (*string*) -- Team name
   * **status** (*string*) -- Team status
   * **created_at** (*string*) -- Creation timestamp

   **Status Codes:**
   
   * **201** -- Team created successfully
   * **400** -- Invalid request data
   * **401** -- Unauthorized
   * **409** -- Team name already exists

List Teams
^^^^^^^^^^

.. http:get:: /api/v1/teams

   Retrieve paginated list of teams with optional filtering.

   **Query Parameters:**
   
   * **page** (*integer*, optional) -- Page number (default: 1)
   * **limit** (*integer*, optional) -- Items per page (default: 20)
   * **team_type** (*string*, optional) -- Filter by team type
   * **status** (*string*, optional) -- Filter by team status

   **Response JSON Object:**
   
   * **teams** (*array*) -- List of team objects
   * **total_count** (*integer*) -- Total number of teams
   * **page** (*integer*) -- Current page number
   * **total_pages** (*integer*) -- Total number of pages

   **Status Codes:**
   
   * **200** -- Teams retrieved successfully
   * **401** -- Unauthorized

Get Team Details
^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/teams/{team_id}

   Retrieve detailed information about a specific team.

   **Path Parameters:**
   
   * **team_id** (*string*) -- Unique team identifier

   **Response JSON Object:**
   
   * **team_id** (*string*) -- Team identifier
   * **name** (*string*) -- Team name
   * **description** (*string*) -- Team description
   * **members** (*array*) -- List of team members
   * **governance_model** (*object*) -- Governance configuration
   * **reward_policy** (*object*) -- Reward distribution policy
   * **statistics** (*object*) -- Team performance statistics

   **Status Codes:**
   
   * **200** -- Team details retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Team not found

Update Team
^^^^^^^^^^^

.. http:put:: /api/v1/teams/{team_id}

   Update team configuration and settings.

   **Path Parameters:**
   
   * **team_id** (*string*) -- Unique team identifier

   **Request JSON Object:**
   
   * **name** (*string*, optional) -- Updated team name
   * **description** (*string*, optional) -- Updated description
   * **governance_model** (*object*, optional) -- Updated governance settings
   * **reward_policy** (*object*, optional) -- Updated reward policy

   **Status Codes:**
   
   * **200** -- Team updated successfully
   * **400** -- Invalid request data
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions
   * **404** -- Team not found

Team Membership Endpoints
-------------------------

Join Team
^^^^^^^^^

.. http:post:: /api/v1/teams/{team_id}/join

   Request to join a team or accept an invitation.

   **Path Parameters:**
   
   * **team_id** (*string*) -- Unique team identifier

   **Request JSON Object:**
   
   * **invitation_code** (*string*, optional) -- Team invitation code
   * **message** (*string*, optional) -- Join request message

   **Status Codes:**
   
   * **200** -- Join request submitted successfully
   * **201** -- Joined team successfully (with invitation)
   * **401** -- Unauthorized
   * **404** -- Team not found
   * **409** -- Already a member

Leave Team
^^^^^^^^^^

.. http:post:: /api/v1/teams/{team_id}/leave

   Leave a team.

   **Path Parameters:**
   
   * **team_id** (*string*) -- Unique team identifier

   **Status Codes:**
   
   * **200** -- Left team successfully
   * **401** -- Unauthorized
   * **404** -- Team not found or not a member

Invite Members
^^^^^^^^^^^^^^

.. http:post:: /api/v1/teams/{team_id}/invite

   Invite users to join the team.

   **Path Parameters:**
   
   * **team_id** (*string*) -- Unique team identifier

   **Request JSON Object:**
   
   * **user_ids** (*array*) -- List of user IDs to invite
   * **role** (*string*, optional) -- Role to assign to invited members
   * **message** (*string*, optional) -- Invitation message

   **Status Codes:**
   
   * **200** -- Invitations sent successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions
   * **404** -- Team not found

Team Governance
---------------

Governance Models
^^^^^^^^^^^^^^^^^

The Teams API supports multiple governance models:

* **Hierarchical** - Traditional top-down structure
* **Democratic** - Voting-based decisions
* **Consensus** - Unanimous agreement required
* **Delegated** - Elected representatives make decisions
* **Hybrid** - Combination of multiple models

Reward Policies
^^^^^^^^^^^^^^^

Teams can configure various reward distribution policies:

* **Equal** - Equal distribution among all members
* **Contribution-based** - Based on individual contributions
* **Role-based** - Based on member roles and responsibilities
* **Performance-based** - Based on measurable performance metrics
* **Custom** - Custom formula defined by team

Team Types
----------

Research Teams
^^^^^^^^^^^^^^

Focused on scientific research and discovery:

* Access to research datasets and tools
* Collaboration with academic institutions
* Publication and peer review workflows
* Grant application and funding management

Development Teams
^^^^^^^^^^^^^^^^^

Focused on software and product development:

* Code repository management
* CI/CD pipeline integration
* Issue tracking and project management
* Release and deployment coordination

Innovation Teams
^^^^^^^^^^^^^^^^

Focused on experimental and cutting-edge projects:

* Rapid prototyping capabilities
* Access to emerging technologies
* Flexible governance and rapid iteration
* Risk-tolerant reward structures

Error Responses
--------------

Team API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "team_not_found",
     "message": "The specified team does not exist",
     "code": 404,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error codes:

* ``team_not_found`` - Team does not exist
* ``insufficient_permissions`` - User lacks required team permissions
* ``team_name_exists`` - Team name already in use
* ``invalid_governance_model`` - Unsupported governance configuration
* ``member_limit_exceeded`` - Team has reached maximum member limit