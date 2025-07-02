Governance API
==============

The Governance API provides endpoints for participating in PRSM's decentralized governance system, including proposal creation, voting, and governance token management.

.. automodule:: prsm.api.governance_api
   :members:
   :undoc-members:
   :show-inheritance:

Proposals
---------

List Proposals
^^^^^^^^^^^^^

.. http:get:: /governance/proposals

   Retrieve a list of governance proposals.

   **Query Parameters:**
   
   * **status** (*string*, optional) -- Filter by proposal status ("active", "passed", "failed", "pending")
   * **category** (*string*, optional) -- Filter by category ("protocol", "treasury", "network", "community")
   * **author** (*string*, optional) -- Filter by proposal author
   * **page** (*integer*, optional) -- Page number for pagination (default: 1)
   * **limit** (*integer*, optional) -- Number of results per page (default: 20, max: 100)

   **Response JSON Object:**
   
   * **proposals** (*array*) -- List of proposal objects
   * **total** (*integer*) -- Total number of proposals
   * **page** (*integer*) -- Current page number
   * **pages** (*integer*) -- Total number of pages

   **Proposal Object:**
   
   * **id** (*string*) -- Unique proposal identifier
   * **title** (*string*) -- Proposal title
   * **description** (*string*) -- Brief proposal description
   * **author** (*string*) -- Proposal author's address
   * **category** (*string*) -- Proposal category
   * **status** (*string*) -- Current status
   * **voting_start** (*string*) -- Voting start timestamp (ISO 8601)
   * **voting_end** (*string*) -- Voting end timestamp (ISO 8601)
   * **votes_for** (*integer*) -- Number of votes in favor
   * **votes_against** (*integer*) -- Number of votes against
   * **quorum_reached** (*boolean*) -- Whether quorum has been reached
   * **created_at** (*string*) -- Proposal creation timestamp

   **Status Codes:**
   
   * **200** -- Proposals retrieved successfully
   * **400** -- Invalid query parameters

Proposal Details
^^^^^^^^^^^^^^^

.. http:get:: /governance/proposals/{proposal_id}

   Get detailed information about a specific proposal.

   **Path Parameters:**
   
   * **proposal_id** (*string*) -- Unique proposal identifier

   **Response JSON Object:**
   
   * **id** (*string*) -- Proposal identifier
   * **title** (*string*) -- Proposal title
   * **description** (*string*) -- Full proposal description
   * **author** (*string*) -- Author's governance address
   * **category** (*string*) -- Proposal category
   * **status** (*string*) -- Current status
   * **execution_payload** (*object*, optional) -- Smart contract execution data
   * **voting_period** (*object*) -- Voting period details
   * **voting_results** (*object*) -- Current voting results
   * **quorum_requirement** (*integer*) -- Required quorum percentage
   * **discussion_url** (*string*, optional) -- Link to discussion forum
   * **created_at** (*string*) -- Creation timestamp
   * **updated_at** (*string*) -- Last update timestamp

   **Status Codes:**
   
   * **200** -- Proposal details retrieved successfully
   * **404** -- Proposal not found

Create Proposal
^^^^^^^^^^^^^^

.. http:post:: /governance/proposals

   Create a new governance proposal.

   **Request Headers:**
   
   * **Authorization** -- Bearer token
   * **Content-Type** -- application/json

   **Request JSON Object:**
   
   * **title** (*string*) -- Proposal title (max 200 characters)
   * **description** (*string*) -- Detailed proposal description
   * **category** (*string*) -- Proposal category
   * **execution_payload** (*object*, optional) -- Smart contract execution data
   * **voting_duration** (*integer*, optional) -- Voting period in hours (default: 168)
   * **discussion_url** (*string*, optional) -- Link to discussion

   **Response JSON Object:**
   
   * **proposal_id** (*string*) -- Created proposal identifier
   * **status** (*string*) -- Initial proposal status
   * **voting_start** (*string*) -- Scheduled voting start time
   * **voting_end** (*string*) -- Scheduled voting end time

   **Status Codes:**
   
   * **201** -- Proposal created successfully
   * **400** -- Invalid proposal data
   * **401** -- Unauthorized
   * **403** -- Insufficient governance token balance

Voting
------

Cast Vote
^^^^^^^^^

.. http:post:: /governance/proposals/{proposal_id}/vote

   Cast a vote on a governance proposal.

   **Path Parameters:**
   
   * **proposal_id** (*string*) -- Proposal identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **vote** (*string*) -- Vote choice ("for", "against", "abstain")
   * **voting_power** (*integer*, optional) -- Amount of tokens to vote with
   * **reason** (*string*, optional) -- Reason for the vote (max 500 characters)

   **Response JSON Object:**
   
   * **vote_id** (*string*) -- Unique vote identifier
   * **voting_power** (*integer*) -- Tokens used for voting
   * **vote** (*string*) -- Vote choice
   * **timestamp** (*string*) -- Vote timestamp

   **Status Codes:**
   
   * **200** -- Vote cast successfully
   * **400** -- Invalid vote data
   * **401** -- Unauthorized
   * **403** -- Voting period not active or insufficient balance
   * **409** -- Already voted on this proposal

Get Vote
^^^^^^^^

.. http:get:: /governance/proposals/{proposal_id}/votes/{user_address}

   Get a user's vote on a specific proposal.

   **Path Parameters:**
   
   * **proposal_id** (*string*) -- Proposal identifier
   * **user_address** (*string*) -- User's governance address

   **Response JSON Object:**
   
   * **vote_id** (*string*) -- Vote identifier
   * **vote** (*string*) -- Vote choice
   * **voting_power** (*integer*) -- Tokens used
   * **reason** (*string*) -- Vote reason
   * **timestamp** (*string*) -- Vote timestamp

   **Status Codes:**
   
   * **200** -- Vote retrieved successfully
   * **404** -- Vote not found

Proposal Votes
^^^^^^^^^^^^^

.. http:get:: /governance/proposals/{proposal_id}/votes

   Get all votes for a specific proposal.

   **Path Parameters:**
   
   * **proposal_id** (*string*) -- Proposal identifier

   **Query Parameters:**
   
   * **vote_type** (*string*, optional) -- Filter by vote type ("for", "against", "abstain")
   * **page** (*integer*, optional) -- Page number (default: 1)
   * **limit** (*integer*, optional) -- Results per page (default: 50, max: 200)

   **Response JSON Object:**
   
   * **votes** (*array*) -- List of vote objects
   * **summary** (*object*) -- Vote count summary
   * **total** (*integer*) -- Total number of votes
   * **page** (*integer*) -- Current page
   * **pages** (*integer*) -- Total pages

   **Status Codes:**
   
   * **200** -- Votes retrieved successfully
   * **404** -- Proposal not found

Governance Tokens
-----------------

Token Balance
^^^^^^^^^^^^

.. http:get:: /governance/tokens/balance

   Get current governance token balance for authenticated user.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **balance** (*integer*) -- Available governance tokens
   * **locked** (*integer*) -- Tokens locked in active votes
   * **delegated_to** (*string*, optional) -- Address tokens are delegated to
   * **delegated_from** (*array*) -- Addresses that delegated to this user
   * **voting_power** (*integer*) -- Total voting power (balance + delegated)

   **Status Codes:**
   
   * **200** -- Balance retrieved successfully
   * **401** -- Unauthorized

Delegate Tokens
^^^^^^^^^^^^^^

.. http:post:: /governance/tokens/delegate

   Delegate governance tokens to another address.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **delegate_to** (*string*) -- Address to delegate tokens to
   * **amount** (*integer*) -- Number of tokens to delegate

   **Response JSON Object:**
   
   * **delegation_id** (*string*) -- Delegation transaction ID
   * **delegated_amount** (*integer*) -- Amount successfully delegated
   * **delegate_address** (*string*) -- Delegate's address

   **Status Codes:**
   
   * **200** -- Delegation successful
   * **400** -- Invalid delegation parameters
   * **401** -- Unauthorized
   * **403** -- Insufficient token balance

Revoke Delegation
^^^^^^^^^^^^^^^^

.. http:delete:: /governance/tokens/delegate/{delegation_id}

   Revoke a token delegation.

   **Path Parameters:**
   
   * **delegation_id** (*string*) -- Delegation identifier

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Status Codes:**
   
   * **200** -- Delegation revoked successfully
   * **401** -- Unauthorized
   * **404** -- Delegation not found
   * **403** -- Cannot revoke delegation (tokens may be locked in active votes)

Treasury Management
------------------

Treasury Status
^^^^^^^^^^^^^^

.. http:get:: /governance/treasury

   Get current treasury status and allocation.

   **Response JSON Object:**
   
   * **total_funds** (*object*) -- Total funds by token type
   * **allocations** (*array*) -- Current budget allocations
   * **recent_transactions** (*array*) -- Recent treasury transactions
   * **pending_proposals** (*array*) -- Proposals affecting treasury

   **Status Codes:**
   
   * **200** -- Treasury status retrieved successfully

Treasury Proposal
^^^^^^^^^^^^^^^^

.. http:post:: /governance/treasury/proposals

   Create a treasury funding proposal.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **title** (*string*) -- Proposal title
   * **description** (*string*) -- Detailed description
   * **requested_amount** (*integer*) -- Amount requested
   * **token_type** (*string*) -- Token type (e.g., "FTNS", "ETH")
   * **recipient_address** (*string*) -- Recipient's address
   * **milestones** (*array*, optional) -- Payment milestones

   **Status Codes:**
   
   * **201** -- Treasury proposal created successfully
   * **400** -- Invalid proposal data
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Network Parameters
-----------------

Get Parameters
^^^^^^^^^^^^^

.. http:get:: /governance/parameters

   Get current network governance parameters.

   **Response JSON Object:**
   
   * **voting_period** (*integer*) -- Default voting period in hours
   * **quorum_threshold** (*integer*) -- Required quorum percentage
   * **proposal_threshold** (*integer*) -- Minimum tokens to create proposal
   * **execution_delay** (*integer*) -- Delay before proposal execution
   * **parameters** (*object*) -- Other configurable network parameters

   **Status Codes:**
   
   * **200** -- Parameters retrieved successfully

Update Parameters
^^^^^^^^^^^^^^^^

.. http:post:: /governance/parameters/proposals

   Create a proposal to update network parameters.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **parameter_changes** (*object*) -- Proposed parameter changes
   * **rationale** (*string*) -- Explanation for changes

   **Status Codes:**
   
   * **201** -- Parameter change proposal created
   * **400** -- Invalid parameter changes
   * **401** -- Unauthorized

Analytics
---------

Governance Statistics
^^^^^^^^^^^^^^^^^^^

.. http:get:: /governance/analytics/stats

   Get governance system statistics.

   **Query Parameters:**
   
   * **period** (*string*, optional) -- Time period ("week", "month", "quarter", default: "month")

   **Response JSON Object:**
   
   * **total_proposals** (*integer*) -- Total number of proposals
   * **active_proposals** (*integer*) -- Currently active proposals
   * **voter_participation** (*float*) -- Average voter participation rate
   * **token_distribution** (*object*) -- Governance token distribution
   * **proposal_success_rate** (*float*) -- Percentage of passed proposals

   **Status Codes:**
   
   * **200** -- Statistics retrieved successfully

User Activity
^^^^^^^^^^^^

.. http:get:: /governance/analytics/user

   Get governance activity for authenticated user.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **proposals_created** (*integer*) -- Number of proposals created
   * **votes_cast** (*integer*) -- Number of votes cast
   * **voting_power_history** (*array*) -- Historical voting power
   * **participation_rate** (*float*) -- User's participation rate

   **Status Codes:**
   
   * **200** -- User activity retrieved successfully
   * **401** -- Unauthorized

Error Handling
-------------

Governance API errors follow the standard format:

.. code-block:: json

   {
     "error": "insufficient_voting_power",
     "message": "Minimum 1000 FTNS tokens required to create proposal",
     "code": 403,
     "required_balance": 1000,
     "current_balance": 500,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error types:

* ``insufficient_voting_power`` - Not enough tokens for action
* ``voting_period_ended`` - Voting period has closed
* ``proposal_not_found`` - Proposal doesn't exist
* ``already_voted`` - User has already voted on proposal
* ``invalid_vote_choice`` - Invalid vote option
* ``delegation_failed`` - Token delegation error

Rate Limits
-----------

Governance endpoints have the following rate limits:

* Proposal creation: 3 per hour per user
* Voting: 100 votes per hour per user
* Token operations: 50 per hour per user
* Read operations: 1000 per hour per user

Security Features
----------------

* All governance actions require authentication
* Proposal creation requires minimum token balance
* Vote delegation uses cryptographic signatures
* Treasury proposals have additional validation
* Rate limiting prevents governance spam attacks