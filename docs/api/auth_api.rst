Authentication API
================

The Authentication API provides secure user authentication and authorization endpoints for the PRSM system.

.. automodule:: prsm.api.auth_api
   :members:
   :undoc-members:
   :show-inheritance:

Authentication Endpoints
-----------------------

Login
^^^^^

.. http:post:: /auth/login

   Authenticate a user with credentials.

   **Request JSON Object:**
   
   * **username** (*string*) -- User's username or email
   * **password** (*string*) -- User's password
   * **mfa_token** (*string*, optional) -- Multi-factor authentication token

   **Response JSON Object:**
   
   * **access_token** (*string*) -- JWT access token
   * **refresh_token** (*string*) -- JWT refresh token
   * **expires_in** (*integer*) -- Token expiration time in seconds
   * **user_id** (*string*) -- Unique user identifier

   **Status Codes:**
   
   * **200** -- Authentication successful
   * **401** -- Invalid credentials
   * **429** -- Rate limit exceeded

Logout
^^^^^^

.. http:post:: /auth/logout

   Invalidate user session and tokens.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Status Codes:**
   
   * **200** -- Logout successful
   * **401** -- Unauthorized

Token Refresh
^^^^^^^^^^^^^

.. http:post:: /auth/refresh

   Refresh an expired access token using a refresh token.

   **Request JSON Object:**
   
   * **refresh_token** (*string*) -- Valid refresh token

   **Response JSON Object:**
   
   * **access_token** (*string*) -- New JWT access token
   * **expires_in** (*integer*) -- Token expiration time in seconds

   **Status Codes:**
   
   * **200** -- Token refreshed successfully
   * **401** -- Invalid refresh token

User Registration
^^^^^^^^^^^^^^^

.. http:post:: /auth/register

   Register a new user account.

   **Request JSON Object:**
   
   * **username** (*string*) -- Desired username
   * **email** (*string*) -- User's email address
   * **password** (*string*) -- User's password (min 8 characters)
   * **first_name** (*string*, optional) -- User's first name
   * **last_name** (*string*, optional) -- User's last name

   **Response JSON Object:**
   
   * **user_id** (*string*) -- Unique user identifier
   * **message** (*string*) -- Registration status message

   **Status Codes:**
   
   * **201** -- User created successfully
   * **400** -- Invalid request data
   * **409** -- Username/email already exists

Authorization Endpoints
----------------------

User Permissions
^^^^^^^^^^^^^^^

.. http:get:: /auth/permissions

   Get current user's permissions and roles.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **permissions** (*array*) -- List of user permissions
   * **roles** (*array*) -- List of user roles
   * **user_id** (*string*) -- User identifier

   **Status Codes:**
   
   * **200** -- Permissions retrieved successfully
   * **401** -- Unauthorized

Role Management
^^^^^^^^^^^^^^

.. http:post:: /auth/roles

   Assign roles to users (admin only).

   **Request Headers:**
   
   * **Authorization** -- Bearer token (admin role required)

   **Request JSON Object:**
   
   * **user_id** (*string*) -- Target user identifier
   * **roles** (*array*) -- Roles to assign

   **Status Codes:**
   
   * **200** -- Roles assigned successfully
   * **401** -- Unauthorized
   * **403** -- Insufficient permissions

Multi-Factor Authentication
--------------------------

MFA Setup
^^^^^^^^^

.. http:post:: /auth/mfa/setup

   Set up multi-factor authentication for user account.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Response JSON Object:**
   
   * **qr_code** (*string*) -- Base64 encoded QR code
   * **secret** (*string*) -- TOTP secret key
   * **backup_codes** (*array*) -- One-time backup codes

   **Status Codes:**
   
   * **200** -- MFA setup initiated
   * **401** -- Unauthorized

MFA Verification
^^^^^^^^^^^^^^

.. http:post:: /auth/mfa/verify

   Verify multi-factor authentication token.

   **Request Headers:**
   
   * **Authorization** -- Bearer token

   **Request JSON Object:**
   
   * **token** (*string*) -- 6-digit TOTP token or backup code

   **Status Codes:**
   
   * **200** -- MFA verified successfully
   * **401** -- Invalid MFA token

Security Features
----------------

Rate Limiting
^^^^^^^^^^^^

All authentication endpoints implement rate limiting to prevent brute force attacks:

* Login attempts: 5 per minute per IP
* Registration: 3 per hour per IP
* Token refresh: 10 per minute per user

Password Security
^^^^^^^^^^^^^^^

* Minimum 8 characters required
* Passwords are hashed using bcrypt with salt
* Password history tracking prevents reuse of last 12 passwords

Session Management
^^^^^^^^^^^^^^^^^

* JWT tokens with configurable expiration
* Automatic token rotation
* Secure session invalidation on logout

Error Responses
--------------

All authentication endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "authentication_failed",
     "message": "Invalid username or password",
     "code": 401,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error codes:

* ``authentication_failed`` - Invalid credentials
* ``token_expired`` - JWT token has expired
* ``insufficient_permissions`` - User lacks required permissions
* ``rate_limit_exceeded`` - Too many requests
* ``validation_error`` - Invalid request data