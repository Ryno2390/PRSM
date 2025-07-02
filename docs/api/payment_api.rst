Payment API
===========

The Payment API provides secure payment processing, FTNS token transactions, and financial management capabilities for the PRSM ecosystem.

.. automodule:: prsm.api.payment_api
   :members:
   :undoc-members:
   :show-inheritance:

Payment Processing Endpoints
----------------------------

Process Payment
^^^^^^^^^^^^^^^

.. http:post:: /api/v1/payments/process

   Process a payment transaction with multiple payment methods.

   **Request JSON Object:**
   
   * **amount** (*decimal*) -- Payment amount (required)
   * **currency** (*string*) -- Payment currency (USD, FTNS)
   * **payment_method** (*object*) -- Payment method details
   * **description** (*string*) -- Payment description
   * **metadata** (*object*, optional) -- Additional payment metadata

   **Response JSON Object:**
   
   * **payment_id** (*string*) -- Unique payment identifier
   * **status** (*string*) -- Payment status
   * **amount** (*decimal*) -- Processed amount
   * **transaction_fee** (*decimal*) -- Applied transaction fee
   * **confirmation_code** (*string*) -- Payment confirmation code

   **Status Codes:**
   
   * **201** -- Payment processed successfully
   * **400** -- Invalid payment data
   * **401** -- Unauthorized
   * **402** -- Payment required (insufficient funds)
   * **422** -- Payment processing failed

Get Payment Status
^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/payments/{payment_id}

   Retrieve detailed payment status and transaction history.

   **Path Parameters:**
   
   * **payment_id** (*string*) -- Unique payment identifier

   **Response JSON Object:**
   
   * **payment_id** (*string*) -- Payment identifier
   * **status** (*string*) -- Current payment status
   * **amount** (*decimal*) -- Payment amount
   * **currency** (*string*) -- Payment currency
   * **created_at** (*string*) -- Payment creation timestamp
   * **updated_at** (*string*) -- Last update timestamp
   * **transaction_history** (*array*) -- Payment status changes

   **Status Codes:**
   
   * **200** -- Payment status retrieved successfully
   * **401** -- Unauthorized
   * **404** -- Payment not found

FTNS Token Management
--------------------

Get FTNS Balance
^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/payments/ftns/balance

   Retrieve current FTNS token balance and staking information.

   **Response JSON Object:**
   
   * **available_balance** (*decimal*) -- Available FTNS tokens
   * **staked_balance** (*decimal*) -- Staked FTNS tokens
   * **total_balance** (*decimal*) -- Total FTNS token balance
   * **pending_transactions** (*array*) -- Pending FTNS transactions
   * **last_updated** (*string*) -- Balance last update timestamp

   **Status Codes:**
   
   * **200** -- Balance retrieved successfully
   * **401** -- Unauthorized

Transfer FTNS Tokens
^^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/payments/ftns/transfer

   Transfer FTNS tokens to another user or external wallet.

   **Request JSON Object:**
   
   * **recipient** (*string*) -- Recipient user ID or wallet address
   * **amount** (*decimal*) -- Amount to transfer (required)
   * **memo** (*string*, optional) -- Transfer memo
   * **priority** (*string*, optional) -- Transaction priority (normal, high)

   **Response JSON Object:**
   
   * **transfer_id** (*string*) -- Unique transfer identifier
   * **status** (*string*) -- Transfer status
   * **transaction_hash** (*string*) -- Blockchain transaction hash
   * **estimated_confirmation** (*string*) -- Estimated confirmation time

   **Status Codes:**
   
   * **201** -- Transfer initiated successfully
   * **400** -- Invalid transfer data
   * **401** -- Unauthorized
   * **402** -- Insufficient FTNS balance

Purchase FTNS Tokens
^^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/payments/ftns/purchase

   Purchase FTNS tokens using fiat currency or cryptocurrency.

   **Request JSON Object:**
   
   * **amount_usd** (*decimal*) -- USD amount to spend
   * **payment_method** (*object*) -- Payment method details
   * **auto_stake** (*boolean*, optional) -- Automatically stake purchased tokens

   **Response JSON Object:**
   
   * **purchase_id** (*string*) -- Unique purchase identifier
   * **ftns_amount** (*decimal*) -- FTNS tokens to be received
   * **exchange_rate** (*decimal*) -- USD to FTNS exchange rate
   * **total_cost** (*decimal*) -- Total cost including fees
   * **estimated_delivery** (*string*) -- Estimated token delivery time

   **Status Codes:**
   
   * **201** -- Purchase initiated successfully
   * **400** -- Invalid purchase data
   * **401** -- Unauthorized
   * **422** -- Payment processing failed

Subscription Management
----------------------

Create Subscription
^^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/payments/subscriptions

   Create a recurring subscription for PRSM services.

   **Request JSON Object:**
   
   * **plan_id** (*string*) -- Subscription plan identifier
   * **payment_method** (*object*) -- Payment method for recurring charges
   * **billing_cycle** (*string*) -- Billing frequency (monthly, quarterly, annual)
   * **auto_renew** (*boolean*, optional) -- Enable automatic renewal

   **Response JSON Object:**
   
   * **subscription_id** (*string*) -- Unique subscription identifier
   * **status** (*string*) -- Subscription status
   * **next_billing_date** (*string*) -- Next billing date
   * **total_amount** (*decimal*) -- Subscription amount per cycle

   **Status Codes:**
   
   * **201** -- Subscription created successfully
   * **400** -- Invalid subscription data
   * **401** -- Unauthorized
   * **402** -- Payment method validation failed

Manage Subscription
^^^^^^^^^^^^^^^^^^

.. http:put:: /api/v1/payments/subscriptions/{subscription_id}

   Update or modify an existing subscription.

   **Path Parameters:**
   
   * **subscription_id** (*string*) -- Unique subscription identifier

   **Request JSON Object:**
   
   * **plan_id** (*string*, optional) -- New subscription plan
   * **payment_method** (*object*, optional) -- Updated payment method
   * **auto_renew** (*boolean*, optional) -- Update auto-renewal setting

   **Status Codes:**
   
   * **200** -- Subscription updated successfully
   * **400** -- Invalid update data
   * **401** -- Unauthorized
   * **404** -- Subscription not found

Cancel Subscription
^^^^^^^^^^^^^^^^^^

.. http:delete:: /api/v1/payments/subscriptions/{subscription_id}

   Cancel an active subscription.

   **Path Parameters:**
   
   * **subscription_id** (*string*) -- Unique subscription identifier

   **Query Parameters:**
   
   * **immediate** (*boolean*, optional) -- Cancel immediately vs. at period end

   **Status Codes:**
   
   * **200** -- Subscription cancelled successfully
   * **401** -- Unauthorized
   * **404** -- Subscription not found

Invoice and Billing
-------------------

Get Invoices
^^^^^^^^^^^^

.. http:get:: /api/v1/payments/invoices

   Retrieve billing invoices with filtering and pagination.

   **Query Parameters:**
   
   * **status** (*string*, optional) -- Filter by invoice status
   * **date_from** (*string*, optional) -- Start date filter
   * **date_to** (*string*, optional) -- End date filter
   * **page** (*integer*, optional) -- Page number
   * **limit** (*integer*, optional) -- Items per page

   **Response JSON Object:**
   
   * **invoices** (*array*) -- List of invoice objects
   * **total_count** (*integer*) -- Total number of invoices
   * **total_amount** (*decimal*) -- Sum of all invoice amounts
   * **page** (*integer*) -- Current page number

   **Status Codes:**
   
   * **200** -- Invoices retrieved successfully
   * **401** -- Unauthorized

Download Invoice
^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/payments/invoices/{invoice_id}/download

   Download invoice as PDF document.

   **Path Parameters:**
   
   * **invoice_id** (*string*) -- Unique invoice identifier

   **Response:**
   
   PDF document download

   **Status Codes:**
   
   * **200** -- Invoice PDF downloaded successfully
   * **401** -- Unauthorized
   * **404** -- Invoice not found

Payment Methods
--------------

Add Payment Method
^^^^^^^^^^^^^^^^^^

.. http:post:: /api/v1/payments/methods

   Add a new payment method to user account.

   **Request JSON Object:**
   
   * **type** (*string*) -- Payment method type (card, bank, crypto)
   * **details** (*object*) -- Payment method specific details
   * **is_default** (*boolean*, optional) -- Set as default payment method
   * **billing_address** (*object*, optional) -- Billing address information

   **Response JSON Object:**
   
   * **method_id** (*string*) -- Unique payment method identifier
   * **type** (*string*) -- Payment method type
   * **last_four** (*string*) -- Last four digits (for cards)
   * **status** (*string*) -- Payment method status

   **Status Codes:**
   
   * **201** -- Payment method added successfully
   * **400** -- Invalid payment method data
   * **401** -- Unauthorized
   * **422** -- Payment method validation failed

List Payment Methods
^^^^^^^^^^^^^^^^^^^^

.. http:get:: /api/v1/payments/methods

   List all payment methods associated with user account.

   **Response JSON Object:**
   
   * **payment_methods** (*array*) -- List of payment method objects
   * **default_method_id** (*string*) -- Default payment method identifier

   **Status Codes:**
   
   * **200** -- Payment methods retrieved successfully
   * **401** -- Unauthorized

Remove Payment Method
^^^^^^^^^^^^^^^^^^^^

.. http:delete:: /api/v1/payments/methods/{method_id}

   Remove a payment method from user account.

   **Path Parameters:**
   
   * **method_id** (*string*) -- Unique payment method identifier

   **Status Codes:**
   
   * **204** -- Payment method removed successfully
   * **401** -- Unauthorized
   * **404** -- Payment method not found
   * **409** -- Cannot remove default payment method

Security and Compliance
-----------------------

PCI DSS Compliance
^^^^^^^^^^^^^^^^^^

All payment processing adheres to PCI DSS standards:

* Card data tokenization and encryption
* Secure payment gateway integration
* No storage of sensitive card information
* Regular security audits and compliance validation

Fraud Prevention
^^^^^^^^^^^^^^^^

Advanced fraud detection and prevention measures:

* Real-time transaction monitoring
* Machine learning-based risk scoring
* Velocity checks and spending limits
* Geolocation and device fingerprinting

FTNS Token Security
^^^^^^^^^^^^^^^^^^

Secure FTNS token management:

* Multi-signature wallet integration
* Hardware security module (HSM) key storage
* Transaction confirmation requirements
* Automated suspicious activity detection

Error Responses
--------------

Payment API endpoints return standardized error responses:

.. code-block:: json

   {
     "error": "insufficient_funds",
     "message": "Insufficient FTNS balance for transaction",
     "code": 402,
     "timestamp": "2025-07-02T10:30:00Z"
   }

Common error codes:

* ``insufficient_funds`` - Insufficient balance for transaction
* ``payment_failed`` - Payment processing failed
* ``invalid_payment_method`` - Payment method is invalid or expired
* ``subscription_not_found`` - Subscription does not exist
* ``invoice_not_found`` - Invoice does not exist