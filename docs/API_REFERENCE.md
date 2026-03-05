# PRSM API Reference

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Node API Endpoints](#node-api-endpoints)
   - [Core Endpoints](#node-core-endpoints)
   - [Compute Endpoints](#node-compute-endpoints)
   - [Content Endpoints](#node-content-endpoints)
   - [Agent Endpoints](#node-agent-endpoints)
   - [Ledger Endpoints](#node-ledger-endpoints)
   - [Staking Endpoints](#node-staking-endpoints)
   - [Bridge Endpoints](#node-bridge-endpoints)
   - [Storage Endpoints](#node-storage-endpoints)
   - [WebSocket Endpoints](#node-websocket-endpoints)
   - [Authentication Endpoints](#node-authentication-endpoints)
4. [Platform API Endpoints](#platform-api-endpoints)
   - [NWTN (Neural Web of Thought Networks)](#nwtn-neural-web-of-thought-networks)
   - [SEAL Technology API](#seal-technology-api)
   - [Model Management](#model-management)
5. [Health & Monitoring](#health--monitoring)
6. [User Management](#user-management)
7. [Marketplace](#marketplace)
8. [Governance](#governance)
9. [Web3 & FTNS](#web3--ftns)
10. [Security](#security)
11. [WebSocket API](#websocket-api)
12. [Error Handling](#error-handling)
13. [Rate Limiting](#rate-limiting)
14. [SDK Examples](#sdk-examples)

## Overview

The PRSM API provides a comprehensive RESTful interface for interacting with the Protocol for Recursive Scientific Modeling, featuring MIT's breakthrough SEAL (Self-Adapting Language Models) technology. All endpoints follow REST conventions and return JSON responses.

### Base URLs

- **Production**: `https://api.prsm.org`
- **Staging**: `https://staging-api.prsm.org`
- **Development**: `http://localhost:8000`

### API Versioning

The API uses URL-based versioning:
- Current version: `v1`
- Full endpoint format: `{base_url}/api/v1/{endpoint}`

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`

## Authentication

### JWT Bearer Authentication

Most endpoints require JWT bearer token authentication:

```http
Authorization: Bearer <jwt_token>
```

### Obtaining a Token

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user123",
    "username": "your_username",
    "email": "user@example.com",
    "role": "researcher"
  }
}
```

#### Register
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "new_user",
  "email": "user@example.com",
  "password": "secure_password",
  "role": "researcher"
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Authorization: Bearer <jwt_token>
```

### API Key Authentication

For programmatic access, use API keys:

```http
X-API-Key: your_api_key
```

---

## Node API Endpoints

The Node API provides endpoints for monitoring and controlling a running PRSM node. These are node-local endpoints distinct from the main PRSM platform API.

**Base URL**: `http://localhost:8000` (default node API port)

### Node Core Endpoints

#### Get API Information
```http
GET /
```

Returns basic API information and available endpoints.

**Response**:
```json
{
  "name": "PRSM Node API",
  "version": "0.2.0",
  "docs": "/docs",
  "openapi": "/openapi.json",
  "websocket": "/ws/status"
}
```

#### Get Node Status
```http
GET /status
```

Returns comprehensive node status including system health and component states.

**Response**:
```json
{
  "node_id": "node_abc123",
  "status": "online",
  "uptime_seconds": 86400,
  "components": {
    "transport": "healthy",
    "discovery": "healthy",
    "ledger": "healthy"
  }
}
```

#### Get Connected Peers
```http
GET /peers
```

Lists all connected and known peers on the P2P network.

**Response**:
```json
{
  "connected": [
    {
      "peer_id": "peer_123",
      "address": "192.168.1.100:8080",
      "display_name": "Research Node 1",
      "connected_at": "2025-06-11T10:00:00Z",
      "last_seen": "2025-06-11T12:00:00Z",
      "outbound": true
    }
  ],
  "known": [
    {
      "node_id": "node_xyz",
      "address": "10.0.0.50:8080",
      "display_name": "Storage Provider",
      "last_seen": "2025-06-11T11:30:00Z"
    }
  ],
  "connected_count": 5,
  "known_count": 12
}
```

#### Get FTNS Balance
```http
GET /balance
```

Returns the node's FTNS balance and recent transaction history.

**Response**:
```json
{
  "wallet_id": "node_abc123",
  "balance": 1250.50,
  "recent_transactions": [
    {
      "tx_id": "tx_001",
      "type": "reward",
      "from": "system",
      "to": "node_abc123",
      "amount": 50.0,
      "description": "Compute job reward",
      "timestamp": "2025-06-11T11:00:00Z"
    }
  ]
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | Node not initialized |

#### Get Transaction History
```http
GET /transactions?limit=50
```

Returns transaction history for the node.

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | integer | No | 50 | Maximum transactions to return (max: 200) |

**Response**:
```json
{
  "transactions": [
    {
      "tx_id": "tx_001",
      "type": "transfer",
      "from": "node_abc123",
      "to": "node_xyz",
      "amount": 100.0,
      "description": "Payment for service",
      "timestamp": "2025-06-11T10:00:00Z"
    }
  ],
  "count": 1
}
```

#### Health Check
```http
GET /health
```

Simple health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "node_id": "node_abc123"
}
```

---

### Node Compute Endpoints

#### Submit Compute Job
```http
POST /compute/submit
Content-Type: application/json

{
  "job_type": "inference",
  "payload": {
    "model": "llama-7b",
    "prompt": "Analyze this data..."
  },
  "ftns_budget": 10.0
}
```

**Request Body**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| job_type | string | Yes | Type: `inference`, `embedding`, or `benchmark` |
| payload | object | No | Job-specific data |
| ftns_budget | float | No | Maximum FTNS to spend (default: 1.0) |

**Response**:
```json
{
  "job_id": "job_abc123",
  "status": "pending",
  "job_type": "inference",
  "ftns_budget": 10.0
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Invalid job type |
| 503 | Compute requester not initialized |

#### Get Job Status
```http
GET /compute/job/{job_id}
```

Returns the status of a submitted compute job.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string | Yes | ID of the job to retrieve |

**Response**:
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "job_type": "inference",
  "provider_id": "provider_xyz",
  "result": {
    "output": "Analysis results..."
  },
  "result_verified": true,
  "error": null,
  "created_at": "2025-06-11T10:00:00Z",
  "completed_at": "2025-06-11T10:05:00Z"
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Job not found |
| 503 | Compute requester not initialized |

#### Get Compute Stats
```http
GET /compute/stats
```

Returns compute provider statistics.

**Response**:
```json
{
  "available": true,
  "jobs_completed": 150,
  "jobs_queued": 3,
  "total_ftns_earned": 500.0,
  "average_job_time_seconds": 45.2
}
```

---

### Node Content Endpoints

#### Upload Content
```http
POST /content/upload
Content-Type: application/json

{
  "text": "Research paper content...",
  "filename": "paper.txt",
  "replicas": 3,
  "royalty_rate": 0.01,
  "parent_cids": ["QmParentCID123"]
}
```

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| text | string | Yes | - | Text content to upload |
| filename | string | No | "document.txt" | Original filename |
| replicas | integer | No | 3 | Number of storage replicas |
| royalty_rate | float | No | 0.01 | FTNS per access (0.001-0.1) |
| parent_cids | string[] | No | [] | Source material CIDs |

**Response**:
```json
{
  "cid": "QmXxx...yyy",
  "filename": "paper.txt",
  "size_bytes": 12345,
  "content_hash": "sha256:abc123...",
  "creator_id": "node_abc123",
  "royalty_rate": 0.01,
  "parent_cids": ["QmParentCID123"]
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 502 | Upload failed - IPFS not running |
| 503 | Content uploader not initialized |

#### Search Content
```http
GET /content/search?q=climate&limit=20
```

Searches the network content index by keyword.

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| q | string | No | "" | Search query |
| limit | integer | No | 20 | Max results (max: 100) |

**Response**:
```json
{
  "query": "climate",
  "results": [
    {
      "cid": "QmXxx...yyy",
      "filename": "climate_paper.pdf",
      "size_bytes": 50000,
      "content_hash": "sha256:abc123...",
      "creator_id": "researcher_1",
      "providers": ["provider_a", "provider_b"],
      "created_at": "2025-06-10T14:00:00Z",
      "metadata": {},
      "royalty_rate": 0.02,
      "parent_cids": []
    }
  ],
  "count": 1
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | Content index not initialized |

#### Get Content Record
```http
GET /content/{cid}
```

Looks up a specific content record by CID.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| cid | string | Yes | IPFS content identifier |

**Response**:
```json
{
  "cid": "QmXxx...yyy",
  "filename": "document.pdf",
  "size_bytes": 12345,
  "content_hash": "sha256:abc123...",
  "creator_id": "node_abc123",
  "providers": ["provider_1"],
  "created_at": "2025-06-11T10:00:00Z",
  "metadata": {},
  "royalty_rate": 0.01,
  "parent_cids": []
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Content not found in index |
| 503 | Content index not initialized |

#### Retrieve Content
```http
GET /content/retrieve/{cid}?timeout=30.0&verify_hash=true
```

Retrieves content from the P2P network by CID.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| cid | string | Yes | IPFS content identifier |

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| timeout | float | No | 30.0 | Seconds to wait for response |
| verify_hash | boolean | No | true | Verify SHA-256 hash |

**Response**:
```json
{
  "cid": "QmXxx...yyy",
  "status": "success",
  "data": "base64-encoded-content...",
  "size_bytes": 12345,
  "content_hash": "sha256:abc123...",
  "filename": "document.pdf",
  "providers_tried": 2
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | Content provider not initialized |
| 504 | Retrieval timed out |

#### Get Content Index Stats
```http
GET /content/index/stats
```

Returns content index statistics.

**Response**:
```json
{
  "total_items": 1500,
  "total_size_bytes": 50000000,
  "categories": {
    "papers": 800,
    "datasets": 400,
    "models": 300
  }
}
```

---

### Node Agent Endpoints

#### List Agents
```http
GET /agents?local_only=false
```

Lists known agents (local and/or remote).

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| local_only | boolean | No | false | Only return local agents |

**Response**:
```json
{
  "agents": [
    {
      "agent_id": "agent_001",
      "agent_name": "Research Assistant",
      "capabilities": ["analysis", "summarization"],
      "status": "online"
    }
  ],
  "count": 1
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | Agent registry not initialized |

#### Search Agents by Capability
```http
GET /agents/search?capability=analysis&limit=20
```

Searches agents by capability.

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| capability | string | Yes | - | Capability to search for |
| limit | integer | No | 20 | Max results (max: 100) |

**Response**:
```json
{
  "capability": "analysis",
  "agents": [
    {
      "agent_id": "agent_001",
      "agent_name": "Research Assistant",
      "capabilities": ["analysis", "summarization"],
      "status": "online"
    }
  ],
  "count": 1
}
```

#### Get Agent Spending
```http
GET /agents/spending
```

Returns aggregate spending dashboard for all local agents.

**Response**:
```json
{
  "agents": [
    {
      "agent_id": "agent_001",
      "agent_name": "Research Assistant",
      "allowance": {
        "amount": 100.0,
        "spent": 25.0,
        "remaining": 75.0,
        "epoch_hours": 24.0
      }
    }
  ],
  "count": 1
}
```

#### Get Agent Details
```http
GET /agents/{agent_id}
```

Gets agent details, spending, and status.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| agent_id | string | Yes | Agent identifier |

**Response**:
```json
{
  "agent_id": "agent_001",
  "agent_name": "Research Assistant",
  "capabilities": ["analysis", "summarization"],
  "status": "online",
  "allowance": {
    "amount": 100.0,
    "spent": 25.0,
    "remaining": 75.0
  }
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Agent not found |
| 503 | Agent registry not initialized |

#### Get Agent Conversations
```http
GET /agents/{agent_id}/conversations?limit=10
```

Gets recent conversation threads involving an agent.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| agent_id | string | Yes | Agent identifier |

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | integer | No | 10 | Max conversations to return |

**Response**:
```json
{
  "conversations": [
    {
      "conversation_id": "conv_001",
      "message_count": 15,
      "messages": [
        {"role": "user", "content": "Analyze this..."},
        {"role": "assistant", "content": "Here's the analysis..."}
      ]
    }
  ],
  "count": 1
}
```

#### Set Agent Allowance
```http
POST /agents/{agent_id}/allowance?amount=100.0&epoch_hours=24.0
```

Sets or updates an agent's spending allowance.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| agent_id | string | Yes | Agent identifier |

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| amount | float | Yes | - | Allowance amount (must be positive) |
| epoch_hours | float | No | 24.0 | Allowance period in hours |

**Response**:
```json
{
  "agent_id": "agent_001",
  "amount": 100.0,
  "epoch_hours": 24.0,
  "remaining": 100.0
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Amount must be positive |
| 503 | Node not initialized |

#### Revoke Agent Allowance
```http
DELETE /agents/{agent_id}/allowance
```

Revokes an agent's spending authority.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| agent_id | string | Yes | Agent identifier |

**Response**:
```json
{
  "agent_id": "agent_001",
  "revoked": true
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Agent allowance not found |
| 503 | Node not initialized |

#### Pause Agent
```http
POST /agents/{agent_id}/pause
```

Temporarily suspends an agent.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| agent_id | string | Yes | Agent identifier |

**Response**:
```json
{
  "agent_id": "agent_001",
  "status": "paused"
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Agent not found |
| 503 | Agent registry not initialized |

#### Resume Agent
```http
POST /agents/{agent_id}/resume
```

Resumes a paused agent.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| agent_id | string | Yes | Agent identifier |

**Response**:
```json
{
  "agent_id": "agent_001",
  "status": "online"
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Agent not found |
| 503 | Agent registry not initialized |

---

### Node Ledger Endpoints

#### Get Ledger Sync Stats
```http
GET /ledger/sync/stats
```

Returns ledger synchronization statistics.

**Response**:
```json
{
  "sync_status": "synced",
  "last_sync": "2025-06-11T12:00:00Z",
  "pending_transactions": 0,
  "sync_errors": 0
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | Ledger sync not initialized |

#### Transfer FTNS
```http
POST /ledger/transfer?to_wallet=node_xyz&amount=50.0
```

Transfers FTNS to another node (signed, gossip-broadcast).

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| to_wallet | string | Yes | Recipient wallet/node ID |
| amount | float | Yes | Amount to transfer (must be positive) |

**Response**:
```json
{
  "tx_id": "tx_abc123",
  "from": "node_abc123",
  "to": "node_xyz",
  "amount": 50.0,
  "timestamp": "2025-06-11T12:00:00Z"
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Amount must be positive; Insufficient balance |
| 503 | Ledger sync not initialized |

---

### Node Staking Endpoints

#### Stake FTNS Tokens
```http
POST /staking/stake
Content-Type: application/json

{
  "amount": 100.0,
  "stake_type": "general",
  "metadata": {}
}
```

Stakes FTNS tokens for the node's identity.

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| amount | float | Yes | - | Amount of FTNS to stake (must be > 0) |
| stake_type | string | No | "general" | Type: `governance`, `validation`, `compute`, `storage`, `liquidity`, `general` |
| metadata | object | No | null | Optional metadata for the stake |

**Response**:
```json
{
  "stake_id": "stake_abc123",
  "user_id": "node_abc123",
  "amount": 100.0,
  "stake_type": "general",
  "status": "active",
  "staked_at": "2025-06-11T12:00:00Z",
  "rewards_earned": 0.0
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Invalid stake_type; Staking validation failed |
| 500 | Staking failed |
| 503 | Staking manager or node identity not initialized |

#### Request Unstake
```http
POST /staking/unstake
Content-Type: application/json

{
  "stake_id": "stake_abc123",
  "amount": null
}
```

Creates an unstake request available after the unstaking period (default: 7 days).

**Request Body**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| stake_id | string | Yes | ID of the stake to unstake |
| amount | float | No | Amount to unstake (null = full stake) |

**Response**:
```json
{
  "request_id": "req_xyz",
  "stake_id": "stake_abc123",
  "user_id": "node_abc123",
  "amount": 100.0,
  "requested_at": "2025-06-11T12:00:00Z",
  "available_at": "2025-06-18T12:00:00Z",
  "status": "pending"
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Unstake validation failed |
| 404 | Stake not found |
| 500 | Unstake failed |
| 503 | Staking manager or node identity not initialized |

#### Get Staking Status
```http
GET /staking/status
```

Returns comprehensive staking information for the node's identity.

**Response**:
```json
{
  "user_id": "node_abc123",
  "total_staked": 500.0,
  "active_stakes": [
    {
      "stake_id": "stake_abc123",
      "amount": 100.0,
      "stake_type": "general",
      "status": "active",
      "staked_at": "2025-06-11T12:00:00Z",
      "rewards_earned": 5.0,
      "rewards_claimed": 0.0
    }
  ],
  "pending_unstake_requests": [
    {
      "request_id": "req_xyz",
      "stake_id": "stake_def456",
      "amount": 50.0,
      "requested_at": "2025-06-10T12:00:00Z",
      "available_at": "2025-06-17T12:00:00Z",
      "status": "pending"
    }
  ],
  "total_rewards_earned": 25.0,
  "total_rewards_claimed": 10.0
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | Staking manager or node identity not initialized |

#### Claim Staking Rewards
```http
POST /staking/claim-rewards?stake_id=stake_abc123
```

Claims accumulated staking rewards.

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| stake_id | string | No | Specific stake ID (null = all stakes) |

**Response**:
```json
{
  "user_id": "node_abc123",
  "total_rewards_claimed": 15.0,
  "stakes_processed": 3
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Claim validation failed |
| 500 | Claim rewards failed |
| 503 | Staking manager or node identity not initialized |

#### Get Specific Stake
```http
GET /staking/stakes/{stake_id}
```

Returns details of a specific stake.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| stake_id | string | Yes | The stake ID |

**Response**:
```json
{
  "stake_id": "stake_abc123",
  "user_id": "node_abc123",
  "amount": 100.0,
  "stake_type": "general",
  "status": "active",
  "staked_at": "2025-06-11T12:00:00Z",
  "rewards_earned": 5.0,
  "rewards_claimed": 0.0,
  "last_reward_calculation": "2025-06-11T12:00:00Z",
  "lock_reason": null,
  "metadata": {}
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Stake not found |
| 503 | Staking manager not initialized |

#### Get Unstake Request
```http
GET /staking/unstake-requests/{request_id}
```

Returns details of a specific unstake request.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| request_id | string | Yes | The unstake request ID |

**Response**:
```json
{
  "request_id": "req_xyz",
  "stake_id": "stake_abc123",
  "user_id": "node_abc123",
  "amount": 100.0,
  "requested_at": "2025-06-11T12:00:00Z",
  "available_at": "2025-06-18T12:00:00Z",
  "status": "pending",
  "completed_at": null,
  "cancellation_reason": null,
  "is_available": false
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Unstake request not found |
| 503 | Staking manager not initialized |

#### Withdraw Unstaked Tokens
```http
POST /staking/withdraw/{request_id}
```

Withdraws unstaked tokens after the unstaking period.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| request_id | string | Yes | The unstake request ID |

**Response**:
```json
{
  "request_id": "req_xyz",
  "success": true,
  "amount_withdrawn": 100.0
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Withdrawal validation failed |
| 404 | Request not found |
| 500 | Withdrawal failed |
| 503 | Staking manager or node identity not initialized |

#### Cancel Unstake Request
```http
POST /staking/cancel-unstake/{request_id}?reason=Changed%20mind
```

Cancels a pending unstake request and restores tokens to active staking.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| request_id | string | Yes | The unstake request ID |

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| reason | string | No | Reason for cancellation |

**Response**:
```json
{
  "request_id": "req_xyz",
  "cancelled": true,
  "reason": "Changed mind"
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Cancellation validation failed |
| 404 | Request not found |
| 500 | Cancellation failed |
| 503 | Staking manager or node identity not initialized |

---

### Node Bridge Endpoints

The Bridge enables FTNS token transfers between the local PRSM network and external blockchain networks (e.g., Polygon).

#### Bridge Deposit
```http
POST /bridge/deposit
Content-Type: application/json

{
  "amount": 100.0,
  "chain_address": "0x742d35cc6bf4532c95a0e96a7bdc86c0b3e11888",
  "destination_chain": 137
}
```

Deposits FTNS tokens from local balance to external chain.

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| amount | float | Yes | - | Amount of FTNS to deposit (must be > 0) |
| chain_address | string | Yes | - | Destination on-chain address |
| destination_chain | integer | No | 137 | Destination chain ID (137 = Polygon mainnet) |

**Response**:
```json
{
  "success": true,
  "transaction": {
    "transaction_id": "tx_bridge_001",
    "direction": "deposit",
    "user_id": "node_abc123",
    "chain_address": "0x742d35...",
    "amount": "100000000000000000000",
    "source_chain": 0,
    "destination_chain": 137,
    "status": "pending",
    "source_tx_hash": null,
    "destination_tx_hash": null,
    "fee_amount": "100000000000000000",
    "created_at": "2025-06-11T12:00:00Z",
    "updated_at": "2025-06-11T12:00:00Z",
    "completed_at": null,
    "error_message": null
  }
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Insufficient balance; Invalid address; Amount outside limits |
| 500 | Bridge deposit failed |
| 503 | FTNS bridge or node identity not initialized |

#### Bridge Withdraw
```http
POST /bridge/withdraw
Content-Type: application/json

{
  "amount": 50.0,
  "chain_address": "0x742d35cc6bf4532c95a0e96a7bdc86c0b3e11888",
  "source_chain": 137
}
```

Withdraws FTNS tokens from external chain to local balance.

**Request Body**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| amount | float | Yes | - | Amount of FTNS to withdraw (must be > 0) |
| chain_address | string | Yes | - | Source on-chain address |
| source_chain | integer | No | 137 | Source chain ID (137 = Polygon mainnet) |

**Response**:
```json
{
  "success": true,
  "transaction": {
    "transaction_id": "tx_bridge_002",
    "direction": "withdraw",
    "user_id": "node_abc123",
    "chain_address": "0x742d35...",
    "amount": "50000000000000000000",
    "source_chain": 137,
    "destination_chain": 0,
    "status": "pending",
    "source_tx_hash": "0xabc123...",
    "destination_tx_hash": null,
    "fee_amount": "50000000000000000",
    "created_at": "2025-06-11T12:00:00Z",
    "updated_at": "2025-06-11T12:00:00Z",
    "completed_at": null,
    "error_message": null
  }
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 400 | Insufficient balance; Invalid address; Amount outside limits |
| 500 | Bridge withdraw failed |
| 503 | FTNS bridge or node identity not initialized |

#### Get Bridge Status
```http
GET /bridge/status
```

Returns bridge status and pending operations.

**Response**:
```json
{
  "stats": {
    "total_deposited": "10000000000000000000000",
    "total_withdrawn": "5000000000000000000000",
    "total_fees_collected": "150000000000000000000",
    "pending_transactions": 2,
    "completed_transactions": 150,
    "failed_transactions": 3
  },
  "limits": {
    "min_amount": "1000000000000000000",
    "max_amount": "100000000000000000000000",
    "daily_limit": "1000000000000000000000000",
    "fee_bps": 100
  },
  "pending_transactions": [],
  "pending_count": 2
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 500 | Failed to get bridge status |
| 503 | FTNS bridge not initialized |

#### Get Bridge Transaction
```http
GET /bridge/transactions/{tx_id}
```

Returns status of a specific bridge transaction.

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tx_id | string | Yes | Transaction ID to look up |

**Response**:
```json
{
  "transaction": {
    "transaction_id": "tx_bridge_001",
    "direction": "deposit",
    "user_id": "node_abc123",
    "chain_address": "0x742d35...",
    "amount": "100000000000000000000",
    "source_chain": 0,
    "destination_chain": 137,
    "status": "completed",
    "source_tx_hash": "0xdef456...",
    "destination_tx_hash": "0xabc123...",
    "fee_amount": "100000000000000000",
    "created_at": "2025-06-11T12:00:00Z",
    "updated_at": "2025-06-11T12:30:00Z",
    "completed_at": "2025-06-11T12:30:00Z",
    "error_message": null
  }
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 404 | Transaction not found |
| 503 | FTNS bridge not initialized |

#### List Bridge Transactions
```http
GET /bridge/transactions?limit=50
```

Lists bridge transactions for the current user.

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | integer | No | 50 | Max transactions (max: 200) |

**Response**:
```json
{
  "transactions": [
    {
      "transaction_id": "tx_bridge_001",
      "direction": "deposit",
      "amount": "100000000000000000000",
      "status": "completed",
      "created_at": "2025-06-11T12:00:00Z"
    }
  ],
  "count": 1
}
```

**Error Codes**:
| Code | Description |
|------|-------------|
| 503 | FTNS bridge or node identity not initialized |

---

### Node Storage Endpoints

#### Get Storage Stats
```http
GET /storage/stats
```

Returns storage provider statistics.

**Response**:
```json
{
  "available": true,
  "pledged_gb": 1000.0,
  "used_gb": 450.5,
  "pinned_count": 150,
  "ftns_earned": 25.50
}
```

**Response (when not initialized)**:
```json
{
  "available": false,
  "pledged_gb": 0,
  "used_gb": 0,
  "pinned_count": 0,
  "message": "Storage provider not initialized"
}
```

---

### Node WebSocket Endpoints

#### Status WebSocket
```websocket
WS /ws/status
```

WebSocket endpoint for real-time status updates.

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/status');
```

**Client Messages**:

Send heartbeat:
```json
{
  "type": "heartbeat"
}
```

Request current status:
```json
{
  "type": "get_status"
}
```

**Server Messages**:

Heartbeat acknowledgment:
```json
{
  "type": "heartbeat_ack",
  "timestamp": "2025-06-11T12:00:00Z"
}
```

Status update:
```json
{
  "type": "status_update",
  "data": {
    "node_id": "node_abc123",
    "status": "online"
  },
  "timestamp": "2025-06-11T12:00:00Z"
}
```

---

### Node Authentication Endpoints

#### Verify JWT Token
```http
GET /auth/verify
Authorization: Bearer <jwt_token>
```

Verifies JWT token and returns user information.

**Response**:
```json
{
  "valid": true,
  "user_id": "user_123",
  "username": "researcher1",
  "role": "researcher",
  "permissions": ["read", "submit_queries"]
}
```

---

## Platform API Endpoints

### NWTN (Neural Web of Thought Networks)

#### Submit Research Query
```http
POST /api/v1/nwtn/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "Analyze the impact of climate change on marine ecosystems",
  "domain": "environmental_science",
  "methodology": "comprehensive_analysis",
  "max_iterations": 5,
  "include_citations": true,
  "seal_enhancement": {
    "enabled": true,
    "autonomous_improvement": true,
    "target_learning_gain": 0.15,
    "restem_methodology": true
  }
}
```

**Response**:
```json
{
  "session_id": "sess_abc123",
  "status": "processing",
  "query": "Analyze the impact of climate change on marine ecosystems",
  "estimated_completion": "2025-06-11T14:30:00Z",
  "cost_estimate": {
    "ftns_tokens": 150,
    "usd_equivalent": 0.75
  },
  "seal_status": {
    "enhancement_enabled": true,
    "autonomous_improvement_active": true,
    "estimated_learning_gain": 0.15,
    "self_edit_generation_rate": 3784
  }
}
```

#### Get Session Status
```http
GET /api/v1/nwtn/sessions/{session_id}
Authorization: Bearer <token>
```

**Response**:
```json
{
  "session_id": "sess_abc123",
  "status": "completed",
  "progress": 100,
  "results": {
    "summary": "Climate change significantly impacts marine ecosystems through...",
    "key_findings": [
      "Ocean acidification affects coral reef biodiversity",
      "Rising temperatures alter fish migration patterns"
    ],
    "citations": [
      {
        "title": "Marine Ecosystem Response to Climate Change",
        "authors": ["Smith, J.", "Doe, A."],
        "journal": "Nature Climate Change",
        "year": 2024
      }
    ],
    "confidence_score": 0.92
  },
  "cost_actual": {
    "ftns_tokens": 145,
    "usd_equivalent": 0.73
  },
  "seal_performance": {
    "autonomous_improvements_applied": 3,
    "learning_gain_achieved": 0.16,
    "knowledge_incorporation_improvement": 0.135,
    "self_edit_examples_generated": 127,
    "restem_policy_updates": 5
  }
}
```

#### List User Sessions
```http
GET /api/v1/nwtn/sessions?limit=20&offset=0&status=completed
Authorization: Bearer <token>
```

### SEAL Technology API

#### Get SEAL Performance Metrics
```http
GET /api/v1/seal/metrics
Authorization: Bearer <token>
```

**Response**:
```json
{
  "seal_system_status": "active",
  "production_metrics": {
    "knowledge_incorporation_baseline": 0.335,
    "knowledge_incorporation_current": 0.470,
    "improvement_percentage": 0.403,
    "few_shot_learning_success_rate": 0.725,
    "self_edit_generation_rate": 3784,
    "autonomous_improvement_cycles_completed": 1547
  },
  "real_time_performance": {
    "restem_policy_updates_per_second": 15327,
    "seal_reward_calculations_per_second": 24130,
    "autonomous_improvement_rate": 0.187
  }
}
```

#### Trigger SEAL Autonomous Improvement
```http
POST /api/v1/seal/improve
Authorization: Bearer <token>
Content-Type: application/json

{
  "domain": "biomedical_research",
  "target_improvement": 0.20,
  "improvement_strategy": "restem_methodology",
  "max_iterations": 10
}
```

#### Get SEAL Enhancement Status for Session
```http
GET /api/v1/seal/sessions/{session_id}/status
Authorization: Bearer <token>
```

### Model Management

#### List Available Models
```http
GET /api/v1/models?category=language&provider=openai&limit=50
```

**Response**:
```json
{
  "models": [
    {
      "id": "model_123",
      "name": "GPT-4 Turbo",
      "provider": "openai",
      "category": "language",
      "description": "Advanced language model for complex reasoning",
      "pricing": {
        "input_tokens": 0.01,
        "output_tokens": 0.03,
        "currency": "USD"
      },
      "capabilities": ["text_generation", "analysis", "reasoning"],
      "context_length": 128000,
      "available": true
    }
  ],
  "total": 45,
  "page": 1,
  "pages": 3
}
```

#### Get Model Details
```http
GET /api/v1/models/{model_id}
```

#### Train Custom Model
```http
POST /api/v1/models/train
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Custom Research Model",
  "base_model": "llama2_7b",
  "training_data": "ipfs://QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "training_parameters": {
    "epochs": 3,
    "learning_rate": 0.0001,
    "batch_size": 16
  },
  "domain": "biomedical_research"
}
```

## Health & Monitoring

### System Health
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "environment": "production",
  "timestamp": "2025-06-11T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12.5,
      "last_check": "2025-06-11T12:00:00Z"
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2.1,
      "last_check": "2025-06-11T12:00:00Z"
    },
    "ipfs": {
      "status": "healthy",
      "response_time_ms": 45.3,
      "last_check": "2025-06-11T12:00:00Z"
    }
  }
}
```

### Liveness & Readiness Probes
```http
GET /health/liveness    # Kubernetes liveness probe
GET /health/readiness   # Kubernetes readiness probe
```

### Detailed Status (Authenticated)
```http
GET /health/detailed
Authorization: Bearer <token>
```

### Metrics (Prometheus)
```http
GET /health/metrics
```

## User Management

### Get Current User Profile
```http
GET /api/v1/users/me
Authorization: Bearer <token>
```

**Response**:
```json
{
  "id": "user_123",
  "username": "researcher1",
  "email": "researcher@university.edu",
  "role": "researcher",
  "organization": "University Research Lab",
  "created_at": "2025-01-15T10:30:00Z",
  "last_login": "2025-06-11T08:15:00Z",
  "ftns_balance": 1250.50,
  "research_credits": 500,
  "permissions": [
    "submit_queries",
    "train_models",
    "access_marketplace"
  ]
}
```

### Update User Profile
```http
PUT /api/v1/users/me
Authorization: Bearer <token>
Content-Type: application/json

{
  "organization": "Updated Research Institution",
  "bio": "Climate science researcher focused on marine ecosystems",
  "research_interests": ["climate_change", "marine_biology", "data_analysis"]
}
```

### Change Password
```http
POST /api/v1/users/me/password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "old_password",
  "new_password": "new_secure_password"
}
```

## Marketplace

### Browse Models
```http
GET /api/v1/marketplace/models?category=language&featured=true&limit=20
```

**Response**:
```json
{
  "models": [
    {
      "id": "marketplace_model_123",
      "name": "Advanced Research Assistant",
      "description": "Specialized model for scientific literature analysis",
      "creator": {
        "username": "ai_researcher",
        "verified": true
      },
      "category": "language",
      "tags": ["research", "analysis", "scientific"],
      "pricing": {
        "ftns_per_request": 5,
        "bulk_discount": 0.1
      },
      "stats": {
        "downloads": 1547,
        "rating": 4.8,
        "reviews": 89
      },
      "featured": true,
      "created_at": "2025-05-15T14:20:00Z"
    }
  ],
  "total": 156,
  "filters": {
    "categories": ["language", "vision", "audio", "scientific"],
    "providers": ["community", "verified", "official"]
  }
}
```

### Rent Model
```http
POST /api/v1/marketplace/models/{model_id}/rent
Authorization: Bearer <token>
Content-Type: application/json

{
  "duration_hours": 24,
  "max_requests": 1000
}
```

### Submit Model to Marketplace
```http
POST /api/v1/marketplace/models
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Custom Scientific Model",
  "description": "Specialized for climate data analysis",
  "category": "scientific",
  "model_file": "ipfs://QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "pricing": {
    "ftns_per_request": 10,
    "revenue_share": 0.7
  },
  "tags": ["climate", "data_analysis", "python"]
}
```

## Governance

### List Active Proposals
```http
GET /api/v1/governance/proposals?status=active&limit=10
```

**Response**:
```json
{
  "proposals": [
    {
      "id": "prop_123",
      "title": "Increase Research Grant Pool",
      "description": "Proposal to allocate additional FTNS tokens for research grants",
      "proposer": "governance_council",
      "status": "active",
      "voting_ends": "2025-06-18T23:59:59Z",
      "votes": {
        "yes": 1250000,
        "no": 340000,
        "abstain": 89000
      },
      "quorum_required": 1000000,
      "threshold": 0.6,
      "category": "funding"
    }
  ]
}
```

### Submit Proposal
```http
POST /api/v1/governance/proposals
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "New Research Domain Integration",
  "description": "Proposal to add quantum computing as a supported research domain",
  "category": "technical",
  "implementation_plan": "Detailed plan for quantum computing integration...",
  "budget_required": 50000
}
```

### Vote on Proposal
```http
POST /api/v1/governance/proposals/{proposal_id}/vote
Authorization: Bearer <token>
Content-Type: application/json

{
  "vote": "yes",
  "voting_power": 1000,
  "comment": "This proposal aligns with our research goals"
}
```

## Web3 & FTNS

### Get FTNS Balance
```http
GET /api/v1/web3/balance
Authorization: Bearer <token>
```

**Response**:
```json
{
  "balance": {
    "ftns_tokens": 1250.50,
    "staked_tokens": 500.00,
    "available_tokens": 750.50
  },
  "wallet_address": "0x742d35cc6bf4532c95a0e96a7bdc86c0b3e11888",
  "network": "polygon",
  "last_updated": "2025-06-11T12:00:00Z"
}
```

### Transfer FTNS Tokens
```http
POST /api/v1/web3/transfer
Authorization: Bearer <token>
Content-Type: application/json

{
  "to_address": "0x456def789ghi012jkl345mno678pqr901stu234vw",
  "amount": 100.0,
  "note": "Research collaboration payment"
}
```

### Purchase FTNS with Fiat
```http
POST /api/v1/payments/purchase
Authorization: Bearer <token>
Content-Type: application/json

{
  "amount_usd": 50.00,
  "payment_method": "stripe",
  "payment_token": "tok_1234567890abcdef"
}
```

### Transaction History
```http
GET /api/v1/web3/transactions?limit=50&type=all
Authorization: Bearer <token>
```

## Security

### Security Status
```http
GET /api/v1/security/status
Authorization: Bearer <token>
```

**Response**:
```json
{
  "security_level": "normal",
  "threats_detected": 0,
  "last_scan": "2025-06-11T11:45:00Z",
  "account_status": "verified",
  "recent_activity": {
    "login_attempts": 1,
    "successful_logins": 1,
    "api_calls": 156,
    "suspicious_activity": false
  }
}
```

### Enable Two-Factor Authentication
```http
POST /api/v1/security/2fa/enable
Authorization: Bearer <token>
```

### Generate API Key
```http
POST /api/v1/security/api-keys
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Research Project API Key",
  "permissions": ["read", "submit_queries"],
  "expires_in_days": 90
}
```

## WebSocket API

### Connection
```javascript
wss://api.prsm.org/ws?token=<jwt_token>
```

### Message Format
```json
{
  "type": "message_type",
  "payload": {
    "key": "value"
  },
  "timestamp": "2025-06-11T12:00:00Z"
}
```

### Subscribe to Session Updates
```json
{
  "type": "subscribe",
  "payload": {
    "channel": "session_updates",
    "session_id": "sess_abc123"
  }
}
```

### Real-time Progress Updates
```json
{
  "type": "session_progress",
  "payload": {
    "session_id": "sess_abc123",
    "progress": 45,
    "current_step": "model_reasoning",
    "estimated_completion": "2025-06-11T14:30:00Z"
  }
}
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "query",
      "issue": "Query text is required"
    },
    "request_id": "req_abc123xyz"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_FAILED` - Invalid credentials
- `INSUFFICIENT_BALANCE` - Not enough FTNS tokens
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `VALIDATION_ERROR` - Invalid input data
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist
- `PERMISSION_DENIED` - Insufficient permissions
- `SERVICE_UNAVAILABLE` - External service temporarily unavailable

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1623456789
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/nwtn/query` | 100 requests | 1 hour |
| `/api/v1/models/train` | 5 requests | 1 hour |
| `/api/v1/auth/login` | 5 attempts | 15 minutes |
| `/api/v1/marketplace/*` | 1000 requests | 1 hour |
| General API | 10000 requests | 1 hour |

### Rate Limit Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "retry_after": 3600
  }
}
```

## SDK Examples

### Python SDK
```python
from prsm_sdk import PRSMClient

# Initialize client
client = PRSMClient(
    api_key="your_api_key",
    base_url="https://api.prsm.org"
)

# Submit research query
response = client.nwtn.submit_query(
    query="Analyze protein folding mechanisms",
    domain="biochemistry",
    max_iterations=3
)

print(f"Session ID: {response.session_id}")

# Monitor progress
for update in client.nwtn.watch_session(response.session_id):
    print(f"Progress: {update.progress}%")
    if update.status == "completed":
        print(f"Results: {update.results}")
        break
```

### JavaScript SDK
```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.prsm.org'
});

// Submit research query
const response = await client.nwtn.submitQuery({
  query: 'Analyze protein folding mechanisms',
  domain: 'biochemistry',
  maxIterations: 3
});

console.log(`Session ID: ${response.sessionId}`);

// WebSocket for real-time updates
const ws = client.createWebSocket();
ws.subscribe('session_updates', response.sessionId, (update) => {
  console.log(`Progress: ${update.progress}%`);
  if (update.status === 'completed') {
    console.log('Results:', update.results);
  }
});
```

### cURL Examples

#### Submit Query
```bash
curl -X POST "https://api.prsm.org/api/v1/nwtn/query" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze climate change impacts",
    "domain": "environmental_science",
    "max_iterations": 5
  }'
```

#### Check Session Status
```bash
curl -X GET "https://api.prsm.org/api/v1/nwtn/sessions/sess_abc123" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### Get FTNS Balance
```bash
curl -X GET "https://api.prsm.org/api/v1/web3/balance" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Document Information

**Version**: 2.0
**Last Updated**: March 5, 2026
**API Version**: v1
**Contact**: api-support@prsm.org

### Endpoint Summary

This documentation covers **two API layers**:

#### Node API (Local Node Management)
| Category | Endpoints | Description |
|----------|-----------|-------------|
| Core | 6 | Status, peers, balance, health, transactions |
| Compute | 3 | Job submission, status, stats |
| Content | 5 | Upload, search, retrieve, index stats |
| Agent | 10 | List, search, manage, allowance control |
| Ledger | 2 | Sync stats, transfers |
| Staking | 8 | Stake, unstake, withdraw, rewards |
| Bridge | 5 | Deposit, withdraw, status, transactions |
| Storage | 1 | Storage stats |
| WebSocket | 1 | Real-time status updates |
| Auth | 1 | Token verification |
| **Total Node API** | **42** | |

#### Platform API (Cloud Services)
| Category | Endpoints | Description |
|----------|-----------|-------------|
| NWTN | 3 | Query processing, sessions |
| SEAL | 3 | Performance metrics, improvement |
| Models | 3 | List, details, training |
| Health | 4 | System health, probes, metrics |
| Users | 3 | Profile, password management |
| Marketplace | 3 | Browse, rent, submit models |
| Governance | 3 | Proposals, voting |
| Web3/FTNS | 4 | Balance, transfer, purchase, history |
| Security | 3 | Status, 2FA, API keys |
| **Total Platform API** | **29** | |

**Grand Total: 71 documented endpoints**

### Related Documentation
- [WebSocket API](#websocket-api) - Real-time communication endpoints
- [Authentication](#authentication) - JWT Bearer and API Key authentication
- [FTNS Tokenomics Documentation](FTNS_API_DOCUMENTATION.md)
- [SDK Documentation](https://docs.prsm.org/sdks)
- [Postman Collection](https://docs.prsm.org/postman)

---

*For support and questions, please contact our API support team at api-support@prsm.org or visit our [Developer Portal](https://developers.prsm.org).*