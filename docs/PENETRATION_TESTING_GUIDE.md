# PRSM Penetration Testing Guide

> **Scope note:** This guide was authored January 2026 and reflects application-layer testing priorities (API, auth, FTNS double-spend, rate limiting, etc.) as they stood pre-v1.6. The application-layer content below remains directionally useful, but a current pen-test engagement against PRSM must **additionally cover the on-chain and protocol-layer surface** that is either new since this guide was written or that became the primary attack surface post-v1.6:
>
> - **Phase 1 smart contracts on Base:** `ProvenanceRegistry.sol` and `RoyaltyDistributor.sol` — critical financial surface. See [`2026-04-11-phase1.3-completion-plan.md`](2026-04-11-phase1.3-completion-plan.md) for bake-in status and [`archive/2026-04-10-phase1.1-codex-fixes-plan.md`](archive/2026-04-10-phase1.1-codex-fixes-plan.md) + [`archive/2026-04-10-phase1.2-codex-rereview-fixes-plan.md`](archive/2026-04-10-phase1.2-codex-rereview-fixes-plan.md) for the 7-round codex audit trail already completed. Recommend a smart-contract-experienced firm (Trail of Bits, Zellic, OpenZeppelin, Certora, or Cantina) for this surface specifically; application-only firms will miss contract-specific attack classes.
> - **Protocol-layer components:** libtorrent seeding / challenge-response proof-of-storage, Wasmtime SPRK sandbox (WASM module escape, fuel-metering bypass), Ed25519 signed-receipt forgery in Phase 2 remote compute dispatch, topology-randomization assumptions, TEE attestation posture.
> - **Activation-inversion attack surface** on sharded inference (per `PRSM_Vision.md` §7 honest limits and research track item R3).
>
> **Related docs in this suite:**
> - [`SECURITY_HARDENING.md`](SECURITY_HARDENING.md) — hardening policy
> - [`SECURITY_HARDENING_CHECKLIST.md`](SECURITY_HARDENING_CHECKLIST.md) — implementation verification
> - [`SECURITY_CONFIGURATION_AUDIT.md`](SECURITY_CONFIGURATION_AUDIT.md) — audit snapshot
> - [`REMEDIATION_HARDENING_MASTER_PLAN.md`](REMEDIATION_HARDENING_MASTER_PLAN.md) — historical 12-week remediation sprint (see post-v1.6 status audit at top)
> - [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) Phase 6 + Phase 7 for the protocol-layer hardening scope still to ship

## Overview

This document provides guidance for third-party security auditors conducting penetration testing on the PRSM platform. It documents security controls, potential attack vectors, and testing priorities. Originally authored January 2026 for application-layer focus; see the scope note above for the broader surface current engagements must cover.

**Recommended Testing Firms by surface:**

*Application layer (API, auth, FTNS application logic, rate limiting, infrastructure):*
- NCC Group
- Trail of Bits
- Bishop Fox
- Cure53
- Doyensec

*Smart-contract layer (Phase 1 `ProvenanceRegistry` / `RoyaltyDistributor`, plus Phase 2+ compute escrow and Phase 7+ slashing contracts):*
- Trail of Bits (also covers application layer)
- Zellic
- OpenZeppelin Security
- Certora (formal verification)
- Cantina (distributed review)

Current Phase 1 contracts already passed 7 rounds of independent codex review reaching SAFE TO DEPLOY verdict before Sepolia deployment; a human-auditor engagement is still recommended before mainnet high-value operation per Risk Register A1 mitigation.

---

## Executive Summary

PRSM is a peer-to-peer infrastructure protocol for open-source collaboration (compute / storage / data, not an AI model hosting platform). The January 2026 application-layer security surface included:
- JWT-based authentication with token revocation
- FTNS token economy with double-spend prevention (application-layer — Phase 1 on-chain surface also in scope now)
- Redis-backed rate limiting
- PostgreSQL for persistent storage
- Post-quantum cryptography (hybrid mode; current posture: R6 research-track defer per [`2026-04-14-phase4plus-research-track.md`](2026-04-14-phase4plus-research-track.md))
- Enterprise SSO (SAML/OIDC)

Additional surface introduced since January 2026 (must be included in scope for a current engagement):
- On-chain `ProvenanceRegistry.sol` + `RoyaltyDistributor.sol` on Base Sepolia / Base mainnet
- BitTorrent-based `libtorrent` data layer with challenge-response proof-of-storage
- Wasmtime WASM sandbox (SPRK runtime) with fuel-metering resource limits
- Ed25519 signed receipts for cross-node compute dispatch (Phase 2)
- TEE attestation surface (SGX / TDX / SEV-SNP / Secure Enclave) — current posture: node-level attestation only; per-inference attestation deferred to Phase 2.1+

---

## Scope Definition

### In-Scope Systems (current — post-v1.6 / Phase 1 bake-in)

| Component | Technology | Priority | Notes |
|-----------|------------|----------|---|
| **`ProvenanceRegistry.sol`** | Solidity on Base (Sepolia + mainnet) | **CRITICAL** | Phase 1. Source of truth for content ownership. Exploit = IP integrity loss. Already passed 7 rounds codex review. |
| **`RoyaltyDistributor.sol`** | Solidity on Base (Sepolia + mainnet) | **CRITICAL** | Phase 1. Atomic three-way FTNS split. Exploit = drain protocol funds. Formal verification recommended. |
| **FTNS ERC-20 contract** | Solidity on Base mainnet (`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`) | **CRITICAL** | Supply integrity; mint-authority bounds. |
| Authentication | JWT/OIDC/SAML | CRITICAL | Application-layer; unchanged from Jan 2026 scope. |
| FTNS application-layer ledger | PostgreSQL | CRITICAL | Must be tested *in addition to* on-chain surface — mixed local/on-chain payment path. |
| API Server | FastAPI/Python | HIGH | 536-line `main.py` post-refactor (was 2204). |
| Rate Limiting | Redis | HIGH | Per `SECURITY_HARDENING_CHECKLIST.md`; verify Redis rate-limiter actually shipped (flagged as "outstanding" in REMEDIATION post-v1.6 audit). |
| **libtorrent data layer** | C++ via Python bindings | HIGH | Challenge-response proof-of-storage, signed manifests, BEP 27 private torrents. New surface since Jan 2026. |
| **Wasmtime SPRK runtime** | Rust/WASM | HIGH | Fuel-metering enforcement, WASI exposure, sandbox escape. New surface since Jan 2026. |
| **Ed25519 signed receipts** (Phase 2) | Python ed25519 | HIGH | Remote-compute-dispatch attestation. Forgery = free compute + cost to requester. |
| WebSocket Server | FastAPI | MEDIUM | Transport layer; libp2p replacement planned Phase 6. |

### Out-of-Scope

- Third-party AI model APIs (OpenAI, Anthropic, OpenRouter) — PRSM does not host models; the reasoning layer is a third-party LLM
- Base L2 chain security itself (inherited from Ethereum)
- Cloud provider infrastructure (RunPod, Lambda, CoreWeave — used by T3 operators but PRSM does not operate)
- FHE / MPC primitives (research track R1, R2 — not in current product)

---

## Authentication & Authorization

### JWT Authentication

**Implementation:** `prsm/core/auth/jwt_handler.py`

**Security Controls:**
- Algorithm whitelist: HS256, HS384, HS512 only
- `none` algorithm explicitly rejected
- Required claims: `exp`, `sub`, `jti`
- Token revocation via Redis + PostgreSQL
- 15-60 minute access token lifetime

**Test Scenarios:**
1. Algorithm confusion attack (try `none` algorithm)
2. Token with missing required claims
3. Expired token acceptance
4. Revoked token acceptance
5. Token replay attacks
6. JWT secret brute force (if weak)
7. Future `iat` handling

**Test Endpoints:**
```
POST /api/v1/auth/login
POST /api/v1/auth/refresh
POST /api/v1/auth/logout
GET /api/v1/auth/me
```

### Enterprise SSO (OIDC)

**Implementation:** `prsm/core/auth/enterprise/sso_provider.py`

**Security Controls:**
- JWKS-based signature verification
- Issuer and audience validation
- Required claims: `exp`, `iat`, `sub`, `aud`
- PKCE support for authorization code flow

**Test Scenarios:**
1. JWKS key injection
2. ID token without signature
3. Audience mismatch
4. Issuer spoofing
5. State parameter manipulation
6. Authorization code replay

### SAML Authentication

**Implementation:** `prsm/core/auth/enterprise/sso_provider.py`

**Security Controls:**
- XML signature verification
- Defusedxml for XXE prevention
- Assertion validation

**Test Scenarios:**
1. XXE injection in SAML response
2. Signature wrapping attacks
3. Comment injection
4. Replay attacks

---

## Financial System (FTNS)

### Double-Spend Prevention

**Implementation:** `prsm/economy/tokenomics/atomic_ftns_service.py`

**Security Controls:**
- `SELECT FOR UPDATE` row-level locking
- Optimistic concurrency control (version column)
- Idempotency keys with 24-hour TTL
- Database CHECK constraints for non-negative balance

**Test Scenarios:**
1. Concurrent deduction race condition
2. Negative balance exploitation
3. Idempotency key collision
4. Version mismatch exploitation
5. Lock timeout attacks
6. Transaction isolation bypass

**Critical Tables:**
```sql
ftns_balances (user_id, balance, locked_balance, version)
ftns_transactions (id, from_user_id, to_user_id, amount, idempotency_key)
ftns_idempotency_keys (idempotency_key, transaction_id, expires_at)
```

**Test Endpoints:**
```
POST /api/v1/ftns/transfer
POST /api/v1/ftns/deduct
GET /api/v1/ftns/balance/{user_id}
```

---

## Rate Limiting

### Sliding Window Implementation

**Implementation:** `prsm/core/security/advanced_rate_limiting.py`

**Security Controls:**
- Redis sorted sets for accurate tracking
- Per-user and per-IP limits
- Tier-based limits (anonymous, free, pro, enterprise)
- Burst protection with token bucket

**Rate Limits:**
| Tier | Requests/Min | Requests/Hour |
|------|--------------|---------------|
| Anonymous | 10 | 100 |
| Free | 60 | 1,000 |
| Pro | 300 | 10,000 |
| Enterprise | 1,000 | 50,000 |

**Test Scenarios:**
1. Rate limit bypass via IP rotation
2. User ID spoofing to avoid limits
3. Distributed rate limit evasion
4. Redis unavailability fallback

---

## Input Validation

### SQL Injection Prevention

**Implementation:** SQLAlchemy ORM + parameterized queries

**Test Areas:**
- All user input fields
- Search/filter parameters
- Sorting parameters
- Pagination parameters

**Test Payloads:**
```
' OR '1'='1
'; DROP TABLE users;--
1; SELECT * FROM ftns_balances--
```

### XSS Prevention

**Implementation:** JSON responses, Content-Type enforcement

**Test Areas:**
- User-provided text fields
- Error messages
- API responses

### Request Size Limits

**Implementation:** `prsm/interface/api/middleware.py:ContentValidationMiddleware`

**Limits:**
- Max request size: 10MB
- JSON depth limit: Configured
- Array size limit: Configured

**Test Scenarios:**
1. Large payload DoS
2. Deeply nested JSON
3. Large array submission

---

## API Security

### Security Headers

**Implementation:** `prsm/interface/api/standards.py`

**Headers Applied:**
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'; ...
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Resource-Policy: same-origin
```

**Test Scenarios:**
1. Clickjacking via iframe
2. MIME type sniffing
3. CSP bypass attempts

### CORS Configuration

**Implementation:** `prsm/interface/api/middleware.py`

**Allowed Origins:**
- `https://prsm.ai`
- `https://app.prsm.ai`

**Test Scenarios:**
1. CORS bypass with null origin
2. Subdomain wildcard exploitation
3. Credential leakage

---

## Cryptography

### Post-Quantum Cryptography

**Implementation:** `prsm/core/cryptography/post_quantum_production.py`

**Modes:**
- DISABLED: Classical only
- REAL: Full PQC (requires liboqs)
- HYBRID: Classical + PQC

**Test Scenarios:**
1. Downgrade attack to classical
2. Key exchange manipulation
3. Side-channel analysis (if applicable)

### Key Management

**Test Areas:**
- Secret key storage (environment variables)
- Key rotation procedures
- Key length validation (min 32 chars)

---

## Infrastructure

### Database Security

**Implementation:** PostgreSQL with SSL

**Test Areas:**
1. SQL injection (covered above)
2. Connection string exposure
3. Privilege escalation
4. Backup security

### Redis Security

**Implementation:** Redis with authentication

**Test Areas:**
1. Unauthorized access
2. Data exfiltration
3. Command injection
4. Cache poisoning

### TLS Configuration

**Implementation:** `prsm/core/security/tls_config.py`

**Requirements:**
- TLS 1.2+ minimum
- Strong cipher suites
- Certificate verification

**Test Scenarios:**
1. Protocol downgrade
2. Cipher suite weaknesses
3. Certificate validation bypass

---

## Attack Vectors Summary

### Critical Priority
1. **JWT Authentication Bypass** - Test algorithm confusion, token forgery
2. **Double-Spend Attack** - Race conditions on FTNS transfers
3. **SQL Injection** - All user input endpoints
4. **OIDC Token Forgery** - JWKS manipulation

### High Priority
5. **Rate Limit Bypass** - Distributed attacks
6. **SSRF via IPFS** - Internal network access
7. **XSS in Error Messages** - Reflected content
8. **Session Hijacking** - Token theft scenarios

### Medium Priority
9. **Information Disclosure** - Error verbosity
10. **DoS via Resource Exhaustion** - Large payloads
11. **CORS Misconfiguration** - Data leakage
12. **Insecure Direct Object Reference** - User data access

---

## Test Environment Setup

### Required Access
- API endpoint access
- Test user accounts (all tiers)
- Database read access (for verification)
- Redis access (for state inspection)

### Test Accounts
```
pentest_admin@prsm.ai - Admin tier
pentest_pro@prsm.ai - Pro tier
pentest_free@prsm.ai - Free tier
pentest_anon - Anonymous access
```

### Initial FTNS Allocation
- 10,000 FTNS per test account
- Reset capability between tests

---

## Reporting Requirements

### Vulnerability Format
```
Title: [Vulnerability Name]
Severity: Critical/High/Medium/Low
CVSS Score: X.X
Affected Component: [file path or endpoint]
Description: [Detailed description]
Steps to Reproduce: [Numbered steps]
Impact: [Business impact]
Recommendation: [Fix suggestion]
```

### Required Deliverables
1. Executive Summary (1-2 pages)
2. Technical Report (full details)
3. Vulnerability List (with CVSS)
4. Remediation Roadmap
5. Retest Confirmation (after fixes)

---

## Contact Information

**Security contact:** to be published on foundation website at launch
**Engineering Lead:** Foundation CTO (role to be filled; see `PRSM_Vision.md` §12 Team)
**Escalation:** Foundation board via Prismatica liaison (see `Prismatica_Vision.md` §9 on Foundation–Prismatica relationship)

---

## Appendix: Security Test Coverage Matrix

| Category | Test Count | Automated | Manual |
|----------|------------|-----------|--------|
| Authentication | 15 | 10 | 5 |
| Authorization | 10 | 6 | 4 |
| FTNS Security | 12 | 8 | 4 |
| Input Validation | 20 | 15 | 5 |
| Rate Limiting | 8 | 5 | 3 |
| API Security | 12 | 8 | 4 |
| Cryptography | 6 | 2 | 4 |
| Infrastructure | 10 | 4 | 6 |
| **Total** | **93** | **58** | **35** |

---

*Document Version: 1.1*
*Originally authored: January 2026. Cross-reference + on-chain scope expansion pass: 2026-04-16.*
*Classification: Security Auditor — Confidential*
*Note: the v1.0 January 2026 application-layer content is retained verbatim below the scope / exec-summary banners. On-chain and protocol-layer scope (Phase 1 contracts, libtorrent, Wasmtime SPRK, Ed25519 receipts) must be added to any current engagement — see scope note at top of document.*
