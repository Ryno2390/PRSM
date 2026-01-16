# PRSM Penetration Testing Guide

## Overview

This document provides guidance for third-party security auditors conducting penetration testing on the PRSM platform. It documents security controls, potential attack vectors, and testing priorities.

**Recommended Testing Firms:**
- NCC Group
- Trail of Bits
- Bishop Fox
- Cure53
- Doyensec

---

## Executive Summary

PRSM is a decentralized AI research platform with:
- JWT-based authentication with token revocation
- FTNS token economy with double-spend prevention
- Redis-backed rate limiting
- PostgreSQL for persistent storage
- Post-quantum cryptography (hybrid mode)
- Enterprise SSO (SAML/OIDC)

---

## Scope Definition

### In-Scope Systems

| Component | Technology | Priority |
|-----------|------------|----------|
| API Server | FastAPI/Python | HIGH |
| Authentication | JWT/OIDC/SAML | CRITICAL |
| FTNS Token System | PostgreSQL | CRITICAL |
| Rate Limiting | Redis | HIGH |
| WebSocket Server | FastAPI | MEDIUM |
| IPFS Integration | External | LOW |
| Web3 Bridge | Ethereum/Smart Contracts | MEDIUM |

### Out-of-Scope
- Third-party AI model APIs (OpenAI, Anthropic)
- IPFS network infrastructure
- Cloud provider infrastructure

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

**Security Team:** security@prsm.ai
**Engineering Lead:** [Redacted for documentation]
**Escalation:** [Redacted for documentation]

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

*Document Version: 1.0*
*Last Updated: January 2026*
*Classification: Security Auditor - Confidential*
