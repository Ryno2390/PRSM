# PRSM Security Hardening Checklist

> **Status:** authored January 2026 following the Technical Due Diligence (TDD) remediation work tracked in the [historical remediation plan](REMEDIATION_HARDENING_MASTER_PLAN.md). The **checkbox state** and **file references** below reflect implementation intent at that snapshot. **Verify against current code before treating any ✅ box as current truth** — the codebase has evolved since January 2026, including the v1.6 scope-alignment sprint (April 2026, deleted ~210K LoC of legacy) and the Phase 1 on-chain provenance work (in Sepolia bake-in as of April 2026). Some file paths may have moved; some checkboxes may be stale in either direction (shipped work now, or regressed).
>
> **Related docs in this suite:**
> - [`SECURITY_HARDENING.md`](SECURITY_HARDENING.md) — security policy and hardening guide
> - [`SECURITY_CONFIGURATION_AUDIT.md`](SECURITY_CONFIGURATION_AUDIT.md) — point-in-time audit snapshot
> - [`PENETRATION_TESTING_GUIDE.md`](PENETRATION_TESTING_GUIDE.md) — pen-test methodology
> - [`REMEDIATION_HARDENING_MASTER_PLAN.md`](REMEDIATION_HARDENING_MASTER_PLAN.md) — sprint-level remediation plan (historical; see post-v1.6 status audit at top of that doc)
> - [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) Phase 6 + Phase 7 — protocol-layer hardening scope

## Overview

This checklist documents the security controls implemented to address findings from the Technical Due Diligence audit. Use this document to verify security posture before deployment and during security reviews. **Treat as implementation-intent reference, not current-state source of truth** — current state must be verified against the codebase.

---

## 1. Authentication & Authorization

### JWT Security
- [x] **Algorithm Confusion Prevention**
  - Only HS256, HS384, HS512 allowed
  - `none` algorithm explicitly rejected
  - File: `prsm/core/auth/jwt_handler.py:ALLOWED_ALGORITHMS`

- [x] **Token Revocation**
  - Tiered revocation checking: Redis cache → PostgreSQL
  - Token hash stored on revocation
  - Revocation survives Redis restart
  - File: `prsm/core/auth/jwt_handler.py:_is_token_revoked()`

- [x] **Required Claims Validation**
  - `exp` (expiration) - enforced
  - `iat` (issued at) - checked for future dates
  - `sub` (subject) - required for user identification
  - `jti` (JWT ID) - required for revocation tracking

- [x] **Token Lifetime**
  - Access tokens: 15-60 minutes (configurable)
  - Refresh tokens: 7 days max
  - Session timeout enforced

### Password Security
- [x] Bcrypt hashing with cost factor ≥12
- [x] Password complexity requirements
- [x] Account lockout after failed attempts
- [x] Password change invalidates all existing tokens

---

## 2. Financial Transaction Security (FTNS)

### Double-Spend Prevention
- [x] **Row-Level Locking**
  - `SELECT FOR UPDATE` on balance reads
  - File: `prsm/economy/tokenomics/atomic_ftns_service.py`
  - Migration: `scripts/migrations/003_atomic_operations_and_revocation.sql`

- [x] **Optimistic Concurrency Control**
  - Version column on `ftns_wallets` table
  - Version check in UPDATE WHERE clause
  - Automatic retry on version mismatch

- [x] **Idempotency Keys**
  - Unique constraint on idempotency_key
  - 24-hour TTL for idempotency records
  - Same key returns cached result
  - Table: `ftns_idempotency_keys`

- [x] **Database Constraints**
  - `CHECK (balance >= 0)` prevents negative balance
  - Foreign key constraints on transactions
  - Transaction isolation level: SERIALIZABLE for critical ops

### Transaction Recovery
- [x] Stuck transaction detection
- [x] Automated recovery service
- [x] Ledger reconciliation
- File: `prsm/economy/tokenomics/transaction_recovery.py`

---

## 3. Rate Limiting

### Sliding Window Algorithm
- [x] Redis sorted sets for accurate tracking
- [x] Per-user rate limiting
- [x] Per-IP rate limiting (for anonymous)
- [x] Per-endpoint rate limiting (optional)
- File: `prsm/core/security/advanced_rate_limiting.py:SlidingWindowRateLimiter`

### Tier-Based Limits
| Tier       | Requests/Min | Requests/Hour | Requests/Day |
|------------|--------------|---------------|--------------|
| Anonymous  | 10           | 100           | 500          |
| Free       | 60           | 1,000         | 10,000       |
| Pro        | 300          | 10,000        | 100,000      |
| Enterprise | 1,000        | 50,000        | 500,000      |

### Adaptive Rate Limiting
- [x] Threat score affects limits
- [x] System load factor adjustment
- [x] Burst allowance with token bucket
- [x] Cooldown periods after violations

---

## 4. Input Validation & Injection Prevention

### SQL Injection
- [x] Parameterized queries only
- [x] ORM (SQLAlchemy) for most operations
- [x] Input sanitization on user-provided data
- [x] No string concatenation in SQL

### XSS Prevention
- [x] Content-Type headers enforced
- [x] JSON responses properly escaped
- [x] HTML content sanitized if rendered

### Request Validation
- [x] Content-Length limits (10MB default)
- [x] Content-Type validation
- [x] JSON schema validation for API inputs
- File: `prsm/interface/api/middleware.py:ContentValidationMiddleware`

---

## 5. API Security

### HTTPS/TLS
- [x] TLS 1.2+ required in production
  - Configurable via `PRSM_TLS_MIN_VERSION`
  - File: `prsm/core/security/tls_config.py`
- [x] HSTS header enabled
  - 1 year max-age with includeSubDomains
  - File: `prsm/interface/api/standards.py`
- [x] TLS configuration for database connections
  - PostgreSQL SSL with verify-full mode
  - File: `prsm/core/security/tls_config.py:get_database_ssl_config()`
- [x] TLS configuration for Redis connections
  - Redis TLS with certificate verification
  - File: `prsm/core/security/tls_config.py:get_redis_ssl_config()`
- [ ] Certificate pinning (mobile clients) - Future enhancement

### Security Headers
- [x] X-Content-Type-Options: nosniff
- [x] X-Frame-Options: DENY
- [x] X-XSS-Protection: 1; mode=block
- [x] Content-Security-Policy configured
- [x] Referrer-Policy: strict-origin-when-cross-origin
- File: `prsm/interface/api/middleware.py:SecurityHeadersMiddleware`

### CORS
- [x] Whitelist-based origin validation
- [x] Credentials allowed only for whitelisted origins
- [x] Preflight caching configured
- File: `prsm/core/security/middleware.py:configure_cors()`

---

## 6. Cryptography

### Post-Quantum Cryptography
- [x] Production mode: No silent mock fallback
- [x] Explicit mode selection: DISABLED, REAL, HYBRID
- [x] liboqs integration when available
- [x] Graceful degradation with warnings
- File: `prsm/core/cryptography/post_quantum_production.py`

### Key Management
- [x] Secrets stored in environment variables
- [x] No hardcoded credentials
- [x] Key rotation procedures documented
- [x] Separate keys per environment

---

## 7. Logging & Monitoring

### Security Logging
- [x] Authentication events logged
- [x] Rate limit violations logged
- [x] Transaction operations logged
- [x] No sensitive data in logs (tokens, passwords)
- File: `prsm/core/security/security_monitoring.py`

### Audit Trail
- [x] User actions tracked
- [x] Admin operations logged
- [x] IP addresses recorded
- [x] Timestamps in UTC

### Alerting
- [x] Failed authentication threshold alerts
- [x] Rate limit breach alerts
- [x] Unusual transaction patterns
- File: `prsm/core/security/security_analytics.py`

---

## 8. Infrastructure Security

### Database
- [x] Connection pooling limits
- [x] Query timeouts configured
- [x] Prepared statements only
- [ ] Database encryption at rest
- [ ] Backup encryption

### Redis
- [x] Authentication required
- [x] Connection limits
- [x] Key expiration policies
- [x] TLS for Redis connections
  - Configurable via `prsm/core/security/tls_config.py:get_redis_ssl_config()`
  - Use `rediss://` protocol for TLS connections

### IPFS
- [x] Content validation
- [x] Size limits
- [x] CID verification
- [x] Pinning controls

---

## 9. Deployment Checklist

### Pre-Production
- [ ] All secrets rotated from development
- [ ] Debug mode disabled
- [ ] Error messages sanitized
- [ ] Rate limits configured for expected load
- [ ] Database indexes verified

### Production
- [ ] HTTPS certificates valid
- [ ] Monitoring dashboards configured
- [ ] Alerting rules active
- [ ] Backup procedures tested
- [ ] Incident response plan reviewed

---

## 10. Security Testing

### Automated Tests
- [x] Double-spend prevention tests
- [x] JWT verification tests
- [x] Rate limiting tests
- Location: `tests/security/`

### Manual Testing
- [ ] Penetration testing scheduled
- [ ] Security code review completed
- [ ] Dependency vulnerability scan
- [ ] OWASP Top 10 checklist verified

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Lead Developer | | | |
| Security Engineer | | | |
| DevOps Engineer | | | |
| Product Manager | | | |

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01 | PRSM Team | Initial checklist from TDD remediation |
| 1.1 | 2026-04-16 | Cross-reference pass | Added status banner, cross-refs to companion docs and master roadmap; flagged checkbox-state as snapshot not source of truth |

---

## References

- Remediation Plan (historical): [`REMEDIATION_HARDENING_MASTER_PLAN.md`](REMEDIATION_HARDENING_MASTER_PLAN.md) — see post-v1.6 status audit at top of that doc for which sprint items actually shipped
- Master Roadmap: [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) — Phase 6 + Phase 7 for protocol-layer hardening
- Companion: [`SECURITY_HARDENING.md`](SECURITY_HARDENING.md), [`SECURITY_CONFIGURATION_AUDIT.md`](SECURITY_CONFIGURATION_AUDIT.md)
- Security Tests: `tests/security/` (verify presence — some referenced test files may not exist)
- Migration Scripts: `scripts/migrations/`
