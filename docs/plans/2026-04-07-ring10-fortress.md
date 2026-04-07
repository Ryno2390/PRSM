# Ring 10 — "The Fortress" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Final production hardening for Rings 7-9: TEE attestation verification, model shard integrity checks, DP privacy budget auditing, pipeline randomization audit trail, and a comprehensive all-rings verification suite.

**Architecture:** A `SecurityAuditor` module provides verification utilities across the full stack. An `IntegrityVerifier` validates model shard checksums. A `PrivacyBudgetTracker` enforces cumulative epsilon limits. A `PipelineAuditLog` records all randomization decisions for provability.

**Tech Stack:** Existing PRSM infrastructure. No new external dependencies.

**Scope note:** Full API authentication, mDNS discovery, OpenTelemetry extensions, and documentation updates are tracked in the spec but deferred — each is its own project.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/security/__init__.py` | Package exports |
| Create | `prsm/security/integrity.py` | `IntegrityVerifier` — shard checksum validation |
| Create | `prsm/security/privacy_budget.py` | `PrivacyBudgetTracker` — cumulative epsilon enforcement |
| Create | `prsm/security/audit_log.py` | `PipelineAuditLog` — randomization decision logging |
| Modify | `prsm/node/node.py` | Wire Ring 10 into PRSMNode |
| Create | `tests/unit/test_security_hardening.py` | Integrity, privacy, audit tests |
| Create | `tests/integration/test_ring10_fortress.py` | Full-stack verification |
| Create | `tests/e2e/test_all_rings_verify.py` | Comprehensive all-rings verification |

---

### Task 1: Security Modules

Create `prsm/security/` with integrity verifier, privacy budget tracker, and audit log.

### Task 2: Node Integration + All-Rings Verification

Wire into PRSMNode, create comprehensive test suites, version bump, push, publish.
