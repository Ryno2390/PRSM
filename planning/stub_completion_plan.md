# Stub Completion Plan

## Overview

Audit identified 9 genuine stub implementations in production code requiring completion.
All are located in concrete, instantiated classes — not abstract base classes.

| Priority | File | Issue |
|----------|------|-------|
| P0 | `teams_api.py:123` | Hardcoded `"current_user_placeholder"` user ID breaks auth |
| P0 | `enhanced_p2p_network.py:1370` | P2P model execution returns placeholder — feature non-functional |
| P1 | `model_executor.py:617` | Safety monitor integration — keyword filter only, no real scoring |
| P1 | `sandbox_manager.py:1308` | Security alerts not sent to external monitoring systems |
| P1 | `secure_api_client_factory.py:407` | Credential validation — structure check only, no API validation |
| P2 | `crypto_sharding.py:193` | Hardcoded `"system"` instead of actual user in audit trail |
| P2 | `crypto_sharding.py:292` | Session expiration field is `None` — no auto-expire |
| P2 | `hierarchical_compiler.py:2292` | Generated code template has `result = None` stub |
| P2 | `distributed_resource_manager.py:289,438` | GPU detection only supports NVIDIA, not AMD/ROCm |

---

## P0: Critical — Broken Functionality

### P0-A: `prsm/interface/api/teams_api.py:123` — Real User ID from Auth

**Problem:** `get_current_user_id()` returns `"current_user_placeholder"` unconditionally.
All teams API endpoints calling this function will attribute actions to a placeholder user
instead of the authenticated caller.

**Fix:** Extract user_id from the `current_user` dependency already available in endpoint
scope. Look at the existing endpoint signatures — they likely already accept
`current_user: User = Depends(get_current_user)`. The helper should accept that user
object or extract from the JWT token.

```python
# Replace:
def get_current_user_id() -> str:
    return "current_user_placeholder"  # Placeholder...

# With (after reading the actual endpoint structure):
def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    return str(current_user.id)
```

**Read first:** `prsm/interface/api/teams_api.py` lines 110–140 to understand how
`get_current_user_id` is called and what type the callers expect.

---

### P0-B: `prsm/compute/federation/enhanced_p2p_network.py:1370` — P2P Model Execution

**Problem:** `_handle_model_execution()` returns a static placeholder dict instead of
actually routing the task to a model executor.

**Fix:** Read lines 1350–1400 to understand the method signature and what data is
available (`request`, `peer_id`, etc.), then connect it to the existing model execution
infrastructure. The `ModelExecutor` class exists in
`prsm/compute/agents/executors/model_executor.py` — instantiate or call it.

Minimum viable fix:
```python
# Instead of returning placeholder, delegate to ModelExecutor:
from prsm.compute.agents.executors.model_executor import ModelExecutor
executor = ModelExecutor()
result = await executor.execute(
    model_id=request.get("model_id"),
    prompt=request.get("prompt"),
    parameters=request.get("parameters", {})
)
return result
```

**Read first:** `prsm/compute/federation/enhanced_p2p_network.py` lines 1340–1420 and
`prsm/compute/agents/executors/model_executor.py` lines 1–50 (class interface).

---

## P1: Important — Incomplete Safety/Security

### P1-A: `prsm/compute/agents/executors/model_executor.py:617` — Safety Monitor

**Problem:** `_validate_response()` uses a hardcoded keyword list and returns a fixed
`safety_score: 0.9`. The TODO says "Integrate with actual safety monitor."

**Fix:** The `prsm/safety/` module already exists. Identify the safety monitor class
there and call it. The response validation should return a real score.

Read `prsm/safety/` directory structure first to identify the correct class/method.
Then update `_validate_response()`:

```python
from prsm.safety.monitor import SafetyMonitor  # adjust path after reading

safety_monitor = SafetyMonitor()
safety_result = await safety_monitor.evaluate(content)
validated_result["safety_score"] = safety_result.score
validated_result["validation_passed"] = safety_result.passed
```

If the safety monitor requires async initialization, use a module-level singleton pattern.

**Read first:** `prsm/safety/` directory, then `model_executor.py` lines 605–660.

---

### P1-B: `prsm/core/integrations/security/sandbox_manager.py:1308` — External Security Notifications

**Problem:** `_notify_security_systems()` has TODO "Integrate with external security
systems" and likely does nothing or logs only.

**Fix:** Read lines 1290–1340 to understand what data is available. At minimum, this
should emit a structured security event to the existing audit logger.

```python
async def _notify_security_systems(self, event_type: str, details: dict):
    await audit_logger.log_security_event(
        event_type=event_type,
        severity=details.get("severity", "medium"),
        details=details
    )
```

Check whether an `audit_logger` is already imported/available in this module.

**Read first:** `sandbox_manager.py` lines 1280–1360.

---

### P1-C: `prsm/core/integrations/security/secure_api_client_factory.py:407` — Credential Validation

**Problem:** `_validate_credentials()` only checks structure (key presence/length) with
TODO "Add platform-specific validation (API calls to test credentials)."

**Fix:** For each supported provider (OpenAI, Anthropic, etc.), make a lightweight
test API call (e.g., list models endpoint) to confirm credentials are actually valid.
Use `httpx.AsyncClient` with a short timeout. Catch auth errors and raise `ValueError`.

```python
async def _validate_credentials(self, provider: str, credentials: dict) -> bool:
    if provider == "openai":
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {credentials['api_key']}"}
            )
            if r.status_code == 401:
                raise ValueError("Invalid OpenAI API key")
    # Add similar checks for other providers
    return True
```

**Read first:** `secure_api_client_factory.py` lines 390–440 to understand which
providers are supported and what credential shapes are expected.

---

## P2: Lower Priority — Incomplete Features

### P2-A: `prsm/compute/collaboration/security/crypto_sharding.py:193` — User Tracking

**Problem:** `"created_by": "system"` — no actual user identity in audit trail.

**Fix:** The shard creation method must already receive a user context somewhere upstream.
Read the method signature at line ~185. If a `user_id` parameter exists or can be added,
use it:

```python
"created_by": user_id if user_id else "system"
```

If user context isn't available in the method, thread it through from the caller.

---

### P2-B: `prsm/compute/collaboration/security/crypto_sharding.py:292` — Session Expiration

**Problem:** `"auto_expire": None` — sessions never automatically expire.

**Fix:** Set a default expiration (e.g., 24 hours from creation):

```python
from datetime import datetime, timezone, timedelta

"auto_expire": datetime.now(timezone.utc) + timedelta(hours=24)
```

Or accept a `ttl_hours: int = 24` parameter in the session creation method.

**Read first:** `crypto_sharding.py` lines 280–310 to understand the session creation
method signature.

---

### P2-C: `prsm/compute/agents/compilers/hierarchical_compiler.py:2292` — Solution Logic Template

**Problem:** `result = None` in generated code template for programming challenges.

**Fix:** Read lines 2280–2310 to understand the template context. The stub is inside
generated code, so the "fix" is to make the template generate better placeholder logic
or a comment guiding the user, not dead code:

```python
# Generated code template should produce:
result = solve(input_data)  # Replace with actual solution logic
```

**Read first:** `hierarchical_compiler.py` lines 2270–2320.

---

### P2-D: `prsm/compute/federation/distributed_resource_manager.py:289,438` — AMD GPU Support

**Problem:** GPU detection uses `pynvml` (NVIDIA only). AMD/ROCm support is TODO.

**Fix:** Add a try/except around `amdsmi` import and detection:

```python
try:
    import amdsmi
    amdsmi.amdsmi_init()
    amd_devices = amdsmi.amdsmi_get_processor_handles()
    # collect AMD GPU info
except ImportError:
    pass  # amdsmi not installed
except Exception:
    pass  # AMD GPUs not present
```

**Read first:** `distributed_resource_manager.py` lines 275–310 and 420–455.

---

## File Checklist

- [ ] `prsm/interface/api/teams_api.py` — Fix `get_current_user_id()` (P0-A)
- [ ] `prsm/compute/federation/enhanced_p2p_network.py` — Fix `_handle_model_execution()` (P0-B)
- [ ] `prsm/compute/agents/executors/model_executor.py` — Connect safety monitor (P1-A)
- [ ] `prsm/core/integrations/security/sandbox_manager.py` — Add security event emission (P1-B)
- [ ] `prsm/core/integrations/security/secure_api_client_factory.py` — Add API credential validation (P1-C)
- [ ] `prsm/compute/collaboration/security/crypto_sharding.py` — User tracking + expiration (P2-A, P2-B)
- [ ] `prsm/compute/agents/compilers/hierarchical_compiler.py` — Fix template stub (P2-C)
- [ ] `prsm/compute/federation/distributed_resource_manager.py` — Add AMD GPU detection (P2-D)

**Expected outcome:** Zero genuine stubs in production code. All tests remain passing.

---

## Implementation Summary

*(To be filled in by you upon completion — rename file to `_completed.md`)*
