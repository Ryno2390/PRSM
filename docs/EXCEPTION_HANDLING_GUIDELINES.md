# PRSM Exception Handling Guidelines

## Overview

This document provides guidelines for proper exception handling in the PRSM codebase. Following these guidelines ensures consistent error handling, better debugging, and improved system reliability.

## Table of Contents

1. [General Principles](#general-principles)
2. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
3. [Recommended Patterns](#recommended-patterns)
4. [Module-Specific Guidelines](#module-specific-guidelines)
5. [Logging Requirements](#logging-requirements)
6. [Custom Exceptions](#custom-exceptions)
7. [Testing Exception Handling](#testing-exception-handling)

---

## General Principles

### 1. Be Specific
Always catch the most specific exception type possible. This prevents accidentally catching unexpected errors and makes debugging easier.

### 2. Never Silence Errors
Every exception handler should either:
- Log the error with context
- Re-raise the exception
- Transform it into a more appropriate exception type
- Return a meaningful default value (with logging)

### 3. Preserve Context
When catching and re-raising or transforming exceptions, always preserve the original error context using `raise ... from e` or including the original error in the log.

### 4. Document Intentional Suppression
If you intentionally suppress an exception (e.g., cleanup code), add a comment explaining why.

---

## Anti-Patterns to Avoid

### ❌ Bare `except:` Statements

**NEVER use bare `except:` statements.** They catch system exceptions like `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`, which should normally propagate.

```python
# ❌ BAD - Catches everything including KeyboardInterrupt
try:
    do_something()
except:
    pass  # Silently swallows ALL exceptions

# ✅ GOOD - Catches only regular exceptions
try:
    do_something()
except Exception as e:
    logger.error("Operation failed", error=str(e))
```

### ❌ Silent `pass` Statements

Never use `pass` without logging in an exception handler.

```python
# ❌ BAD - Silent failure, impossible to debug
try:
    process_data(data)
except Exception:
    pass

# ✅ GOOD - Logged with context
try:
    process_data(data)
except Exception as e:
    logger.warning("Failed to process data, using default", 
                   error=str(e), data_id=data.get("id"))
```

### ❌ Returning `None` Without Context

```python
# ❌ BAD - Caller can't distinguish between "not found" and "error"
def get_user(user_id):
    try:
        return db.query(User).get(user_id)
    except Exception:
        return None

# ✅ GOOD - Log and return None with clear semantics
def get_user(user_id):
    try:
        return db.query(User).get(user_id)
    except DatabaseError as e:
        logger.error("Database error fetching user", 
                     error=str(e), user_id=user_id)
        return None
```

### ❌ Catching Too Broad

```python
# ❌ BAD - Too broad, catches typos in the try block
try:
    result = complex_calculation(data)
    log_resut(result)  # Typo! This error is silently caught
except Exception:
    return None

# ✅ GOOD - Catch only expected exceptions
try:
    result = complex_calculation(data)
    log_result(result)
except (ValueError, CalculationError) as e:
    logger.warning("Calculation failed", error=str(e))
    return None
```

---

## Recommended Patterns

### Pattern 1: Specific Exception with Logging

```python
try:
    await process_transaction(tx)
except InsufficientBalanceError as e:
    logger.warning("Transaction rejected - insufficient balance",
                   user_id=tx.user_id,
                   amount=tx.amount,
                   balance=e.available_balance)
    raise  # Re-raise for caller to handle
except ValidationError as e:
    logger.error("Transaction validation failed",
                 tx_id=tx.id,
                 errors=e.errors)
    raise TransactionError(f"Invalid transaction: {e}") from e
```

### Pattern 2: Cleanup with Exception Preservation

```python
resource = None
try:
    resource = await acquire_resource()
    await use_resource(resource)
finally:
    if resource:
        try:
            await resource.close()
        except Exception as e:
            # Log but don't mask the original exception
            logger.warning("Failed to close resource", error=str(e))
```

### Pattern 3: Retry with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def fetch_with_retry(url: str) -> Response:
    try:
        return await http_client.get(url)
    except (ConnectionError, TimeoutError) as e:
        logger.warning("Request failed, retrying",
                       url=url, attempt=retry.state.attempt_number)
        raise
```

### Pattern 4: Context Manager for Resource Handling

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_connection():
    conn = None
    try:
        conn = await create_connection()
        yield conn
    except ConnectionError as e:
        logger.error("Connection error", error=str(e))
        raise
    finally:
        if conn:
            try:
                await conn.close()
            except Exception as e:
                logger.debug("Error closing connection (may be expected)", error=str(e))
```

### Pattern 5: Graceful Degradation

```python
async def get_recommendations(user_id: str) -> list[Recommendation]:
    try:
        # Try ML-based recommendations first
        return await ml_engine.get_recommendations(user_id)
    except (MLEngineError, TimeoutError) as e:
        logger.warning("ML recommendations unavailable, falling back to basic",
                       user_id=user_id, error=str(e))
        try:
            # Fall back to simpler algorithm
            return await basic_recommender.get(user_id)
        except Exception as e:
            logger.error("All recommendation methods failed",
                         user_id=user_id, error=str(e))
            return []  # Empty list as last resort
```

---

## Module-Specific Guidelines

### DAG Ledger (`prsm/node/dag_ledger.py`)

```python
# Database operations should catch specific sqlite errors
import sqlite3

try:
    await self._db.execute(query, params)
except sqlite3.IntegrityError as e:
    logger.error("Database integrity violation", query=query, error=str(e))
    raise DuplicateTransactionError(f"Transaction already exists: {tx_id}") from e
except sqlite3.OperationalError as e:
    logger.error("Database operation failed", error=str(e))
    raise DatabaseError(f"Database operation failed: {e}") from e
```

### P2P Network (`prsm/compute/federation/`)

```python
# Network operations should distinguish between connection and protocol errors
try:
    await peer.send_message(message)
except ConnectionError as e:
    logger.warning("Connection lost to peer", peer_id=peer.id, error=str(e))
    await self.handle_peer_disconnect(peer)
except asyncio.TimeoutError as e:
    logger.warning("Message send timeout", peer_id=peer.id)
    raise MessageDeliveryError(f"Timeout sending to {peer.id}") from e
```

### Marketplace (`prsm/economy/`)

```python
# Financial operations must never silently fail
try:
    await self.atomic_deduct(user_id, amount, reason)
except InsufficientBalanceError as e:
    logger.warning("Insufficient balance for purchase",
                   user_id=user_id, amount=amount, balance=e.balance)
    raise  # Always propagate financial errors
except Exception as e:
    logger.error("Unexpected error in financial operation",
                 user_id=user_id, amount=amount, error=str(e), exc_info=True)
    raise FinancialError(f"Transaction failed: {e}") from e
```

### NWTN Orchestrator (`prsm/compute/nwtn/`)

```python
# AI operations should handle model-specific errors
try:
    response = await model.generate(prompt)
except ModelLoadError as e:
    logger.error("Failed to load model", model=model.name, error=str(e))
    raise ServiceUnavailableError("AI service temporarily unavailable") from e
except ContextLengthError as e:
    logger.warning("Context too long", tokens=e.token_count, limit=e.limit)
    # Truncate and retry
    return await self.generate_with_truncation(prompt, max_tokens=e.limit)
```

---

## Logging Requirements

### Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | Detailed diagnostic information (development only) |
| `INFO` | Normal operational events |
| `WARNING` | Unexpected but handled situations |
| `ERROR` | Errors that affect operation but are recoverable |
| `CRITICAL` | Errors that may require immediate attention |

### Structured Logging

Always use structured logging with context:

```python
# ❌ BAD - Unstructured, hard to query
logger.error(f"Failed to process user {user_id}: {e}")

# ✅ GOOD - Structured, queryable
logger.error("Failed to process user",
             user_id=user_id,
             error=str(e),
             error_type=type(e).__name__,
             exc_info=True)  # Include stack trace for errors
```

### Required Context Fields

Always include these fields when available:
- `user_id` - User affected by the error
- `request_id` - Request being processed
- `operation` - Operation that failed
- `error` - Error message
- `error_type` - Exception class name

---

## Custom Exceptions

### Creating Custom Exceptions

```python
class PRSMBaseException(Exception):
    """Base exception for all PRSM exceptions."""
    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}

class FinancialError(PRSMBaseException):
    """Base for financial operation errors."""
    pass

class InsufficientBalanceError(FinancialError):
    """Raised when user has insufficient balance."""
    def __init__(self, user_id: str, required: float, available: float):
        super().__init__(
            f"Insufficient balance: required {required}, available {available}",
            context={"user_id": user_id, "required": required, "available": available}
        )
        self.user_id = user_id
        self.required = required
        self.available = available
```

### Exception Hierarchy

```
PRSMBaseException
├── FinancialError
│   ├── InsufficientBalanceError
│   ├── DuplicateTransactionError
│   └── TransactionValidationError
├── NetworkError
│   ├── PeerDisconnectedError
│   ├── MessageDeliveryError
│   └── ConsensusError
├── StorageError
│   ├── IPFSError
│   └── DatabaseError
└── ServiceError
    ├── ServiceUnavailableError
    └── RateLimitError
```

---

## Testing Exception Handling

### Unit Testing Exceptions

```python
import pytest

async def test_insufficient_balance_raises_error():
    """Test that insufficient balance raises InsufficientBalanceError."""
    ledger = DAGLedger(db_path=':memory:')
    await ledger.initialize()
    
    # User has 100 tokens
    await ledger.credit("user1", 100.0, "initial")
    
    # Trying to spend 200 should fail
    with pytest.raises(InsufficientBalanceError) as exc_info:
        await ledger.debit("user1", 200.0, "purchase")
    
    assert exc_info.value.available == 100.0
    assert exc_info.value.required == 200.0
```

### Testing Error Logging

```python
async def test_error_is_logged(caplog):
    """Test that errors are properly logged."""
    with caplog.at_level(logging.ERROR):
        await failing_operation()
    
    assert len(caplog.records) == 1
    assert "operation failed" in caplog.records[0].message.lower()
```

### Integration Testing

```python
async def test_concurrent_operations_handle_errors():
    """Test that concurrent operations handle errors correctly."""
    results = await asyncio.gather(
        operation_that_fails(),
        operation_that_succeeds(),
        return_exceptions=True
    )
    
    # First should be an exception
    assert isinstance(results[0], ExpectedError)
    # Second should succeed
    assert results[1] == expected_result
```

---

## Checklist for Code Review

When reviewing exception handling code, verify:

- [ ] No bare `except:` statements
- [ ] No silent `pass` without comment
- [ ] Specific exception types are caught
- [ ] Errors are logged with context
- [ ] Original exceptions are preserved with `raise ... from e`
- [ ] Cleanup code uses `finally` blocks
- [ ] Custom exceptions follow the hierarchy
- [ ] Tests cover error scenarios

---

## References

- [Python Exception Handling Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [Structured Logging with structlog](https://www.structlog.org/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)

---

*Last updated: 2026-02-27*
*Author: PRSM Development Team*
