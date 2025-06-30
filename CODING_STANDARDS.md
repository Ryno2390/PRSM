# PRSM Coding Standards

## Overview
This document establishes consistent coding standards for the PRSM codebase to improve maintainability, readability, and collaboration.

## Python Code Style

### 1. Formatting
- **Line Length**: Maximum 120 characters
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Prefer double quotes `"` for regular strings, single quotes `'` for internal/temporary strings
- **Trailing Commas**: Use in multi-line structures

### 2. Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `ALL_CAPS_WITH_UNDERSCORES`
- **Private Methods**: `_leading_underscore`
- **Magic Methods**: `__double_underscore__`

### 3. Import Organization
```python
# Standard library imports
import asyncio
import time
from typing import Dict, List, Optional

# Third-party imports
import structlog
from pydantic import BaseModel

# Local application imports
from prsm.core.models import UserInput
from prsm.agents.base import BaseAgent
from .local_module import LocalClass
```

### 4. Documentation
- **All public methods**: Must have docstrings
- **Classes**: Must have class-level docstring explaining purpose
- **Complex functions**: Include type hints and parameter descriptions

### 5. Error Handling
- **Never use bare `except:`** - Always specify exception types
- **Don't silently fail** - Log errors appropriately
- **Use proper exception chaining** with `raise ... from e`
- **Validate inputs** at function boundaries

```python
# Good
try:
    result = risky_operation()
    return result
except SpecificError as e:
    logger.error("Operation failed", error=str(e))
    raise RuntimeError(f"Failed to complete operation: {str(e)}") from e

# Bad
try:
    result = risky_operation()
    return result
except:
    return None
```

### 6. Logging
- **Use structured logging** with `structlog`
- **No print statements** in production code
- **Include context** in log messages

```python
logger.info("Operation completed successfully",
           session_id=session.session_id,
           duration_ms=duration * 1000)
```

### 7. Type Hints
- **All function signatures** should include type hints
- **Use Optional** for parameters that can be None
- **Import from typing** module as needed

```python
async def process_request(
    data: Dict[str, Any], 
    session_id: Optional[str] = None
) -> Tuple[bool, str]:
```

### 8. Async/Await
- **Use async/await** consistently
- **Don't block the event loop** - use async alternatives
- **Handle AsyncIO timeouts** appropriately

## Testing Standards

### 1. Test Naming
- **Descriptive names**: `test_user_authentication_with_invalid_credentials`
- **Not status-based**: Avoid `test_working`, `test_fixed`, `test_broken`

### 2. Assertions
- **Use proper assertions** - not just print statements
- **Include failure messages** in assertions
- **Test error conditions** explicitly

```python
# Good
assert user.is_authenticated, "User should be authenticated after login"

# Bad  
print("User authenticated: passed")
```

## File Organization

### 1. Directory Structure
- Keep related functionality together
- Use `__init__.py` for package imports
- Separate tests from implementation code

### 2. Module Size
- **Maximum ~500 lines** per module
- **Split large files** by responsibility
- **Group related classes** together

## Documentation

### 1. README Files
- Each major component should have a README
- Include setup/usage instructions
- Document dependencies clearly

### 2. Code Comments
- **Explain why, not what** - code should be self-documenting
- **Update comments** when changing code
- **Remove outdated comments**

## Security Standards

### 1. Secrets Management
- **Never commit secrets** to version control
- **Use environment variables** or secure credential managers
- **Log security events** appropriately

### 2. Input Validation
- **Validate all inputs** at boundaries
- **Sanitize user data** before processing
- **Use parameterized queries** for database operations

## Performance Guidelines

### 1. Database Operations
- **Use connection pooling**
- **Implement proper error handling** with rollbacks
- **Log slow queries** for optimization

### 2. API Design
- **Include timeout handling**
- **Implement rate limiting** where appropriate
- **Use proper HTTP status codes**

## Compliance

All code must follow these standards. Code reviews should enforce:
- Proper error handling
- Appropriate logging
- Type safety
- Documentation completeness
- Security best practices

## Tools

Recommended tools for maintaining standards:
- **Black**: Code formatting
- **mypy**: Type checking  
- **flake8**: Linting
- **pytest**: Testing framework
- **structlog**: Structured logging

---

**Note**: These standards were established to address technical debt and improve code quality across the PRSM codebase. All new code should follow these guidelines, and existing code should be updated incrementally during maintenance.