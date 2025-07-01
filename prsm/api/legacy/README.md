# Legacy Marketplace APIs

⚠️ **DEPRECATED**: These APIs are deprecated and should not be used in production.

## Moved to Legacy (2025-01-01)

### `marketplace_api.py`
- **Status**: Deprecated
- **Reason**: Legacy implementation with incomplete features
- **Replacement**: Use `unified_marketplace_api.py` for all marketplace operations
- **Migration**: All endpoints have been consolidated into the unified API

### `marketplace_api_launch.py` 
- **Status**: Deprecated (Simulation Only)
- **Reason**: FTNS marketplace simulation for development/testing only
- **Replacement**: Use `unified_marketplace_api.py` for real marketplace operations
- **Note**: This was never intended for production use

## Current Production APIs

✅ **`unified_marketplace_api.py`** - Complete consolidated marketplace API
- Universal resource management (all 9 asset types)
- Advanced search and filtering
- Real database operations
- Marketplace launch functionality
- Analytics and statistics
- Admin moderation tools

✅ **`real_marketplace_api.py`** - Will be replaced by unified API
✅ **`marketplace_launch_api.py`** - Will be merged into unified API

## Migration Guide

All marketplace functionality is now available through the unified API:

```python
# OLD (deprecated)
from prsm.api.marketplace_api import router as marketplace_router
from prsm.api.marketplace_api_launch import router as ftns_router

# NEW (production)
from prsm.api.unified_marketplace_api import router as marketplace_router
```

The unified API provides all functionality with:
- Better error handling
- Comprehensive validation
- Real database persistence
- Universal resource support
- Enhanced security
- Complete documentation