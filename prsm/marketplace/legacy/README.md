# Legacy Marketplace Modules

⚠️ **DEPRECATED** - These modules are no longer used in production.

## Deprecation Notice

These marketplace modules have been **deprecated** in favor of the consolidated `real_marketplace_service.py` implementation. They are kept here for reference only.

### Deprecated Modules

- `marketplace_service.py` - Legacy service with mock implementations
- `ftns_marketplace.py` - Simulation-only FTNS marketplace (not real transactions)
- `model_marketplace.py` - Phase 3 MVP model marketplace
- `tool_marketplace.py` - Standalone tool marketplace

### Migration

All functionality from these modules has been consolidated into:
- **`real_marketplace_service.py`** - Production-ready service with real database operations
- **`real_marketplace_api.py`** - Production API with standardized error handling

### Audit Compliance

Per the Gemini 2.5 Pro audit recommendations:
1. ✅ **Marketplace consolidation completed** - All logic moved to real_marketplace_service.py
2. ✅ **Deprecated modules removed** - Moved to legacy folder to prevent usage
3. ✅ **Single source of truth** - All marketplace functionality routes through real service
4. ✅ **Error handling standardized** - Consistent patterns across all APIs

## For Developers

**DO NOT** import or use these legacy modules. Use the production service instead:

```python
# ✅ CORRECT - Use production service
from prsm.marketplace.real_marketplace_service import RealMarketplaceService

# ❌ INCORRECT - Do not use legacy modules
# from prsm.marketplace.legacy.marketplace_service import marketplace_service
```

All marketplace functionality is available through the real service with complete database operations, comprehensive error handling, and production-ready features.