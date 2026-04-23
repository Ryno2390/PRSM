"""
Cross-database type compatibility for PRSM models.

JSONB and UUID from postgresql dialects only work with PostgreSQL.
These wrappers fall back to JSON/String for SQLite (dev/test) while
preserving the PostgreSQL-native types in production.
"""

from sqlalchemy import types


class _JSONBType(types.TypeDecorator):
    """
    Cross-database JSONB column type.
    Uses PostgreSQL's native JSONB in production; falls back to JSON for SQLite.
    """
    impl = types.JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
            return dialect.type_descriptor(PG_JSONB())
        return dialect.type_descriptor(types.JSON())


class _UUIDType(types.TypeDecorator):
    """
    Cross-database UUID column type.
    Uses PostgreSQL's native UUID in production; falls back to String(36) for SQLite.
    """
    impl = types.String
    cache_ok = True

    def __init__(self, *args, **kwargs):
        # Absorb any UUID-specific kwargs (as_uuid=True, etc.)
        kwargs.pop("as_uuid", None)
        super().__init__(36, *args, **kwargs)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import UUID as PG_UUID
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(types.String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        import uuid
        if isinstance(value, str):
            return uuid.UUID(value)
        return value


# Public aliases used in model imports
JSONB = _JSONBType
UUID = _UUIDType
