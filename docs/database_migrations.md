# PRSM Database Migrations

This guide covers database migration management for PRSM using Alembic.

## Overview

PRSM uses Alembic for database schema versioning and migrations. This ensures consistent database schemas across development, testing, and production environments.

## Migration Structure

```
alembic/
├── versions/           # Migration files
├── env.py             # Alembic environment configuration
├── script.py.mako     # Migration template
└── README
alembic.ini            # Alembic configuration
```

## Configuration

### Database URL

Alembic automatically uses the PRSM database configuration from `prsm.core.config`. The database URL is set via:

- Environment variable: `PRSM_DATABASE_URL`
- Default: `sqlite:///./prsm_test.db`

### Supported Databases

- **SQLite** (development/testing): `sqlite:///./database.db`
- **PostgreSQL** (production): `postgresql://user:pass@host:port/db`

## Common Commands

### Check Current Migration Status

```bash
alembic current
```

### Create New Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Create empty migration (manual changes)
alembic revision -m "Description of changes"
```

### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade <revision_id>

# Upgrade by relative amount
alembic upgrade +2
```

### Rollback Migrations

```bash
# Downgrade by one migration
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>

# Downgrade to base (empty database)
alembic downgrade base
```

### View Migration History

```bash
# Show migration history
alembic history

# Show verbose history
alembic history --verbose
```

## Development Workflow

### 1. Making Model Changes

1. Modify models in `prsm/core/database.py`
2. Generate migration:
   ```bash
   alembic revision --autogenerate -m "Add new feature table"
   ```
3. Review generated migration in `alembic/versions/`
4. Edit migration if needed (add data migrations, etc.)
5. Test migration:
   ```bash
   alembic upgrade head
   ```

### 2. Testing Migrations

Always test migrations in development:

```bash
# Test upgrade
alembic upgrade head

# Test rollback
alembic downgrade -1

# Test re-upgrade
alembic upgrade head
```

### 3. Database Schema Changes

#### Adding Tables/Columns
- Use `alembic revision --autogenerate`
- Alembic automatically detects new tables and columns

#### Removing Tables/Columns
- Use `alembic revision --autogenerate`
- Verify the migration before applying to production

#### Data Migrations
- Create manual migration with `alembic revision`
- Add data transformation logic in `upgrade()` and `downgrade()`

## Production Deployment

### Pre-deployment Checklist

1. **Backup Database**: Always backup before migrations
2. **Test in Staging**: Apply migrations to staging environment first
3. **Review Migration**: Ensure migration is reversible if possible
4. **Monitor Performance**: Large table changes may require maintenance windows

### Deployment Process

```bash
# 1. Check current status
alembic current

# 2. Apply migrations
alembic upgrade head

# 3. Verify application functionality
```

### Rollback Strategy

If issues occur after migration:

```bash
# Immediate rollback
alembic downgrade -1

# Or rollback to specific known-good revision
alembic downgrade <previous_revision>
```

## Migration Best Practices

### DO:
- ✅ Always review auto-generated migrations
- ✅ Test migrations locally before deployment
- ✅ Include both upgrade and downgrade operations
- ✅ Use descriptive migration messages
- ✅ Backup production databases before migrations
- ✅ Test rollback procedures

### DON'T:
- ❌ Edit existing migration files after they're applied
- ❌ Skip migration testing
- ❌ Deploy without backup strategies
- ❌ Create migrations with breaking changes without planning
- ❌ Delete migration files from version control

## Example Migration

### Auto-generated Migration

```python
"""Add user preferences table

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2024-12-10 10:30:00.000000
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    """Upgrade schema."""
    op.create_table('user_preferences',
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('user_id')
    )

def downgrade():
    """Downgrade schema."""
    op.drop_table('user_preferences')
```

### Manual Data Migration

```python
"""Migrate user data format

Revision ID: def456ghi789
Revises: abc123def456
Create Date: 2024-12-10 11:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column

def upgrade():
    """Convert old format to new format."""
    # Define table structure for data migration
    user_table = table('users',
        column('id', sa.String),
        column('old_field', sa.String),
        column('new_field', sa.String)
    )
    
    # Migration logic
    connection = op.get_bind()
    users = connection.execute(
        sa.select(user_table.c.id, user_table.c.old_field)
    ).fetchall()
    
    for user in users:
        # Transform data
        new_value = transform_data(user.old_field)
        connection.execute(
            user_table.update()
            .where(user_table.c.id == user.id)
            .values(new_field=new_value)
        )

def downgrade():
    """Reverse data migration."""
    # Implement reverse transformation
    pass
```

## Troubleshooting

### Common Issues

#### Migration Conflicts
- **Issue**: Multiple developers creating migrations simultaneously
- **Solution**: Coordinate migration creation, merge conflicts manually

#### Schema Drift
- **Issue**: Database schema doesn't match models
- **Solution**: Use `alembic check` to detect differences

#### Rollback Failures
- **Issue**: Downgrade operation fails
- **Solution**: Manually fix database state or restore from backup

### Getting Help

1. Check `alembic history` for migration timeline
2. Use `alembic current` to see current state
3. Review migration files in `alembic/versions/`
4. Check Alembic logs for error details

## Integration with PRSM

### Database Service Compatibility

The DatabaseService in `prsm.core.database_service` is designed to work with the migrated schema. Key integration points:

- **Model Imports**: All models are imported in `alembic/env.py`
- **Configuration**: Uses same database URL as PRSM application
- **Schema Sync**: Migrations keep database in sync with model definitions

### Testing Integration

Run database tests after migrations:

```bash
# Apply migrations
alembic upgrade head

# Run database service tests
python -m pytest tests/test_database_service.py

# Run full test suite
python -m pytest
```

---

For more information on Alembic, see the [official documentation](https://alembic.sqlalchemy.org/).