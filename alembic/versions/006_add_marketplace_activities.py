"""Add marketplace_activities table

Revision ID: 006_add_marketplace_activities
Revises: 005_add_reputation_tables
Create Date: 2026-03-18 18:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '006_add_marketplace_activities'
down_revision: Union[str, None] = '005_add_reputation_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Create marketplace_activities table."""
    
    # Create marketplace_activities table
    op.create_table(
        'marketplace_activities',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('activity_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('activity_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('activity_metadata', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('activity_id')
    )
    
    # Create indexes for marketplace_activities
    op.create_index(op.f('ix_marketplace_activities_activity_id'), 'marketplace_activities', ['activity_id'], unique=False)
    op.create_index(op.f('ix_marketplace_activities_user_id'), 'marketplace_activities', ['user_id'], unique=False)
    op.create_index(op.f('ix_marketplace_activities_activity_type'), 'marketplace_activities', ['activity_type'], unique=False)
    op.create_index(op.f('ix_marketplace_activities_resource_id'), 'marketplace_activities', ['resource_id'], unique=False)
    op.create_index(op.f('ix_marketplace_activities_session_id'), 'marketplace_activities', ['session_id'], unique=False)
    op.create_index(op.f('ix_marketplace_activities_timestamp'), 'marketplace_activities', ['timestamp'], unique=False)
    
    # Create composite indexes for common query patterns
    op.create_index('idx_marketplace_activities_user_ts', 'marketplace_activities', ['user_id', 'timestamp'], unique=False)
    op.create_index('idx_marketplace_activities_type_ts', 'marketplace_activities', ['activity_type', 'timestamp'], unique=False)
    op.create_index('idx_marketplace_activities_resource', 'marketplace_activities', ['resource_id', 'timestamp'], unique=False)


def downgrade() -> None:
    """Downgrade schema: Drop marketplace_activities table."""
    
    # Drop composite indexes
    op.drop_index('idx_marketplace_activities_resource', table_name='marketplace_activities')
    op.drop_index('idx_marketplace_activities_type_ts', table_name='marketplace_activities')
    op.drop_index('idx_marketplace_activities_user_ts', table_name='marketplace_activities')
    
    # Drop single-column indexes
    op.drop_index(op.f('ix_marketplace_activities_timestamp'), table_name='marketplace_activities')
    op.drop_index(op.f('ix_marketplace_activities_session_id'), table_name='marketplace_activities')
    op.drop_index(op.f('ix_marketplace_activities_resource_id'), table_name='marketplace_activities')
    op.drop_index(op.f('ix_marketplace_activities_activity_type'), table_name='marketplace_activities')
    op.drop_index(op.f('ix_marketplace_activities_user_id'), table_name='marketplace_activities')
    op.drop_index(op.f('ix_marketplace_activities_activity_id'), table_name='marketplace_activities')
    
    # Drop table
    op.drop_table('marketplace_activities')