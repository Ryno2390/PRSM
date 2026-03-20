"""Add user_api_configs table

Persists per-user, per-provider LLM API configurations to the database.
Replaces the in-memory _user_api_configs dict in DatabaseService.store_user_api_config()
(Post-audit Phase 2 Item 2a fix).

Revision ID: 008_add_user_api_configs
Revises: 007_add_phase2_tables
Create Date: 2026-03-20
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '008_add_user_api_configs'
down_revision: Union[str, None] = '007_add_phase2_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_api_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('config_data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'provider', name='uq_user_api_config_provider'),
    )
    op.create_index('idx_user_api_config_user', 'user_api_configs', ['user_id'])


def downgrade() -> None:
    op.drop_index('idx_user_api_config_user', table_name='user_api_configs')
    op.drop_table('user_api_configs')
