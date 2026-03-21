"""Add pq_identities table for post-quantum keypair persistence

Persists PostQuantumAuthManager.identities dict to DB so users retain
their post-quantum identity across restarts (Phase 3 Item 3f fix).

Revision ID: 010_add_pq_identities
Revises: 009_add_staking_tables
Create Date: 2026-03-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '010_add_pq_identities'
down_revision = '009_add_staking_tables'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'pq_identities',
        sa.Column('user_id',        postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('security_level', sa.String(50),  nullable=False),
        sa.Column('keypair_json',   sa.Text(),       nullable=False),
        sa.Column('signature_type', sa.String(50),  nullable=False),
        sa.Column('created_at',     sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_used',      sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at',     sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('user_id'),
    )
    op.create_index('idx_pq_identity_user', 'pq_identities', ['user_id'])


def downgrade():
    op.drop_index('idx_pq_identity_user', table_name='pq_identities')
    op.drop_table('pq_identities')
