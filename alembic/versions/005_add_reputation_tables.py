"""Add reputation tables

Revision ID: 005_add_reputation_tables
Revises: 8377a6cba565
Create Date: 2026-03-18 17:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_add_reputation_tables'
down_revision: Union[str, None] = '8377a6cba565'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Create user_reputations and reputation_events tables."""
    
    # Create user_reputations table
    op.create_table(
        'user_reputations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False, server_default='50.0'),
        sa.Column('trust_level', sa.String(length=50), nullable=False, server_default='NEWCOMER'),
        sa.Column('dimension_scores', sa.JSON(), nullable=True),
        sa.Column('badges', sa.JSON(), nullable=True),
        sa.Column('verification_status', sa.JSON(), nullable=True),
        sa.Column('reputation_history', sa.JSON(), nullable=True),
        sa.Column('last_calculated', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_review', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    
    # Create indexes for user_reputations
    op.create_index('idx_user_reputations_trust_level', 'user_reputations', ['trust_level'], unique=False)
    op.create_index('idx_user_reputations_overall_score', 'user_reputations', ['overall_score'], unique=False)
    op.create_index(op.f('ix_user_reputations_user_id'), 'user_reputations', ['user_id'], unique=False)
    
    # Create reputation_events table
    op.create_table(
        'reputation_events',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('transaction_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('dimension', sa.String(length=100), nullable=True),
        sa.Column('score_change', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('evidence', sa.JSON(), nullable=True),
        sa.Column('source_user_id', sa.String(length=255), nullable=True),
        sa.Column('validated', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('transaction_id')
    )
    
    # Create indexes for reputation_events
    op.create_index('idx_reputation_events_user_type', 'reputation_events', ['user_id', 'event_type'], unique=False)
    op.create_index('idx_reputation_events_user_timestamp', 'reputation_events', ['user_id', 'timestamp'], unique=False)
    op.create_index(op.f('ix_reputation_events_user_id'), 'reputation_events', ['user_id'], unique=False)
    op.create_index(op.f('ix_reputation_events_event_type'), 'reputation_events', ['event_type'], unique=False)
    op.create_index(op.f('ix_reputation_events_source_user_id'), 'reputation_events', ['source_user_id'], unique=False)
    op.create_index(op.f('ix_reputation_events_timestamp'), 'reputation_events', ['timestamp'], unique=False)


def downgrade() -> None:
    """Downgrade schema: Drop reputation_events and user_reputations tables."""
    
    # Drop reputation_events table and its indexes
    op.drop_index(op.f('ix_reputation_events_timestamp'), table_name='reputation_events')
    op.drop_index(op.f('ix_reputation_events_source_user_id'), table_name='reputation_events')
    op.drop_index(op.f('ix_reputation_events_event_type'), table_name='reputation_events')
    op.drop_index(op.f('ix_reputation_events_user_id'), table_name='reputation_events')
    op.drop_index('idx_reputation_events_user_timestamp', table_name='reputation_events')
    op.drop_index('idx_reputation_events_user_type', table_name='reputation_events')
    op.drop_table('reputation_events')
    
    # Drop user_reputations table and its indexes
    op.drop_index(op.f('ix_user_reputations_user_id'), table_name='user_reputations')
    op.drop_index('idx_user_reputations_overall_score', table_name='user_reputations')
    op.drop_index('idx_user_reputations_trust_level', table_name='user_reputations')
    op.drop_table('user_reputations')
