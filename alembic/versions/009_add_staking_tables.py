"""Add staking tables: ftns_stakes, ftns_unstake_requests, ftns_slash_events

ftns_stakes:           User stake records with rewards tracking
                       (supports general and specialized staking)

ftns_unstake_requests: Unstake request queue with cooldown period
                       (manages the unstaking workflow)

ftns_slash_events:     Slashing event records with appeal process
                       (tracks stake slashing and appeals)

Revision ID: 009_add_staking_tables
Revises: 008_add_user_api_configs
Create Date: 2026-03-20
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '009_add_staking_tables'
down_revision: Union[str, None] = '008_add_user_api_configs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── ftns_stakes ────────────────────────────────────────────────────────────
    op.create_table(
        'ftns_stakes',
        sa.Column('stake_id',               postgresql.UUID(as_uuid=True),  nullable=False),
        sa.Column('user_id',                sa.String(255),                 nullable=False),
        sa.Column('amount',                 sa.Float(),                     nullable=False),
        sa.Column('stake_type',             sa.String(50),                  nullable=False, server_default='general'),
        sa.Column('status',                 sa.String(50),                  nullable=False, server_default='active'),
        sa.Column('rewards_earned',         sa.Float(),                     nullable=False, server_default='0.0'),
        sa.Column('rewards_claimed',        sa.Float(),                     nullable=False, server_default='0.0'),
        sa.Column('last_reward_calculation', sa.DateTime(timezone=True),    nullable=False),
        sa.Column('staked_at',              sa.DateTime(timezone=True),     nullable=False),
        sa.Column('unstake_requested_at',   sa.DateTime(timezone=True),     nullable=True),
        sa.Column('withdrawn_at',           sa.DateTime(timezone=True),     nullable=True),
        sa.Column('lock_reason',            sa.Text(),                      nullable=True),
        sa.Column('stake_metadata',         sa.JSON(),                      nullable=True),
        sa.PrimaryKeyConstraint('stake_id'),
    )
    op.create_index('idx_stake_user_status', 'ftns_stakes', ['user_id', 'status'])
    op.create_index('idx_stake_staked_at',    'ftns_stakes', ['staked_at'])
    op.create_index(op.f('ix_ftns_stakes_user_id'), 'ftns_stakes', ['user_id'])
    op.create_index(op.f('ix_ftns_stakes_status'),   'ftns_stakes', ['status'])

    # ── ftns_unstake_requests ─────────────────────────────────────────────────
    op.create_table(
        'ftns_unstake_requests',
        sa.Column('request_id',           postgresql.UUID(as_uuid=True),  nullable=False),
        sa.Column('stake_id',             postgresql.UUID(as_uuid=True),  nullable=False),
        sa.Column('user_id',              sa.String(255),                 nullable=False),
        sa.Column('amount',               sa.Float(),                     nullable=False),
        sa.Column('status',               sa.String(50),                  nullable=False, server_default='pending'),
        sa.Column('requested_at',         sa.DateTime(timezone=True),     nullable=False),
        sa.Column('available_at',         sa.DateTime(timezone=True),     nullable=False),
        sa.Column('completed_at',         sa.DateTime(timezone=True),     nullable=True),
        sa.Column('cancellation_reason', sa.Text(),                      nullable=True),
        sa.Column('request_metadata',    sa.JSON(),                      nullable=True),
        sa.PrimaryKeyConstraint('request_id'),
    )
    op.create_index('idx_unstake_user_status',  'ftns_unstake_requests', ['user_id', 'status'])
    op.create_index('idx_unstake_available_at', 'ftns_unstake_requests', ['available_at'])
    op.create_index(op.f('ix_ftns_unstake_requests_stake_id'), 'ftns_unstake_requests', ['stake_id'])
    op.create_index(op.f('ix_ftns_unstake_requests_user_id'),  'ftns_unstake_requests', ['user_id'])
    op.create_index(op.f('ix_ftns_unstake_requests_status'),   'ftns_unstake_requests', ['status'])

    # ── ftns_slash_events ─────────────────────────────────────────────────────
    op.create_table(
        'ftns_slash_events',
        sa.Column('slash_id',          postgresql.UUID(as_uuid=True),  nullable=False),
        sa.Column('stake_id',          postgresql.UUID(as_uuid=True),  nullable=False),
        sa.Column('user_id',           sa.String(255),                 nullable=False),
        sa.Column('amount_slashed',   sa.Float(),                     nullable=False),
        sa.Column('reason',            sa.String(50),                  nullable=False),
        sa.Column('slash_rate',        sa.Float(),                     nullable=False),
        sa.Column('slashed_by',        sa.String(255),                 nullable=False),
        sa.Column('evidence',          sa.JSON(),                      nullable=True),
        sa.Column('slashed_at',        sa.DateTime(timezone=True),     nullable=False),
        sa.Column('appeal_deadline',   sa.DateTime(timezone=True),     nullable=True),
        sa.Column('appeal_status',     sa.String(50),                  nullable=True),
        sa.Column('appeal_evidence',   sa.JSON(),                      nullable=True),
        sa.Column('slash_metadata',    sa.JSON(),                      nullable=True),
        sa.PrimaryKeyConstraint('slash_id'),
    )
    op.create_index('idx_slash_user_slashed_at', 'ftns_slash_events', ['user_id', 'slashed_at'])
    op.create_index('idx_slash_stake',            'ftns_slash_events', ['stake_id'])
    op.create_index(op.f('ix_ftns_slash_events_stake_id'), 'ftns_slash_events', ['stake_id'])
    op.create_index(op.f('ix_ftns_slash_events_user_id'),  'ftns_slash_events', ['user_id'])


def downgrade() -> None:
    # Drop ftns_slash_events
    op.drop_index(op.f('ix_ftns_slash_events_user_id'),  table_name='ftns_slash_events')
    op.drop_index(op.f('ix_ftns_slash_events_stake_id'), table_name='ftns_slash_events')
    op.drop_index('idx_slash_stake',            table_name='ftns_slash_events')
    op.drop_index('idx_slash_user_slashed_at',  table_name='ftns_slash_events')
    op.drop_table('ftns_slash_events')

    # Drop ftns_unstake_requests
    op.drop_index(op.f('ix_ftns_unstake_requests_status'),   table_name='ftns_unstake_requests')
    op.drop_index(op.f('ix_ftns_unstake_requests_user_id'),  table_name='ftns_unstake_requests')
    op.drop_index(op.f('ix_ftns_unstake_requests_stake_id'),  table_name='ftns_unstake_requests')
    op.drop_index('idx_unstake_available_at', table_name='ftns_unstake_requests')
    op.drop_index('idx_unstake_user_status',  table_name='ftns_unstake_requests')
    op.drop_table('ftns_unstake_requests')

    # Drop ftns_stakes
    op.drop_index(op.f('ix_ftns_stakes_status'),   table_name='ftns_stakes')
    op.drop_index(op.f('ix_ftns_stakes_user_id'),  table_name='ftns_stakes')
    op.drop_index('idx_stake_staked_at',   table_name='ftns_stakes')
    op.drop_index('idx_stake_user_status', table_name='ftns_stakes')
    op.drop_table('ftns_stakes')
