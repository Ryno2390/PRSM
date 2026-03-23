"""Add federation and distillation tables

federation_peers:         Peer discovery and tracking for distributed RLT network
federation_messages:      Message log for federation P2P communication
distillation_jobs:        Distillation job lifecycle tracking
distillation_results:     Distillation result metrics and outcomes
emergency_protocol_actions: Emergency protocol audit trail

Revision ID: 014_add_federation_distillation_tables
Revises: 013_add_improvement_tables
Create Date: 2026-03-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '014_add_federation_distillation_tables'
down_revision: Union[str, None] = '013_add_improvement_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # federation_peers - Peer discovery and tracking
    op.create_table(
        'federation_peers',
        sa.Column('id',              postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('peer_id',         sa.Text(),                      nullable=False),
        sa.Column('address',         sa.Text(),                      nullable=False),
        sa.Column('port',            sa.Integer(),                   nullable=False),
        sa.Column('node_type',       sa.Text(),                      nullable=False, server_default='standard'),
        sa.Column('last_seen',       sa.Float(),                     nullable=False, server_default='0.0'),
        sa.Column('quality_score',   sa.Float(),                     nullable=False, server_default='0.5'),
        sa.Column('capabilities',    postgresql.JSONB(),             nullable=False, server_default='{}'),
        sa.Column('is_active',       sa.Boolean(),                   nullable=False, server_default='true'),
        sa.Column('created_at',      sa.Float(),                     nullable=False, server_default=sa.text('extract(epoch from now())')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('peer_id'),
    )
    op.create_index(op.f('ix_federation_peers_peer_id'), 'federation_peers', ['peer_id'])
    op.create_index(op.f('ix_federation_peers_last_seen'), 'federation_peers', ['last_seen'])

    # federation_messages - Message log for P2P communication
    op.create_table(
        'federation_messages',
        sa.Column('id',              postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('message_id',      sa.Text(),                      nullable=False),
        sa.Column('message_type',    sa.Text(),                      nullable=False),
        sa.Column('sender_id',       sa.Text(),                      nullable=False),
        sa.Column('recipient_id',    sa.Text(),                      nullable=True),  # NULL = broadcast
        sa.Column('payload',         postgresql.JSONB(),             nullable=False, server_default='{}'),
        sa.Column('sent_at',         sa.Float(),                     nullable=False),
        sa.Column('received_at',     sa.Float(),                     nullable=True),
        sa.Column('processed_at',    sa.Float(),                     nullable=True),
        sa.Column('status',          sa.Text(),                      nullable=False, server_default='pending'),  # pending/processed/failed
        sa.Column('error',           sa.Text(),                      nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('message_id'),
    )
    op.create_index(op.f('ix_federation_messages_message_type'), 'federation_messages', ['message_type'])
    op.create_index(op.f('ix_federation_messages_sender_id'), 'federation_messages', ['sender_id'])
    op.create_index(op.f('ix_federation_messages_status'), 'federation_messages', ['status'])

    # distillation_jobs - Distillation job lifecycle tracking
    op.create_table(
        'distillation_jobs',
        sa.Column('job_id',          sa.Text(),                      nullable=False),
        sa.Column('user_id',         sa.Text(),                      nullable=False),
        sa.Column('teacher_model_id', sa.Text(),                     nullable=False),
        sa.Column('student_model_id', sa.Text(),                     nullable=False),
        sa.Column('strategy',        sa.Text(),                      nullable=False),
        sa.Column('status',          sa.Text(),                      nullable=False, server_default='pending'),
        sa.Column('priority',        sa.Integer(),                   nullable=False, server_default='5'),
        sa.Column('config',          postgresql.JSONB(),             nullable=False, server_default='{}'),
        sa.Column('result',          postgresql.JSONB(),             nullable=True),
        sa.Column('error',           sa.Text(),                      nullable=True),
        sa.Column('created_at',      sa.Float(),                     nullable=False),
        sa.Column('started_at',      sa.Float(),                     nullable=True),
        sa.Column('completed_at',    sa.Float(),                     nullable=True),
        sa.PrimaryKeyConstraint('job_id'),
    )
    op.create_index(op.f('ix_distillation_jobs_user_id'), 'distillation_jobs', ['user_id'])
    op.create_index(op.f('ix_distillation_jobs_status'), 'distillation_jobs', ['status'])

    # distillation_results - Distillation result metrics
    op.create_table(
        'distillation_results',
        sa.Column('id',              postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('job_id',          sa.Text(),                      nullable=False),
        sa.Column('teacher_model_id', sa.Text(),                     nullable=False),
        sa.Column('student_model_id', sa.Text(),                     nullable=False),
        sa.Column('strategy',        sa.Text(),                      nullable=False),
        sa.Column('accuracy_score',  sa.Float(),                     nullable=False, server_default='0.0'),
        sa.Column('compression_ratio', sa.Float(),                   nullable=False, server_default='1.0'),
        sa.Column('training_loss',   sa.Float(),                     nullable=False, server_default='0.0'),
        sa.Column('validation_loss', sa.Float(),                     nullable=False, server_default='0.0'),
        sa.Column('tokens_used',     sa.Integer(),                   nullable=False, server_default='0'),
        sa.Column('ftns_cost',       sa.Numeric(20, 8),              nullable=False, server_default='0'),
        sa.Column('created_at',      sa.Float(),                     nullable=False),
        sa.Column('metadata',        postgresql.JSONB(),             nullable=False, server_default='{}'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['job_id'], ['distillation_jobs.job_id'], ondelete='CASCADE'),
    )
    op.create_index(op.f('ix_distillation_results_job_id'), 'distillation_results', ['job_id'])
    op.create_index(op.f('ix_distillation_results_strategy'), 'distillation_results', ['strategy'])

    # emergency_protocol_actions - Emergency protocol audit trail
    op.create_table(
        'emergency_protocol_actions',
        sa.Column('id',              postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('action_type',     sa.Text(),                      nullable=False),  # 'halt'/'limit_reduction'
        sa.Column('triggered_by',    sa.Text(),                      nullable=False),
        sa.Column('reason',          sa.Text(),                      nullable=True),
        sa.Column('original_value',  postgresql.JSONB(),             nullable=True),
        sa.Column('new_value',       postgresql.JSONB(),             nullable=True),
        sa.Column('created_at',      sa.Float(),                     nullable=False, server_default=sa.text('extract(epoch from now())')),
        sa.Column('resolved_at',     sa.Float(),                     nullable=True),
        sa.Column('resolved_by',     sa.Text(),                      nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_emergency_protocol_actions_action_type'), 'emergency_protocol_actions', ['action_type'])
    op.create_index(op.f('ix_emergency_protocol_actions_created_at'), 'emergency_protocol_actions', ['created_at'])


def downgrade() -> None:
    # Drop emergency_protocol_actions
    op.drop_index(op.f('ix_emergency_protocol_actions_created_at'), table_name='emergency_protocol_actions')
    op.drop_index(op.f('ix_emergency_protocol_actions_action_type'), table_name='emergency_protocol_actions')
    op.drop_table('emergency_protocol_actions')

    # Drop distillation_results
    op.drop_index(op.f('ix_distillation_results_strategy'), table_name='distillation_results')
    op.drop_index(op.f('ix_distillation_results_job_id'), table_name='distillation_results')
    op.drop_table('distillation_results')

    # Drop distillation_jobs
    op.drop_index(op.f('ix_distillation_jobs_status'), table_name='distillation_jobs')
    op.drop_index(op.f('ix_distillation_jobs_user_id'), table_name='distillation_jobs')
    op.drop_table('distillation_jobs')

    # Drop federation_messages
    op.drop_index(op.f('ix_federation_messages_status'), table_name='federation_messages')
    op.drop_index(op.f('ix_federation_messages_sender_id'), table_name='federation_messages')
    op.drop_index(op.f('ix_federation_messages_message_type'), table_name='federation_messages')
    op.drop_table('federation_messages')

    # Drop federation_peers
    op.drop_index(op.f('ix_federation_peers_last_seen'), table_name='federation_peers')
    op.drop_index(op.f('ix_federation_peers_peer_id'), table_name='federation_peers')
    op.drop_table('federation_peers')
