"""Add all missing tables for Phase 7 production hardening

Revision ID: 005_add_missing_tables
Revises: 004_add_comprehensive_marketplace_system
Create Date: 2026-03-25

This migration adds all tables that exist in prsm/core/database.py ORM models
but were missing from the Alembic migration chain. This enables:
- Clean database initialization via `alembic upgrade head`
- Proper schema version control for production deployments
- Rollback capability for all tables

Tables added:
- Core session tables: prsm_sessions, reasoning_steps, safety_flags, architect_tasks
- FTNS system tables: ftns_balances, ftns_idempotency_keys, ftns_stakes,
  ftns_unstake_requests, ftns_slash_events
- Model ecosystem: teacher_models, model_registry, content_provenance
- Governance & safety: governance_proposals, circuit_breaker_events
- User configuration: user_api_configs
- P2P federation: peer_nodes, pq_identities, federation_peers, federation_messages
- Distillation: distillation_jobs, distillation_results
- Emergency protocol: emergency_protocol_actions
- Teams: teams, team_members, team_wallets, team_tasks, team_governance
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = '005_add_missing_tables'
down_revision: Union[str, None] = '004_add_comprehensive_marketplace_system'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add all missing tables"""

    # === Core Session Tables ===

    # prsm_sessions
    op.create_table(
        'prsm_sessions',
        sa.Column('session_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('nwtn_context_allocation', sa.Integer, default=0),
        sa.Column('context_used', sa.Integer, default=0),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('model_metadata', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_session_user_created', 'prsm_sessions', ['user_id', 'created_at'])
    op.create_index('idx_session_status_created', 'prsm_sessions', ['status', 'created_at'])
    op.create_index(op.f('ix_prsm_sessions_user_id'), 'prsm_sessions', ['user_id'])
    op.create_index(op.f('ix_prsm_sessions_status'), 'prsm_sessions', ['status'])

    # reasoning_steps
    op.create_table(
        'reasoning_steps',
        sa.Column('step_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', UUID(as_uuid=True), sa.ForeignKey('prsm_sessions.session_id'), nullable=False),
        sa.Column('agent_type', sa.String(50), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('input_data', sa.JSON, nullable=False),
        sa.Column('output_data', sa.JSON, nullable=False),
        sa.Column('execution_time', sa.Float, nullable=False),
        sa.Column('confidence_score', sa.Float),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_reasoning_session_timestamp', 'reasoning_steps', ['session_id', 'timestamp'])
    op.create_index('idx_reasoning_agent_type', 'reasoning_steps', ['agent_type'])

    # safety_flags
    op.create_table(
        'safety_flags',
        sa.Column('flag_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', UUID(as_uuid=True), sa.ForeignKey('prsm_sessions.session_id'), nullable=False),
        sa.Column('level', sa.String(20), nullable=False),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('triggered_by', sa.String(255), nullable=False),
        sa.Column('resolved', sa.Boolean, default=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_safety_level_timestamp', 'safety_flags', ['level', 'timestamp'])
    op.create_index('idx_safety_resolved', 'safety_flags', ['resolved'])
    op.create_index(op.f('ix_safety_flags_level'), 'safety_flags', ['level'])
    op.create_index(op.f('ix_safety_flags_resolved'), 'safety_flags', ['resolved'])

    # architect_tasks
    op.create_table(
        'architect_tasks',
        sa.Column('task_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', UUID(as_uuid=True), sa.ForeignKey('prsm_sessions.session_id'), nullable=False),
        sa.Column('parent_task_id', UUID(as_uuid=True), sa.ForeignKey('architect_tasks.task_id')),
        sa.Column('level', sa.Integer, default=0),
        sa.Column('instruction', sa.Text, nullable=False),
        sa.Column('complexity_score', sa.Float, default=0.0),
        sa.Column('dependencies', sa.JSON, default=list),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('assigned_agent', sa.String(255)),
        sa.Column('result', sa.JSON),
        sa.Column('execution_time', sa.Float),
        sa.Column('model_metadata', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_task_session_level', 'architect_tasks', ['session_id', 'level'])
    op.create_index('idx_task_status_created', 'architect_tasks', ['status', 'created_at'])
    op.create_index('idx_task_parent', 'architect_tasks', ['parent_task_id'])
    op.create_index(op.f('ix_architect_tasks_status'), 'architect_tasks', ['status'])

    # === FTNS System Tables ===

    # ftns_balances (separate from ftns_wallets in migration 003)
    op.create_table(
        'ftns_balances',
        sa.Column('user_id', sa.String(255), primary_key=True),
        sa.Column('balance', sa.Float, default=0.0, nullable=False),
        sa.Column('locked_balance', sa.Float, default=0.0, nullable=False),
        sa.Column('total_earned', sa.Float, default=0.0, nullable=False),
        sa.Column('total_spent', sa.Float, default=0.0, nullable=False),
        sa.Column('version', sa.Integer, default=1, nullable=False),
        sa.Column('last_transaction_id', UUID(as_uuid=True)),
        sa.Column('last_dividend', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_ftns_balance_updated', 'ftns_balances', ['updated_at'])

    # ftns_idempotency_keys
    op.create_table(
        'ftns_idempotency_keys',
        sa.Column('idempotency_key', sa.String(255), primary_key=True),
        sa.Column('transaction_id', sa.String(255), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('operation_type', sa.String(50), nullable=False),
        sa.Column('amount', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), default='completed'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    )

    # ftns_stakes
    op.create_table(
        'ftns_stakes',
        sa.Column('stake_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('amount', sa.Float, nullable=False),
        sa.Column('stake_type', sa.String(50), nullable=False, default='general'),
        sa.Column('status', sa.String(50), nullable=False, default='active'),
        sa.Column('rewards_earned', sa.Float, nullable=False, default=0.0),
        sa.Column('rewards_claimed', sa.Float, nullable=False, default=0.0),
        sa.Column('last_reward_calculation', sa.DateTime(timezone=True), nullable=False),
        sa.Column('staked_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('unstake_requested_at', sa.DateTime(timezone=True)),
        sa.Column('withdrawn_at', sa.DateTime(timezone=True)),
        sa.Column('lock_reason', sa.Text),
        sa.Column('stake_metadata', sa.JSON),
    )
    op.create_index('idx_stake_user_status', 'ftns_stakes', ['user_id', 'status'])
    op.create_index('idx_stake_staked_at', 'ftns_stakes', ['staked_at'])
    op.create_index(op.f('ix_ftns_stakes_user_id'), 'ftns_stakes', ['user_id'])
    op.create_index(op.f('ix_ftns_stakes_status'), 'ftns_stakes', ['status'])

    # ftns_unstake_requests
    op.create_table(
        'ftns_unstake_requests',
        sa.Column('request_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('stake_id', UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('amount', sa.Float, nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('requested_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('cancellation_reason', sa.Text),
        sa.Column('request_metadata', sa.JSON),
    )
    op.create_index('idx_unstake_user_status', 'ftns_unstake_requests', ['user_id', 'status'])
    op.create_index('idx_unstake_available_at', 'ftns_unstake_requests', ['available_at'])
    op.create_index(op.f('ix_ftns_unstake_requests_stake_id'), 'ftns_unstake_requests', ['stake_id'])
    op.create_index(op.f('ix_ftns_unstake_requests_user_id'), 'ftns_unstake_requests', ['user_id'])
    op.create_index(op.f('ix_ftns_unstake_requests_status'), 'ftns_unstake_requests', ['status'])

    # ftns_slash_events
    op.create_table(
        'ftns_slash_events',
        sa.Column('slash_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('stake_id', UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('amount_slashed', sa.Float, nullable=False),
        sa.Column('reason', sa.String(50), nullable=False),
        sa.Column('slash_rate', sa.Float, nullable=False),
        sa.Column('slashed_by', sa.String(255), nullable=False),
        sa.Column('evidence', sa.JSON),
        sa.Column('slashed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('appeal_deadline', sa.DateTime(timezone=True)),
        sa.Column('appeal_status', sa.String(50)),
        sa.Column('appeal_evidence', sa.JSON),
        sa.Column('slash_metadata', sa.JSON),
    )
    op.create_index('idx_slash_user_slashed_at', 'ftns_slash_events', ['user_id', 'slashed_at'])
    op.create_index('idx_slash_stake', 'ftns_slash_events', ['stake_id'])
    op.create_index(op.f('ix_ftns_slash_events_stake_id'), 'ftns_slash_events', ['stake_id'])
    op.create_index(op.f('ix_ftns_slash_events_user_id'), 'ftns_slash_events', ['user_id'])

    # === Model Ecosystem Tables ===

    # teacher_models
    op.create_table(
        'teacher_models',
        sa.Column('teacher_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('specialization', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), default='teacher'),
        sa.Column('performance_score', sa.Float, default=0.0),
        sa.Column('curriculum_ids', sa.JSON, default=list),
        sa.Column('student_models', sa.JSON, default=list),
        sa.Column('rlvr_score', sa.Float),
        sa.Column('ipfs_cid', sa.String(255)),
        sa.Column('version', sa.String(50), default='1.0.0'),
        sa.Column('active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_teacher_specialization_active', 'teacher_models', ['specialization', 'active'])
    op.create_index('idx_teacher_performance', 'teacher_models', ['performance_score'])
    op.create_unique_constraint('uq_teacher_name_version', 'teacher_models', ['name', 'version'])
    op.create_index(op.f('ix_teacher_models_specialization'), 'teacher_models', ['specialization'])
    op.create_index(op.f('ix_teacher_models_model_type'), 'teacher_models', ['model_type'])
    op.create_index(op.f('ix_teacher_models_ipfs_cid'), 'teacher_models', ['ipfs_cid'])
    op.create_index(op.f('ix_teacher_models_active'), 'teacher_models', ['active'])

    # model_registry
    op.create_table(
        'model_registry',
        sa.Column('model_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('specialization', sa.String(255)),
        sa.Column('owner_id', sa.String(255), nullable=False),
        sa.Column('ipfs_cid', sa.String(255)),
        sa.Column('version', sa.String(50), default='1.0.0'),
        sa.Column('performance_metrics', sa.JSON, default=dict),
        sa.Column('resource_requirements', sa.JSON, default=dict),
        sa.Column('pricing_model', sa.JSON, default=dict),
        sa.Column('availability_status', sa.String(50), default='available'),
        sa.Column('total_usage_hours', sa.Float, default=0.0),
        sa.Column('reputation_score', sa.Float, default=0.5),
        sa.Column('active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_model_type_specialization', 'model_registry', ['model_type', 'specialization'])
    op.create_index('idx_model_owner_active', 'model_registry', ['owner_id', 'active'])
    op.create_index('idx_model_availability', 'model_registry', ['availability_status'])
    op.create_unique_constraint('uq_model_name_version_owner', 'model_registry', ['name', 'version', 'owner_id'])
    op.create_index(op.f('ix_model_registry_model_type'), 'model_registry', ['model_type'])
    op.create_index(op.f('ix_model_registry_specialization'), 'model_registry', ['specialization'])
    op.create_index(op.f('ix_model_registry_owner_id'), 'model_registry', ['owner_id'])
    op.create_index(op.f('ix_model_registry_ipfs_cid'), 'model_registry', ['ipfs_cid'])
    op.create_index(op.f('ix_model_registry_availability_status'), 'model_registry', ['availability_status'])
    op.create_index(op.f('ix_model_registry_active'), 'model_registry', ['active'])

    # content_provenance
    op.create_table(
        'content_provenance',
        sa.Column('cid', sa.String(255), primary_key=True),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('size_bytes', sa.BigInteger, nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('creator_id', sa.String(255), nullable=False),
        sa.Column('provenance_signature', sa.Text, nullable=False),
        sa.Column('royalty_rate', sa.Float, nullable=False, default=0.01),
        sa.Column('parent_cids', sa.JSON, default=list),
        sa.Column('access_count', sa.Integer, nullable=False, default=0),
        sa.Column('total_royalties', sa.Float, nullable=False, default=0.0),
        sa.Column('is_sharded', sa.Boolean, nullable=False, default=False),
        sa.Column('manifest_cid', sa.String(255)),
        sa.Column('total_shards', sa.Integer, nullable=False, default=0),
        sa.Column('embedding_id', sa.String(255)),
        sa.Column('near_duplicate_of', sa.String(255)),
        sa.Column('near_duplicate_similarity', sa.Float),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_provenance_creator', 'content_provenance', ['creator_id'])
    op.create_index('idx_provenance_hash', 'content_provenance', ['content_hash'])
    op.create_index('idx_provenance_created', 'content_provenance', ['created_at'])
    op.create_index(op.f('ix_content_provenance_creator_id'), 'content_provenance', ['creator_id'])
    op.create_index(op.f('ix_content_provenance_manifest_cid'), 'content_provenance', ['manifest_cid'])

    # === Governance & Safety Tables ===

    # governance_proposals
    op.create_table(
        'governance_proposals',
        sa.Column('proposal_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('proposer_id', sa.String(255), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('proposal_type', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='active'),
        sa.Column('votes_for', sa.Integer, nullable=False, default=0),
        sa.Column('votes_against', sa.Integer, nullable=False, default=0),
        sa.Column('total_voting_power', sa.Float, nullable=False, default=0.0),
        sa.Column('required_quorum', sa.Float),
        sa.Column('voting_starts', sa.DateTime(timezone=True)),
        sa.Column('voting_ends', sa.DateTime(timezone=True)),
        sa.Column('proposal_metadata', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_governance_status', 'governance_proposals', ['status'])
    op.create_index('idx_governance_proposer', 'governance_proposals', ['proposer_id'])
    op.create_index('idx_governance_created', 'governance_proposals', ['created_at'])
    op.create_index(op.f('ix_governance_proposals_proposal_type'), 'governance_proposals', ['proposal_type'])
    op.create_index(op.f('ix_governance_proposals_status'), 'governance_proposals', ['status'])
    op.create_index(op.f('ix_governance_proposals_proposer_id'), 'governance_proposals', ['proposer_id'])

    # circuit_breaker_events
    op.create_table(
        'circuit_breaker_events',
        sa.Column('event_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('triggered_by', sa.String(255), nullable=False),
        sa.Column('safety_level', sa.String(20), nullable=False),
        sa.Column('reason', sa.Text, nullable=False),
        sa.Column('affected_components', sa.JSON, default=list),
        sa.Column('resolution_action', sa.Text),
        sa.Column('resolved_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_circuit_breaker_level_created', 'circuit_breaker_events', ['safety_level', 'created_at'])
    op.create_index('idx_circuit_breaker_resolved', 'circuit_breaker_events', ['resolved_at'])
    op.create_index(op.f('ix_circuit_breaker_events_safety_level'), 'circuit_breaker_events', ['safety_level'])

    # === User Configuration ===

    # user_api_configs
    op.create_table(
        'user_api_configs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('config_data', sa.JSON, nullable=False, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_unique_constraint('uq_user_api_config_provider', 'user_api_configs', ['user_id', 'provider'])
    op.create_index('idx_user_api_config_user', 'user_api_configs', ['user_id'])
    op.create_index(op.f('ix_user_api_configs_user_id'), 'user_api_configs', ['user_id'])

    # === P2P Federation Tables ===

    # peer_nodes
    op.create_table(
        'peer_nodes',
        sa.Column('node_id', sa.String(255), primary_key=True),
        sa.Column('peer_id', sa.String(255), nullable=False, unique=True),
        sa.Column('multiaddr', sa.String(500), nullable=False),
        sa.Column('capabilities', sa.JSON, default=list),
        sa.Column('reputation_score', sa.Float, default=0.5),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_peer_active_last_seen', 'peer_nodes', ['active', 'last_seen'])
    op.create_index('idx_peer_reputation', 'peer_nodes', ['reputation_score'])
    op.create_index(op.f('ix_peer_nodes_active'), 'peer_nodes', ['active'])

    # pq_identities
    op.create_table(
        'pq_identities',
        sa.Column('user_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('security_level', sa.String(50), nullable=False),
        sa.Column('keypair_json', sa.Text, nullable=False),
        sa.Column('signature_type', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_used', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_pq_identity_user', 'pq_identities', ['user_id'])

    # federation_peers
    op.create_table(
        'federation_peers',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('peer_id', sa.String(255), nullable=False, unique=True),
        sa.Column('address', sa.String(255), nullable=False),
        sa.Column('port', sa.Integer, nullable=False),
        sa.Column('node_type', sa.String(50), nullable=False, default='standard'),
        sa.Column('last_seen', sa.Float, nullable=False, default=0.0),
        sa.Column('quality_score', sa.Float, nullable=False, default=0.5),
        sa.Column('capabilities', sa.JSON, nullable=False, default=dict),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.Float, nullable=False),
    )
    op.create_index('ix_federation_peers_last_seen', 'federation_peers', ['last_seen'])

    # federation_messages
    op.create_table(
        'federation_messages',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('message_id', sa.String(255), nullable=False, unique=True),
        sa.Column('message_type', sa.String(100), nullable=False),
        sa.Column('sender_id', sa.String(255), nullable=False),
        sa.Column('recipient_id', sa.String(255)),
        sa.Column('payload', sa.JSON, nullable=False, default=dict),
        sa.Column('sent_at', sa.Float, nullable=False),
        sa.Column('received_at', sa.Float),
        sa.Column('processed_at', sa.Float),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('error', sa.Text),
    )
    op.create_index('ix_federation_messages_message_type', 'federation_messages', ['message_type'])
    op.create_index('ix_federation_messages_sender_id', 'federation_messages', ['sender_id'])
    op.create_index('ix_federation_messages_status', 'federation_messages', ['status'])

    # === Distillation Tables ===

    # distillation_jobs
    op.create_table(
        'distillation_jobs',
        sa.Column('job_id', sa.String(255), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('teacher_model_id', sa.String(255), nullable=False),
        sa.Column('student_model_id', sa.String(255), nullable=False),
        sa.Column('strategy', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('priority', sa.Integer, nullable=False, default=5),
        sa.Column('config', sa.JSON, nullable=False, default=dict),
        sa.Column('result', sa.JSON),
        sa.Column('error', sa.Text),
        sa.Column('created_at', sa.Float, nullable=False),
        sa.Column('started_at', sa.Float),
        sa.Column('completed_at', sa.Float),
    )
    op.create_index('ix_distillation_jobs_user_id', 'distillation_jobs', ['user_id'])
    op.create_index('ix_distillation_jobs_status', 'distillation_jobs', ['status'])

    # distillation_results
    op.create_table(
        'distillation_results',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', sa.String(255), sa.ForeignKey('distillation_jobs.job_id', ondelete='CASCADE'), nullable=False),
        sa.Column('teacher_model_id', sa.String(255), nullable=False),
        sa.Column('student_model_id', sa.String(255), nullable=False),
        sa.Column('strategy', sa.String(100), nullable=False),
        sa.Column('accuracy_score', sa.Float, nullable=False, default=0.0),
        sa.Column('compression_ratio', sa.Float, nullable=False, default=1.0),
        sa.Column('training_loss', sa.Float, nullable=False, default=0.0),
        sa.Column('validation_loss', sa.Float, nullable=False, default=0.0),
        sa.Column('tokens_used', sa.Integer, nullable=False, default=0),
        sa.Column('ftns_cost', sa.Float, nullable=False, default=0.0),
        sa.Column('created_at', sa.Float, nullable=False),
        sa.Column('extra_metadata', sa.JSON, nullable=False, default=dict),
    )
    op.create_index('ix_distillation_results_job_id', 'distillation_results', ['job_id'])
    op.create_index('ix_distillation_results_strategy', 'distillation_results', ['strategy'])

    # === Emergency Protocol ===

    # emergency_protocol_actions
    op.create_table(
        'emergency_protocol_actions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('action_type', sa.String(50), nullable=False),
        sa.Column('triggered_by', sa.String(255), nullable=False),
        sa.Column('reason', sa.Text),
        sa.Column('original_value', sa.JSON),
        sa.Column('new_value', sa.JSON),
        sa.Column('created_at', sa.Float, nullable=False),
        sa.Column('resolved_at', sa.Float),
        sa.Column('resolved_by', sa.String(255)),
    )
    op.create_index('ix_emergency_protocol_actions_action_type', 'emergency_protocol_actions', ['action_type'])
    op.create_index('ix_emergency_protocol_actions_created_at', 'emergency_protocol_actions', ['created_at'])

    # === Teams Tables ===

    # teams
    op.create_table(
        'teams',
        sa.Column('team_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('team_type', sa.String(50), default='research'),
        sa.Column('avatar_url', sa.String(500)),
        sa.Column('logo_url', sa.String(500)),
        sa.Column('governance_model', sa.String(50), default='democratic'),
        sa.Column('reward_policy', sa.String(50), default='proportional'),
        sa.Column('is_public', sa.Boolean, default=True),
        sa.Column('max_members', sa.Integer),
        sa.Column('entry_stake_required', sa.Float, default=0.0),
        sa.Column('research_domains', sa.JSON, default=list),
        sa.Column('keywords', sa.JSON, default=list),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('founding_date', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('member_count', sa.Integer, default=0),
        sa.Column('total_ftns_earned', sa.Float, default=0.0),
        sa.Column('total_tasks_completed', sa.Integer, default=0),
        sa.Column('impact_score', sa.Float, default=0.0),
        sa.Column('external_links', sa.JSON, default=dict),
        sa.Column('contact_info', sa.JSON, default=dict),
        sa.Column('model_metadata', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index(op.f('ix_teams_team_type'), 'teams', ['team_type'])
    op.create_index(op.f('ix_teams_is_public'), 'teams', ['is_public'])
    op.create_index(op.f('ix_teams_is_active'), 'teams', ['is_active'])
    op.create_index(op.f('ix_teams_impact_score'), 'teams', ['impact_score'])

    # team_members
    op.create_table(
        'team_members',
        sa.Column('membership_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('team_id', UUID(as_uuid=True), sa.ForeignKey('teams.team_id'), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('role', sa.String(50), default='member'),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('invited_by', sa.String(255)),
        sa.Column('invitation_message', sa.Text),
        sa.Column('invited_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('joined_at', sa.DateTime(timezone=True)),
        sa.Column('left_at', sa.DateTime(timezone=True)),
        sa.Column('ftns_contributed', sa.Float, default=0.0),
        sa.Column('tasks_completed', sa.Integer, default=0),
        sa.Column('models_contributed', sa.Integer, default=0),
        sa.Column('datasets_uploaded', sa.Integer, default=0),
        sa.Column('performance_score', sa.Float, default=0.0),
        sa.Column('reputation_score', sa.Float, default=0.5),
        sa.Column('collaboration_score', sa.Float, default=0.0),
        sa.Column('can_invite_members', sa.Boolean, default=False),
        sa.Column('can_manage_tasks', sa.Boolean, default=False),
        sa.Column('can_access_treasury', sa.Boolean, default=False),
        sa.Column('can_vote', sa.Boolean, default=True),
        sa.Column('bio', sa.Text),
        sa.Column('expertise_areas', sa.JSON, default=list),
        sa.Column('public_profile', sa.Boolean, default=True),
        sa.Column('model_metadata', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_team_member_team_user', 'team_members', ['team_id', 'user_id'])
    op.create_index('idx_team_member_status', 'team_members', ['status'])
    op.create_index('idx_team_member_role', 'team_members', ['role'])
    op.create_index(op.f('ix_team_members_user_id'), 'team_members', ['user_id'])
    op.create_index(op.f('ix_team_members_role'), 'team_members', ['role'])
    op.create_index(op.f('ix_team_members_status'), 'team_members', ['status'])

    # team_wallets
    op.create_table(
        'team_wallets',
        sa.Column('wallet_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('team_id', UUID(as_uuid=True), sa.ForeignKey('teams.team_id'), nullable=False, unique=True),
        sa.Column('is_multisig', sa.Boolean, default=True),
        sa.Column('required_signatures', sa.Integer, default=1),
        sa.Column('authorized_signers', sa.JSON, default=list),
        sa.Column('total_balance', sa.Float, default=0.0),
        sa.Column('available_balance', sa.Float, default=0.0),
        sa.Column('locked_balance', sa.Float, default=0.0),
        sa.Column('reward_policy', sa.String(50), default='proportional'),
        sa.Column('policy_config', sa.JSON, default=dict),
        sa.Column('distribution_metrics', sa.JSON, default=list),
        sa.Column('metric_weights', sa.JSON, default=list),
        sa.Column('auto_distribution_enabled', sa.Boolean, default=False),
        sa.Column('distribution_frequency_days', sa.Integer, default=30),
        sa.Column('last_distribution', sa.DateTime(timezone=True)),
        sa.Column('wallet_address', sa.String(255)),
        sa.Column('spending_limits', sa.JSON, default=dict),
        sa.Column('emergency_freeze', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_team_wallet_balance', 'team_wallets', ['total_balance'])
    op.create_index('idx_team_wallet_emergency', 'team_wallets', ['emergency_freeze'])

    # team_tasks
    op.create_table(
        'team_tasks',
        sa.Column('task_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('team_id', UUID(as_uuid=True), sa.ForeignKey('teams.team_id'), nullable=False),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('task_type', sa.String(50), default='research'),
        sa.Column('assigned_to', sa.JSON, default=list),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('priority', sa.String(20), default='medium'),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('progress_percentage', sa.Float, default=0.0),
        sa.Column('ftns_budget', sa.Float, default=0.0),
        sa.Column('ftns_spent', sa.Float, default=0.0),
        sa.Column('due_date', sa.DateTime(timezone=True)),
        sa.Column('estimated_hours', sa.Float),
        sa.Column('actual_hours', sa.Float),
        sa.Column('output_artifacts', sa.JSON, default=list),
        sa.Column('output_models', sa.JSON, default=list),
        sa.Column('performance_metrics', sa.JSON, default=dict),
        sa.Column('requires_consensus', sa.Boolean, default=False),
        sa.Column('consensus_threshold', sa.Float, default=0.6),
        sa.Column('votes_for', sa.Integer, default=0),
        sa.Column('votes_against', sa.Integer, default=0),
        sa.Column('tags', sa.JSON, default=list),
        sa.Column('external_links', sa.JSON, default=dict),
        sa.Column('model_metadata', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index(op.f('ix_team_tasks_task_type'), 'team_tasks', ['task_type'])
    op.create_index(op.f('ix_team_tasks_priority'), 'team_tasks', ['priority'])
    op.create_index(op.f('ix_team_tasks_status'), 'team_tasks', ['status'])

    # team_governance
    op.create_table(
        'team_governance',
        sa.Column('governance_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('team_id', UUID(as_uuid=True), sa.ForeignKey('teams.team_id'), nullable=False, unique=True),
        sa.Column('model', sa.String(50), default='democratic'),
        sa.Column('constitution', sa.JSON, default=dict),
        sa.Column('voting_period_days', sa.Integer, default=7),
        sa.Column('quorum_percentage', sa.Float, default=0.5),
        sa.Column('approval_threshold', sa.Float, default=0.6),
        sa.Column('role_assignments', sa.JSON, default=dict),
        sa.Column('role_term_limits', sa.JSON, default=dict),
        sa.Column('proposal_types', sa.JSON, default=list),
        sa.Column('type_thresholds', sa.JSON, default=dict),
        sa.Column('emergency_roles', sa.JSON, default=list),
        sa.Column('emergency_procedures', sa.JSON, default=dict),
        sa.Column('max_owner_power', sa.Float, default=0.4),
        sa.Column('member_protection_threshold', sa.Float, default=0.25),
        sa.Column('active_proposals', sa.JSON, default=list),
        sa.Column('total_proposals', sa.Integer, default=0),
        sa.Column('proposals_passed', sa.Integer, default=0),
        sa.Column('average_participation', sa.Float, default=0.0),
        sa.Column('last_vote', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_team_governance_model', 'team_governance', ['model'])

    print("✅ All missing tables created successfully")
    print("📊 Created 27 tables covering:")
    print("   - Core session tables (4)")
    print("   - FTNS system tables (5)")
    print("   - Model ecosystem tables (3)")
    print("   - Governance & safety tables (2)")
    print("   - User configuration tables (1)")
    print("   - P2P federation tables (4)")
    print("   - Distillation tables (2)")
    print("   - Emergency protocol tables (1)")
    print("   - Teams tables (5)")


def downgrade() -> None:
    """Remove all missing tables in reverse order"""

    # Teams tables (reverse order)
    op.drop_table('team_governance')
    op.drop_table('team_tasks')
    op.drop_table('team_wallets')
    op.drop_table('team_members')
    op.drop_table('teams')

    # Emergency protocol
    op.drop_table('emergency_protocol_actions')

    # Distillation tables
    op.drop_table('distillation_results')
    op.drop_table('distillation_jobs')

    # P2P federation tables
    op.drop_table('federation_messages')
    op.drop_table('federation_peers')
    op.drop_table('pq_identities')
    op.drop_table('peer_nodes')

    # User configuration
    op.drop_table('user_api_configs')

    # Governance & safety tables
    op.drop_table('circuit_breaker_events')
    op.drop_table('governance_proposals')

    # Model ecosystem tables
    op.drop_table('content_provenance')
    op.drop_table('model_registry')
    op.drop_table('teacher_models')

    # FTNS system tables
    op.drop_table('ftns_slash_events')
    op.drop_table('ftns_unstake_requests')
    op.drop_table('ftns_stakes')
    op.drop_table('ftns_idempotency_keys')
    op.drop_table('ftns_balances')

    # Core session tables (reverse order)
    op.drop_table('architect_tasks')
    op.drop_table('safety_flags')
    op.drop_table('reasoning_steps')
    op.drop_table('prsm_sessions')

    print("❌ All missing tables removed")
