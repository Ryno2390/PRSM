"""Add provenance tables: provenance_records, reasoning_chains

provenance_records:    Individual provenance records for data lineage tracking
                       (supports trust scoring, content hashing, verification)

reasoning_chains:      Reasoning chain provenance for step-by-step tracking
                       (supports multi-step reasoning with verification)

Revision ID: 012_add_provenance_tables
Revises: 011_add_bittorrent_tables
Create Date: 2026-03-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '012_add_provenance_tables'
down_revision: Union[str, None] = '011_add_bittorrent_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # provenance_records
    op.create_table(
        'provenance_records',
        sa.Column('record_id',           sa.String(64),                 nullable=False),
        sa.Column('provenance_type',     sa.String(50),                 nullable=False),
        sa.Column('source_entity',       sa.String(255),                nullable=False),
        sa.Column('target_entity',       sa.String(255),                nullable=False),
        sa.Column('operation',           sa.String(255),                nullable=False),
        sa.Column('inputs',              sa.JSON(),                     nullable=True),
        sa.Column('outputs',             sa.JSON(),                     nullable=True),
        sa.Column('metadata',            sa.JSON(),                     nullable=True),
        sa.Column('trust_level',         sa.String(32),                 nullable=False, server_default='credible'),
        sa.Column('verification_status', sa.Boolean(),                  nullable=False, server_default='false'),
        sa.Column('content_hash',        sa.String(64),                 nullable=True),
        sa.Column('timestamp',           sa.DateTime(timezone=True),    nullable=False),
        sa.Column('active',              sa.Boolean(),                  nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('record_id'),
    )
    op.create_index('idx_provenance_source_entity', 'provenance_records', ['source_entity'])
    op.create_index('idx_provenance_target_entity', 'provenance_records', ['target_entity'])
    op.create_index('idx_provenance_trust_level',   'provenance_records', ['trust_level'])
    op.create_index('idx_provenance_timestamp',     'provenance_records', ['timestamp'])
    op.create_index('idx_provenance_content_hash',  'provenance_records', ['content_hash'])
    op.create_index(op.f('ix_provenance_records_record_id'), 'provenance_records', ['record_id'])
    op.create_index(op.f('ix_provenance_records_source_entity'), 'provenance_records', ['source_entity'])

    # reasoning_chains
    op.create_table(
        'reasoning_chains',
        sa.Column('chain_id',          postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('node_id',           sa.String(64),                 nullable=True),
        sa.Column('query',             sa.Text(),                     nullable=False),
        sa.Column('final_conclusion',  sa.Text(),                     nullable=True),
        sa.Column('confidence_score',  sa.Float(),                    nullable=True),
        sa.Column('total_steps',       sa.Integer(),                  nullable=False, server_default='0'),
        sa.Column('step_count',        sa.Integer(),                  nullable=False, server_default='0'),
        sa.Column('finalized',         sa.Boolean(),                  nullable=False, server_default='false'),
        sa.Column('finalized_at',      sa.DateTime(timezone=True),    nullable=True),
        sa.Column('metadata',          sa.JSON(),                     nullable=True),
        sa.Column('created_at',        sa.DateTime(timezone=True),    nullable=False),
        sa.PrimaryKeyConstraint('chain_id'),
    )
    op.create_index('idx_chains_node_id', 'reasoning_chains', ['node_id'])
    op.create_index('idx_chains_created_at', 'reasoning_chains', ['created_at'])
    op.create_index(op.f('ix_reasoning_chains_chain_id'), 'reasoning_chains', ['chain_id'])
    op.create_index(op.f('ix_reasoning_chains_node_id'), 'reasoning_chains', ['node_id'])


def downgrade() -> None:
    # Drop reasoning_chains
    op.drop_index(op.f('ix_reasoning_chains_node_id'), table_name='reasoning_chains')
    op.drop_index(op.f('ix_reasoning_chains_chain_id'), table_name='reasoning_chains')
    op.drop_index('idx_chains_created_at', table_name='reasoning_chains')
    op.drop_index('idx_chains_node_id', table_name='reasoning_chains')
    op.drop_table('reasoning_chains')

    # Drop provenance_records
    op.drop_index(op.f('ix_provenance_records_source_entity'), table_name='provenance_records')
    op.drop_index(op.f('ix_provenance_records_record_id'), table_name='provenance_records')
    op.drop_index('idx_provenance_content_hash', table_name='provenance_records')
    op.drop_index('idx_provenance_timestamp', table_name='provenance_records')
    op.drop_index('idx_provenance_trust_level', table_name='provenance_records')
    op.drop_index('idx_provenance_target_entity', table_name='provenance_records')
    op.drop_index('idx_provenance_source_entity', table_name='provenance_records')
    op.drop_table('provenance_records')
