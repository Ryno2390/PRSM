"""Add Phase 2 tables: content_provenance, governance_proposals, ftns_idempotency_keys

content_provenance:    IPFS upload provenance records for royalty persistence
                       (added Phase 2 Item 1 — previously in-memory only)

governance_proposals:  Governance proposal lifecycle storage
                       (added Phase 2 Item 3 — previously in-memory only)

ftns_idempotency_keys: Idempotency log for AtomicFTNSService operations
                       (referenced in atomic_ftns_service.py but absent from all
                       prior migrations; included here to close that gap)

Revision ID: 007_add_phase2_tables
Revises: 006_add_marketplace_activities
Create Date: 2026-03-20
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '007_add_phase2_tables'
down_revision: Union[str, None] = '006_add_marketplace_activities'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── content_provenance ────────────────────────────────────────────────────
    op.create_table(
        'content_provenance',
        sa.Column('cid',                      sa.String(255),               nullable=False),
        sa.Column('filename',                 sa.String(500),               nullable=False),
        sa.Column('size_bytes',               sa.BigInteger(),              nullable=False),
        sa.Column('content_hash',             sa.String(64),                nullable=False),
        sa.Column('creator_id',               sa.String(255),               nullable=False),
        sa.Column('provenance_signature',     sa.Text(),                    nullable=False),
        sa.Column('royalty_rate',             sa.Float(),                   nullable=False, server_default='0.01'),
        sa.Column('parent_cids',              sa.JSON(),                    nullable=True),
        sa.Column('access_count',             sa.Integer(),                 nullable=False, server_default='0'),
        sa.Column('total_royalties',          sa.Float(),                   nullable=False, server_default='0'),
        sa.Column('is_sharded',               sa.Boolean(),                 nullable=False, server_default='false'),
        sa.Column('manifest_cid',             sa.String(255),               nullable=True),
        sa.Column('total_shards',             sa.Integer(),                 nullable=False, server_default='0'),
        sa.Column('embedding_id',             sa.String(255),               nullable=True),
        sa.Column('near_duplicate_of',        sa.String(255),               nullable=True),
        sa.Column('near_duplicate_similarity',sa.Float(),                   nullable=True),
        sa.Column('created_at',               sa.DateTime(timezone=True),   nullable=False),
        sa.Column('updated_at',               sa.DateTime(timezone=True),   server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('cid'),
    )
    op.create_index('idx_provenance_creator',  'content_provenance', ['creator_id'])
    op.create_index('idx_provenance_hash',     'content_provenance', ['content_hash'])
    op.create_index('idx_provenance_created',  'content_provenance', ['created_at'])
    op.create_index('idx_provenance_manifest', 'content_provenance', ['manifest_cid'])

    # ── governance_proposals ─────────────────────────────────────────────────
    op.create_table(
        'governance_proposals',
        sa.Column('proposal_id',        postgresql.UUID(as_uuid=True),  nullable=False),
        sa.Column('proposer_id',        sa.String(255),                 nullable=False),
        sa.Column('title',              sa.String(500),                 nullable=False),
        sa.Column('description',        sa.Text(),                      nullable=False),
        sa.Column('proposal_type',      sa.String(100),                 nullable=False),
        sa.Column('status',             sa.String(50),                  nullable=False, server_default='active'),
        sa.Column('votes_for',          sa.Integer(),                   nullable=False, server_default='0'),
        sa.Column('votes_against',      sa.Integer(),                   nullable=False, server_default='0'),
        sa.Column('total_voting_power', sa.Float(),                     nullable=False, server_default='0'),
        sa.Column('required_quorum',    sa.Float(),                     nullable=True),
        sa.Column('voting_starts',      sa.DateTime(timezone=True),     nullable=True),
        sa.Column('voting_ends',        sa.DateTime(timezone=True),     nullable=True),
        sa.Column('proposal_metadata',  sa.JSON(),                      nullable=True),
        sa.Column('created_at',         sa.DateTime(timezone=True),     server_default=sa.text('now()')),
        sa.Column('updated_at',         sa.DateTime(timezone=True),     server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('proposal_id'),
    )
    op.create_index('idx_governance_status',   'governance_proposals', ['status'])
    op.create_index('idx_governance_proposer', 'governance_proposals', ['proposer_id'])
    op.create_index('idx_governance_created',  'governance_proposals', ['created_at'])

    # ── ftns_idempotency_keys ────────────────────────────────────────────────
    # Referenced in AtomicFTNSService._check_idempotency() but absent from all
    # prior migrations. Closes the gap so atomic FTNS operations don't fail on
    # a fresh database that was created via alembic rather than create_all().
    op.create_table(
        'ftns_idempotency_keys',
        sa.Column('idempotency_key',  sa.String(255),               nullable=False),
        sa.Column('transaction_id',   sa.String(255),               nullable=False),
        sa.Column('created_at',       sa.DateTime(timezone=True),   server_default=sa.text('now()')),
        sa.Column('expires_at',       sa.DateTime(timezone=True),   nullable=False),
        sa.PrimaryKeyConstraint('idempotency_key'),
    )
    op.create_index('idx_idempotency_expires', 'ftns_idempotency_keys', ['expires_at'])


def downgrade() -> None:
    op.drop_index('idx_idempotency_expires',   'ftns_idempotency_keys')
    op.drop_table('ftns_idempotency_keys')

    op.drop_index('idx_governance_created',    'governance_proposals')
    op.drop_index('idx_governance_proposer',   'governance_proposals')
    op.drop_index('idx_governance_status',     'governance_proposals')
    op.drop_table('governance_proposals')

    op.drop_index('idx_provenance_manifest',   'content_provenance')
    op.drop_index('idx_provenance_created',    'content_provenance')
    op.drop_index('idx_provenance_hash',       'content_provenance')
    op.drop_index('idx_provenance_creator',    'content_provenance')
    op.drop_table('content_provenance')
