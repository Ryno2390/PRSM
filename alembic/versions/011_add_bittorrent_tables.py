"""Add BitTorrent tables: torrent_manifests, torrent_seeder_log

torrent_manifests:  Torrent metadata and manifest storage
                    (tracks torrents available for P2P distribution)

torrent_seeder_log: Seeding activity and reward tracking
                    (records upload stats and FTNS rewards for seeders)

Revision ID: 011_add_bittorrent_tables
Revises: 010_add_pq_identities
Create Date: 2026-03-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '011_add_bittorrent_tables'
down_revision: Union[str, None] = '010_add_pq_identities'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── torrent_manifests ────────────────────────────────────────────────────────
    op.create_table(
        'torrent_manifests',
        sa.Column('infohash',         sa.String(40),                  nullable=False),  # SHA-1 hex
        sa.Column('name',             sa.Text(),                      nullable=False),
        sa.Column('total_size_bytes', sa.BigInteger(),                nullable=False),
        sa.Column('piece_length',     sa.Integer(),                   nullable=False),
        sa.Column('num_pieces',       sa.Integer(),                   nullable=False),
        sa.Column('magnet_uri',       sa.Text(),                      nullable=False),
        sa.Column('torrent_bytes',    sa.LargeBinary(),               nullable=True),
        sa.Column('ipfs_cid',         sa.String(64),                  nullable=True),   # Optional: .torrent pinned to IPFS
        sa.Column('created_at',       sa.DateTime(timezone=True),     nullable=False),
        sa.Column('created_by',       sa.String(64),                  nullable=False),  # node_id
        sa.Column('provenance_id',    postgresql.UUID(as_uuid=True),  nullable=True),
        sa.Column('metadata',         sa.JSON(),                      nullable=True),
        sa.PrimaryKeyConstraint('infohash'),
    )
    op.create_index('idx_torrent_created_at',    'torrent_manifests', ['created_at'])
    op.create_index('idx_torrent_created_by',    'torrent_manifests', ['created_by'])
    op.create_index(op.f('ix_torrent_manifests_ipfs_cid'), 'torrent_manifests', ['ipfs_cid'])

    # ── torrent_seeder_log ───────────────────────────────────────────────────────
    op.create_table(
        'torrent_seeder_log',
        sa.Column('id',              postgresql.UUID(as_uuid=True),  nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('infohash',        sa.String(40),                  nullable=False),
        sa.Column('seeder_node_id',  sa.String(64),                  nullable=False),
        sa.Column('bytes_uploaded',  sa.BigInteger(),                nullable=False, server_default='0'),
        sa.Column('reward_paid',     sa.Numeric(20, 8),              nullable=False, server_default='0'),
        sa.Column('started_at',      sa.DateTime(timezone=True),     nullable=False),
        sa.Column('last_seen_at',    sa.DateTime(timezone=True),     nullable=False),
        sa.Column('active',          sa.Boolean(),                   nullable=False, server_default='true'),
        sa.Column('seeder_metadata', sa.JSON(),                      nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['infohash'], ['torrent_manifests.infohash'], ondelete='CASCADE'),
    )
    op.create_index('idx_seeder_infohash_active',      'torrent_seeder_log', ['infohash', 'active'])
    op.create_index('idx_seeder_node_active',          'torrent_seeder_log', ['seeder_node_id', 'active'])
    op.create_index('idx_seeder_last_seen',            'torrent_seeder_log', ['last_seen_at'])
    op.create_index(op.f('ix_torrent_seeder_log_infohash'), 'torrent_seeder_log', ['infohash'])


def downgrade() -> None:
    # Drop torrent_seeder_log
    op.drop_index(op.f('ix_torrent_seeder_log_infohash'), table_name='torrent_seeder_log')
    op.drop_index('idx_seeder_last_seen',            table_name='torrent_seeder_log')
    op.drop_index('idx_seeder_node_active',          table_name='torrent_seeder_log')
    op.drop_index('idx_seeder_infohash_active',      table_name='torrent_seeder_log')
    op.drop_table('torrent_seeder_log')

    # Drop torrent_manifests
    op.drop_index(op.f('ix_torrent_manifests_ipfs_cid'), table_name='torrent_manifests')
    op.drop_index('idx_torrent_created_by',    table_name='torrent_manifests')
    op.drop_index('idx_torrent_created_at',    table_name='torrent_manifests')
    op.drop_table('torrent_manifests')
