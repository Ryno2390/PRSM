"""Add provenance_hash column to content_provenance

Phase 1.3 Task 2 — closes the deferred P2 from the Phase 1.2 codex
re-review. Adds a nullable provenance_hash column so the canonical
on-chain hash (keccak256(creator_address || sha3_256(file_bytes)))
survives node restarts.

Before this migration, provenance_hash was computed and gossiped at
upload time but dropped on the DB write, so a node restart silently
lost on-chain royalty routing for all uploads since the last restart.

Column is nullable. Legacy rows (uploaded before Phase 1.3 or without
a configured creator 0x address) stay NULL and fall back to local
royalty distribution. There is no backfill path — operators who want
on-chain routing for legacy uploads must re-register them via
`prsm provenance register`.

Revision ID: 015_add_provenance_hash_column
Revises: 014_add_federation_distillation_tables
Create Date: 2026-04-11
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '015_add_provenance_hash_column'
down_revision: Union[str, None] = '014_add_federation_distillation_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite requires batch_alter_table for ALTER TABLE ADD COLUMN with
    # non-trivial column specs; Postgres handles either path transparently.
    with op.batch_alter_table("content_provenance") as batch_op:
        batch_op.add_column(
            sa.Column(
                "provenance_hash",
                sa.String(length=66),
                nullable=True,
                comment=(
                    "Canonical on-chain provenance hash "
                    "(keccak256(creator_address || sha3_256(file_bytes))), "
                    "0x-prefixed hex. NULL for legacy rows and rows without "
                    "a configured creator 0x address — those fall back to "
                    "local royalty distribution."
                ),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("content_provenance") as batch_op:
        batch_op.drop_column("provenance_hash")
