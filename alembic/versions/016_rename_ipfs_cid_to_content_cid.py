"""Rename ipfs_cid columns to content_cid

Native-storage migration PR 5 — closes the IPFS removal sweep. PRSM no
longer uses IPFS as a content backend (replaced by the proprietary
ContentStore + BitTorrent layer); the legacy column name is now
misleading. This rename keeps the SQLAlchemy model attribute and the
on-disk column name aligned.

Affected tables:
- ftns_transactions.ipfs_cid → content_cid
- teacher_models.ipfs_cid → content_cid (indexed)
- model_registry.ipfs_cid → content_cid (indexed)

The values themselves are unchanged — they remain content-addressed
hashes, just no longer IPFS-specific. Indexes are recreated under
the new column name where they existed.

Revision ID: 016_rename_ipfs_cid_to_content_cid
Revises: 015_add_provenance_hash_column
Create Date: 2026-05-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '016_rename_ipfs_cid_to_content_cid'
down_revision: Union[str, None] = '015_add_provenance_hash_column'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("ftns_transactions") as batch_op:
        batch_op.alter_column("ipfs_cid", new_column_name="content_cid")

    with op.batch_alter_table("teacher_models") as batch_op:
        batch_op.alter_column("ipfs_cid", new_column_name="content_cid")

    with op.batch_alter_table("model_registry") as batch_op:
        batch_op.alter_column("ipfs_cid", new_column_name="content_cid")


def downgrade() -> None:
    with op.batch_alter_table("ftns_transactions") as batch_op:
        batch_op.alter_column("content_cid", new_column_name="ipfs_cid")

    with op.batch_alter_table("teacher_models") as batch_op:
        batch_op.alter_column("content_cid", new_column_name="ipfs_cid")

    with op.batch_alter_table("model_registry") as batch_op:
        batch_op.alter_column("content_cid", new_column_name="ipfs_cid")
