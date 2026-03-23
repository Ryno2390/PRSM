"""Add improvement tables: ab_test_runs, ab_routing_assignments

ab_test_runs:            A/B test lifecycle tracking with rollout phases
                         (supports gradual rollout with rollback capability)

ab_routing_assignments:  Request routing assignments for A/B tests
                         (supports consistent hashing-based routing)

Revision ID: 013_add_improvement_tables
Revises: 012_add_provenance_tables
Create Date: 2026-03-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '013_add_improvement_tables'
down_revision: Union[str, None] = '012_add_provenance_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ab_test_runs
    op.create_table(
        'ab_test_runs',
        sa.Column('test_id',          postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name',             sa.Text(),                      nullable=False),
        sa.Column('status',           sa.String(32),                  nullable=False, server_default='staging'),
        sa.Column('variant_id',       sa.String(64),                  nullable=False),
        sa.Column('control_id',       sa.String(64),                  nullable=False),
        sa.Column('current_phase',    sa.Integer(),                   nullable=False, server_default='0'),
        sa.Column('rollout_pct',      sa.Float(),                     nullable=False, server_default='0'),
        sa.Column('started_at',       sa.DateTime(timezone=True),     nullable=True),
        sa.Column('completed_at',     sa.DateTime(timezone=True),     nullable=True),
        sa.Column('rollback_reason',  sa.Text(),                      nullable=True),
        sa.Column('metrics',          sa.JSON(),                      nullable=True),
        sa.Column('created_at',       sa.DateTime(timezone=True),     nullable=False),
        sa.PrimaryKeyConstraint('test_id'),
    )
    op.create_index('idx_ab_test_status', 'ab_test_runs', ['status'])
    op.create_index('idx_ab_test_created_at', 'ab_test_runs', ['created_at'])
    op.create_index(op.f('ix_ab_test_runs_test_id'), 'ab_test_runs', ['test_id'])
    op.create_index(op.f('ix_ab_test_runs_status'), 'ab_test_runs', ['status'])

    # ab_routing_assignments
    op.create_table(
        'ab_routing_assignments',
        sa.Column('assignment_id',    postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('test_id',          postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_hash',     sa.String(16),                 nullable=False),
        sa.Column('variant_id',       sa.String(64),                 nullable=False),
        sa.Column('assigned_at',      sa.DateTime(timezone=True),    nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('assignment_id'),
        sa.ForeignKeyConstraint(['test_id'], ['ab_test_runs.test_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_ab_routing_test_hash', 'ab_routing_assignments', ['test_id', 'request_hash'])
    op.create_index(op.f('ix_ab_routing_assignments_test_id'), 'ab_routing_assignments', ['test_id'])
    op.create_index(op.f('ix_ab_routing_assignments_request_hash'), 'ab_routing_assignments', ['request_hash'])


def downgrade() -> None:
    # Drop ab_routing_assignments
    op.drop_index(op.f('ix_ab_routing_assignments_request_hash'), table_name='ab_routing_assignments')
    op.drop_index(op.f('ix_ab_routing_assignments_test_id'), table_name='ab_routing_assignments')
    op.drop_index('idx_ab_routing_test_hash', table_name='ab_routing_assignments')
    op.drop_table('ab_routing_assignments')

    # Drop ab_test_runs
    op.drop_index(op.f('ix_ab_test_runs_status'), table_name='ab_test_runs')
    op.drop_index(op.f('ix_ab_test_runs_test_id'), table_name='ab_test_runs')
    op.drop_index('idx_ab_test_created_at', table_name='ab_test_runs')
    op.drop_index('idx_ab_test_status', table_name='ab_test_runs')
    op.drop_table('ab_test_runs')
