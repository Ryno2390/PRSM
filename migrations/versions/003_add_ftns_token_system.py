"""Add FTNS Token System

Revision ID: 003_add_ftns_token_system
Revises: 002_add_auth_system
Create Date: 2025-06-10 12:00:00.000000

This migration adds the complete FTNS (Fungible Tokens for Node Support) 
database schema to support:

- User wallet management with balance tracking
- Complete transaction history with blockchain integration
- Provenance tracking for royalty calculations  
- Marketplace listings and rental transactions
- Dividend distribution management
- Governance voting records
- Comprehensive audit logging
- Price history for market analytics

The schema is designed to support both current simulation mode
and future blockchain integration with production-ready features.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

# revision identifiers
revision = '003_add_ftns_token_system'
down_revision = '002_add_auth_system'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add FTNS token system tables"""
    
    # Create FTNS wallets table
    op.create_table(
        'ftns_wallets',
        sa.Column('wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('wallet_type', sa.String(length=50), nullable=False, server_default='standard'),
        sa.Column('balance', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('locked_balance', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('staked_balance', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('blockchain_address', sa.String(length=255), nullable=True),
        sa.Column('public_key', sa.Text(), nullable=True),
        sa.Column('wallet_version', sa.String(length=20), nullable=False, server_default='1.0'),
        sa.Column('multisig_threshold', sa.Integer(), nullable=True),
        sa.Column('multisig_participants', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('security_level', sa.String(length=20), nullable=False, server_default='standard'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_transaction', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.CheckConstraint('balance >= 0', name='positive_balance'),
        sa.CheckConstraint('locked_balance >= 0', name='positive_locked_balance'),
        sa.CheckConstraint('staked_balance >= 0', name='positive_staked_balance'),
        sa.PrimaryKeyConstraint('wallet_id'),
        sa.UniqueConstraint('user_id'),
        sa.UniqueConstraint('blockchain_address')
    )
    
    # Create indexes for wallets
    op.create_index('idx_ftns_wallets_user_type', 'ftns_wallets', ['user_id', 'wallet_type'])
    op.create_index(op.f('ix_ftns_wallets_user_id'), 'ftns_wallets', ['user_id'])
    op.create_index(op.f('ix_ftns_wallets_blockchain_address'), 'ftns_wallets', ['blockchain_address'])
    
    # Create FTNS transactions table
    op.create_table(
        'ftns_transactions',
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('transaction_hash', sa.String(length=255), nullable=True),
        sa.Column('from_wallet_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('to_wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('transaction_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('block_number', sa.Integer(), nullable=True),
        sa.Column('block_hash', sa.String(length=255), nullable=True),
        sa.Column('gas_fee', sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('confirmation_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('context_units', sa.Integer(), nullable=True),
        sa.Column('reference_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('confirmed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('signature', sa.Text(), nullable=True),
        sa.Column('nonce', sa.Integer(), nullable=True),
        sa.Column('fee_paid', sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.CheckConstraint('amount > 0', name='positive_amount'),
        sa.CheckConstraint('confirmation_count >= 0', name='positive_confirmations'),
        sa.ForeignKeyConstraint(['from_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.ForeignKeyConstraint(['to_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.PrimaryKeyConstraint('transaction_id'),
        sa.UniqueConstraint('transaction_hash')
    )
    
    # Create indexes for transactions
    op.create_index('idx_ftns_transactions_type_status', 'ftns_transactions', ['transaction_type', 'status'])
    op.create_index('idx_ftns_transactions_created', 'ftns_transactions', ['created_at'])
    op.create_index('idx_ftns_transactions_amount', 'ftns_transactions', ['amount'])
    op.create_index(op.f('ix_ftns_transactions_transaction_hash'), 'ftns_transactions', ['transaction_hash'])
    op.create_index(op.f('ix_ftns_transactions_transaction_type'), 'ftns_transactions', ['transaction_type'])
    op.create_index(op.f('ix_ftns_transactions_status'), 'ftns_transactions', ['status'])
    op.create_index(op.f('ix_ftns_transactions_block_number'), 'ftns_transactions', ['block_number'])
    op.create_index(op.f('ix_ftns_transactions_block_hash'), 'ftns_transactions', ['block_hash'])
    op.create_index(op.f('ix_ftns_transactions_reference_id'), 'ftns_transactions', ['reference_id'])
    
    # Create FTNS provenance records table
    op.create_table(
        'ftns_provenance_records',
        sa.Column('record_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_cid', sa.String(length=255), nullable=False),
        sa.Column('content_type', sa.String(length=50), nullable=False),
        sa.Column('creator_wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('original_creator_id', sa.String(length=255), nullable=False),
        sa.Column('access_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('download_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('citation_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('computational_usage_hours', sa.DECIMAL(precision=10, scale=2), nullable=False, server_default='0.0'),
        sa.Column('total_royalties_earned', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('total_royalties_paid', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('royalty_rate', sa.DECIMAL(precision=5, scale=4), nullable=False, server_default='0.05'),
        sa.Column('impact_score', sa.DECIMAL(precision=10, scale=4), nullable=False, server_default='0.0'),
        sa.Column('quality_score', sa.DECIMAL(precision=3, scale=2), nullable=False, server_default='1.0'),
        sa.Column('academic_citations', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('industry_applications', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_royalty_calculation', sa.DateTime(timezone=True), nullable=True),
        sa.Column('content_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('licensing_terms', sa.Text(), nullable=True),
        sa.Column('geographical_restrictions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint('access_count >= 0', name='positive_access_count'),
        sa.CheckConstraint('royalty_rate >= 0 AND royalty_rate <= 1', name='valid_royalty_rate'),
        sa.CheckConstraint('quality_score >= 0 AND quality_score <= 5', name='valid_quality_score'),
        sa.ForeignKeyConstraint(['creator_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.PrimaryKeyConstraint('record_id')
    )
    
    # Create indexes for provenance records
    op.create_index('idx_provenance_content_creator', 'ftns_provenance_records', ['content_cid', 'creator_wallet_id'])
    op.create_index('idx_provenance_impact', 'ftns_provenance_records', ['impact_score'])
    op.create_index(op.f('ix_ftns_provenance_records_content_cid'), 'ftns_provenance_records', ['content_cid'])
    op.create_index(op.f('ix_ftns_provenance_records_original_creator_id'), 'ftns_provenance_records', ['original_creator_id'])
    
    # Create dividend distributions table
    op.create_table(
        'ftns_dividend_distributions',
        sa.Column('distribution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('quarter', sa.String(length=20), nullable=False),
        sa.Column('total_pool', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('distribution_method', sa.String(length=50), nullable=False, server_default='proportional'),
        sa.Column('minimum_holding_period_days', sa.Integer(), nullable=False, server_default='30'),
        sa.Column('minimum_balance', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='1.0'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('eligible_wallets_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_distributed', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('calculation_started', sa.DateTime(timezone=True), nullable=True),
        sa.Column('distribution_started', sa.DateTime(timezone=True), nullable=True),
        sa.Column('distribution_completed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('bonus_multipliers', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('distribution_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint('total_pool > 0', name='positive_pool'),
        sa.CheckConstraint('minimum_holding_period_days >= 0', name='positive_holding_period'),
        sa.CheckConstraint('total_distributed >= 0', name='positive_distributed'),
        sa.PrimaryKeyConstraint('distribution_id'),
        sa.UniqueConstraint('quarter')
    )
    
    # Create indexes for dividend distributions
    op.create_index(op.f('ix_ftns_dividend_distributions_quarter'), 'ftns_dividend_distributions', ['quarter'])
    op.create_index(op.f('ix_ftns_dividend_distributions_status'), 'ftns_dividend_distributions', ['status'])
    
    # Create dividend payments table
    op.create_table(
        'ftns_dividend_payments',
        sa.Column('payment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('distribution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('base_amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('bonus_amount', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('total_amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('wallet_balance', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('holding_period_days', sa.Integer(), nullable=False),
        sa.Column('bonus_multiplier', sa.DECIMAL(precision=5, scale=4), nullable=False, server_default='1.0'),
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('base_amount >= 0', name='positive_base_amount'),
        sa.CheckConstraint('total_amount >= 0', name='positive_total_amount'),
        sa.CheckConstraint('holding_period_days >= 0', name='positive_holding_period'),
        sa.ForeignKeyConstraint(['distribution_id'], ['ftns_dividend_distributions.distribution_id'], ),
        sa.ForeignKeyConstraint(['transaction_id'], ['ftns_transactions.transaction_id'], ),
        sa.ForeignKeyConstraint(['wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.PrimaryKeyConstraint('payment_id'),
        sa.UniqueConstraint('distribution_id', 'wallet_id', name='unique_distribution_wallet')
    )
    
    # Create indexes for dividend payments
    op.create_index('idx_dividend_payments_distribution', 'ftns_dividend_payments', ['distribution_id'])
    
    # Create royalty payments table
    op.create_table(
        'ftns_royalty_payments',
        sa.Column('payment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provenance_record_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('total_usage', sa.DECIMAL(precision=15, scale=6), nullable=False),
        sa.Column('usage_type', sa.String(length=50), nullable=False),
        sa.Column('unique_users', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('royalty_rate', sa.DECIMAL(precision=5, scale=4), nullable=False),
        sa.Column('base_amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('impact_multiplier', sa.DECIMAL(precision=5, scale=4), nullable=False, server_default='1.0'),
        sa.Column('quality_multiplier', sa.DECIMAL(precision=5, scale=4), nullable=False, server_default='1.0'),
        sa.Column('bonus_amount', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('total_amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('calculation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint('total_usage >= 0', name='positive_usage'),
        sa.CheckConstraint('royalty_rate >= 0 AND royalty_rate <= 1', name='valid_royalty_rate'),
        sa.CheckConstraint('total_amount >= 0', name='positive_total_amount'),
        sa.ForeignKeyConstraint(['provenance_record_id'], ['ftns_provenance_records.record_id'], ),
        sa.ForeignKeyConstraint(['transaction_id'], ['ftns_transactions.transaction_id'], ),
        sa.PrimaryKeyConstraint('payment_id')
    )
    
    # Create indexes for royalty payments
    op.create_index('idx_royalty_payments_period', 'ftns_royalty_payments', ['period_start', 'period_end'])
    op.create_index('idx_royalty_payments_status', 'ftns_royalty_payments', ['status'])
    
    # Create marketplace listings table
    op.create_table(
        'ftns_marketplace_listings',
        sa.Column('listing_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=False),
        sa.Column('owner_wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('pricing_model', sa.String(length=50), nullable=False),
        sa.Column('base_price', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('currency', sa.String(length=10), nullable=False, server_default='FTNS'),
        sa.Column('availability_status', sa.String(length=50), nullable=False, server_default='available'),
        sa.Column('maximum_concurrent_users', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('minimum_rental_duration', sa.Integer(), nullable=True),
        sa.Column('maximum_rental_duration', sa.Integer(), nullable=True),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('resource_requirements', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('supported_features', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('geographical_restrictions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('total_revenue', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('total_rentals', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('average_rating', sa.DECIMAL(precision=3, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_rented', sa.DateTime(timezone=True), nullable=True),
        sa.Column('terms_of_service', sa.Text(), nullable=True),
        sa.Column('listing_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint('base_price > 0', name='positive_base_price'),
        sa.CheckConstraint('maximum_concurrent_users > 0', name='positive_concurrent_users'),
        sa.CheckConstraint('average_rating IS NULL OR (average_rating >= 0 AND average_rating <= 5)', name='valid_rating'),
        sa.ForeignKeyConstraint(['owner_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.PrimaryKeyConstraint('listing_id')
    )
    
    # Create indexes for marketplace listings
    op.create_index('idx_marketplace_listings_price', 'ftns_marketplace_listings', ['base_price'])
    op.create_index('idx_marketplace_listings_status', 'ftns_marketplace_listings', ['availability_status'])
    op.create_index(op.f('ix_ftns_marketplace_listings_model_id'), 'ftns_marketplace_listings', ['model_id'])
    
    # Create marketplace transactions table
    op.create_table(
        'ftns_marketplace_transactions',
        sa.Column('marketplace_transaction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('listing_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('buyer_wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('seller_wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('transaction_type', sa.String(length=50), nullable=False),
        sa.Column('amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('platform_fee', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('escrow_amount', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('rental_duration_hours', sa.Integer(), nullable=True),
        sa.Column('rental_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rental_ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('payment_transaction_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('usage_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('completion_rating', sa.DECIMAL(precision=3, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('amount > 0', name='positive_amount'),
        sa.CheckConstraint('platform_fee >= 0', name='positive_platform_fee'),
        sa.CheckConstraint('completion_rating IS NULL OR (completion_rating >= 0 AND completion_rating <= 5)', name='valid_completion_rating'),
        sa.ForeignKeyConstraint(['buyer_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.ForeignKeyConstraint(['listing_id'], ['ftns_marketplace_listings.listing_id'], ),
        sa.ForeignKeyConstraint(['payment_transaction_id'], ['ftns_transactions.transaction_id'], ),
        sa.ForeignKeyConstraint(['seller_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.PrimaryKeyConstraint('marketplace_transaction_id')
    )
    
    # Create indexes for marketplace transactions
    op.create_index('idx_marketplace_transactions_listing', 'ftns_marketplace_transactions', ['listing_id'])
    op.create_index('idx_marketplace_transactions_status', 'ftns_marketplace_transactions', ['status'])
    
    # Create price history table
    op.create_table(
        'ftns_price_history',
        sa.Column('price_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('price_usd', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('price_btc', sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('price_eth', sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('volume_24h', sa.DECIMAL(precision=20, scale=8), nullable=False, server_default='0.0'),
        sa.Column('market_cap', sa.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('circulating_supply', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('exchange_name', sa.String(length=100), nullable=True),
        sa.Column('trading_pair', sa.String(length=20), nullable=False, server_default='FTNS/USD'),
        sa.Column('volatility_24h', sa.DECIMAL(precision=10, scale=6), nullable=True),
        sa.Column('price_change_24h', sa.DECIMAL(precision=10, scale=6), nullable=True),
        sa.CheckConstraint('price_usd > 0', name='positive_price_usd'),
        sa.CheckConstraint('volume_24h >= 0', name='positive_volume'),
        sa.CheckConstraint('circulating_supply > 0', name='positive_supply'),
        sa.PrimaryKeyConstraint('price_id')
    )
    
    # Create indexes for price history
    op.create_index('idx_price_history_timestamp', 'ftns_price_history', ['timestamp'])
    op.create_index('idx_price_history_exchange_pair', 'ftns_price_history', ['exchange_name', 'trading_pair'])
    
    # Create governance votes table
    op.create_table(
        'ftns_governance_votes',
        sa.Column('vote_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('proposal_id', sa.String(length=255), nullable=False),
        sa.Column('voter_wallet_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vote_choice', sa.Boolean(), nullable=False),
        sa.Column('voting_power', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('staked_amount', sa.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('vote_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('voted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('stake_locked_until', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('voting_power > 0', name='positive_voting_power'),
        sa.CheckConstraint('staked_amount > 0', name='positive_staked_amount'),
        sa.ForeignKeyConstraint(['voter_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.PrimaryKeyConstraint('vote_id'),
        sa.UniqueConstraint('proposal_id', 'voter_wallet_id', name='unique_proposal_vote')
    )
    
    # Create indexes for governance votes
    op.create_index('idx_governance_votes_proposal', 'ftns_governance_votes', ['proposal_id'])
    op.create_index(op.f('ix_ftns_governance_votes_proposal_id'), 'ftns_governance_votes', ['proposal_id'])
    
    # Create audit logs table
    op.create_table(
        'ftns_audit_logs',
        sa.Column('log_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('event_category', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False, server_default='info'),
        sa.Column('actor_wallet_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('actor_ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('event_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('before_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('after_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('related_transaction_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('related_entity_type', sa.String(length=100), nullable=True),
        sa.Column('related_entity_id', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('signature', sa.Text(), nullable=True),
        sa.Column('checksum', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['actor_wallet_id'], ['ftns_wallets.wallet_id'], ),
        sa.ForeignKeyConstraint(['related_transaction_id'], ['ftns_transactions.transaction_id'], ),
        sa.PrimaryKeyConstraint('log_id')
    )
    
    # Create indexes for audit logs
    op.create_index('idx_audit_logs_event_type', 'ftns_audit_logs', ['event_type'])
    op.create_index('idx_audit_logs_category_severity', 'ftns_audit_logs', ['event_category', 'severity'])
    op.create_index('idx_audit_logs_timestamp', 'ftns_audit_logs', ['timestamp'])
    op.create_index('idx_audit_logs_actor', 'ftns_audit_logs', ['actor_wallet_id'])
    
    # Create performance indexes for common queries
    op.execute(text("""
        -- Index for transaction history queries by user
        CREATE INDEX CONCURRENTLY idx_ftns_transactions_user_created 
        ON ftns_transactions(to_wallet_id, created_at DESC);
        
        -- Index for marketplace search queries
        CREATE INDEX CONCURRENTLY idx_marketplace_listings_search 
        ON ftns_marketplace_listings(availability_status, base_price, average_rating);
        
        -- Index for dividend eligibility checks
        CREATE INDEX CONCURRENTLY idx_ftns_wallets_balance_created 
        ON ftns_wallets(balance, created_at) WHERE is_active = true;
        
        -- Index for royalty calculation queries
        CREATE INDEX CONCURRENTLY idx_provenance_records_royalty_calc 
        ON ftns_provenance_records(last_royalty_calculation, total_royalties_earned);
        
        -- Index for audit log queries by time and category
        CREATE INDEX CONCURRENTLY idx_audit_logs_time_category 
        ON ftns_audit_logs(timestamp DESC, event_category, severity);
    """))
    
    print("✅ FTNS Token System tables created successfully")
    print("📊 Created 11 main tables with comprehensive indexing")
    print("🔐 Added security constraints and data validation")
    print("💰 Ready for production token economy implementation")


def downgrade() -> None:
    """Remove FTNS token system tables"""
    
    # Drop indexes first
    op.drop_index('idx_audit_logs_time_category', table_name='ftns_audit_logs')
    op.drop_index('idx_provenance_records_royalty_calc', table_name='ftns_provenance_records')
    op.drop_index('idx_ftns_wallets_balance_created', table_name='ftns_wallets')
    op.drop_index('idx_marketplace_listings_search', table_name='ftns_marketplace_listings')
    op.drop_index('idx_ftns_transactions_user_created', table_name='ftns_transactions')
    
    # Drop tables in reverse dependency order
    op.drop_table('ftns_audit_logs')
    op.drop_table('ftns_governance_votes')
    op.drop_table('ftns_price_history')
    op.drop_table('ftns_marketplace_transactions')
    op.drop_table('ftns_marketplace_listings')
    op.drop_table('ftns_royalty_payments')
    op.drop_table('ftns_dividend_payments')
    op.drop_table('ftns_dividend_distributions')
    op.drop_table('ftns_provenance_records')
    op.drop_table('ftns_transactions')
    op.drop_table('ftns_wallets')
    
    print("❌ FTNS Token System tables removed")