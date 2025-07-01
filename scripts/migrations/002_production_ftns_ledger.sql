-- Production FTNS Ledger Database Schema
-- =====================================
-- 
-- Creates production-grade database schema for FTNS token ledger
-- Addresses Gemini audit finding: "The core ledger for balances and 
-- transactions is an in-memory Python dictionary, which would be wiped 
-- on every server restart."

-- FTNS Balances table - Persistent user balances
CREATE TABLE IF NOT EXISTS ftns_balances (
    user_id VARCHAR(255) PRIMARY KEY,
    balance DECIMAL(28, 18) NOT NULL DEFAULT 0,
    locked_balance DECIMAL(28, 18) NOT NULL DEFAULT 0,
    total_earned DECIMAL(28, 18) NOT NULL DEFAULT 0,
    total_spent DECIMAL(28, 18) NOT NULL DEFAULT 0,
    account_type VARCHAR(20) NOT NULL DEFAULT 'user', -- 'user', 'system'
    last_transaction_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT balance_non_negative CHECK (balance >= 0),
    CONSTRAINT locked_balance_non_negative CHECK (locked_balance >= 0),
    CONSTRAINT locked_not_exceeding_balance CHECK (locked_balance <= balance),
    CONSTRAINT total_earned_non_negative CHECK (total_earned >= 0),
    CONSTRAINT total_spent_non_negative CHECK (total_spent >= 0)
);

-- FTNS Transactions table - Complete transaction history
CREATE TABLE IF NOT EXISTS ftns_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_user_id VARCHAR(255) NOT NULL,
    to_user_id VARCHAR(255),
    amount DECIMAL(28, 18) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL, -- 'transfer', 'mint', 'burn', 'reward', 'fee'
    description TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    reference_id VARCHAR(255), -- External reference (API call, marketplace transaction)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT amount_positive CHECK (amount > 0),
    CONSTRAINT valid_transaction_type CHECK (transaction_type IN ('transfer', 'mint', 'burn', 'reward', 'fee')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'completed', 'failed')),
    
    -- Foreign key relationships
    CONSTRAINT fk_from_user FOREIGN KEY (from_user_id) REFERENCES ftns_balances(user_id),
    CONSTRAINT fk_to_user FOREIGN KEY (to_user_id) REFERENCES ftns_balances(user_id)
);

-- Transaction locks table - Prevent race conditions
CREATE TABLE IF NOT EXISTS ftns_transaction_locks (
    lock_key VARCHAR(255) PRIMARY KEY,
    transaction_id UUID NOT NULL,
    locked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    CONSTRAINT expires_in_future CHECK (expires_at > locked_at)
);

-- Market orders table - For marketplace transactions
CREATE TABLE IF NOT EXISTS ftns_market_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    order_type VARCHAR(20) NOT NULL, -- 'buy', 'sell'
    asset_type VARCHAR(50) NOT NULL, -- 'ai_model', 'dataset', 'compute_time'
    asset_id VARCHAR(255) NOT NULL,
    amount_ftns DECIMAL(28, 18) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price_per_unit DECIMAL(28, 18) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'open', -- 'open', 'partial', 'filled', 'cancelled'
    filled_quantity DECIMAL(18, 8) NOT NULL DEFAULT 0,
    total_fees DECIMAL(28, 18) NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT amount_positive CHECK (amount_ftns > 0),
    CONSTRAINT quantity_positive CHECK (quantity > 0),
    CONSTRAINT price_positive CHECK (price_per_unit > 0),
    CONSTRAINT filled_not_exceeding_quantity CHECK (filled_quantity <= quantity),
    CONSTRAINT valid_order_type CHECK (order_type IN ('buy', 'sell')),
    CONSTRAINT valid_order_status CHECK (status IN ('open', 'partial', 'filled', 'cancelled')),
    
    -- Foreign key
    CONSTRAINT fk_order_user FOREIGN KEY (user_id) REFERENCES ftns_balances(user_id)
);

-- Staking records table - For FTNS staking rewards
CREATE TABLE IF NOT EXISTS ftns_staking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    staked_amount DECIMAL(28, 18) NOT NULL,
    stake_type VARCHAR(50) NOT NULL, -- 'governance', 'liquidity', 'validator'
    annual_rate DECIMAL(8, 6) NOT NULL, -- Annual percentage rate
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE,
    last_reward_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_rewards DECIMAL(28, 18) NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'ended', 'slashed'
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT staked_amount_positive CHECK (staked_amount > 0),
    CONSTRAINT annual_rate_valid CHECK (annual_rate >= 0 AND annual_rate <= 1),
    CONSTRAINT total_rewards_non_negative CHECK (total_rewards >= 0),
    CONSTRAINT valid_stake_status CHECK (status IN ('active', 'ended', 'slashed')),
    CONSTRAINT end_after_start CHECK (end_date IS NULL OR end_date > start_date),
    
    -- Foreign key
    CONSTRAINT fk_staking_user FOREIGN KEY (user_id) REFERENCES ftns_balances(user_id)
);

-- Governance proposals table - For FTNS governance voting
CREATE TABLE IF NOT EXISTS ftns_governance_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    proposer_id VARCHAR(255) NOT NULL,
    proposal_type VARCHAR(50) NOT NULL, -- 'parameter_change', 'feature_addition', 'funding'
    voting_starts TIMESTAMP WITH TIME ZONE NOT NULL,
    voting_ends TIMESTAMP WITH TIME ZONE NOT NULL,
    min_participation_rate DECIMAL(5, 4) NOT NULL DEFAULT 0.1, -- 10% minimum
    approval_threshold DECIMAL(5, 4) NOT NULL DEFAULT 0.5, -- 50% approval
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'active', 'passed', 'failed', 'executed'
    execution_data JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT voting_period_valid CHECK (voting_ends > voting_starts),
    CONSTRAINT participation_rate_valid CHECK (min_participation_rate >= 0 AND min_participation_rate <= 1),
    CONSTRAINT approval_threshold_valid CHECK (approval_threshold > 0 AND approval_threshold <= 1),
    CONSTRAINT valid_proposal_status CHECK (status IN ('pending', 'active', 'passed', 'failed', 'executed')),
    
    -- Foreign key
    CONSTRAINT fk_proposer FOREIGN KEY (proposer_id) REFERENCES ftns_balances(user_id)
);

-- Governance votes table - Individual votes on proposals
CREATE TABLE IF NOT EXISTS ftns_governance_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id UUID NOT NULL,
    voter_id VARCHAR(255) NOT NULL,
    vote VARCHAR(10) NOT NULL, -- 'yes', 'no', 'abstain'
    voting_power DECIMAL(28, 18) NOT NULL, -- FTNS amount used for voting
    vote_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT valid_vote CHECK (vote IN ('yes', 'no', 'abstain')),
    CONSTRAINT voting_power_positive CHECK (voting_power > 0),
    UNIQUE(proposal_id, voter_id), -- One vote per user per proposal
    
    -- Foreign keys
    CONSTRAINT fk_vote_proposal FOREIGN KEY (proposal_id) REFERENCES ftns_governance_proposals(id),
    CONSTRAINT fk_voter FOREIGN KEY (voter_id) REFERENCES ftns_balances(user_id)
);

-- Performance indexes for production workloads

-- Balances table indexes
CREATE INDEX IF NOT EXISTS idx_ftns_balances_account_type ON ftns_balances(account_type);
CREATE INDEX IF NOT EXISTS idx_ftns_balances_balance ON ftns_balances(balance) WHERE balance > 0;
CREATE INDEX IF NOT EXISTS idx_ftns_balances_updated ON ftns_balances(updated_at);

-- Transactions table indexes
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_from_user ON ftns_transactions(from_user_id);
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_to_user ON ftns_transactions(to_user_id);
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_type ON ftns_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_status ON ftns_transactions(status);
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_created ON ftns_transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_reference ON ftns_transactions(reference_id) WHERE reference_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_user_activity ON ftns_transactions(from_user_id, to_user_id, created_at);

-- Compound index for user transaction history
CREATE INDEX IF NOT EXISTS idx_ftns_transactions_user_history 
ON ftns_transactions(from_user_id, created_at DESC) 
WHERE status = 'completed';

CREATE INDEX IF NOT EXISTS idx_ftns_transactions_user_history_incoming 
ON ftns_transactions(to_user_id, created_at DESC) 
WHERE status = 'completed';

-- Market orders indexes
CREATE INDEX IF NOT EXISTS idx_ftns_market_orders_user ON ftns_market_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_ftns_market_orders_asset ON ftns_market_orders(asset_type, asset_id);
CREATE INDEX IF NOT EXISTS idx_ftns_market_orders_status ON ftns_market_orders(status);
CREATE INDEX IF NOT EXISTS idx_ftns_market_orders_type_status ON ftns_market_orders(order_type, status);

-- Staking indexes
CREATE INDEX IF NOT EXISTS idx_ftns_staking_user ON ftns_staking(user_id);
CREATE INDEX IF NOT EXISTS idx_ftns_staking_status ON ftns_staking(status);
CREATE INDEX IF NOT EXISTS idx_ftns_staking_type ON ftns_staking(stake_type);

-- Governance indexes
CREATE INDEX IF NOT EXISTS idx_ftns_governance_status ON ftns_governance_proposals(status);
CREATE INDEX IF NOT EXISTS idx_ftns_governance_voting_period ON ftns_governance_proposals(voting_starts, voting_ends);
CREATE INDEX IF NOT EXISTS idx_ftns_governance_votes_proposal ON ftns_governance_votes(proposal_id);

-- Functions for business logic and maintenance

-- Calculate user voting power based on staked FTNS
CREATE OR REPLACE FUNCTION calculate_voting_power(p_user_id VARCHAR(255))
RETURNS DECIMAL(28, 18) AS $$
DECLARE
    total_power DECIMAL(28, 18) := 0;
BEGIN
    -- Base voting power from staked tokens
    SELECT COALESCE(SUM(staked_amount), 0) INTO total_power
    FROM ftns_staking
    WHERE user_id = p_user_id AND status = 'active';
    
    -- Add delegation power if implemented
    -- (Future feature for delegated voting)
    
    RETURN total_power;
END;
$$ LANGUAGE plpgsql;

-- Update balance totals (reconciliation function)
CREATE OR REPLACE FUNCTION reconcile_user_balance(p_user_id VARCHAR(255))
RETURNS BOOLEAN AS $$
DECLARE
    calculated_earned DECIMAL(28, 18) := 0;
    calculated_spent DECIMAL(28, 18) := 0;
    calculated_balance DECIMAL(28, 18) := 0;
BEGIN
    -- Calculate total earned (incoming transactions)
    SELECT COALESCE(SUM(amount), 0) INTO calculated_earned
    FROM ftns_transactions
    WHERE to_user_id = p_user_id 
    AND status = 'completed'
    AND transaction_type IN ('mint', 'transfer', 'reward');
    
    -- Calculate total spent (outgoing transactions)
    SELECT COALESCE(SUM(amount), 0) INTO calculated_spent
    FROM ftns_transactions
    WHERE from_user_id = p_user_id 
    AND status = 'completed'
    AND transaction_type IN ('burn', 'transfer', 'fee');
    
    calculated_balance := calculated_earned - calculated_spent;
    
    -- Update balance record
    UPDATE ftns_balances
    SET 
        total_earned = calculated_earned,
        total_spent = calculated_spent,
        balance = calculated_balance,
        updated_at = NOW()
    WHERE user_id = p_user_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Get market depth for asset
CREATE OR REPLACE FUNCTION get_market_depth(
    p_asset_type VARCHAR(50),
    p_asset_id VARCHAR(255),
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    order_type VARCHAR(20),
    price_per_unit DECIMAL(28, 18),
    total_quantity DECIMAL(18, 8),
    order_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mo.order_type,
        mo.price_per_unit,
        SUM(mo.quantity - mo.filled_quantity) as total_quantity,
        COUNT(*)::INTEGER as order_count
    FROM ftns_market_orders mo
    WHERE mo.asset_type = p_asset_type
    AND mo.asset_id = p_asset_id
    AND mo.status IN ('open', 'partial')
    AND mo.quantity > mo.filled_quantity
    GROUP BY mo.order_type, mo.price_per_unit
    ORDER BY 
        CASE WHEN mo.order_type = 'buy' THEN mo.price_per_unit END DESC,
        CASE WHEN mo.order_type = 'sell' THEN mo.price_per_unit END ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Cleanup expired transaction locks
CREATE OR REPLACE FUNCTION cleanup_expired_locks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ftns_transaction_locks
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Views for reporting and analytics

-- User portfolio view
CREATE OR REPLACE VIEW user_portfolio AS
SELECT 
    fb.user_id,
    fb.balance as available_balance,
    fb.locked_balance,
    COALESCE(fs.staked_amount, 0) as staked_balance,
    (fb.balance + fb.locked_balance + COALESCE(fs.staked_amount, 0)) as total_balance,
    fb.total_earned,
    fb.total_spent,
    fb.updated_at as last_activity
FROM ftns_balances fb
LEFT JOIN (
    SELECT user_id, SUM(staked_amount) as staked_amount
    FROM ftns_staking
    WHERE status = 'active'
    GROUP BY user_id
) fs ON fb.user_id = fs.user_id
WHERE fb.account_type = 'user';

-- Transaction summary view
CREATE OR REPLACE VIEW transaction_summary_24h AS
SELECT 
    transaction_type,
    COUNT(*) as transaction_count,
    SUM(amount) as total_volume,
    AVG(amount) as average_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount
FROM ftns_transactions
WHERE created_at > NOW() - INTERVAL '24 hours'
AND status = 'completed'
GROUP BY transaction_type;

-- Market activity view
CREATE OR REPLACE VIEW market_activity_summary AS
SELECT 
    asset_type,
    order_type,
    COUNT(*) as order_count,
    SUM(amount_ftns) as total_value,
    SUM(quantity) as total_quantity,
    AVG(price_per_unit) as avg_price,
    MIN(price_per_unit) as min_price,
    MAX(price_per_unit) as max_price
FROM ftns_market_orders
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY asset_type, order_type;

-- Grant permissions for application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO prsm_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO prsm_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO prsm_app;

-- Insert system accounts
INSERT INTO ftns_balances (user_id, account_type, balance, total_earned)
VALUES 
    ('system_mint', 'system', 1000000000, 1000000000),      -- 1B FTNS for minting
    ('system_burn', 'system', 0, 0),                        -- Burn account
    ('system_rewards', 'system', 100000000, 100000000),     -- 100M FTNS for rewards
    ('system_fees', 'system', 0, 0)                         -- Fee collection account
ON CONFLICT (user_id) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE ftns_balances IS 'Persistent FTNS token balances replacing in-memory dictionary';
COMMENT ON TABLE ftns_transactions IS 'Complete FTNS transaction history with ACID guarantees';
COMMENT ON TABLE ftns_market_orders IS 'Marketplace orders for real value transfer';
COMMENT ON TABLE ftns_staking IS 'FTNS staking for governance and rewards';
COMMENT ON TABLE ftns_governance_proposals IS 'Community governance proposals';
COMMENT ON TABLE ftns_governance_votes IS 'Individual votes on governance proposals';

-- Success message
SELECT 'Production FTNS Ledger schema created successfully!' as status;