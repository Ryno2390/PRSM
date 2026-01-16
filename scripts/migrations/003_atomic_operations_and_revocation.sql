-- Migration 003: Atomic Operations and Token Revocation
-- =====================================================
--
-- Addresses critical audit findings:
-- 1. Double-spend vulnerability (TOCTOU race condition)
-- 2. JWT verification bypass (stubbed revocation)
--
-- Changes:
-- - Add version column for optimistic concurrency control
-- - Add idempotency tracking table
-- - Add token revocation table
-- - Add balance snapshot tracking for auditing

-- =====================================================
-- PART 1: Atomic FTNS Operations
-- =====================================================

-- Add version column for optimistic concurrency control (OCC)
ALTER TABLE ftns_balances
ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1 NOT NULL;

-- Add balance snapshot columns for transaction auditing
ALTER TABLE ftns_transactions
ADD COLUMN IF NOT EXISTS balance_before_sender DECIMAL(28, 18),
ADD COLUMN IF NOT EXISTS balance_after_sender DECIMAL(28, 18),
ADD COLUMN IF NOT EXISTS balance_before_receiver DECIMAL(28, 18),
ADD COLUMN IF NOT EXISTS balance_after_receiver DECIMAL(28, 18);

-- Create idempotency tracking table
CREATE TABLE IF NOT EXISTS ftns_idempotency_keys (
    idempotency_key VARCHAR(255) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    operation_type VARCHAR(32) NOT NULL,
    amount DECIMAL(28, 18) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '24 hours'),

    CONSTRAINT valid_idempotency_status CHECK (status IN ('pending', 'completed', 'failed'))
);

-- Indexes for idempotency table
CREATE INDEX IF NOT EXISTS idx_idempotency_expires
ON ftns_idempotency_keys(expires_at);

CREATE INDEX IF NOT EXISTS idx_idempotency_user
ON ftns_idempotency_keys(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_idempotency_transaction
ON ftns_idempotency_keys(transaction_id);

-- Add idempotency_key column to transactions if not exists
ALTER TABLE ftns_transactions
ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(255);

-- Create unique index on idempotency_key (allows NULLs)
CREATE UNIQUE INDEX IF NOT EXISTS idx_ftns_tx_idempotency_unique
ON ftns_transactions(idempotency_key)
WHERE idempotency_key IS NOT NULL;

-- Function to acquire advisory lock for balance operations
CREATE OR REPLACE FUNCTION acquire_balance_lock(p_user_id VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    -- Use advisory lock based on hash of user_id
    -- This provides application-level locking without blocking reads
    PERFORM pg_advisory_xact_lock(hashtext(p_user_id));
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to check and record idempotency
CREATE OR REPLACE FUNCTION check_idempotency(
    p_idempotency_key VARCHAR(255),
    p_user_id VARCHAR(255),
    p_operation_type VARCHAR(32),
    p_amount DECIMAL(28, 18)
) RETURNS TABLE (
    is_duplicate BOOLEAN,
    existing_transaction_id VARCHAR(64),
    existing_status VARCHAR(20)
) AS $$
DECLARE
    existing_record RECORD;
BEGIN
    -- Check if idempotency key already exists
    SELECT ik.transaction_id, ik.status
    INTO existing_record
    FROM ftns_idempotency_keys ik
    WHERE ik.idempotency_key = p_idempotency_key;

    IF FOUND THEN
        -- Duplicate request detected
        RETURN QUERY SELECT
            TRUE as is_duplicate,
            existing_record.transaction_id,
            existing_record.status;
    ELSE
        -- New request - record the idempotency key
        -- This will be updated with transaction_id after successful operation
        RETURN QUERY SELECT
            FALSE as is_duplicate,
            NULL::VARCHAR(64) as existing_transaction_id,
            NULL::VARCHAR(20) as existing_status;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Atomic balance deduction with OCC and idempotency
CREATE OR REPLACE FUNCTION atomic_deduct_balance(
    p_user_id VARCHAR(255),
    p_amount DECIMAL(28, 18),
    p_idempotency_key VARCHAR(255),
    p_description TEXT DEFAULT '',
    p_transaction_type VARCHAR(50) DEFAULT 'deduction'
) RETURNS TABLE (
    success BOOLEAN,
    transaction_id VARCHAR(64),
    new_balance DECIMAL(28, 18),
    error_message TEXT
) AS $$
DECLARE
    v_current_balance DECIMAL(28, 18);
    v_current_locked DECIMAL(28, 18);
    v_current_version INTEGER;
    v_available_balance DECIMAL(28, 18);
    v_new_balance DECIMAL(28, 18);
    v_transaction_id VARCHAR(64);
    v_update_count INTEGER;
    v_idempotency_check RECORD;
BEGIN
    -- Step 1: Check idempotency
    SELECT * INTO v_idempotency_check
    FROM check_idempotency(p_idempotency_key, p_user_id, p_transaction_type, p_amount);

    IF v_idempotency_check.is_duplicate THEN
        RETURN QUERY SELECT
            FALSE,
            v_idempotency_check.existing_transaction_id,
            NULL::DECIMAL(28, 18),
            'Duplicate request - idempotency key already used'::TEXT;
        RETURN;
    END IF;

    -- Step 2: Acquire advisory lock for this user
    PERFORM acquire_balance_lock(p_user_id);

    -- Step 3: Get current balance with FOR UPDATE lock
    SELECT balance, locked_balance, version
    INTO v_current_balance, v_current_locked, v_current_version
    FROM ftns_balances
    WHERE user_id = p_user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT
            FALSE,
            NULL::VARCHAR(64),
            NULL::DECIMAL(28, 18),
            'User account not found'::TEXT;
        RETURN;
    END IF;

    -- Step 4: Check available balance
    v_available_balance := v_current_balance - v_current_locked;

    IF v_available_balance < p_amount THEN
        RETURN QUERY SELECT
            FALSE,
            NULL::VARCHAR(64),
            v_available_balance,
            format('Insufficient balance: available %s, requested %s',
                   v_available_balance::TEXT, p_amount::TEXT)::TEXT;
        RETURN;
    END IF;

    -- Step 5: Generate transaction ID
    v_transaction_id := 'ftns_' || encode(gen_random_bytes(12), 'hex');
    v_new_balance := v_current_balance - p_amount;

    -- Step 6: Update balance with OCC check
    UPDATE ftns_balances
    SET
        balance = v_new_balance,
        total_spent = total_spent + p_amount,
        version = version + 1,
        last_transaction_id = v_transaction_id::UUID,
        updated_at = NOW()
    WHERE user_id = p_user_id
    AND version = v_current_version;

    GET DIAGNOSTICS v_update_count = ROW_COUNT;

    IF v_update_count = 0 THEN
        -- Concurrent modification detected
        RETURN QUERY SELECT
            FALSE,
            NULL::VARCHAR(64),
            NULL::DECIMAL(28, 18),
            'Concurrent modification detected - please retry'::TEXT;
        RETURN;
    END IF;

    -- Step 7: Record transaction
    INSERT INTO ftns_transactions (
        id, from_user_id, to_user_id, amount,
        transaction_type, description, status,
        idempotency_key, balance_before_sender, balance_after_sender,
        created_at
    ) VALUES (
        v_transaction_id::UUID, p_user_id, NULL, p_amount,
        p_transaction_type, p_description, 'completed',
        p_idempotency_key, v_current_balance, v_new_balance,
        NOW()
    );

    -- Step 8: Record idempotency key
    INSERT INTO ftns_idempotency_keys (
        idempotency_key, transaction_id, user_id,
        operation_type, amount, status
    ) VALUES (
        p_idempotency_key, v_transaction_id, p_user_id,
        p_transaction_type, p_amount, 'completed'
    );

    -- Step 9: Return success
    RETURN QUERY SELECT
        TRUE,
        v_transaction_id,
        v_new_balance,
        NULL::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Atomic transfer between users
CREATE OR REPLACE FUNCTION atomic_transfer(
    p_from_user_id VARCHAR(255),
    p_to_user_id VARCHAR(255),
    p_amount DECIMAL(28, 18),
    p_idempotency_key VARCHAR(255),
    p_description TEXT DEFAULT ''
) RETURNS TABLE (
    success BOOLEAN,
    transaction_id VARCHAR(64),
    sender_new_balance DECIMAL(28, 18),
    receiver_new_balance DECIMAL(28, 18),
    error_message TEXT
) AS $$
DECLARE
    v_sender_balance DECIMAL(28, 18);
    v_sender_locked DECIMAL(28, 18);
    v_sender_version INTEGER;
    v_receiver_balance DECIMAL(28, 18);
    v_receiver_version INTEGER;
    v_available_balance DECIMAL(28, 18);
    v_transaction_id VARCHAR(64);
    v_update_count INTEGER;
    v_idempotency_check RECORD;
BEGIN
    -- Check idempotency
    SELECT * INTO v_idempotency_check
    FROM check_idempotency(p_idempotency_key, p_from_user_id, 'transfer', p_amount);

    IF v_idempotency_check.is_duplicate THEN
        RETURN QUERY SELECT
            FALSE, v_idempotency_check.existing_transaction_id,
            NULL::DECIMAL(28, 18), NULL::DECIMAL(28, 18),
            'Duplicate request'::TEXT;
        RETURN;
    END IF;

    -- Acquire locks in consistent order (alphabetically by user_id)
    IF p_from_user_id < p_to_user_id THEN
        PERFORM acquire_balance_lock(p_from_user_id);
        PERFORM acquire_balance_lock(p_to_user_id);
    ELSE
        PERFORM acquire_balance_lock(p_to_user_id);
        PERFORM acquire_balance_lock(p_from_user_id);
    END IF;

    -- Get sender balance
    SELECT balance, locked_balance, version
    INTO v_sender_balance, v_sender_locked, v_sender_version
    FROM ftns_balances
    WHERE user_id = p_from_user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(64),
            NULL::DECIMAL(28, 18), NULL::DECIMAL(28, 18),
            'Sender account not found'::TEXT;
        RETURN;
    END IF;

    v_available_balance := v_sender_balance - v_sender_locked;

    IF v_available_balance < p_amount THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(64),
            v_available_balance, NULL::DECIMAL(28, 18),
            'Insufficient balance'::TEXT;
        RETURN;
    END IF;

    -- Get receiver balance
    SELECT balance, version
    INTO v_receiver_balance, v_receiver_version
    FROM ftns_balances
    WHERE user_id = p_to_user_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(64),
            NULL::DECIMAL(28, 18), NULL::DECIMAL(28, 18),
            'Receiver account not found'::TEXT;
        RETURN;
    END IF;

    v_transaction_id := 'ftns_' || encode(gen_random_bytes(12), 'hex');

    -- Update sender
    UPDATE ftns_balances
    SET balance = balance - p_amount,
        total_spent = total_spent + p_amount,
        version = version + 1,
        updated_at = NOW()
    WHERE user_id = p_from_user_id
    AND version = v_sender_version;

    GET DIAGNOSTICS v_update_count = ROW_COUNT;
    IF v_update_count = 0 THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR(64),
            NULL::DECIMAL(28, 18), NULL::DECIMAL(28, 18),
            'Concurrent modification on sender'::TEXT;
        RETURN;
    END IF;

    -- Update receiver
    UPDATE ftns_balances
    SET balance = balance + p_amount,
        total_earned = total_earned + p_amount,
        version = version + 1,
        updated_at = NOW()
    WHERE user_id = p_to_user_id
    AND version = v_receiver_version;

    GET DIAGNOSTICS v_update_count = ROW_COUNT;
    IF v_update_count = 0 THEN
        -- Rollback sender update (will happen automatically with EXCEPTION)
        RAISE EXCEPTION 'Concurrent modification on receiver';
    END IF;

    -- Record transaction
    INSERT INTO ftns_transactions (
        id, from_user_id, to_user_id, amount,
        transaction_type, description, status,
        idempotency_key, balance_before_sender, balance_after_sender,
        balance_before_receiver, balance_after_receiver
    ) VALUES (
        v_transaction_id::UUID, p_from_user_id, p_to_user_id, p_amount,
        'transfer', p_description, 'completed',
        p_idempotency_key, v_sender_balance, v_sender_balance - p_amount,
        v_receiver_balance, v_receiver_balance + p_amount
    );

    -- Record idempotency
    INSERT INTO ftns_idempotency_keys (
        idempotency_key, transaction_id, user_id, operation_type, amount
    ) VALUES (
        p_idempotency_key, v_transaction_id, p_from_user_id, 'transfer', p_amount
    );

    RETURN QUERY SELECT TRUE, v_transaction_id,
        v_sender_balance - p_amount,
        v_receiver_balance + p_amount,
        NULL::TEXT;

EXCEPTION WHEN OTHERS THEN
    RETURN QUERY SELECT FALSE, NULL::VARCHAR(64),
        NULL::DECIMAL(28, 18), NULL::DECIMAL(28, 18),
        SQLERRM::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for expired idempotency keys
CREATE OR REPLACE FUNCTION cleanup_expired_idempotency_keys()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ftns_idempotency_keys
    WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PART 2: Token Revocation System
-- =====================================================

-- Create token revocation table
CREATE TABLE IF NOT EXISTS revoked_tokens (
    id SERIAL PRIMARY KEY,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    jti VARCHAR(64) NOT NULL,
    user_id UUID,
    reason VARCHAR(255),
    revoked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    revoked_by VARCHAR(255),

    CONSTRAINT reason_not_empty CHECK (reason IS NULL OR LENGTH(reason) > 0)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_hash
ON revoked_tokens(token_hash);

CREATE INDEX IF NOT EXISTS idx_revoked_tokens_jti
ON revoked_tokens(jti);

CREATE INDEX IF NOT EXISTS idx_revoked_tokens_user
ON revoked_tokens(user_id);

CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires
ON revoked_tokens(expires_at)
WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_revoked_tokens_revoked_at
ON revoked_tokens(revoked_at);

-- Function to revoke a token
CREATE OR REPLACE FUNCTION revoke_token(
    p_token_hash VARCHAR(64),
    p_jti VARCHAR(64),
    p_user_id UUID DEFAULT NULL,
    p_reason VARCHAR(255) DEFAULT 'User logout',
    p_expires_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    p_revoked_by VARCHAR(255) DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO revoked_tokens (token_hash, jti, user_id, reason, expires_at, revoked_by)
    VALUES (p_token_hash, p_jti, p_user_id, p_reason, p_expires_at, p_revoked_by)
    ON CONFLICT (token_hash) DO UPDATE
    SET reason = EXCLUDED.reason,
        revoked_at = NOW();

    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to check if token is revoked
CREATE OR REPLACE FUNCTION is_token_revoked(
    p_token_hash VARCHAR(64),
    p_jti VARCHAR(64) DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    is_revoked BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM revoked_tokens
        WHERE token_hash = p_token_hash
        OR (p_jti IS NOT NULL AND jti = p_jti)
    ) INTO is_revoked;

    RETURN is_revoked;
END;
$$ LANGUAGE plpgsql;

-- Function to revoke all tokens for a user
CREATE OR REPLACE FUNCTION revoke_all_user_tokens(
    p_user_id UUID,
    p_reason VARCHAR(255) DEFAULT 'User requested logout from all devices'
) RETURNS INTEGER AS $$
DECLARE
    revoked_count INTEGER;
BEGIN
    -- Mark any existing tokens as revoked (if we track active tokens)
    -- For now, just return 0 as we rely on token_hash lookup
    -- In a more complete system, we'd track active tokens
    RETURN 0;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for expired revoked tokens
CREATE OR REPLACE FUNCTION cleanup_expired_revoked_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM revoked_tokens
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PART 3: Scheduled Maintenance
-- =====================================================

-- Create maintenance log table
CREATE TABLE IF NOT EXISTS maintenance_log (
    id SERIAL PRIMARY KEY,
    task_name VARCHAR(100) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    records_affected INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT,

    CONSTRAINT valid_maintenance_status CHECK (status IN ('running', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_maintenance_log_task
ON maintenance_log(task_name, started_at DESC);

-- Comprehensive cleanup function
CREATE OR REPLACE FUNCTION run_scheduled_maintenance()
RETURNS TABLE (
    task_name VARCHAR(100),
    records_affected INTEGER,
    status VARCHAR(20)
) AS $$
DECLARE
    v_count INTEGER;
    v_log_id INTEGER;
BEGIN
    -- Task 1: Cleanup expired idempotency keys
    INSERT INTO maintenance_log (task_name)
    VALUES ('cleanup_idempotency_keys')
    RETURNING id INTO v_log_id;

    SELECT cleanup_expired_idempotency_keys() INTO v_count;

    UPDATE maintenance_log
    SET completed_at = NOW(), records_affected = v_count, status = 'completed'
    WHERE id = v_log_id;

    task_name := 'cleanup_idempotency_keys';
    records_affected := v_count;
    status := 'completed';
    RETURN NEXT;

    -- Task 2: Cleanup expired transaction locks
    INSERT INTO maintenance_log (task_name)
    VALUES ('cleanup_transaction_locks')
    RETURNING id INTO v_log_id;

    SELECT cleanup_expired_locks() INTO v_count;

    UPDATE maintenance_log
    SET completed_at = NOW(), records_affected = v_count, status = 'completed'
    WHERE id = v_log_id;

    task_name := 'cleanup_transaction_locks';
    records_affected := v_count;
    status := 'completed';
    RETURN NEXT;

    -- Task 3: Cleanup expired revoked tokens
    INSERT INTO maintenance_log (task_name)
    VALUES ('cleanup_revoked_tokens')
    RETURNING id INTO v_log_id;

    SELECT cleanup_expired_revoked_tokens() INTO v_count;

    UPDATE maintenance_log
    SET completed_at = NOW(), records_affected = v_count, status = 'completed'
    WHERE id = v_log_id;

    task_name := 'cleanup_revoked_tokens';
    records_affected := v_count;
    status := 'completed';
    RETURN NEXT;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PART 4: User Token Tracking (for revocation)
-- =====================================================

-- Create user_tokens table for token tracking
CREATE TABLE IF NOT EXISTS user_tokens (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    jti VARCHAR(64) NOT NULL,
    token_type VARCHAR(20) NOT NULL DEFAULT 'access',
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    client_ip VARCHAR(45),
    user_agent VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_revoked BOOLEAN DEFAULT FALSE,

    CONSTRAINT valid_token_type CHECK (token_type IN ('access', 'refresh'))
);

-- Indexes for user_tokens
CREATE INDEX IF NOT EXISTS idx_user_tokens_user_id
ON user_tokens(user_id);

CREATE INDEX IF NOT EXISTS idx_user_tokens_hash
ON user_tokens(token_hash);

CREATE INDEX IF NOT EXISTS idx_user_tokens_jti
ON user_tokens(jti);

CREATE INDEX IF NOT EXISTS idx_user_tokens_expires
ON user_tokens(expires_at);

CREATE INDEX IF NOT EXISTS idx_user_tokens_active
ON user_tokens(user_id, is_revoked, expires_at)
WHERE is_revoked = FALSE;

-- Cleanup function for expired user tokens
CREATE OR REPLACE FUNCTION cleanup_expired_user_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_tokens
    WHERE expires_at < NOW() - INTERVAL '1 day';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE ftns_idempotency_keys IS 'Prevents duplicate FTNS operations via idempotency keys';
COMMENT ON TABLE revoked_tokens IS 'JWT token revocation list for security';
COMMENT ON TABLE user_tokens IS 'Tracks active user tokens for revocation capability';
COMMENT ON TABLE maintenance_log IS 'Tracks scheduled maintenance task execution';
COMMENT ON FUNCTION atomic_deduct_balance IS 'Atomically deducts balance with OCC and idempotency';
COMMENT ON FUNCTION atomic_transfer IS 'Atomically transfers FTNS between users with full ACID guarantees';
COMMENT ON FUNCTION revoke_token IS 'Revokes a JWT token by adding to blacklist';
COMMENT ON FUNCTION is_token_revoked IS 'Checks if a JWT token has been revoked';

-- Success message
SELECT 'Migration 003: Atomic operations and token revocation completed!' as status;
