-- Production RBAC Database Schema Migration
-- ========================================
-- 
-- Creates production-grade database schema for RBAC system
-- Addresses Gemini audit finding about in-memory authorization

-- User permissions table for granular access control
CREATE TABLE IF NOT EXISTS user_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    granted_by UUID NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    revoked_by UUID,
    revoked_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_permissions_user_id ON user_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_resource ON user_permissions(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_active ON user_permissions(is_active, expires_at);
CREATE INDEX IF NOT EXISTS idx_user_permissions_action ON user_permissions(action);

-- Audit logs table for compliance and security monitoring
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL,
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for audit queries
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_success ON audit_logs(success);
CREATE INDEX IF NOT EXISTS idx_audit_logs_ip ON audit_logs(ip_address);

-- Rate limiting log table (fallback when Redis unavailable)
CREATE TABLE IF NOT EXISTS rate_limit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    ip_address INET,
    request_count INTEGER DEFAULT 1,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for rate limiting queries
CREATE INDEX IF NOT EXISTS idx_rate_limit_user_endpoint ON rate_limit_log(user_id, endpoint);
CREATE INDEX IF NOT EXISTS idx_rate_limit_timestamp ON rate_limit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_rate_limit_ip ON rate_limit_log(ip_address);

-- Blocked IPs table for security
CREATE TABLE IF NOT EXISTS blocked_ips (
    ip_address INET PRIMARY KEY,
    reason TEXT NOT NULL,
    blocked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    blocked_by UUID,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Security events table for threat detection
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL, -- failed_login, rate_limit_exceeded, suspicious_activity
    user_id UUID,
    ip_address INET,
    user_agent TEXT,
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    details JSONB NOT NULL,
    resolved BOOLEAN DEFAULT false,
    resolved_by UUID,
    resolved_at TIMESTAMP WITH TIME ZONE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for security monitoring
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_resolved ON security_events(resolved);
CREATE INDEX IF NOT EXISTS idx_security_events_ip ON security_events(ip_address);

-- User sessions table for session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for session management
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, expires_at);

-- Role assignments table (many-to-many users and roles)
CREATE TABLE IF NOT EXISTS user_role_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    role VARCHAR(50) NOT NULL,
    assigned_by UUID NOT NULL,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    
    UNIQUE(user_id, role)
);

-- Indexes for role assignments
CREATE INDEX IF NOT EXISTS idx_user_role_assignments_user_id ON user_role_assignments(user_id);
CREATE INDEX IF NOT EXISTS idx_user_role_assignments_role ON user_role_assignments(role);
CREATE INDEX IF NOT EXISTS idx_user_role_assignments_active ON user_role_assignments(is_active);

-- Functions for maintenance and cleanup

-- Cleanup expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    UPDATE user_sessions 
    SET is_active = false 
    WHERE expires_at < NOW() AND is_active = true;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup old rate limit logs (keep only last 7 days)
CREATE OR REPLACE FUNCTION cleanup_rate_limit_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM rate_limit_log 
    WHERE timestamp < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup old audit logs (keep only last 2 years for compliance)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs()
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Archive old logs to separate table before deletion (if needed)
    -- For now, just mark as archived
    UPDATE audit_logs 
    SET metadata = metadata || '{"archived": true}'::jsonb
    WHERE timestamp < NOW() - INTERVAL '2 years'
    AND NOT (metadata ? 'archived');
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Security alert function
CREATE OR REPLACE FUNCTION create_security_event(
    p_event_type VARCHAR(100),
    p_user_id UUID,
    p_ip_address INET,
    p_severity VARCHAR(20),
    p_details JSONB
)
RETURNS UUID AS $$
DECLARE
    event_id UUID;
BEGIN
    INSERT INTO security_events (event_type, user_id, ip_address, severity, details)
    VALUES (p_event_type, p_user_id, p_ip_address, p_severity, p_details)
    RETURNING id INTO event_id;
    
    -- Auto-block IPs for critical events (can be customized)
    IF p_severity = 'critical' AND p_ip_address IS NOT NULL THEN
        INSERT INTO blocked_ips (ip_address, reason, blocked_by)
        VALUES (p_ip_address, 'Auto-blocked for critical security event', NULL)
        ON CONFLICT (ip_address) DO NOTHING;
    END IF;
    
    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

-- User permission checking function (optimized for performance)
CREATE OR REPLACE FUNCTION check_user_permission(
    p_user_id UUID,
    p_resource_type VARCHAR(100),
    p_action VARCHAR(50),
    p_resource_id VARCHAR(255) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM user_permissions
        WHERE user_id = p_user_id
        AND resource_type = p_resource_type
        AND action = p_action
        AND (p_resource_id IS NULL OR resource_id IS NULL OR resource_id = p_resource_id)
        AND is_active = true
        AND (expires_at IS NULL OR expires_at > NOW())
    );
END;
$$ LANGUAGE plpgsql;

-- Views for monitoring and reporting

-- Active user permissions view
CREATE OR REPLACE VIEW active_user_permissions AS
SELECT 
    up.user_id,
    up.resource_type,
    up.resource_id,
    up.action,
    up.granted_by,
    up.granted_at,
    up.expires_at
FROM user_permissions up
WHERE up.is_active = true
  AND (up.expires_at IS NULL OR up.expires_at > NOW());

-- Security events summary view
CREATE OR REPLACE VIEW security_events_summary AS
SELECT 
    event_type,
    severity,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT ip_address) as unique_ips,
    MAX(timestamp) as latest_event
FROM security_events
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY event_type, severity
ORDER BY event_count DESC;

-- Rate limiting stats view
CREATE OR REPLACE VIEW rate_limit_stats AS
SELECT 
    user_id,
    endpoint,
    COUNT(*) as request_count,
    MAX(timestamp) as last_request,
    COUNT(DISTINCT ip_address) as unique_ips
FROM rate_limit_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY user_id, endpoint
ORDER BY request_count DESC;

-- Grant permissions for PRSM application user
-- (Assumes application connects with user 'prsm_app')
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO prsm_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO prsm_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO prsm_app;

-- Create indexes for performance optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_permissions_composite 
ON user_permissions(user_id, resource_type, action, is_active) 
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_recent 
ON audit_logs(timestamp DESC, user_id) 
WHERE timestamp > NOW() - INTERVAL '30 days';

-- Comments for documentation
COMMENT ON TABLE user_permissions IS 'Granular user permissions for RBAC system';
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for compliance and security';
COMMENT ON TABLE rate_limit_log IS 'Rate limiting fallback when Redis unavailable';
COMMENT ON TABLE blocked_ips IS 'Blocked IP addresses for security';
COMMENT ON TABLE security_events IS 'Security incidents and threat detection';
COMMENT ON TABLE user_sessions IS 'Active user sessions management';
COMMENT ON TABLE user_role_assignments IS 'User role assignments with expiration';

-- Insert initial admin permissions (run once during setup)
-- This creates foundational admin access for system bootstrap
INSERT INTO user_permissions (user_id, resource_type, action, granted_by)
SELECT 
    u.id,
    '*',
    'admin',
    u.id
FROM users u
WHERE u.role = 'admin'
  AND NOT EXISTS (
    SELECT 1 FROM user_permissions up
    WHERE up.user_id = u.id AND up.resource_type = '*' AND up.action = 'admin'
  );