-- PRSM PostgreSQL Initialization Script
-- Creates additional databases and configurations for PRSM

-- Create additional databases for testing and development
CREATE DATABASE prsm_test;
CREATE DATABASE prsm_dev;

-- Create read-only user for monitoring
CREATE USER prsm_monitor WITH PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE prsm TO prsm_monitor;
GRANT USAGE ON SCHEMA public TO prsm_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO prsm_monitor;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO prsm_monitor;

-- Create backup user
CREATE USER prsm_backup WITH PASSWORD 'backup_password';
GRANT CONNECT ON DATABASE prsm TO prsm_backup;
GRANT USAGE ON SCHEMA public TO prsm_backup;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO prsm_backup;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO prsm_backup;

-- Performance optimizations
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Configure for PRSM workload
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- Reload configuration
SELECT pg_reload_conf();