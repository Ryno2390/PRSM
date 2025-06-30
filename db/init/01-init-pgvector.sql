-- Initialize pgvector extension and PRSM vector store schema
-- This script runs automatically when the PostgreSQL container starts

\echo 'Initializing PRSM Vector Store Database...'

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

\echo 'pgvector extension installed successfully'

-- Create PRSM vector store schema
CREATE SCHEMA IF NOT EXISTS prsm_vector;

\echo 'Created prsm_vector schema'

-- Set default search path
ALTER DATABASE prsm_vector_dev SET search_path TO prsm_vector, public;

\echo 'Set search path to include prsm_vector schema'

-- Create enum types for content classification
CREATE TYPE prsm_vector.content_type AS ENUM (
    'text',
    'image', 
    'audio',
    'video',
    'code',
    'research_paper',
    'dataset'
);

\echo 'Created content_type enum'

-- Create the main vector content table
CREATE TABLE IF NOT EXISTS prsm_vector.content_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_cid TEXT NOT NULL UNIQUE,
    embedding vector(384),  -- Default to 384 dimensions (can be adjusted)
    
    -- Content metadata
    title TEXT,
    description TEXT,
    content_type prsm_vector.content_type DEFAULT 'text',
    
    -- Provenance and creator information
    creator_id TEXT,
    royalty_rate DECIMAL(5,4) DEFAULT 0.08,  -- 8% default royalty
    license TEXT,
    
    -- Quality and performance metrics
    quality_score DECIMAL(3,2) CHECK (quality_score >= 0 AND quality_score <= 1),
    peer_review_score DECIMAL(3,2) CHECK (peer_review_score >= 0 AND peer_review_score <= 1),
    citation_count INTEGER DEFAULT 0,
    
    -- Access tracking
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    
    -- Administrative
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    
    -- Additional metadata as JSONB for flexibility
    metadata JSONB DEFAULT '{}'::jsonb
);

\echo 'Created content_vectors table'

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS content_vectors_embedding_idx 
ON prsm_vector.content_vectors 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

\echo 'Created HNSW index for vector similarity search'

-- Create btree indexes for common queries
CREATE INDEX IF NOT EXISTS content_vectors_content_cid_idx 
ON prsm_vector.content_vectors (content_cid);

CREATE INDEX IF NOT EXISTS content_vectors_creator_id_idx 
ON prsm_vector.content_vectors (creator_id);

CREATE INDEX IF NOT EXISTS content_vectors_content_type_idx 
ON prsm_vector.content_vectors (content_type);

CREATE INDEX IF NOT EXISTS content_vectors_quality_score_idx 
ON prsm_vector.content_vectors (quality_score);

CREATE INDEX IF NOT EXISTS content_vectors_created_at_idx 
ON prsm_vector.content_vectors (created_at);

\echo 'Created btree indexes for efficient filtering'

-- Create GIN index for metadata queries
CREATE INDEX IF NOT EXISTS content_vectors_metadata_idx 
ON prsm_vector.content_vectors 
USING GIN (metadata);

\echo 'Created GIN index for metadata queries'

-- Create function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION prsm_vector.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update updated_at
CREATE TRIGGER content_vectors_updated_at_trigger
    BEFORE UPDATE ON prsm_vector.content_vectors
    FOR EACH ROW
    EXECUTE FUNCTION prsm_vector.update_updated_at();

\echo 'Created updated_at trigger'

-- Create collection statistics view
CREATE OR REPLACE VIEW prsm_vector.collection_stats AS
SELECT 
    COUNT(*) as total_vectors,
    COUNT(DISTINCT creator_id) as unique_creators,
    COUNT(DISTINCT content_type) as content_types,
    AVG(citation_count) as average_citations,
    AVG(quality_score) as average_quality,
    MIN(created_at) as earliest_content,
    MAX(created_at) as latest_content,
    SUM(access_count) as total_accesses,
    pg_size_pretty(pg_total_relation_size('prsm_vector.content_vectors')) as table_size
FROM prsm_vector.content_vectors 
WHERE active = TRUE;

\echo 'Created collection_stats view'

-- Insert some sample data for testing
INSERT INTO prsm_vector.content_vectors 
(content_cid, title, content_type, creator_id, royalty_rate, quality_score, citation_count, metadata)
VALUES 
('QmSample1_AIEthics', 'AI Ethics in Modern Systems', 'research_paper', 'sample_researcher_1', 0.08, 0.95, 42, '{"keywords": ["AI", "ethics", "governance"], "journal": "AI Ethics Quarterly"}'),
('QmSample2_ClimateData', 'Global Climate Dataset 2024', 'dataset', 'climate_research_org', 0.06, 0.98, 156, '{"data_size": "2.5TB", "format": "NetCDF", "resolution": "1km"}'),
('QmSample3_VectorDB', 'Efficient Vector Database Implementation', 'code', 'open_source_dev', 0.05, 0.87, 73, '{"language": "Python", "framework": "PostgreSQL", "license": "MIT"}')
ON CONFLICT (content_cid) DO NOTHING;

\echo 'Inserted sample data'

-- Create function for similarity search
CREATE OR REPLACE FUNCTION prsm_vector.search_similar_content(
    query_embedding vector(384),
    content_types prsm_vector.content_type[] DEFAULT NULL,
    creator_ids TEXT[] DEFAULT NULL,
    min_quality_score DECIMAL DEFAULT NULL,
    max_royalty_rate DECIMAL DEFAULT NULL,
    top_k INTEGER DEFAULT 10
)
RETURNS TABLE (
    content_cid TEXT,
    similarity_score DECIMAL,
    title TEXT,
    content_type prsm_vector.content_type,
    creator_id TEXT,
    royalty_rate DECIMAL,
    quality_score DECIMAL,
    citation_count INTEGER,
    access_count INTEGER,
    last_accessed TIMESTAMP WITH TIME ZONE,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cv.content_cid,
        (1 - (cv.embedding <=> query_embedding))::DECIMAL as similarity_score,
        cv.title,
        cv.content_type,
        cv.creator_id,
        cv.royalty_rate,
        cv.quality_score,
        cv.citation_count,
        cv.access_count,
        cv.last_accessed,
        cv.metadata
    FROM prsm_vector.content_vectors cv
    WHERE cv.active = TRUE
        AND (content_types IS NULL OR cv.content_type = ANY(content_types))
        AND (creator_ids IS NULL OR cv.creator_id = ANY(creator_ids))
        AND (min_quality_score IS NULL OR cv.quality_score >= min_quality_score)
        AND (max_royalty_rate IS NULL OR cv.royalty_rate <= max_royalty_rate)
    ORDER BY cv.embedding <=> query_embedding
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

\echo 'Created search_similar_content function'

-- Create function to track content access
CREATE OR REPLACE FUNCTION prsm_vector.track_content_access(
    p_content_cid TEXT,
    p_user_id TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    rows_updated INTEGER;
BEGIN
    UPDATE prsm_vector.content_vectors 
    SET 
        access_count = access_count + 1,
        last_accessed = NOW(),
        metadata = metadata || jsonb_build_object('last_user_id', p_user_id)
    WHERE content_cid = p_content_cid AND active = TRUE;
    
    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    RETURN rows_updated > 0;
END;
$$ LANGUAGE plpgsql;

\echo 'Created track_content_access function'

-- Grant permissions to postgres user (development setup)
GRANT ALL PRIVILEGES ON SCHEMA prsm_vector TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA prsm_vector TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA prsm_vector TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA prsm_vector TO postgres;

\echo 'Granted permissions to postgres user'

-- Display initialization summary
\echo ''
\echo '============================================'
\echo 'PRSM Vector Store Database Initialized!'
\echo '============================================'
\echo 'Database: prsm_vector_dev'
\echo 'Schema: prsm_vector'
\echo 'Main table: content_vectors'
\echo 'Vector dimensions: 384 (configurable)'
\echo 'Index: HNSW for cosine similarity'
\echo 'Sample data: 3 test records inserted'
\echo ''
\echo 'Ready for PRSM vector store operations!'
\echo '============================================'