-- Supabase Database Schema for Knowledge Ingestion System
-- This script creates all necessary tables, indexes, and functions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT,
    content TEXT,
    structure JSONB,
    parsing_method VARCHAR(50),
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    domain_classification VARCHAR(100),
    ingestion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_status VARCHAR(50) DEFAULT 'pending' CHECK (
        processing_status IN ('pending', 'processing', 'completed', 'failed', 'skipped')
    ),
    file_size_bytes BIGINT,
    page_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create chunks table with embedding vectors
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- Default OpenAI embedding dimension
    chunk_order INTEGER NOT NULL,
    section_header TEXT,
    semantic_metadata JSONB,
    token_count INTEGER,
    embedding_model VARCHAR(100),
    embedding_dimension INTEGER DEFAULT 1536,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    math_elements JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chunks_chunk_order_positive CHECK (chunk_order >= 0),
    CONSTRAINT chunks_token_count_positive CHECK (token_count > 0),
    CONSTRAINT chunks_embedding_dimension_positive CHECK (embedding_dimension > 0),
    CONSTRAINT chunks_unique_document_order UNIQUE (document_id, chunk_order)
);

-- Create ingestion logs table
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id VARCHAR(255) NOT NULL,
    phase VARCHAR(50) NOT NULL CHECK (
        phase IN ('discovery', 'download', 'parsing', 'chunking', 'embedding', 'storage', 'audit')
    ),
    status VARCHAR(50) NOT NULL CHECK (
        status IN ('started', 'completed', 'failed', 'skipped', 'retrying')
    ),
    error_message TEXT,
    error_code VARCHAR(50),
    processing_time_ms INTEGER,
    metadata JSONB,
    correlation_id VARCHAR(100),
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT ingestion_logs_processing_time_positive CHECK (processing_time_ms >= 0),
    CONSTRAINT ingestion_logs_retry_count_positive CHECK (retry_count >= 0)
);

-- Create trigger function for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create standard indexes for query optimization
-- Documents table indexes
CREATE INDEX IF NOT EXISTS idx_documents_file_id ON documents(file_id);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_domain_classification ON documents(domain_classification);
CREATE INDEX IF NOT EXISTS idx_documents_ingestion_timestamp ON documents(ingestion_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_quality_score ON documents(quality_score);

-- Chunks table indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_order ON chunks(document_id, chunk_order);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model);
CREATE INDEX IF NOT EXISTS idx_chunks_quality_score ON chunks(quality_score);
CREATE INDEX IF NOT EXISTS idx_chunks_token_count ON chunks(token_count);

-- Ingestion logs indexes
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_file_id ON ingestion_logs(file_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_phase_status ON ingestion_logs(phase, status);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_correlation_id ON ingestion_logs(correlation_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_created_at ON ingestion_logs(created_at);

-- JSONB indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_documents_structure_gin ON documents USING gin(structure);
CREATE INDEX IF NOT EXISTS idx_chunks_semantic_metadata_gin ON chunks USING gin(semantic_metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_math_elements_gin ON chunks USING gin(math_elements);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_metadata_gin ON ingestion_logs USING gin(metadata);

-- Create vector similarity search function
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    document_id uuid,
    content text,
    chunk_order int,
    section_header text,
    semantic_metadata jsonb,
    token_count int,
    embedding_model varchar(100),
    quality_score float,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        chunks.id,
        chunks.document_id,
        chunks.content,
        chunks.chunk_order,
        chunks.section_header,
        chunks.semantic_metadata,
        chunks.token_count,
        chunks.embedding_model,
        chunks.quality_score,
        1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM chunks
    WHERE 1 - (chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY chunks.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Create HNSW vector indexes for efficient similarity search
-- OpenAI text-embedding-3-large (1536 dimensions)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_1536 
   ON chunks USING hnsw (embedding vector_cosine_ops) 
   WITH (m = 16, ef_construction = 64)
   WHERE embedding_dimension = 1536;

-- BAAI/bge-large-en-v1.5 (1024 dimensions) - for future use
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_1024 
--    ON chunks USING hnsw (embedding vector_cosine_ops) 
--    WITH (m = 16, ef_construction = 64)
--    WHERE embedding_dimension = 1024;

-- Generic HNSW index for any dimension
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_generic 
   ON chunks USING hnsw (embedding vector_cosine_ops) 
   WITH (m = 16, ef_construction = 64);

-- Insert a test record to verify schema is working
-- This will be cleaned up after testing
INSERT INTO documents (
    file_id, 
    title, 
    content, 
    parsing_method, 
    domain_classification, 
    processing_status
) VALUES (
    'test_schema_setup',
    'Schema Setup Test Document',
    'This is a test document to verify the schema is working correctly.',
    'test',
    'General Technical',
    'completed'
) ON CONFLICT (file_id) DO NOTHING;

-- Clean up test record
DELETE FROM documents WHERE file_id = 'test_schema_setup';