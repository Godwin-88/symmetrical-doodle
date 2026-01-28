"""
Supabase database schema management service.

This module handles database schema initialization, migration, and validation
for the Google Drive Knowledge Base Ingestion system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from supabase import create_client, Client
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pgvector

from core.config import SupabaseConfig
from core.logging import get_logger


@dataclass
class SchemaValidationResult:
    """Result of schema validation"""
    valid: bool
    missing_tables: List[str]
    missing_indexes: List[str]
    missing_extensions: List[str]
    errors: List[str]
    warnings: List[str]


@dataclass
class MigrationResult:
    """Result of schema migration"""
    success: bool
    applied_migrations: List[str]
    errors: List[str]
    execution_time_ms: int


class SupabaseSchemaManager:
    """
    Manages database schema initialization, migration, and validation for Supabase.
    
    Handles:
    - Schema initialization for documents, chunks, and ingestion_logs tables
    - HNSW vector indexes for efficient similarity search
    - Schema migration and validation functionality
    - Proper relationships, indexes, and constraints
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self._client: Optional[Client] = None
        self._pg_connection: Optional[Any] = None
        
    async def initialize_client(self) -> bool:
        """Initialize Supabase client and PostgreSQL connection"""
        try:
            # Initialize Supabase client
            self._client = create_client(
                self.config.url,
                self.config.service_role_key or self.config.key
            )
            
            # Initialize direct PostgreSQL connection for schema operations
            if self.config.database_url:
                self._pg_connection = psycopg2.connect(
                    self.config.database_url,
                    connect_timeout=self.config.timeout
                )
                self._pg_connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            self.logger.info("Supabase client and PostgreSQL connection initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase client: {e}")
            return False
    
    async def initialize_schema(self) -> bool:
        """
        Initialize complete database schema with all required tables, indexes, and constraints.
        
        Returns:
            bool: True if schema initialization successful, False otherwise
        """
        try:
            if not await self.initialize_client():
                return False
            
            self.logger.info("Starting database schema initialization")
            
            # Enable required extensions
            if not await self._enable_extensions():
                return False
            
            # Create tables in dependency order
            if not await self._create_documents_table():
                return False
                
            if not await self._create_chunks_table():
                return False
                
            if not await self._create_ingestion_logs_table():
                return False
            
            # Create indexes
            if not await self._create_indexes():
                return False
            
            # Create HNSW vector indexes
            if not await self._create_vector_indexes():
                return False
            
            self.logger.info("Database schema initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {e}")
            return False
    
    async def _enable_extensions(self) -> bool:
        """Enable required PostgreSQL extensions"""
        try:
            extensions = [
                "CREATE EXTENSION IF NOT EXISTS vector;",
                "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
            ]
            
            for extension_sql in extensions:
                await self._execute_sql(extension_sql)
            
            self.logger.info("Required extensions enabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable extensions: {e}")
            return False
    
    async def _create_documents_table(self) -> bool:
        """Create documents table with all required fields and constraints"""
        try:
            sql = """
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
            
            -- Create trigger to update updated_at timestamp
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            
            DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
            CREATE TRIGGER update_documents_updated_at
                BEFORE UPDATE ON documents
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
            
            await self._execute_sql(sql)
            self.logger.info("Documents table created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create documents table: {e}")
            return False
    
    async def _create_chunks_table(self) -> bool:
        """Create chunks table with embedding vectors and semantic metadata"""
        try:
            sql = """
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
            """
            
            await self._execute_sql(sql)
            self.logger.info("Chunks table created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create chunks table: {e}")
            return False
    
    async def _create_ingestion_logs_table(self) -> bool:
        """Create ingestion logs table for tracking processing status"""
        try:
            sql = """
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
            """
            
            await self._execute_sql(sql)
            self.logger.info("Ingestion logs table created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create ingestion logs table: {e}")
            return False
    
    async def _create_indexes(self) -> bool:
        """Create standard indexes for query optimization"""
        try:
            indexes = [
                # Documents table indexes
                "CREATE INDEX IF NOT EXISTS idx_documents_file_id ON documents(file_id);",
                "CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);",
                "CREATE INDEX IF NOT EXISTS idx_documents_domain_classification ON documents(domain_classification);",
                "CREATE INDEX IF NOT EXISTS idx_documents_ingestion_timestamp ON documents(ingestion_timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_documents_quality_score ON documents(quality_score);",
                
                # Chunks table indexes
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);",
                "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_order ON chunks(document_id, chunk_order);",
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model ON chunks(embedding_model);",
                "CREATE INDEX IF NOT EXISTS idx_chunks_quality_score ON chunks(quality_score);",
                "CREATE INDEX IF NOT EXISTS idx_chunks_token_count ON chunks(token_count);",
                
                # Ingestion logs indexes
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_file_id ON ingestion_logs(file_id);",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_phase_status ON ingestion_logs(phase, status);",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_correlation_id ON ingestion_logs(correlation_id);",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_created_at ON ingestion_logs(created_at);",
                
                # JSONB indexes for metadata queries
                "CREATE INDEX IF NOT EXISTS idx_documents_structure_gin ON documents USING gin(structure);",
                "CREATE INDEX IF NOT EXISTS idx_chunks_semantic_metadata_gin ON chunks USING gin(semantic_metadata);",
                "CREATE INDEX IF NOT EXISTS idx_chunks_math_elements_gin ON chunks USING gin(math_elements);",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_metadata_gin ON ingestion_logs USING gin(metadata);"
            ]
            
            for index_sql in indexes:
                await self._execute_sql(index_sql)
            
            self.logger.info("Standard indexes created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
            return False
    
    async def _create_vector_indexes(self) -> bool:
        """Create HNSW vector indexes for efficient similarity search"""
        try:
            # Create vector similarity search function
            similarity_function = """
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
            """
            
            await self._execute_sql(similarity_function)
            
            # HNSW indexes for different embedding dimensions
            vector_indexes = [
                # OpenAI text-embedding-3-large (1536 dimensions)
                """CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_1536 
                   ON chunks USING hnsw (embedding vector_cosine_ops) 
                   WITH (m = 16, ef_construction = 64)
                   WHERE embedding_dimension = 1536;""",
                
                # BAAI/bge-large-en-v1.5 (1024 dimensions)
                """CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_1024 
                   ON chunks USING hnsw (embedding vector_cosine_ops) 
                   WITH (m = 16, ef_construction = 64)
                   WHERE embedding_dimension = 1024;""",
                
                # sentence-transformers/all-mpnet-base-v2 (768 dimensions)
                """CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_768 
                   ON chunks USING hnsw (embedding vector_cosine_ops) 
                   WITH (m = 16, ef_construction = 64)
                   WHERE embedding_dimension = 768;""",
                
                # Generic HNSW index for any dimension
                """CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_generic 
                   ON chunks USING hnsw (embedding vector_cosine_ops) 
                   WITH (m = 16, ef_construction = 64);"""
            ]
            
            for index_sql in vector_indexes:
                await self._execute_sql(index_sql)
            
            self.logger.info("HNSW vector indexes and similarity function created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create vector indexes: {e}")
            return False
    
    async def validate_schema(self) -> SchemaValidationResult:
        """
        Validate database schema completeness and integrity.
        
        Returns:
            SchemaValidationResult: Validation results with missing components and errors
        """
        result = SchemaValidationResult(
            valid=True,
            missing_tables=[],
            missing_indexes=[],
            missing_extensions=[],
            errors=[],
            warnings=[]
        )
        
        try:
            if not await self.initialize_client():
                result.valid = False
                result.errors.append("Failed to initialize database connection")
                return result
            
            # Check required extensions
            missing_extensions = await self._check_extensions()
            result.missing_extensions = missing_extensions
            if missing_extensions:
                result.valid = False
                result.errors.extend([f"Missing extension: {ext}" for ext in missing_extensions])
            
            # Check required tables
            missing_tables = await self._check_tables()
            result.missing_tables = missing_tables
            if missing_tables:
                result.valid = False
                result.errors.extend([f"Missing table: {table}" for table in missing_tables])
            
            # Check required indexes
            missing_indexes = await self._check_indexes()
            result.missing_indexes = missing_indexes
            if missing_indexes:
                result.warnings.extend([f"Missing index: {index}" for index in missing_indexes])
            
            # Check table constraints and relationships
            constraint_errors = await self._check_constraints()
            if constraint_errors:
                result.errors.extend(constraint_errors)
                result.valid = False
            
            self.logger.info(f"Schema validation completed. Valid: {result.valid}")
            return result
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Schema validation failed: {e}")
            self.logger.error(f"Schema validation error: {e}")
            return result
    
    async def _check_extensions(self) -> List[str]:
        """Check for required PostgreSQL extensions"""
        try:
            sql = """
            SELECT extname FROM pg_extension 
            WHERE extname IN ('vector', 'uuid-ossp');
            """
            
            result = await self._execute_query(sql)
            existing_extensions = [row[0] for row in result]
            
            required_extensions = ['vector', 'uuid-ossp']
            missing = [ext for ext in required_extensions if ext not in existing_extensions]
            
            return missing
            
        except Exception as e:
            self.logger.error(f"Failed to check extensions: {e}")
            return ['vector', 'uuid-ossp']  # Assume missing on error
    
    async def _check_tables(self) -> List[str]:
        """Check for required tables"""
        try:
            sql = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('documents', 'chunks', 'ingestion_logs');
            """
            
            result = await self._execute_query(sql)
            existing_tables = [row[0] for row in result]
            
            required_tables = ['documents', 'chunks', 'ingestion_logs']
            missing = [table for table in required_tables if table not in existing_tables]
            
            return missing
            
        except Exception as e:
            self.logger.error(f"Failed to check tables: {e}")
            return ['documents', 'chunks', 'ingestion_logs']  # Assume missing on error
    
    async def _check_indexes(self) -> List[str]:
        """Check for required indexes"""
        try:
            sql = """
            SELECT indexname FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND tablename IN ('documents', 'chunks', 'ingestion_logs');
            """
            
            result = await self._execute_query(sql)
            existing_indexes = [row[0] for row in result]
            
            # Key indexes to check for
            required_indexes = [
                'idx_documents_file_id',
                'idx_chunks_document_id',
                'idx_chunks_embedding_hnsw_generic',
                'idx_ingestion_logs_file_id'
            ]
            
            missing = [idx for idx in required_indexes if idx not in existing_indexes]
            return missing
            
        except Exception as e:
            self.logger.error(f"Failed to check indexes: {e}")
            return required_indexes  # Assume missing on error
    
    async def _check_constraints(self) -> List[str]:
        """Check table constraints and relationships"""
        errors = []
        
        try:
            # Check foreign key constraints
            sql = """
            SELECT conname, conrelid::regclass, confrelid::regclass
            FROM pg_constraint 
            WHERE contype = 'f' 
            AND conrelid::regclass::text IN ('documents', 'chunks', 'ingestion_logs');
            """
            
            result = await self._execute_query(sql)
            
            # Check if chunks -> documents foreign key exists
            fk_exists = any(
                'chunks' in str(row[1]) and 'documents' in str(row[2]) 
                for row in result
            )
            
            if not fk_exists:
                errors.append("Missing foreign key constraint: chunks.document_id -> documents.id")
            
        except Exception as e:
            errors.append(f"Failed to check constraints: {e}")
        
        return errors
    
    async def migrate_schema(self, target_version: Optional[str] = None) -> MigrationResult:
        """
        Apply schema migrations to bring database to target version.
        
        Args:
            target_version: Target schema version (None for latest)
            
        Returns:
            MigrationResult: Migration execution results
        """
        start_time = datetime.now()
        
        result = MigrationResult(
            success=True,
            applied_migrations=[],
            errors=[],
            execution_time_ms=0
        )
        
        try:
            # For now, we'll implement a simple migration that ensures schema is up to date
            # In a production system, this would track migration versions
            
            self.logger.info("Starting schema migration")
            
            # Check current schema state
            validation = await self.validate_schema()
            
            if validation.valid:
                self.logger.info("Schema is already up to date")
                result.applied_migrations.append("Schema validation passed - no migrations needed")
            else:
                # Apply missing components
                if validation.missing_extensions:
                    if await self._enable_extensions():
                        result.applied_migrations.append("Enabled missing extensions")
                    else:
                        result.success = False
                        result.errors.append("Failed to enable extensions")
                
                if validation.missing_tables:
                    if await self.initialize_schema():
                        result.applied_migrations.append("Created missing tables and indexes")
                    else:
                        result.success = False
                        result.errors.append("Failed to create missing tables")
            
            end_time = datetime.now()
            result.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self.logger.info(f"Schema migration completed in {result.execution_time_ms}ms")
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Migration failed: {e}")
            self.logger.error(f"Schema migration error: {e}")
            
            end_time = datetime.now()
            result.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return result
    
    async def _execute_sql(self, sql: str) -> None:
        """Execute SQL statement using direct PostgreSQL connection"""
        if self._pg_connection:
            cursor = self._pg_connection.cursor()
            cursor.execute(sql)
            cursor.close()
        else:
            # Fallback to Supabase client if direct connection not available
            if self._client:
                # Note: Supabase client doesn't support DDL operations directly
                # This is a limitation - we need direct PostgreSQL access for schema operations
                raise Exception("Direct PostgreSQL connection required for schema operations")
    
    async def _execute_query(self, sql: str) -> List[Tuple]:
        """Execute SQL query and return results"""
        if self._pg_connection:
            cursor = self._pg_connection.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            return result
        else:
            raise Exception("Direct PostgreSQL connection required for schema queries")
    
    def close(self):
        """Close database connections"""
        if self._pg_connection:
            self._pg_connection.close()
            self._pg_connection = None
        
        self._client = None
        self.logger.info("Database connections closed")