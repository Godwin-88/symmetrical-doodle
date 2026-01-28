"""
Multi-Source Schema Migration

This module handles database schema migration to support multi-source document management.
It extends the existing documents table with fields needed for source attribution,
user-managed metadata, and enhanced document management capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from supabase import create_client, Client
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from core.config import get_settings
from core.logging import get_logger


class MultiSourceSchemaMigration:
    """Handles schema migration for multi-source document management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self._client: Optional[Client] = None
        self._pg_connection: Optional[Any] = None
    
    async def initialize_client(self) -> bool:
        """Initialize database connections"""
        try:
            # Initialize Supabase client
            self._client = create_client(
                self.settings.supabase.url,
                self.settings.supabase.service_role_key or self.settings.supabase.key
            )
            
            # Initialize direct PostgreSQL connection for schema operations
            if self.settings.supabase.database_url:
                self._pg_connection = psycopg2.connect(
                    self.settings.supabase.database_url,
                    connect_timeout=self.settings.supabase.timeout
                )
                self._pg_connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            self.logger.info("Database connections initialized for migration")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connections: {e}")
            return False
    
    async def migrate_to_multi_source_support(self) -> bool:
        """
        Migrate database schema to support multi-source document management.
        
        Adds fields for:
        - Source attribution (source_type, source_path, connection_id, etc.)
        - User-managed metadata (tags, categories, notes, is_favorite)
        - Enhanced processing information
        - Source-specific metadata storage
        """
        try:
            if not await self.initialize_client():
                return False
            
            self.logger.info("Starting multi-source schema migration")
            
            # Check if migration is needed
            if await self._check_migration_status():
                self.logger.info("Multi-source schema migration already applied")
                return True
            
            # Apply migration steps
            migration_steps = [
                self._add_source_attribution_fields,
                self._add_user_metadata_fields,
                self._add_enhanced_processing_fields,
                self._add_source_specific_metadata_fields,
                self._create_multi_source_indexes,
                self._update_existing_documents,
                self._mark_migration_complete
            ]
            
            for step in migration_steps:
                if not await step():
                    self.logger.error(f"Migration step failed: {step.__name__}")
                    return False
                self.logger.info(f"Migration step completed: {step.__name__}")
            
            self.logger.info("Multi-source schema migration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Multi-source schema migration failed: {e}")
            return False
    
    async def _check_migration_status(self) -> bool:
        """Check if multi-source migration has already been applied"""
        try:
            # Check if source_type column exists
            result = await self._execute_query("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'documents' 
                AND column_name = 'source_type'
            """)
            
            return len(result) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to check migration status: {e}")
            return False
    
    async def _add_source_attribution_fields(self) -> bool:
        """Add fields for source attribution"""
        try:
            sql = """
            -- Add source attribution fields
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS source_type VARCHAR(50) DEFAULT 'individual_upload',
            ADD COLUMN IF NOT EXISTS source_path TEXT,
            ADD COLUMN IF NOT EXISTS source_id VARCHAR(255),
            ADD COLUMN IF NOT EXISTS connection_id VARCHAR(255),
            ADD COLUMN IF NOT EXISTS parent_folders JSONB DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS access_url TEXT,
            ADD COLUMN IF NOT EXISTS is_accessible BOOLEAN DEFAULT true,
            ADD COLUMN IF NOT EXISTS access_permissions JSONB DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP WITH TIME ZONE,
            ADD COLUMN IF NOT EXISTS checksum VARCHAR(64);
            
            -- Add constraint for source_type
            ALTER TABLE documents 
            ADD CONSTRAINT IF NOT EXISTS documents_source_type_check 
            CHECK (source_type IN (
                'google_drive', 'local_directory', 'local_zip', 
                'individual_upload', 'aws_s3', 'azure_blob', 
                'google_cloud_storage'
            ));
            """
            
            await self._execute_sql(sql)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add source attribution fields: {e}")
            return False
    
    async def _add_user_metadata_fields(self) -> bool:
        """Add fields for user-managed metadata"""
        try:
            sql = """
            -- Add user-managed metadata fields
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS categories JSONB DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS notes TEXT,
            ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN DEFAULT false,
            ADD COLUMN IF NOT EXISTS content_type VARCHAR(100),
            ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en';
            """
            
            await self._execute_sql(sql)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add user metadata fields: {e}")
            return False
    
    async def _add_enhanced_processing_fields(self) -> bool:
        """Add enhanced processing information fields"""
        try:
            sql = """
            -- Add enhanced processing fields
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
            ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER,
            ADD COLUMN IF NOT EXISTS ingestion_job_id VARCHAR(255),
            ADD COLUMN IF NOT EXISTS modified_time TIMESTAMP WITH TIME ZONE,
            ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) DEFAULT 'application/pdf';
            
            -- Add constraints
            ALTER TABLE documents 
            ADD CONSTRAINT IF NOT EXISTS documents_chunk_count_positive 
            CHECK (chunk_count >= 0),
            ADD CONSTRAINT IF NOT EXISTS documents_processing_time_positive 
            CHECK (processing_time_ms >= 0);
            """
            
            await self._execute_sql(sql)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add enhanced processing fields: {e}")
            return False
    
    async def _add_source_specific_metadata_fields(self) -> bool:
        """Add field for source-specific metadata storage"""
        try:
            sql = """
            -- Add source-specific metadata field
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS source_specific_metadata JSONB DEFAULT '{}'::jsonb;
            """
            
            await self._execute_sql(sql)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add source-specific metadata fields: {e}")
            return False
    
    async def _create_multi_source_indexes(self) -> bool:
        """Create indexes for multi-source queries"""
        try:
            indexes = [
                # Source attribution indexes
                "CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);",
                "CREATE INDEX IF NOT EXISTS idx_documents_connection_id ON documents(connection_id) WHERE connection_id IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id) WHERE source_id IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_is_accessible ON documents(is_accessible);",
                "CREATE INDEX IF NOT EXISTS idx_documents_last_accessed ON documents(last_accessed) WHERE last_accessed IS NOT NULL;",
                
                # User metadata indexes
                "CREATE INDEX IF NOT EXISTS idx_documents_tags_gin ON documents USING gin(tags);",
                "CREATE INDEX IF NOT EXISTS idx_documents_categories_gin ON documents USING gin(categories);",
                "CREATE INDEX IF NOT EXISTS idx_documents_is_favorite ON documents(is_favorite) WHERE is_favorite = true;",
                "CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type) WHERE content_type IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language);",
                
                # Enhanced processing indexes
                "CREATE INDEX IF NOT EXISTS idx_documents_embedding_model ON documents(embedding_model) WHERE embedding_model IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_chunk_count ON documents(chunk_count);",
                "CREATE INDEX IF NOT EXISTS idx_documents_ingestion_job_id ON documents(ingestion_job_id) WHERE ingestion_job_id IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_modified_time ON documents(modified_time) WHERE modified_time IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_mime_type ON documents(mime_type);",
                
                # Source-specific metadata index
                "CREATE INDEX IF NOT EXISTS idx_documents_source_metadata_gin ON documents USING gin(source_specific_metadata);",
                
                # Composite indexes for common queries
                "CREATE INDEX IF NOT EXISTS idx_documents_source_status ON documents(source_type, processing_status);",
                "CREATE INDEX IF NOT EXISTS idx_documents_source_domain ON documents(source_type, domain_classification) WHERE domain_classification IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_documents_connection_status ON documents(connection_id, processing_status) WHERE connection_id IS NOT NULL;"
            ]
            
            for index_sql in indexes:
                await self._execute_sql(index_sql)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-source indexes: {e}")
            return False
    
    async def _update_existing_documents(self) -> bool:
        """Update existing documents with default multi-source values"""
        try:
            sql = """
            -- Update existing documents with default values
            UPDATE documents 
            SET 
                source_type = COALESCE(source_type, 'individual_upload'),
                source_path = COALESCE(source_path, file_id),
                source_id = COALESCE(source_id, file_id),
                is_accessible = COALESCE(is_accessible, true),
                tags = COALESCE(tags, '[]'::jsonb),
                categories = COALESCE(categories, '[]'::jsonb),
                is_favorite = COALESCE(is_favorite, false),
                language = COALESCE(language, 'en'),
                chunk_count = COALESCE(chunk_count, 0),
                mime_type = COALESCE(mime_type, 'application/pdf'),
                source_specific_metadata = COALESCE(source_specific_metadata, '{}'::jsonb),
                modified_time = COALESCE(modified_time, created_at)
            WHERE 
                source_type IS NULL 
                OR source_path IS NULL 
                OR source_id IS NULL;
            """
            
            await self._execute_sql(sql)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update existing documents: {e}")
            return False
    
    async def _mark_migration_complete(self) -> bool:
        """Mark migration as complete by creating a migration record"""
        try:
            # Create migrations table if it doesn't exist
            sql = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                migration_name VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                version VARCHAR(50) NOT NULL
            );
            
            -- Insert migration record
            INSERT INTO schema_migrations (migration_name, version)
            VALUES ('multi_source_support', '1.0.0')
            ON CONFLICT (migration_name) DO NOTHING;
            """
            
            await self._execute_sql(sql)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to mark migration complete: {e}")
            return False
    
    async def _execute_sql(self, sql: str) -> None:
        """Execute SQL statement"""
        if self._pg_connection:
            cursor = self._pg_connection.cursor()
            cursor.execute(sql)
            cursor.close()
        else:
            # Fallback to Supabase client (limited functionality)
            raise Exception("Direct PostgreSQL connection required for schema operations")
    
    async def _execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        if self._pg_connection:
            cursor = self._pg_connection.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            return results
        else:
            raise Exception("Direct PostgreSQL connection required for schema operations")
    
    async def rollback_migration(self) -> bool:
        """Rollback multi-source schema migration (for testing/development)"""
        try:
            if not await self.initialize_client():
                return False
            
            self.logger.warning("Rolling back multi-source schema migration")
            
            # Drop added columns (be careful with this in production!)
            sql = """
            ALTER TABLE documents 
            DROP COLUMN IF EXISTS source_type,
            DROP COLUMN IF EXISTS source_path,
            DROP COLUMN IF EXISTS source_id,
            DROP COLUMN IF EXISTS connection_id,
            DROP COLUMN IF EXISTS parent_folders,
            DROP COLUMN IF EXISTS access_url,
            DROP COLUMN IF EXISTS is_accessible,
            DROP COLUMN IF EXISTS access_permissions,
            DROP COLUMN IF EXISTS last_accessed,
            DROP COLUMN IF EXISTS checksum,
            DROP COLUMN IF EXISTS tags,
            DROP COLUMN IF EXISTS categories,
            DROP COLUMN IF EXISTS notes,
            DROP COLUMN IF EXISTS is_favorite,
            DROP COLUMN IF EXISTS content_type,
            DROP COLUMN IF EXISTS language,
            DROP COLUMN IF EXISTS embedding_model,
            DROP COLUMN IF EXISTS chunk_count,
            DROP COLUMN IF EXISTS processing_time_ms,
            DROP COLUMN IF EXISTS ingestion_job_id,
            DROP COLUMN IF EXISTS modified_time,
            DROP COLUMN IF EXISTS mime_type,
            DROP COLUMN IF EXISTS source_specific_metadata;
            
            -- Remove migration record
            DELETE FROM schema_migrations WHERE migration_name = 'multi_source_support';
            """
            
            await self._execute_sql(sql)
            
            self.logger.info("Multi-source schema migration rolled back")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback migration: {e}")
            return False


async def apply_multi_source_migration():
    """Apply multi-source schema migration"""
    migration = MultiSourceSchemaMigration()
    return await migration.migrate_to_multi_source_support()


async def rollback_multi_source_migration():
    """Rollback multi-source schema migration"""
    migration = MultiSourceSchemaMigration()
    return await migration.rollback_migration()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        result = asyncio.run(rollback_multi_source_migration())
    else:
        result = asyncio.run(apply_multi_source_migration())
    
    if result:
        print("✅ Migration completed successfully")
    else:
        print("❌ Migration failed")
        sys.exit(1)