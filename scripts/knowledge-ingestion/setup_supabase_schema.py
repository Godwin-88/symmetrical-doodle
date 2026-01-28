#!/usr/bin/env python3
"""
Supabase Database Schema Setup Script

This script initializes the database schema for the knowledge ingestion system.
It creates all necessary tables, indexes, and constraints.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.supabase_schema import SupabaseSchemaManager
from core.config import load_config
from core.logging import configure_logging, get_logger


async def setup_schema():
    """Setup Supabase database schema"""
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting Supabase database schema setup")
    
    try:
        # Load configuration
        settings = load_config("development")
        
        if not settings.supabase.url or not settings.supabase.key:
            logger.error("Supabase configuration missing. Please check .env file.")
            return False
        
        logger.info(f"Connecting to Supabase: {settings.supabase.url}")
        
        # Initialize schema manager
        schema_manager = SupabaseSchemaManager(settings.supabase)
        
        # Validate current schema
        logger.info("Validating current database schema...")
        validation_result = await schema_manager.validate_schema()
        
        if validation_result.valid:
            logger.info("✅ Database schema is already valid and up-to-date!")
            print("\n🎉 Database schema is ready!")
            return True
        
        # Display validation results
        logger.info("Schema validation results:")
        if validation_result.missing_extensions:
            logger.warning(f"Missing extensions: {validation_result.missing_extensions}")
        if validation_result.missing_tables:
            logger.warning(f"Missing tables: {validation_result.missing_tables}")
        if validation_result.missing_indexes:
            logger.warning(f"Missing indexes: {validation_result.missing_indexes}")
        if validation_result.errors:
            for error in validation_result.errors:
                logger.error(f"Schema error: {error}")
        
        # Apply migrations to fix schema
        logger.info("Applying database migrations...")
        migration_result = await schema_manager.migrate_schema()
        
        if migration_result.success:
            logger.info("✅ Database migrations applied successfully!")
            logger.info(f"Applied migrations: {migration_result.applied_migrations}")
            logger.info(f"Migration completed in {migration_result.execution_time_ms}ms")
            
            # Validate schema again
            logger.info("Re-validating database schema...")
            final_validation = await schema_manager.validate_schema()
            
            if final_validation.valid:
                logger.info("✅ Database schema validation passed!")
                print("\n" + "="*80)
                print("DATABASE SCHEMA SETUP COMPLETED SUCCESSFULLY")
                print("="*80)
                print("✅ All required tables created")
                print("✅ All indexes and constraints applied")
                print("✅ Vector search functionality enabled")
                print("✅ Schema validation passed")
                print("\n🎯 Next Steps:")
                print("   1. Run the local ZIP ingestion test again")
                print("   2. Documents should now be stored successfully in Supabase")
                print("   3. Proceed with full pipeline processing")
                return True
            else:
                logger.error("❌ Schema validation still failing after migration")
                for error in final_validation.errors:
                    logger.error(f"Remaining error: {error}")
                return False
        else:
            logger.error("❌ Database migration failed!")
            for error in migration_result.errors:
                logger.error(f"Migration error: {error}")
            return False
    
    except Exception as e:
        logger.error(f"Schema setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up connections
        if 'schema_manager' in locals():
            schema_manager.close()


async def main():
    """Main function"""
    success = await setup_schema()
    
    if success:
        print("\n🎉 Supabase database schema setup completed successfully!")
        print("The knowledge ingestion system is now ready to store documents.")
    else:
        print("\n❌ Schema setup failed. Please check the logs and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())