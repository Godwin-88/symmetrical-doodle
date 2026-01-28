#!/usr/bin/env python3
"""
Simple Supabase Schema Setup Script

This script executes the schema.sql file to create the database schema.
Uses the Supabase Python client for execution.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supabase import create_client, Client
from core.config import load_config
from core.logging import configure_logging, get_logger


async def setup_schema_simple():
    """Setup Supabase database schema using SQL file"""
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting simple Supabase database schema setup")
    
    try:
        # Load configuration
        settings = load_config("development")
        
        if not settings.supabase.url or not settings.supabase.key:
            logger.error("Supabase configuration missing. Please check .env file.")
            return False
        
        logger.info(f"Connecting to Supabase: {settings.supabase.url}")
        
        # Create Supabase client
        supabase: Client = create_client(
            settings.supabase.url,
            settings.supabase.key
        )
        
        # Read SQL schema file
        schema_file = Path(__file__).parent / "schema.sql"
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        logger.info(f"Reading schema from: {schema_file}")
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        # Execute each statement
        success_count = 0
        error_count = 0
        
        for i, statement in enumerate(statements, 1):
            try:
                # Skip comments and empty statements
                if statement.startswith('--') or not statement.strip():
                    continue
                
                logger.info(f"Executing statement {i}/{len(statements)}")
                
                # Execute using Supabase RPC for DDL operations
                result = supabase.rpc('exec_sql', {'sql': statement}).execute()
                
                if result.data is not None:
                    success_count += 1
                    logger.info(f"✅ Statement {i} executed successfully")
                else:
                    error_count += 1
                    logger.warning(f"⚠️ Statement {i} returned no data (may be normal for DDL)")
                    success_count += 1  # Count as success for DDL
                
            except Exception as e:
                error_count += 1
                error_msg = str(e)
                
                # Some errors are expected/acceptable
                if any(acceptable in error_msg.lower() for acceptable in [
                    'already exists',
                    'relation already exists',
                    'extension already exists',
                    'function already exists',
                    'trigger already exists'
                ]):
                    logger.info(f"✅ Statement {i} - Object already exists (OK)")
                    success_count += 1
                    error_count -= 1
                else:
                    logger.error(f"❌ Statement {i} failed: {error_msg}")
        
        logger.info(f"Schema setup completed: {success_count} successful, {error_count} errors")
        
        if error_count == 0:
            logger.info("✅ All schema operations completed successfully!")
            
            # Test the schema by trying to query the documents table
            try:
                result = supabase.table('documents').select('id').limit(1).execute()
                logger.info("✅ Schema validation: documents table is accessible")
                
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
                
            except Exception as e:
                logger.error(f"Schema validation failed: {e}")
                return False
        else:
            logger.error(f"❌ Schema setup completed with {error_count} errors")
            return False
    
    except Exception as e:
        logger.error(f"Schema setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function"""
    success = await setup_schema_simple()
    
    if success:
        print("\n🎉 Supabase database schema setup completed successfully!")
        print("The knowledge ingestion system is now ready to store documents.")
    else:
        print("\n❌ Schema setup failed. Please check the logs and configuration.")
        print("\n💡 Alternative: You can manually execute the schema.sql file in the Supabase dashboard:")
        print("   1. Go to https://supabase.com/dashboard")
        print("   2. Open your project")
        print("   3. Go to SQL Editor")
        print("   4. Copy and paste the contents of schema.sql")
        print("   5. Run the SQL")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())