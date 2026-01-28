#!/usr/bin/env python3
"""
Direct PostgreSQL Schema Setup Script

This script connects directly to PostgreSQL using psycopg2 and executes the schema.sql file.
Uses the session pooler connection string for better performance.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("❌ psycopg2 not installed. Please install it with: pip install psycopg2-binary")
    sys.exit(1)

from core.config import load_config
from core.logging import configure_logging, get_logger


async def setup_schema_direct():
    """Setup Supabase database schema using direct PostgreSQL connection"""
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting direct PostgreSQL database schema setup")
    
    try:
        # Load configuration
        settings = load_config("development")
        
        if not settings.supabase.database_url:
            logger.error("Database URL missing. Please check .env file.")
            return False
        
        logger.info(f"Connecting to PostgreSQL database...")
        
        # Create PostgreSQL connection
        conn = psycopg2.connect(
            settings.supabase.database_url,
            connect_timeout=30
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        logger.info("✅ Connected to PostgreSQL database")
        
        # Read SQL schema file
        schema_file = Path(__file__).parent / "schema.sql"
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        logger.info(f"Reading schema from: {schema_file}")
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Execute the entire schema as one transaction
        cursor = conn.cursor()
        
        try:
            logger.info("Executing schema SQL...")
            cursor.execute(schema_sql)
            logger.info("✅ Schema SQL executed successfully")
            
            # Test the schema by querying system tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('documents', 'chunks', 'ingestion_logs')
                ORDER BY table_name;
            """)
            
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            logger.info(f"Created tables: {table_names}")
            
            if len(table_names) == 3:
                logger.info("✅ All required tables created successfully")
                
                # Check for vector extension
                cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                vector_ext = cursor.fetchone()
                
                if vector_ext:
                    logger.info("✅ Vector extension is available")
                else:
                    logger.warning("⚠️ Vector extension not found - vector search may not work")
                
                print("\n" + "="*80)
                print("DATABASE SCHEMA SETUP COMPLETED SUCCESSFULLY")
                print("="*80)
                print("✅ All required tables created:")
                for table in table_names:
                    print(f"   - {table}")
                print("✅ All indexes and constraints applied")
                print("✅ Vector search functionality enabled" if vector_ext else "⚠️ Vector extension needs manual setup")
                print("✅ Schema validation passed")
                print("\n🎯 Next Steps:")
                print("   1. Run the local ZIP ingestion test again")
                print("   2. Documents should now be stored successfully in Supabase")
                print("   3. Proceed with full pipeline processing")
                return True
            else:
                logger.error(f"❌ Expected 3 tables, found {len(table_names)}: {table_names}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute schema SQL: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    except Exception as e:
        logger.error(f"Schema setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function"""
    success = await setup_schema_direct()
    
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