"""
Test script for Supabase storage services.

This script tests the database schema management and storage services
to ensure they work correctly with the Supabase backend.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import get_settings
from services.supabase_schema import SupabaseSchemaManager
from services.supabase_storage import (
    SupabaseStorageService, TransactionManager,
    DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry,
    ProcessingStatus, IngestionPhase, IngestionStatus
)


async def test_schema_management():
    """Test database schema initialization and validation"""
    print("Testing Supabase schema management...")
    
    try:
        settings = get_settings()
        schema_manager = SupabaseSchemaManager(settings.supabase)
        
        # Test client initialization
        print("1. Testing client initialization...")
        if await schema_manager.initialize_client():
            print("   ✓ Client initialized successfully")
        else:
            print("   ✗ Client initialization failed")
            return False
        
        # Test schema validation
        print("2. Testing schema validation...")
        validation_result = await schema_manager.validate_schema()
        print(f"   Schema valid: {validation_result.valid}")
        
        if validation_result.missing_tables:
            print(f"   Missing tables: {validation_result.missing_tables}")
        if validation_result.missing_indexes:
            print(f"   Missing indexes: {validation_result.missing_indexes}")
        if validation_result.errors:
            print(f"   Errors: {validation_result.errors}")
        
        # Test schema initialization if needed
        if not validation_result.valid:
            print("3. Testing schema initialization...")
            if await schema_manager.initialize_schema():
                print("   ✓ Schema initialized successfully")
            else:
                print("   ✗ Schema initialization failed")
                return False
        else:
            print("3. Schema already valid, skipping initialization")
        
        # Test migration
        print("4. Testing schema migration...")
        migration_result = await schema_manager.migrate_schema()
        print(f"   Migration successful: {migration_result.success}")
        print(f"   Applied migrations: {migration_result.applied_migrations}")
        print(f"   Execution time: {migration_result.execution_time_ms}ms")
        
        schema_manager.close()
        return True
        
    except Exception as e:
        print(f"Schema management test failed: {e}")
        return False


async def test_storage_services():
    """Test document and chunk storage services"""
    print("\nTesting Supabase storage services...")
    
    try:
        settings = get_settings()
        storage_service = SupabaseStorageService(settings.supabase)
        
        # Test client initialization
        print("1. Testing storage client initialization...")
        if await storage_service.initialize_client():
            print("   ✓ Storage client initialized successfully")
        else:
            print("   ✗ Storage client initialization failed")
            return False
        
        # Test document storage
        print("2. Testing document storage...")
        test_document = DocumentMetadata(
            file_id="test_file_123",
            title="Test Document",
            source_url="https://drive.google.com/file/d/test_file_123",
            content="This is a test document content.",
            parsing_method="marker",
            quality_score=0.95,
            domain_classification="ML",
            file_size_bytes=1024,
            page_count=5
        )
        
        doc_result = await storage_service.store_document(test_document)
        if doc_result.success:
            print(f"   ✓ Document stored with ID: {doc_result.record_id}")
            document_id = doc_result.record_id
        else:
            print(f"   ✗ Document storage failed: {doc_result.error_message}")
            return False
        
        # Test chunk storage
        print("3. Testing chunk storage...")
        test_chunks = [
            EmbeddedChunk(
                chunk_data=ChunkData(
                    document_id=document_id,
                    content="This is the first chunk of the test document.",
                    chunk_order=0,
                    section_header="Introduction",
                    token_count=12,
                    embedding_model="text-embedding-3-large",
                    embedding_dimension=1536,
                    quality_score=0.9
                ),
                embedding_vector=[0.1] * 1536  # Mock embedding vector
            ),
            EmbeddedChunk(
                chunk_data=ChunkData(
                    document_id=document_id,
                    content="This is the second chunk of the test document.",
                    chunk_order=1,
                    section_header="Methods",
                    token_count=13,
                    embedding_model="text-embedding-3-large",
                    embedding_dimension=1536,
                    quality_score=0.85
                ),
                embedding_vector=[0.2] * 1536  # Mock embedding vector
            )
        ]
        
        chunk_results = await storage_service.store_chunks(test_chunks)
        successful_chunks = [r for r in chunk_results if r.success]
        
        if len(successful_chunks) == len(test_chunks):
            print(f"   ✓ All {len(test_chunks)} chunks stored successfully")
        else:
            print(f"   ✗ Only {len(successful_chunks)}/{len(test_chunks)} chunks stored")
        
        # Test ingestion logging
        print("4. Testing ingestion logging...")
        log_entry = IngestionLogEntry(
            file_id="test_file_123",
            phase=IngestionPhase.STORAGE,
            status=IngestionStatus.COMPLETED,
            processing_time_ms=1500,
            metadata={"chunks_stored": len(successful_chunks)}
        )
        
        log_result = await storage_service.log_ingestion_status(log_entry)
        if log_result.success:
            print(f"   ✓ Ingestion log stored with ID: {log_result.record_id}")
        else:
            print(f"   ✗ Ingestion logging failed: {log_result.error_message}")
        
        # Test retrieval operations
        print("5. Testing data retrieval...")
        
        # Retrieve document by file_id
        retrieved_doc = await storage_service.get_document_by_file_id("test_file_123")
        if retrieved_doc:
            print(f"   ✓ Document retrieved: {retrieved_doc['title']}")
        else:
            print("   ✗ Document retrieval failed")
        
        # Retrieve chunks by document_id
        retrieved_chunks = await storage_service.get_chunks_by_document_id(document_id)
        if len(retrieved_chunks) == len(test_chunks):
            print(f"   ✓ All {len(retrieved_chunks)} chunks retrieved")
        else:
            print(f"   ✗ Only {len(retrieved_chunks)}/{len(test_chunks)} chunks retrieved")
        
        # Test storage statistics
        print("6. Testing storage statistics...")
        stats = await storage_service.get_storage_statistics()
        if stats:
            print(f"   ✓ Statistics retrieved: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks")
        else:
            print("   ✗ Statistics retrieval failed")
        
        return True
        
    except Exception as e:
        print(f"Storage services test failed: {e}")
        return False


async def test_transaction_management():
    """Test atomic transaction operations"""
    print("\nTesting transaction management...")
    
    try:
        settings = get_settings()
        storage_service = SupabaseStorageService(settings.supabase)
        transaction_manager = TransactionManager(storage_service)
        
        await storage_service.initialize_client()
        
        # Test atomic document + chunks storage
        print("1. Testing atomic document and chunks storage...")
        
        test_document = DocumentMetadata(
            file_id="test_transaction_456",
            title="Transaction Test Document",
            source_url="https://drive.google.com/file/d/test_transaction_456",
            content="This is a transaction test document.",
            parsing_method="pymupdf",
            quality_score=0.8,
            domain_classification="finance"
        )
        
        test_chunks = [
            EmbeddedChunk(
                chunk_data=ChunkData(
                    document_id="",  # Will be set by transaction
                    content="Transaction test chunk 1",
                    chunk_order=0,
                    token_count=5,
                    embedding_model="bge-large-en-v1.5",
                    embedding_dimension=1024,
                    quality_score=0.8
                ),
                embedding_vector=[0.3] * 1024
            ),
            EmbeddedChunk(
                chunk_data=ChunkData(
                    document_id="",  # Will be set by transaction
                    content="Transaction test chunk 2",
                    chunk_order=1,
                    token_count=5,
                    embedding_model="bge-large-en-v1.5",
                    embedding_dimension=1024,
                    quality_score=0.75
                ),
                embedding_vector=[0.4] * 1024
            )
        ]
        
        transaction_result = await storage_service.store_document_with_chunks(test_document, test_chunks)
        
        if transaction_result.success:
            print(f"   ✓ Transaction completed successfully")
            print(f"     Document ID: {transaction_result.document_id}")
            print(f"     Chunk IDs: {len(transaction_result.chunk_ids)} chunks")
        else:
            print(f"   ✗ Transaction failed: {transaction_result.error_message}")
            if transaction_result.rollback_performed:
                print("     Rollback was performed")
        
        return transaction_result.success
        
    except Exception as e:
        print(f"Transaction management test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("Starting Supabase storage tests...\n")
    
    # Check if configuration is available
    try:
        settings = get_settings()
        if not settings.supabase.url or not settings.supabase.key:
            print("❌ Supabase configuration not found. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
            return
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Run tests
    tests = [
        ("Schema Management", test_schema_management),
        ("Storage Services", test_storage_services),
        ("Transaction Management", test_transaction_management)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Running {test_name} Tests")
        print(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} tests PASSED")
            else:
                print(f"❌ {test_name} tests FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} tests ERROR: {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print(f"{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())