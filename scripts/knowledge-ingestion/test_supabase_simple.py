"""
Simple test for Supabase storage services without full configuration.

This test checks if the services can be imported and basic functionality works.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Supabase services can be imported"""
    print("Testing Supabase service imports...")
    
    try:
        from services.supabase_schema import (
            SupabaseSchemaManager, SchemaValidationResult, MigrationResult
        )
        print("✓ Schema management services imported successfully")
        
        from services.supabase_storage import (
            SupabaseStorageService, TransactionManager,
            DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry,
            StorageResult, TransactionResult,
            ProcessingStatus, IngestionPhase, IngestionStatus
        )
        print("✓ Storage services imported successfully")
        
        # Test data model creation
        doc = DocumentMetadata(
            file_id="test_123",
            title="Test Document",
            quality_score=0.95
        )
        print(f"✓ DocumentMetadata created: {doc.file_id}")
        
        chunk = ChunkData(
            document_id="doc_123",
            content="Test chunk content",
            chunk_order=0,
            token_count=4
        )
        print(f"✓ ChunkData created: order {chunk.chunk_order}")
        
        embedded_chunk = EmbeddedChunk(
            chunk_data=chunk,
            embedding_vector=[0.1, 0.2, 0.3]
        )
        print(f"✓ EmbeddedChunk created with {len(embedded_chunk.embedding_vector)} dimensions")
        
        log_entry = IngestionLogEntry(
            file_id="test_123",
            phase=IngestionPhase.STORAGE,
            status=IngestionStatus.COMPLETED
        )
        print(f"✓ IngestionLogEntry created: {log_entry.phase.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_data_serialization():
    """Test data model serialization"""
    print("\nTesting data model serialization...")
    
    try:
        from services.supabase_storage import (
            DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry,
            ProcessingStatus, IngestionPhase, IngestionStatus
        )
        
        # Test document serialization
        doc = DocumentMetadata(
            file_id="test_456",
            title="Serialization Test",
            processing_status=ProcessingStatus.COMPLETED,
            quality_score=0.85,
            domain_classification="ML"
        )
        
        doc_dict = doc.to_dict()
        assert doc_dict['file_id'] == "test_456"
        assert doc_dict['processing_status'] == "completed"
        print("✓ Document serialization works")
        
        # Test chunk serialization
        chunk = ChunkData(
            document_id="doc_456",
            content="Serialization test content",
            chunk_order=1,
            embedding_model="test-model",
            quality_score=0.9
        )
        
        chunk_dict = chunk.to_dict()
        assert chunk_dict['document_id'] == "doc_456"
        assert chunk_dict['chunk_order'] == 1
        print("✓ Chunk serialization works")
        
        # Test embedded chunk serialization
        embedded = EmbeddedChunk(
            chunk_data=chunk,
            embedding_vector=[0.1, 0.2, 0.3, 0.4]
        )
        
        embedded_dict = embedded.to_dict()
        assert 'embedding' in embedded_dict
        assert len(embedded_dict['embedding']) == 4
        print("✓ Embedded chunk serialization works")
        
        # Test log entry serialization
        log = IngestionLogEntry(
            file_id="test_456",
            phase=IngestionPhase.EMBEDDING,
            status=IngestionStatus.FAILED,
            error_message="Test error",
            processing_time_ms=1500
        )
        
        log_dict = log.to_dict()
        assert log_dict['phase'] == "embedding"
        assert log_dict['status'] == "failed"
        print("✓ Log entry serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Serialization test failed: {e}")
        return False


def test_enum_values():
    """Test enum value consistency"""
    print("\nTesting enum values...")
    
    try:
        from services.supabase_storage import ProcessingStatus, IngestionPhase, IngestionStatus
        
        # Test ProcessingStatus
        expected_statuses = ["pending", "processing", "completed", "failed", "skipped"]
        actual_statuses = [status.value for status in ProcessingStatus]
        
        for status in expected_statuses:
            assert status in actual_statuses, f"Missing status: {status}"
        print(f"✓ ProcessingStatus has all expected values: {actual_statuses}")
        
        # Test IngestionPhase
        expected_phases = ["discovery", "download", "parsing", "chunking", "embedding", "storage", "audit"]
        actual_phases = [phase.value for phase in IngestionPhase]
        
        for phase in expected_phases:
            assert phase in actual_phases, f"Missing phase: {phase}"
        print(f"✓ IngestionPhase has all expected values: {actual_phases}")
        
        # Test IngestionStatus
        expected_ing_statuses = ["started", "completed", "failed", "skipped", "retrying"]
        actual_ing_statuses = [status.value for status in IngestionStatus]
        
        for status in expected_ing_statuses:
            assert status in actual_ing_statuses, f"Missing ingestion status: {status}"
        print(f"✓ IngestionStatus has all expected values: {actual_ing_statuses}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enum test failed: {e}")
        return False


def main():
    """Run all basic tests"""
    print("Running basic Supabase storage tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Serialization Test", test_data_serialization),
        ("Enum Values Test", test_enum_values)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*40}")
        print(f"Running {test_name}")
        print(f"{'='*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print(f"{'='*40}")
    print("Test Summary")
    print(f"{'='*40}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\nThe Supabase storage services are ready for use.")
        print("To test with a real database, configure SUPABASE_URL and SUPABASE_KEY environment variables.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")


if __name__ == "__main__":
    main()