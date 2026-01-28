"""
Test chunk storage with content cleaning fix.

This test specifically verifies that chunks with problematic characters
can be stored successfully after the content cleaning fix.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.supabase_storage import (
    SupabaseStorageService, 
    ChunkData, 
    EmbeddedChunk,
    DocumentMetadata,
    ProcessingStatus
)
from core.config import load_config
from core.logging import configure_logging, get_logger


async def test_chunk_storage_with_problematic_content():
    """Test chunk storage with content that contains null bytes and other problematic characters"""
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    logger.info("Testing chunk storage with problematic content")
    
    try:
        # Load configuration
        settings = load_config("development")
        
        # Initialize Supabase Storage
        storage_service = SupabaseStorageService(settings.supabase)
        
        schema_success = await storage_service.initialize_client()
        if not schema_success:
            logger.error("Failed to initialize Supabase client")
            return False
        
        logger.info("✅ Supabase client initialized")
        
        # Create a test document first (if it doesn't exist)
        test_document = DocumentMetadata(
            file_id="test_chunk_fix_001",
            title="Test Document for Chunk Fix",
            content="Test document for verifying chunk storage fix",
            parsing_method="test",
            domain_classification="Test",
            processing_status=ProcessingStatus.COMPLETED,
            file_size_bytes=1000
        )
        
        # Try to store the document (might fail if it already exists, which is fine)
        doc_result = await storage_service.store_document(test_document)
        
        if doc_result.success:
            document_id = doc_result.record_id
            logger.info(f"✅ Test document created: {document_id}")
        else:
            # Document might already exist, try to get it
            existing_doc = await storage_service.get_document_by_file_id("test_chunk_fix_001")
            if existing_doc:
                document_id = existing_doc['id']
                logger.info(f"✅ Using existing test document: {document_id}")
            else:
                logger.error("Failed to create or find test document")
                return False
        
        # Create test chunks with problematic content
        problematic_chunks = []
        
        # Chunk 1: Contains null bytes
        chunk1_content = "This is a test chunk with null bytes\x00 and other problematic characters."
        chunk1 = ChunkData(
            document_id=document_id,
            content=chunk1_content,
            chunk_order=1,
            section_header="Test Section with \x00 null bytes",
            token_count=50
        )
        
        # Chunk 2: Contains various control characters
        chunk2_content = "This chunk has control chars: \x01\x02\x03\x04\x05 and unicode issues."
        chunk2 = ChunkData(
            document_id=document_id,
            content=chunk2_content,
            chunk_order=2,
            section_header="Control Characters Test",
            token_count=45
        )
        
        # Chunk 3: Contains problematic Unicode sequences
        chunk3_content = "Unicode test: \u0000\u0001\u0002 and normal text."
        chunk3 = ChunkData(
            document_id=document_id,
            content=chunk3_content,
            chunk_order=3,
            section_header="Unicode Test",
            token_count=40
        )
        
        # Create embedded chunks with dummy embeddings
        for i, chunk_data in enumerate([chunk1, chunk2, chunk3], 1):
            embedded_chunk = EmbeddedChunk(
                chunk_data=chunk_data,
                embedding_vector=[0.1 * i] * 1536  # Dummy embedding vector
            )
            problematic_chunks.append(embedded_chunk)
        
        logger.info(f"Created {len(problematic_chunks)} test chunks with problematic content")
        
        # Test the content cleaning by checking what gets cleaned
        for i, chunk in enumerate(problematic_chunks, 1):
            original_content = chunk.chunk_data.content
            cleaned_dict = chunk.to_dict()
            cleaned_content = cleaned_dict['content']
            
            logger.info(f"Chunk {i}:")
            logger.info(f"  Original: {repr(original_content)}")
            logger.info(f"  Cleaned:  {repr(cleaned_content)}")
            logger.info(f"  Null bytes removed: {'\\x00' not in cleaned_content}")
        
        # Store the chunks
        logger.info("Storing chunks with cleaned content...")
        chunk_results = await storage_service.store_chunks(problematic_chunks)
        
        # Check results
        successful_chunks = [r for r in chunk_results if r.success]
        failed_chunks = [r for r in chunk_results if not r.success]
        
        logger.info(f"✅ Successfully stored {len(successful_chunks)} chunks")
        
        if failed_chunks:
            logger.error(f"❌ Failed to store {len(failed_chunks)} chunks")
            for i, result in enumerate(chunk_results):
                if not result.success:
                    logger.error(f"  Chunk {i+1} error: {result.error_message}")
            return False
        
        # Verify the chunks were stored correctly
        logger.info("Verifying stored chunks...")
        stored_chunks = await storage_service.get_chunks_by_document_id(document_id)
        
        # Filter to only our test chunks (by chunk_order)
        test_chunk_orders = [1, 2, 3]
        our_chunks = [c for c in stored_chunks if c.get('chunk_order') in test_chunk_orders]
        
        logger.info(f"✅ Retrieved {len(our_chunks)} stored chunks")
        
        for chunk in our_chunks:
            content = chunk.get('content', '')
            logger.info(f"  Stored chunk {chunk['chunk_order']}: {repr(content[:50])}...")
            
            # Verify no null bytes in stored content
            if '\x00' in content:
                logger.error(f"❌ Null bytes still present in stored chunk {chunk['chunk_order']}")
                return False
        
        logger.info("✅ All chunks stored successfully without problematic characters")
        
        # Clean up test chunks (optional)
        try:
            # Delete the test chunks
            for chunk in our_chunks:
                chunk_id = chunk['id']
                # Note: We don't have a delete_chunk method, so we'll leave them for now
                pass
        except Exception as e:
            logger.warning(f"Could not clean up test chunks: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    success = await test_chunk_storage_with_problematic_content()
    
    if success:
        print("\n🎉 Chunk storage fix test completed successfully!")
        print("✅ Chunks with problematic characters can now be stored in Supabase")
        print("✅ Content cleaning is working correctly")
    else:
        print("\n❌ Chunk storage fix test failed")
        print("❌ There are still issues with storing chunks containing problematic characters")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())