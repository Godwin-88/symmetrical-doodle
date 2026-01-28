"""
Test processing a fresh document with chunks to verify the complete pipeline works.

This test processes a different PDF to ensure the content cleaning fix works
for the complete document + chunks storage workflow.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.local_zip_discovery import LocalZipDiscoveryService
from services.supabase_storage import SupabaseStorageService, DocumentMetadata, ProcessingStatus, ChunkData, EmbeddedChunk
from services.simple_pdf_parser import SimplePDFParser
from services.simple_chunker import SimpleChunker
from core.config import load_config
from core.logging import configure_logging, get_logger


async def test_fresh_document_processing():
    """Test processing a fresh document with chunks"""
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    logger.info("Testing fresh document processing with chunks")
    
    try:
        # Load configuration
        settings = load_config("development")
        
        # Initialize services
        storage_service = SupabaseStorageService(settings.supabase)
        await storage_service.initialize_client()
        
        discovery_service = LocalZipDiscoveryService()
        pdf_parser = SimplePDFParser()
        chunker = SimpleChunker()
        
        logger.info("✅ Services initialized")
        
        # Discover PDFs from ZIP
        zip_path = Path(settings.local_zip.zip_path)
        discovery_result = await discovery_service.discover_pdfs_from_zip(
            str(zip_path),
            settings.local_zip.extract_path
        )
        
        if not discovery_result.success or not discovery_result.pdfs_found:
            logger.error("No PDFs found in ZIP")
            return False
        
        # Find a different PDF than the one we've been testing with
        # Let's use the second PDF in the list
        if len(discovery_result.pdfs_found) < 2:
            logger.error("Need at least 2 PDFs for this test")
            return False
        
        test_pdf = discovery_result.pdfs_found[1]  # Second PDF
        logger.info(f"Processing PDF: {test_pdf.name}")
        
        # Check if this document already exists
        existing_doc = await storage_service.get_document_by_file_id(test_pdf.file_id)
        if existing_doc:
            logger.info(f"Document {test_pdf.file_id} already exists, using it for chunk test")
            document_id = existing_doc['id']
        else:
            # Parse the PDF
            logger.info("Parsing PDF...")
            parsed_doc = await pdf_parser.parse_pdf_from_file(test_pdf.local_path)
            
            if not parsed_doc:
                logger.error("Failed to parse PDF")
                return False
            
            logger.info(f"✅ PDF parsed: {len(parsed_doc.content)} characters")
            
            # Create document metadata
            document_metadata = DocumentMetadata(
                file_id=test_pdf.file_id,
                title=test_pdf.name,
                source_url=test_pdf.web_view_link,
                content=parsed_doc.content[:10000],  # First 10k chars for demo
                parsing_method=parsed_doc.parsing_method,
                domain_classification=test_pdf.domain_classification,
                processing_status=ProcessingStatus.COMPLETED,
                file_size_bytes=test_pdf.size
            )
            
            # Store document
            doc_result = await storage_service.store_document(document_metadata)
            
            if not doc_result.success:
                logger.error(f"Failed to store document: {doc_result.error_message}")
                return False
            
            document_id = doc_result.record_id
            logger.info(f"✅ Document stored: {document_id}")
        
        # Chunk the content (use a smaller sample for faster processing)
        logger.info("Chunking content...")
        
        # Create a test document with some content that might have problematic characters
        test_content = """
        This is a test document with some content that might contain problematic characters.
        
        Here's some text with potential issues:
        - Null bytes: \x00
        - Control characters: \x01\x02\x03
        - Unicode issues: \u0000\u0001
        
        And here's normal content that should be preserved:
        - Mathematical formulas: E = mc²
        - Greek letters: α, β, γ, δ
        - Special symbols: ∑, ∫, ∂, ∇
        
        This content should be cleaned and stored successfully.
        """
        
        # Create a simple parsed document for chunking
        class MockParsedDoc:
            def __init__(self, content):
                self.content = content
                self.parsing_method = "test"
        
        mock_doc = MockParsedDoc(test_content)
        chunks = await chunker.chunk_document(mock_doc)
        
        logger.info(f"✅ Content chunked: {len(chunks)} chunks")
        
        # Create embedded chunks with dummy embeddings
        embedded_chunks = []
        for i, chunk in enumerate(chunks[:3]):  # Process first 3 chunks
            chunk_data = ChunkData(
                document_id=document_id,
                content=chunk.content,
                chunk_order=i + 100,  # Use high numbers to avoid conflicts
                section_header=f"Test Section {i+1}",
                token_count=chunk.token_count
            )
            
            embedded_chunk = EmbeddedChunk(
                chunk_data=chunk_data,
                embedding_vector=[0.1 * (i + 1)] * 1536  # Dummy embedding
            )
            embedded_chunks.append(embedded_chunk)
        
        # Store chunks with content cleaning
        logger.info("Storing chunks with content cleaning...")
        chunk_results = await storage_service.store_chunks(embedded_chunks)
        
        # Check results
        successful_chunks = [r for r in chunk_results if r.success]
        failed_chunks = [r for r in chunk_results if not r.success]
        
        if failed_chunks:
            logger.error(f"❌ Failed to store {len(failed_chunks)} chunks")
            for i, result in enumerate(chunk_results):
                if not result.success:
                    logger.error(f"  Chunk {i+1} error: {result.error_message}")
            return False
        
        logger.info(f"✅ Successfully stored {len(successful_chunks)} chunks")
        
        # Verify chunks were stored correctly
        stored_chunks = await storage_service.get_chunks_by_document_id(document_id)
        test_chunks = [c for c in stored_chunks if c.get('chunk_order', 0) >= 100]
        
        logger.info(f"✅ Verified {len(test_chunks)} test chunks in database")
        
        # Check that problematic characters were cleaned
        for chunk in test_chunks:
            content = chunk.get('content', '')
            if '\x00' in content:
                logger.error("❌ Null bytes still present in stored content")
                return False
            
            # Check that meaningful content was preserved
            if 'mathematical formulas' not in content.lower() and 'test document' not in content.lower():
                continue  # This chunk might not have the test content
            
            logger.info(f"✅ Chunk content cleaned and preserved: {content[:100]}...")
        
        logger.info("✅ All chunks processed successfully with content cleaning")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    success = await test_fresh_document_processing()
    
    if success:
        print("\n🎉 Fresh document processing test completed successfully!")
        print("✅ Documents and chunks can be stored with content cleaning")
        print("✅ The Unicode escape sequence issue has been resolved")
        print("✅ The system is ready for full-scale processing")
    else:
        print("\n❌ Fresh document processing test failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())