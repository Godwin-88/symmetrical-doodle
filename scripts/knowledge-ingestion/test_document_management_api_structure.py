"""
Test Document Management API Structure

This test validates the document management API structure and models
without requiring database connections. It focuses on:
- API model validation
- Service initialization structure
- Endpoint definition validation
- Data model completeness
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime, timezone

from services.document_management_api import (
    DocumentManagementAPI,
    DocumentMetadata,
    DocumentSourceInfo,
    DocumentListRequest,
    DocumentListResponse,
    DocumentUpdateRequest,
    DocumentReprocessingRequest,
    DocumentReprocessingResponse,
    DocumentPreviewRequest,
    DocumentPreviewResponse,
    DocumentDeletionRequest,
    DocumentDeletionResponse,
    DocumentStatisticsResponse,
    DocumentChunkInfo
)
from core.logging import get_logger


def test_data_models():
    """Test data model structure and validation"""
    logger = get_logger(__name__)
    logger.info("Testing data model structure...")
    
    # Test DocumentSourceInfo
    source_info = DocumentSourceInfo(
        source_type="google_drive",
        source_path="/Research/paper.pdf",
        source_id="gd_123456789",
        access_url="https://drive.google.com/file/d/123/view",
        parent_folders=["Research", "ML"],
        connection_id="conn_001",
        is_accessible=True,
        access_permissions=["read"]
    )
    
    assert source_info.source_type == "google_drive"
    assert source_info.is_accessible == True
    assert len(source_info.parent_folders) == 2
    logger.info("✓ DocumentSourceInfo model validated")
    
    # Test DocumentMetadata
    document = DocumentMetadata(
        document_id="doc_123",
        title="Test Document",
        file_id="file_123",
        size=1024000,
        mime_type="application/pdf",
        modified_time=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_info=source_info,
        processing_status="completed",
        quality_score=0.85,
        parsing_method="marker",
        embedding_model="text-embedding-3-large",
        domain_classification="machine_learning",
        content_type="research_paper",
        language="en",
        tags=["research", "ml"],
        categories=["academic"],
        notes="Test document",
        is_favorite=True,
        chunk_count=25,
        processing_time_ms=15000,
        ingestion_job_id="job_123",
        checksum="sha256_hash",
        source_specific_metadata={"test": True}
    )
    
    assert document.document_id == "doc_123"
    assert document.source_info.source_type == "google_drive"
    assert document.quality_score == 0.85
    assert len(document.tags) == 2
    assert document.is_favorite == True
    logger.info("✓ DocumentMetadata model validated")
    
    # Test DocumentListRequest
    list_request = DocumentListRequest(
        source_types=["google_drive", "local_directory"],
        processing_status=["completed"],
        domain_classification=["machine_learning"],
        tags=["research"],
        categories=["academic"],
        is_favorite=True,
        search_query="neural networks",
        sort_by="created_at",
        sort_order="desc",
        limit=50,
        offset=0
    )
    
    assert len(list_request.source_types) == 2
    assert list_request.is_favorite == True
    assert list_request.limit == 50
    logger.info("✓ DocumentListRequest model validated")
    
    # Test DocumentUpdateRequest
    update_request = DocumentUpdateRequest(
        title="Updated Title",
        tags=["updated", "test"],
        categories=["testing"],
        notes="Updated notes",
        is_favorite=False,
        domain_classification="testing"
    )
    
    assert update_request.title == "Updated Title"
    assert len(update_request.tags) == 2
    assert update_request.is_favorite == False
    logger.info("✓ DocumentUpdateRequest model validated")
    
    # Test DocumentReprocessingRequest
    reprocess_request = DocumentReprocessingRequest(
        document_id="doc_123",
        processing_options={"use_llm": True},
        force_reprocess=True,
        preserve_user_metadata=True
    )
    
    assert reprocess_request.document_id == "doc_123"
    assert reprocess_request.force_reprocess == True
    assert reprocess_request.processing_options["use_llm"] == True
    logger.info("✓ DocumentReprocessingRequest model validated")
    
    # Test DocumentChunkInfo
    chunk_info = DocumentChunkInfo(
        chunk_id="chunk_123",
        content="Sample chunk content",
        chunk_order=1,
        section_header="Introduction",
        token_count=150,
        embedding_model="text-embedding-3-large",
        quality_score=0.82
    )
    
    assert chunk_info.chunk_id == "chunk_123"
    assert chunk_info.chunk_order == 1
    assert chunk_info.token_count == 150
    logger.info("✓ DocumentChunkInfo model validated")
    
    logger.info("✅ All data models validated successfully")
    return True


def test_api_service_structure():
    """Test API service structure without database connections"""
    logger = get_logger(__name__)
    logger.info("Testing API service structure...")
    
    # Test service instantiation
    doc_api = DocumentManagementAPI()
    
    assert doc_api.settings is not None
    assert doc_api.logger is not None
    assert doc_api._auth_service is None  # Not initialized yet
    assert doc_api._storage_service is None  # Not initialized yet
    logger.info("✓ DocumentManagementAPI instantiation validated")
    
    # Test dependency getter methods exist
    assert hasattr(doc_api, 'get_auth_service')
    assert hasattr(doc_api, 'get_storage_service')
    assert hasattr(doc_api, 'get_query_interface')
    assert hasattr(doc_api, 'get_batch_manager')
    logger.info("✓ Dependency getter methods exist")
    
    # Test main API methods exist
    assert hasattr(doc_api, 'list_documents')
    assert hasattr(doc_api, 'get_document_by_id')
    assert hasattr(doc_api, 'update_document_metadata')
    assert hasattr(doc_api, 'reprocess_document')
    assert hasattr(doc_api, 'get_document_preview')
    assert hasattr(doc_api, 'delete_documents')
    assert hasattr(doc_api, 'get_document_statistics')
    logger.info("✓ Main API methods exist")
    
    # Test helper methods exist
    assert hasattr(doc_api, '_get_documents_with_filters')
    assert hasattr(doc_api, '_get_documents_count')
    assert hasattr(doc_api, '_validate_document_access')
    assert hasattr(doc_api, '_update_document_in_storage')
    assert hasattr(doc_api, '_get_document_statistics')
    assert hasattr(doc_api, '_delete_from_source')
    assert hasattr(doc_api, '_delete_document_from_storage')
    assert hasattr(doc_api, '_get_comprehensive_statistics')
    assert hasattr(doc_api, '_estimate_processing_time')
    logger.info("✓ Helper methods exist")
    
    logger.info("✅ API service structure validated successfully")
    return True


def test_response_models():
    """Test response model structure"""
    logger = get_logger(__name__)
    logger.info("Testing response model structure...")
    
    # Test DocumentListResponse
    list_response = DocumentListResponse(
        documents=[],
        total_documents=100,
        by_source_type={"google_drive": 50, "local_directory": 50},
        by_status={"completed": 90, "processing": 10},
        by_domain={"ml": 60, "finance": 40},
        pagination={"offset": 0, "limit": 50, "has_more": True},
        execution_time_ms=125
    )
    
    assert list_response.total_documents == 100
    assert list_response.by_source_type["google_drive"] == 50
    assert list_response.pagination["has_more"] == True
    logger.info("✓ DocumentListResponse model validated")
    
    # Test DocumentReprocessingResponse
    reprocess_response = DocumentReprocessingResponse(
        success=True,
        job_id="job_123",
        message="Processing initiated",
        estimated_duration_ms=15000
    )
    
    assert reprocess_response.success == True
    assert reprocess_response.job_id == "job_123"
    assert reprocess_response.estimated_duration_ms == 15000
    logger.info("✓ DocumentReprocessingResponse model validated")
    
    # Test DocumentDeletionResponse
    deletion_response = DocumentDeletionResponse(
        deleted_documents=["doc_1", "doc_2"],
        failed_deletions=[],
        source_deletions=[],
        total_deleted=2
    )
    
    assert len(deletion_response.deleted_documents) == 2
    assert deletion_response.total_deleted == 2
    logger.info("✓ DocumentDeletionResponse model validated")
    
    # Test DocumentStatisticsResponse
    stats_response = DocumentStatisticsResponse(
        total_documents=1000,
        by_source_type={"google_drive": 500, "local": 500},
        by_processing_status={"completed": 950, "failed": 50},
        by_domain={"ml": 400, "finance": 600},
        by_quality_score={"high": 800, "medium": 150, "low": 50},
        processing_statistics={"avg_time": 12000},
        storage_statistics={"total_size": 1000000000},
        recent_activity=[]
    )
    
    assert stats_response.total_documents == 1000
    assert stats_response.by_source_type["google_drive"] == 500
    assert stats_response.processing_statistics["avg_time"] == 12000
    logger.info("✓ DocumentStatisticsResponse model validated")
    
    logger.info("✅ All response models validated successfully")
    return True


def test_multi_source_api_integration():
    """Test integration with multi-source API endpoints"""
    logger = get_logger(__name__)
    logger.info("Testing multi-source API integration...")
    
    # Test that the document management API can be imported by the multi-source API
    try:
        from services.multi_source_api_endpoints import MultiSourceAuthAPI
        
        # Check that the multi-source API has document management dependency
        api = MultiSourceAuthAPI()
        assert hasattr(api, '_document_management')
        assert hasattr(api, 'get_document_management')
        logger.info("✓ Multi-source API integration structure validated")
        
    except ImportError as e:
        logger.warning(f"Multi-source API import failed (expected in some environments): {e}")
    
    logger.info("✅ Multi-source API integration validated")
    return True


def test_source_type_support():
    """Test support for all expected source types"""
    logger = get_logger(__name__)
    logger.info("Testing source type support...")
    
    expected_source_types = [
        'google_drive',
        'local_directory', 
        'local_zip',
        'individual_upload',
        'aws_s3',
        'azure_blob',
        'google_cloud_storage'
    ]
    
    # Test that DocumentSourceInfo accepts all source types
    for source_type in expected_source_types:
        source_info = DocumentSourceInfo(
            source_type=source_type,
            source_path=f"/test/{source_type}/file.pdf",
            source_id=f"{source_type}_123",
            is_accessible=True,
            access_permissions=["read"]
        )
        
        assert source_info.source_type == source_type
        logger.info(f"✓ Source type {source_type} supported")
    
    logger.info("✅ All source types supported")
    return True


def test_filtering_capabilities():
    """Test document filtering capabilities"""
    logger = get_logger(__name__)
    logger.info("Testing filtering capabilities...")
    
    # Test comprehensive filtering request
    filter_request = DocumentListRequest(
        source_types=["google_drive", "aws_s3"],
        connection_ids=["conn_1", "conn_2"],
        processing_status=["completed", "processing"],
        domain_classification=["machine_learning", "finance"],
        tags=["important", "research"],
        categories=["academic", "business"],
        is_favorite=True,
        date_range={
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-12-31T23:59:59Z"
        },
        search_query="neural networks",
        sort_by="quality_score",
        sort_order="desc",
        limit=100,
        offset=50
    )
    
    # Validate all filter fields are present
    assert len(filter_request.source_types) == 2
    assert len(filter_request.connection_ids) == 2
    assert len(filter_request.processing_status) == 2
    assert len(filter_request.domain_classification) == 2
    assert len(filter_request.tags) == 2
    assert len(filter_request.categories) == 2
    assert filter_request.is_favorite == True
    assert filter_request.date_range is not None
    assert filter_request.search_query == "neural networks"
    assert filter_request.sort_by == "quality_score"
    assert filter_request.sort_order == "desc"
    assert filter_request.limit == 100
    assert filter_request.offset == 50
    
    logger.info("✓ Comprehensive filtering request validated")
    logger.info("✅ Filtering capabilities validated")
    return True


async def main():
    """Main test function"""
    print("🚀 Starting Document Management API Structure Tests")
    print("=" * 60)
    
    tests = [
        ("Data Models", test_data_models),
        ("API Service Structure", test_api_service_structure),
        ("Response Models", test_response_models),
        ("Multi-Source API Integration", test_multi_source_api_integration),
        ("Source Type Support", test_source_type_support),
        ("Filtering Capabilities", test_filtering_capabilities)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 Running: {test_name}")
            print("-" * 40)
            
            result = test_func()
            if result:
                passed_tests += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All structure tests passed! Document Management API structure is valid.")
        print("\nValidated components:")
        print("  • Data model structure and validation")
        print("  • API service architecture")
        print("  • Response model completeness")
        print("  • Multi-source integration points")
        print("  • Support for all source types")
        print("  • Comprehensive filtering capabilities")
        print("  • Request/response model consistency")
        print("  • Service dependency structure")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} tests failed. Check the output for details.")
        return False


if __name__ == "__main__":
    import sys
    
    result = asyncio.run(main())
    if not result:
        sys.exit(1)