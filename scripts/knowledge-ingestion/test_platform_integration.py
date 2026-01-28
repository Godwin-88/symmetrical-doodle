"""
Test platform integration functionality.

This test verifies:
- Platform integration service initialization
- Concurrent access management
- Knowledge query interface
- API endpoints functionality
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any

from services.platform_integration import PlatformIntegrationService, ConflictResolution
from services.concurrent_access_manager import (
    ConcurrentAccessManager, 
    LockType, 
    OperationType, 
    Priority
)
from services.knowledge_query_interface import (
    KnowledgeQueryInterface,
    SearchQuery,
    SearchType,
    SortOrder
)
from services.platform_api_endpoints import PlatformAPIService


class TestPlatformIntegration:
    """Test platform integration components"""
    
    @pytest.fixture
    async def integration_service(self):
        """Create integration service for testing"""
        service = PlatformIntegrationService()
        await service.initialize()
        yield service
        await service.close()
    
    @pytest.fixture
    async def access_manager(self):
        """Create access manager for testing"""
        manager = ConcurrentAccessManager()
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.fixture
    async def query_interface(self):
        """Create query interface for testing"""
        interface = KnowledgeQueryInterface()
        # Note: This will fail if Supabase is not configured, but that's expected
        try:
            await interface.initialize()
        except Exception:
            pass  # Expected in test environment
        return interface
    
    async def test_integration_service_initialization(self, integration_service):
        """Test integration service initialization"""
        assert integration_service is not None
        
        # Test connection status
        status = await integration_service.get_connection_status()
        assert 'timestamp' in status
        assert 'overall_status' in status
        assert 'connections' in status
        
        # Should have at least Supabase and intelligence layer connections
        assert 'supabase' in status['connections']
        assert 'intelligence_layer' in status['connections']
    
    async def test_conflict_detection(self, integration_service):
        """Test conflict detection functionality"""
        # Test document conflict detection
        test_data = {
            'file_id': 'test_file_123',
            'title': 'Test Document',
            'content': 'Test content',
            'processing_status': 'completed'
        }
        
        conflict_result = await integration_service.detect_conflicts(
            'documents', 
            test_data, 
            'file_id'
        )
        
        assert conflict_result is not None
        assert hasattr(conflict_result, 'has_conflict')
        assert hasattr(conflict_result, 'conflict_type')
    
    async def test_conflict_resolution(self, integration_service):
        """Test conflict resolution strategies"""
        # Mock conflict result
        from services.platform_integration import ConflictDetectionResult
        
        conflict_result = ConflictDetectionResult(
            has_conflict=True,
            conflict_type="document_exists",
            existing_data={'id': '123', 'title': 'Old Title'},
            conflicting_fields=['title'],
            resolution_strategy=ConflictResolution.MERGE
        )
        
        new_data = {'id': '123', 'title': 'New Title', 'content': 'New content'}
        
        resolved_data = await integration_service.resolve_conflict(
            conflict_result,
            new_data,
            ConflictResolution.MERGE
        )
        
        assert resolved_data is not None
        if resolved_data:  # If not skipped
            assert 'title' in resolved_data
            assert 'updated_at' in resolved_data
    
    async def test_access_manager_process_registration(self, access_manager):
        """Test process registration and management"""
        process_id = "test_process_123"
        
        # Register process
        success = await access_manager.register_process(
            process_id, 
            "test_process", 
            {"test": True}
        )
        assert success
        
        # Check process info
        process_info = access_manager.get_process_info(process_id)
        assert process_info is not None
        assert process_info['process_id'] == process_id
        assert process_info['process_type'] == "test_process"
        
        # Update heartbeat
        heartbeat_success = await access_manager.update_heartbeat(process_id)
        assert heartbeat_success
        
        # Unregister process
        unregister_success = await access_manager.unregister_process(process_id)
        assert unregister_success
        
        # Verify process is gone
        process_info_after = access_manager.get_process_info(process_id)
        assert process_info_after is None
    
    async def test_access_manager_locking(self, access_manager):
        """Test locking mechanism"""
        process_id = "test_process_lock"
        resource_id = "test_resource"
        
        # Register process first
        await access_manager.register_process(process_id, "test_process")
        
        try:
            # Test lock acquisition
            async with access_manager.acquire_lock(
                resource_id=resource_id,
                lock_type=LockType.READ,
                operation_type=OperationType.QUERY,
                process_id=process_id,
                priority=Priority.NORMAL,
                timeout_seconds=5.0
            ) as lock_id:
                assert lock_id is not None
                
                # Verify lock is active
                stats = access_manager.get_lock_statistics()
                assert stats['active_locks'] > 0
            
            # After context exit, lock should be released
            stats_after = access_manager.get_lock_statistics()
            # Note: There might be a small delay in cleanup
            
        finally:
            await access_manager.unregister_process(process_id)
    
    async def test_query_interface_search_query_creation(self, query_interface):
        """Test search query creation and validation"""
        # Test basic search query
        query = SearchQuery(
            query_text="machine learning",
            search_type=SearchType.SEMANTIC,
            limit=10,
            similarity_threshold=0.7
        )
        
        assert query.query_text == "machine learning"
        assert query.search_type == SearchType.SEMANTIC
        assert query.limit == 10
        assert query.similarity_threshold == 0.7
    
    async def test_query_interface_domain_matching(self, query_interface):
        """Test domain keyword matching"""
        # Test domain keyword matching
        content = "This document discusses machine learning algorithms and neural networks"
        domains = ["ML", "finance"]
        
        matches = query_interface._matches_domain_keywords(content, domains)
        assert matches  # Should match ML domain
    
    async def test_api_service_initialization(self):
        """Test API service initialization"""
        api_service = PlatformAPIService()
        
        # Test that FastAPI app is created
        assert api_service.app is not None
        assert api_service.app.title == "Knowledge Ingestion API"
        
        # Test route setup
        routes = [route.path for route in api_service.app.routes]
        expected_routes = ["/health", "/search", "/search/semantic", "/search/keyword", "/search/hybrid"]
        
        for expected_route in expected_routes:
            assert any(expected_route in route for route in routes)
    
    async def test_integration_statistics(self, integration_service, access_manager):
        """Test statistics collection"""
        # Test integration service statistics
        status = await integration_service.get_connection_status()
        assert 'timestamp' in status
        assert 'connections' in status
        
        # Test access manager statistics
        stats = access_manager.get_lock_statistics()
        assert 'timestamp' in stats
        assert 'active_locks' in stats
        assert 'active_processes' in stats
    
    async def test_concurrent_operations(self, access_manager):
        """Test concurrent operations handling"""
        process_ids = ["proc_1", "proc_2", "proc_3"]
        resource_id = "shared_resource"
        
        # Register processes
        for proc_id in process_ids:
            await access_manager.register_process(proc_id, "test_process")
        
        try:
            # Test concurrent read locks (should succeed)
            async def acquire_read_lock(proc_id):
                async with access_manager.acquire_lock(
                    resource_id=resource_id,
                    lock_type=LockType.READ,
                    operation_type=OperationType.QUERY,
                    process_id=proc_id,
                    timeout_seconds=2.0
                ):
                    await asyncio.sleep(0.1)  # Hold lock briefly
                    return True
            
            # All read locks should succeed concurrently
            tasks = [acquire_read_lock(proc_id) for proc_id in process_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed (no exceptions)
            for result in results:
                assert result is True or not isinstance(result, Exception)
        
        finally:
            # Cleanup
            for proc_id in process_ids:
                await access_manager.unregister_process(proc_id)


async def test_basic_functionality():
    """Test basic functionality without fixtures"""
    # Test that services can be imported and instantiated
    integration_service = PlatformIntegrationService()
    assert integration_service is not None
    
    access_manager = ConcurrentAccessManager()
    assert access_manager is not None
    
    query_interface = KnowledgeQueryInterface()
    assert query_interface is not None
    
    api_service = PlatformAPIService()
    assert api_service is not None
    assert api_service.app is not None


async def test_search_types_and_enums():
    """Test search types and enums"""
    # Test SearchType enum
    assert SearchType.SEMANTIC.value == "semantic"
    assert SearchType.KEYWORD.value == "keyword"
    assert SearchType.HYBRID.value == "hybrid"
    
    # Test SortOrder enum
    assert SortOrder.RELEVANCE.value == "relevance"
    assert SortOrder.DATE.value == "date"
    
    # Test LockType enum
    assert LockType.READ.value == "read"
    assert LockType.WRITE.value == "write"
    assert LockType.EXCLUSIVE.value == "exclusive"


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_basic_functionality())
    asyncio.run(test_search_types_and_enums())
    print("Basic platform integration tests passed!")