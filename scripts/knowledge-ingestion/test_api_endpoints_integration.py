#!/usr/bin/env python3
"""
Integration test for unified browsing API endpoints

This script tests the FastAPI endpoints to ensure they respond correctly
and handle requests properly.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import httpx
import pytest

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.multi_source_api_endpoints import MultiSourceAuthAPI
from core.logging import get_logger


class APIEndpointTester:
    """Test class for API endpoint integration"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.api_service = None
        self.base_url = "http://localhost:8001"
    
    async def setup(self):
        """Setup test environment"""
        try:
            self.logger.info("Setting up API endpoint integration test")
            
            # Initialize API service
            self.api_service = MultiSourceAuthAPI()
            await self.api_service.initialize()
            
            self.logger.info("API endpoint test environment setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        self.logger.info("Testing health endpoint")
        
        try:
            # Test health endpoint directly on the app
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            assert "service" in data
            assert "timestamp" in data
            
            self.logger.info("✓ Health endpoint test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Health endpoint test failed: {e}")
            return False
    
    async def test_file_listing_endpoint(self):
        """Test file listing endpoint"""
        self.logger.info("Testing file listing endpoint")
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            # Test file listing with empty request
            request_data = {
                "connection_ids": [],
                "source_types": [],
                "folder_path": None,
                "include_subfolders": True,
                "file_types": ["application/pdf"],
                "limit": 10,
                "offset": 0
            }
            
            response = client.post("/browse/files", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "files" in data
            assert "total_files" in data
            assert "execution_time_ms" in data
            assert "pagination" in data
            
            # Should return empty list for no connections
            assert isinstance(data["files"], list)
            assert data["total_files"] == 0
            
            self.logger.info("✓ File listing endpoint test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ File listing endpoint test failed: {e}")
            return False
    
    async def test_file_tree_endpoint(self):
        """Test file tree endpoint"""
        self.logger.info("Testing file tree endpoint")
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            # Test file tree endpoint
            response = client.get("/browse/tree?max_depth=3")
            assert response.status_code == 200
            
            data = response.json()
            assert "sources" in data
            assert "total_files" in data
            assert "total_sources" in data
            assert "execution_time_ms" in data
            assert "last_updated" in data
            
            # Should return empty list for no connections
            assert isinstance(data["sources"], list)
            assert data["total_sources"] == 0
            
            self.logger.info("✓ File tree endpoint test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ File tree endpoint test failed: {e}")
            return False
    
    async def test_search_endpoint(self):
        """Test search endpoint"""
        self.logger.info("Testing search endpoint")
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            # Test search endpoint
            request_data = {
                "query": "test",
                "connection_ids": [],
                "source_types": [],
                "search_fields": ["name", "content"],
                "file_types": ["application/pdf"],
                "limit": 10,
                "offset": 0
            }
            
            response = client.post("/browse/search", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "results" in data
            assert "total_results" in data
            assert "query" in data
            assert "search_time_ms" in data
            assert "sources_searched" in data
            
            # Should return empty results for no connections
            assert isinstance(data["results"], list)
            assert data["total_results"] == 0
            assert data["query"] == "test"
            
            self.logger.info("✓ Search endpoint test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Search endpoint test failed: {e}")
            return False
    
    async def test_cache_endpoints(self):
        """Test cache management endpoints"""
        self.logger.info("Testing cache endpoints")
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            # Test cache stats endpoint
            response = client.get("/browse/cache/stats")
            assert response.status_code == 200
            
            data = response.json()
            assert "total_entries" in data
            assert "hit_rate" in data
            assert "memory_usage_mb" in data
            assert "oldest_entry_age_seconds" in data
            
            # Test cache invalidation endpoint
            response = client.post("/browse/cache/invalidate", json=None)
            assert response.status_code == 200
            
            data = response.json()
            assert "success" in data
            assert data["success"] is True
            assert "message" in data
            assert "timestamp" in data
            
            self.logger.info("✓ Cache endpoints test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Cache endpoints test failed: {e}")
            return False
    
    async def test_source_capabilities_endpoint(self):
        """Test source capabilities endpoint"""
        self.logger.info("Testing source capabilities endpoint")
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            # Test list all source types
            response = client.get("/sources")
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
            
            # Should have entries for all supported source types
            source_types = [item["source_type"] for item in data]
            expected_types = [
                "google_drive", "local_zip", "local_directory", 
                "individual_upload", "aws_s3", "azure_blob", "google_cloud_storage"
            ]
            
            for expected_type in expected_types:
                assert expected_type in source_types
            
            self.logger.info("✓ Source capabilities endpoint test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Source capabilities endpoint test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling in endpoints"""
        self.logger.info("Testing error handling in endpoints")
        
        try:
            from fastapi.testclient import TestClient
            client = TestClient(self.api_service.app)
            
            # Test invalid file metadata request
            response = client.get("/browse/files/invalid_file_id/metadata?connection_id=invalid_connection")
            assert response.status_code == 500  # Should return error for invalid connection
            
            # Test invalid access validation request
            request_data = {
                "file_ids": ["invalid_file"],
                "connection_id": "invalid_connection"
            }
            
            response = client.post("/browse/validate-access", json=request_data)
            assert response.status_code == 500  # Should return error for invalid connection
            
            self.logger.info("✓ Error handling test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error handling test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        self.logger.info("Starting API endpoint integration tests")
        
        tests = [
            self.test_health_endpoint,
            self.test_file_listing_endpoint,
            self.test_file_tree_endpoint,
            self.test_search_endpoint,
            self.test_cache_endpoints,
            self.test_source_capabilities_endpoint,
            self.test_error_handling
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"Test {test.__name__} failed with exception: {e}")
                failed += 1
        
        self.logger.info(f"Integration test results: {passed} passed, {failed} failed")
        
        if failed == 0:
            self.logger.info("🎉 All API endpoint integration tests passed!")
            return True
        else:
            self.logger.error(f"❌ {failed} integration tests failed")
            return False


async def main():
    """Main test function"""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = get_logger(__name__)
    
    logger.info("Starting API endpoint integration test suite")
    
    # Create and run tester
    tester = APIEndpointTester()
    
    # Setup test environment
    if not await tester.setup():
        logger.error("Failed to setup test environment")
        return False
    
    # Run all tests
    success = await tester.run_all_tests()
    
    if success:
        logger.info("✅ API endpoint integration test suite completed successfully")
        return True
    else:
        logger.error("❌ API endpoint integration test suite failed")
        return False


if __name__ == "__main__":
    import sys
    
    # Run the test
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)