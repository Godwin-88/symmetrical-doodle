#!/usr/bin/env python3
"""
Test script for unified source browsing API endpoints

This script tests the new unified browsing functionality including:
- File listing across sources
- Hierarchical file tree navigation
- Cross-source search
- File access validation
- Metadata retrieval
- Caching functionality
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.multi_source_api_endpoints import MultiSourceAuthAPI
from services.unified_browsing_service import UnifiedBrowsingService, get_browsing_service
from services.multi_source_auth import get_auth_service, DataSourceType
from core.logging import get_logger
from core.config import get_settings


class UnifiedBrowsingAPITester:
    """Test class for unified browsing API functionality"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.api_service = None
        self.browsing_service = None
        self.auth_service = None
    
    async def setup(self):
        """Setup test environment"""
        try:
            self.logger.info("Setting up unified browsing API test environment")
            
            # Initialize services
            self.auth_service = get_auth_service()
            self.browsing_service = get_browsing_service()
            await self.browsing_service.initialize()
            
            # Initialize API service
            self.api_service = MultiSourceAuthAPI()
            await self.api_service.initialize()
            
            self.logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def test_browsing_service_initialization(self):
        """Test browsing service initialization"""
        self.logger.info("Testing browsing service initialization")
        
        try:
            # Test cache functionality
            cache_stats = self.browsing_service.get_cache_stats()
            assert isinstance(cache_stats, dict)
            assert 'total_entries' in cache_stats
            assert 'hit_rate' in cache_stats
            
            self.logger.info("✓ Browsing service initialization test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Browsing service initialization test failed: {e}")
            return False
    
    async def test_file_listing_endpoint(self):
        """Test file listing endpoint structure"""
        self.logger.info("Testing file listing endpoint structure")
        
        try:
            # Test with empty connections (should return empty list)
            files = await self.browsing_service.list_files(
                connection_ids=[],
                source_types=None,
                folder_path=None,
                include_subfolders=True,
                file_types=["application/pdf"],
                limit=10,
                offset=0
            )
            
            assert isinstance(files, list)
            self.logger.info(f"✓ File listing returned {len(files)} files (expected 0 for no connections)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ File listing endpoint test failed: {e}")
            return False
    
    async def test_file_tree_endpoint(self):
        """Test file tree endpoint structure"""
        self.logger.info("Testing file tree endpoint structure")
        
        try:
            # Test with empty connections (should return empty list)
            trees = await self.browsing_service.get_file_tree(
                connection_ids=[],
                source_types=None,
                max_depth=3
            )
            
            assert isinstance(trees, list)
            self.logger.info(f"✓ File tree returned {len(trees)} source trees (expected 0 for no connections)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ File tree endpoint test failed: {e}")
            return False
    
    async def test_search_endpoint(self):
        """Test search endpoint structure"""
        self.logger.info("Testing search endpoint structure")
        
        try:
            # Test with empty connections (should return empty list)
            results = await self.browsing_service.search_files(
                query="test",
                connection_ids=[],
                source_types=None,
                search_fields=["name", "content"],
                file_types=["application/pdf"],
                limit=10,
                offset=0
            )
            
            assert isinstance(results, list)
            self.logger.info(f"✓ Search returned {len(results)} results (expected 0 for no connections)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Search endpoint test failed: {e}")
            return False
    
    async def test_cache_functionality(self):
        """Test caching functionality"""
        self.logger.info("Testing cache functionality")
        
        try:
            # Test cache stats
            initial_stats = self.browsing_service.get_cache_stats()
            assert isinstance(initial_stats, dict)
            
            # Test cache invalidation
            self.browsing_service.invalidate_cache()
            
            # Test cache stats after clear
            cleared_stats = self.browsing_service.get_cache_stats()
            assert cleared_stats['total_entries'] == 0
            
            self.logger.info("✓ Cache functionality test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Cache functionality test failed: {e}")
            return False
    
    async def test_data_source_types(self):
        """Test data source type handling"""
        self.logger.info("Testing data source type handling")
        
        try:
            # Test all supported data source types
            supported_types = [
                DataSourceType.GOOGLE_DRIVE,
                DataSourceType.LOCAL_ZIP,
                DataSourceType.LOCAL_DIRECTORY,
                DataSourceType.INDIVIDUAL_UPLOAD,
                DataSourceType.AWS_S3,
                DataSourceType.AZURE_BLOB,
                DataSourceType.GOOGLE_CLOUD_STORAGE
            ]
            
            for source_type in supported_types:
                # Test that each source type is properly handled
                files = await self.browsing_service.list_files(
                    connection_ids=None,
                    source_types=[source_type],
                    folder_path=None,
                    include_subfolders=True,
                    file_types=["application/pdf"],
                    limit=1,
                    offset=0
                )
                assert isinstance(files, list)
            
            self.logger.info(f"✓ Data source types test passed for {len(supported_types)} types")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Data source types test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling in browsing service"""
        self.logger.info("Testing error handling")
        
        try:
            # Test invalid connection ID
            try:
                await self.browsing_service.get_file_metadata("invalid_file_id", "invalid_connection_id")
                # Should raise an exception
                assert False, "Expected exception for invalid connection ID"
            except ValueError:
                # Expected behavior
                pass
            
            # Test file access validation with invalid connection
            try:
                await self.browsing_service.validate_file_access(["file1"], "invalid_connection_id")
                # Should raise an exception
                assert False, "Expected exception for invalid connection ID"
            except ValueError:
                # Expected behavior
                pass
            
            self.logger.info("✓ Error handling test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error handling test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        self.logger.info("Starting unified browsing API tests")
        
        tests = [
            self.test_browsing_service_initialization,
            self.test_file_listing_endpoint,
            self.test_file_tree_endpoint,
            self.test_search_endpoint,
            self.test_cache_functionality,
            self.test_data_source_types,
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
        
        self.logger.info(f"Test results: {passed} passed, {failed} failed")
        
        if failed == 0:
            self.logger.info("🎉 All unified browsing API tests passed!")
            return True
        else:
            self.logger.error(f"❌ {failed} tests failed")
            return False


async def main():
    """Main test function"""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = get_logger(__name__)
    
    logger.info("Starting unified browsing API test suite")
    
    # Create and run tester
    tester = UnifiedBrowsingAPITester()
    
    # Setup test environment
    if not await tester.setup():
        logger.error("Failed to setup test environment")
        return False
    
    # Run all tests
    success = await tester.run_all_tests()
    
    if success:
        logger.info("✅ Unified browsing API test suite completed successfully")
        return True
    else:
        logger.error("❌ Unified browsing API test suite failed")
        return False


if __name__ == "__main__":
    import sys
    
    # Run the test
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)