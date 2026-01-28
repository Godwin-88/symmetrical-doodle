#!/usr/bin/env python3
"""
Test script for Google Drive integration components.

This script tests the authentication and discovery services to ensure
they work correctly with the configured Google Drive API.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from services.google_drive_auth import GoogleDriveAuthService
from services.google_drive_discovery import GoogleDriveDiscoveryService
from core.config import get_settings
from core.logging import get_logger


async def test_authentication():
    """Test Google Drive authentication"""
    logger = get_logger(__name__)
    logger.info("Testing Google Drive authentication...")
    
    auth_service = GoogleDriveAuthService()
    
    # Test authentication
    auth_result = await auth_service.authenticate()
    
    if auth_result.success:
        logger.info(f"✅ Authentication successful using {auth_result.auth_method.value}")
        logger.info(f"Credentials expire at: {auth_result.expires_at}")
        
        # Get auth info
        auth_info = auth_service.get_auth_info()
        logger.info(f"Auth info: {auth_info}")
        
        return auth_service
    else:
        logger.error(f"❌ Authentication failed: {auth_result.error_message}")
        return None


async def test_discovery(auth_service: GoogleDriveAuthService):
    """Test file discovery"""
    logger = get_logger(__name__)
    logger.info("Testing file discovery...")
    
    discovery_service = GoogleDriveDiscoveryService(auth_service)
    
    # Get settings to check if folder IDs are configured
    settings = get_settings()
    
    if not settings.google_drive.folder_ids:
        logger.warning("No folder IDs configured in settings. Skipping discovery test.")
        logger.info("To test discovery, add folder IDs to your configuration.")
        return
    
    # Test discovery
    discovery_result = await discovery_service.discover_pdfs()
    
    if discovery_result.success:
        logger.info(f"✅ Discovery successful!")
        logger.info(f"Found {len(discovery_result.pdfs_found)} PDF files")
        logger.info(f"Scanned {discovery_result.total_files_scanned} total files")
        logger.info(f"Scanned {len(discovery_result.folders_scanned)} folders")
        logger.info(f"Found {len(discovery_result.inaccessible_files)} inaccessible files")
        
        # Show first few PDFs found
        for i, pdf in enumerate(discovery_result.pdfs_found[:3]):
            logger.info(f"PDF {i+1}: {pdf.name} ({pdf.size} bytes, {pdf.access_status.value})")
        
        if discovery_result.inaccessible_files:
            logger.warning("Inaccessible files:")
            for file_info in discovery_result.inaccessible_files[:3]:
                logger.warning(f"  - {file_info.get('name', 'Unknown')}: {file_info.get('reason', 'Unknown error')}")
        
        if discovery_result.errors:
            logger.warning("Errors encountered:")
            for error in discovery_result.errors[:3]:
                logger.warning(f"  - {error}")
    else:
        logger.error("❌ Discovery failed")
        for error in discovery_result.errors:
            logger.error(f"  - {error}")


async def main():
    """Main test function"""
    logger = get_logger(__name__)
    logger.info("Starting Google Drive integration tests...")
    
    try:
        # Test authentication
        auth_service = await test_authentication()
        
        if auth_service:
            # Test discovery
            await test_discovery(auth_service)
        
        logger.info("Tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())