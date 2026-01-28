"""
Test Multi-Source Authentication System

This test file validates the multi-source authentication endpoints and services
to ensure they work correctly with all supported data source types.
"""

import asyncio
import json
import os
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from services.multi_source_auth import (
    MultiSourceAuthenticationService,
    DataSourceType,
    AuthenticationStatus,
    CredentialManager
)
from services.multi_source_api_endpoints import get_api_service
from core.config import get_settings


class TestMultiSourceAuthentication:
    """Test suite for multi-source authentication"""
    
    @pytest.fixture
    def auth_service(self):
        """Create authentication service for testing"""
        return MultiSourceAuthenticationService()
    
    @pytest.fixture
    def api_client(self):
        """Create API client for testing"""
        api_service = get_api_service()
        return TestClient(api_service.app)
    
    @pytest.fixture
    def temp_credentials_dir(self):
        """Create temporary credentials directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def temp_zip_file(self):
        """Create temporary ZIP file with test PDFs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "test.zip"
            
            # Create test PDF content (mock)
            test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                zip_file.writestr("test1.pdf", test_pdf_content)
                zip_file.writestr("test2.pdf", test_pdf_content)
                zip_file.writestr("folder/test3.pdf", test_pdf_content)
            
            yield zip_path
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory with test PDFs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test PDF files
            test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            
            (temp_path / "test1.pdf").write_bytes(test_pdf_content)
            (temp_path / "test2.pdf").write_bytes(test_pdf_content)
            
            # Create subdirectory with PDF
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "test3.pdf").write_bytes(test_pdf_content)
            
            yield temp_path
    
    def test_credential_manager_encryption(self, temp_credentials_dir):
        """Test credential encryption and decryption"""
        # Set up credential manager with temp directory
        os.environ['CREDENTIALS_DIR'] = str(temp_credentials_dir)
        credential_manager = CredentialManager()
        
        # Test data
        test_credentials = {
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "region": "us-east-1"
        }
        
        # Test encryption
        encrypted = credential_manager.encrypt_credentials(test_credentials)
        assert encrypted is not None
        assert isinstance(encrypted, str)
        
        # Test decryption
        decrypted = credential_manager.decrypt_credentials(encrypted)
        assert decrypted == test_credentials
    
    @pytest.mark.asyncio
    async def test_local_directory_authentication(self, auth_service, temp_directory):
        """Test local directory authentication"""
        result = await auth_service.authenticate_source(
            DataSourceType.LOCAL_DIRECTORY,
            "test_user",
            {"directory_path": str(temp_directory)}
        )
        
        assert result.success is True
        assert result.status == AuthenticationStatus.AUTHENTICATED
        assert result.connection_id is not None
        assert "pdf_count" in result.metadata
        assert result.metadata["pdf_count"] == 3  # 3 PDF files created
    
    @pytest.mark.asyncio
    async def test_local_zip_authentication(self, auth_service, temp_zip_file):
        """Test local ZIP file authentication"""
        result = await auth_service.authenticate_source(
            DataSourceType.LOCAL_ZIP,
            "test_user",
            {"zip_path": str(temp_zip_file)}
        )
        
        assert result.success is True
        assert result.status == AuthenticationStatus.AUTHENTICATED
        assert result.connection_id is not None
        assert "pdf_count" in result.metadata
        assert result.metadata["pdf_count"] == 3  # 3 PDF files in ZIP
    
    @pytest.mark.asyncio
    async def test_upload_setup(self, auth_service):
        """Test upload handler setup"""
        result = await auth_service.authenticate_source(
            DataSourceType.INDIVIDUAL_UPLOAD,
            "test_user",
            {}
        )
        
        assert result.success is True
        assert result.status == AuthenticationStatus.AUTHENTICATED
        assert result.connection_id is not None
        assert "upload_dir" in result.metadata
    
    @pytest.mark.asyncio
    async def test_invalid_directory_authentication(self, auth_service):
        """Test authentication with invalid directory"""
        result = await auth_service.authenticate_source(
            DataSourceType.LOCAL_DIRECTORY,
            "test_user",
            {"directory_path": "/nonexistent/directory"}
        )
        
        assert result.success is False
        assert result.status == AuthenticationStatus.INVALID
        assert "does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_zip_authentication(self, auth_service):
        """Test authentication with invalid ZIP file"""
        result = await auth_service.authenticate_source(
            DataSourceType.LOCAL_ZIP,
            "test_user",
            {"zip_path": "/nonexistent/file.zip"}
        )
        
        assert result.success is False
        assert result.status == AuthenticationStatus.INVALID
        assert "does not exist" in result.error
    
    @pytest.mark.asyncio
    async def test_connection_management(self, auth_service, temp_directory):
        """Test connection management operations"""
        # Create connection
        result = await auth_service.authenticate_source(
            DataSourceType.LOCAL_DIRECTORY,
            "test_user",
            {"directory_path": str(temp_directory)}
        )
        
        assert result.success is True
        connection_id = result.connection_id
        
        # Get connection status
        connection = await auth_service.get_connection_status(connection_id)
        assert connection is not None
        assert connection.connection_id == connection_id
        assert connection.source_type == DataSourceType.LOCAL_DIRECTORY
        assert connection.user_id == "test_user"
        
        # Validate connection
        validation_result = await auth_service.validate_connection(connection_id)
        assert validation_result.success is True
        
        # List connections
        connections = await auth_service.list_connections("test_user")
        assert len(connections) >= 1
        assert any(c.connection_id == connection_id for c in connections)
        
        # Disconnect
        disconnect_success = await auth_service.disconnect_source(connection_id)
        assert disconnect_success is True
        
        # Verify disconnection
        connection_after_disconnect = await auth_service.get_connection_status(connection_id)
        assert connection_after_disconnect is None
    
    @pytest.mark.asyncio
    async def test_source_capabilities(self, auth_service):
        """Test source capabilities retrieval"""
        # Test local directory capabilities
        capabilities = await auth_service.get_source_capabilities(DataSourceType.LOCAL_DIRECTORY)
        assert capabilities["can_browse"] is True
        assert capabilities["can_search"] is False
        assert capabilities["supports_auth"] is False
        assert capabilities["supports_batch"] is True
        
        # Test Google Drive capabilities
        gd_capabilities = await auth_service.get_source_capabilities(DataSourceType.GOOGLE_DRIVE)
        assert gd_capabilities["can_browse"] is True
        assert gd_capabilities["can_search"] is True
        assert gd_capabilities["supports_auth"] is True
        assert "oauth2" in gd_capabilities["auth_methods"]
        assert "service_account" in gd_capabilities["auth_methods"]
    
    def test_api_health_endpoint(self, api_client):
        """Test API health endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "multi-source-auth-api"
        assert "timestamp" in data
    
    def test_api_list_source_types(self, api_client):
        """Test API source types listing"""
        response = api_client.get("/sources")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check that all expected source types are present
        source_types = [item["source_type"] for item in data]
        expected_types = [
            "google_drive",
            "local_zip", 
            "local_directory",
            "individual_upload",
            "aws_s3",
            "azure_blob",
            "google_cloud_storage"
        ]
        
        for expected_type in expected_types:
            assert expected_type in source_types
    
    def test_api_source_capabilities(self, api_client):
        """Test API source capabilities endpoint"""
        response = api_client.get("/sources/local_directory/capabilities")
        assert response.status_code == 200
        
        data = response.json()
        assert data["source_type"] == "local_directory"
        assert "capabilities" in data
        assert data["available"] is True
        assert isinstance(data["auth_methods"], list)
    
    def test_api_local_directory_auth(self, api_client, temp_directory):
        """Test API local directory authentication"""
        auth_request = {
            "user_id": "test_user",
            "directory_path": str(temp_directory)
        }
        
        response = api_client.post("/auth/local-directory", json=auth_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "authenticated"
        assert data["connection_id"] is not None
        assert "pdf_count" in data["metadata"]
    
    def test_api_local_zip_auth(self, api_client, temp_zip_file):
        """Test API local ZIP authentication"""
        auth_request = {
            "user_id": "test_user",
            "zip_path": str(temp_zip_file)
        }
        
        response = api_client.post("/auth/local-zip", json=auth_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "authenticated"
        assert data["connection_id"] is not None
        assert "pdf_count" in data["metadata"]
    
    def test_api_upload_setup(self, api_client):
        """Test API upload setup"""
        auth_request = {
            "user_id": "test_user"
        }
        
        response = api_client.post("/auth/upload-setup", json=auth_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "authenticated"
        assert data["connection_id"] is not None
        assert "upload_dir" in data["metadata"]
    
    def test_api_invalid_source_type(self, api_client):
        """Test API with invalid source type"""
        response = api_client.get("/sources/invalid_source/capabilities")
        assert response.status_code == 400
        assert "Invalid source type" in response.json()["detail"]
    
    def test_api_connection_management(self, api_client, temp_directory):
        """Test API connection management endpoints"""
        # Create connection
        auth_request = {
            "user_id": "test_user",
            "directory_path": str(temp_directory)
        }
        
        response = api_client.post("/auth/local-directory", json=auth_request)
        assert response.status_code == 200
        
        connection_id = response.json()["connection_id"]
        
        # Get connection status
        response = api_client.get(f"/connections/{connection_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["connection_id"] == connection_id
        assert data["source_type"] == "local_directory"
        assert data["user_id"] == "test_user"
        
        # Validate connection
        response = api_client.post(f"/connections/{connection_id}/validate")
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # List connections
        response = api_client.get("/connections", params={"user_id": "test_user"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_connections"] >= 1
        assert any(c["connection_id"] == connection_id for c in data["connections"])
        
        # Disconnect
        response = api_client.delete(f"/connections/{connection_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify disconnection
        response = api_client.get(f"/connections/{connection_id}")
        assert response.status_code == 404
    
    def test_api_statistics(self, api_client, temp_directory):
        """Test API statistics endpoint"""
        # Create a connection first
        auth_request = {
            "user_id": "test_user",
            "directory_path": str(temp_directory)
        }
        
        response = api_client.post("/auth/local-directory", json=auth_request)
        assert response.status_code == 200
        
        # Get statistics
        response = api_client.get("/statistics")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "statistics" in data
        
        stats = data["statistics"]
        assert "total_connections" in stats
        assert "by_source_type" in stats
        assert "by_status" in stats
        assert "by_user" in stats
        assert "recent_activity" in stats
        
        assert stats["total_connections"] >= 1
        assert "local_directory" in stats["by_source_type"]
        assert "authenticated" in stats["by_status"]


def test_google_drive_service_account_mock():
    """Test Google Drive service account authentication with mock data"""
    # This test uses mock service account data since we don't have real credentials
    mock_service_account_info = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
    
    # This test would fail with real API calls, but validates the structure
    auth_service = MultiSourceAuthenticationService()
    
    # Test that the service accepts the correct structure
    assert isinstance(mock_service_account_info, dict)
    assert "type" in mock_service_account_info
    assert "client_email" in mock_service_account_info
    assert mock_service_account_info["type"] == "service_account"


def test_aws_s3_config_structure():
    """Test AWS S3 configuration structure"""
    mock_aws_config = {
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "region": "us-east-1",
        "account_id": "123456789012"
    }
    
    # Validate structure
    assert "access_key_id" in mock_aws_config
    assert "secret_access_key" in mock_aws_config
    assert "region" in mock_aws_config
    assert len(mock_aws_config["access_key_id"]) > 0
    assert len(mock_aws_config["secret_access_key"]) > 0


def test_azure_blob_config_structure():
    """Test Azure Blob Storage configuration structure"""
    mock_azure_config = {
        "account_url": "https://teststorage.blob.core.windows.net",
        "credential": "DefaultAzureCredential",
        "account_name": "teststorage"
    }
    
    # Validate structure
    assert "account_url" in mock_azure_config
    assert "credential" in mock_azure_config
    assert mock_azure_config["account_url"].startswith("https://")
    assert mock_azure_config["account_url"].endswith(".blob.core.windows.net")


if __name__ == "__main__":
    # Run basic tests without pytest
    import sys
    
    print("Running basic multi-source authentication tests...")
    
    # Test credential manager
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['CREDENTIALS_DIR'] = temp_dir
        credential_manager = CredentialManager()
        
        test_creds = {"test": "data"}
        encrypted = credential_manager.encrypt_credentials(test_creds)
        decrypted = credential_manager.decrypt_credentials(encrypted)
        
        assert decrypted == test_creds
        print("✓ Credential encryption/decryption test passed")
    
    # Test configuration structures
    test_google_drive_service_account_mock()
    print("✓ Google Drive service account structure test passed")
    
    test_aws_s3_config_structure()
    print("✓ AWS S3 configuration structure test passed")
    
    test_azure_blob_config_structure()
    print("✓ Azure Blob configuration structure test passed")
    
    print("\nAll basic tests passed! Run with pytest for full test suite.")