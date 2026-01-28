"""
Multi-Source API Endpoints

This module provides FastAPI endpoints for multi-source authentication and unified
source browsing that integrate with the existing intelligence layer architecture.
It follows the same patterns as the main intelligence API while providing comprehensive
authentication support and unified file browsing across all data source types.

Features:
- Multi-source authentication endpoints
- Unified file listing and navigation
- Cross-source search and metadata retrieval
- File access validation and permission checking
- Caching for improved performance
- Consistent response formats across all source types
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Query, UploadFile, File, Form, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import hashlib
import json
from pathlib import Path as PathLib

from core.config import get_settings
from core.logging import get_logger
from .multi_source_auth import (
    get_auth_service,
    MultiSourceAuthenticationService,
    DataSourceType,
    AuthenticationStatus,
    AuthenticationResult,
    ConnectionInfo
)
from .batch_ingestion_manager import (
    get_batch_manager,
    BatchIngestionManager,
    BatchIngestionJob,
    JobStatus,
    JobPriority,
    ProcessingOptions,
    FileJobItem,
    SourceProgress,
    JobStatistics,
    WebSocketConnection
)
from .document_management_api import (
    get_document_management_api,
    DocumentManagementAPI,
    DocumentMetadata,
    DocumentListRequest,
    DocumentListResponse,
    DocumentUpdateRequest,
    DocumentReprocessingRequest,
    DocumentReprocessingResponse,
    DocumentPreviewRequest,
    DocumentPreviewResponse,
    DocumentDeletionRequest,
    DocumentDeletionResponse,
    DocumentStatisticsResponse
)


# Pydantic models for API requests/responses

class AuthenticationRequest(BaseModel):
    """Authentication request model"""
    source_type: str = Field(..., description="Data source type")
    user_id: str = Field(..., description="User identifier")
    auth_config: Dict[str, Any] = Field(..., description="Authentication configuration")


class GoogleDriveOAuth2Request(BaseModel):
    """Google Drive OAuth2 authentication request"""
    user_id: str = Field(..., description="User identifier")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(..., description="OAuth2 client secret")
    redirect_uri: str = Field(default="http://localhost:8080/auth/callback", description="OAuth2 redirect URI")


class GoogleDriveServiceAccountRequest(BaseModel):
    """Google Drive service account authentication request"""
    user_id: str = Field(..., description="User identifier")
    service_account_info: Dict[str, Any] = Field(..., description="Service account JSON info")


class AWSS3AuthRequest(BaseModel):
    """AWS S3 authentication request"""
    user_id: str = Field(..., description="User identifier")
    access_key_id: str = Field(..., description="AWS access key ID")
    secret_access_key: str = Field(..., description="AWS secret access key")
    region: str = Field(default="us-east-1", description="AWS region")
    account_id: Optional[str] = Field(default=None, description="AWS account ID")


class AzureBlobAuthRequest(BaseModel):
    """Azure Blob Storage authentication request"""
    user_id: str = Field(..., description="User identifier")
    account_url: str = Field(..., description="Azure storage account URL")
    credential: str = Field(..., description="Azure credential (connection string or SAS token)")
    account_name: Optional[str] = Field(default=None, description="Azure storage account name")


class GoogleCloudStorageAuthRequest(BaseModel):
    """Google Cloud Storage authentication request"""
    user_id: str = Field(..., description="User identifier")
    project_id: Optional[str] = Field(default=None, description="GCP project ID")
    service_account_info: Optional[Dict[str, Any]] = Field(default=None, description="Service account JSON info")


class LocalDirectoryAuthRequest(BaseModel):
    """Local directory authentication request"""
    user_id: str = Field(..., description="User identifier")
    directory_path: str = Field(..., description="Local directory path")


class LocalZipAuthRequest(BaseModel):
    """Local ZIP file authentication request"""
    user_id: str = Field(..., description="User identifier")
    zip_path: str = Field(..., description="Local ZIP file path")


class UploadSetupRequest(BaseModel):
    """Upload setup request"""
    user_id: str = Field(..., description="User identifier")


class AuthenticationResponse(BaseModel):
    """Authentication response model"""
    success: bool
    status: str
    connection_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    permissions: List[str] = []
    expires_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ConnectionStatusResponse(BaseModel):
    """Connection status response model"""
    connection_id: str
    source_type: str
    user_id: str
    status: str
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime] = None
    user_info: Dict[str, Any] = {}
    permissions: List[str] = []
    quota_info: Dict[str, Any] = {}
    error_info: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ConnectionListResponse(BaseModel):
    """Connection list response model"""
    connections: List[ConnectionStatusResponse]
    total_connections: int
    by_source_type: Dict[str, int] = {}
    by_status: Dict[str, int] = {}


class SourceCapabilitiesResponse(BaseModel):
    """Source capabilities response model"""
    source_type: str
    capabilities: Dict[str, Any]
    available: bool = True
    auth_methods: List[str] = []


# Unified Source Browsing Models

class UniversalFileMetadata(BaseModel):
    """Universal file metadata across all data sources"""
    file_id: str = Field(..., description="Unique file identifier")
    name: str = Field(..., description="File name")
    size: int = Field(..., description="File size in bytes")
    modified_time: datetime = Field(..., description="Last modified timestamp")
    source_type: str = Field(..., description="Data source type")
    source_path: str = Field(..., description="Source-specific path")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    access_url: Optional[str] = Field(default=None, description="Access URL if available")
    parent_folders: List[str] = Field(default_factory=list, description="Parent folder hierarchy")
    domain_classification: Optional[str] = Field(default=None, description="Content domain classification")
    checksum: Optional[str] = Field(default=None, description="File checksum")
    source_specific_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")
    processing_status: Optional[str] = Field(default=None, description="Processing status")
    is_accessible: bool = Field(default=True, description="Whether file is accessible")


class FolderMetadata(BaseModel):
    """Folder metadata for hierarchical navigation"""
    folder_id: str = Field(..., description="Unique folder identifier")
    name: str = Field(..., description="Folder name")
    source_type: str = Field(..., description="Data source type")
    source_path: str = Field(..., description="Source-specific path")
    parent_folder_id: Optional[str] = Field(default=None, description="Parent folder ID")
    child_count: int = Field(default=0, description="Number of child items")
    pdf_count: int = Field(default=0, description="Number of PDF files")
    modified_time: Optional[datetime] = Field(default=None, description="Last modified timestamp")
    source_specific_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")


class FileTreeNode(BaseModel):
    """File tree node for hierarchical display"""
    id: str = Field(..., description="Node identifier")
    name: str = Field(..., description="Node name")
    type: str = Field(..., description="Node type: 'folder' or 'file'")
    source_type: str = Field(..., description="Data source type")
    children: List['FileTreeNode'] = Field(default_factory=list, description="Child nodes")
    metadata: Optional[Union[UniversalFileMetadata, FolderMetadata]] = Field(default=None, description="Node metadata")
    is_expanded: bool = Field(default=False, description="Whether node is expanded in UI")


class SourceFileTree(BaseModel):
    """File tree for a specific data source"""
    source_type: str = Field(..., description="Data source type")
    source_name: str = Field(..., description="Human-readable source name")
    connection_id: str = Field(..., description="Connection identifier")
    root_folders: List[FileTreeNode] = Field(default_factory=list, description="Root folder nodes")
    total_files: int = Field(default=0, description="Total file count")
    total_folders: int = Field(default=0, description="Total folder count")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")


class UnifiedFileTree(BaseModel):
    """Unified file tree across all sources"""
    sources: List[SourceFileTree] = Field(default_factory=list, description="Source file trees")
    total_files: int = Field(default=0, description="Total files across all sources")
    total_sources: int = Field(default=0, description="Number of connected sources")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")


class FileListRequest(BaseModel):
    """File listing request"""
    connection_ids: Optional[List[str]] = Field(default=None, description="Specific connection IDs to query")
    source_types: Optional[List[str]] = Field(default=None, description="Filter by source types")
    folder_path: Optional[str] = Field(default=None, description="Specific folder path")
    include_subfolders: bool = Field(default=True, description="Include subfolders recursively")
    file_types: List[str] = Field(default=["application/pdf"], description="MIME types to include")
    limit: Optional[int] = Field(default=None, description="Maximum number of files to return")
    offset: int = Field(default=0, description="Pagination offset")


class SearchRequest(BaseModel):
    """Cross-source search request"""
    query: str = Field(..., description="Search query")
    connection_ids: Optional[List[str]] = Field(default=None, description="Specific connection IDs to search")
    source_types: Optional[List[str]] = Field(default=None, description="Filter by source types")
    search_fields: List[str] = Field(default=["name", "content"], description="Fields to search in")
    file_types: List[str] = Field(default=["application/pdf"], description="MIME types to include")
    limit: int = Field(default=50, description="Maximum number of results")
    offset: int = Field(default=0, description="Pagination offset")


class SearchResult(BaseModel):
    """Search result item"""
    file_metadata: UniversalFileMetadata = Field(..., description="File metadata")
    relevance_score: float = Field(..., description="Search relevance score")
    matched_fields: List[str] = Field(default_factory=list, description="Fields that matched the query")
    snippet: Optional[str] = Field(default=None, description="Content snippet if available")


class SearchResponse(BaseModel):
    """Cross-source search response"""
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(default=0, description="Total number of matching results")
    query: str = Field(..., description="Original search query")
    search_time_ms: int = Field(..., description="Search execution time in milliseconds")
    sources_searched: List[str] = Field(default_factory=list, description="Source types that were searched")


class FileAccessValidationRequest(BaseModel):
    """File access validation request"""
    file_ids: List[str] = Field(..., description="File IDs to validate")
    connection_id: str = Field(..., description="Connection to use for validation")


class FileAccessValidationResult(BaseModel):
    """File access validation result"""
    file_id: str = Field(..., description="File identifier")
    is_accessible: bool = Field(..., description="Whether file is accessible")
    access_level: str = Field(..., description="Access level: 'read', 'write', 'none'")
    error: Optional[str] = Field(default=None, description="Error message if not accessible")


class FileAccessValidationResponse(BaseModel):
    """File access validation response"""
    results: List[FileAccessValidationResult] = Field(default_factory=list, description="Validation results")
    connection_id: str = Field(..., description="Connection used for validation")
    validation_time_ms: int = Field(..., description="Validation execution time in milliseconds")


class CacheStats(BaseModel):
    """Cache statistics"""
    total_entries: int = Field(default=0, description="Total cache entries")
    hit_rate: float = Field(default=0.0, description="Cache hit rate percentage")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    oldest_entry_age_seconds: int = Field(default=0, description="Age of oldest entry in seconds")


# Update FileTreeNode to handle forward references
FileTreeNode.model_rebuild()


# Batch Ingestion Management Models

class BatchJobRequest(BaseModel):
    """Batch ingestion job creation request"""
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Job name")
    description: Optional[str] = Field(default=None, description="Job description")
    priority: str = Field(default="normal", description="Job priority: low, normal, high, urgent")
    file_selections: List[Dict[str, Any]] = Field(..., description="File selections with source info")
    processing_options: Optional[Dict[str, Any]] = Field(default=None, description="Processing configuration")


class BatchJobResponse(BaseModel):
    """Batch ingestion job response"""
    job_id: str = Field(..., description="Job identifier")
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Job name")
    description: Optional[str] = Field(default=None, description="Job description")
    status: str = Field(..., description="Job status")
    priority: str = Field(..., description="Job priority")
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(default=None, description="Start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")
    total_files: int = Field(default=0, description="Total files in job")
    completed_files: int = Field(default=0, description="Completed files")
    failed_files: int = Field(default=0, description="Failed files")
    skipped_files: int = Field(default=0, description="Skipped files")
    progress_percentage: float = Field(default=0.0, description="Overall progress percentage")
    estimated_duration_ms: Optional[int] = Field(default=None, description="Estimated duration in milliseconds")
    actual_duration_ms: Optional[int] = Field(default=None, description="Actual duration in milliseconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    source_progress: List[Dict[str, Any]] = Field(default_factory=list, description="Progress by source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class JobControlRequest(BaseModel):
    """Job control operation request"""
    user_id: str = Field(..., description="User identifier")
    action: str = Field(..., description="Action: start, pause, resume, cancel, retry")
    retry_failed_only: bool = Field(default=True, description="For retry action: only retry failed files")


class JobListRequest(BaseModel):
    """Job list request"""
    user_id: str = Field(..., description="User identifier")
    status_filter: Optional[List[str]] = Field(default=None, description="Filter by job statuses")
    limit: Optional[int] = Field(default=None, description="Maximum number of jobs to return")
    offset: int = Field(default=0, description="Pagination offset")


class JobListResponse(BaseModel):
    """Job list response"""
    jobs: List[BatchJobResponse] = Field(default_factory=list, description="List of jobs")
    total_jobs: int = Field(default=0, description="Total number of jobs")
    has_more: bool = Field(default=False, description="Whether there are more jobs")


class JobStatisticsResponse(BaseModel):
    """Job statistics response"""
    total_jobs: int = Field(default=0, description="Total number of jobs")
    active_jobs: int = Field(default=0, description="Number of active jobs")
    completed_jobs: int = Field(default=0, description="Number of completed jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    cancelled_jobs: int = Field(default=0, description="Number of cancelled jobs")
    total_files_processed: int = Field(default=0, description="Total files processed")
    average_processing_time_ms: float = Field(default=0.0, description="Average processing time")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    by_source_type: Dict[str, int] = Field(default_factory=dict, description="Statistics by source type")
    by_priority: Dict[str, int] = Field(default_factory=dict, description="Statistics by priority")
    by_status: Dict[str, int] = Field(default_factory=dict, description="Statistics by status")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: str = Field(..., description="Message timestamp")


# Update FileTreeNode to handle forward references
FileTreeNode.model_rebuild()


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    connection_id: str = Field(..., description="Connection identifier")


class DisconnectRequest(BaseModel):
    """Disconnect request"""
    connection_id: str = Field(..., description="Connection identifier")


class MultiSourceAuthAPI:
    """Multi-source authentication API service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Service dependencies
        self._auth_service: Optional[MultiSourceAuthenticationService] = None
        self._browsing_service: Optional['UnifiedBrowsingService'] = None
        self._batch_manager: Optional[BatchIngestionManager] = None
        self._document_management: Optional[DocumentManagementAPI] = None
        
        # FastAPI app
        self.app = FastAPI(
            title="Multi-Source Knowledge Ingestion API",
            description="Authentication and unified browsing endpoints for multi-source knowledge ingestion",
            version="1.0.0"
        )
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:5174",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    async def initialize(self) -> bool:
        """Initialize the API service"""
        try:
            self.logger.info("Initializing multi-source authentication API")
            
            # Initialize authentication service
            self._auth_service = get_auth_service()
            
            # Initialize browsing service
            from .unified_browsing_service import get_browsing_service
            self._browsing_service = get_browsing_service()
            await self._browsing_service.initialize()
            
            # Initialize batch ingestion manager
            self._batch_manager = get_batch_manager()
            await self._batch_manager.initialize()
            
            # Initialize document management API
            self._document_management = get_document_management_api()
            await self._document_management.initialize()
            
            self.logger.info("Multi-source authentication API initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize authentication API: {e}")
            return False
    
    def get_auth_service(self) -> MultiSourceAuthenticationService:
        """Dependency to get authentication service"""
        if self._auth_service is None:
            raise HTTPException(status_code=503, detail="Authentication service not initialized")
        return self._auth_service
    
    def get_browsing_service(self):
        """Dependency to get browsing service"""
        if self._browsing_service is None:
            raise HTTPException(status_code=503, detail="Browsing service not initialized")
        return self._browsing_service
    
    def get_batch_manager(self) -> BatchIngestionManager:
        """Dependency to get batch ingestion manager"""
        if self._batch_manager is None:
            raise HTTPException(status_code=503, detail="Batch ingestion manager not initialized")
        return self._batch_manager
    
    def get_document_management(self) -> DocumentManagementAPI:
        """Dependency to get document management API"""
        if self._document_management is None:
            raise HTTPException(status_code=503, detail="Document management API not initialized")
        return self._document_management
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "multi-source-auth-api",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Google Drive Authentication Endpoints
        @self.app.post("/auth/google-drive/oauth2", response_model=AuthenticationResponse)
        async def authenticate_google_drive_oauth2(
            request: GoogleDriveOAuth2Request,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with Google Drive using OAuth2"""
            try:
                auth_config = {
                    'client_config': {
                        'web': {
                            'client_id': request.client_id,
                            'client_secret': request.client_secret,
                            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                            'token_uri': 'https://oauth2.googleapis.com/token',
                            'redirect_uris': [request.redirect_uri]
                        }
                    }
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.GOOGLE_DRIVE,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Google Drive OAuth2 authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        @self.app.post("/auth/google-drive/service-account", response_model=AuthenticationResponse)
        async def authenticate_google_drive_service_account(
            request: GoogleDriveServiceAccountRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with Google Drive using service account"""
            try:
                auth_config = {
                    'service_account_info': request.service_account_info
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.GOOGLE_DRIVE,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Google Drive service account authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        # Cloud Storage Authentication Endpoints
        @self.app.post("/auth/aws-s3", response_model=AuthenticationResponse)
        async def authenticate_aws_s3(
            request: AWSS3AuthRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with AWS S3"""
            try:
                auth_config = {
                    'access_key_id': request.access_key_id,
                    'secret_access_key': request.secret_access_key,
                    'region': request.region,
                    'account_id': request.account_id
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.AWS_S3,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"AWS S3 authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        @self.app.post("/auth/azure-blob", response_model=AuthenticationResponse)
        async def authenticate_azure_blob(
            request: AzureBlobAuthRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with Azure Blob Storage"""
            try:
                auth_config = {
                    'account_url': request.account_url,
                    'credential': request.credential,
                    'account_name': request.account_name
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.AZURE_BLOB,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Azure Blob authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        @self.app.post("/auth/google-cloud-storage", response_model=AuthenticationResponse)
        async def authenticate_google_cloud_storage(
            request: GoogleCloudStorageAuthRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with Google Cloud Storage"""
            try:
                auth_config = {
                    'project_id': request.project_id,
                    'service_account_info': request.service_account_info
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.GOOGLE_CLOUD_STORAGE,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Google Cloud Storage authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        # Local Source Authentication Endpoints
        @self.app.post("/auth/local-directory", response_model=AuthenticationResponse)
        async def authenticate_local_directory(
            request: LocalDirectoryAuthRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with local directory"""
            try:
                auth_config = {
                    'directory_path': request.directory_path
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.LOCAL_DIRECTORY,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Local directory authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        @self.app.post("/auth/local-zip", response_model=AuthenticationResponse)
        async def authenticate_local_zip(
            request: LocalZipAuthRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Authenticate with local ZIP file"""
            try:
                auth_config = {
                    'zip_path': request.zip_path
                }
                
                result = await auth_service.authenticate_source(
                    DataSourceType.LOCAL_ZIP,
                    request.user_id,
                    auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Local ZIP authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        @self.app.post("/auth/upload-setup", response_model=AuthenticationResponse)
        async def setup_upload_handler(
            request: UploadSetupRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Setup individual file upload handler"""
            try:
                result = await auth_service.authenticate_source(
                    DataSourceType.INDIVIDUAL_UPLOAD,
                    request.user_id,
                    {}
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Upload setup failed: {e}")
                raise HTTPException(status_code=500, detail=f"Upload setup failed: {str(e)}")
        
        # Generic Authentication Endpoint
        @self.app.post("/auth/authenticate", response_model=AuthenticationResponse)
        async def authenticate_generic(
            request: AuthenticationRequest,
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Generic authentication endpoint for any source type"""
            try:
                source_type = DataSourceType(request.source_type)
                
                result = await auth_service.authenticate_source(
                    source_type,
                    request.user_id,
                    request.auth_config
                )
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid source type: {str(e)}")
            except Exception as e:
                self.logger.error(f"Generic authentication failed: {e}")
                raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
        # Connection Management Endpoints
        @self.app.post("/connections/{connection_id}/refresh", response_model=AuthenticationResponse)
        async def refresh_connection(
            connection_id: str = Path(..., description="Connection identifier"),
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Refresh connection credentials"""
            try:
                result = await auth_service.refresh_connection(connection_id)
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Connection refresh failed: {e}")
                raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")
        
        @self.app.post("/connections/{connection_id}/validate", response_model=AuthenticationResponse)
        async def validate_connection(
            connection_id: str = Path(..., description="Connection identifier"),
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Validate existing connection"""
            try:
                result = await auth_service.validate_connection(connection_id)
                
                return AuthenticationResponse(
                    success=result.success,
                    status=result.status.value,
                    connection_id=result.connection_id,
                    user_info=result.user_info,
                    permissions=result.permissions,
                    expires_at=result.expires_at,
                    error=result.error,
                    metadata=result.metadata
                )
                
            except Exception as e:
                self.logger.error(f"Connection validation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
        
        @self.app.delete("/connections/{connection_id}")
        async def disconnect_source(
            connection_id: str = Path(..., description="Connection identifier"),
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Disconnect from a data source"""
            try:
                success = await auth_service.disconnect_source(connection_id)
                
                if success:
                    return {"success": True, "message": f"Connection {connection_id} disconnected"}
                else:
                    raise HTTPException(status_code=404, detail="Connection not found")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Disconnect failed: {e}")
                raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")
        
        @self.app.get("/connections/{connection_id}", response_model=ConnectionStatusResponse)
        async def get_connection_status(
            connection_id: str = Path(..., description="Connection identifier"),
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Get connection status information"""
            try:
                connection = await auth_service.get_connection_status(connection_id)
                
                if not connection:
                    raise HTTPException(status_code=404, detail="Connection not found")
                
                return ConnectionStatusResponse(
                    connection_id=connection.connection_id,
                    source_type=connection.source_type.value,
                    user_id=connection.user_id,
                    status=connection.status.value,
                    created_at=connection.created_at,
                    last_accessed=connection.last_accessed,
                    expires_at=connection.expires_at,
                    user_info=connection.user_info,
                    permissions=connection.permissions,
                    quota_info=connection.quota_info,
                    error_info=connection.error_info,
                    metadata=connection.metadata
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get connection status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get connection status: {str(e)}")
        
        @self.app.get("/connections", response_model=ConnectionListResponse)
        async def list_connections(
            user_id: Optional[str] = Query(None, description="Filter by user ID"),
            source_type: Optional[str] = Query(None, description="Filter by source type"),
            status: Optional[str] = Query(None, description="Filter by status"),
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """List all connections with optional filtering"""
            try:
                connections = await auth_service.list_connections(user_id)
                
                # Apply additional filters
                if source_type:
                    connections = [c for c in connections if c.source_type.value == source_type]
                
                if status:
                    connections = [c for c in connections if c.status.value == status]
                
                # Generate statistics
                by_source_type = {}
                by_status = {}
                
                for conn in connections:
                    source_key = conn.source_type.value
                    status_key = conn.status.value
                    
                    by_source_type[source_key] = by_source_type.get(source_key, 0) + 1
                    by_status[status_key] = by_status.get(status_key, 0) + 1
                
                # Convert to response format
                connection_responses = [
                    ConnectionStatusResponse(
                        connection_id=conn.connection_id,
                        source_type=conn.source_type.value,
                        user_id=conn.user_id,
                        status=conn.status.value,
                        created_at=conn.created_at,
                        last_accessed=conn.last_accessed,
                        expires_at=conn.expires_at,
                        user_info=conn.user_info,
                        permissions=conn.permissions,
                        quota_info=conn.quota_info,
                        error_info=conn.error_info,
                        metadata=conn.metadata
                    ) for conn in connections
                ]
                
                return ConnectionListResponse(
                    connections=connection_responses,
                    total_connections=len(connections),
                    by_source_type=by_source_type,
                    by_status=by_status
                )
                
            except Exception as e:
                self.logger.error(f"Failed to list connections: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list connections: {str(e)}")
        
        # Source Capabilities Endpoints
        @self.app.get("/sources/{source_type}/capabilities", response_model=SourceCapabilitiesResponse)
        async def get_source_capabilities(
            source_type: str = Path(..., description="Data source type"),
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Get capabilities for a data source type"""
            try:
                source_enum = DataSourceType(source_type)
                capabilities = await auth_service.get_source_capabilities(source_enum)
                
                return SourceCapabilitiesResponse(
                    source_type=source_type,
                    capabilities=capabilities,
                    available=capabilities.get('available', True),
                    auth_methods=capabilities.get('auth_methods', [])
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid source type: {str(e)}")
            except Exception as e:
                self.logger.error(f"Failed to get source capabilities: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")
        
        @self.app.get("/sources", response_model=List[SourceCapabilitiesResponse])
        async def list_source_types(
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """List all supported source types and their capabilities"""
            try:
                source_types = []
                
                for source_type in DataSourceType:
                    capabilities = await auth_service.get_source_capabilities(source_type)
                    
                    source_types.append(SourceCapabilitiesResponse(
                        source_type=source_type.value,
                        capabilities=capabilities,
                        available=capabilities.get('available', True),
                        auth_methods=capabilities.get('auth_methods', [])
                    ))
                
                return source_types
                
            except Exception as e:
                self.logger.error(f"Failed to list source types: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list source types: {str(e)}")
        
        # Statistics and Monitoring Endpoints
        @self.app.get("/statistics")
        async def get_authentication_statistics(
            auth_service: MultiSourceAuthenticationService = Depends(self.get_auth_service)
        ):
            """Get authentication statistics"""
            try:
                connections = await auth_service.list_connections()
                
                stats = {
                    'total_connections': len(connections),
                    'by_source_type': {},
                    'by_status': {},
                    'by_user': {},
                    'recent_activity': []
                }
                
                # Calculate statistics
                for conn in connections:
                    source_key = conn.source_type.value
                    status_key = conn.status.value
                    user_key = conn.user_id
                    
                    stats['by_source_type'][source_key] = stats['by_source_type'].get(source_key, 0) + 1
                    stats['by_status'][status_key] = stats['by_status'].get(status_key, 0) + 1
                    stats['by_user'][user_key] = stats['by_user'].get(user_key, 0) + 1
                
                # Recent activity (last 10 connections by last_accessed)
                recent_connections = sorted(connections, key=lambda x: x.last_accessed, reverse=True)[:10]
                stats['recent_activity'] = [
                    {
                        'connection_id': conn.connection_id,
                        'source_type': conn.source_type.value,
                        'user_id': conn.user_id,
                        'last_accessed': conn.last_accessed.isoformat()
                    } for conn in recent_connections
                ]
                
                return {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'statistics': stats
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get statistics: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
        
        # Unified Source Browsing Endpoints
        @self.app.post("/browse/files")
        async def list_files(
            request: FileListRequest,
            browsing_service = Depends(self.get_browsing_service)
        ):
            """List files across specified sources with unified metadata format"""
            try:
                start_time = time.time()
                
                # Convert string source types to enums
                source_types = None
                if request.source_types:
                    source_types = [DataSourceType(st) for st in request.source_types]
                
                files = await browsing_service.list_files(
                    connection_ids=request.connection_ids,
                    source_types=source_types,
                    folder_path=request.folder_path,
                    include_subfolders=request.include_subfolders,
                    file_types=request.file_types,
                    limit=request.limit,
                    offset=request.offset
                )
                
                # Convert to response format
                file_responses = []
                for file_meta in files:
                    file_responses.append({
                        'file_id': file_meta.file_id,
                        'name': file_meta.name,
                        'size': file_meta.size,
                        'modified_time': file_meta.modified_time.isoformat(),
                        'source_type': file_meta.source_type.value,
                        'source_path': file_meta.source_path,
                        'mime_type': file_meta.mime_type,
                        'access_url': file_meta.access_url,
                        'parent_folders': file_meta.parent_folders,
                        'domain_classification': file_meta.domain_classification,
                        'checksum': file_meta.checksum,
                        'source_specific_metadata': file_meta.source_specific_metadata,
                        'processing_status': file_meta.processing_status,
                        'is_accessible': file_meta.is_accessible
                    })
                
                execution_time = int((time.time() - start_time) * 1000)
                
                return {
                    'files': file_responses,
                    'total_files': len(file_responses),
                    'execution_time_ms': execution_time,
                    'pagination': {
                        'offset': request.offset,
                        'limit': request.limit,
                        'has_more': len(file_responses) == request.limit if request.limit else False
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Failed to list files: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
        
        @self.app.get("/browse/tree")
        async def get_file_tree(
            connection_ids: Optional[str] = Query(None, description="Comma-separated connection IDs"),
            source_types: Optional[str] = Query(None, description="Comma-separated source types"),
            max_depth: int = Query(3, description="Maximum tree depth"),
            browsing_service = Depends(self.get_browsing_service)
        ):
            """Get hierarchical file tree for sources that support it"""
            try:
                start_time = time.time()
                
                # Parse query parameters
                connection_id_list = connection_ids.split(',') if connection_ids else None
                source_type_list = None
                if source_types:
                    source_type_list = [DataSourceType(st.strip()) for st in source_types.split(',')]
                
                trees = await browsing_service.get_file_tree(
                    connection_ids=connection_id_list,
                    source_types=source_type_list,
                    max_depth=max_depth
                )
                
                # Convert to response format
                tree_responses = []
                total_files = 0
                total_folders = 0
                
                for tree in trees:
                    tree_response = {
                        'source_type': tree.source_type.value,
                        'source_name': tree.source_name,
                        'connection_id': tree.connection_id,
                        'root_folders': self._convert_tree_nodes(tree.root_folders),
                        'total_files': tree.total_files,
                        'total_folders': tree.total_folders,
                        'last_updated': tree.last_updated.isoformat()
                    }
                    tree_responses.append(tree_response)
                    total_files += tree.total_files
                    total_folders += tree.total_folders
                
                execution_time = int((time.time() - start_time) * 1000)
                
                return {
                    'sources': tree_responses,
                    'total_files': total_files,
                    'total_sources': len(tree_responses),
                    'total_folders': total_folders,
                    'execution_time_ms': execution_time,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get file tree: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get file tree: {str(e)}")
        
        @self.app.post("/browse/search", response_model=SearchResponse)
        async def search_files(
            request: SearchRequest,
            browsing_service = Depends(self.get_browsing_service)
        ):
            """Search files across all specified sources"""
            try:
                start_time = time.time()
                
                # Convert string source types to enums
                source_types = None
                if request.source_types:
                    source_types = [DataSourceType(st) for st in request.source_types]
                
                results = await browsing_service.search_files(
                    query=request.query,
                    connection_ids=request.connection_ids,
                    source_types=source_types,
                    search_fields=request.search_fields,
                    file_types=request.file_types,
                    limit=request.limit,
                    offset=request.offset
                )
                
                # Convert to response format
                result_responses = []
                sources_searched = set()
                
                for result in results:
                    sources_searched.add(result.file_metadata.source_type.value)
                    result_responses.append({
                        'file_metadata': {
                            'file_id': result.file_metadata.file_id,
                            'name': result.file_metadata.name,
                            'size': result.file_metadata.size,
                            'modified_time': result.file_metadata.modified_time.isoformat(),
                            'source_type': result.file_metadata.source_type.value,
                            'source_path': result.file_metadata.source_path,
                            'mime_type': result.file_metadata.mime_type,
                            'access_url': result.file_metadata.access_url,
                            'parent_folders': result.file_metadata.parent_folders,
                            'domain_classification': result.file_metadata.domain_classification,
                            'checksum': result.file_metadata.checksum,
                            'source_specific_metadata': result.file_metadata.source_specific_metadata,
                            'processing_status': result.file_metadata.processing_status,
                            'is_accessible': result.file_metadata.is_accessible
                        },
                        'relevance_score': result.relevance_score,
                        'matched_fields': result.matched_fields,
                        'snippet': result.snippet
                    })
                
                execution_time = int((time.time() - start_time) * 1000)
                
                return SearchResponse(
                    results=result_responses,
                    total_results=len(result_responses),
                    query=request.query,
                    search_time_ms=execution_time,
                    sources_searched=list(sources_searched)
                )
                
            except Exception as e:
                self.logger.error(f"Failed to search files: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to search files: {str(e)}")
        
        @self.app.post("/browse/validate-access", response_model=FileAccessValidationResponse)
        async def validate_file_access(
            request: FileAccessValidationRequest,
            browsing_service = Depends(self.get_browsing_service)
        ):
            """Validate access to specific files"""
            try:
                start_time = time.time()
                
                validation_results = await browsing_service.validate_file_access(
                    file_ids=request.file_ids,
                    connection_id=request.connection_id
                )
                
                # Convert to response format
                result_list = []
                for file_id, result in validation_results.items():
                    result_list.append(FileAccessValidationResult(
                        file_id=file_id,
                        is_accessible=result['is_accessible'],
                        access_level=result['access_level'],
                        error=result.get('error')
                    ))
                
                execution_time = int((time.time() - start_time) * 1000)
                
                return FileAccessValidationResponse(
                    results=result_list,
                    connection_id=request.connection_id,
                    validation_time_ms=execution_time
                )
                
            except Exception as e:
                self.logger.error(f"Failed to validate file access: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to validate file access: {str(e)}")
        
        @self.app.get("/browse/files/{file_id}/metadata")
        async def get_file_metadata(
            file_id: str = Path(..., description="File identifier"),
            connection_id: str = Query(..., description="Connection identifier"),
            browsing_service = Depends(self.get_browsing_service)
        ):
            """Get detailed metadata for a specific file"""
            try:
                metadata = await browsing_service.get_file_metadata(file_id, connection_id)
                
                if not metadata:
                    raise HTTPException(status_code=404, detail="File not found")
                
                return {
                    'file_id': metadata.file_id,
                    'name': metadata.name,
                    'size': metadata.size,
                    'modified_time': metadata.modified_time.isoformat(),
                    'source_type': metadata.source_type.value,
                    'source_path': metadata.source_path,
                    'mime_type': metadata.mime_type,
                    'access_url': metadata.access_url,
                    'parent_folders': metadata.parent_folders,
                    'domain_classification': metadata.domain_classification,
                    'checksum': metadata.checksum,
                    'source_specific_metadata': metadata.source_specific_metadata,
                    'processing_status': metadata.processing_status,
                    'is_accessible': metadata.is_accessible
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get file metadata: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get file metadata: {str(e)}")
        
        @self.app.post("/browse/cache/invalidate")
        async def invalidate_cache(
            connection_id: Optional[str] = Body(None, description="Connection ID to invalidate (all if None)"),
            browsing_service = Depends(self.get_browsing_service)
        ):
            """Invalidate browsing cache"""
            try:
                browsing_service.invalidate_cache(connection_id)
                
                return {
                    'success': True,
                    'message': f"Cache invalidated for connection {connection_id}" if connection_id else "All cache cleared",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to invalidate cache: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")
        
        @self.app.get("/browse/cache/stats", response_model=CacheStats)
        async def get_cache_stats(
            browsing_service = Depends(self.get_browsing_service)
        ):
            """Get cache performance statistics"""
            try:
                stats = browsing_service.get_cache_stats()
                
                return CacheStats(
                    total_entries=stats['total_entries'],
                    hit_rate=stats['hit_rate'],
                    memory_usage_mb=stats['memory_usage_mb'],
                    oldest_entry_age_seconds=stats['oldest_entry_age_seconds']
                )
                
            except Exception as e:
                self.logger.error(f"Failed to get cache stats: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")
        
        # Batch Ingestion Management Endpoints
        @self.app.post("/batch/jobs", response_model=BatchJobResponse)
        async def create_batch_job(
            request: BatchJobRequest,
            batch_manager: BatchIngestionManager = Depends(self.get_batch_manager)
        ):
            """Create a new batch ingestion job"""
            try:
                # Convert processing options
                processing_options = None
                if request.processing_options:
                    from .batch_ingestion_manager import ProcessingOptions
                    processing_options = ProcessingOptions(**request.processing_options)
                
                # Convert priority
                from .batch_ingestion_manager import JobPriority
                priority = JobPriority(request.priority)
                
                # Create job
                job_id = await batch_manager.create_job(
                    user_id=request.user_id,
                    name=request.name,
                    file_selections=request.file_selections,
                    processing_options=processing_options,
                    description=request.description,
                    priority=priority
                )
                
                # Get created job
                job = await batch_manager.get_job(job_id, request.user_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Job not found after creation")
                
                return BatchJobResponse(**job.to_dict())
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Failed to create batch job: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create batch job: {str(e)}")
        
        @self.app.post("/batch/jobs/{job_id}/control")
        async def control_batch_job(
            job_id: str = Path(..., description="Job identifier"),
            request: JobControlRequest = Body(...),
            batch_manager: BatchIngestionManager = Depends(self.get_batch_manager)
        ):
            """Control batch ingestion job (start, pause, resume, cancel, retry)"""
            try:
                success = False
                
                if request.action == "start":
                    success = await batch_manager.start_job(job_id, request.user_id)
                elif request.action == "pause":
                    success = await batch_manager.pause_job(job_id, request.user_id)
                elif request.action == "resume":
                    success = await batch_manager.resume_job(job_id, request.user_id)
                elif request.action == "cancel":
                    success = await batch_manager.cancel_job(job_id, request.user_id)
                elif request.action == "retry":
                    success = await batch_manager.retry_job(job_id, request.user_id, request.retry_failed_only)
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
                
                if success:
                    return {"success": True, "message": f"Job {request.action} successful"}
                else:
                    raise HTTPException(status_code=400, detail=f"Job {request.action} failed")
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Failed to control job {job_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to control job: {str(e)}")
        
        @self.app.get("/batch/jobs/{job_id}", response_model=BatchJobResponse)
        async def get_batch_job(
            job_id: str = Path(..., description="Job identifier"),
            user_id: str = Query(..., description="User identifier"),
            batch_manager: BatchIngestionManager = Depends(self.get_batch_manager)
        ):
            """Get batch ingestion job details"""
            try:
                job = await batch_manager.get_job(job_id, user_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                return BatchJobResponse(**job.to_dict())
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get job {job_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")
        
        @self.app.post("/batch/jobs/list", response_model=JobListResponse)
        async def list_batch_jobs(
            request: JobListRequest,
            batch_manager: BatchIngestionManager = Depends(self.get_batch_manager)
        ):
            """List batch ingestion jobs for a user"""
            try:
                # Convert status filter
                status_filter = None
                if request.status_filter:
                    from .batch_ingestion_manager import JobStatus
                    status_filter = [JobStatus(status) for status in request.status_filter]
                
                jobs = await batch_manager.list_jobs(
                    user_id=request.user_id,
                    status_filter=status_filter,
                    limit=request.limit,
                    offset=request.offset
                )
                
                job_responses = [BatchJobResponse(**job.to_dict()) for job in jobs]
                
                # Check if there are more jobs
                has_more = False
                if request.limit and len(jobs) == request.limit:
                    # Check if there's at least one more job
                    next_jobs = await batch_manager.list_jobs(
                        user_id=request.user_id,
                        status_filter=status_filter,
                        limit=1,
                        offset=request.offset + request.limit
                    )
                    has_more = len(next_jobs) > 0
                
                return JobListResponse(
                    jobs=job_responses,
                    total_jobs=len(job_responses),
                    has_more=has_more
                )
                
            except Exception as e:
                self.logger.error(f"Failed to list jobs: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")
        
        @self.app.get("/batch/statistics", response_model=JobStatisticsResponse)
        async def get_batch_statistics(
            user_id: Optional[str] = Query(None, description="User identifier (None for global stats)"),
            batch_manager: BatchIngestionManager = Depends(self.get_batch_manager)
        ):
            """Get batch ingestion statistics"""
            try:
                stats = await batch_manager.get_job_statistics(user_id)
                
                return JobStatisticsResponse(
                    total_jobs=stats.total_jobs,
                    active_jobs=stats.active_jobs,
                    completed_jobs=stats.completed_jobs,
                    failed_jobs=stats.failed_jobs,
                    cancelled_jobs=stats.cancelled_jobs,
                    total_files_processed=stats.total_files_processed,
                    average_processing_time_ms=stats.average_processing_time_ms,
                    success_rate=stats.success_rate,
                    by_source_type=stats.by_source_type,
                    by_priority=stats.by_priority,
                    by_status=stats.by_status
                )
                
            except Exception as e:
                self.logger.error(f"Failed to get batch statistics: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get batch statistics: {str(e)}")
        
        # Document Management Endpoints
        @self.app.post("/documents/list", response_model=DocumentListResponse)
        async def list_documents(
            request: DocumentListRequest,
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """List documents with multi-source filtering and attribution"""
            try:
                return await document_management.list_documents(request)
                
            except Exception as e:
                self.logger.error(f"Failed to list documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
        
        @self.app.get("/documents/{document_id}", response_model=DocumentMetadata)
        async def get_document_by_id(
            document_id: str = Path(..., description="Document identifier"),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Get document by ID with full multi-source metadata"""
            try:
                return await document_management.get_document_by_id(document_id)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get document {document_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")
        
        @self.app.put("/documents/{document_id}", response_model=DocumentMetadata)
        async def update_document_metadata(
            document_id: str = Path(..., description="Document identifier"),
            request: DocumentUpdateRequest = Body(...),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Update document metadata with multi-source awareness"""
            try:
                return await document_management.update_document_metadata(document_id, request)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to update document {document_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")
        
        @self.app.post("/documents/reprocess", response_model=DocumentReprocessingResponse)
        async def reprocess_document(
            request: DocumentReprocessingRequest,
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Re-process document with source-appropriate methods"""
            try:
                return await document_management.reprocess_document(request)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to reprocess document: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to reprocess document: {str(e)}")
        
        @self.app.post("/documents/preview", response_model=DocumentPreviewResponse)
        async def get_document_preview(
            request: DocumentPreviewRequest,
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Get document preview with processing statistics and source information"""
            try:
                return await document_management.get_document_preview(request)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get document preview: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get document preview: {str(e)}")
        
        @self.app.post("/documents/delete", response_model=DocumentDeletionResponse)
        async def delete_documents(
            request: DocumentDeletionRequest,
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Delete documents with source-aware cleanup"""
            try:
                return await document_management.delete_documents(request)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to delete documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")
        
        @self.app.get("/documents/statistics", response_model=DocumentStatisticsResponse)
        async def get_document_statistics(
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Get comprehensive document statistics across all sources"""
            try:
                return await document_management.get_document_statistics()
                
            except Exception as e:
                self.logger.error(f"Failed to get document statistics: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")
        
        @self.app.get("/documents/{document_id}/validate-access")
        async def validate_document_access(
            document_id: str = Path(..., description="Document identifier"),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Validate document access through its source"""
            try:
                document = await document_management.get_document_by_id(document_id)
                
                if not document.source_info.connection_id:
                    return {
                        'document_id': document_id,
                        'is_accessible': True,
                        'message': 'Local document, always accessible'
                    }
                
                validation_result = await document_management._validate_document_access(
                    document_id, 
                    document.source_info.connection_id
                )
                
                return {
                    'document_id': document_id,
                    'is_accessible': validation_result.get('is_accessible', False),
                    'permissions': validation_result.get('permissions', []),
                    'error': validation_result.get('error')
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to validate document access: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to validate access: {str(e)}")
        
        @self.app.get("/documents/{document_id}/source-link")
        async def get_document_source_link(
            document_id: str = Path(..., description="Document identifier"),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Get source-specific link for document"""
            try:
                document = await document_management.get_document_by_id(document_id)
                
                source_link = {
                    'document_id': document_id,
                    'source_type': document.source_info.source_type,
                    'source_path': document.source_info.source_path,
                    'access_url': document.source_info.access_url,
                    'is_accessible': document.source_info.is_accessible
                }
                
                # Add source-specific link information
                if document.source_info.source_type == 'google_drive' and document.source_info.access_url:
                    source_link['google_drive_link'] = document.source_info.access_url
                    source_link['can_open_in_browser'] = True
                elif document.source_info.source_type in ['local_directory', 'local_zip']:
                    source_link['local_path'] = document.source_info.source_path
                    source_link['can_open_in_browser'] = False
                elif document.source_info.source_type in ['aws_s3', 'azure_blob', 'google_cloud_storage']:
                    source_link['cloud_storage_path'] = document.source_info.source_path
                    source_link['can_open_in_browser'] = bool(document.source_info.access_url)
                
                return source_link
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get document source link: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get source link: {str(e)}")
        
        @self.app.get("/documents/by-source/{source_type}")
        async def list_documents_by_source(
            source_type: str = Path(..., description="Source type"),
            connection_id: Optional[str] = Query(None, description="Connection ID filter"),
            processing_status: Optional[str] = Query(None, description="Processing status filter"),
            limit: int = Query(50, description="Maximum results"),
            offset: int = Query(0, description="Pagination offset"),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """List documents from a specific source type"""
            try:
                # Create request with source type filter
                request = DocumentListRequest(
                    source_types=[source_type],
                    connection_ids=[connection_id] if connection_id else None,
                    processing_status=[processing_status] if processing_status else None,
                    limit=limit,
                    offset=offset
                )
                
                return await document_management.list_documents(request)
                
            except Exception as e:
                self.logger.error(f"Failed to list documents by source {source_type}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
        
        @self.app.post("/documents/bulk-update")
        async def bulk_update_documents(
            document_ids: List[str] = Body(..., description="Document IDs to update"),
            updates: DocumentUpdateRequest = Body(..., description="Updates to apply"),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Bulk update multiple documents"""
            try:
                results = []
                
                for document_id in document_ids:
                    try:
                        updated_doc = await document_management.update_document_metadata(document_id, updates)
                        results.append({
                            'document_id': document_id,
                            'success': True,
                            'document': updated_doc
                        })
                    except Exception as e:
                        results.append({
                            'document_id': document_id,
                            'success': False,
                            'error': str(e)
                        })
                
                successful_updates = [r for r in results if r['success']]
                failed_updates = [r for r in results if not r['success']]
                
                return {
                    'total_documents': len(document_ids),
                    'successful_updates': len(successful_updates),
                    'failed_updates': len(failed_updates),
                    'results': results
                }
                
            except Exception as e:
                self.logger.error(f"Failed to bulk update documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to bulk update: {str(e)}")
        
        @self.app.post("/documents/bulk-reprocess")
        async def bulk_reprocess_documents(
            document_ids: List[str] = Body(..., description="Document IDs to reprocess"),
            processing_options: Optional[Dict[str, Any]] = Body(None, description="Processing options"),
            force_reprocess: bool = Body(False, description="Force reprocessing"),
            document_management: DocumentManagementAPI = Depends(self.get_document_management)
        ):
            """Bulk reprocess multiple documents"""
            try:
                results = []
                
                for document_id in document_ids:
                    try:
                        request = DocumentReprocessingRequest(
                            document_id=document_id,
                            processing_options=processing_options,
                            force_reprocess=force_reprocess
                        )
                        
                        result = await document_management.reprocess_document(request)
                        results.append({
                            'document_id': document_id,
                            'success': result.success,
                            'job_id': result.job_id,
                            'message': result.message
                        })
                    except Exception as e:
                        results.append({
                            'document_id': document_id,
                            'success': False,
                            'error': str(e)
                        })
                
                successful_reprocessing = [r for r in results if r['success']]
                failed_reprocessing = [r for r in results if not r['success']]
                
                return {
                    'total_documents': len(document_ids),
                    'successful_reprocessing': len(successful_reprocessing),
                    'failed_reprocessing': len(failed_reprocessing),
                    'results': results
                }
                
            except Exception as e:
                self.logger.error(f"Failed to bulk reprocess documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to bulk reprocess: {str(e)}")
        
        @self.app.websocket("/batch/ws/{user_id}")
        async def websocket_endpoint(
            websocket,
            user_id: str = Path(..., description="User identifier"),
            job_id: Optional[str] = Query(None, description="Specific job ID to monitor"),
            batch_manager: BatchIngestionManager = Depends(self.get_batch_manager)
        ):
            """WebSocket endpoint for real-time batch ingestion progress updates"""
            await websocket.accept()
            
            try:
                # Add WebSocket connection
                connection = await batch_manager.add_websocket_connection(websocket, user_id, job_id)
                
                # Send initial connection confirmation
                await websocket.send_text(json.dumps({
                    'type': 'connection_established',
                    'user_id': user_id,
                    'job_id': job_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }))
                
                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        # Wait for messages from client (ping/pong, etc.)
                        message = await websocket.receive_text()
                        data = json.loads(message)
                        
                        if data.get('type') == 'ping':
                            await websocket.send_text(json.dumps({
                                'type': 'pong',
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }))
                        
                    except Exception as e:
                        self.logger.debug(f"WebSocket message handling error: {e}")
                        break
                
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
            finally:
                # Remove WebSocket connection
                await batch_manager.remove_websocket_connection(websocket, user_id)
    
    def _convert_tree_nodes(self, nodes) -> List[Dict[str, Any]]:
        """Convert FileTreeNode objects to dictionary format"""
        result = []
        
        for node in nodes:
            node_dict = {
                'id': node.id,
                'name': node.name,
                'type': node.type,
                'source_type': node.source_type.value,
                'is_expanded': node.is_expanded,
                'children': self._convert_tree_nodes(node.children)
            }
            
            # Add metadata if available
            if node.metadata:
                if node.type == 'file':
                    node_dict['metadata'] = {
                        'file_id': node.metadata.file_id,
                        'name': node.metadata.name,
                        'size': node.metadata.size,
                        'modified_time': node.metadata.modified_time.isoformat(),
                        'source_type': node.metadata.source_type.value,
                        'source_path': node.metadata.source_path,
                        'mime_type': node.metadata.mime_type,
                        'access_url': node.metadata.access_url,
                        'parent_folders': node.metadata.parent_folders,
                        'domain_classification': node.metadata.domain_classification,
                        'checksum': node.metadata.checksum,
                        'source_specific_metadata': node.metadata.source_specific_metadata,
                        'processing_status': node.metadata.processing_status,
                        'is_accessible': node.metadata.is_accessible
                    }
                else:  # folder
                    node_dict['metadata'] = {
                        'folder_id': node.metadata.folder_id,
                        'name': node.metadata.name,
                        'source_type': node.metadata.source_type.value,
                        'source_path': node.metadata.source_path,
                        'parent_folder_id': node.metadata.parent_folder_id,
                        'child_count': node.metadata.child_count,
                        'pdf_count': node.metadata.pdf_count,
                        'modified_time': node.metadata.modified_time.isoformat() if node.metadata.modified_time else None,
                        'source_specific_metadata': node.metadata.source_specific_metadata
                    }
            
            result.append(node_dict)
        
        return result
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the API server"""
        try:
            await self.initialize()
            
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            self.logger.info(f"Starting multi-source authentication API server on {host}:{port}")
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise


# Global API service instance
_api_service: Optional[MultiSourceAuthAPI] = None


def get_api_service() -> MultiSourceAuthAPI:
    """Get or create global API service instance"""
    global _api_service
    
    if _api_service is None:
        _api_service = MultiSourceAuthAPI()
    
    return _api_service


async def start_multi_source_auth_api(host: str = "0.0.0.0", port: int = 8001):
    """Start the multi-source authentication API server"""
    service = get_api_service()
    await service.start_server(host, port)


if __name__ == "__main__":
    import asyncio
    asyncio.run(start_multi_source_auth_api())