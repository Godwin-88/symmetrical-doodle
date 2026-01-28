"""
Intelligence Layer Integration for Multi-Source Authentication

This module integrates multi-source authentication endpoints with the existing
intelligence layer FastAPI application, following the same patterns and
maintaining consistency with the existing API structure.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Query
from fastapi.responses import JSONResponse

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
from .multi_source_api_endpoints import (
    AuthenticationRequest,
    GoogleDriveOAuth2Request,
    GoogleDriveServiceAccountRequest,
    AWSS3AuthRequest,
    AzureBlobAuthRequest,
    GoogleCloudStorageAuthRequest,
    LocalDirectoryAuthRequest,
    LocalZipAuthRequest,
    UploadSetupRequest,
    AuthenticationResponse,
    ConnectionStatusResponse,
    ConnectionListResponse,
    SourceCapabilitiesResponse
)


class IntelligenceLayerIntegration:
    """Integration service for adding multi-source auth to intelligence layer"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self._auth_service: Optional[MultiSourceAuthenticationService] = None
    
    async def initialize(self) -> bool:
        """Initialize the integration service"""
        try:
            self.logger.info("Initializing intelligence layer integration")
            
            # Initialize authentication service
            self._auth_service = get_auth_service()
            
            self.logger.info("Intelligence layer integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize intelligence layer integration: {e}")
            return False
    
    def get_auth_service(self) -> MultiSourceAuthenticationService:
        """Dependency to get authentication service"""
        if self._auth_service is None:
            raise HTTPException(status_code=503, detail="Authentication service not initialized")
        return self._auth_service
    
    def add_routes_to_app(self, app: FastAPI):
        """Add multi-source authentication routes to existing FastAPI app"""
        
        # ============================================================================
        # MULTI-SOURCE AUTHENTICATION API ENDPOINTS
        # ============================================================================
        
        @app.get("/multi-source/health")
        async def multi_source_health_check():
            """Multi-source authentication health check"""
            return {
                "status": "healthy",
                "service": "multi-source-auth",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "integration": "intelligence-layer"
            }
        
        # Google Drive Authentication
        @app.post("/multi-source/auth/google-drive/oauth2", response_model=AuthenticationResponse)
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
        
        @app.post("/multi-source/auth/google-drive/service-account", response_model=AuthenticationResponse)
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
        
        # Cloud Storage Authentication
        @app.post("/multi-source/auth/aws-s3", response_model=AuthenticationResponse)
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
        
        @app.post("/multi-source/auth/azure-blob", response_model=AuthenticationResponse)
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
        
        @app.post("/multi-source/auth/google-cloud-storage", response_model=AuthenticationResponse)
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
        
        # Local Source Authentication
        @app.post("/multi-source/auth/local-directory", response_model=AuthenticationResponse)
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
        
        @app.post("/multi-source/auth/local-zip", response_model=AuthenticationResponse)
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
        
        @app.post("/multi-source/auth/upload-setup", response_model=AuthenticationResponse)
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
        
        # Generic Authentication
        @app.post("/multi-source/auth/authenticate", response_model=AuthenticationResponse)
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
        
        # Connection Management
        @app.post("/multi-source/connections/{connection_id}/refresh", response_model=AuthenticationResponse)
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
        
        @app.post("/multi-source/connections/{connection_id}/validate", response_model=AuthenticationResponse)
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
        
        @app.delete("/multi-source/connections/{connection_id}")
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
        
        @app.get("/multi-source/connections/{connection_id}", response_model=ConnectionStatusResponse)
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
        
        @app.get("/multi-source/connections", response_model=ConnectionListResponse)
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
        
        # Source Capabilities
        @app.get("/multi-source/sources/{source_type}/capabilities", response_model=SourceCapabilitiesResponse)
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
        
        @app.get("/multi-source/sources", response_model=List[SourceCapabilitiesResponse])
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
        
        # Statistics
        @app.get("/multi-source/statistics")
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


# Global integration instance
_integration: Optional[IntelligenceLayerIntegration] = None


def get_integration() -> IntelligenceLayerIntegration:
    """Get or create global integration instance"""
    global _integration
    
    if _integration is None:
        _integration = IntelligenceLayerIntegration()
    
    return _integration


async def integrate_with_intelligence_layer(app: FastAPI) -> bool:
    """Integrate multi-source authentication with intelligence layer app"""
    try:
        integration = get_integration()
        await integration.initialize()
        integration.add_routes_to_app(app)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to integrate with intelligence layer: {e}")
        return False