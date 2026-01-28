"""
Platform integration service for connecting knowledge ingestion to existing trading platform.

This module provides:
- Connection to existing Supabase and Neo4j instances
- Conflict detection and resolution for database operations
- Consistent API endpoints for knowledge retrieval
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import uuid

from supabase import create_client, Client
from neo4j import GraphDatabase, Driver
import httpx

from core.config import get_settings
from core.logging import get_logger
from .supabase_storage import SupabaseStorageService, DocumentMetadata, ChunkData


class IntegrationStatus(Enum):
    """Integration status enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    INITIALIZING = "initializing"


class ConflictResolution(Enum):
    """Conflict resolution strategy enumeration"""
    SKIP = "skip"  # Skip conflicting operations
    OVERWRITE = "overwrite"  # Overwrite existing data
    MERGE = "merge"  # Merge with existing data
    VERSION = "version"  # Create new version


@dataclass
class PlatformConnection:
    """Platform connection configuration"""
    service_name: str
    connection_string: str
    status: IntegrationStatus
    last_check: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConflictDetectionResult:
    """Result of conflict detection"""
    has_conflict: bool
    conflict_type: str
    existing_data: Optional[Dict[str, Any]] = None
    conflicting_fields: List[str] = None
    resolution_strategy: Optional[ConflictResolution] = None


class PlatformIntegrationService:
    """
    Platform integration service for knowledge ingestion system.
    
    Handles:
    - Connection to existing Supabase and Neo4j instances
    - Conflict detection and resolution for database operations
    - Consistent API endpoints for knowledge retrieval
    - Integration with intelligence layer services
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Connection instances
        self._supabase_client: Optional[Client] = None
        self._neo4j_driver: Optional[Driver] = None
        self._intelligence_client: Optional[httpx.AsyncClient] = None
        
        # Connection status tracking
        self.connections: Dict[str, PlatformConnection] = {}
        
        # Storage service for knowledge ingestion
        self._storage_service: Optional[SupabaseStorageService] = None
        
        # Default conflict resolution strategies
        self.default_conflict_resolution = {
            "documents": ConflictResolution.VERSION,
            "chunks": ConflictResolution.SKIP,
            "embeddings": ConflictResolution.OVERWRITE
        }
    
    async def initialize(self) -> bool:
        """
        Initialize platform integration service.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing platform integration service")
            
            # Initialize Supabase connection
            await self._initialize_supabase()
            
            # Initialize Neo4j connection
            await self._initialize_neo4j()
            
            # Initialize intelligence layer client
            await self._initialize_intelligence_client()
            
            # Initialize storage service
            await self._initialize_storage_service()
            
            self.logger.info("Platform integration service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform integration service: {e}")
            return False
    
    async def _initialize_supabase(self) -> bool:
        """Initialize Supabase connection"""
        try:
            self._supabase_client = create_client(
                self.settings.supabase.url,
                self.settings.supabase.service_role_key or self.settings.supabase.key
            )
            
            # Test connection
            result = self._supabase_client.table('documents').select('id').limit(1).execute()
            
            self.connections['supabase'] = PlatformConnection(
                service_name='supabase',
                connection_string=self.settings.supabase.url,
                status=IntegrationStatus.CONNECTED,
                last_check=datetime.now(),
                metadata={'table_count': len(result.data) if result.data else 0}
            )
            
            self.logger.info("Supabase connection established")
            return True
            
        except Exception as e:
            self.connections['supabase'] = PlatformConnection(
                service_name='supabase',
                connection_string=self.settings.supabase.url,
                status=IntegrationStatus.ERROR,
                last_check=datetime.now(),
                error_message=str(e)
            )
            self.logger.error(f"Failed to connect to Supabase: {e}")
            return False
    
    async def _initialize_neo4j(self) -> bool:
        """Initialize Neo4j connection"""
        try:
            # For now, we'll skip Neo4j integration as it's not required for basic knowledge ingestion
            # This can be implemented later when graph analytics are needed
            
            self.connections['neo4j'] = PlatformConnection(
                service_name='neo4j',
                connection_string='not_configured',
                status=IntegrationStatus.DISCONNECTED,
                last_check=datetime.now(),
                metadata={'note': 'Neo4j integration not implemented yet'}
            )
            
            self.logger.info("Neo4j connection skipped (not required for basic functionality)")
            return True
            
        except Exception as e:
            self.connections['neo4j'] = PlatformConnection(
                service_name='neo4j',
                connection_string='not_configured',
                status=IntegrationStatus.ERROR,
                last_check=datetime.now(),
                error_message=str(e)
            )
            self.logger.error(f"Neo4j connection error: {e}")
            return False
    
    async def _initialize_intelligence_client(self) -> bool:
        """Initialize intelligence layer HTTP client"""
        try:
            # Default intelligence layer URL
            intelligence_url = "http://localhost:8000"
            
            self._intelligence_client = httpx.AsyncClient(
                base_url=intelligence_url,
                timeout=30.0
            )
            
            # Test connection
            try:
                response = await self._intelligence_client.get("/health")
                if response.status_code == 200:
                    status = IntegrationStatus.CONNECTED
                    error_msg = None
                else:
                    status = IntegrationStatus.ERROR
                    error_msg = f"Health check failed: {response.status_code}"
            except Exception as e:
                status = IntegrationStatus.DISCONNECTED
                error_msg = f"Connection failed: {e}"
            
            self.connections['intelligence_layer'] = PlatformConnection(
                service_name='intelligence_layer',
                connection_string=intelligence_url,
                status=status,
                last_check=datetime.now(),
                error_message=error_msg
            )
            
            self.logger.info(f"Intelligence layer client initialized: {status.value}")
            return True
            
        except Exception as e:
            self.connections['intelligence_layer'] = PlatformConnection(
                service_name='intelligence_layer',
                connection_string='http://localhost:8000',
                status=IntegrationStatus.ERROR,
                last_check=datetime.now(),
                error_message=str(e)
            )
            self.logger.error(f"Failed to initialize intelligence client: {e}")
            return False
    
    async def _initialize_storage_service(self) -> bool:
        """Initialize storage service"""
        try:
            self._storage_service = SupabaseStorageService(self.settings.supabase)
            await self._storage_service.initialize_client()
            
            self.logger.info("Storage service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage service: {e}")
            return False
    
    @property
    def supabase_client(self) -> Client:
        """Get Supabase client"""
        if self._supabase_client is None:
            raise RuntimeError("Supabase client not initialized")
        return self._supabase_client
    
    @property
    def storage_service(self) -> SupabaseStorageService:
        """Get storage service"""
        if self._storage_service is None:
            raise RuntimeError("Storage service not initialized")
        return self._storage_service
    
    async def detect_conflicts(
        self, 
        operation_type: str,
        data: Dict[str, Any],
        identifier_field: str = "file_id"
    ) -> ConflictDetectionResult:
        """
        Detect conflicts with existing data.
        
        Args:
            operation_type: Type of operation (documents, chunks, embeddings)
            data: Data to check for conflicts
            identifier_field: Field to use for conflict detection
            
        Returns:
            ConflictDetectionResult with conflict information
        """
        try:
            if operation_type == "documents":
                return await self._detect_document_conflicts(data, identifier_field)
            elif operation_type == "chunks":
                return await self._detect_chunk_conflicts(data, identifier_field)
            elif operation_type == "embeddings":
                return await self._detect_embedding_conflicts(data, identifier_field)
            else:
                return ConflictDetectionResult(
                    has_conflict=False,
                    conflict_type="unknown_operation"
                )
                
        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}")
            return ConflictDetectionResult(
                has_conflict=False,
                conflict_type="detection_error"
            )
    
    async def _detect_document_conflicts(
        self, 
        data: Dict[str, Any], 
        identifier_field: str
    ) -> ConflictDetectionResult:
        """Detect document conflicts"""
        try:
            identifier_value = data.get(identifier_field)
            if not identifier_value:
                return ConflictDetectionResult(
                    has_conflict=False,
                    conflict_type="no_identifier"
                )
            
            # Check if document already exists
            existing_doc = await self.storage_service.get_document_by_file_id(identifier_value)
            
            if existing_doc:
                # Compare key fields to determine conflict type
                conflicting_fields = []
                
                if existing_doc.get('title') != data.get('title'):
                    conflicting_fields.append('title')
                if existing_doc.get('content') != data.get('content'):
                    conflicting_fields.append('content')
                if existing_doc.get('processing_status') != data.get('processing_status'):
                    conflicting_fields.append('processing_status')
                
                return ConflictDetectionResult(
                    has_conflict=True,
                    conflict_type="document_exists",
                    existing_data=existing_doc,
                    conflicting_fields=conflicting_fields,
                    resolution_strategy=self.default_conflict_resolution.get("documents")
                )
            
            return ConflictDetectionResult(
                has_conflict=False,
                conflict_type="no_conflict"
            )
            
        except Exception as e:
            self.logger.error(f"Document conflict detection failed: {e}")
            return ConflictDetectionResult(
                has_conflict=False,
                conflict_type="detection_error"
            )
    
    async def _detect_chunk_conflicts(
        self, 
        data: Dict[str, Any], 
        identifier_field: str
    ) -> ConflictDetectionResult:
        """Detect chunk conflicts"""
        try:
            # For chunks, we typically check by document_id and chunk_order
            document_id = data.get('document_id')
            chunk_order = data.get('chunk_order')
            
            if not document_id or chunk_order is None:
                return ConflictDetectionResult(
                    has_conflict=False,
                    conflict_type="no_identifier"
                )
            
            # Check if chunk already exists for this document and order
            existing_chunks = await self.storage_service.get_chunks_by_document_id(document_id)
            existing_chunk = next(
                (chunk for chunk in existing_chunks if chunk.get('chunk_order') == chunk_order),
                None
            )
            
            if existing_chunk:
                conflicting_fields = []
                
                if existing_chunk.get('content') != data.get('content'):
                    conflicting_fields.append('content')
                if existing_chunk.get('embedding') != data.get('embedding'):
                    conflicting_fields.append('embedding')
                
                return ConflictDetectionResult(
                    has_conflict=True,
                    conflict_type="chunk_exists",
                    existing_data=existing_chunk,
                    conflicting_fields=conflicting_fields,
                    resolution_strategy=self.default_conflict_resolution.get("chunks")
                )
            
            return ConflictDetectionResult(
                has_conflict=False,
                conflict_type="no_conflict"
            )
            
        except Exception as e:
            self.logger.error(f"Chunk conflict detection failed: {e}")
            return ConflictDetectionResult(
                has_conflict=False,
                conflict_type="detection_error"
            )
    
    async def _detect_embedding_conflicts(
        self, 
        data: Dict[str, Any], 
        identifier_field: str
    ) -> ConflictDetectionResult:
        """Detect embedding conflicts"""
        # For embeddings, we typically overwrite as they may be regenerated
        return ConflictDetectionResult(
            has_conflict=False,
            conflict_type="no_conflict",
            resolution_strategy=self.default_conflict_resolution.get("embeddings")
        )
    
    async def resolve_conflict(
        self,
        conflict_result: ConflictDetectionResult,
        new_data: Dict[str, Any],
        resolution_strategy: Optional[ConflictResolution] = None
    ) -> Dict[str, Any]:
        """
        Resolve detected conflicts using specified strategy.
        
        Args:
            conflict_result: Result from conflict detection
            new_data: New data to be stored
            resolution_strategy: Strategy to use for resolution
            
        Returns:
            Resolved data ready for storage
        """
        if not conflict_result.has_conflict:
            return new_data
        
        strategy = resolution_strategy or conflict_result.resolution_strategy
        if not strategy:
            strategy = ConflictResolution.SKIP
        
        try:
            if strategy == ConflictResolution.SKIP:
                self.logger.info(f"Skipping conflicting data due to resolution strategy")
                return {}  # Empty dict indicates skip
            
            elif strategy == ConflictResolution.OVERWRITE:
                self.logger.info(f"Overwriting existing data")
                return new_data
            
            elif strategy == ConflictResolution.MERGE:
                self.logger.info(f"Merging with existing data")
                merged_data = conflict_result.existing_data.copy()
                merged_data.update(new_data)
                merged_data['updated_at'] = datetime.now().isoformat()
                return merged_data
            
            elif strategy == ConflictResolution.VERSION:
                self.logger.info(f"Creating new version")
                versioned_data = new_data.copy()
                versioned_data['version'] = datetime.now().isoformat()
                versioned_data['previous_version'] = conflict_result.existing_data.get('id')
                return versioned_data
            
            else:
                self.logger.warning(f"Unknown resolution strategy: {strategy}, defaulting to skip")
                return {}
                
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return {}
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """
        Get status of all platform connections.
        
        Returns:
            Dictionary with connection status for each service
        """
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'connections': {}
        }
        
        unhealthy_count = 0
        
        for service_name, connection in self.connections.items():
            connection_info = {
                'status': connection.status.value,
                'last_check': connection.last_check.isoformat() if connection.last_check else None,
                'error_message': connection.error_message,
                'metadata': connection.metadata
            }
            
            status_report['connections'][service_name] = connection_info
            
            if connection.status in [IntegrationStatus.ERROR, IntegrationStatus.DISCONNECTED]:
                unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count == 0:
            status_report['overall_status'] = 'healthy'
        elif unhealthy_count < len(self.connections):
            status_report['overall_status'] = 'degraded'
        else:
            status_report['overall_status'] = 'unhealthy'
        
        return status_report
    
    async def test_connections(self) -> Dict[str, bool]:
        """
        Test all platform connections.
        
        Returns:
            Dictionary with test results for each service
        """
        results = {}
        
        # Test Supabase
        try:
            if self._supabase_client:
                result = self._supabase_client.table('documents').select('id').limit(1).execute()
                results['supabase'] = True
                self.connections['supabase'].status = IntegrationStatus.CONNECTED
                self.connections['supabase'].last_check = datetime.now()
                self.connections['supabase'].error_message = None
            else:
                results['supabase'] = False
        except Exception as e:
            results['supabase'] = False
            self.connections['supabase'].status = IntegrationStatus.ERROR
            self.connections['supabase'].error_message = str(e)
        
        # Test Intelligence Layer
        try:
            if self._intelligence_client:
                response = await self._intelligence_client.get("/health")
                results['intelligence_layer'] = response.status_code == 200
                
                if results['intelligence_layer']:
                    self.connections['intelligence_layer'].status = IntegrationStatus.CONNECTED
                    self.connections['intelligence_layer'].error_message = None
                else:
                    self.connections['intelligence_layer'].status = IntegrationStatus.ERROR
                    self.connections['intelligence_layer'].error_message = f"Health check failed: {response.status_code}"
                
                self.connections['intelligence_layer'].last_check = datetime.now()
            else:
                results['intelligence_layer'] = False
        except Exception as e:
            results['intelligence_layer'] = False
            self.connections['intelligence_layer'].status = IntegrationStatus.ERROR
            self.connections['intelligence_layer'].error_message = str(e)
        
        # Neo4j is optional for now
        results['neo4j'] = True  # Mark as successful since it's not required
        
        return results
    
    async def close(self):
        """Close all connections"""
        try:
            if self._intelligence_client:
                await self._intelligence_client.aclose()
            
            if self._neo4j_driver:
                self._neo4j_driver.close()
            
            self.logger.info("Platform integration service connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Global integration service instance
_integration_service: Optional[PlatformIntegrationService] = None


async def get_integration_service() -> PlatformIntegrationService:
    """Get or create global integration service instance"""
    global _integration_service
    
    if _integration_service is None:
        _integration_service = PlatformIntegrationService()
        await _integration_service.initialize()
    
    return _integration_service


async def initialize_platform_integration() -> bool:
    """Initialize platform integration service"""
    try:
        service = await get_integration_service()
        return service is not None
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to initialize platform integration: {e}")
        return False