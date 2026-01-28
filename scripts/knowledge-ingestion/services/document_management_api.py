"""
Document Management API for Multi-Source Support

This module extends the existing document management functionality to support
documents from multiple data sources with proper source attribution and management
capabilities. It provides unified document listing, filtering, re-processing,
and metadata management across all supported data source types.

Features:
- Extended document endpoints with multi-source metadata
- Unified document listing with source type filtering
- Document re-processing with source-appropriate methods
- Source-specific link preservation and validation
- Document metadata editing across all source types
- Document deletion with source-aware cleanup
- Document preview with source information
- Document categorization and tagging
- Document statistics and processing information
- Backward compatibility with existing document API
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from fastapi import HTTPException, Depends, Body, Path, Query
from pydantic import BaseModel, Field
import json
from pathlib import Path as PathLib

from core.config import get_settings
from core.logging import get_logger
from .multi_source_auth import DataSourceType, get_auth_service, MultiSourceAuthenticationService
from .supabase_storage import SupabaseStorageService
from .knowledge_query_interface import get_query_interface, KnowledgeQueryInterface
from .batch_ingestion_manager import get_batch_manager, BatchIngestionManager


# Pydantic models for document management

class DocumentSourceInfo(BaseModel):
    """Source information for a document"""
    source_type: str = Field(..., description="Data source type")
    source_path: str = Field(..., description="Original source path")
    source_id: str = Field(..., description="Source-specific identifier")
    access_url: Optional[str] = Field(default=None, description="Direct access URL if available")
    parent_folders: List[str] = Field(default_factory=list, description="Parent folder hierarchy")
    connection_id: Optional[str] = Field(default=None, description="Connection used to access this document")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access timestamp")
    is_accessible: bool = Field(default=True, description="Whether document is currently accessible")
    access_permissions: List[str] = Field(default_factory=list, description="Access permissions")


class DocumentMetadata(BaseModel):
    """Extended document metadata with multi-source support"""
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    file_id: str = Field(..., description="Original file identifier")
    size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    modified_time: datetime = Field(..., description="Last modified timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Source information
    source_info: DocumentSourceInfo = Field(..., description="Source information")
    
    # Processing information
    processing_status: str = Field(..., description="Processing status")
    quality_score: Optional[float] = Field(default=None, description="Processing quality score")
    parsing_method: Optional[str] = Field(default=None, description="Parsing method used")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model used")
    
    # Content classification
    domain_classification: Optional[str] = Field(default=None, description="Content domain")
    content_type: Optional[str] = Field(default=None, description="Content type classification")
    language: Optional[str] = Field(default="en", description="Document language")
    
    # User-managed metadata
    tags: List[str] = Field(default_factory=list, description="User-assigned tags")
    categories: List[str] = Field(default_factory=list, description="User-assigned categories")
    notes: Optional[str] = Field(default=None, description="User notes")
    is_favorite: bool = Field(default=False, description="Whether document is marked as favorite")
    
    # Processing statistics
    chunk_count: int = Field(default=0, description="Number of chunks created")
    processing_time_ms: Optional[int] = Field(default=None, description="Processing time in milliseconds")
    ingestion_job_id: Optional[str] = Field(default=None, description="Ingestion job that processed this document")
    
    # Additional metadata
    checksum: Optional[str] = Field(default=None, description="File checksum")
    source_specific_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")


class DocumentListRequest(BaseModel):
    """Document listing request with multi-source filtering"""
    source_types: Optional[List[str]] = Field(default=None, description="Filter by source types")
    connection_ids: Optional[List[str]] = Field(default=None, description="Filter by connection IDs")
    processing_status: Optional[List[str]] = Field(default=None, description="Filter by processing status")
    domain_classification: Optional[List[str]] = Field(default=None, description="Filter by domain")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    categories: Optional[List[str]] = Field(default=None, description="Filter by categories")
    is_favorite: Optional[bool] = Field(default=None, description="Filter by favorite status")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Date range filter")
    search_query: Optional[str] = Field(default=None, description="Search in title and content")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order: asc or desc")
    limit: Optional[int] = Field(default=50, description="Maximum number of results")
    offset: int = Field(default=0, description="Pagination offset")


class DocumentListResponse(BaseModel):
    """Document listing response"""
    documents: List[DocumentMetadata] = Field(default_factory=list, description="Document list")
    total_documents: int = Field(default=0, description="Total number of matching documents")
    by_source_type: Dict[str, int] = Field(default_factory=dict, description="Count by source type")
    by_status: Dict[str, int] = Field(default_factory=dict, description="Count by processing status")
    by_domain: Dict[str, int] = Field(default_factory=dict, description="Count by domain")
    pagination: Dict[str, Any] = Field(default_factory=dict, description="Pagination information")
    execution_time_ms: int = Field(default=0, description="Query execution time")


class DocumentUpdateRequest(BaseModel):
    """Document metadata update request"""
    title: Optional[str] = Field(default=None, description="New title")
    tags: Optional[List[str]] = Field(default=None, description="New tags")
    categories: Optional[List[str]] = Field(default=None, description="New categories")
    notes: Optional[str] = Field(default=None, description="New notes")
    is_favorite: Optional[bool] = Field(default=None, description="New favorite status")
    domain_classification: Optional[str] = Field(default=None, description="New domain classification")


class DocumentReprocessingRequest(BaseModel):
    """Document re-processing request"""
    document_id: str = Field(..., description="Document to re-process")
    processing_options: Optional[Dict[str, Any]] = Field(default=None, description="Processing options")
    force_reprocess: bool = Field(default=False, description="Force re-processing even if already completed")
    preserve_user_metadata: bool = Field(default=True, description="Preserve user-assigned metadata")


class DocumentReprocessingResponse(BaseModel):
    """Document re-processing response"""
    success: bool = Field(..., description="Whether re-processing was initiated")
    job_id: Optional[str] = Field(default=None, description="Processing job ID")
    message: str = Field(..., description="Status message")
    estimated_duration_ms: Optional[int] = Field(default=None, description="Estimated processing time")


class DocumentPreviewRequest(BaseModel):
    """Document preview request"""
    document_id: str = Field(..., description="Document identifier")
    include_chunks: bool = Field(default=True, description="Include chunk information")
    include_statistics: bool = Field(default=True, description="Include processing statistics")
    chunk_limit: int = Field(default=10, description="Maximum chunks to include")


class DocumentChunkInfo(BaseModel):
    """Document chunk information"""
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    chunk_order: int = Field(..., description="Chunk order in document")
    section_header: Optional[str] = Field(default=None, description="Section header")
    token_count: int = Field(default=0, description="Token count")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model used")
    quality_score: Optional[float] = Field(default=None, description="Chunk quality score")


class DocumentPreviewResponse(BaseModel):
    """Document preview response"""
    document: DocumentMetadata = Field(..., description="Document metadata")
    chunks: List[DocumentChunkInfo] = Field(default_factory=list, description="Document chunks")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    source_validation: Dict[str, Any] = Field(default_factory=dict, description="Source access validation")


class DocumentDeletionRequest(BaseModel):
    """Document deletion request"""
    document_ids: List[str] = Field(..., description="Documents to delete")
    delete_from_source: bool = Field(default=False, description="Also delete from original source")
    confirm_deletion: bool = Field(default=False, description="Confirmation flag")


class DocumentDeletionResponse(BaseModel):
    """Document deletion response"""
    deleted_documents: List[str] = Field(default_factory=list, description="Successfully deleted document IDs")
    failed_deletions: List[Dict[str, str]] = Field(default_factory=list, description="Failed deletions with errors")
    source_deletions: List[Dict[str, Any]] = Field(default_factory=list, description="Source deletion results")
    total_deleted: int = Field(default=0, description="Total documents deleted")


class DocumentStatisticsResponse(BaseModel):
    """Document statistics response"""
    total_documents: int = Field(default=0, description="Total document count")
    by_source_type: Dict[str, int] = Field(default_factory=dict, description="Count by source type")
    by_processing_status: Dict[str, int] = Field(default_factory=dict, description="Count by processing status")
    by_domain: Dict[str, int] = Field(default_factory=dict, description="Count by domain")
    by_quality_score: Dict[str, int] = Field(default_factory=dict, description="Count by quality score ranges")
    processing_statistics: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    storage_statistics: Dict[str, Any] = Field(default_factory=dict, description="Storage statistics")
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list, description="Recent document activity")


class DocumentManagementAPI:
    """Document management API service with multi-source support"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Service dependencies
        self._auth_service: Optional[MultiSourceAuthenticationService] = None
        self._storage_service: Optional[SupabaseStorageService] = None
        self._query_interface: Optional[KnowledgeQueryInterface] = None
        self._batch_manager: Optional[BatchIngestionManager] = None
    
    async def initialize(self) -> bool:
        """Initialize the document management API service"""
        try:
            self.logger.info("Initializing document management API")
            
            # Initialize service dependencies
            self._auth_service = get_auth_service()
            self._storage_service = SupabaseStorageService(self.settings.supabase)
            await self._storage_service.initialize_client()
            self._query_interface = get_query_interface()
            self._batch_manager = get_batch_manager()
            
            # Initialize query interface if needed
            if hasattr(self._query_interface, 'initialize'):
                await self._query_interface.initialize()
            
            self.logger.info("Document management API initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize document management API: {e}")
            return False
    
    def get_auth_service(self) -> MultiSourceAuthenticationService:
        """Dependency to get authentication service"""
        if self._auth_service is None:
            raise HTTPException(status_code=503, detail="Authentication service not initialized")
        return self._auth_service
    
    def get_storage_service(self) -> SupabaseStorageService:
        """Dependency to get storage service"""
        if self._storage_service is None:
            raise HTTPException(status_code=503, detail="Storage service not initialized")
        return self._storage_service
    
    def get_query_interface(self) -> KnowledgeQueryInterface:
        """Dependency to get query interface"""
        if self._query_interface is None:
            raise HTTPException(status_code=503, detail="Query interface not initialized")
        return self._query_interface
    
    def get_batch_manager(self) -> BatchIngestionManager:
        """Dependency to get batch manager"""
        if self._batch_manager is None:
            raise HTTPException(status_code=503, detail="Batch manager not initialized")
        return self._batch_manager
    
    async def list_documents(self, request: DocumentListRequest) -> DocumentListResponse:
        """List documents with multi-source filtering and attribution"""
        try:
            start_time = time.time()
            self.logger.info(f"Listing documents with filters: {request.dict()}")
            
            # Build query filters
            filters = {}
            
            # Source type filtering
            if request.source_types:
                filters['source_type'] = request.source_types
            
            # Processing status filtering
            if request.processing_status:
                filters['processing_status'] = request.processing_status
            
            # Domain filtering
            if request.domain_classification:
                filters['domain_classification'] = request.domain_classification
            
            # Date range filtering
            if request.date_range:
                filters['date_range'] = request.date_range
            
            # Get documents from storage
            documents_data = await self._get_documents_with_filters(
                filters=filters,
                search_query=request.search_query,
                sort_by=request.sort_by,
                sort_order=request.sort_order,
                limit=request.limit,
                offset=request.offset
            )
            
            # Convert to response format with source attribution
            documents = []
            by_source_type = {}
            by_status = {}
            by_domain = {}
            
            for doc_data in documents_data:
                # Create source info
                source_info = DocumentSourceInfo(
                    source_type=doc_data.get('source_type', 'unknown'),
                    source_path=doc_data.get('source_path', ''),
                    source_id=doc_data.get('file_id', ''),
                    access_url=doc_data.get('source_url'),
                    parent_folders=doc_data.get('parent_folders', []),
                    connection_id=doc_data.get('connection_id'),
                    is_accessible=doc_data.get('is_accessible', True),
                    access_permissions=doc_data.get('access_permissions', [])
                )
                
                # Create document metadata
                document = DocumentMetadata(
                    document_id=doc_data['id'],
                    title=doc_data['title'],
                    file_id=doc_data['file_id'],
                    size=doc_data.get('size', 0),
                    mime_type=doc_data.get('mime_type', 'application/pdf'),
                    modified_time=datetime.fromisoformat(doc_data['modified_time']) if doc_data.get('modified_time') else datetime.now(timezone.utc),
                    created_at=datetime.fromisoformat(doc_data['created_at']),
                    updated_at=datetime.fromisoformat(doc_data['updated_at']),
                    source_info=source_info,
                    processing_status=doc_data['processing_status'],
                    quality_score=doc_data.get('quality_score'),
                    parsing_method=doc_data.get('parsing_method'),
                    embedding_model=doc_data.get('embedding_model'),
                    domain_classification=doc_data.get('domain_classification'),
                    content_type=doc_data.get('content_type'),
                    language=doc_data.get('language', 'en'),
                    tags=doc_data.get('tags', []),
                    categories=doc_data.get('categories', []),
                    notes=doc_data.get('notes'),
                    is_favorite=doc_data.get('is_favorite', False),
                    chunk_count=doc_data.get('chunk_count', 0),
                    processing_time_ms=doc_data.get('processing_time_ms'),
                    ingestion_job_id=doc_data.get('ingestion_job_id'),
                    checksum=doc_data.get('checksum'),
                    source_specific_metadata=doc_data.get('source_specific_metadata', {})
                )
                
                documents.append(document)
                
                # Update statistics
                source_type = source_info.source_type
                status = document.processing_status
                domain = document.domain_classification or 'unknown'
                
                by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1
                by_domain[domain] = by_domain.get(domain, 0) + 1
            
            # Get total count for pagination
            total_count = await self._get_documents_count(filters, request.search_query)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return DocumentListResponse(
                documents=documents,
                total_documents=total_count,
                by_source_type=by_source_type,
                by_status=by_status,
                by_domain=by_domain,
                pagination={
                    'offset': request.offset,
                    'limit': request.limit,
                    'has_more': len(documents) == request.limit,
                    'total_pages': (total_count + request.limit - 1) // request.limit if request.limit else 1
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
    
    async def get_document_by_id(self, document_id: str) -> DocumentMetadata:
        """Get document by ID with full multi-source metadata"""
        try:
            self.logger.info(f"Getting document by ID: {document_id}")
            
            # Get document from query interface
            document_data = await self._query_interface.get_document_by_id(document_id)
            
            if not document_data:
                raise HTTPException(status_code=404, detail="Document not found")
            
            doc = document_data['document']
            
            # Create source info
            source_info = DocumentSourceInfo(
                source_type=doc.get('source_type', 'unknown'),
                source_path=doc.get('source_path', ''),
                source_id=doc.get('file_id', ''),
                access_url=doc.get('source_url'),
                parent_folders=doc.get('parent_folders', []),
                connection_id=doc.get('connection_id'),
                is_accessible=doc.get('is_accessible', True),
                access_permissions=doc.get('access_permissions', [])
            )
            
            # Validate source access
            if source_info.connection_id:
                try:
                    validation_result = await self._validate_document_access(document_id, source_info.connection_id)
                    source_info.is_accessible = validation_result.get('is_accessible', True)
                    source_info.access_permissions = validation_result.get('permissions', [])
                except Exception as e:
                    self.logger.warning(f"Failed to validate document access: {e}")
            
            # Create document metadata
            document = DocumentMetadata(
                document_id=doc['id'],
                title=doc['title'],
                file_id=doc['file_id'],
                size=doc.get('size', 0),
                mime_type=doc.get('mime_type', 'application/pdf'),
                modified_time=datetime.fromisoformat(doc['modified_time']) if doc.get('modified_time') else datetime.now(timezone.utc),
                created_at=datetime.fromisoformat(doc['created_at']),
                updated_at=datetime.fromisoformat(doc['updated_at']),
                source_info=source_info,
                processing_status=doc['processing_status'],
                quality_score=doc.get('quality_score'),
                parsing_method=doc.get('parsing_method'),
                embedding_model=doc.get('embedding_model'),
                domain_classification=doc.get('domain_classification'),
                content_type=doc.get('content_type'),
                language=doc.get('language', 'en'),
                tags=doc.get('tags', []),
                categories=doc.get('categories', []),
                notes=doc.get('notes'),
                is_favorite=doc.get('is_favorite', False),
                chunk_count=len(document_data.get('chunks', [])),
                processing_time_ms=doc.get('processing_time_ms'),
                ingestion_job_id=doc.get('ingestion_job_id'),
                checksum=doc.get('checksum'),
                source_specific_metadata=doc.get('source_specific_metadata', {})
            )
            
            return document
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")
    
    async def update_document_metadata(self, document_id: str, request: DocumentUpdateRequest) -> DocumentMetadata:
        """Update document metadata with multi-source awareness"""
        try:
            self.logger.info(f"Updating document metadata: {document_id}")
            
            # Get current document
            current_doc = await self.get_document_by_id(document_id)
            
            # Prepare update data
            update_data = {}
            
            if request.title is not None:
                update_data['title'] = request.title
            
            if request.tags is not None:
                update_data['tags'] = request.tags
            
            if request.categories is not None:
                update_data['categories'] = request.categories
            
            if request.notes is not None:
                update_data['notes'] = request.notes
            
            if request.is_favorite is not None:
                update_data['is_favorite'] = request.is_favorite
            
            if request.domain_classification is not None:
                update_data['domain_classification'] = request.domain_classification
            
            # Update in storage
            result = await self._update_document_in_storage(document_id, update_data)
            
            if not result:
                raise HTTPException(status_code=500, detail="Failed to update document")
            
            # Return updated document
            return await self.get_document_by_id(document_id)
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update document {document_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")
    
    async def reprocess_document(self, request: DocumentReprocessingRequest) -> DocumentReprocessingResponse:
        """Re-process document with source-appropriate methods"""
        try:
            self.logger.info(f"Re-processing document: {request.document_id}")
            
            # Get current document
            document = await self.get_document_by_id(request.document_id)
            
            # Check if re-processing is needed
            if not request.force_reprocess and document.processing_status == 'completed':
                return DocumentReprocessingResponse(
                    success=False,
                    message="Document already processed. Use force_reprocess=true to override."
                )
            
            # Validate source access
            if not document.source_info.is_accessible:
                return DocumentReprocessingResponse(
                    success=False,
                    message="Document source is not accessible for re-processing"
                )
            
            # Create re-processing job
            file_selections = [{
                'file_id': document.file_id,
                'source_type': document.source_info.source_type,
                'connection_id': document.source_info.connection_id,
                'source_path': document.source_info.source_path,
                'reprocess': True,
                'preserve_user_metadata': request.preserve_user_metadata
            }]
            
            # Create batch job for re-processing
            job_id = await self._batch_manager.create_job(
                user_id="system",  # System-initiated re-processing
                name=f"Reprocess: {document.title}",
                file_selections=file_selections,
                processing_options=request.processing_options,
                description=f"Re-processing document {document.document_id}"
            )
            
            # Start the job
            await self._batch_manager.start_job(job_id, "system")
            
            return DocumentReprocessingResponse(
                success=True,
                job_id=job_id,
                message="Document re-processing initiated successfully",
                estimated_duration_ms=self._estimate_processing_time(document)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to reprocess document {request.document_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to reprocess document: {str(e)}")
    
    async def get_document_preview(self, request: DocumentPreviewRequest) -> DocumentPreviewResponse:
        """Get document preview with processing statistics and source information"""
        try:
            self.logger.info(f"Getting document preview: {request.document_id}")
            
            # Get document metadata
            document = await self.get_document_by_id(request.document_id)
            
            # Get chunks if requested
            chunks = []
            if request.include_chunks:
                chunks_data = await self._storage_service.get_chunks_by_document_id(request.document_id)
                
                for chunk_data in chunks_data[:request.chunk_limit]:
                    chunk_info = DocumentChunkInfo(
                        chunk_id=chunk_data['id'],
                        content=chunk_data['content'][:500] + "..." if len(chunk_data['content']) > 500 else chunk_data['content'],
                        chunk_order=chunk_data.get('chunk_order', 0),
                        section_header=chunk_data.get('section_header'),
                        token_count=chunk_data.get('token_count', 0),
                        embedding_model=chunk_data.get('embedding_model'),
                        quality_score=chunk_data.get('quality_score')
                    )
                    chunks.append(chunk_info)
            
            # Get processing statistics if requested
            statistics = {}
            if request.include_statistics:
                statistics = await self._get_document_statistics(request.document_id)
            
            # Validate source access
            source_validation = {}
            if document.source_info.connection_id:
                try:
                    source_validation = await self._validate_document_access(
                        request.document_id, 
                        document.source_info.connection_id
                    )
                except Exception as e:
                    source_validation = {
                        'is_accessible': False,
                        'error': str(e)
                    }
            
            return DocumentPreviewResponse(
                document=document,
                chunks=chunks,
                statistics=statistics,
                source_validation=source_validation
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get document preview {request.document_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get document preview: {str(e)}")
    
    async def delete_documents(self, request: DocumentDeletionRequest) -> DocumentDeletionResponse:
        """Delete documents with source-aware cleanup"""
        try:
            self.logger.info(f"Deleting documents: {request.document_ids}")
            
            if not request.confirm_deletion:
                raise HTTPException(status_code=400, detail="Deletion must be confirmed")
            
            deleted_documents = []
            failed_deletions = []
            source_deletions = []
            
            for document_id in request.document_ids:
                try:
                    # Get document info
                    document = await self.get_document_by_id(document_id)
                    
                    # Delete from source if requested
                    if request.delete_from_source and document.source_info.is_accessible:
                        source_result = await self._delete_from_source(document)
                        source_deletions.append({
                            'document_id': document_id,
                            'source_type': document.source_info.source_type,
                            'success': source_result.get('success', False),
                            'message': source_result.get('message', '')
                        })
                    
                    # Delete from storage (this will cascade to chunks)
                    storage_result = await self._delete_document_from_storage(document_id)
                    
                    if storage_result:
                        deleted_documents.append(document_id)
                        self.logger.info(f"Successfully deleted document: {document_id}")
                    else:
                        failed_deletions.append({
                            'document_id': document_id,
                            'error': 'Failed to delete from storage'
                        })
                
                except Exception as e:
                    failed_deletions.append({
                        'document_id': document_id,
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to delete document {document_id}: {e}")
            
            return DocumentDeletionResponse(
                deleted_documents=deleted_documents,
                failed_deletions=failed_deletions,
                source_deletions=source_deletions,
                total_deleted=len(deleted_documents)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")
    
    async def get_document_statistics(self) -> DocumentStatisticsResponse:
        """Get comprehensive document statistics across all sources"""
        try:
            self.logger.info("Getting document statistics")
            
            # Get basic counts
            stats = await self._get_comprehensive_statistics()
            
            return DocumentStatisticsResponse(
                total_documents=stats['total_documents'],
                by_source_type=stats['by_source_type'],
                by_processing_status=stats['by_processing_status'],
                by_domain=stats['by_domain'],
                by_quality_score=stats['by_quality_score'],
                processing_statistics=stats['processing_statistics'],
                storage_statistics=stats['storage_statistics'],
                recent_activity=stats['recent_activity']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get document statistics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
    
    # Helper methods
    
    async def _get_documents_with_filters(
        self, 
        filters: Dict[str, Any], 
        search_query: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get documents from storage with filters"""
        # This would be implemented to query the Supabase storage
        # For now, return a placeholder implementation
        try:
            # Build query based on filters
            query = self._storage_service.client.table('documents').select('*')
            
            # Apply filters
            if 'source_type' in filters:
                query = query.in_('source_type', filters['source_type'])
            
            if 'processing_status' in filters:
                query = query.in_('processing_status', filters['processing_status'])
            
            if 'domain_classification' in filters:
                query = query.in_('domain_classification', filters['domain_classification'])
            
            # Apply search
            if search_query:
                query = query.ilike('title', f'%{search_query}%')
            
            # Apply sorting
            if sort_order == 'desc':
                query = query.order(sort_by, desc=True)
            else:
                query = query.order(sort_by)
            
            # Apply pagination
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            self.logger.error(f"Failed to get documents with filters: {e}")
            return []
    
    async def _get_documents_count(self, filters: Dict[str, Any], search_query: Optional[str] = None) -> int:
        """Get total count of documents matching filters"""
        try:
            query = self._storage_service.client.table('documents').select('id', count='exact')
            
            # Apply same filters as in _get_documents_with_filters
            if 'source_type' in filters:
                query = query.in_('source_type', filters['source_type'])
            
            if 'processing_status' in filters:
                query = query.in_('processing_status', filters['processing_status'])
            
            if 'domain_classification' in filters:
                query = query.in_('domain_classification', filters['domain_classification'])
            
            if search_query:
                query = query.ilike('title', f'%{search_query}%')
            
            result = query.execute()
            return result.count or 0
            
        except Exception as e:
            self.logger.error(f"Failed to get documents count: {e}")
            return 0
    
    async def _validate_document_access(self, document_id: str, connection_id: str) -> Dict[str, Any]:
        """Validate document access through its source"""
        try:
            # Get connection info
            connection = await self._auth_service.get_connection_status(connection_id)
            
            if not connection or connection.status.value != 'connected':
                return {
                    'is_accessible': False,
                    'error': 'Connection not available'
                }
            
            # For now, assume accessible if connection is valid
            return {
                'is_accessible': True,
                'permissions': connection.permissions
            }
            
        except Exception as e:
            return {
                'is_accessible': False,
                'error': str(e)
            }
    
    async def _update_document_in_storage(self, document_id: str, update_data: Dict[str, Any]) -> bool:
        """Update document in storage"""
        try:
            update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            result = self._storage_service.client.table('documents').update(update_data).eq('id', document_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"Failed to update document in storage: {e}")
            return False
    
    async def _get_document_statistics(self, document_id: str) -> Dict[str, Any]:
        """Get processing statistics for a document"""
        try:
            # Get chunks count and statistics
            chunks = await self._storage_service.get_chunks_by_document_id(document_id)
            
            stats = {
                'chunk_count': len(chunks),
                'total_tokens': sum(chunk.get('token_count', 0) for chunk in chunks),
                'embedding_models_used': list(set(chunk.get('embedding_model') for chunk in chunks if chunk.get('embedding_model'))),
                'average_chunk_quality': sum(chunk.get('quality_score', 0) for chunk in chunks) / len(chunks) if chunks else 0,
                'processing_phases': {
                    'parsing': 'completed',
                    'chunking': 'completed',
                    'embedding': 'completed',
                    'storage': 'completed'
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get document statistics: {e}")
            return {}
    
    async def _delete_from_source(self, document: DocumentMetadata) -> Dict[str, Any]:
        """Delete document from its original source"""
        try:
            source_type = document.source_info.source_type
            
            # For now, we don't actually delete from sources
            # This would be implemented based on source type
            return {
                'success': False,
                'message': f'Source deletion not implemented for {source_type}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    async def _delete_document_from_storage(self, document_id: str) -> bool:
        """Delete document from storage (cascades to chunks)"""
        try:
            result = self._storage_service.client.table('documents').delete().eq('id', document_id).execute()
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"Failed to delete document from storage: {e}")
            return False
    
    async def _get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all documents"""
        try:
            # Get all documents
            all_docs = self._storage_service.client.table('documents').select('*').execute()
            documents = all_docs.data or []
            
            # Calculate statistics
            stats = {
                'total_documents': len(documents),
                'by_source_type': {},
                'by_processing_status': {},
                'by_domain': {},
                'by_quality_score': {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0},
                'processing_statistics': {
                    'total_chunks': 0,
                    'average_processing_time': 0,
                    'success_rate': 0
                },
                'storage_statistics': {
                    'total_size_bytes': 0,
                    'average_document_size': 0
                },
                'recent_activity': []
            }
            
            total_processing_time = 0
            successful_docs = 0
            total_size = 0
            
            for doc in documents:
                # Source type stats
                source_type = doc.get('source_type', 'unknown')
                stats['by_source_type'][source_type] = stats['by_source_type'].get(source_type, 0) + 1
                
                # Processing status stats
                status = doc.get('processing_status', 'unknown')
                stats['by_processing_status'][status] = stats['by_processing_status'].get(status, 0) + 1
                
                # Domain stats
                domain = doc.get('domain_classification', 'unknown')
                stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1
                
                # Quality score stats
                quality = doc.get('quality_score')
                if quality is None:
                    stats['by_quality_score']['unknown'] += 1
                elif quality >= 0.8:
                    stats['by_quality_score']['high'] += 1
                elif quality >= 0.6:
                    stats['by_quality_score']['medium'] += 1
                else:
                    stats['by_quality_score']['low'] += 1
                
                # Processing statistics
                if status == 'completed':
                    successful_docs += 1
                
                processing_time = doc.get('processing_time_ms', 0)
                if processing_time:
                    total_processing_time += processing_time
                
                # Storage statistics
                size = doc.get('size', 0)
                total_size += size
            
            # Calculate averages
            if documents:
                stats['processing_statistics']['average_processing_time'] = total_processing_time / len(documents)
                stats['processing_statistics']['success_rate'] = (successful_docs / len(documents)) * 100
                stats['storage_statistics']['average_document_size'] = total_size / len(documents)
            
            stats['storage_statistics']['total_size_bytes'] = total_size
            
            # Recent activity (last 10 documents by updated_at)
            recent_docs = sorted(documents, key=lambda x: x.get('updated_at', ''), reverse=True)[:10]
            stats['recent_activity'] = [
                {
                    'document_id': doc['id'],
                    'title': doc['title'],
                    'action': 'updated',
                    'timestamp': doc.get('updated_at'),
                    'source_type': doc.get('source_type')
                } for doc in recent_docs
            ]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive statistics: {e}")
            return {
                'total_documents': 0,
                'by_source_type': {},
                'by_processing_status': {},
                'by_domain': {},
                'by_quality_score': {},
                'processing_statistics': {},
                'storage_statistics': {},
                'recent_activity': []
            }
    
    def _estimate_processing_time(self, document: DocumentMetadata) -> int:
        """Estimate processing time for a document"""
        # Base time estimate based on document size
        base_time = max(5000, document.size // 1000)  # 5 seconds minimum, 1ms per KB
        
        # Adjust based on source type
        source_multiplier = {
            'google_drive': 1.2,  # Network overhead
            'local_directory': 1.0,
            'local_zip': 1.1,
            'individual_upload': 1.0,
            'aws_s3': 1.3,
            'azure_blob': 1.3,
            'google_cloud_storage': 1.3
        }
        
        multiplier = source_multiplier.get(document.source_info.source_type, 1.0)
        
        return int(base_time * multiplier)


# Global service instance
_document_management_api: Optional[DocumentManagementAPI] = None


def get_document_management_api() -> DocumentManagementAPI:
    """Get or create global document management API instance"""
    global _document_management_api
    
    if _document_management_api is None:
        _document_management_api = DocumentManagementAPI()
    
    return _document_management_api