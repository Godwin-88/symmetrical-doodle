"""
Supabase storage services for documents and chunks.

This module handles document metadata storage, chunk storage with embedding vectors,
transaction management, and ingestion status tracking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import uuid

from supabase import create_client, Client
import numpy as np

from core.config import SupabaseConfig
from core.logging import get_logger


class ProcessingStatus(Enum):
    """Document processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IngestionPhase(Enum):
    """Ingestion pipeline phase enumeration"""
    DISCOVERY = "discovery"
    DOWNLOAD = "download"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    AUDIT = "audit"


class IngestionStatus(Enum):
    """Ingestion operation status enumeration"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class DocumentMetadata:
    """Document metadata for storage"""
    file_id: str
    title: str
    source_url: Optional[str] = None
    content: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None
    parsing_method: Optional[str] = None
    quality_score: Optional[float] = None
    domain_classification: Optional[str] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    file_size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        data['processing_status'] = self.processing_status.value
        
        # Clean content to remove null bytes and other problematic characters
        if data.get('content'):
            data['content'] = self._clean_content(data['content'])
        
        return data
    
    def _clean_content(self, content: str) -> str:
        """Clean content to remove problematic characters for database storage"""
        if not content:
            return content
        
        # Remove null bytes and other control characters that cause database issues
        cleaned = content.replace('\x00', '')  # Remove null bytes
        
        # Remove other problematic control characters but keep useful ones like newlines and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
        
        # Ensure the content is valid UTF-8
        try:
            cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')
        except UnicodeError:
            # If there are still encoding issues, use a more aggressive approach
            cleaned = cleaned.encode('ascii', errors='ignore').decode('ascii')
        
        return cleaned


@dataclass
class ChunkData:
    """Chunk data for storage"""
    document_id: str
    content: str
    chunk_order: int
    section_header: Optional[str] = None
    semantic_metadata: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    quality_score: Optional[float] = None
    math_elements: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        
        # Clean content to remove null bytes and other problematic characters
        if data.get('content'):
            data['content'] = self._clean_content(data['content'])
        
        # Also clean section_header if present
        if data.get('section_header'):
            data['section_header'] = self._clean_content(data['section_header'])
        
        return data
    
    def _clean_content(self, content: str) -> str:
        """Clean content to remove problematic characters for database storage"""
        if not content:
            return content
        
        # Remove null bytes and other control characters that cause database issues
        cleaned = content.replace('\x00', '')  # Remove null bytes
        
        # Remove other problematic control characters but keep useful ones like newlines and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
        
        # Ensure the content is valid UTF-8
        try:
            cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')
        except UnicodeError:
            # If there are still encoding issues, use a more aggressive approach
            cleaned = cleaned.encode('ascii', errors='ignore').decode('ascii')
        
        return cleaned


@dataclass
class EmbeddedChunk:
    """Chunk with embedding vector"""
    chunk_data: ChunkData
    embedding_vector: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = self.chunk_data.to_dict()
        data['embedding'] = self.embedding_vector
        return data


@dataclass
class IngestionLogEntry:
    """Ingestion log entry"""
    file_id: str
    phase: IngestionPhase
    status: IngestionStatus
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    processing_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        data['phase'] = self.phase.value
        data['status'] = self.status.value
        return data


@dataclass
class StorageResult:
    """Result of storage operation"""
    success: bool
    record_id: Optional[str] = None
    error_message: Optional[str] = None
    affected_rows: int = 0


@dataclass
class TransactionResult:
    """Result of transaction operation"""
    success: bool
    document_id: Optional[str] = None
    chunk_ids: List[str] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False


class SupabaseStorageService:
    """
    Supabase storage service for documents and chunks.
    
    Handles:
    - Document metadata storage with all required fields
    - Chunk storage with embedding vectors and semantic metadata
    - Transaction management for atomic operations
    - Ingestion status tracking and logging
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self._client: Optional[Client] = None
        
    async def initialize_client(self) -> bool:
        """Initialize Supabase client"""
        try:
            self._client = create_client(
                self.config.url,
                self.config.service_role_key or self.config.key
            )
            
            self.logger.info("Supabase storage client initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase client: {e}")
            return False
    
    @property
    def client(self) -> Client:
        """Get Supabase client, initializing if necessary"""
        if self._client is None:
            raise RuntimeError("Supabase client not initialized. Call initialize_client() first.")
        return self._client
    
    async def store_document(self, document: DocumentMetadata) -> StorageResult:
        """
        Store document metadata in the documents table.
        
        Args:
            document: Document metadata to store
            
        Returns:
            StorageResult: Storage operation result with document ID
        """
        try:
            self.logger.info(f"Storing document metadata for file_id: {document.file_id}")
            
            # Prepare document data
            doc_data = document.to_dict()
            doc_data['ingestion_timestamp'] = datetime.now().isoformat()
            
            # Insert document
            result = self.client.table('documents').insert(doc_data).execute()
            
            if result.data:
                document_id = result.data[0]['id']
                self.logger.info(f"Document stored successfully with ID: {document_id}")
                
                return StorageResult(
                    success=True,
                    record_id=document_id,
                    affected_rows=1
                )
            else:
                error_msg = "No data returned from document insert"
                self.logger.error(error_msg)
                return StorageResult(success=False, error_message=error_msg)
                
        except Exception as e:
            error_msg = f"Failed to store document: {e}"
            self.logger.error(error_msg)
            return StorageResult(success=False, error_message=error_msg)
    
    async def store_chunks(self, chunks: List[EmbeddedChunk]) -> List[StorageResult]:
        """
        Store multiple chunks with embedding vectors.
        
        Args:
            chunks: List of embedded chunks to store
            
        Returns:
            List[StorageResult]: Storage results for each chunk
        """
        results = []
        
        try:
            self.logger.info(f"Storing {len(chunks)} chunks")
            
            # Prepare chunk data for batch insert
            chunks_data = []
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                chunk_dict['created_at'] = datetime.now().isoformat()
                chunks_data.append(chunk_dict)
            
            # Batch insert chunks
            result = self.client.table('chunks').insert(chunks_data).execute()
            
            if result.data:
                for i, chunk_record in enumerate(result.data):
                    chunk_id = chunk_record['id']
                    results.append(StorageResult(
                        success=True,
                        record_id=chunk_id,
                        affected_rows=1
                    ))
                
                self.logger.info(f"Successfully stored {len(result.data)} chunks")
            else:
                error_msg = "No data returned from chunks insert"
                self.logger.error(error_msg)
                for _ in chunks:
                    results.append(StorageResult(success=False, error_message=error_msg))
                
        except Exception as e:
            error_msg = f"Failed to store chunks: {e}"
            self.logger.error(error_msg)
            for _ in chunks:
                results.append(StorageResult(success=False, error_message=error_msg))
        
        return results
    
    async def store_document_with_chunks(
        self, 
        document: DocumentMetadata, 
        chunks: List[EmbeddedChunk]
    ) -> TransactionResult:
        """
        Store document and chunks in a single atomic transaction.
        
        Args:
            document: Document metadata to store
            chunks: List of embedded chunks to store
            
        Returns:
            TransactionResult: Transaction result with document and chunk IDs
        """
        try:
            self.logger.info(f"Starting atomic transaction for document {document.file_id} with {len(chunks)} chunks")
            
            # Store document first
            doc_result = await self.store_document(document)
            
            if not doc_result.success:
                return TransactionResult(
                    success=False,
                    error_message=f"Failed to store document: {doc_result.error_message}"
                )
            
            document_id = doc_result.record_id
            
            # Update chunks with document_id
            for chunk in chunks:
                chunk.chunk_data.document_id = document_id
            
            # Store chunks
            chunk_results = await self.store_chunks(chunks)
            
            # Check if all chunks were stored successfully
            failed_chunks = [r for r in chunk_results if not r.success]
            
            if failed_chunks:
                # Rollback: delete the document
                await self._delete_document(document_id)
                
                return TransactionResult(
                    success=False,
                    error_message=f"Failed to store {len(failed_chunks)} chunks, transaction rolled back",
                    rollback_performed=True
                )
            
            # Collect successful chunk IDs
            chunk_ids = [r.record_id for r in chunk_results if r.record_id]
            
            self.logger.info(f"Transaction completed successfully: document {document_id}, {len(chunk_ids)} chunks")
            
            return TransactionResult(
                success=True,
                document_id=document_id,
                chunk_ids=chunk_ids
            )
            
        except Exception as e:
            error_msg = f"Transaction failed: {e}"
            self.logger.error(error_msg)
            
            # Attempt rollback if we have a document_id
            rollback_performed = False
            if 'document_id' in locals() and document_id:
                try:
                    await self._delete_document(document_id)
                    rollback_performed = True
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            return TransactionResult(
                success=False,
                error_message=error_msg,
                rollback_performed=rollback_performed
            )
    
    async def log_ingestion_status(self, log_entry: IngestionLogEntry) -> StorageResult:
        """
        Log ingestion status for tracking and debugging.
        
        Args:
            log_entry: Ingestion log entry to store
            
        Returns:
            StorageResult: Storage operation result
        """
        try:
            # Prepare log data
            log_data = log_entry.to_dict()
            log_data['created_at'] = datetime.now().isoformat()
            
            # Generate correlation ID if not provided
            if not log_data.get('correlation_id'):
                log_data['correlation_id'] = str(uuid.uuid4())
            
            # Insert log entry
            result = self.client.table('ingestion_logs').insert(log_data).execute()
            
            if result.data:
                log_id = result.data[0]['id']
                return StorageResult(
                    success=True,
                    record_id=log_id,
                    affected_rows=1
                )
            else:
                error_msg = "No data returned from ingestion log insert"
                return StorageResult(success=False, error_message=error_msg)
                
        except Exception as e:
            error_msg = f"Failed to log ingestion status: {e}"
            self.logger.error(error_msg)
            return StorageResult(success=False, error_message=error_msg)
    
    async def update_document_status(
        self, 
        document_id: str, 
        status: ProcessingStatus,
        quality_score: Optional[float] = None
    ) -> StorageResult:
        """
        Update document processing status and quality score.
        
        Args:
            document_id: Document ID to update
            status: New processing status
            quality_score: Optional quality score to update
            
        Returns:
            StorageResult: Update operation result
        """
        try:
            update_data = {
                'processing_status': status.value,
                'updated_at': datetime.now().isoformat()
            }
            
            if quality_score is not None:
                update_data['quality_score'] = quality_score
            
            result = self.client.table('documents').update(update_data).eq('id', document_id).execute()
            
            if result.data:
                return StorageResult(
                    success=True,
                    record_id=document_id,
                    affected_rows=len(result.data)
                )
            else:
                error_msg = f"No document found with ID: {document_id}"
                return StorageResult(success=False, error_message=error_msg)
                
        except Exception as e:
            error_msg = f"Failed to update document status: {e}"
            self.logger.error(error_msg)
            return StorageResult(success=False, error_message=error_msg)
    
    async def get_document_by_file_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by Google Drive file ID.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            Document record or None if not found
        """
        try:
            result = self.client.table('documents').select('*').eq('file_id', file_id).execute()
            
            if result.data:
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve document by file_id {file_id}: {e}")
            return None
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk records
        """
        try:
            result = self.client.table('chunks').select('*').eq('document_id', document_id).order('chunk_order').execute()
            
            return result.data or []
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks for document {document_id}: {e}")
            return []
    
    async def search_similar_chunks(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            embedding_model: Filter by embedding model
            
        Returns:
            List of similar chunks with similarity scores
        """
        try:
            # Convert query vector to string format for pgvector
            vector_str = f"[{','.join(map(str, query_vector))}]"
            
            # Build query
            query = self.client.table('chunks').select(
                '*, documents!inner(title, domain_classification)'
            ).rpc(
                'match_chunks',
                {
                    'query_embedding': vector_str,
                    'match_threshold': similarity_threshold,
                    'match_count': limit
                }
            )
            
            # Add embedding model filter if specified
            if embedding_model:
                query = query.eq('embedding_model', embedding_model)
            
            result = query.execute()
            
            return result.data or []
                
        except Exception as e:
            self.logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    async def get_ingestion_logs(
        self, 
        file_id: Optional[str] = None,
        phase: Optional[IngestionPhase] = None,
        status: Optional[IngestionStatus] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ingestion logs with optional filters.
        
        Args:
            file_id: Filter by file ID
            phase: Filter by ingestion phase
            status: Filter by status
            correlation_id: Filter by correlation ID
            limit: Maximum number of results
            
        Returns:
            List of ingestion log records
        """
        try:
            query = self.client.table('ingestion_logs').select('*')
            
            if file_id:
                query = query.eq('file_id', file_id)
            if phase:
                query = query.eq('phase', phase.value)
            if status:
                query = query.eq('status', status.value)
            if correlation_id:
                query = query.eq('correlation_id', correlation_id)
            
            result = query.order('created_at', desc=True).limit(limit).execute()
            
            return result.data or []
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve ingestion logs: {e}")
            return []
    
    async def _delete_document(self, document_id: str) -> bool:
        """
        Delete document and associated chunks (for rollback).
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Delete document (chunks will be deleted by CASCADE)
            result = self.client.table('documents').delete().eq('id', document_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics for monitoring and reporting.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {}
            
            # Document statistics
            doc_result = self.client.table('documents').select('processing_status', count='exact').execute()
            stats['total_documents'] = doc_result.count or 0
            
            # Status breakdown
            status_counts = {}
            for status in ProcessingStatus:
                status_result = self.client.table('documents').select('id', count='exact').eq('processing_status', status.value).execute()
                status_counts[status.value] = status_result.count or 0
            stats['document_status_breakdown'] = status_counts
            
            # Chunk statistics
            chunk_result = self.client.table('chunks').select('id', count='exact').execute()
            stats['total_chunks'] = chunk_result.count or 0
            
            # Embedding model breakdown
            model_result = self.client.table('chunks').select('embedding_model', count='exact').execute()
            stats['embedding_models'] = {}
            if model_result.data:
                for row in model_result.data:
                    model = row.get('embedding_model', 'unknown')
                    stats['embedding_models'][model] = stats['embedding_models'].get(model, 0) + 1
            
            # Recent ingestion activity
            recent_logs = await self.get_ingestion_logs(limit=10)
            stats['recent_activity'] = len(recent_logs)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {}


class TransactionManager:
    """
    Transaction manager for atomic operations across multiple tables.
    
    Provides higher-level transaction management for complex operations
    that span multiple storage operations.
    """
    
    def __init__(self, storage_service: SupabaseStorageService):
        self.storage = storage_service
        self.logger = get_logger(__name__)
    
    async def process_document_batch(
        self, 
        documents_with_chunks: List[Tuple[DocumentMetadata, List[EmbeddedChunk]]]
    ) -> List[TransactionResult]:
        """
        Process multiple documents with chunks in separate transactions.
        
        Args:
            documents_with_chunks: List of (document, chunks) tuples
            
        Returns:
            List of transaction results
        """
        results = []
        
        for i, (document, chunks) in enumerate(documents_with_chunks):
            self.logger.info(f"Processing document batch {i+1}/{len(documents_with_chunks)}: {document.file_id}")
            
            result = await self.storage.store_document_with_chunks(document, chunks)
            results.append(result)
            
            # Log the transaction result
            log_entry = IngestionLogEntry(
                file_id=document.file_id,
                phase=IngestionPhase.STORAGE,
                status=IngestionStatus.COMPLETED if result.success else IngestionStatus.FAILED,
                error_message=result.error_message,
                metadata={
                    'document_id': result.document_id,
                    'chunk_count': len(chunks),
                    'rollback_performed': result.rollback_performed
                }
            )
            
            await self.storage.log_ingestion_status(log_entry)
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Batch processing completed: {successful}/{len(results)} successful")
        
        return results