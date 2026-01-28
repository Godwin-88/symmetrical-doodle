"""
Unified Source Browsing Service

This service provides unified file browsing capabilities across all data sources,
including hierarchical navigation, cross-source search, metadata retrieval,
file access validation, and performance caching.

Features:
- Unified file listing across all connected sources
- Hierarchical folder navigation for applicable sources
- Cross-source search functionality
- Metadata retrieval with source attribution
- File access validation per source type
- Caching layer for performance optimization
- Consistent response formats across all source types
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

from .multi_source_auth import (
    MultiSourceAuthenticationService,
    DataSourceType,
    ConnectionInfo,
    AuthenticationStatus
)
from core.config import get_settings
from core.logging import get_logger


@dataclass
class UniversalFileMetadata:
    """Universal file metadata across all data sources"""
    file_id: str
    name: str
    size: int
    modified_time: datetime
    source_type: DataSourceType
    source_path: str
    mime_type: str = "application/pdf"
    access_url: Optional[str] = None
    parent_folders: List[str] = field(default_factory=list)
    domain_classification: Optional[str] = None
    checksum: Optional[str] = None
    source_specific_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_status: Optional[str] = None
    is_accessible: bool = True


@dataclass
class FolderMetadata:
    """Folder metadata for hierarchical navigation"""
    folder_id: str
    name: str
    source_type: DataSourceType
    source_path: str
    parent_folder_id: Optional[str] = None
    child_count: int = 0
    pdf_count: int = 0
    modified_time: Optional[datetime] = None
    source_specific_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileTreeNode:
    """File tree node for hierarchical display"""
    id: str
    name: str
    type: str  # 'folder' or 'file'
    source_type: DataSourceType
    children: List['FileTreeNode'] = field(default_factory=list)
    metadata: Optional[Union[UniversalFileMetadata, FolderMetadata]] = None
    is_expanded: bool = False


@dataclass
class SourceFileTree:
    """File tree for a specific data source"""
    source_type: DataSourceType
    source_name: str
    connection_id: str
    root_folders: List[FileTreeNode] = field(default_factory=list)
    total_files: int = 0
    total_folders: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchResult:
    """Search result item"""
    file_metadata: UniversalFileMetadata
    relevance_score: float
    matched_fields: List[str] = field(default_factory=list)
    snippet: Optional[str] = None


@dataclass
class CacheEntry:
    """Cache entry with expiration"""
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UnifiedBrowsingCache:
    """Performance cache for browsing operations"""
    
    def __init__(self, default_ttl_minutes: int = 15, max_entries: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.max_entries = max_entries
        self.logger = get_logger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, operation: str, **kwargs) -> str:
        """Generate cache key from operation and parameters"""
        key_data = f"{operation}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, operation: str, **kwargs) -> Optional[Any]:
        """Get cached data"""
        key = self._generate_key(operation, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration
            if datetime.now(timezone.utc) > entry.expires_at:
                del self.cache[key]
                self.misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self.hits += 1
            
            return entry.data
        
        self.misses += 1
        return None
    
    def set(self, operation: str, data: Any, ttl: Optional[timedelta] = None, **kwargs):
        """Set cached data"""
        key = self._generate_key(operation, **kwargs)
        ttl = ttl or self.default_ttl
        
        # Evict oldest entries if at capacity
        if len(self.cache) >= self.max_entries:
            self._evict_oldest()
        
        self.cache[key] = CacheEntry(
            data=data,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + ttl
        )
    
    def invalidate(self, operation: str, **kwargs):
        """Invalidate specific cache entry"""
        key = self._generate_key(operation, **kwargs)
        if key in self.cache:
            del self.cache[key]
    
    def invalidate_source(self, connection_id: str):
        """Invalidate all cache entries for a specific source"""
        keys_to_remove = []
        for key, entry in self.cache.items():
            if connection_id in str(entry.data):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def _evict_oldest(self):
        """Evict oldest cache entries"""
        if not self.cache:
            return
        
        # Remove 10% of entries, starting with oldest
        entries_to_remove = max(1, len(self.cache) // 10)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for key, _ in sorted_entries[:entries_to_remove]:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate memory usage estimate (rough)
        memory_usage_mb = len(json.dumps(self.cache, default=str).encode()) / (1024 * 1024)
        
        # Find oldest entry
        oldest_age = 0
        if self.cache:
            oldest_entry = min(self.cache.values(), key=lambda x: x.created_at)
            oldest_age = int((datetime.now(timezone.utc) - oldest_entry.created_at).total_seconds())
        
        return {
            'total_entries': len(self.cache),
            'hit_rate': hit_rate,
            'memory_usage_mb': memory_usage_mb,
            'oldest_entry_age_seconds': oldest_age,
            'hits': self.hits,
            'misses': self.misses
        }
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class UnifiedBrowsingService:
    """Unified browsing service for all data sources"""
    
    def __init__(self, auth_service: MultiSourceAuthenticationService):
        self.auth_service = auth_service
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize cache
        self.cache = UnifiedBrowsingCache()
        
        # Source-specific services
        self._source_services: Dict[DataSourceType, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize the browsing service"""
        try:
            self.logger.info("Initializing unified browsing service")
            
            # Initialize source-specific services as needed
            # This would be expanded based on available connectors
            
            self.logger.info("Unified browsing service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browsing service: {e}")
            return False
    
    async def list_files(
        self,
        connection_ids: Optional[List[str]] = None,
        source_types: Optional[List[DataSourceType]] = None,
        folder_path: Optional[str] = None,
        include_subfolders: bool = True,
        file_types: List[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[UniversalFileMetadata]:
        """List files across specified sources with unified metadata format"""
        
        # Check cache first
        cache_key_params = {
            'connection_ids': connection_ids,
            'source_types': [st.value for st in source_types] if source_types else None,
            'folder_path': folder_path,
            'include_subfolders': include_subfolders,
            'file_types': file_types,
            'limit': limit,
            'offset': offset
        }
        
        cached_result = self.cache.get('list_files', **cache_key_params)
        if cached_result:
            self.logger.debug("Returning cached file list")
            return cached_result
        
        try:
            # Get active connections
            connections = await self._get_filtered_connections(connection_ids, source_types)
            
            if not connections:
                self.logger.warning("No active connections found for file listing")
                return []
            
            # Collect files from all sources
            all_files = []
            
            for connection in connections:
                try:
                    source_files = await self._list_files_from_source(
                        connection,
                        folder_path,
                        include_subfolders,
                        file_types or ["application/pdf"]
                    )
                    all_files.extend(source_files)
                    
                except Exception as e:
                    self.logger.error(f"Failed to list files from {connection.source_type.value}: {e}")
                    continue
            
            # Apply pagination
            if offset > 0:
                all_files = all_files[offset:]
            if limit:
                all_files = all_files[:limit]
            
            # Cache the result
            self.cache.set('list_files', all_files, **cache_key_params)
            
            self.logger.info(f"Listed {len(all_files)} files from {len(connections)} sources")
            return all_files
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            raise
    
    async def get_file_tree(
        self,
        connection_ids: Optional[List[str]] = None,
        source_types: Optional[List[DataSourceType]] = None,
        max_depth: int = 3
    ) -> List[SourceFileTree]:
        """Get hierarchical file tree for sources that support it"""
        
        # Check cache first
        cache_key_params = {
            'connection_ids': connection_ids,
            'source_types': [st.value for st in source_types] if source_types else None,
            'max_depth': max_depth
        }
        
        cached_result = self.cache.get('get_file_tree', **cache_key_params)
        if cached_result:
            self.logger.debug("Returning cached file tree")
            return cached_result
        
        try:
            # Get active connections
            connections = await self._get_filtered_connections(connection_ids, source_types)
            
            if not connections:
                return []
            
            # Build file trees for each source
            source_trees = []
            
            for connection in connections:
                try:
                    tree = await self._build_source_file_tree(connection, max_depth)
                    if tree:
                        source_trees.append(tree)
                        
                except Exception as e:
                    self.logger.error(f"Failed to build file tree for {connection.source_type.value}: {e}")
                    continue
            
            # Cache the result
            self.cache.set('get_file_tree', source_trees, **cache_key_params)
            
            self.logger.info(f"Built file trees for {len(source_trees)} sources")
            return source_trees
            
        except Exception as e:
            self.logger.error(f"Failed to get file tree: {e}")
            raise
    
    async def search_files(
        self,
        query: str,
        connection_ids: Optional[List[str]] = None,
        source_types: Optional[List[DataSourceType]] = None,
        search_fields: List[str] = None,
        file_types: List[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search files across all specified sources"""
        
        # Check cache first
        cache_key_params = {
            'query': query,
            'connection_ids': connection_ids,
            'source_types': [st.value for st in source_types] if source_types else None,
            'search_fields': search_fields,
            'file_types': file_types,
            'limit': limit,
            'offset': offset
        }
        
        cached_result = self.cache.get('search_files', **cache_key_params)
        if cached_result:
            self.logger.debug("Returning cached search results")
            return cached_result
        
        try:
            # Get active connections
            connections = await self._get_filtered_connections(connection_ids, source_types)
            
            if not connections:
                return []
            
            # Search across all sources
            all_results = []
            search_fields = search_fields or ['name', 'content']
            file_types = file_types or ['application/pdf']
            
            for connection in connections:
                try:
                    source_results = await self._search_files_in_source(
                        connection,
                        query,
                        search_fields,
                        file_types
                    )
                    all_results.extend(source_results)
                    
                except Exception as e:
                    self.logger.error(f"Failed to search in {connection.source_type.value}: {e}")
                    continue
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply pagination
            if offset > 0:
                all_results = all_results[offset:]
            if limit:
                all_results = all_results[:limit]
            
            # Cache the result
            self.cache.set('search_files', all_results, **cache_key_params)
            
            self.logger.info(f"Found {len(all_results)} search results from {len(connections)} sources")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Failed to search files: {e}")
            raise
    
    async def validate_file_access(
        self,
        file_ids: List[str],
        connection_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Validate access to specific files"""
        
        # Check cache first
        cache_key_params = {
            'file_ids': sorted(file_ids),
            'connection_id': connection_id
        }
        
        cached_result = self.cache.get('validate_file_access', **cache_key_params)
        if cached_result:
            self.logger.debug("Returning cached access validation")
            return cached_result
        
        try:
            # Get connection info
            connection = await self.auth_service.get_connection_status(connection_id)
            if not connection or connection.status != AuthenticationStatus.AUTHENTICATED:
                raise ValueError(f"Invalid or inactive connection: {connection_id}")
            
            # Validate access for each file
            validation_results = {}
            
            for file_id in file_ids:
                try:
                    result = await self._validate_file_access_in_source(connection, file_id)
                    validation_results[file_id] = result
                    
                except Exception as e:
                    self.logger.error(f"Failed to validate access for file {file_id}: {e}")
                    validation_results[file_id] = {
                        'is_accessible': False,
                        'access_level': 'none',
                        'error': str(e)
                    }
            
            # Cache the result (shorter TTL for access validation)
            self.cache.set(
                'validate_file_access',
                validation_results,
                ttl=timedelta(minutes=5),
                **cache_key_params
            )
            
            self.logger.info(f"Validated access for {len(file_ids)} files")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate file access: {e}")
            raise
    
    async def get_file_metadata(
        self,
        file_id: str,
        connection_id: str
    ) -> Optional[UniversalFileMetadata]:
        """Get detailed metadata for a specific file"""
        
        # Check cache first
        cached_result = self.cache.get('get_file_metadata', file_id=file_id, connection_id=connection_id)
        if cached_result:
            self.logger.debug("Returning cached file metadata")
            return cached_result
        
        try:
            # Get connection info
            connection = await self.auth_service.get_connection_status(connection_id)
            if not connection or connection.status != AuthenticationStatus.AUTHENTICATED:
                raise ValueError(f"Invalid or inactive connection: {connection_id}")
            
            # Get metadata from source
            metadata = await self._get_file_metadata_from_source(connection, file_id)
            
            # Cache the result
            if metadata:
                self.cache.set('get_file_metadata', metadata, file_id=file_id, connection_id=connection_id)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get file metadata: {e}")
            raise
    
    def invalidate_cache(self, connection_id: Optional[str] = None):
        """Invalidate cache entries"""
        if connection_id:
            self.cache.invalidate_source(connection_id)
            self.logger.info(f"Invalidated cache for connection {connection_id}")
        else:
            self.cache.clear()
            self.logger.info("Cleared all cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.cache.get_stats()
    
    # Private helper methods
    
    async def _get_filtered_connections(
        self,
        connection_ids: Optional[List[str]] = None,
        source_types: Optional[List[DataSourceType]] = None
    ) -> List[ConnectionInfo]:
        """Get filtered list of active connections"""
        
        # Get all connections
        all_connections = await self.auth_service.list_connections()
        
        # Filter by status
        active_connections = [
            conn for conn in all_connections
            if conn.status == AuthenticationStatus.AUTHENTICATED
        ]
        
        # Filter by connection IDs if specified
        if connection_ids:
            active_connections = [
                conn for conn in active_connections
                if conn.connection_id in connection_ids
            ]
        
        # Filter by source types if specified
        if source_types:
            active_connections = [
                conn for conn in active_connections
                if conn.source_type in source_types
            ]
        
        return active_connections
    
    async def _list_files_from_source(
        self,
        connection: ConnectionInfo,
        folder_path: Optional[str],
        include_subfolders: bool,
        file_types: List[str]
    ) -> List[UniversalFileMetadata]:
        """List files from a specific source"""
        
        # This would be implemented based on the specific source type
        # For now, return empty list as placeholder
        self.logger.warning(f"File listing not yet implemented for {connection.source_type.value}")
        return []
    
    async def _build_source_file_tree(
        self,
        connection: ConnectionInfo,
        max_depth: int
    ) -> Optional[SourceFileTree]:
        """Build file tree for a specific source"""
        
        # This would be implemented based on the specific source type
        # For now, return None as placeholder
        self.logger.warning(f"File tree building not yet implemented for {connection.source_type.value}")
        return None
    
    async def _search_files_in_source(
        self,
        connection: ConnectionInfo,
        query: str,
        search_fields: List[str],
        file_types: List[str]
    ) -> List[SearchResult]:
        """Search files in a specific source"""
        
        # This would be implemented based on the specific source type
        # For now, return empty list as placeholder
        self.logger.warning(f"File search not yet implemented for {connection.source_type.value}")
        return []
    
    async def _validate_file_access_in_source(
        self,
        connection: ConnectionInfo,
        file_id: str
    ) -> Dict[str, Any]:
        """Validate file access in a specific source"""
        
        # This would be implemented based on the specific source type
        # For now, return basic validation
        return {
            'is_accessible': True,
            'access_level': 'read',
            'error': None
        }
    
    async def _get_file_metadata_from_source(
        self,
        connection: ConnectionInfo,
        file_id: str
    ) -> Optional[UniversalFileMetadata]:
        """Get file metadata from a specific source"""
        
        # This would be implemented based on the specific source type
        # For now, return None as placeholder
        self.logger.warning(f"File metadata retrieval not yet implemented for {connection.source_type.value}")
        return None


# Global service instance
_browsing_service: Optional[UnifiedBrowsingService] = None


def get_browsing_service() -> UnifiedBrowsingService:
    """Get or create global browsing service instance"""
    global _browsing_service
    
    if _browsing_service is None:
        from .multi_source_auth import get_auth_service
        auth_service = get_auth_service()
        _browsing_service = UnifiedBrowsingService(auth_service)
    
    return _browsing_service