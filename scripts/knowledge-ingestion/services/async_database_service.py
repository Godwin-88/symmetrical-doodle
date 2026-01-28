"""
Async Database Service with Connection Pooling and Optimization

This module provides optimized database operations for the knowledge ingestion
system using asyncio, connection pooling, batch operations, and performance
monitoring for Supabase integration.

Features:
- Async connection pooling for Supabase
- Batch database operations with transaction management
- Concurrent query execution with backpressure handling
- Performance monitoring and query optimization
- Automatic retry logic and error handling
- Connection health monitoring and recovery

Requirements: 10.1, 10.2, 10.4
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from contextlib import asynccontextmanager
import json
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor

import asyncpg
import aiohttp
from supabase import create_client, Client
import numpy as np

from core.config import get_settings, SupabaseConfig
from core.logging import get_logger
from .supabase_storage import (
    DocumentMetadata, ChunkData, EmbeddedChunk, IngestionLogEntry,
    ProcessingStatus, IngestionPhase, IngestionStatus, StorageResult, TransactionResult
)


class ConnectionState(Enum):
    """Database connection state"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"


class QueryType(Enum):
    """Database query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BATCH_INSERT = "batch_insert"
    TRANSACTION = "transaction"
    VECTOR_SEARCH = "vector_search"


@dataclass
class DatabaseQuery:
    """Database query definition"""
    query_id: str
    query_type: QueryType
    table: str
    data: Optional[Any] = None
    filters: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
    priority: int = 0
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    callback: Optional[Callable] = None


@dataclass
class DatabaseConnection:
    """Database connection wrapper"""
    connection_id: str
    client: Client
    state: ConnectionState = ConnectionState.IDLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_count: int = 0
    error_count: int = 0
    current_query: Optional[str] = None


@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    busy_connections: int = 0
    error_connections: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_query_time: float = 0.0
    connection_wait_time: float = 0.0
    pool_utilization: float = 0.0
    queries_per_second: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BatchOperation:
    """Batch database operation"""
    batch_id: str
    operation_type: QueryType
    table: str
    records: List[Dict[str, Any]]
    batch_size: int = 100
    parallel_batches: int = 4
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    callback: Optional[Callable] = None


class AsyncConnectionPool:
    """Async connection pool for Supabase"""
    
    def __init__(self, config: SupabaseConfig, pool_size: int = 10):
        self.config = config
        self.pool_size = pool_size
        self.logger = get_logger(f"{__name__}.ConnectionPool")
        
        # Connection management
        self._connections: Dict[str, DatabaseConnection] = {}
        self._available_connections: asyncio.Queue = asyncio.Queue()
        self._connection_lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(pool_size)
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30.0  # seconds
        
        # Metrics
        self._metrics = ConnectionPoolMetrics()
        self._metrics_lock = asyncio.Lock()
        
        # Shutdown
        self._shutdown = False
    
    async def initialize(self) -> bool:
        """Initialize the connection pool"""
        try:
            self.logger.info(f"Initializing connection pool with {self.pool_size} connections")
            
            # Create initial connections
            for i in range(self.pool_size):
                connection = await self._create_connection()
                if connection:
                    self._connections[connection.connection_id] = connection
                    await self._available_connections.put(connection.connection_id)
                else:
                    self.logger.error(f"Failed to create connection {i}")
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            async with self._metrics_lock:
                self._metrics.total_connections = len(self._connections)
                self._metrics.idle_connections = len(self._connections)
            
            self.logger.info(f"Connection pool initialized with {len(self._connections)} connections")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            return False
    
    async def _create_connection(self) -> Optional[DatabaseConnection]:
        """Create a new database connection"""
        try:
            client = create_client(
                self.config.url,
                self.config.service_role_key or self.config.key
            )
            
            # Test connection
            result = client.table('documents').select('id').limit(1).execute()
            
            connection = DatabaseConnection(
                connection_id=str(uuid.uuid4()),
                client=client,
                state=ConnectionState.IDLE
            )
            
            return connection
            
        except Exception as e:
            self.logger.error(f"Failed to create database connection: {e}")
            return None
    
    @asynccontextmanager
    async def acquire_connection(self, timeout: float = 30.0):
        """Acquire a connection from the pool"""
        connection_id = None
        start_time = time.time()
        
        try:
            # Wait for available connection
            connection_id = await asyncio.wait_for(
                self._available_connections.get(),
                timeout=timeout
            )
            
            async with self._connection_lock:
                connection = self._connections.get(connection_id)
                if not connection:
                    raise Exception(f"Connection {connection_id} not found in pool")
                
                connection.state = ConnectionState.ACTIVE
                connection.last_used = datetime.now(timezone.utc)
            
            # Update metrics
            wait_time = time.time() - start_time
            async with self._metrics_lock:
                self._metrics.connection_wait_time = (
                    0.9 * self._metrics.connection_wait_time + 0.1 * wait_time
                )
                self._metrics.active_connections += 1
                self._metrics.idle_connections -= 1
            
            yield connection
            
        except asyncio.TimeoutError:
            self.logger.error(f"Connection acquisition timeout after {timeout}s")
            raise
        except Exception as e:
            self.logger.error(f"Error acquiring connection: {e}")
            raise
        finally:
            # Return connection to pool
            if connection_id:
                async with self._connection_lock:
                    connection = self._connections.get(connection_id)
                    if connection:
                        connection.state = ConnectionState.IDLE
                        connection.current_query = None
                        await self._available_connections.put(connection_id)
                
                async with self._metrics_lock:
                    self._metrics.active_connections = max(0, self._metrics.active_connections - 1)
                    self._metrics.idle_connections += 1
    
    async def _health_monitor(self):
        """Monitor connection health"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Check each connection
                unhealthy_connections = []
                
                async with self._connection_lock:
                    for conn_id, connection in self._connections.items():
                        if connection.state == ConnectionState.ERROR:
                            unhealthy_connections.append(conn_id)
                        elif connection.state == ConnectionState.IDLE:
                            # Test idle connection
                            try:
                                result = connection.client.table('documents').select('id').limit(1).execute()
                                connection.error_count = 0
                            except Exception as e:
                                self.logger.warning(f"Connection {conn_id} health check failed: {e}")
                                connection.error_count += 1
                                if connection.error_count >= 3:
                                    connection.state = ConnectionState.ERROR
                                    unhealthy_connections.append(conn_id)
                
                # Replace unhealthy connections
                for conn_id in unhealthy_connections:
                    await self._replace_connection(conn_id)
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    async def _replace_connection(self, connection_id: str):
        """Replace an unhealthy connection"""
        try:
            self.logger.info(f"Replacing unhealthy connection {connection_id}")
            
            # Create new connection
            new_connection = await self._create_connection()
            if not new_connection:
                self.logger.error(f"Failed to create replacement connection")
                return
            
            async with self._connection_lock:
                # Remove old connection
                old_connection = self._connections.pop(connection_id, None)
                if old_connection:
                    try:
                        # Close old connection if possible
                        pass  # Supabase client doesn't have explicit close
                    except Exception:
                        pass
                
                # Add new connection
                self._connections[new_connection.connection_id] = new_connection
                await self._available_connections.put(new_connection.connection_id)
            
            self.logger.info(f"Replaced connection {connection_id} with {new_connection.connection_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to replace connection {connection_id}: {e}")
    
    async def _metrics_collector(self):
        """Collect connection pool metrics"""
        while not self._shutdown:
            try:
                async with self._metrics_lock:
                    # Update connection state counts
                    self._metrics.idle_connections = 0
                    self._metrics.active_connections = 0
                    self._metrics.busy_connections = 0
                    self._metrics.error_connections = 0
                    
                    for connection in self._connections.values():
                        if connection.state == ConnectionState.IDLE:
                            self._metrics.idle_connections += 1
                        elif connection.state == ConnectionState.ACTIVE:
                            self._metrics.active_connections += 1
                        elif connection.state == ConnectionState.BUSY:
                            self._metrics.busy_connections += 1
                        elif connection.state == ConnectionState.ERROR:
                            self._metrics.error_connections += 1
                    
                    # Calculate pool utilization
                    if self._metrics.total_connections > 0:
                        active_ratio = (self._metrics.active_connections + self._metrics.busy_connections) / self._metrics.total_connections
                        self._metrics.pool_utilization = active_ratio * 100
                    
                    self._metrics.last_updated = datetime.now(timezone.utc)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting pool metrics: {e}")
                await asyncio.sleep(5)
    
    async def get_metrics(self) -> ConnectionPoolMetrics:
        """Get current pool metrics"""
        async with self._metrics_lock:
            return self._metrics
    
    async def shutdown(self):
        """Shutdown the connection pool"""
        self.logger.info("Shutting down connection pool")
        
        self._shutdown = True
        
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._connection_lock:
            for connection in self._connections.values():
                try:
                    connection.state = ConnectionState.CLOSED
                    # Supabase client doesn't have explicit close method
                except Exception as e:
                    self.logger.warning(f"Error closing connection: {e}")
            
            self._connections.clear()
        
        self.logger.info("Connection pool shutdown complete")


class AsyncDatabaseService:
    """
    Main async database service providing optimized database operations
    with connection pooling, batch processing, and performance monitoring.
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Connection pool
        self.connection_pool: Optional[AsyncConnectionPool] = None
        
        # Query processing
        self._query_queue: asyncio.Queue = asyncio.Queue()
        self._query_processors: List[asyncio.Task] = []
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_processors: List[asyncio.Task] = []
        
        # Performance optimization
        self._query_cache: Dict[str, Any] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_ttl = 300  # 5 minutes
        
        # Metrics
        self._service_metrics = ConnectionPoolMetrics()
        self._metrics_lock = asyncio.Lock()
        
        # Configuration
        self.max_query_processors = 8
        self.max_batch_processors = 4
        self.default_batch_size = 100
        self.max_concurrent_batches = 4
        
        # Shutdown
        self._shutdown = False
    
    async def initialize(self) -> bool:
        """Initialize the async database service"""
        try:
            self.logger.info("Initializing async database service")
            
            # Initialize connection pool
            pool_size = min(20, max(5, self.config.max_connections or 10))
            self.connection_pool = AsyncConnectionPool(self.config, pool_size)
            
            if not await self.connection_pool.initialize():
                raise Exception("Failed to initialize connection pool")
            
            # Start query processors
            for i in range(self.max_query_processors):
                processor = asyncio.create_task(self._query_processor(f"query_processor_{i}"))
                self._query_processors.append(processor)
            
            # Start batch processors
            for i in range(self.max_batch_processors):
                processor = asyncio.create_task(self._batch_processor(f"batch_processor_{i}"))
                self._batch_processors.append(processor)
            
            # Start metrics collection
            asyncio.create_task(self._service_metrics_collector())
            
            self.logger.info("Async database service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize async database service: {e}")
            return False
    
    async def execute_query_async(
        self,
        query_type: QueryType,
        table: str,
        data: Optional[Any] = None,
        filters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        timeout: float = 30.0
    ) -> Any:
        """Execute database query asynchronously"""
        query_id = f"query_{int(time.time() * 1000000)}"
        
        # Create future for result
        result_future = asyncio.Future()
        
        # Create callback to resolve future
        def result_callback(result: Any, error: Optional[Exception] = None):
            if not result_future.done():
                if error:
                    result_future.set_exception(error)
                else:
                    result_future.set_result(result)
        
        # Create query
        query = DatabaseQuery(
            query_id=query_id,
            query_type=query_type,
            table=table,
            data=data,
            filters=filters,
            options=options,
            priority=priority,
            timeout=timeout,
            callback=result_callback
        )
        
        # Submit query
        await self._query_queue.put(query)
        
        # Wait for result
        try:
            result = await asyncio.wait_for(result_future, timeout=timeout + 5)
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Query {query_id} timeout")
            raise
    
    async def store_document_async(self, document: DocumentMetadata) -> StorageResult:
        """Store document asynchronously"""
        try:
            result = await self.execute_query_async(
                query_type=QueryType.INSERT,
                table='documents',
                data=document.to_dict()
            )
            
            if result and result.data:
                document_id = result.data[0]['id']
                return StorageResult(
                    success=True,
                    record_id=document_id,
                    affected_rows=1
                )
            else:
                return StorageResult(
                    success=False,
                    error_message="No data returned from document insert"
                )
                
        except Exception as e:
            return StorageResult(
                success=False,
                error_message=f"Failed to store document: {e}"
            )
    
    async def store_chunks_batch_async(self, chunks: List[EmbeddedChunk]) -> List[StorageResult]:
        """Store chunks in optimized batches"""
        if not chunks:
            return []
        
        batch_id = f"batch_{int(time.time() * 1000000)}"
        
        # Prepare chunk data
        chunks_data = []
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            chunk_dict['created_at'] = datetime.now().isoformat()
            chunks_data.append(chunk_dict)
        
        # Create batch operation
        batch_op = BatchOperation(
            batch_id=batch_id,
            operation_type=QueryType.BATCH_INSERT,
            table='chunks',
            records=chunks_data,
            batch_size=self.default_batch_size,
            parallel_batches=min(self.max_concurrent_batches, len(chunks_data) // self.default_batch_size + 1)
        )
        
        # Create future for result
        result_future = asyncio.Future()
        
        def batch_callback(results: List[StorageResult]):
            if not result_future.done():
                result_future.set_result(results)
        
        batch_op.callback = batch_callback
        
        # Submit batch
        await self._batch_queue.put(batch_op)
        
        # Wait for results
        try:
            results = await asyncio.wait_for(result_future, timeout=300)  # 5 minute timeout
            return results
        except asyncio.TimeoutError:
            self.logger.error(f"Batch {batch_id} timeout")
            return [StorageResult(success=False, error_message="Batch timeout") for _ in chunks]
    
    async def store_document_with_chunks_async(
        self,
        document: DocumentMetadata,
        chunks: List[EmbeddedChunk]
    ) -> TransactionResult:
        """Store document and chunks in atomic transaction"""
        try:
            self.logger.info(f"Starting async transaction for document {document.file_id} with {len(chunks)} chunks")
            
            # Store document first
            doc_result = await self.store_document_async(document)
            
            if not doc_result.success:
                return TransactionResult(
                    success=False,
                    error_message=f"Failed to store document: {doc_result.error_message}"
                )
            
            document_id = doc_result.record_id
            
            # Update chunks with document_id
            for chunk in chunks:
                chunk.chunk_data.document_id = document_id
            
            # Store chunks in batches
            chunk_results = await self.store_chunks_batch_async(chunks)
            
            # Check if all chunks were stored successfully
            failed_chunks = [r for r in chunk_results if not r.success]
            
            if failed_chunks:
                # Rollback: delete the document
                await self._delete_document_async(document_id)
                
                return TransactionResult(
                    success=False,
                    error_message=f"Failed to store {len(failed_chunks)} chunks, transaction rolled back",
                    rollback_performed=True
                )
            
            # Collect successful chunk IDs
            chunk_ids = [r.record_id for r in chunk_results if r.record_id]
            
            self.logger.info(f"Async transaction completed: document {document_id}, {len(chunk_ids)} chunks")
            
            return TransactionResult(
                success=True,
                document_id=document_id,
                chunk_ids=chunk_ids
            )
            
        except Exception as e:
            error_msg = f"Async transaction failed: {e}"
            self.logger.error(error_msg)
            
            # Attempt rollback if we have a document_id
            rollback_performed = False
            if 'document_id' in locals() and document_id:
                try:
                    await self._delete_document_async(document_id)
                    rollback_performed = True
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            return TransactionResult(
                success=False,
                error_message=error_msg,
                rollback_performed=rollback_performed
            )
    
    async def search_similar_chunks_async(
        self,
        query_vector: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        embedding_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Create cache key
            cache_key = f"vector_search_{hash(str(query_vector[:10]))}{limit}{similarity_threshold}{embedding_model}"
            
            # Check cache
            async with self._cache_lock:
                cached_result = self._query_cache.get(cache_key)
                if cached_result and (time.time() - cached_result['timestamp']) < self._cache_ttl:
                    return cached_result['data']
            
            # Prepare search parameters
            search_params = {
                'query_embedding': f"[{','.join(map(str, query_vector))}]",
                'match_threshold': similarity_threshold,
                'match_count': limit
            }
            
            if embedding_model:
                search_params['embedding_model'] = embedding_model
            
            # Execute vector search
            result = await self.execute_query_async(
                query_type=QueryType.VECTOR_SEARCH,
                table='chunks',
                data=search_params,
                options={'rpc_function': 'match_chunks'}
            )
            
            search_results = result.data if result and result.data else []
            
            # Cache result
            async with self._cache_lock:
                self._query_cache[cache_key] = {
                    'data': search_results,
                    'timestamp': time.time()
                }
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def _query_processor(self, processor_name: str):
        """Process database queries"""
        self.logger.debug(f"Starting query processor: {processor_name}")
        
        while not self._shutdown:
            try:
                # Get query from queue
                query = await self._query_queue.get()
                
                # Process query
                await self._process_query(query, processor_name)
                
                # Mark task as done
                self._query_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in query processor {processor_name}: {e}")
                await asyncio.sleep(1)
    
    async def _process_query(self, query: DatabaseQuery, processor_name: str):
        """Process a single database query"""
        start_time = time.time()
        query.started_at = datetime.now(timezone.utc)
        
        try:
            self.logger.debug(f"Processing query {query.query_id} on {processor_name}")
            
            # Acquire connection from pool
            async with self.connection_pool.acquire_connection(timeout=query.timeout) as connection:
                connection.current_query = query.query_id
                connection.state = ConnectionState.BUSY
                
                # Execute query based on type
                if query.query_type == QueryType.SELECT:
                    result = await self._execute_select(connection, query)
                elif query.query_type == QueryType.INSERT:
                    result = await self._execute_insert(connection, query)
                elif query.query_type == QueryType.UPDATE:
                    result = await self._execute_update(connection, query)
                elif query.query_type == QueryType.DELETE:
                    result = await self._execute_delete(connection, query)
                elif query.query_type == QueryType.VECTOR_SEARCH:
                    result = await self._execute_vector_search(connection, query)
                else:
                    raise ValueError(f"Unsupported query type: {query.query_type}")
                
                # Update connection stats
                connection.query_count += 1
                connection.state = ConnectionState.ACTIVE
                
                # Handle result
                query.completed_at = datetime.now(timezone.utc)
                processing_time = time.time() - start_time
                
                if query.callback:
                    query.callback(result)
                
                # Update metrics
                async with self._metrics_lock:
                    self._service_metrics.successful_queries += 1
                    self._service_metrics.total_queries += 1
                    
                    # Update average query time
                    if self._service_metrics.total_queries == 1:
                        self._service_metrics.average_query_time = processing_time
                    else:
                        alpha = 0.1
                        self._service_metrics.average_query_time = (
                            alpha * processing_time + 
                            (1 - alpha) * self._service_metrics.average_query_time
                        )
                
                self.logger.debug(f"Query {query.query_id} completed in {processing_time:.2f}s")
                
        except Exception as e:
            # Handle query failure
            query.completed_at = datetime.now(timezone.utc)
            
            # Retry logic
            if query.retry_count < query.max_retries:
                query.retry_count += 1
                query.started_at = None
                query.completed_at = None
                
                self.logger.warning(f"Retrying query {query.query_id} (attempt {query.retry_count})")
                await self._query_queue.put(query)
            else:
                self.logger.error(f"Query {query.query_id} failed after {query.retry_count} retries: {e}")
                
                if query.callback:
                    query.callback(None, e)
                
                async with self._metrics_lock:
                    self._service_metrics.failed_queries += 1
                    self._service_metrics.total_queries += 1
    
    async def _execute_select(self, connection: DatabaseConnection, query: DatabaseQuery) -> Any:
        """Execute SELECT query"""
        table_query = connection.client.table(query.table).select('*')
        
        # Apply filters
        if query.filters:
            for key, value in query.filters.items():
                table_query = table_query.eq(key, value)
        
        # Apply options
        if query.options:
            if 'limit' in query.options:
                table_query = table_query.limit(query.options['limit'])
            if 'order' in query.options:
                table_query = table_query.order(query.options['order'])
        
        return table_query.execute()
    
    async def _execute_insert(self, connection: DatabaseConnection, query: DatabaseQuery) -> Any:
        """Execute INSERT query"""
        return connection.client.table(query.table).insert(query.data).execute()
    
    async def _execute_update(self, connection: DatabaseConnection, query: DatabaseQuery) -> Any:
        """Execute UPDATE query"""
        table_query = connection.client.table(query.table).update(query.data)
        
        # Apply filters
        if query.filters:
            for key, value in query.filters.items():
                table_query = table_query.eq(key, value)
        
        return table_query.execute()
    
    async def _execute_delete(self, connection: DatabaseConnection, query: DatabaseQuery) -> Any:
        """Execute DELETE query"""
        table_query = connection.client.table(query.table).delete()
        
        # Apply filters
        if query.filters:
            for key, value in query.filters.items():
                table_query = table_query.eq(key, value)
        
        return table_query.execute()
    
    async def _execute_vector_search(self, connection: DatabaseConnection, query: DatabaseQuery) -> Any:
        """Execute vector similarity search"""
        rpc_function = query.options.get('rpc_function', 'match_chunks')
        return connection.client.rpc(rpc_function, query.data).execute()
    
    async def _batch_processor(self, processor_name: str):
        """Process batch operations"""
        self.logger.debug(f"Starting batch processor: {processor_name}")
        
        while not self._shutdown:
            try:
                # Get batch from queue
                batch = await self._batch_queue.get()
                
                # Process batch
                await self._process_batch(batch, processor_name)
                
                # Mark task as done
                self._batch_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in batch processor {processor_name}: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: BatchOperation, processor_name: str):
        """Process batch operation"""
        try:
            self.logger.debug(f"Processing batch {batch.batch_id} with {len(batch.records)} records")
            
            # Split records into sub-batches
            sub_batches = []
            for i in range(0, len(batch.records), batch.batch_size):
                sub_batch = batch.records[i:i + batch.batch_size]
                sub_batches.append(sub_batch)
            
            # Process sub-batches concurrently
            semaphore = asyncio.Semaphore(batch.parallel_batches)
            
            async def process_sub_batch(sub_batch_records):
                async with semaphore:
                    return await self._process_sub_batch(batch.table, sub_batch_records, batch.operation_type)
            
            # Execute all sub-batches
            tasks = [process_sub_batch(sub_batch) for sub_batch in sub_batches]
            sub_batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_results = []
            for sub_result in sub_batch_results:
                if isinstance(sub_result, Exception):
                    # Create error results for this sub-batch
                    sub_batch_size = len(sub_batches[len(all_results)])
                    error_results = [
                        StorageResult(success=False, error_message=str(sub_result))
                        for _ in range(sub_batch_size)
                    ]
                    all_results.extend(error_results)
                else:
                    all_results.extend(sub_result)
            
            # Call batch callback
            if batch.callback:
                batch.callback(all_results)
            
            self.logger.debug(f"Batch {batch.batch_id} completed with {len(all_results)} results")
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch.batch_id}: {e}")
            
            # Create error results for all records
            error_results = [
                StorageResult(success=False, error_message=str(e))
                for _ in batch.records
            ]
            
            if batch.callback:
                batch.callback(error_results)
    
    async def _process_sub_batch(
        self,
        table: str,
        records: List[Dict[str, Any]],
        operation_type: QueryType
    ) -> List[StorageResult]:
        """Process a sub-batch of records"""
        try:
            async with self.connection_pool.acquire_connection() as connection:
                if operation_type == QueryType.BATCH_INSERT:
                    result = connection.client.table(table).insert(records).execute()
                    
                    if result.data:
                        return [
                            StorageResult(success=True, record_id=record.get('id'), affected_rows=1)
                            for record in result.data
                        ]
                    else:
                        return [
                            StorageResult(success=False, error_message="No data returned")
                            for _ in records
                        ]
                else:
                    raise ValueError(f"Unsupported batch operation: {operation_type}")
                    
        except Exception as e:
            return [
                StorageResult(success=False, error_message=str(e))
                for _ in records
            ]
    
    async def _delete_document_async(self, document_id: str) -> bool:
        """Delete document (for rollback)"""
        try:
            result = await self.execute_query_async(
                query_type=QueryType.DELETE,
                table='documents',
                filters={'id': document_id}
            )
            return bool(result and result.data)
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def _service_metrics_collector(self):
        """Collect service-level metrics"""
        while not self._shutdown:
            try:
                # Get pool metrics
                if self.connection_pool:
                    pool_metrics = await self.connection_pool.get_metrics()
                    
                    async with self._metrics_lock:
                        # Merge pool metrics into service metrics
                        self._service_metrics.total_connections = pool_metrics.total_connections
                        self._service_metrics.active_connections = pool_metrics.active_connections
                        self._service_metrics.idle_connections = pool_metrics.idle_connections
                        self._service_metrics.busy_connections = pool_metrics.busy_connections
                        self._service_metrics.error_connections = pool_metrics.error_connections
                        self._service_metrics.pool_utilization = pool_metrics.pool_utilization
                        self._service_metrics.connection_wait_time = pool_metrics.connection_wait_time
                        
                        # Calculate queries per second
                        elapsed_time = (datetime.now(timezone.utc) - self._service_metrics.last_updated).total_seconds()
                        if elapsed_time > 0:
                            self._service_metrics.queries_per_second = self._service_metrics.total_queries / elapsed_time
                        
                        self._service_metrics.last_updated = datetime.now(timezone.utc)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting service metrics: {e}")
                await asyncio.sleep(10)
    
    async def get_service_metrics(self) -> ConnectionPoolMetrics:
        """Get current service metrics"""
        async with self._metrics_lock:
            return self._service_metrics
    
    async def shutdown(self):
        """Shutdown the async database service"""
        self.logger.info("Shutting down async database service")
        
        self._shutdown = True
        
        # Wait for queues to empty
        await self._query_queue.join()
        await self._batch_queue.join()
        
        # Cancel processors
        all_processors = self._query_processors + self._batch_processors
        for processor in all_processors:
            processor.cancel()
        
        # Wait for processors to finish
        await asyncio.gather(*all_processors, return_exceptions=True)
        
        # Shutdown connection pool
        if self.connection_pool:
            await self.connection_pool.shutdown()
        
        self.logger.info("Async database service shutdown complete")


# Global service instance
_async_database_service: Optional[AsyncDatabaseService] = None


async def get_async_database_service() -> AsyncDatabaseService:
    """Get or create global async database service instance"""
    global _async_database_service
    
    if _async_database_service is None:
        from core.config import load_config
        config = load_config()
        _async_database_service = AsyncDatabaseService(config.supabase)
        await _async_database_service.initialize()
    
    return _async_database_service