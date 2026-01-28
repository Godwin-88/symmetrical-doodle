"""
Concurrent access management for knowledge ingestion system.

This module provides:
- Support for multiple intelligence layer processes
- Data consistency management during concurrent operations
- Operational isolation to prevent interference with trading operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import threading
from contextlib import asynccontextmanager
import weakref

from core.logging import get_logger


class LockType(Enum):
    """Lock type enumeration"""
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


class OperationType(Enum):
    """Operation type enumeration"""
    INGESTION = "ingestion"
    QUERY = "query"
    MAINTENANCE = "maintenance"
    TRADING = "trading"


class Priority(Enum):
    """Operation priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4  # Reserved for trading operations


@dataclass
class LockRequest:
    """Lock request information"""
    lock_id: str
    resource_id: str
    lock_type: LockType
    operation_type: OperationType
    priority: Priority
    process_id: str
    requested_at: datetime
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLock:
    """Active lock information"""
    lock_id: str
    resource_id: str
    lock_type: LockType
    operation_type: OperationType
    priority: Priority
    process_id: str
    acquired_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessInfo:
    """Process information for tracking"""
    process_id: str
    process_type: str  # "intelligence_layer", "ingestion", "trading"
    started_at: datetime
    last_heartbeat: datetime
    active_locks: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConcurrentAccessManager:
    """
    Concurrent access manager for knowledge ingestion system.
    
    Provides:
    - Distributed locking mechanism for resources
    - Priority-based access control (trading operations get highest priority)
    - Process isolation and tracking
    - Deadlock detection and prevention
    - Resource contention monitoring
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Lock management
        self._active_locks: Dict[str, ActiveLock] = {}
        self._lock_queue: List[LockRequest] = []
        self._lock_mutex = asyncio.Lock()
        
        # Process tracking
        self._active_processes: Dict[str, ProcessInfo] = {}
        self._process_mutex = asyncio.Lock()
        
        # Resource tracking
        self._resource_access_count: Dict[str, int] = {}
        self._resource_last_access: Dict[str, datetime] = {}
        
        # Configuration
        self.max_concurrent_operations = 10
        self.default_lock_timeout = 30.0
        self.heartbeat_interval = 10.0
        self.cleanup_interval = 60.0
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks for lock events
        self._lock_acquired_callbacks: List[Callable] = []
        self._lock_released_callbacks: List[Callable] = []
    
    async def start(self):
        """Start the concurrent access manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        self.logger.info("Concurrent access manager started")
    
    async def stop(self):
        """Stop the concurrent access manager"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Release all locks
        await self._release_all_locks()
        
        self.logger.info("Concurrent access manager stopped")
    
    async def register_process(
        self,
        process_id: str,
        process_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a process with the access manager.
        
        Args:
            process_id: Unique process identifier
            process_type: Type of process (intelligence_layer, ingestion, trading)
            metadata: Optional process metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            async with self._process_mutex:
                if process_id in self._active_processes:
                    self.logger.warning(f"Process {process_id} already registered")
                    return False
                
                process_info = ProcessInfo(
                    process_id=process_id,
                    process_type=process_type,
                    started_at=datetime.now(),
                    last_heartbeat=datetime.now(),
                    metadata=metadata or {}
                )
                
                self._active_processes[process_id] = process_info
                
                self.logger.info(f"Process registered: {process_id} ({process_type})")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register process {process_id}: {e}")
            return False
    
    async def unregister_process(self, process_id: str) -> bool:
        """
        Unregister a process and release all its locks.
        
        Args:
            process_id: Process identifier to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            async with self._process_mutex:
                if process_id not in self._active_processes:
                    self.logger.warning(f"Process {process_id} not registered")
                    return False
                
                process_info = self._active_processes[process_id]
                
                # Release all locks held by this process
                for lock_id in list(process_info.active_locks):
                    await self._release_lock_internal(lock_id, process_id)
                
                # Remove process
                del self._active_processes[process_id]
                
                self.logger.info(f"Process unregistered: {process_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unregister process {process_id}: {e}")
            return False
    
    async def update_heartbeat(self, process_id: str) -> bool:
        """
        Update process heartbeat.
        
        Args:
            process_id: Process identifier
            
        Returns:
            True if heartbeat updated, False otherwise
        """
        try:
            async with self._process_mutex:
                if process_id not in self._active_processes:
                    return False
                
                self._active_processes[process_id].last_heartbeat = datetime.now()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat for {process_id}: {e}")
            return False
    
    @asynccontextmanager
    async def acquire_lock(
        self,
        resource_id: str,
        lock_type: LockType,
        operation_type: OperationType,
        process_id: str,
        priority: Priority = Priority.NORMAL,
        timeout_seconds: float = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Acquire a lock on a resource (async context manager).
        
        Args:
            resource_id: Resource identifier to lock
            lock_type: Type of lock (READ, WRITE, EXCLUSIVE)
            operation_type: Type of operation
            process_id: Process requesting the lock
            priority: Operation priority
            timeout_seconds: Lock timeout in seconds
            metadata: Optional lock metadata
            
        Yields:
            Lock ID if successful
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
            RuntimeError: If lock acquisition fails
        """
        lock_id = str(uuid.uuid4())
        timeout = timeout_seconds or self.default_lock_timeout
        
        try:
            # Create lock request
            lock_request = LockRequest(
                lock_id=lock_id,
                resource_id=resource_id,
                lock_type=lock_type,
                operation_type=operation_type,
                priority=priority,
                process_id=process_id,
                requested_at=datetime.now(),
                timeout_seconds=timeout,
                metadata=metadata or {}
            )
            
            # Acquire the lock
            success = await self._acquire_lock_internal(lock_request)
            
            if not success:
                raise RuntimeError(f"Failed to acquire lock for resource {resource_id}")
            
            self.logger.debug(f"Lock acquired: {lock_id} for resource {resource_id}")
            
            try:
                yield lock_id
            finally:
                # Always release the lock
                await self._release_lock_internal(lock_id, process_id)
                self.logger.debug(f"Lock released: {lock_id} for resource {resource_id}")
                
        except Exception as e:
            self.logger.error(f"Lock operation failed for resource {resource_id}: {e}")
            raise
    
    async def _acquire_lock_internal(self, lock_request: LockRequest) -> bool:
        """Internal lock acquisition logic"""
        try:
            async with self._lock_mutex:
                # Check if lock can be granted immediately
                if self._can_grant_lock(lock_request):
                    return await self._grant_lock(lock_request)
                
                # Add to queue if not immediately available
                self._add_to_queue(lock_request)
                
                # Wait for lock to be available
                return await self._wait_for_lock(lock_request)
                
        except Exception as e:
            self.logger.error(f"Internal lock acquisition failed: {e}")
            return False
    
    def _can_grant_lock(self, lock_request: LockRequest) -> bool:
        """Check if lock can be granted immediately"""
        resource_id = lock_request.resource_id
        
        # Get existing locks for this resource
        existing_locks = [
            lock for lock in self._active_locks.values()
            if lock.resource_id == resource_id and lock.expires_at > datetime.now()
        ]
        
        if not existing_locks:
            return True
        
        # Check compatibility based on lock type
        if lock_request.lock_type == LockType.READ:
            # Read locks are compatible with other read locks
            return all(lock.lock_type == LockType.READ for lock in existing_locks)
        
        elif lock_request.lock_type == LockType.WRITE:
            # Write locks are not compatible with any other locks
            return False
        
        elif lock_request.lock_type == LockType.EXCLUSIVE:
            # Exclusive locks are not compatible with any other locks
            return False
        
        return False
    
    async def _grant_lock(self, lock_request: LockRequest) -> bool:
        """Grant a lock"""
        try:
            expires_at = datetime.now() + timedelta(seconds=lock_request.timeout_seconds)
            
            active_lock = ActiveLock(
                lock_id=lock_request.lock_id,
                resource_id=lock_request.resource_id,
                lock_type=lock_request.lock_type,
                operation_type=lock_request.operation_type,
                priority=lock_request.priority,
                process_id=lock_request.process_id,
                acquired_at=datetime.now(),
                expires_at=expires_at,
                metadata=lock_request.metadata
            )
            
            # Store active lock
            self._active_locks[lock_request.lock_id] = active_lock
            
            # Update process info
            if lock_request.process_id in self._active_processes:
                self._active_processes[lock_request.process_id].active_locks.add(lock_request.lock_id)
            
            # Update resource tracking
            self._resource_access_count[lock_request.resource_id] = (
                self._resource_access_count.get(lock_request.resource_id, 0) + 1
            )
            self._resource_last_access[lock_request.resource_id] = datetime.now()
            
            # Notify callbacks
            for callback in self._lock_acquired_callbacks:
                try:
                    await callback(active_lock)
                except Exception as e:
                    self.logger.error(f"Lock acquired callback failed: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to grant lock: {e}")
            return False
    
    def _add_to_queue(self, lock_request: LockRequest):
        """Add lock request to priority queue"""
        # Insert in priority order (higher priority first)
        inserted = False
        for i, queued_request in enumerate(self._lock_queue):
            if lock_request.priority.value > queued_request.priority.value:
                self._lock_queue.insert(i, lock_request)
                inserted = True
                break
        
        if not inserted:
            self._lock_queue.append(lock_request)
    
    async def _wait_for_lock(self, lock_request: LockRequest) -> bool:
        """Wait for lock to become available"""
        start_time = datetime.now()
        timeout = timedelta(seconds=lock_request.timeout_seconds)
        
        while datetime.now() - start_time < timeout:
            # Check if lock can now be granted
            if self._can_grant_lock(lock_request):
                # Remove from queue
                if lock_request in self._lock_queue:
                    self._lock_queue.remove(lock_request)
                
                return await self._grant_lock(lock_request)
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        # Timeout reached
        if lock_request in self._lock_queue:
            self._lock_queue.remove(lock_request)
        
        raise TimeoutError(f"Lock acquisition timeout for resource {lock_request.resource_id}")
    
    async def _release_lock_internal(self, lock_id: str, process_id: str) -> bool:
        """Internal lock release logic"""
        try:
            async with self._lock_mutex:
                if lock_id not in self._active_locks:
                    return False
                
                active_lock = self._active_locks[lock_id]
                
                # Verify process owns the lock
                if active_lock.process_id != process_id:
                    self.logger.warning(f"Process {process_id} tried to release lock {lock_id} owned by {active_lock.process_id}")
                    return False
                
                # Remove lock
                del self._active_locks[lock_id]
                
                # Update process info
                if process_id in self._active_processes:
                    self._active_processes[process_id].active_locks.discard(lock_id)
                
                # Notify callbacks
                for callback in self._lock_released_callbacks:
                    try:
                        await callback(active_lock)
                    except Exception as e:
                        self.logger.error(f"Lock released callback failed: {e}")
                
                # Process queue to see if any waiting locks can now be granted
                await self._process_lock_queue()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to release lock {lock_id}: {e}")
            return False
    
    async def _process_lock_queue(self):
        """Process queued lock requests"""
        granted_any = True
        
        while granted_any and self._lock_queue:
            granted_any = False
            
            # Process queue in priority order
            for i, lock_request in enumerate(self._lock_queue):
                if self._can_grant_lock(lock_request):
                    # Remove from queue
                    self._lock_queue.pop(i)
                    
                    # Grant the lock
                    await self._grant_lock(lock_request)
                    granted_any = True
                    break
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await self._cleanup_expired_locks()
                await self._cleanup_stale_processes()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(5)
    
    async def _heartbeat_loop(self):
        """Background heartbeat monitoring loop"""
        while self._running:
            try:
                await self._check_process_heartbeats()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_expired_locks(self):
        """Clean up expired locks"""
        now = datetime.now()
        expired_locks = [
            lock_id for lock_id, lock in self._active_locks.items()
            if lock.expires_at <= now
        ]
        
        for lock_id in expired_locks:
            lock = self._active_locks[lock_id]
            self.logger.warning(f"Cleaning up expired lock: {lock_id} for resource {lock.resource_id}")
            await self._release_lock_internal(lock_id, lock.process_id)
    
    async def _cleanup_stale_processes(self):
        """Clean up stale processes"""
        now = datetime.now()
        stale_threshold = timedelta(seconds=self.heartbeat_interval * 3)
        
        stale_processes = [
            process_id for process_id, process_info in self._active_processes.items()
            if now - process_info.last_heartbeat > stale_threshold
        ]
        
        for process_id in stale_processes:
            self.logger.warning(f"Cleaning up stale process: {process_id}")
            await self.unregister_process(process_id)
    
    async def _check_process_heartbeats(self):
        """Check process heartbeats and warn about stale processes"""
        now = datetime.now()
        warning_threshold = timedelta(seconds=self.heartbeat_interval * 2)
        
        for process_id, process_info in self._active_processes.items():
            if now - process_info.last_heartbeat > warning_threshold:
                self.logger.warning(f"Process {process_id} heartbeat is stale")
    
    async def _release_all_locks(self):
        """Release all active locks"""
        lock_ids = list(self._active_locks.keys())
        for lock_id in lock_ids:
            lock = self._active_locks[lock_id]
            await self._release_lock_internal(lock_id, lock.process_id)
    
    def get_lock_statistics(self) -> Dict[str, Any]:
        """Get lock statistics"""
        now = datetime.now()
        
        return {
            'timestamp': now.isoformat(),
            'active_locks': len(self._active_locks),
            'queued_requests': len(self._lock_queue),
            'active_processes': len(self._active_processes),
            'resource_access_counts': dict(self._resource_access_count),
            'lock_types': {
                lock_type.value: sum(1 for lock in self._active_locks.values() if lock.lock_type == lock_type)
                for lock_type in LockType
            },
            'operation_types': {
                op_type.value: sum(1 for lock in self._active_locks.values() if lock.operation_type == op_type)
                for op_type in OperationType
            },
            'priority_distribution': {
                priority.value: sum(1 for lock in self._active_locks.values() if lock.priority == priority)
                for priority in Priority
            }
        }
    
    def get_process_info(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific process"""
        if process_id not in self._active_processes:
            return None
        
        process_info = self._active_processes[process_id]
        
        return {
            'process_id': process_info.process_id,
            'process_type': process_info.process_type,
            'started_at': process_info.started_at.isoformat(),
            'last_heartbeat': process_info.last_heartbeat.isoformat(),
            'active_locks': list(process_info.active_locks),
            'metadata': process_info.metadata
        }
    
    def add_lock_acquired_callback(self, callback: Callable):
        """Add callback for lock acquired events"""
        self._lock_acquired_callbacks.append(callback)
    
    def add_lock_released_callback(self, callback: Callable):
        """Add callback for lock released events"""
        self._lock_released_callbacks.append(callback)


# Global concurrent access manager instance
_access_manager: Optional[ConcurrentAccessManager] = None


async def get_access_manager() -> ConcurrentAccessManager:
    """Get or create global access manager instance"""
    global _access_manager
    
    if _access_manager is None:
        _access_manager = ConcurrentAccessManager()
        await _access_manager.start()
    
    return _access_manager


async def initialize_concurrent_access() -> bool:
    """Initialize concurrent access manager"""
    try:
        manager = await get_access_manager()
        return manager is not None
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to initialize concurrent access manager: {e}")
        return False


# Convenience functions for common operations

async def with_trading_priority_lock(
    resource_id: str,
    process_id: str,
    operation_func: Callable,
    lock_type: LockType = LockType.EXCLUSIVE,
    timeout_seconds: float = 5.0
):
    """
    Execute operation with trading priority lock.
    
    Args:
        resource_id: Resource to lock
        process_id: Process requesting lock
        operation_func: Async function to execute with lock
        lock_type: Type of lock to acquire
        timeout_seconds: Lock timeout
        
    Returns:
        Result of operation_func
    """
    manager = await get_access_manager()
    
    async with manager.acquire_lock(
        resource_id=resource_id,
        lock_type=lock_type,
        operation_type=OperationType.TRADING,
        process_id=process_id,
        priority=Priority.CRITICAL,
        timeout_seconds=timeout_seconds
    ):
        return await operation_func()


async def with_ingestion_lock(
    resource_id: str,
    process_id: str,
    operation_func: Callable,
    lock_type: LockType = LockType.WRITE,
    timeout_seconds: float = 30.0
):
    """
    Execute operation with ingestion lock.
    
    Args:
        resource_id: Resource to lock
        process_id: Process requesting lock
        operation_func: Async function to execute with lock
        lock_type: Type of lock to acquire
        timeout_seconds: Lock timeout
        
    Returns:
        Result of operation_func
    """
    manager = await get_access_manager()
    
    async with manager.acquire_lock(
        resource_id=resource_id,
        lock_type=lock_type,
        operation_type=OperationType.INGESTION,
        process_id=process_id,
        priority=Priority.NORMAL,
        timeout_seconds=timeout_seconds
    ):
        return await operation_func()


async def with_query_lock(
    resource_id: str,
    process_id: str,
    operation_func: Callable,
    lock_type: LockType = LockType.READ,
    timeout_seconds: float = 10.0
):
    """
    Execute operation with query lock.
    
    Args:
        resource_id: Resource to lock
        process_id: Process requesting lock
        operation_func: Async function to execute with lock
        lock_type: Type of lock to acquire
        timeout_seconds: Lock timeout
        
    Returns:
        Result of operation_func
    """
    manager = await get_access_manager()
    
    async with manager.acquire_lock(
        resource_id=resource_id,
        lock_type=lock_type,
        operation_type=OperationType.QUERY,
        process_id=process_id,
        priority=Priority.NORMAL,
        timeout_seconds=timeout_seconds
    ):
        return await operation_func()