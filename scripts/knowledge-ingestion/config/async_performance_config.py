"""
Async Performance Configuration

This module provides configuration management for all async performance
optimizations including worker pools, GPU acceleration, database connections,
and resource management settings.

Requirements: 10.1, 10.2, 10.4
"""

import os
import multiprocessing as mp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import psutil


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class WorkerPoolSettings:
    """Worker pool configuration settings"""
    # I/O-bound operations (file downloads, API calls)
    io_pool_size: int = 32
    io_queue_size: int = 1000
    io_timeout_seconds: int = 300
    
    # CPU-bound operations (parsing, chunking)
    cpu_pool_size: int = mp.cpu_count() or 1
    cpu_queue_size: int = 500
    cpu_timeout_seconds: int = 600
    
    # GPU-accelerated operations (embeddings)
    gpu_pool_size: int = 4
    gpu_queue_size: int = 200
    gpu_timeout_seconds: int = 120
    gpu_batch_size: int = 32
    gpu_memory_limit_mb: int = 4096
    
    # Database operations
    db_pool_size: int = 16
    db_queue_size: int = 500
    db_timeout_seconds: int = 180
    db_batch_size: int = 100
    db_connection_pool_size: int = 10


@dataclass
class ConcurrencySettings:
    """Concurrency control settings"""
    # File processing concurrency
    max_concurrent_files: int = 16
    max_concurrent_sources: int = 4
    max_concurrent_jobs: int = 3
    
    # Batch processing
    adaptive_batch_size: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 32
    batch_timeout_seconds: float = 2.0
    
    # Queue management
    enable_priority_queuing: bool = True
    enable_intelligent_scheduling: bool = True
    
    # Backpressure handling
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8
    backpressure_recovery_threshold: float = 0.6


@dataclass
class ResourceLimits:
    """System resource limits and thresholds"""
    # Memory limits
    max_memory_usage_mb: int = 8192
    memory_warning_threshold_mb: int = 6144
    memory_critical_threshold_mb: int = 7680
    
    # CPU limits
    max_cpu_usage_percent: float = 80.0
    cpu_warning_threshold_percent: float = 60.0
    cpu_throttle_threshold_percent: float = 85.0
    
    # GPU limits (if available)
    max_gpu_memory_usage_percent: float = 90.0
    gpu_warning_threshold_percent: float = 70.0
    
    # Network limits
    max_network_connections: int = 100
    network_timeout_seconds: int = 30
    
    # Disk I/O limits
    max_disk_usage_percent: float = 90.0
    temp_file_cleanup_interval_seconds: int = 300


@dataclass
class PerformanceMonitoring:
    """Performance monitoring settings"""
    # Metrics collection
    enable_metrics_collection: bool = True
    metrics_collection_interval_seconds: int = 5
    metrics_history_size: int = 1000
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    resource_monitoring_interval_seconds: int = 5
    
    # Performance alerts
    enable_performance_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 80.0,
        'gpu_usage': 90.0,
        'queue_size': 500,
        'error_rate': 5.0
    })
    
    # Logging
    enable_performance_logging: bool = True
    log_level: str = "INFO"
    log_detailed_metrics: bool = False


@dataclass
class OptimizationFeatures:
    """Feature flags for various optimizations"""
    # Async processing
    enable_async_processing: bool = True
    enable_concurrent_file_processing: bool = True
    enable_concurrent_source_processing: bool = True
    
    # GPU acceleration
    enable_gpu_acceleration: bool = True
    auto_detect_gpu: bool = True
    prefer_gpu_for_embeddings: bool = True
    
    # Rust integration
    enable_rust_bindings: bool = True
    auto_detect_rust: bool = True
    prefer_rust_for_vector_ops: bool = True
    rust_fallback_on_error: bool = True
    rust_performance_benchmarking: bool = True
    
    # Vector operations optimization
    enable_vector_optimization: bool = True
    vector_cache_size_mb: int = 512
    enable_numpy_optimization: bool = True
    enable_scipy_optimization: bool = True
    vector_batch_size: int = 1000
    
    # Database optimizations
    enable_connection_pooling: bool = True
    enable_batch_database_operations: bool = True
    enable_query_caching: bool = True
    query_cache_ttl_seconds: int = 300
    
    # Intelligent features
    enable_adaptive_batching: bool = True
    enable_intelligent_queuing: bool = True
    enable_resource_aware_scheduling: bool = True
    enable_predictive_scaling: bool = True
    
    # Error handling and recovery
    enable_graceful_degradation: bool = True
    enable_automatic_fallback: bool = True
    enable_retry_with_backoff: bool = True
    max_retry_attempts: int = 3
    
    # Caching and optimization
    enable_result_caching: bool = True
    enable_preprocessing_cache: bool = True
    cache_cleanup_interval_seconds: int = 600


@dataclass
class AsyncPerformanceConfig:
    """Complete async performance configuration"""
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    worker_pools: WorkerPoolSettings = field(default_factory=WorkerPoolSettings)
    concurrency: ConcurrencySettings = field(default_factory=ConcurrencySettings)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    monitoring: PerformanceMonitoring = field(default_factory=PerformanceMonitoring)
    features: OptimizationFeatures = field(default_factory=OptimizationFeatures)
    
    def __post_init__(self):
        """Apply optimization level presets"""
        if self.optimization_level != OptimizationLevel.CUSTOM:
            self._apply_optimization_preset()
    
    def _apply_optimization_preset(self):
        """Apply predefined optimization presets"""
        if self.optimization_level == OptimizationLevel.MINIMAL:
            self._apply_minimal_preset()
        elif self.optimization_level == OptimizationLevel.BALANCED:
            self._apply_balanced_preset()
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self._apply_aggressive_preset()
    
    def _apply_minimal_preset(self):
        """Apply minimal optimization settings"""
        # Reduce worker pool sizes
        self.worker_pools.io_pool_size = min(8, self.worker_pools.io_pool_size)
        self.worker_pools.cpu_pool_size = min(2, self.worker_pools.cpu_pool_size)
        self.worker_pools.gpu_pool_size = min(2, self.worker_pools.gpu_pool_size)
        self.worker_pools.db_pool_size = min(4, self.worker_pools.db_pool_size)
        
        # Reduce concurrency
        self.concurrency.max_concurrent_files = min(4, self.concurrency.max_concurrent_files)
        self.concurrency.max_concurrent_sources = min(2, self.concurrency.max_concurrent_sources)
        self.concurrency.max_concurrent_jobs = min(1, self.concurrency.max_concurrent_jobs)
        
        # Conservative resource limits
        self.resource_limits.max_cpu_usage_percent = 60.0
        self.resource_limits.max_memory_usage_mb = min(4096, self.resource_limits.max_memory_usage_mb)
        
        # Disable some advanced features
        self.features.enable_predictive_scaling = False
        self.features.enable_intelligent_queuing = False
    
    def _apply_balanced_preset(self):
        """Apply balanced optimization settings (default)"""
        # Use moderate settings - already set in defaults
        pass
    
    def _apply_aggressive_preset(self):
        """Apply aggressive optimization settings"""
        # Increase worker pool sizes
        cpu_count = mp.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        self.worker_pools.io_pool_size = min(64, cpu_count * 8)
        self.worker_pools.cpu_pool_size = cpu_count
        self.worker_pools.gpu_pool_size = min(8, cpu_count // 2)
        self.worker_pools.db_pool_size = min(32, cpu_count * 4)
        
        # Increase concurrency
        self.concurrency.max_concurrent_files = min(32, cpu_count * 4)
        self.concurrency.max_concurrent_sources = min(8, cpu_count)
        self.concurrency.max_concurrent_jobs = min(5, max(2, cpu_count // 2))
        
        # Higher resource limits
        self.resource_limits.max_cpu_usage_percent = 90.0
        self.resource_limits.max_memory_usage_mb = min(int(memory_gb * 1024 * 0.8), 16384)
        
        # Enable all advanced features
        self.features.enable_predictive_scaling = True
        self.features.enable_intelligent_queuing = True
        self.features.enable_resource_aware_scheduling = True
    
    @classmethod
    def from_environment(cls) -> 'AsyncPerformanceConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Optimization level
        opt_level = os.getenv('ASYNC_OPTIMIZATION_LEVEL', 'balanced').lower()
        if opt_level in [level.value for level in OptimizationLevel]:
            config.optimization_level = OptimizationLevel(opt_level)
        
        # Worker pool settings
        config.worker_pools.io_pool_size = int(os.getenv('ASYNC_IO_POOL_SIZE', config.worker_pools.io_pool_size))
        config.worker_pools.cpu_pool_size = int(os.getenv('ASYNC_CPU_POOL_SIZE', config.worker_pools.cpu_pool_size))
        config.worker_pools.gpu_pool_size = int(os.getenv('ASYNC_GPU_POOL_SIZE', config.worker_pools.gpu_pool_size))
        config.worker_pools.db_pool_size = int(os.getenv('ASYNC_DB_POOL_SIZE', config.worker_pools.db_pool_size))
        
        # Concurrency settings
        config.concurrency.max_concurrent_files = int(os.getenv('ASYNC_MAX_CONCURRENT_FILES', config.concurrency.max_concurrent_files))
        config.concurrency.max_concurrent_sources = int(os.getenv('ASYNC_MAX_CONCURRENT_SOURCES', config.concurrency.max_concurrent_sources))
        config.concurrency.max_concurrent_jobs = int(os.getenv('ASYNC_MAX_CONCURRENT_JOBS', config.concurrency.max_concurrent_jobs))
        
        # Resource limits
        config.resource_limits.max_memory_usage_mb = int(os.getenv('ASYNC_MAX_MEMORY_MB', config.resource_limits.max_memory_usage_mb))
        config.resource_limits.max_cpu_usage_percent = float(os.getenv('ASYNC_MAX_CPU_PERCENT', config.resource_limits.max_cpu_usage_percent))
        
        # Feature flags
        config.features.enable_gpu_acceleration = os.getenv('ASYNC_ENABLE_GPU', 'true').lower() == 'true'
        config.features.enable_async_processing = os.getenv('ASYNC_ENABLE_ASYNC', 'true').lower() == 'true'
        config.features.enable_connection_pooling = os.getenv('ASYNC_ENABLE_POOLING', 'true').lower() == 'true'
        config.features.enable_rust_bindings = os.getenv('ASYNC_ENABLE_RUST', 'true').lower() == 'true'
        config.features.enable_vector_optimization = os.getenv('ASYNC_ENABLE_VECTOR_OPT', 'true').lower() == 'true'
        config.features.vector_batch_size = int(os.getenv('ASYNC_VECTOR_BATCH_SIZE', config.features.vector_batch_size))
        config.features.vector_cache_size_mb = int(os.getenv('ASYNC_VECTOR_CACHE_MB', config.features.vector_cache_size_mb))
        
        return config
    
    def auto_configure_for_system(self) -> 'AsyncPerformanceConfig':
        """Auto-configure based on system capabilities"""
        cpu_count = mp.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Adjust worker pools based on system resources
        if memory_gb < 4:
            # Low memory system
            self.worker_pools.io_pool_size = min(8, cpu_count * 2)
            self.worker_pools.cpu_pool_size = max(1, cpu_count // 2)
            self.concurrency.max_concurrent_files = min(4, cpu_count)
            self.resource_limits.max_memory_usage_mb = int(memory_gb * 1024 * 0.6)
        elif memory_gb < 8:
            # Medium memory system
            self.worker_pools.io_pool_size = min(16, cpu_count * 3)
            self.worker_pools.cpu_pool_size = cpu_count
            self.concurrency.max_concurrent_files = min(8, cpu_count * 2)
            self.resource_limits.max_memory_usage_mb = int(memory_gb * 1024 * 0.7)
        else:
            # High memory system
            self.worker_pools.io_pool_size = min(32, cpu_count * 4)
            self.worker_pools.cpu_pool_size = cpu_count
            self.concurrency.max_concurrent_files = min(16, cpu_count * 3)
            self.resource_limits.max_memory_usage_mb = min(int(memory_gb * 1024 * 0.8), 16384)
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.features.enable_gpu_acceleration = True
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.worker_pools.gpu_memory_limit_mb = min(int(gpu_memory_gb * 1024 * 0.8), 8192)
            else:
                self.features.enable_gpu_acceleration = False
        except ImportError:
            self.features.enable_gpu_acceleration = False
        
        # Adjust based on CPU count
        if cpu_count <= 2:
            # Low-end system
            self.concurrency.max_concurrent_sources = 1
            self.concurrency.max_concurrent_jobs = 1
            self.features.enable_predictive_scaling = False
        elif cpu_count <= 4:
            # Mid-range system
            self.concurrency.max_concurrent_sources = 2
            self.concurrency.max_concurrent_jobs = 2
        else:
            # High-end system
            self.concurrency.max_concurrent_sources = min(4, cpu_count // 2)
            self.concurrency.max_concurrent_jobs = min(3, cpu_count // 4)
        
        return self
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Check system resources
        cpu_count = mp.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Validate worker pool sizes
        if self.worker_pools.io_pool_size > cpu_count * 8:
            warnings.append(f"I/O pool size ({self.worker_pools.io_pool_size}) may be too large for {cpu_count} CPUs")
        
        if self.worker_pools.cpu_pool_size > cpu_count * 2:
            warnings.append(f"CPU pool size ({self.worker_pools.cpu_pool_size}) exceeds recommended limit for {cpu_count} CPUs")
        
        # Validate memory limits
        if self.resource_limits.max_memory_usage_mb > memory_gb * 1024 * 0.9:
            warnings.append(f"Memory limit ({self.resource_limits.max_memory_usage_mb}MB) may exceed available memory ({memory_gb:.1f}GB)")
        
        # Validate concurrency settings
        if self.concurrency.max_concurrent_files > self.worker_pools.io_pool_size:
            warnings.append("Max concurrent files exceeds I/O pool size")
        
        # Validate GPU settings
        if self.features.enable_gpu_acceleration:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("GPU acceleration enabled but CUDA not available")
            except ImportError:
                warnings.append("GPU acceleration enabled but PyTorch not installed")
        
        return warnings
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'optimization_level': self.optimization_level.value,
            'worker_pools': {
                'io_pool_size': self.worker_pools.io_pool_size,
                'cpu_pool_size': self.worker_pools.cpu_pool_size,
                'gpu_pool_size': self.worker_pools.gpu_pool_size,
                'db_pool_size': self.worker_pools.db_pool_size,
            },
            'concurrency': {
                'max_concurrent_files': self.concurrency.max_concurrent_files,
                'max_concurrent_sources': self.concurrency.max_concurrent_sources,
                'max_concurrent_jobs': self.concurrency.max_concurrent_jobs,
            },
            'resource_limits': {
                'max_memory_mb': self.resource_limits.max_memory_usage_mb,
                'max_cpu_percent': self.resource_limits.max_cpu_usage_percent,
            },
            'features': {
                'async_processing': self.features.enable_async_processing,
                'gpu_acceleration': self.features.enable_gpu_acceleration,
                'rust_bindings': self.features.enable_rust_bindings,
                'vector_optimization': self.features.enable_vector_optimization,
                'connection_pooling': self.features.enable_connection_pooling,
                'adaptive_batching': self.features.enable_adaptive_batching,
            }
        }


def create_default_config() -> AsyncPerformanceConfig:
    """Create default configuration"""
    return AsyncPerformanceConfig().auto_configure_for_system()


def create_config_from_env() -> AsyncPerformanceConfig:
    """Create configuration from environment variables"""
    return AsyncPerformanceConfig.from_environment().auto_configure_for_system()


def create_minimal_config() -> AsyncPerformanceConfig:
    """Create minimal resource configuration"""
    return AsyncPerformanceConfig(optimization_level=OptimizationLevel.MINIMAL)


def create_aggressive_config() -> AsyncPerformanceConfig:
    """Create aggressive performance configuration"""
    return AsyncPerformanceConfig(optimization_level=OptimizationLevel.AGGRESSIVE)


# Global configuration instance
_global_config: Optional[AsyncPerformanceConfig] = None


def get_async_performance_config() -> AsyncPerformanceConfig:
    """Get global async performance configuration"""
    global _global_config
    
    if _global_config is None:
        _global_config = create_config_from_env()
    
    return _global_config


def set_async_performance_config(config: AsyncPerformanceConfig):
    """Set global async performance configuration"""
    global _global_config
    _global_config = config