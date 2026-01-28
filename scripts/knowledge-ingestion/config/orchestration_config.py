"""
Orchestration Configuration Management

This module provides configuration management specifically for the multi-source
pipeline orchestration system, including default configurations, environment-specific
overrides, and validation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import timedelta

from core.config import get_settings
from services.multi_source_auth import DataSourceType


class OrchestrationMode(Enum):
    """Orchestration execution modes"""
    SEQUENTIAL = "sequential"  # Process sources one by one
    PARALLEL = "parallel"     # Process sources concurrently
    ADAPTIVE = "adaptive"     # Adapt based on resources


class ProgressReportingLevel(Enum):
    """Progress reporting detail levels"""
    MINIMAL = "minimal"       # Basic status only
    STANDARD = "standard"     # Standard progress updates
    DETAILED = "detailed"     # Detailed metrics and timing
    VERBOSE = "verbose"       # All available information


@dataclass
class SourceConfiguration:
    """Configuration for a specific data source"""
    source_type: DataSourceType
    enabled: bool = True
    priority: int = 0  # Higher numbers = higher priority
    max_concurrent_files: int = 16
    retry_attempts: int = 3
    timeout_seconds: int = 300
    
    # Source-specific settings
    source_specific_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfiguration:
    """Performance optimization configuration"""
    # Concurrency settings
    max_concurrent_sources: int = 4
    max_concurrent_files_global: int = 64
    max_concurrent_files_per_source: int = 16
    
    # Resource management
    enable_resource_monitoring: bool = True
    memory_threshold_mb: int = 8192
    cpu_threshold_percent: float = 80.0
    
    # Adaptive processing
    enable_adaptive_batching: bool = True
    enable_intelligent_queuing: bool = True
    enable_backpressure_handling: bool = True
    
    # GPU acceleration
    enable_gpu_acceleration: bool = True
    gpu_memory_limit_mb: int = 4096
    gpu_batch_size: int = 32
    
    # Database optimization
    enable_batch_database_ops: bool = True
    database_batch_size: int = 100
    max_database_connections: int = 10


@dataclass
class MonitoringConfiguration:
    """Monitoring and logging configuration"""
    # Progress reporting
    progress_update_interval_seconds: int = 5
    reporting_level: ProgressReportingLevel = ProgressReportingLevel.STANDARD
    enable_websocket_updates: bool = True
    
    # Logging
    enable_detailed_logging: bool = True
    enable_performance_logging: bool = True
    enable_error_tracking: bool = True
    log_correlation_ids: bool = True
    
    # Metrics collection
    enable_metrics_collection: bool = True
    metrics_collection_interval_seconds: int = 10
    metrics_retention_hours: int = 24
    
    # Health monitoring
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30


@dataclass
class ErrorHandlingConfiguration:
    """Error handling and recovery configuration"""
    # Retry behavior
    max_retries_per_source: int = 3
    max_retries_per_file: int = 2
    retry_delay_seconds: float = 5.0
    exponential_backoff: bool = True
    max_retry_delay_seconds: float = 60.0
    
    # Failure handling
    continue_on_source_failure: bool = True
    continue_on_file_failure: bool = True
    fail_fast_on_critical_errors: bool = False
    
    # Recovery options
    enable_checkpoint_recovery: bool = True
    enable_graceful_degradation: bool = True
    fallback_to_sync_processing: bool = True


@dataclass
class QualityConfiguration:
    """Quality assurance and audit configuration"""
    # Quality audit
    enable_quality_audit: bool = True
    audit_sample_size: int = 100
    audit_confidence_level: float = 0.95
    
    # Coverage analysis
    enable_coverage_analysis: bool = True
    coverage_domains: List[str] = field(default_factory=lambda: [
        "machine_learning", "deep_learning", "nlp", "finance", "mathematics"
    ])
    
    # Readiness assessment
    generate_readiness_memo: bool = True
    readiness_threshold_score: float = 0.8
    
    # Content validation
    enable_content_validation: bool = True
    validate_embeddings: bool = True
    validate_mathematical_content: bool = True


@dataclass
class OrchestrationConfiguration:
    """Complete orchestration configuration"""
    # Basic settings
    orchestration_mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    execution_timeout_hours: int = 24
    enable_state_persistence: bool = True
    
    # Source configurations
    source_configs: Dict[str, SourceConfiguration] = field(default_factory=dict)
    default_source_config: SourceConfiguration = field(default_factory=SourceConfiguration)
    
    # Component configurations
    performance: PerformanceConfiguration = field(default_factory=PerformanceConfiguration)
    monitoring: MonitoringConfiguration = field(default_factory=MonitoringConfiguration)
    error_handling: ErrorHandlingConfiguration = field(default_factory=ErrorHandlingConfiguration)
    quality: QualityConfiguration = field(default_factory=QualityConfiguration)
    
    # Integration settings
    enable_websocket_integration: bool = True
    enable_api_integration: bool = True
    enable_ui_integration: bool = True
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class OrchestrationConfigManager:
    """Manager for orchestration configuration"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.settings = get_settings()
        self._config: Optional[OrchestrationConfiguration] = None
        self._config_file_path: Optional[Path] = None
    
    def load_config(
        self,
        config_file: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None
    ) -> OrchestrationConfiguration:
        """Load orchestration configuration from file or create default"""
        try:
            # Determine config file path
            if config_file:
                self._config_file_path = Path(config_file)
            else:
                env = environment or self.settings.environment
                self._config_file_path = self.config_dir / f"orchestration_{env}.yaml"
                
                # Fallback to default if environment-specific doesn't exist
                if not self._config_file_path.exists():
                    self._config_file_path = self.config_dir / "orchestration_default.yaml"
            
            # Load from file if exists
            if self._config_file_path.exists():
                self._config = self._load_from_file(self._config_file_path)
            else:
                # Create default configuration
                self._config = self._create_default_config()
                self._save_to_file(self._config_file_path, self._config)
            
            # Apply environment-specific overrides
            if environment and environment in self._config.environment_overrides:
                self._apply_overrides(self._config.environment_overrides[environment])
            
            return self._config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load orchestration configuration: {e}")
    
    def _load_from_file(self, file_path: Path) -> OrchestrationConfiguration:
        """Load configuration from YAML or JSON file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return self._dict_to_config(data)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {e}")
    
    def _save_to_file(self, file_path: Path, config: OrchestrationConfiguration):
        """Save configuration to file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = self._config_to_dict(config)
            
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2, default=str)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {file_path}: {e}")
    
    def _create_default_config(self) -> OrchestrationConfiguration:
        """Create default orchestration configuration"""
        config = OrchestrationConfiguration()
        
        # Configure default source settings for each supported type
        for source_type in DataSourceType:
            source_config = SourceConfiguration(
                source_type=source_type,
                enabled=True,
                priority=self._get_default_source_priority(source_type),
                max_concurrent_files=self._get_default_concurrent_files(source_type)
            )
            config.source_configs[source_type.value] = source_config
        
        # Environment-specific overrides
        config.environment_overrides = {
            "development": {
                "performance.max_concurrent_sources": 2,
                "performance.max_concurrent_files_per_source": 8,
                "monitoring.progress_update_interval_seconds": 2,
                "monitoring.reporting_level": "verbose"
            },
            "production": {
                "performance.max_concurrent_sources": 8,
                "performance.max_concurrent_files_per_source": 32,
                "monitoring.progress_update_interval_seconds": 10,
                "monitoring.reporting_level": "standard",
                "error_handling.continue_on_source_failure": True
            }
        }
        
        return config
    
    def _get_default_source_priority(self, source_type: DataSourceType) -> int:
        """Get default priority for source type"""
        priority_map = {
            DataSourceType.GOOGLE_DRIVE: 10,
            DataSourceType.LOCAL_DIRECTORY: 8,
            DataSourceType.LOCAL_ZIP: 6,
            DataSourceType.INDIVIDUAL_UPLOAD: 4,
            DataSourceType.AWS_S3: 7,
            DataSourceType.AZURE_BLOB: 7,
            DataSourceType.GOOGLE_CLOUD_STORAGE: 7
        }
        return priority_map.get(source_type, 5)
    
    def _get_default_concurrent_files(self, source_type: DataSourceType) -> int:
        """Get default concurrent files for source type"""
        concurrent_map = {
            DataSourceType.GOOGLE_DRIVE: 8,   # API rate limits
            DataSourceType.LOCAL_DIRECTORY: 32,  # Fast local access
            DataSourceType.LOCAL_ZIP: 16,    # I/O bound
            DataSourceType.INDIVIDUAL_UPLOAD: 4,  # Usually small batches
            DataSourceType.AWS_S3: 16,       # Good throughput
            DataSourceType.AZURE_BLOB: 16,   # Good throughput
            DataSourceType.GOOGLE_CLOUD_STORAGE: 16  # Good throughput
        }
        return concurrent_map.get(source_type, 16)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> OrchestrationConfiguration:
        """Convert dictionary to configuration object"""
        # This would implement proper deserialization
        # For now, create a basic implementation
        config = OrchestrationConfiguration()
        
        # Update fields from data
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _config_to_dict(self, config: OrchestrationConfiguration) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        return asdict(config)
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply environment-specific overrides"""
        for key, value in overrides.items():
            self._set_nested_value(self._config, key, value)
    
    def _set_nested_value(self, obj: Any, key_path: str, value: Any):
        """Set nested value using dot notation"""
        keys = key_path.split('.')
        current = obj
        
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return  # Skip if path doesn't exist
        
        if hasattr(current, keys[-1]):
            setattr(current, keys[-1], value)
    
    def validate_config(self, config: Optional[OrchestrationConfiguration] = None) -> Dict[str, Any]:
        """Validate orchestration configuration"""
        config = config or self._config
        if not config:
            return {"valid": False, "errors": ["No configuration loaded"]}
        
        errors = []
        warnings = []
        
        # Validate performance settings
        if config.performance.max_concurrent_sources <= 0:
            errors.append("max_concurrent_sources must be positive")
        
        if config.performance.max_concurrent_files_per_source <= 0:
            errors.append("max_concurrent_files_per_source must be positive")
        
        if config.performance.memory_threshold_mb <= 0:
            errors.append("memory_threshold_mb must be positive")
        
        # Validate monitoring settings
        if config.monitoring.progress_update_interval_seconds <= 0:
            errors.append("progress_update_interval_seconds must be positive")
        
        # Validate error handling settings
        if config.error_handling.max_retries_per_source < 0:
            errors.append("max_retries_per_source cannot be negative")
        
        if config.error_handling.retry_delay_seconds <= 0:
            warnings.append("retry_delay_seconds should be positive")
        
        # Validate source configurations
        for source_type_str, source_config in config.source_configs.items():
            try:
                DataSourceType(source_type_str)
            except ValueError:
                errors.append(f"Invalid source type: {source_type_str}")
            
            if source_config.max_concurrent_files <= 0:
                errors.append(f"max_concurrent_files for {source_type_str} must be positive")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def get_config(self) -> Optional[OrchestrationConfiguration]:
        """Get current configuration"""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> OrchestrationConfiguration:
        """Update configuration with new values"""
        if not self._config:
            raise RuntimeError("No configuration loaded")
        
        for key, value in updates.items():
            self._set_nested_value(self._config, key, value)
        
        # Save updated configuration
        if self._config_file_path:
            self._save_to_file(self._config_file_path, self._config)
        
        return self._config


# Global configuration manager instance
_config_manager: Optional[OrchestrationConfigManager] = None


def get_orchestration_config_manager() -> OrchestrationConfigManager:
    """Get global orchestration configuration manager"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = OrchestrationConfigManager()
    
    return _config_manager


def load_orchestration_config(
    config_file: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> OrchestrationConfiguration:
    """Load orchestration configuration"""
    manager = get_orchestration_config_manager()
    return manager.load_config(config_file, environment)