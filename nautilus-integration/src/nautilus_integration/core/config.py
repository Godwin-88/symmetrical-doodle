"""
Configuration management for NautilusTrader integration.

This module provides comprehensive configuration management following the
patterns established in the knowledge-ingestion system.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration for NautilusTrader integration."""
    
    postgres_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/trading_system",
        description="PostgreSQL connection URL"
    )
    neo4j_url: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URL"
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )


class NautilusEngineConfig(BaseModel):
    """NautilusTrader engine configuration."""
    
    # BacktestEngine configuration
    backtest_engine_id: str = Field(default="BACKTEST", description="Backtest engine ID")
    backtest_log_level: str = Field(default="INFO", description="Backtest logging level")
    backtest_cache_database: bool = Field(default=True, description="Enable backtest cache database")
    backtest_cache_database_flush: bool = Field(default=False, description="Flush cache database on start")
    
    # TradingNode configuration
    trading_node_id: str = Field(default="LIVE", description="Trading node ID")
    trading_log_level: str = Field(default="INFO", description="Trading logging level")
    trading_cache_database: bool = Field(default=True, description="Enable trading cache database")
    
    # Data configuration
    data_catalog_path: str = Field(
        default="./data/catalog",
        description="Path to Nautilus data catalog"
    )
    parquet_compression: str = Field(
        default="snappy",
        description="Parquet compression algorithm"
    )
    
    @validator("backtest_log_level", "trading_log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class SignalRouterConfig(BaseModel):
    """Signal Router & Buffer configuration."""
    
    # Buffer configuration
    buffer_max_size_mb: int = Field(
        default=100,
        description="Maximum buffer size in MB"
    )
    buffer_cleanup_interval: int = Field(
        default=300,
        description="Buffer cleanup interval in seconds"
    )
    buffer_retention_hours: int = Field(
        default=24,
        description="Buffer retention period in hours"
    )
    
    # Delivery configuration
    delivery_timeout: float = Field(
        default=30.0,
        description="Signal delivery timeout in seconds"
    )
    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for failed deliveries"
    )
    retry_delay_base: float = Field(
        default=1.0,
        description="Base retry delay in seconds (exponential backoff)"
    )
    max_pending_deliveries: int = Field(
        default=1000,
        description="Maximum pending deliveries queue size"
    )
    
    # Rate limiting
    default_max_signals_per_minute: int = Field(
        default=60,
        description="Default maximum signals per minute per strategy"
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limiting window in seconds"
    )
    
    # F5 Integration
    f5_connection_enabled: bool = Field(
        default=True,
        description="Enable F5 Intelligence Layer connection"
    )
    f5_connection_timeout: float = Field(
        default=10.0,
        description="F5 connection timeout in seconds"
    )
    f5_heartbeat_interval: float = Field(
        default=30.0,
        description="F5 heartbeat interval in seconds"
    )
    
    # Signal validation
    validation_enabled: bool = Field(
        default=True,
        description="Enable signal validation"
    )
    min_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for signals"
    )
    max_signal_age_seconds: int = Field(
        default=300,
        description="Maximum signal age in seconds"
    )


class SignalValidationConfig(BaseModel):
    """Signal validation configuration."""
    
    # Validation thresholds
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    max_age_seconds: int = Field(
        default=300,
        description="Maximum signal age in seconds"
    )
    min_metadata_completeness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum metadata completeness score"
    )
    
    # Quality assessment
    quality_assessment_enabled: bool = Field(
        default=True,
        description="Enable signal quality assessment"
    )
    historical_consistency_enabled: bool = Field(
        default=True,
        description="Enable historical consistency checking"
    )
    source_reliability_tracking: bool = Field(
        default=True,
        description="Enable source reliability tracking"
    )
    
    # Validation rules
    strict_format_validation: bool = Field(
        default=True,
        description="Enable strict format validation"
    )
    content_validation_enabled: bool = Field(
        default=True,
        description="Enable content validation"
    )
    outlier_detection_enabled: bool = Field(
        default=True,
        description="Enable statistical outlier detection"
    )


class IntegrationConfig(BaseModel):
    """Integration layer configuration."""
    
    # Strategy translation
    strategy_translation_enabled: bool = Field(
        default=True,
        description="Enable F6 to Nautilus strategy translation"
    )
    strategy_validation_enabled: bool = Field(
        default=True,
        description="Enable strategy validation before deployment"
    )
    strategy_hot_swap_enabled: bool = Field(
        default=True,
        description="Enable hot-swapping of strategies"
    )
    
    # Signal routing
    signal_routing_enabled: bool = Field(
        default=True,
        description="Enable AI signal routing from F5"
    )
    signal_buffer_size: int = Field(
        default=10000,
        description="Signal buffer size for backtesting"
    )
    signal_delivery_timeout: float = Field(
        default=1.0,
        description="Signal delivery timeout in seconds"
    )
    
    # Risk management
    risk_integration_enabled: bool = Field(
        default=True,
        description="Enable F8 risk management integration"
    )
    position_sync_interval: float = Field(
        default=1.0,
        description="Position synchronization interval in seconds"
    )
    
    # Performance monitoring
    performance_monitoring_enabled: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    metrics_collection_interval: float = Field(
        default=5.0,
        description="Metrics collection interval in seconds"
    )


class ErrorHandlingConfig(BaseModel):
    """Error handling and resilience configuration."""
    
    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(
        default=2.0,
        description="Exponential backoff factor"
    )
    retry_max_delay: float = Field(
        default=60.0,
        description="Maximum retry delay in seconds"
    )
    
    # Circuit breaker configuration
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Circuit breaker failure threshold"
    )
    circuit_breaker_recovery_timeout: float = Field(
        default=30.0,
        description="Circuit breaker recovery timeout in seconds"
    )
    
    # Graceful degradation
    graceful_degradation_enabled: bool = Field(
        default=True,
        description="Enable graceful degradation on failures"
    )
    fallback_to_legacy_engine: bool = Field(
        default=True,
        description="Fallback to legacy F7 engine on Nautilus failures"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    correlation_id_enabled: bool = Field(
        default=True,
        description="Enable correlation ID tracking"
    )
    structured_logging: bool = Field(
        default=True,
        description="Enable structured logging with JSON output"
    )
    log_file_path: Optional[str] = Field(
        default=None,
        description="Path to log file (None for stdout only)"
    )
    log_rotation_enabled: bool = Field(
        default=True,
        description="Enable log file rotation"
    )
    log_max_size: str = Field(
        default="100MB",
        description="Maximum log file size before rotation"
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    
    @validator("level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""
    
    # Metrics collection
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    max_metric_history: int = Field(
        default=10000,
        description="Maximum number of metrics to keep in memory"
    )
    metrics_retention_hours: int = Field(
        default=24,
        description="Metrics retention period in hours"
    )
    
    # Performance monitoring
    latency_tracking_enabled: bool = Field(
        default=True,
        description="Enable nanosecond precision latency tracking"
    )
    performance_monitoring_interval: float = Field(
        default=5.0,
        description="Performance monitoring interval in seconds"
    )
    
    # Health checks
    health_checks_enabled: bool = Field(
        default=True,
        description="Enable health check monitoring"
    )
    health_check_interval: float = Field(
        default=30.0,
        description="Health check interval in seconds"
    )
    health_check_timeout: float = Field(
        default=10.0,
        description="Health check timeout in seconds"
    )
    
    # System monitoring
    system_monitoring_enabled: bool = Field(
        default=True,
        description="Enable system-level monitoring"
    )
    system_monitor_interval: int = Field(
        default=30,
        description="System monitoring interval in seconds"
    )
    
    # Alerting
    alerting_enabled: bool = Field(
        default=True,
        description="Enable alerting system"
    )
    alert_retention_hours: int = Field(
        default=168,  # 7 days
        description="Alert retention period in hours"
    )
    
    # Thresholds
    cpu_warning_threshold: float = Field(
        default=80.0,
        description="CPU usage warning threshold (percentage)"
    )
    cpu_critical_threshold: float = Field(
        default=95.0,
        description="CPU usage critical threshold (percentage)"
    )
    memory_warning_threshold: float = Field(
        default=85.0,
        description="Memory usage warning threshold (percentage)"
    )
    memory_critical_threshold: float = Field(
        default=95.0,
        description="Memory usage critical threshold (percentage)"
    )
    disk_warning_threshold: float = Field(
        default=80.0,
        description="Disk usage warning threshold (percentage)"
    )
    disk_critical_threshold: float = Field(
        default=90.0,
        description="Disk usage critical threshold (percentage)"
    )
    
    # Latency thresholds (in milliseconds)
    latency_warning_threshold: float = Field(
        default=100.0,
        description="Latency warning threshold in milliseconds"
    )
    latency_critical_threshold: float = Field(
        default=1000.0,
        description="Latency critical threshold in milliseconds"
    )
    
    # Notification settings
    notification_enabled: bool = Field(
        default=True,
        description="Enable alert notifications"
    )
    notification_rate_limit: int = Field(
        default=10,
        description="Maximum notifications per minute"
    )
    
    @validator("cpu_warning_threshold", "cpu_critical_threshold", 
              "memory_warning_threshold", "memory_critical_threshold",
              "disk_warning_threshold", "disk_critical_threshold")
    def validate_percentage_thresholds(cls, v: float) -> float:
        """Validate percentage thresholds."""
        if not (0.0 <= v <= 100.0):
            raise ValueError("Percentage thresholds must be between 0 and 100")
        return v
    
    @validator("latency_warning_threshold", "latency_critical_threshold")
    def validate_latency_thresholds(cls, v: float) -> float:
        """Validate latency thresholds."""
        if v < 0.0:
            raise ValueError("Latency thresholds must be non-negative")
        return v


class FeatureFlagConfig(BaseModel):
    """Feature flag configuration."""
    
    enabled: bool = Field(default=True, description="Enable feature flag service")
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/trading_system",
        description="Feature flag database URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for feature flag caching"
    )
    config_file_path: Optional[str] = Field(
        default="./config/feature_flags.json",
        description="Path to feature flag configuration file"
    )
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging for feature flag changes"
    )
    
    # A/B testing configuration
    ab_testing_enabled: bool = Field(
        default=True,
        description="Enable A/B testing capabilities"
    )
    default_traffic_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default traffic split for A/B tests"
    )
    
    # Rollout configuration
    gradual_rollout_enabled: bool = Field(
        default=True,
        description="Enable gradual rollout capabilities"
    )
    default_rollout_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Default rollout percentage for new flags"
    )
    
    # Approval workflow
    approval_workflow_enabled: bool = Field(
        default=False,
        description="Enable approval workflow for flag changes"
    )
    require_approval_for_production: bool = Field(
        default=True,
        description="Require approval for production flag changes"
    )


class DependencyManagementConfig(BaseModel):
    """Dependency management configuration."""
    
    enabled: bool = Field(default=True, description="Enable dependency management")
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/trading_system",
        description="Dependency management database URL"
    )
    
    # Environment configuration
    environments: List[str] = Field(
        default=["development", "testing", "staging", "production"],
        description="List of environments to manage"
    )
    
    # Monitoring configuration
    check_interval: int = Field(
        default=3600,
        description="Health check interval in seconds"
    )
    vulnerability_scan_interval: int = Field(
        default=7200,
        description="Vulnerability scan interval in seconds"
    )
    
    # Vulnerability sources
    vulnerability_sources: List[str] = Field(
        default=[
            "https://pyup.io/api/v1/safety/",
            "https://rustsec.org/advisories/",
            "https://registry.npmjs.org/-/npm/v1/security/audits/"
        ],
        description="List of vulnerability data sources"
    )
    
    # Alert configuration
    alert_on_critical: bool = Field(
        default=True,
        description="Send alerts for critical dependency issues"
    )
    alert_on_vulnerabilities: bool = Field(
        default=True,
        description="Send alerts for security vulnerabilities"
    )
    alert_on_compatibility_issues: bool = Field(
        default=True,
        description="Send alerts for compatibility issues"
    )
    
    # Rollback configuration
    auto_snapshot_enabled: bool = Field(
        default=True,
        description="Enable automatic dependency snapshots"
    )
    snapshot_retention_days: int = Field(
        default=30,
        description="Number of days to retain snapshots"
    )
    rollback_validation_enabled: bool = Field(
        default=True,
        description="Enable validation before rollback operations"
    )
    
    # Compatibility testing
    compatibility_testing_enabled: bool = Field(
        default=True,
        description="Enable automated compatibility testing"
    )
    compatibility_cache_ttl: int = Field(
        default=86400,
        description="Compatibility test cache TTL in seconds"
    )


class NautilusConfig(BaseSettings):
    """
    Main configuration class for NautilusTrader integration.
    
    This class follows the patterns established in the knowledge-ingestion
    system for robust configuration management.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="NAUTILUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )
    
    # Environment
    environment: str = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    nautilus_engine: NautilusEngineConfig = Field(default_factory=NautilusEngineConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    signal_router: SignalRouterConfig = Field(default_factory=SignalRouterConfig)
    signal_validation: SignalValidationConfig = Field(default_factory=SignalValidationConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    feature_flags: FeatureFlagConfig = Field(default_factory=FeatureFlagConfig)
    dependency_management: DependencyManagementConfig = Field(default_factory=DependencyManagementConfig)
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8002, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Health check configuration
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health check endpoint"
    )
    health_check_interval: float = Field(
        default=30.0,
        description="Health check interval in seconds"
    )
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment values."""
        valid_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v.lower()
    
    @classmethod
    def from_env_file(cls, env_file: Union[str, Path]) -> "NautilusConfig":
        """Load configuration from environment file."""
        return cls(_env_file=env_file)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of validation errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate data catalog path
        catalog_path = Path(self.nautilus_engine.data_catalog_path)
        if not catalog_path.parent.exists():
            errors.append(f"Data catalog parent directory does not exist: {catalog_path.parent}")
        
        # Validate database URLs
        if not self.database.postgres_url.startswith(("postgresql://", "postgres://")):
            errors.append("Invalid PostgreSQL URL format")
        
        if not self.database.neo4j_url.startswith("bolt://"):
            errors.append("Invalid Neo4j URL format")
        
        if not self.database.redis_url.startswith("redis://"):
            errors.append("Invalid Redis URL format")
        
        # Validate port ranges
        if not (1024 <= self.api_port <= 65535):
            errors.append("API port must be between 1024 and 65535")
        
        # Validate worker count
        if self.api_workers < 1:
            errors.append("API workers must be at least 1")
        
        return errors
    
    def create_data_directories(self) -> None:
        """Create necessary data directories."""
        catalog_path = Path(self.nautilus_engine.data_catalog_path)
        catalog_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (catalog_path / "bars").mkdir(exist_ok=True)
        (catalog_path / "ticks").mkdir(exist_ok=True)
        (catalog_path / "order_book").mkdir(exist_ok=True)
        (catalog_path / "instruments").mkdir(exist_ok=True)
        
        # Create logs directory if log file is configured
        if self.logging.log_file_path:
            log_path = Path(self.logging.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)


def load_config(
    env_file: Optional[Union[str, Path]] = None,
    validate: bool = True
) -> NautilusConfig:
    """
    Load and validate NautilusTrader integration configuration.
    
    Args:
        env_file: Optional path to environment file
        validate: Whether to validate configuration
        
    Returns:
        Loaded and validated configuration
        
    Raises:
        ValueError: If configuration validation fails
    """
    if env_file:
        config = NautilusConfig.from_env_file(env_file)
    else:
        config = NautilusConfig()
    
    if validate:
        errors = config.validate_configuration()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    # Create necessary directories
    config.create_data_directories()
    
    return config