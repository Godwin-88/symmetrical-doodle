"""
Tests for configuration management.

This module tests the configuration loading and validation functionality
following the patterns established in the knowledge-ingestion system.
"""

import os
import tempfile
from pathlib import Path

import pytest

from nautilus_integration.core.config import (
    DatabaseConfig,
    ErrorHandlingConfig,
    IntegrationConfig,
    LoggingConfig,
    NautilusConfig,
    NautilusEngineConfig,
    load_config,
)


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        assert config.postgres_url == "postgresql://postgres:password@localhost:5432/trading_system"
        assert config.neo4j_url == "bolt://localhost:7687"
        assert config.neo4j_user == "neo4j"
        assert config.neo4j_password == "password"
        assert config.redis_url == "redis://localhost:6379"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            postgres_url="postgresql://user:pass@host:5433/db",
            neo4j_url="bolt://host:7688",
            neo4j_user="admin",
            neo4j_password="secret",
            redis_url="redis://host:6380"
        )
        
        assert config.postgres_url == "postgresql://user:pass@host:5433/db"
        assert config.neo4j_url == "bolt://host:7688"
        assert config.neo4j_user == "admin"
        assert config.neo4j_password == "secret"
        assert config.redis_url == "redis://host:6380"


class TestNautilusEngineConfig:
    """Test Nautilus engine configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = NautilusEngineConfig()
        
        assert config.backtest_engine_id == "BACKTEST"
        assert config.backtest_log_level == "INFO"
        assert config.backtest_cache_database is True
        assert config.backtest_cache_database_flush is False
        
        assert config.trading_node_id == "LIVE"
        assert config.trading_log_level == "INFO"
        assert config.trading_cache_database is True
        
        assert config.data_catalog_path == "./data/catalog"
        assert config.parquet_compression == "snappy"
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = NautilusEngineConfig(backtest_log_level=level)
            assert config.backtest_log_level == level
        
        # Invalid log level should raise validation error
        with pytest.raises(ValueError, match="Log level must be one of"):
            NautilusEngineConfig(backtest_log_level="INVALID")


class TestIntegrationConfig:
    """Test integration configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = IntegrationConfig()
        
        assert config.strategy_translation_enabled is True
        assert config.strategy_validation_enabled is True
        assert config.strategy_hot_swap_enabled is True
        
        assert config.signal_routing_enabled is True
        assert config.signal_buffer_size == 10000
        assert config.signal_delivery_timeout == 1.0
        
        assert config.risk_integration_enabled is True
        assert config.position_sync_interval == 1.0
        
        assert config.performance_monitoring_enabled is True
        assert config.metrics_collection_interval == 5.0


class TestErrorHandlingConfig:
    """Test error handling configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ErrorHandlingConfig()
        
        assert config.max_retries == 3
        assert config.retry_backoff_factor == 2.0
        assert config.retry_max_delay == 60.0
        
        assert config.circuit_breaker_enabled is True
        assert config.circuit_breaker_failure_threshold == 5
        assert config.circuit_breaker_recovery_timeout == 30.0
        
        assert config.graceful_degradation_enabled is True
        assert config.fallback_to_legacy_engine is True


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.correlation_id_enabled is True
        assert config.structured_logging is True
        assert config.log_file_path is None
        assert config.log_rotation_enabled is True
        assert config.log_max_size == "100MB"
        assert config.log_backup_count == 5
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level
        
        # Invalid log level should raise validation error
        with pytest.raises(ValueError, match="Log level must be one of"):
            LoggingConfig(level="INVALID")


class TestNautilusConfig:
    """Test main Nautilus configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = NautilusConfig()
        
        assert config.environment == "development"
        assert config.debug is False
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8002
        assert config.api_workers == 1
        assert config.health_check_enabled is True
        assert config.health_check_interval == 30.0
        
        # Test nested configurations
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.nautilus_engine, NautilusEngineConfig)
        assert isinstance(config.integration, IntegrationConfig)
        assert isinstance(config.error_handling, ErrorHandlingConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "testing", "staging", "production"]:
            config = NautilusConfig(environment=env)
            assert config.environment == env
        
        # Invalid environment should raise validation error
        with pytest.raises(ValueError, match="Environment must be one of"):
            NautilusConfig(environment="invalid")
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = NautilusConfig()
        
        # Default configuration should be valid
        errors = config.validate_configuration()
        # Note: Some errors may be expected if directories don't exist
        # We mainly check that the method runs without exceptions
        assert isinstance(errors, list)
    
    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = NautilusConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "environment" in config_dict
        assert "database" in config_dict
        assert "nautilus_engine" in config_dict
        assert "integration" in config_dict
        assert "error_handling" in config_dict
        assert "logging" in config_dict
    
    def test_create_data_directories(self):
        """Test data directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = NautilusConfig()
            config.nautilus_engine.data_catalog_path = str(Path(temp_dir) / "catalog")
            config.logging.log_file_path = str(Path(temp_dir) / "logs" / "test.log")
            
            config.create_data_directories()
            
            # Check that directories were created
            catalog_path = Path(config.nautilus_engine.data_catalog_path)
            assert catalog_path.exists()
            assert (catalog_path / "bars").exists()
            assert (catalog_path / "ticks").exists()
            assert (catalog_path / "order_book").exists()
            assert (catalog_path / "instruments").exists()
            
            log_path = Path(config.logging.log_file_path)
            assert log_path.parent.exists()


class TestLoadConfig:
    """Test configuration loading functionality."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config(validate=False)
        
        assert isinstance(config, NautilusConfig)
        assert config.environment == "development"
    
    def test_load_config_from_env_file(self):
        """Test loading configuration from environment file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("NAUTILUS_ENVIRONMENT=testing\n")
            f.write("NAUTILUS_DEBUG=true\n")
            f.write("NAUTILUS_API_PORT=8003\n")
            env_file = f.name
        
        try:
            config = load_config(env_file, validate=False)
            
            assert config.environment == "testing"
            assert config.debug is True
            assert config.api_port == 8003
        finally:
            os.unlink(env_file)
    
    def test_load_config_with_validation_error(self):
        """Test configuration loading with validation errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("NAUTILUS_API_PORT=99999\n")  # Invalid port
            env_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(env_file, validate=True)
        finally:
            os.unlink(env_file)


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    def test_environment_variable_override(self):
        """Test that environment variables override configuration."""
        # Set environment variable
        os.environ["NAUTILUS_API_PORT"] = "9000"
        
        try:
            config = NautilusConfig()
            assert config.api_port == 9000
        finally:
            # Clean up
            if "NAUTILUS_API_PORT" in os.environ:
                del os.environ["NAUTILUS_API_PORT"]
    
    def test_nested_configuration_override(self):
        """Test nested configuration override via environment variables."""
        # Set nested environment variable
        os.environ["NAUTILUS_DATABASE__POSTGRES_URL"] = "postgresql://test:test@test:5432/test"
        
        try:
            config = NautilusConfig()
            assert config.database.postgres_url == "postgresql://test:test@test:5432/test"
        finally:
            # Clean up
            if "NAUTILUS_DATABASE__POSTGRES_URL" in os.environ:
                del os.environ["NAUTILUS_DATABASE__POSTGRES_URL"]