"""
Test infrastructure components for Google Drive Knowledge Base Ingestion.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.config import ConfigManager, KnowledgeIngestionSettings, GoogleDriveConfig
from core.logging import LoggingManager, get_logger, set_correlation_id, get_correlation_id


class TestConfigManager:
    """Test configuration management system"""
    
    def test_default_config_creation(self):
        """Test that default configuration can be created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigManager(config_dir)
            
            # Create default config files
            manager.create_default_config_files()
            
            # Verify files were created
            assert (config_dir / ".env.example").exists()
            assert (config_dir / "config.yaml").exists()
    
    def test_config_loading(self):
        """Test configuration loading from environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigManager(config_dir)
            
            # Create a test .env file
            env_file = config_dir / ".env"
            env_content = """
ENVIRONMENT=development
DEBUG=true
SUPABASE_URL=https://test.supabase.co
SUPABASE_KEY=test_key
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            # Load configuration
            settings = manager.load_config("development")
            
            assert settings.environment == "development"
            assert settings.debug is True
            assert settings.supabase.url == "https://test.supabase.co"
            assert settings.supabase.key == "test_key"
    
    def test_config_validation(self):
        """Test configuration validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigManager(config_dir)
            
            # Test with missing required fields
            validation_results = manager.validate_config()
            assert not validation_results["valid"]
            assert len(validation_results["errors"]) > 0
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigManager(config_dir)
            
            # Create base config
            base_config = config_dir / "config.yaml"
            with open(base_config, 'w') as f:
                f.write("""
logging:
  level: INFO
  format: json
""")
            
            # Create development-specific config
            dev_config = config_dir / "config.development.yaml"
            with open(dev_config, 'w') as f:
                f.write("""
logging:
  level: DEBUG
  format: console
""")
            
            # Load development configuration
            settings = manager.load_config("development")
            
            assert settings.logging.level == "DEBUG"
            assert settings.logging.format == "console"


class TestLoggingManager:
    """Test structured logging system"""
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        manager = LoggingManager()
        manager.configure_logging()
        
        # Get a logger
        logger = manager.get_logger("test", component="test_component")
        assert logger is not None
    
    def test_correlation_id_management(self):
        """Test correlation ID management"""
        manager = LoggingManager()
        
        # Test setting correlation ID
        correlation_id = manager.set_correlation_id("test-123")
        assert correlation_id == "test-123"
        assert manager.get_correlation_id() == "test-123"
        
        # Test auto-generation
        auto_id = manager.set_correlation_id()
        assert auto_id is not None
        assert len(auto_id) > 0
        
        # Test clearing
        manager.clear_correlation_id()
        assert manager.get_correlation_id() is None
    
    def test_log_context_creation(self):
        """Test log context creation"""
        manager = LoggingManager()
        
        context = manager.create_log_context(
            component="test",
            operation="test_operation",
            file_id="test_file_123"
        )
        
        assert context.component == "test"
        assert context.operation == "test_operation"
        assert context.file_id == "test_file_123"
        assert context.correlation_id is not None
    
    def test_global_logging_functions(self):
        """Test global logging functions"""
        # Test correlation ID functions
        correlation_id = set_correlation_id("global-test-123")
        assert correlation_id == "global-test-123"
        assert get_correlation_id() == "global-test-123"
        
        # Test logger creation
        logger = get_logger("global_test", component="global_component")
        assert logger is not None


class TestDataModels:
    """Test data model validation"""
    
    def test_google_drive_config_creation(self):
        """Test GoogleDriveConfig creation"""
        config = GoogleDriveConfig(
            credentials_path="./test/credentials.json",
            folder_ids=["folder1", "folder2"]
        )
        
        assert config.credentials_path == "./test/credentials.json"
        assert config.folder_ids == ["folder1", "folder2"]
        assert "https://www.googleapis.com/auth/drive.readonly" in config.scopes
    
    def test_settings_validation(self):
        """Test KnowledgeIngestionSettings validation"""
        # Test valid environment
        settings = KnowledgeIngestionSettings(environment="development")
        assert settings.environment == "development"
        
        # Test invalid environment should raise validation error
        with pytest.raises(ValueError):
            KnowledgeIngestionSettings(environment="invalid")


class TestIntegration:
    """Test integration between components"""
    
    def test_config_and_logging_integration(self):
        """Test that configuration and logging work together"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create configuration
            manager = ConfigManager(config_dir)
            settings = manager.load_config("development")
            
            # Configure logging with settings
            logging_manager = LoggingManager(settings.logging)
            logging_manager.configure_logging()
            
            # Get logger and test logging
            logger = logging_manager.get_logger("integration_test")
            
            # Set correlation ID and log message
            correlation_id = logging_manager.set_correlation_id()
            logger.info("Integration test message", test_data="success")
            
            assert logging_manager.get_correlation_id() == correlation_id
    
    def test_environment_setup_simulation(self):
        """Test simulated environment setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create basic project structure
            (project_root / "requirements.txt").write_text("pytest==7.4.3\n")
            
            # Test that we can create the basic structure
            config_dir = project_root / "config"
            config_dir.mkdir()
            
            manager = ConfigManager(config_dir)
            manager.create_default_config_files()
            
            # Verify structure
            assert (config_dir / ".env.example").exists()
            assert (config_dir / "config.yaml").exists()


if __name__ == "__main__":
    pytest.main([__file__])