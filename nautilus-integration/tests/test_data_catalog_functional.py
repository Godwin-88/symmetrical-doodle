"""
Functional tests for Data Catalog Adapter implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nautilus_integration.services.data_catalog_adapter import (
    DataCatalogAdapter,
    DataMigrationConfig,
    DataMigrationType,
    DataQualityStatus
)


class TestDataCatalogAdapterFunctional:
    """Functional test suite for DataCatalogAdapter."""
    
    @pytest.fixture
    def adapter_config(self):
        """Create test configuration."""
        return DataMigrationConfig(
            source_schema="test_schema",
            target_path="./test_catalog",
            batch_size=100,
            validation_enabled=True
        )
    
    @pytest.fixture
    def adapter(self, adapter_config):
        """Create DataCatalogAdapter instance."""
        return DataCatalogAdapter(adapter_config)
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.config.source_schema == "test_schema"
        assert adapter.config.batch_size == 100
        assert adapter.config.target_path == "./test_catalog"
        assert isinstance(adapter._migration_state, dict)
    
    def test_enums(self):
        """Test enum definitions."""
        # Test DataMigrationType
        assert DataMigrationType.FULL_MIGRATION.value == "full_migration"
        assert DataMigrationType.INCREMENTAL_UPDATE.value == "incremental_update"
        assert DataMigrationType.VALIDATION_ONLY.value == "validation_only"
        assert DataMigrationType.ROLLBACK.value == "rollback"
        
        # Test DataQualityStatus
        assert DataQualityStatus.VALID.value == "valid"
        assert DataQualityStatus.WARNING.value == "warning"
        assert DataQualityStatus.INVALID.value == "invalid"
        assert DataQualityStatus.CORRUPTED.value == "corrupted"
    
    @pytest.mark.asyncio
    async def test_migrate_market_data(self, adapter):
        """Test market data migration."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        progress = await adapter.migrate_market_data(
            start_date=start_date,
            end_date=end_date,
            migration_type=DataMigrationType.FULL_MIGRATION
        )
        
        assert progress.operation_type == DataMigrationType.FULL_MIGRATION
        assert progress.current_status == "completed"
        assert progress.migration_id in adapter._migration_state
        assert progress.total_batches == 1
        assert progress.completed_batches == 1
        assert progress.failed_batches == 0
    
    @pytest.mark.asyncio
    async def test_validate_data_quality(self, adapter):
        """Test data quality validation."""
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        result = await adapter.validate_data_quality(
            start_date=start_date,
            end_date=end_date,
            instruments=["EURUSD", "GBPUSD"]
        )
        
        assert result.status == DataQualityStatus.VALID
        assert result.total_records == 1000
        assert result.valid_records == 950
        assert result.invalid_records == 50
        assert result.quality_score == 0.95
        assert not result.corruption_detected
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)
        assert "instruments_validated" in result.metadata
    
    @pytest.mark.asyncio
    async def test_optimize_parquet_storage(self, adapter):
        """Test Parquet storage optimization."""
        result = await adapter.optimize_parquet_storage(
            instruments=["EURUSD", "GBPUSD"],
            compression_type="snappy"
        )
        
        assert isinstance(result, dict)
        assert "optimized_instruments" in result
        assert "compression_savings" in result
        assert "performance_improvements" in result
        assert "errors" in result
        assert result["optimized_instruments"] == ["EURUSD", "GBPUSD"]
    
    @pytest.mark.asyncio
    async def test_handle_incremental_updates(self, adapter):
        """Test incremental data updates."""
        new_data_start = datetime.utcnow() - timedelta(hours=1)
        
        result = await adapter.handle_incremental_updates(
            new_data_start=new_data_start,
            conflict_resolution="latest_wins"
        )
        
        assert isinstance(result, dict)
        assert "processed_records" in result
        assert "conflicts_resolved" in result
        assert "errors" in result
        assert "instruments_updated" in result
        assert "update_strategy" in result
        assert result["update_strategy"] == "latest_wins"
    
    @pytest.mark.asyncio
    async def test_migration_status_tracking(self, adapter):
        """Test migration status tracking."""
        # First create a migration
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        progress = await adapter.migrate_market_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Test getting migration status
        status = await adapter.get_migration_status(progress.migration_id)
        assert status is not None
        assert status.migration_id == progress.migration_id
        assert status.current_status == "completed"
        
        # Test listing migrations
        migrations = await adapter.list_migrations()
        assert len(migrations) >= 1
        assert any(m.migration_id == progress.migration_id for m in migrations)
    
    @pytest.mark.asyncio
    async def test_cleanup_failed_migrations(self, adapter):
        """Test cleanup of failed migrations."""
        # Create a mock failed migration
        from nautilus_integration.services.data_catalog_adapter import MigrationProgress
        
        failed_migration = MigrationProgress(
            migration_id="failed_test",
            operation_type=DataMigrationType.FULL_MIGRATION,
            total_batches=1,
            completed_batches=0,
            failed_batches=1,
            start_time=datetime.utcnow(),
            estimated_completion=None,
            current_status="failed",
            error_details=["Test error"],
            performance_metrics={}
        )
        
        adapter._migration_state["failed_test"] = failed_migration
        
        # Test cleanup
        cleaned_count = await adapter.cleanup_failed_migrations()
        assert cleaned_count == 1
        assert "failed_test" not in adapter._migration_state
    
    def test_data_migration_config_defaults(self):
        """Test DataMigrationConfig default values."""
        config = DataMigrationConfig(
            source_schema="test",
            target_path="./test"
        )
        
        assert config.batch_size == 10000
        assert config.validation_enabled is True
        assert config.compression == "snappy"
        assert config.quality_threshold == 0.95
        assert config.max_retries == 3
        assert config.timeout_seconds == 300
        assert config.partition_cols == ["asset_id", "date"]


if __name__ == "__main__":
    pytest.main([__file__])