"""
Basic tests for Data Catalog Adapter without NautilusTrader dependencies.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Test basic imports without NautilusTrader
try:
    from nautilus_integration.services.data_catalog_adapter import (
        DataMigrationType,
        DataQualityStatus,
        NautilusDataType,
        CompressionStrategy,
        OptimizationConfig,
        DataMigrationConfig,
        DataValidationResult,
        MigrationProgress,
        PerformanceMetrics
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestBasicDataStructures:
    """Test basic data structures without NautilusTrader dependencies."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_enums(self):
        """Test enum definitions."""
        # Test DataMigrationType
        assert DataMigrationType.FULL_MIGRATION.value == "full_migration"
        assert DataMigrationType.INCREMENTAL_UPDATE.value == "incremental_update"
        
        # Test DataQualityStatus
        assert DataQualityStatus.VALID.value == "valid"
        assert DataQualityStatus.INVALID.value == "invalid"
        
        # Test NautilusDataType
        assert NautilusDataType.BARS.value == "bars"
        assert NautilusDataType.TICKS.value == "ticks"
        
        # Test CompressionStrategy
        assert CompressionStrategy.ULTRA_FAST.value == "lz4"
        assert CompressionStrategy.BALANCED.value == "snappy"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_data_migration_config(self):
        """Test DataMigrationConfig."""
        config = DataMigrationConfig(
            source_schema="test_schema",
            target_path="./test_path",
            batch_size=5000
        )
        
        assert config.source_schema == "test_schema"
        assert config.target_path == "./test_path"
        assert config.batch_size == 5000
        assert config.validation_enabled is True
        assert config.partition_cols == ["asset_id", "date"]  # Default value
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_optimization_config(self):
        """Test OptimizationConfig."""
        config = OptimizationConfig()
        
        assert config.compression_strategy == CompressionStrategy.BALANCED
        assert config.row_group_size == 100000
        assert config.use_dictionary_encoding is True
        assert config.partition_strategy == ["date_partition", "venue"]
        assert config.sort_columns == ["ts_event"]
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_performance_metrics(self):
        """Test PerformanceMetrics."""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()
        
        metrics = PerformanceMetrics(
            operation_type="test_operation",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=1.5,
            records_processed=1000,
            throughput_records_per_second=666.67,
            memory_usage_mb=50.0,
            cpu_usage_percent=25.0,
            disk_io_mb=10.0,
            compression_ratio=3.2,
            error_count=0,
            warnings_count=1
        )
        
        assert metrics.operation_type == "test_operation"
        assert metrics.records_processed == 1000
        assert metrics.compression_ratio == 3.2
        
        # Test conversion to dict
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['operation_type'] == "test_operation"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_data_validation_result(self):
        """Test DataValidationResult."""
        result = DataValidationResult(
            status=DataQualityStatus.VALID,
            total_records=1000,
            valid_records=950,
            invalid_records=50,
            corruption_detected=False,
            quality_score=0.95,
            errors=[],
            warnings=["Minor issue detected"],
            metadata={"test": "value"},
            timestamp=datetime.utcnow()
        )
        
        assert result.status == DataQualityStatus.VALID
        assert result.total_records == 1000
        assert result.quality_score == 0.95
        assert len(result.warnings) == 1
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_migration_progress(self):
        """Test MigrationProgress."""
        progress = MigrationProgress(
            migration_id="test_migration_123",
            operation_type=DataMigrationType.FULL_MIGRATION,
            total_batches=10,
            completed_batches=5,
            failed_batches=0,
            start_time=datetime.utcnow(),
            estimated_completion=None,
            current_status="in_progress",
            error_details=[],
            performance_metrics={"throughput": 1000.0}
        )
        
        assert progress.migration_id == "test_migration_123"
        assert progress.operation_type == DataMigrationType.FULL_MIGRATION
        assert progress.total_batches == 10
        assert progress.completed_batches == 5
        assert progress.current_status == "in_progress"


if __name__ == "__main__":
    pytest.main([__file__])