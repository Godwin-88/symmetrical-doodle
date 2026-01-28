"""
Tests for Data Catalog Adapter (DCA) implementation.

This module tests the core functionality of the DataCatalogAdapter including
data migration, validation, optimization, and performance monitoring.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import pyarrow as pa
import tempfile
import shutil

from nautilus_integration.services.data_catalog_adapter import (
    DataCatalogAdapter,
    DataMigrationConfig,
    DataMigrationType,
    DataQualityStatus,
    NautilusDataType,
    CompressionStrategy,
    OptimizationConfig,
    NautilusSchema,
    PerformanceMetrics
)


class TestDataCatalogAdapter:
    """Test suite for DataCatalogAdapter."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def adapter_config(self, temp_dir):
        """Create test configuration."""
        return DataMigrationConfig(
            source_schema="test_schema",
            target_path=str(temp_dir / "catalog"),
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
        assert adapter.optimization_config is not None
        assert len(adapter._schema_cache) == 4  # Four data types
    
    def test_nautilus_schemas(self):
        """Test Nautilus schema definitions."""
        # Test bar schema
        bar_schema = NautilusSchema.get_bar_schema()
        assert isinstance(bar_schema, pa.Schema)
        assert 'instrument_id' in bar_schema.names
        assert 'ts_event' in bar_schema.names
        assert 'open' in bar_schema.names
        
        # Test tick schema
        tick_schema = NautilusSchema.get_tick_schema()
        assert isinstance(tick_schema, pa.Schema)
        assert 'price' in tick_schema.names
        assert 'size' in tick_schema.names
        
        # Test quote tick schema
        quote_schema = NautilusSchema.get_quote_tick_schema()
        assert isinstance(quote_schema, pa.Schema)
        assert 'bid_price' in quote_schema.names
        assert 'ask_price' in quote_schema.names
        
        # Test order book delta schema
        ob_schema = NautilusSchema.get_order_book_delta_schema()
        assert isinstance(ob_schema, pa.Schema)
        assert 'action' in ob_schema.names
        assert 'sequence' in ob_schema.names
    
    def test_optimization_config(self):
        """Test optimization configuration."""
        config = OptimizationConfig()
        assert config.compression_strategy == CompressionStrategy.BALANCED
        assert config.row_group_size == 100000
        assert config.use_dictionary_encoding is True
        assert config.partition_strategy == ["date_partition", "venue"]
    
    @pytest.mark.asyncio
    async def test_adapter_lifecycle(self, adapter):
        """Test adapter initialization and shutdown."""
        # Mock database pool and catalog
        with patch('asyncpg.create_pool') as mock_pool, \
             patch('nautilus_trader.persistence.catalog.ParquetDataCatalog') as mock_catalog:
            
            mock_pool.return_value = AsyncMock()
            mock_catalog.return_value = Mock()
            
            # Test initialization
            await adapter.initialize()
            assert adapter.db_pool is not None
            assert adapter.catalog is not None
            
            # Test shutdown
            await adapter.shutdown()
            mock_pool.return_value.close.assert_called_once()
    
    def test_performance_metrics(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            operation_type="test_operation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
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
    
    @pytest.mark.asyncio
    async def test_migration_progress_tracking(self, adapter):
        """Test migration progress tracking."""
        # Mock database operations
        with patch.object(adapter, 'db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            # Mock query results
            mock_conn.fetch.return_value = [{'symbol': 'EURUSD'}]
            mock_conn.fetchval.return_value = 100
            
            # Mock batch data
            mock_conn.fetch.side_effect = [
                [{'symbol': 'EURUSD'}],  # instruments query
                []  # batch data (empty to avoid processing)
            ]
            
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            # Test migration
            progress = await adapter.migrate_market_data(
                start_date=start_date,
                end_date=end_date,
                migration_type=DataMigrationType.FULL_MIGRATION
            )
            
            assert progress.operation_type == DataMigrationType.FULL_MIGRATION
            assert progress.current_status == "completed"
            assert progress.migration_id in adapter._migration_state
    
    @pytest.mark.asyncio
    async def test_data_validation(self, adapter):
        """Test data quality validation."""
        with patch.object(adapter, 'db_pool') as mock_pool, \
             patch.object(adapter, 'catalog') as mock_catalog:
            
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            # Mock validation query results
            mock_conn.fetch.return_value = [{'symbol': 'EURUSD'}]
            mock_conn.fetchrow.return_value = {
                'count': 1000,
                'null_count': 0,
                'invalid_ohlc': 0,
                'negative_volume': 0
            }
            
            # Mock catalog data
            mock_catalog.bars.return_value = [Mock() for _ in range(1000)]
            
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            result = await adapter.validate_data_quality(
                start_date=start_date,
                end_date=end_date
            )
            
            assert isinstance(result.status, DataQualityStatus)
            assert result.total_records == 1000
            assert result.quality_score >= 0.0
            assert isinstance(result.errors, list)
            assert isinstance(result.warnings, list)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, adapter):
        """Test performance monitoring functionality."""
        # Add some mock metrics
        metrics1 = PerformanceMetrics(
            operation_type="test_operation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=1.0,
            records_processed=1000,
            throughput_records_per_second=1000.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            disk_io_mb=20.0,
            compression_ratio=2.0,
            error_count=0,
            warnings_count=0
        )
        
        metrics2 = PerformanceMetrics(
            operation_type="test_operation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=2.0,
            records_processed=2000,
            throughput_records_per_second=1000.0,
            memory_usage_mb=150.0,
            cpu_usage_percent=60.0,
            disk_io_mb=40.0,
            compression_ratio=2.5,
            error_count=0,
            warnings_count=1
        )
        
        adapter._performance_metrics = [metrics1, metrics2]
        
        # Test monitoring
        stats = await adapter.monitor_performance("test_operation")
        
        assert stats["operation_name"] == "test_operation"
        assert stats["total_operations"] == 2
        assert stats["duration_stats"]["total"] == 3.0
        assert stats["throughput_stats"]["avg"] == 1000.0
        assert stats["total_records_processed"] == 3000
    
    @pytest.mark.asyncio
    async def test_auto_tuning(self, adapter):
        """Test automatic performance tuning."""
        # Add mock metrics with high memory usage
        high_memory_metrics = [
            PerformanceMetrics(
                operation_type="test_op",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                records_processed=1000,
                throughput_records_per_second=500.0,  # Low throughput
                memory_usage_mb=1500.0,  # High memory
                cpu_usage_percent=50.0,
                disk_io_mb=20.0,
                compression_ratio=1.5,  # Poor compression
                error_count=0,
                warnings_count=0
            ) for _ in range(10)
        ]
        
        adapter._performance_metrics = high_memory_metrics
        
        # Test auto-tuning
        tuned_config = await adapter.auto_tune_performance()
        
        # Should reduce row group size due to high memory usage
        assert tuned_config.row_group_size <= 50000
        # Should use faster compression due to low throughput and poor compression
        assert tuned_config.compression_strategy == CompressionStrategy.ULTRA_FAST
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, adapter, temp_dir):
        """Test performance report generation."""
        # Add mock metrics
        adapter._performance_metrics = [
            PerformanceMetrics(
                operation_type="migration",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                records_processed=1000,
                throughput_records_per_second=1000.0,
                memory_usage_mb=100.0,
                cpu_usage_percent=50.0,
                disk_io_mb=20.0,
                compression_ratio=2.0,
                error_count=0,
                warnings_count=0
            )
        ]
        
        # Test report generation
        report_path = temp_dir / "performance_report.json"
        report = await adapter.generate_performance_report(report_path)
        
        assert "report_timestamp" in report
        assert "total_operations" in report
        assert "performance_summary" in report
        assert "recommendations" in report
        assert report["total_operations"] == 1
        
        # Check if file was created
        assert report_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])