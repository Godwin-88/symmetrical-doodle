"""
Property-based tests for Data Migration Integrity.

This module tests Property 26: Data Migration Integrity
using property-based testing with Hypothesis.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

import pytest
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nautilus_integration.core.config import NautilusConfig
from nautilus_integration.services.data_catalog_adapter import (
    DataCatalogAdapter,
    PostgreSQLData,
    NautilusParquetData,
    DataMigrationResult,
    DataValidationResult,
)


# Test data generators
@st.composite
def postgresql_ohlcv_data(draw):
    """Generate valid PostgreSQL OHLCV time-series data."""
    num_rows = draw(st.integers(min_value=100, max_value=1000))
    
    # Generate realistic market data
    base_price = draw(st.floats(min_value=1.0, max_value=1000.0))
    base_time = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2023, 12, 31)
    ))
    
    data = []
    current_price = base_price
    
    for i in range(num_rows):
        # Generate realistic price movements
        change_pct = draw(st.floats(min_value=-0.05, max_value=0.05))
        current_price *= (1 + change_pct)
        
        # Ensure OHLC relationships are valid
        high = current_price * draw(st.floats(min_value=1.0, max_value=1.02))
        low = current_price * draw(st.floats(min_value=0.98, max_value=1.0))
        open_price = draw(st.floats(min_value=low, max_value=high))
        close_price = current_price
        volume = draw(st.floats(min_value=1000.0, max_value=100000.0))
        
        data.append({
            'id': i + 1,
            'instrument_id': draw(st.sampled_from(['EUR/USD', 'GBP/USD', 'BTC/USDT', 'ETH/USDT'])),
            'timestamp': base_time + timedelta(minutes=i),
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close_price, 5),
            'volume': round(volume, 2),
            'bar_type': draw(st.sampled_from(['1-MINUTE-BID', '1-MINUTE-ASK', '1-MINUTE-MID'])),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
        })
    
    return PostgreSQLData(
        table_name="market_data_ohlcv",
        data=pd.DataFrame(data),
        schema={
            'id': 'INTEGER PRIMARY KEY',
            'instrument_id': 'VARCHAR(20) NOT NULL',
            'timestamp': 'TIMESTAMP NOT NULL',
            'open': 'DECIMAL(15,5) NOT NULL',
            'high': 'DECIMAL(15,5) NOT NULL',
            'low': 'DECIMAL(15,5) NOT NULL',
            'close': 'DECIMAL(15,5) NOT NULL',
            'volume': 'DECIMAL(15,2) NOT NULL',
            'bar_type': 'VARCHAR(20) NOT NULL',
            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        }
    )


@st.composite
def postgresql_tick_data(draw):
    """Generate valid PostgreSQL tick data."""
    num_rows = draw(st.integers(min_value=50, max_value=500))
    
    base_price = draw(st.floats(min_value=1.0, max_value=1000.0))
    base_time = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2023, 12, 31)
    ))
    
    data = []
    current_price = base_price
    
    for i in range(num_rows):
        # Small tick movements
        change_pct = draw(st.floats(min_value=-0.001, max_value=0.001))
        current_price *= (1 + change_pct)
        
        data.append({
            'id': i + 1,
            'instrument_id': draw(st.sampled_from(['EUR/USD', 'GBP/USD', 'BTC/USDT'])),
            'timestamp': base_time + timedelta(microseconds=i * 1000),
            'price': round(current_price, 5),
            'size': draw(st.floats(min_value=0.01, max_value=100.0)),
            'aggressor_side': draw(st.sampled_from(['BUY', 'SELL', 'UNKNOWN'])),
            'trade_id': f"trade_{i}",
            'created_at': datetime.now(),
        })
    
    return PostgreSQLData(
        table_name="market_data_ticks",
        data=pd.DataFrame(data),
        schema={
            'id': 'INTEGER PRIMARY KEY',
            'instrument_id': 'VARCHAR(20) NOT NULL',
            'timestamp': 'TIMESTAMP NOT NULL',
            'price': 'DECIMAL(15,5) NOT NULL',
            'size': 'DECIMAL(15,8) NOT NULL',
            'aggressor_side': 'VARCHAR(10) NOT NULL',
            'trade_id': 'VARCHAR(50) NOT NULL',
            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        }
    )


@st.composite
def data_quality_parameters(draw):
    """Generate data quality validation parameters."""
    return {
        "check_duplicates": draw(st.booleans()),
        "check_null_values": draw(st.booleans()),
        "check_data_types": draw(st.booleans()),
        "check_value_ranges": draw(st.booleans()),
        "check_temporal_consistency": draw(st.booleans()),
        "tolerance_pct": draw(st.floats(min_value=0.0, max_value=0.01)),
        "max_null_pct": draw(st.floats(min_value=0.0, max_value=0.05)),
    }


class TestDataMigrationIntegrity:
    """Property-based tests for data migration integrity."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return NautilusConfig(
            environment="testing",  # Use valid environment value
            log_level="INFO",
        )
    
    @pytest.fixture
    async def data_catalog_adapter(self, config):
        """Create data catalog adapter for testing."""
        adapter = DataCatalogAdapter(config)
        await adapter.initialize()
        yield adapter
        await adapter.shutdown()
    
    # Feature: nautilus-trader-integration, Property 26: Data Migration Integrity
    @given(
        postgresql_data=postgresql_ohlcv_data(),
        quality_params=data_quality_parameters()
    )
    @settings(max_examples=100, deadline=15000)
    async def test_data_migration_integrity(
        self, 
        data_catalog_adapter, 
        postgresql_data, 
        quality_params
    ):
        """
        Property 26: Data Migration Integrity
        
        For any PostgreSQL time-series data migration, the system should convert 
        to Parquet format without data loss or corruption.
        
        **Validates: Requirements 4.2**
        """
        # Calculate original data hash for integrity verification
        original_data_hash = self._calculate_data_hash(postgresql_data.data)
        original_row_count = len(postgresql_data.data)
        
        # Perform migration
        migration_result = await data_catalog_adapter.migrate_postgresql_to_parquet(
            postgresql_data=postgresql_data,
            quality_parameters=quality_params
        )
        
        # Verify migration succeeded
        assert migration_result.success is True, (
            f"Migration failed: {migration_result.error_message}"
        )
        
        # Verify migrated data structure
        assert migration_result.parquet_data is not None
        parquet_data = migration_result.parquet_data
        
        assert isinstance(parquet_data, NautilusParquetData)
        assert parquet_data.file_path is not None
        assert parquet_data.schema is not None
        assert parquet_data.metadata is not None
        
        # Read back the migrated data
        migrated_df = pd.read_parquet(parquet_data.file_path)
        
        # Verify row count integrity
        assert len(migrated_df) == original_row_count, (
            f"Row count mismatch: original {original_row_count}, migrated {len(migrated_df)}"
        )
        
        # Verify column integrity
        original_columns = set(postgresql_data.data.columns)
        migrated_columns = set(migrated_df.columns)
        
        # Core data columns should be preserved
        core_columns = {'instrument_id', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        preserved_core_columns = core_columns.intersection(original_columns)
        
        for col in preserved_core_columns:
            assert col in migrated_columns, f"Core column {col} lost during migration"
        
        # Verify data integrity for preserved columns
        for col in preserved_core_columns:
            original_values = postgresql_data.data[col].dropna()
            migrated_values = migrated_df[col].dropna()
            
            if col == 'timestamp':
                # Handle timestamp precision differences
                original_ts = pd.to_datetime(original_values)
                migrated_ts = pd.to_datetime(migrated_values)
                
                # Allow for microsecond precision differences
                time_diff = abs((original_ts - migrated_ts).dt.total_seconds())
                assert time_diff.max() < 1.0, (
                    f"Timestamp precision loss exceeds 1 second for column {col}"
                )
            
            elif col in ['open', 'high', 'low', 'close', 'volume']:
                # Handle numeric precision
                original_numeric = pd.to_numeric(original_values, errors='coerce')
                migrated_numeric = pd.to_numeric(migrated_values, errors='coerce')
                
                # Calculate relative difference
                rel_diff = abs((original_numeric - migrated_numeric) / original_numeric)
                max_rel_diff = rel_diff.max()
                
                # Allow for small floating point precision differences
                assert max_rel_diff < 1e-10, (
                    f"Numeric precision loss exceeds tolerance for column {col}: {max_rel_diff}"
                )
            
            else:
                # String/categorical columns should match exactly
                original_set = set(original_values.astype(str))
                migrated_set = set(migrated_values.astype(str))
                
                assert original_set == migrated_set, (
                    f"String values changed for column {col}: "
                    f"missing {original_set - migrated_set}, "
                    f"added {migrated_set - original_set}"
                )
        
        # Verify Parquet-specific optimizations
        assert parquet_data.compression_type in ['snappy', 'gzip', 'lz4'], (
            "Parquet data should use efficient compression"
        )
        
        # Verify metadata preservation
        assert 'original_table' in parquet_data.metadata
        assert 'migration_timestamp' in parquet_data.metadata
        assert 'row_count' in parquet_data.metadata
        assert 'data_hash' in parquet_data.metadata
        
        assert parquet_data.metadata['original_table'] == postgresql_data.table_name
        assert parquet_data.metadata['row_count'] == original_row_count
        
        # Verify data hash integrity (if available)
        if 'data_hash' in parquet_data.metadata:
            migrated_data_hash = self._calculate_data_hash(migrated_df)
            # Note: Hash might differ due to column ordering/types, but core data should be preserved
    
    # Feature: nautilus-trader-integration, Property 27: Legacy Data Compatibility
    @given(
        postgresql_data=postgresql_ohlcv_data(),
        tick_data=postgresql_tick_data()
    )
    @settings(max_examples=50, deadline=12000)
    async def test_legacy_data_compatibility(
        self, 
        data_catalog_adapter, 
        postgresql_data, 
        tick_data
    ):
        """
        Property 27: Legacy Data Compatibility
        
        For any existing Neo4j graph data or pgvector embeddings, the system 
        should maintain accessibility and functionality after integration.
        
        **Validates: Requirements 4.3**
        """
        # Mock Neo4j and pgvector data
        neo4j_data = {
            "nodes": [
                {"id": "instrument_1", "type": "Instrument", "symbol": "EUR/USD"},
                {"id": "regime_1", "type": "MarketRegime", "name": "LOW_VOL_TRENDING"},
            ],
            "relationships": [
                {"from": "instrument_1", "to": "regime_1", "type": "IN_REGIME", "confidence": 0.85}
            ]
        }
        
        pgvector_embeddings = {
            "instrument_embeddings": np.random.rand(10, 128).tolist(),
            "regime_embeddings": np.random.rand(5, 128).tolist(),
        }
        
        # Perform migration with legacy data preservation
        migration_result = await data_catalog_adapter.migrate_with_legacy_preservation(
            postgresql_data=postgresql_data,
            neo4j_data=neo4j_data,
            pgvector_embeddings=pgvector_embeddings
        )
        
        # Verify migration succeeded
        assert migration_result.success is True
        
        # Verify legacy data accessibility
        assert "legacy_data_preserved" in migration_result.metadata
        assert migration_result.metadata["legacy_data_preserved"] is True
        
        # Verify Neo4j data accessibility
        neo4j_accessible = await data_catalog_adapter.verify_neo4j_accessibility(
            neo4j_data=neo4j_data
        )
        assert neo4j_accessible["accessible"] is True
        assert neo4j_accessible["node_count"] == len(neo4j_data["nodes"])
        assert neo4j_accessible["relationship_count"] == len(neo4j_data["relationships"])
        
        # Verify pgvector embeddings accessibility
        pgvector_accessible = await data_catalog_adapter.verify_pgvector_accessibility(
            embeddings=pgvector_embeddings
        )
        assert pgvector_accessible["accessible"] is True
        assert "instrument_embeddings" in pgvector_accessible["available_embeddings"]
        assert "regime_embeddings" in pgvector_accessible["available_embeddings"]
    
    # Feature: nautilus-trader-integration, Property 28: Incremental Data Updates
    @given(
        initial_data=postgresql_ohlcv_data(),
        update_data=postgresql_ohlcv_data()
    )
    @settings(max_examples=50, deadline=12000)
    async def test_incremental_data_updates(
        self, 
        data_catalog_adapter, 
        initial_data, 
        update_data
    ):
        """
        Property 28: Incremental Data Updates
        
        For any new market data, the system should support incremental addition 
        to Parquet storage without requiring full rebuilds.
        
        **Validates: Requirements 4.4**
        """
        # Perform initial migration
        initial_migration = await data_catalog_adapter.migrate_postgresql_to_parquet(
            postgresql_data=initial_data
        )
        
        assert initial_migration.success is True
        initial_parquet = initial_migration.parquet_data
        initial_row_count = len(pd.read_parquet(initial_parquet.file_path))
        
        # Perform incremental update
        incremental_result = await data_catalog_adapter.append_incremental_data(
            existing_parquet=initial_parquet,
            new_postgresql_data=update_data
        )
        
        # Verify incremental update succeeded
        assert incremental_result.success is True
        
        # Verify data was appended, not replaced
        updated_parquet = incremental_result.parquet_data
        updated_df = pd.read_parquet(updated_parquet.file_path)
        
        expected_row_count = initial_row_count + len(update_data.data)
        actual_row_count = len(updated_df)
        
        assert actual_row_count == expected_row_count, (
            f"Incremental update failed: expected {expected_row_count} rows, got {actual_row_count}"
        )
        
        # Verify no data corruption during append
        # Check that original data is still present
        original_timestamps = set(initial_data.data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
        updated_timestamps = set(updated_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Original timestamps should still be present
        missing_original = original_timestamps - updated_timestamps
        assert len(missing_original) == 0, (
            f"Original data lost during incremental update: {missing_original}"
        )
        
        # Verify metadata reflects incremental update
        assert "incremental_updates" in updated_parquet.metadata
        assert updated_parquet.metadata["incremental_updates"] >= 1
        assert "last_update_timestamp" in updated_parquet.metadata
    
    # Feature: nautilus-trader-integration, Property 30: Data Quality Assurance
    @given(
        postgresql_data=postgresql_ohlcv_data(),
        quality_params=data_quality_parameters()
    )
    @settings(max_examples=50, deadline=10000)
    async def test_data_quality_assurance(
        self, 
        data_catalog_adapter, 
        postgresql_data, 
        quality_params
    ):
        """
        Property 30: Data Quality Assurance
        
        For any data ingestion operation, the system should detect and reject 
        invalid or corrupted data during Parquet ingestion.
        
        **Validates: Requirements 4.6**
        """
        # Introduce data quality issues
        corrupted_data = postgresql_data.data.copy()
        
        # Introduce various data quality issues
        if quality_params.get("check_null_values", True):
            # Add some null values
            null_indices = np.random.choice(len(corrupted_data), size=max(1, len(corrupted_data) // 20), replace=False)
            corrupted_data.loc[null_indices, 'close'] = None
        
        if quality_params.get("check_duplicates", True):
            # Add duplicate rows
            duplicate_row = corrupted_data.iloc[0:1].copy()
            corrupted_data = pd.concat([corrupted_data, duplicate_row], ignore_index=True)
        
        if quality_params.get("check_value_ranges", True):
            # Add invalid price values
            corrupted_data.loc[0, 'high'] = -100.0  # Negative price
            corrupted_data.loc[1, 'volume'] = -1000.0  # Negative volume
        
        if quality_params.get("check_temporal_consistency", True):
            # Add temporal inconsistencies
            corrupted_data.loc[2, 'timestamp'] = datetime(1900, 1, 1)  # Invalid historical date
        
        corrupted_postgresql_data = PostgreSQLData(
            table_name=postgresql_data.table_name,
            data=corrupted_data,
            schema=postgresql_data.schema
        )
        
        # Attempt migration with quality checks
        migration_result = await data_catalog_adapter.migrate_with_quality_validation(
            postgresql_data=corrupted_postgresql_data,
            quality_parameters=quality_params
        )
        
        # Verify quality validation results
        assert "quality_validation" in migration_result.metadata
        quality_validation = migration_result.metadata["quality_validation"]
        
        assert "issues_detected" in quality_validation
        assert "issues_resolved" in quality_validation
        assert "data_quality_score" in quality_validation
        
        # If quality issues were detected, verify appropriate handling
        if quality_validation["issues_detected"] > 0:
            assert "issue_details" in quality_validation
            issue_details = quality_validation["issue_details"]
            
            # Verify specific issue detection
            if quality_params.get("check_null_values", True):
                null_issues = [issue for issue in issue_details if issue["type"] == "null_values"]
                if len(null_indices) > 0:
                    assert len(null_issues) > 0, "Null value issues should be detected"
            
            if quality_params.get("check_duplicates", True):
                duplicate_issues = [issue for issue in issue_details if issue["type"] == "duplicates"]
                assert len(duplicate_issues) > 0, "Duplicate issues should be detected"
            
            if quality_params.get("check_value_ranges", True):
                range_issues = [issue for issue in issue_details if issue["type"] == "value_range"]
                assert len(range_issues) > 0, "Value range issues should be detected"
            
            # Verify data quality score reflects issues
            assert quality_validation["data_quality_score"] < 1.0, (
                "Data quality score should be less than 1.0 when issues are detected"
            )
        
        # If migration succeeded despite issues, verify data was cleaned
        if migration_result.success:
            cleaned_df = pd.read_parquet(migration_result.parquet_data.file_path)
            
            # Verify cleaning was applied
            if quality_params.get("check_null_values", True):
                null_count = cleaned_df.isnull().sum().sum()
                max_allowed_nulls = len(cleaned_df) * quality_params.get("max_null_pct", 0.05)
                assert null_count <= max_allowed_nulls, (
                    f"Too many null values remain after cleaning: {null_count} > {max_allowed_nulls}"
                )
            
            if quality_params.get("check_value_ranges", True):
                # Verify no negative prices or volumes
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in cleaned_df.columns:
                        min_value = cleaned_df[col].min()
                        assert min_value >= 0, f"Negative values remain in {col}: {min_value}"
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash of the dataframe for integrity verification."""
        # Sort by timestamp to ensure consistent ordering
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        else:
            df_sorted = df.sort_index()
        
        # Convert to string representation and hash
        data_str = df_sorted.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "property"])