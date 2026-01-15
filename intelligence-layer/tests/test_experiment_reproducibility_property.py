"""
Property-based tests for experiment reproducibility.

**Feature: algorithmic-trading-system, Property 20: Experiment Reproducibility**
**Validates: Requirements 7.5**

This module tests that experiments with identical configurations produce
reproducible results and can be properly tracked and validated.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings
from typing import Dict, Any

from intelligence_layer.experiment_config import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentManager,
    ExperimentStatus,
    create_experiment_config
)


# Strategies for generating test data
@st.composite
def experiment_parameters(draw):
    """Generate realistic experiment parameters."""
    return {
        'learning_rate': draw(st.floats(min_value=0.0001, max_value=0.1)),
        'batch_size': draw(st.integers(min_value=16, max_value=512)),
        'epochs': draw(st.integers(min_value=1, max_value=100)),
        'model_type': draw(st.sampled_from(['tcn', 'lstm', 'transformer'])),
        'hidden_dim': draw(st.integers(min_value=32, max_value=512)),
        'dropout': draw(st.floats(min_value=0.0, max_value=0.5)),
        'seed': draw(st.integers(min_value=0, max_value=2**32 - 1))
    }


@st.composite
def data_config(draw):
    """Generate data configuration."""
    return {
        'window_size': draw(st.integers(min_value=32, max_value=256)),
        'features': draw(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10)),
        'normalization': draw(st.sampled_from(['zscore', 'minmax', 'robust'])),
        'train_split': draw(st.floats(min_value=0.6, max_value=0.8)),
        'validation_split': draw(st.floats(min_value=0.1, max_value=0.2))
    }


@st.composite
def experiment_config_strategy(draw):
    """Generate complete experiment configuration."""
    name = draw(st.text(min_size=1, max_size=50))
    description = draw(st.text(min_size=0, max_size=200))
    tags = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    
    parameters = draw(experiment_parameters())
    data_cfg = draw(data_config())
    
    return create_experiment_config(
        name=name,
        description=description,
        parameters=parameters,
        data_config=data_cfg,
        tags=tags,
        created_by="test_user"
    )


class TestExperimentReproducibility:
    """Property-based tests for experiment reproducibility."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = ExperimentManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=50)
    def test_configuration_hash_determinism(self, config: ExperimentConfig):
        """
        **Property 20a: Configuration Hash Determinism**
        
        For any experiment configuration, calculating the hash multiple times
        should produce the same result, ensuring deterministic reproducibility.
        
        **Validates: Requirements 7.5**
        """
        # Calculate hash multiple times
        hash1 = config.calculate_hash()
        hash2 = config.calculate_hash()
        hash3 = config.calculate_hash()
        
        # All hashes should be identical
        assert hash1 == hash2 == hash3
        
        # Hash should be non-empty and reasonable length
        assert len(hash1) == 64  # SHA256 hex digest
        assert all(c in '0123456789abcdef' for c in hash1)
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=50)
    def test_configuration_serialization_roundtrip(self, config: ExperimentConfig):
        """
        **Property 20b: Configuration Serialization Round-trip**
        
        For any experiment configuration, serializing to dict and back
        should preserve all configuration parameters exactly.
        
        **Validates: Requirements 7.5**
        """
        # Serialize and deserialize
        config_dict = config.to_dict()
        restored_config = ExperimentConfig.from_dict(config_dict)
        
        # All parameters should be preserved
        assert restored_config.parameters == config.parameters
        assert restored_config.data_config == config.data_config
        assert restored_config.model_config == config.model_config
        assert restored_config.training_config == config.training_config
        assert restored_config.evaluation_config == config.evaluation_config
        
        # Hash should be preserved
        assert restored_config.config_hash == config.config_hash
        
        # Metadata should be preserved
        assert restored_config.name == config.name
        assert restored_config.description == config.description
        assert restored_config.tags == config.tags
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=30)
    def test_experiment_persistence_and_retrieval(self, config: ExperimentConfig):
        """
        **Property 20c: Experiment Persistence and Retrieval**
        
        For any experiment configuration, saving and loading should
        preserve all configuration data exactly.
        
        **Validates: Requirements 7.5**
        """
        # Create and save experiment
        experiment_id = self.manager.create_experiment(config)
        
        # Load experiment
        loaded_config = self.manager.load_experiment(experiment_id)
        
        # All configuration should be preserved
        assert loaded_config.experiment_id == config.experiment_id
        assert loaded_config.parameters == config.parameters
        assert loaded_config.data_config == config.data_config
        assert loaded_config.config_hash == config.config_hash
        
        # Should be findable by hash
        matching_experiments = self.manager.find_experiments_by_hash(config.config_hash)
        assert len(matching_experiments) >= 1
        assert any(exp.experiment_id == experiment_id for exp in matching_experiments)
    
    @given(
        config1=experiment_config_strategy(),
        config2=experiment_config_strategy()
    )
    @settings(max_examples=30)
    def test_identical_configurations_same_hash(self, config1: ExperimentConfig, config2: ExperimentConfig):
        """
        **Property 20d: Identical Configurations Same Hash**
        
        For any two experiment configurations with identical parameters,
        they should produce the same configuration hash.
        
        **Validates: Requirements 7.5**
        """
        # Make configurations identical by copying parameters
        config2.parameters = config1.parameters.copy()
        config2.data_config = config1.data_config.copy()
        config2.model_config = config1.model_config.copy()
        config2.training_config = config1.training_config.copy()
        config2.evaluation_config = config1.evaluation_config.copy()
        config2.config_version = config1.config_version
        config2.git_commit = config1.git_commit
        
        # Recalculate hashes
        hash1 = config1.calculate_hash()
        hash2 = config2.calculate_hash()
        
        # Hashes should be identical
        assert hash1 == hash2
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=30)
    def test_experiment_result_tracking(self, config: ExperimentConfig):
        """
        **Property 20e: Experiment Result Tracking**
        
        For any experiment configuration, results should be properly
        tracked and associated with the correct experiment.
        
        **Validates: Requirements 7.5**
        """
        # Create experiment
        experiment_id = self.manager.create_experiment(config)
        
        # Create and save result
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=123.45,
            metrics={
                'accuracy': 0.85,
                'loss': 0.23,
                'sharpe_ratio': 1.45
            },
            artifacts={
                'model': 'model.pkl',
                'predictions': 'predictions.csv'
            }
        )
        
        self.manager.save_result(result)
        
        # Load and verify result
        loaded_result = self.manager.load_result(experiment_id)
        assert loaded_result is not None
        assert loaded_result.experiment_id == experiment_id
        assert loaded_result.status == ExperimentStatus.COMPLETED
        assert loaded_result.metrics == result.metrics
        assert loaded_result.artifacts == result.artifacts
        
        # Experiment status should be updated
        updated_config = self.manager.load_experiment(experiment_id)
        assert updated_config.status == ExperimentStatus.COMPLETED
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=20)
    def test_reproducibility_validation(self, config: ExperimentConfig):
        """
        **Property 20f: Reproducibility Validation**
        
        For any experiment configuration with proper metadata,
        reproducibility validation should pass.
        
        **Validates: Requirements 7.5**
        """
        # Set required metadata for reproducibility
        config.git_commit = "abc123def456"
        config.config_version = "1.0"
        
        # Create experiment
        experiment_id = self.manager.create_experiment(config)
        
        # Validate reproducibility
        validation_report = self.manager.validate_reproducibility(experiment_id)
        
        # Should be reproducible with no issues
        assert validation_report['reproducible'] is True
        assert len(validation_report['issues']) == 0
        assert validation_report['config_hash'] == config.config_hash
    
    @given(
        configs=st.lists(experiment_config_strategy(), min_size=2, max_size=5)
    )
    @settings(max_examples=20)
    def test_experiment_listing_and_filtering(self, configs):
        """
        **Property 20g: Experiment Listing and Filtering**
        
        For any list of experiment configurations, they should be
        properly stored, listed, and filterable by status.
        
        **Validates: Requirements 7.5**
        """
        experiment_ids = []
        
        # Create all experiments
        for config in configs:
            experiment_id = self.manager.create_experiment(config)
            experiment_ids.append(experiment_id)
        
        # List all experiments
        all_experiments = self.manager.list_experiments()
        assert len(all_experiments) >= len(configs)
        
        # All created experiments should be in the list
        listed_ids = {exp.experiment_id for exp in all_experiments}
        for exp_id in experiment_ids:
            assert exp_id in listed_ids
        
        # Filter by status
        created_experiments = self.manager.list_experiments(status=ExperimentStatus.CREATED)
        assert len(created_experiments) >= len(configs)
        
        # All should have CREATED status
        for exp in created_experiments:
            if exp.experiment_id in experiment_ids:
                assert exp.status == ExperimentStatus.CREATED
    
    def test_configuration_hash_excludes_metadata(self):
        """
        **Property 20h: Configuration Hash Excludes Metadata**
        
        Configuration hash should only depend on parameters that affect
        reproducibility, not metadata like name, description, or timestamps.
        
        **Validates: Requirements 7.5**
        """
        # Create base configuration
        config1 = create_experiment_config(
            name="Test Experiment 1",
            description="First description",
            parameters={'param1': 'value1'},
            tags=['tag1', 'tag2']
        )
        
        # Create identical configuration with different metadata
        config2 = create_experiment_config(
            name="Test Experiment 2",
            description="Second description",
            parameters={'param1': 'value1'},
            tags=['tag3', 'tag4']
        )
        
        # Different experiment IDs and timestamps
        config2.experiment_id = "different_id"
        config2.created_at = datetime.now(timezone.utc)
        
        # Hashes should be identical despite different metadata
        assert config1.calculate_hash() == config2.calculate_hash()
    
    def test_parameter_change_affects_hash(self):
        """
        **Property 20i: Parameter Change Affects Hash**
        
        Any change to parameters that affect reproducibility should
        result in a different configuration hash.
        
        **Validates: Requirements 7.5**
        """
        # Create base configuration
        config1 = create_experiment_config(
            name="Test Experiment",
            parameters={'learning_rate': 0.01, 'batch_size': 32}
        )
        
        # Create configuration with different parameters
        config2 = create_experiment_config(
            name="Test Experiment",
            parameters={'learning_rate': 0.02, 'batch_size': 32}  # Different learning rate
        )
        
        # Hashes should be different
        assert config1.calculate_hash() != config2.calculate_hash()
        
        # Test data config change
        config3 = create_experiment_config(
            name="Test Experiment",
            parameters={'learning_rate': 0.01, 'batch_size': 32},
            data_config={'window_size': 64}
        )
        
        assert config1.calculate_hash() != config3.calculate_hash()


if __name__ == "__main__":
    pytest.main([__file__])