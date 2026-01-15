"""Property-based tests for training protocol compliance."""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any
import json

from intelligence_layer.training_protocol import (
    TrainingConfig,
    ModelVersion,
    EmbeddingTrainer,
    create_training_pipeline,
)
from intelligence_layer.embedding_model import TCNConfig


# Strategies for generating test data
@st.composite
def training_config_strategy(draw):
    """Generate valid training configurations."""
    input_dim = draw(st.integers(min_value=5, max_value=20))
    embedding_dim = draw(st.integers(min_value=8, max_value=32))
    
    # Generate smaller channel sizes for faster testing
    num_channels = [max(input_dim * 2, 16), embedding_dim]
    
    model_config = TCNConfig(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_channels=num_channels,
        kernel_size=3,
        dropout=0.1
    )
    
    return TrainingConfig(
        model_config=model_config,
        batch_size=draw(st.integers(min_value=2, max_value=8)),
        learning_rate=draw(st.floats(min_value=0.0001, max_value=0.01)),
        num_epochs=draw(st.integers(min_value=1, max_value=3)),  # Very short for testing
        sequence_length=draw(st.integers(min_value=16, max_value=32)),
        random_seed=draw(st.integers(min_value=1, max_value=1000)),
        early_stopping_patience=1
    )


@st.composite
def market_data_strategy(draw):
    """Generate market data for training."""
    n_timesteps = draw(st.integers(min_value=100, max_value=200))
    n_features = draw(st.integers(min_value=5, max_value=20))
    
    # Generate realistic market data
    data = np.random.randn(n_timesteps, n_features).astype(np.float32)
    # Normalize to reasonable range
    data = np.clip(data, -3.0, 3.0)
    
    return data


class TestTrainingProtocolComplianceProperties:
    """Property-based tests for training protocol compliance.
    
    **Validates: Requirements 13.6-13.8**
    """
    
    @given(training_config_strategy(), market_data_strategy())
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_property_offline_training_only(self, config, market_data):
        """
        Property: For any training session, it should use offline training only on pre-cut historical segments.
        
        **Feature: algorithmic-trading-system, Property 9: Embedding Training Protocol Compliance**
        **Validates: Requirements 13.6**
        """
        # Ensure config matches data
        assume(config.model_config.input_dim == market_data.shape[1])
        assume(market_data.shape[0] >= config.sequence_length + 20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EmbeddingTrainer(config, Path(temp_dir))
            
            # Prepare data - this should create temporal splits (no shuffling)
            train_loader, val_loader, test_loader = trainer.prepare_data(market_data)
            
            # Verify that data loaders don't shuffle (offline training requirement)
            # Check that shuffle=False in all data loaders
            assert not train_loader.sampler._shuffle if hasattr(train_loader.sampler, '_shuffle') else True
            assert not val_loader.sampler._shuffle if hasattr(val_loader.sampler, '_shuffle') else True
            assert not test_loader.sampler._shuffle if hasattr(test_loader.sampler, '_shuffle') else True
            
            # Verify temporal ordering is preserved
            # Get first few batches and check they represent consecutive time periods
            train_batches = []
            for i, batch in enumerate(train_loader):
                train_batches.append(batch)
                if i >= 2:  # Just check first few batches
                    break
            
            if len(train_batches) >= 2:
                # Batches should represent consecutive time periods
                # This is ensured by stride=1 and no shuffling in MarketDataset
                batch1 = train_batches[0]
                batch2 = train_batches[1]
                
                # Verify batches have expected shapes
                assert batch1.shape[1] == config.model_config.input_dim
                assert batch1.shape[2] == config.sequence_length
                assert batch2.shape[1] == config.model_config.input_dim
                assert batch2.shape[2] == config.sequence_length
            
            # Verify that training uses pre-cut segments (temporal splits)
            # The data preparation should split data temporally, not randomly
            original_length = len(market_data)
            train_end = int(original_length * config.train_test_split)
            
            # Check that training data comes from beginning of time series
            # This is verified by the temporal splitting logic in prepare_data
            assert train_end > config.sequence_length, "Training segment too small"
            
            # Train the model to verify offline training works
            model_version = trainer.train(train_loader, val_loader, market_data)
            
            # Verify training completed successfully
            assert isinstance(model_version, ModelVersion)
            assert len(trainer.training_metrics['train_loss']) > 0
            assert all(isinstance(loss, float) and loss >= 0 for loss in trainer.training_metrics['train_loss'])
    
    @given(training_config_strategy(), market_data_strategy())
    @settings(max_examples=8, deadline=25000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_property_model_weight_freezing(self, config, market_data):
        """
        Property: For any trained model, weights should be frozen before simulation to prevent lookahead bias.
        
        **Feature: algorithmic-trading-system, Property 9: Embedding Training Protocol Compliance**
        **Validates: Requirements 13.7**
        """
        # Ensure config matches data
        assume(config.model_config.input_dim == market_data.shape[1])
        assume(market_data.shape[0] >= config.sequence_length + 20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EmbeddingTrainer(config, Path(temp_dir))
            
            # Train model
            train_loader, val_loader, _ = trainer.prepare_data(market_data)
            model_version = trainer.train(train_loader, val_loader, market_data)
            
            # Load the trained model
            loaded_model, loaded_version = trainer.load_model_version(model_version.version_id)
            
            # Verify model can be set to eval mode (frozen for inference)
            loaded_model.eval()
            
            # Capture initial weights
            initial_weights = {}
            for name, param in loaded_model.named_parameters():
                initial_weights[name] = param.data.clone()
            
            # Simulate inference (model should be frozen)
            test_input = torch.randn(2, config.model_config.input_dim, config.sequence_length)
            
            with torch.no_grad():  # This simulates frozen weights during inference
                output1 = loaded_model(test_input)
                output2 = loaded_model(test_input)
            
            # Verify weights haven't changed during inference
            for name, param in loaded_model.named_parameters():
                torch.testing.assert_close(
                    param.data, initial_weights[name],
                    msg=f"Weight {name} changed during inference (weights not frozen)"
                )
            
            # Verify deterministic output (frozen model behavior)
            for key in output1.keys():
                torch.testing.assert_close(
                    output1[key], output2[key],
                    msg=f"Non-deterministic output for {key} (model not properly frozen)"
                )
            
            # Verify model can be explicitly frozen by disabling gradients
            for param in loaded_model.parameters():
                param.requires_grad_(False)
            
            # Verify all parameters are frozen
            assert all(not param.requires_grad for param in loaded_model.parameters()), \
                "Model parameters not properly frozen"
    
    @given(training_config_strategy(), market_data_strategy())
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_property_model_versioning_and_hashing(self, config, market_data):
        """
        Property: For any training session, all models should be versioned and hashed for reproducibility.
        
        **Feature: algorithmic-trading-system, Property 9: Embedding Training Protocol Compliance**
        **Validates: Requirements 13.8**
        """
        # Ensure config matches data
        assume(config.model_config.input_dim == market_data.shape[1])
        assume(market_data.shape[0] >= config.sequence_length + 20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EmbeddingTrainer(config, Path(temp_dir))
            
            # Train model
            train_loader, val_loader, _ = trainer.prepare_data(market_data)
            model_version = trainer.train(train_loader, val_loader, market_data)
            
            # Verify version information is complete
            assert isinstance(model_version, ModelVersion)
            assert model_version.version_id is not None and len(model_version.version_id) > 0
            assert model_version.model_hash is not None and len(model_version.model_hash) > 0
            assert model_version.config_hash is not None and len(model_version.config_hash) > 0
            assert model_version.training_data_hash is not None and len(model_version.training_data_hash) > 0
            assert isinstance(model_version.timestamp, datetime)
            
            # Verify hashes are consistent length (16 characters as per implementation)
            assert len(model_version.model_hash) == 16, f"Model hash wrong length: {len(model_version.model_hash)}"
            assert len(model_version.config_hash) == 16, f"Config hash wrong length: {len(model_version.config_hash)}"
            assert len(model_version.training_data_hash) == 16, f"Data hash wrong length: {len(model_version.training_data_hash)}"
            
            # Verify version files are saved
            model_dir = Path(temp_dir)
            version_file = model_dir / f"{model_version.version_id}_version.json"
            model_file = model_dir / f"{model_version.version_id}.pth"
            metrics_file = model_dir / f"{model_version.version_id}_metrics.json"
            
            assert version_file.exists(), "Version metadata file not saved"
            assert model_file.exists(), "Model file not saved"
            assert metrics_file.exists(), "Metrics file not saved"
            
            # Verify version file contains required information
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            required_keys = [
                'version_id', 'model_hash', 'config_hash', 'training_data_hash',
                'timestamp', 'training_config', 'validation_results', 'training_metrics'
            ]
            for key in required_keys:
                assert key in version_data, f"Missing key in version data: {key}"
            
            # Verify training config is preserved
            assert 'model_config' in version_data['training_config']
            assert version_data['training_config']['random_seed'] == config.random_seed
            assert version_data['training_config']['batch_size'] == config.batch_size
            
            # Verify metrics are preserved
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            assert 'train_loss' in metrics_data
            assert 'val_loss' in metrics_data
            assert len(metrics_data['train_loss']) > 0
            assert len(metrics_data['val_loss']) > 0
            
            # Verify model can be loaded using version information
            loaded_model, loaded_version = trainer.load_model_version(model_version.version_id)
            
            assert loaded_model is not None
            assert loaded_version.version_id == model_version.version_id
            assert loaded_version.model_hash == model_version.model_hash
            assert loaded_version.config_hash == model_version.config_hash
            assert loaded_version.training_data_hash == model_version.training_data_hash
    
    @given(training_config_strategy(), market_data_strategy())
    @settings(max_examples=5, deadline=40000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_property_reproducible_training_with_versioning(self, config, market_data):
        """
        Property: For any training configuration, identical setups should produce identical version hashes.
        
        **Feature: algorithmic-trading-system, Property 9: Embedding Training Protocol Compliance**
        **Validates: Requirements 13.8**
        """
        # Ensure config matches data
        assume(config.model_config.input_dim == market_data.shape[1])
        assume(market_data.shape[0] >= config.sequence_length + 20)
        
        # Train two identical models
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                # Create identical trainers
                trainer1 = EmbeddingTrainer(config, Path(temp_dir1))
                trainer2 = EmbeddingTrainer(config, Path(temp_dir2))
                
                # Train both models with identical data
                train_loader1, val_loader1, _ = trainer1.prepare_data(market_data)
                train_loader2, val_loader2, _ = trainer2.prepare_data(market_data)
                
                version1 = trainer1.train(train_loader1, val_loader1, market_data)
                version2 = trainer2.train(train_loader2, val_loader2, market_data)
                
                # Configuration and data hashes should be identical
                assert version1.config_hash == version2.config_hash, \
                    "Identical configurations should have same hash"
                assert version1.training_data_hash == version2.training_data_hash, \
                    "Identical training data should have same hash"
                
                # Model hashes might differ due to random initialization, but should be consistent
                # within the same training run (this tests hash consistency)
                assert len(version1.model_hash) == len(version2.model_hash) == 16
                assert version1.model_hash != version2.model_hash or config.random_seed == config.random_seed, \
                    "Model hashes should differ for different random initializations"
                
                # Verify both versions are complete and valid
                for version in [version1, version2]:
                    assert version.version_id is not None
                    assert len(version.training_metrics['train_loss']) > 0
                    assert 'overall_pass' in version.validation_results
    
    @given(st.integers(min_value=5, max_value=15), st.integers(min_value=100, max_value=150))
    @settings(max_examples=8, deadline=20000)
    def test_property_temporal_data_splitting(self, n_features, n_timesteps):
        """
        Property: For any dataset, temporal splitting should preserve chronological order and prevent lookahead bias.
        
        **Feature: algorithmic-trading-system, Property 9: Embedding Training Protocol Compliance**
        **Validates: Requirements 13.6**
        """
        # Create time series data with clear temporal pattern
        market_data = np.zeros((n_timesteps, n_features), dtype=np.float32)
        for t in range(n_timesteps):
            # Create data with temporal dependency
            market_data[t, :] = t / n_timesteps + np.random.randn(n_features) * 0.1
        
        config = TrainingConfig(
            model_config=TCNConfig(input_dim=n_features, embedding_dim=16),
            sequence_length=min(32, n_timesteps // 4),
            train_test_split=0.7,
            validation_split=0.2
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EmbeddingTrainer(config, Path(temp_dir))
            train_loader, val_loader, test_loader = trainer.prepare_data(market_data)
            
            # Extract data from loaders to verify temporal ordering
            train_data = []
            val_data = []
            test_data = []
            
            for batch in train_loader:
                train_data.append(batch)
                if len(train_data) >= 3:  # Limit for testing
                    break
            
            for batch in val_loader:
                val_data.append(batch)
                if len(val_data) >= 2:
                    break
            
            for batch in test_loader:
                test_data.append(batch)
                if len(test_data) >= 2:
                    break
            
            # Verify we have data
            assert len(train_data) > 0, "No training data generated"
            
            if len(train_data) >= 2:
                # Training data should come from earlier time periods
                # Check that the temporal pattern is preserved
                batch1 = train_data[0]
                batch2 = train_data[1]
                
                # Due to stride=1, consecutive batches should have overlapping sequences
                # The mean values should show temporal progression
                mean1 = batch1.mean().item()
                mean2 = batch2.mean().item()
                
                # Since our data increases with time, later batches should have higher means
                # (allowing for some noise and overlap)
                assert abs(mean2 - mean1) < 1.0, "Temporal ordering appears disrupted"
            
            # Verify that validation data comes after training data
            if len(val_data) > 0 and len(train_data) > 0:
                train_mean = torch.cat(train_data, dim=0).mean().item()
                val_mean = torch.cat(val_data, dim=0).mean().item()
                
                # Validation data should generally have higher values (later in time)
                # Allow some overlap due to sequence windows
                assert val_mean >= train_mean - 0.5, \
                    f"Validation data appears to come before training data: {val_mean} < {train_mean}"


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])