"""Property-based tests for embedding model validation."""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import List, Tuple

from intelligence_layer.embedding_model import (
    TCNConfig,
    MarketEmbeddingTCN,
    EmbeddingValidator,
    create_tcn_model,
    prepare_market_data_for_tcn,
)


# Strategies for generating test data
@st.composite
def tcn_config_strategy(draw):
    """Generate valid TCN configurations."""
    input_dim = draw(st.integers(min_value=5, max_value=50))
    embedding_dim = draw(st.integers(min_value=8, max_value=128))
    
    # Generate channel sizes that make sense
    num_layers = draw(st.integers(min_value=2, max_value=4))
    channels = []
    current_dim = max(input_dim * 2, 32)
    
    for _ in range(num_layers - 1):
        channels.append(current_dim)
        current_dim = max(current_dim // 2, embedding_dim)
    channels.append(embedding_dim)
    
    return TCNConfig(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_channels=channels,
        kernel_size=draw(st.integers(min_value=3, max_value=7)),
        dropout=draw(st.floats(min_value=0.0, max_value=0.5)),
        activation=draw(st.sampled_from(["relu", "gelu", "tanh"]))
    )


@st.composite
def market_features_strategy(draw):
    """Generate market feature arrays."""
    n_timesteps = draw(st.integers(min_value=50, max_value=150))  # Further reduced max size
    n_features = draw(st.integers(min_value=5, max_value=10))  # Further reduced max features
    
    # Generate realistic market features (normalized) more efficiently
    base_value = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    noise_scale = draw(st.floats(min_value=0.01, max_value=0.5))
    
    # Create base array and add noise
    feature_array = np.full((n_timesteps, n_features), base_value, dtype=np.float32)
    noise = np.random.normal(0, noise_scale, (n_timesteps, n_features)).astype(np.float32)
    feature_array += noise
    
    # Clip to reasonable range
    feature_array = np.clip(feature_array, -2.0, 2.0)
    
    return feature_array


@st.composite
def sequence_length_strategy(draw):
    """Generate valid sequence lengths."""
    return draw(st.integers(min_value=16, max_value=128))


class TestEmbeddingModelValidationProperties:
    """Property-based tests for embedding model validation.
    
    **Validates: Requirements 13.9-13.10**
    """
    
    @given(tcn_config_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_property_model_creation_consistency(self, config):
        """
        Property: For any valid TCN configuration, model creation should be consistent and deterministic.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        # Create two models with same config
        model1 = create_tcn_model(config)
        model2 = create_tcn_model(config)
        
        # Models should have same architecture
        assert type(model1) == type(model2)
        assert model1.config.input_dim == model2.config.input_dim
        assert model1.config.embedding_dim == model2.config.embedding_dim
        assert model1.config.num_channels == model2.config.num_channels
        
        # Parameter counts should be identical
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())
        assert params1 == params2, f"Parameter counts differ: {params1} vs {params2}"
        
        # Models should be trainable
        assert all(p.requires_grad for p in model1.parameters())
        assert all(p.requires_grad for p in model2.parameters())
    
    @given(tcn_config_strategy(), market_features_strategy(), sequence_length_strategy())
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_property_embedding_output_validity(self, config, features, seq_length):
        """
        Property: For any valid model and input, embeddings should satisfy basic validity constraints.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        # Ensure config matches features
        assume(config.input_dim == features.shape[1])
        assume(features.shape[0] >= seq_length + 10)  # Enough data for sequences
        
        # Skip if all features are zero (edge case)
        assume(not np.all(features == 0))
        
        # Create model and prepare data
        model = create_tcn_model(config)
        model.eval()
        
        # Prepare input tensor
        input_tensor = prepare_market_data_for_tcn(features, seq_length, stride=10)
        assume(input_tensor.size(0) > 0)  # Must have at least one sequence
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Check output structure
        assert 'embeddings' in outputs
        assert 'reconstruction' in outputs
        assert 'sequence_embedding' in outputs
        assert 'contrastive_projection' in outputs
        
        embeddings = outputs['embeddings']
        reconstruction = outputs['reconstruction']
        sequence_embedding = outputs['sequence_embedding']
        
        # Check shapes
        batch_size = input_tensor.size(0)
        assert embeddings.shape == (batch_size, config.embedding_dim, seq_length)
        assert reconstruction.shape == input_tensor.shape
        assert sequence_embedding.shape == (batch_size, config.embedding_dim)
        
        # Check for valid values (no NaN or infinite)
        assert not torch.isnan(embeddings).any(), "Embeddings contain NaN values"
        assert not torch.isinf(embeddings).any(), "Embeddings contain infinite values"
        assert not torch.isnan(reconstruction).any(), "Reconstruction contains NaN values"
        assert not torch.isinf(reconstruction).any(), "Reconstruction contains infinite values"
        assert not torch.isnan(sequence_embedding).any(), "Sequence embeddings contain NaN values"
        assert not torch.isinf(sequence_embedding).any(), "Sequence embeddings contain infinite values"
        
        # Embeddings should have reasonable magnitude
        embedding_norm = torch.norm(embeddings, dim=1).mean()
        input_norm = torch.norm(input_tensor, dim=1).mean()
        
        # Zero input should produce zero or near-zero embeddings (valid behavior)
        if input_norm < 1e-6:
            # For zero input, embeddings can be zero (this is valid)
            assert embedding_norm < 1.0, f"Embeddings should be small for zero input: {embedding_norm}"
        else:
            # For non-zero input, embeddings should be non-zero
            assert embedding_norm > 0, "Embeddings have zero norm for non-zero input"
        
        # Embeddings should not be excessively large
        assert embedding_norm < 100, f"Embeddings have excessive norm: {embedding_norm}"
    
    @given(tcn_config_strategy(), market_features_strategy())
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much])
    def test_property_embedding_quality_validation(self, config, features):
        """
        Property: For any trained model, embedding quality validation should provide meaningful metrics.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        # Ensure we have enough data
        assume(config.input_dim == features.shape[1])
        assume(features.shape[0] >= 100)  # Need substantial data for quality metrics
        
        # Skip if all features are zero
        assume(not np.all(features == 0))
        
        # Create model
        model = create_tcn_model(config)
        model.eval()
        
        # Prepare validation data
        seq_length = min(32, features.shape[0] // 4)  # Smaller sequences for faster testing
        validation_tensor = prepare_market_data_for_tcn(features, seq_length, stride=5)
        assume(validation_tensor.size(0) >= 5)  # Need multiple sequences
        
        # Validate embedding quality
        results = EmbeddingValidator.validate_embedding_quality(
            model,
            validation_tensor,
            min_continuity=0.0,  # Very relaxed thresholds for random model
            min_diversity=0.0
        )
        
        # Check result structure
        required_keys = [
            'continuity_score', 'diversity_score', 'continuity_pass',
            'diversity_pass', 'overall_pass', 'validation_timestamp'
        ]
        for key in required_keys:
            assert key in results, f"Missing key in validation results: {key}"
        
        # Check value types and ranges
        assert isinstance(results['continuity_score'], float)
        assert isinstance(results['diversity_score'], float)
        assert isinstance(results['continuity_pass'], bool)
        assert isinstance(results['diversity_pass'], bool)
        assert isinstance(results['overall_pass'], bool)
        
        # Scores should be in reasonable ranges
        assert 0.0 <= results['continuity_score'] <= 1.0, f"Continuity score out of range: {results['continuity_score']}"
        assert 0.0 <= results['diversity_score'] <= 1.0, f"Diversity score out of range: {results['diversity_score']}"
        
        # Overall pass should be consistent with individual passes
        expected_overall = results['continuity_pass'] and results['diversity_pass']
        assert results['overall_pass'] == expected_overall, "Overall pass inconsistent with individual passes"
    
    @given(tcn_config_strategy())
    @settings(max_examples=15, deadline=10000)
    def test_property_model_determinism(self, config):
        """
        Property: For any model with fixed weights, identical inputs should produce identical outputs.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        # Create model and fix random seed
        torch.manual_seed(42)
        model = create_tcn_model(config)
        model.eval()
        
        # Generate test input
        batch_size = 4
        seq_length = 32
        test_input = torch.randn(batch_size, config.input_dim, seq_length)
        
        # Run inference twice
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)
        
        # Outputs should be identical
        for key in output1.keys():
            torch.testing.assert_close(
                output1[key], output2[key],
                msg=f"Non-deterministic output for {key}"
            )
    
    @given(
        st.integers(min_value=8, max_value=32),  # input_dim
        st.integers(min_value=16, max_value=64),  # embedding_dim
        st.integers(min_value=50, max_value=200)  # n_timesteps
    )
    @settings(max_examples=10, deadline=15000)
    def test_property_temporal_continuity_measurement(self, input_dim, embedding_dim, n_timesteps):
        """
        Property: For any embedding sequence, temporal continuity should be measurable and meaningful.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        # Create embeddings with known continuity properties
        batch_size = 8
        seq_length = min(64, n_timesteps // 2)
        
        # High continuity embeddings (similar consecutive embeddings)
        high_continuity_embeddings = torch.randn(batch_size, embedding_dim, seq_length)
        for i in range(seq_length - 1):
            # Make consecutive embeddings similar
            high_continuity_embeddings[:, :, i + 1] = (
                high_continuity_embeddings[:, :, i] + torch.randn(batch_size, embedding_dim) * 0.1
            )
        
        # Low continuity embeddings (random consecutive embeddings)
        low_continuity_embeddings = torch.randn(batch_size, embedding_dim, seq_length)
        
        # Measure continuity
        high_continuity_score = EmbeddingValidator.temporal_continuity_score(high_continuity_embeddings)
        low_continuity_score = EmbeddingValidator.temporal_continuity_score(low_continuity_embeddings)
        
        # Scores should be valid
        assert isinstance(high_continuity_score, float)
        assert isinstance(low_continuity_score, float)
        assert 0.0 <= high_continuity_score <= 1.0
        assert 0.0 <= low_continuity_score <= 1.0
        
        # High continuity should score higher than low continuity
        assert high_continuity_score > low_continuity_score, (
            f"High continuity score ({high_continuity_score}) should be > "
            f"low continuity score ({low_continuity_score})"
        )
        
        # High continuity should be reasonably high
        assert high_continuity_score > 0.7, f"High continuity score too low: {high_continuity_score}"
    
    @given(
        st.integers(min_value=16, max_value=64),  # embedding_dim
        st.integers(min_value=10, max_value=50)   # batch_size
    )
    @settings(max_examples=15, deadline=8000)
    def test_property_diversity_measurement(self, embedding_dim, batch_size):
        """
        Property: For any set of embeddings, diversity should be measurable and reflect actual diversity.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        # High diversity embeddings (random, different)
        high_diversity_embeddings = torch.randn(batch_size, embedding_dim)
        
        # Low diversity embeddings (similar to each other)
        base_embedding = torch.randn(1, embedding_dim)
        low_diversity_embeddings = base_embedding.repeat(batch_size, 1) + torch.randn(batch_size, embedding_dim) * 0.1
        
        # Measure diversity
        high_diversity_score = EmbeddingValidator.embedding_diversity_score(high_diversity_embeddings)
        low_diversity_score = EmbeddingValidator.embedding_diversity_score(low_diversity_embeddings)
        
        # Scores should be valid
        assert isinstance(high_diversity_score, float)
        assert isinstance(low_diversity_score, float)
        assert 0.0 <= high_diversity_score <= 1.0
        assert 0.0 <= low_diversity_score <= 1.0
        
        # High diversity should score higher than low diversity
        assert high_diversity_score > low_diversity_score, (
            f"High diversity score ({high_diversity_score}) should be > "
            f"low diversity score ({low_diversity_score})"
        )
    
    @given(market_features_strategy())
    @settings(max_examples=5, deadline=10000, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large])
    def test_property_data_preparation_consistency(self, features):
        """
        Property: For any feature array, data preparation should be consistent and preserve information.
        
        **Feature: algorithmic-trading-system, Property 8: Embedding Model Validation**
        **Validates: Requirements 13.9-13.10**
        """
        assume(features.shape[0] >= 50)  # Need enough data
        
        seq_length = min(32, features.shape[0] // 3)  # Smaller sequences
        stride = max(1, seq_length // 4)
        
        # Prepare data twice
        tensor1 = prepare_market_data_for_tcn(features, seq_length, stride)
        tensor2 = prepare_market_data_for_tcn(features, seq_length, stride)
        
        # Should be identical
        torch.testing.assert_close(tensor1, tensor2, msg="Data preparation is not deterministic")
        
        # Check shapes
        expected_sequences = len(range(0, features.shape[0] - seq_length + 1, stride))
        assert tensor1.shape == (expected_sequences, features.shape[1], seq_length)
        
        # Check that data is preserved (no NaN/inf introduced)
        assert not torch.isnan(tensor1).any(), "Data preparation introduced NaN values"
        assert not torch.isinf(tensor1).any(), "Data preparation introduced infinite values"
        
        # Check that sequences are properly extracted
        if expected_sequences > 0:
            # First sequence should match first part of original data
            first_sequence = tensor1[0].T  # Transpose back to (seq_len, features)
            original_first_part = features[:seq_length]
            
            np.testing.assert_allclose(
                first_sequence.numpy(), original_first_part,
                err_msg="First sequence doesn't match original data"
            )


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])