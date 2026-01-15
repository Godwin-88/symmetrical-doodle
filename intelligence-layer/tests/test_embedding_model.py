"""Tests for TCN embedding model."""

import pytest
import torch
import numpy as np
from typing import Dict

from intelligence_layer.embedding_model import (
    TCNConfig,
    TemporalBlock,
    TemporalConvNet,
    MarketEmbeddingTCN,
    EmbeddingLoss,
    EmbeddingValidator,
    create_tcn_model,
    prepare_market_data_for_tcn,
)


class TestTCNConfig:
    """Test TCN configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TCNConfig()
        
        assert config.input_dim == 64
        assert config.embedding_dim == 128
        assert config.num_channels == [256, 128, 64, 128]
        assert config.kernel_size == 3
        assert config.dropout == 0.2
        assert config.activation == "relu"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TCNConfig(
            input_dim=32,
            embedding_dim=64,
            num_channels=[128, 64, 32],
            kernel_size=5,
            dropout=0.1,
            activation="gelu"
        )
        
        assert config.input_dim == 32
        assert config.embedding_dim == 64
        assert config.num_channels == [128, 64, 32]
        assert config.kernel_size == 5
        assert config.dropout == 0.1
        assert config.activation == "gelu"


class TestTemporalBlock:
    """Test temporal block component."""
    
    def test_temporal_block_forward(self):
        """Test temporal block forward pass."""
        block = TemporalBlock(
            n_inputs=10,
            n_outputs=20,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=2,
            dropout=0.1
        )
        
        # Test input
        x = torch.randn(4, 10, 50)  # (batch, channels, sequence)
        output = block(x)
        
        assert output.shape == (4, 20, 50)
        assert not torch.isnan(output).any()
    
    def test_temporal_block_residual(self):
        """Test residual connection in temporal block."""
        # Same input/output dimensions (no downsampling)
        block = TemporalBlock(
            n_inputs=20,
            n_outputs=20,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=2
        )
        
        x = torch.randn(2, 20, 30)
        output = block(x)
        
        assert output.shape == x.shape
        # Output should be different from input due to processing
        assert not torch.allclose(output, x)


class TestTemporalConvNet:
    """Test TCN network."""
    
    def test_tcn_forward(self):
        """Test TCN forward pass."""
        config = TCNConfig(
            input_dim=10,
            embedding_dim=32,
            num_channels=[64, 32]
        )
        
        tcn = TemporalConvNet(config)
        x = torch.randn(4, 10, 50)  # (batch, input_dim, sequence)
        
        output = tcn(x)
        
        assert output.shape == (4, 32, 50)  # (batch, embedding_dim, sequence)
        assert not torch.isnan(output).any()
    
    def test_tcn_different_sequence_lengths(self):
        """Test TCN with different sequence lengths."""
        config = TCNConfig(input_dim=5, embedding_dim=16)
        tcn = TemporalConvNet(config)
        
        # Test different sequence lengths
        for seq_len in [10, 32, 64, 100]:
            x = torch.randn(2, 5, seq_len)
            output = tcn(x)
            assert output.shape == (2, 16, seq_len)


class TestMarketEmbeddingTCN:
    """Test complete market embedding model."""
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        config = TCNConfig(input_dim=20, embedding_dim=64)
        model = MarketEmbeddingTCN(config)
        
        # Check model components exist
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'projection_head')
        assert hasattr(model, 'global_pool')
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_model_forward(self):
        """Test complete model forward pass."""
        config = TCNConfig(input_dim=15, embedding_dim=32)
        model = MarketEmbeddingTCN(config)
        
        x = torch.randn(3, 15, 40)  # (batch, features, sequence)
        outputs = model(x)
        
        # Check all expected outputs
        assert 'embeddings' in outputs
        assert 'reconstruction' in outputs
        assert 'sequence_embedding' in outputs
        assert 'contrastive_projection' in outputs
        
        # Check shapes
        assert outputs['embeddings'].shape == (3, 32, 40)
        assert outputs['reconstruction'].shape == (3, 15, 40)
        assert outputs['sequence_embedding'].shape == (3, 32)
        assert outputs['contrastive_projection'].shape[0] == 3  # Batch dimension
    
    def test_encode_decode(self):
        """Test encode and decode methods."""
        config = TCNConfig(input_dim=10, embedding_dim=20)
        model = MarketEmbeddingTCN(config)
        
        x = torch.randn(2, 10, 30)
        
        # Test encoding
        embeddings = model.encode(x)
        assert embeddings.shape == (2, 20, 30)
        
        # Test decoding
        reconstruction = model.decode(embeddings)
        assert reconstruction.shape == (2, 10, 30)
    
    def test_sequence_embedding(self):
        """Test sequence-level embedding extraction."""
        config = TCNConfig(input_dim=8, embedding_dim=16)
        model = MarketEmbeddingTCN(config)
        
        x = torch.randn(4, 8, 50)
        seq_embedding = model.get_sequence_embedding(x)
        
        assert seq_embedding.shape == (4, 16)
        assert not torch.isnan(seq_embedding).any()


class TestEmbeddingLoss:
    """Test embedding loss components."""
    
    def test_reconstruction_loss(self):
        """Test reconstruction loss calculation."""
        config = TCNConfig()
        loss_fn = EmbeddingLoss(config)
        
        original = torch.randn(2, 10, 20)
        reconstructed = torch.randn(2, 10, 20)
        
        recon_loss = loss_fn.reconstruction_loss(original, reconstructed)
        
        assert isinstance(recon_loss, torch.Tensor)
        assert recon_loss.item() >= 0
    
    def test_contrastive_loss(self):
        """Test contrastive loss calculation."""
        config = TCNConfig()
        loss_fn = EmbeddingLoss(config)
        
        projections = torch.randn(6, 32)  # 6 samples, 32-dim projections
        
        contrast_loss = loss_fn.contrastive_loss(projections)
        
        assert isinstance(contrast_loss, torch.Tensor)
        assert contrast_loss.item() >= 0
    
    def test_temporal_smoothness_loss(self):
        """Test temporal smoothness loss."""
        config = TCNConfig()
        loss_fn = EmbeddingLoss(config)
        
        embeddings = torch.randn(3, 16, 25)  # (batch, embed_dim, seq_len)
        
        smooth_loss = loss_fn.temporal_smoothness_loss(embeddings)
        
        assert isinstance(smooth_loss, torch.Tensor)
        assert smooth_loss.item() >= 0
    
    def test_total_loss(self):
        """Test total loss computation."""
        config = TCNConfig(input_dim=10, embedding_dim=16)
        model = MarketEmbeddingTCN(config)
        loss_fn = EmbeddingLoss(config)
        
        x = torch.randn(4, 10, 30)
        outputs = model(x)
        
        losses = loss_fn(x, outputs)
        
        # Check all loss components
        assert 'total_loss' in losses
        assert 'reconstruction_loss' in losses
        assert 'contrastive_loss' in losses
        assert 'temporal_smoothness_loss' in losses
        
        # Check values are reasonable
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.item() >= 0


class TestEmbeddingValidator:
    """Test embedding quality validation."""
    
    def test_temporal_continuity_score(self):
        """Test temporal continuity measurement."""
        # Create embeddings with high continuity (similar consecutive embeddings)
        embeddings = torch.randn(2, 16, 20)
        
        # Make consecutive embeddings similar
        for i in range(embeddings.size(2) - 1):
            embeddings[:, :, i + 1] = embeddings[:, :, i] + torch.randn(2, 16) * 0.1
        
        continuity = EmbeddingValidator.temporal_continuity_score(embeddings)
        
        assert isinstance(continuity, float)
        assert 0 <= continuity <= 1
        assert continuity > 0.5  # Should be high due to similar consecutive embeddings
    
    def test_embedding_diversity_score(self):
        """Test embedding diversity measurement."""
        # Create diverse embeddings
        embeddings = torch.randn(10, 32)
        
        diversity = EmbeddingValidator.embedding_diversity_score(embeddings)
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
    
    def test_validate_embedding_quality(self):
        """Test complete embedding quality validation."""
        config = TCNConfig(input_dim=8, embedding_dim=16)
        model = MarketEmbeddingTCN(config)
        
        validation_data = torch.randn(5, 8, 30)
        
        results = EmbeddingValidator.validate_embedding_quality(
            model, validation_data,
            min_continuity=0.1,  # Low threshold for random model
            min_diversity=0.1
        )
        
        # Check result structure
        assert 'continuity_score' in results
        assert 'diversity_score' in results
        assert 'continuity_pass' in results
        assert 'diversity_pass' in results
        assert 'overall_pass' in results
        assert 'validation_timestamp' in results
        
        # Check types
        assert isinstance(results['continuity_score'], float)
        assert isinstance(results['diversity_score'], float)
        assert isinstance(results['continuity_pass'], bool)
        assert isinstance(results['diversity_pass'], bool)
        assert isinstance(results['overall_pass'], bool)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_tcn_model(self):
        """Test TCN model factory function."""
        config = TCNConfig(input_dim=12, embedding_dim=24)
        model = create_tcn_model(config)
        
        assert isinstance(model, MarketEmbeddingTCN)
        assert model.config == config
        
        # Test that model can process data
        x = torch.randn(2, 12, 40)
        outputs = model(x)
        assert 'embeddings' in outputs
    
    def test_prepare_market_data_for_tcn(self):
        """Test market data preparation for TCN."""
        # Create sample feature data
        features = np.random.randn(100, 15)  # 100 timesteps, 15 features
        
        tensor = prepare_market_data_for_tcn(
            features, 
            sequence_length=32, 
            stride=1
        )
        
        # Check shape: (n_sequences, n_features, sequence_length)
        expected_sequences = 100 - 32 + 1  # 69 sequences
        assert tensor.shape == (expected_sequences, 15, 32)
        assert isinstance(tensor, torch.Tensor)
    
    def test_prepare_market_data_with_stride(self):
        """Test market data preparation with stride."""
        features = np.random.randn(50, 10)
        
        tensor = prepare_market_data_for_tcn(
            features,
            sequence_length=16,
            stride=4
        )
        
        # With stride=4, we get fewer sequences
        expected_sequences = len(range(0, 50 - 16 + 1, 4))  # 9 sequences
        assert tensor.shape == (expected_sequences, 10, 16)


class TestModelIntegration:
    """Integration tests for the complete model."""
    
    def test_training_step_simulation(self):
        """Simulate a training step to ensure everything works together."""
        config = TCNConfig(input_dim=20, embedding_dim=32)
        model = create_tcn_model(config)
        loss_fn = EmbeddingLoss(config)
        
        # Create batch of data
        batch_size = 8
        sequence_length = 64
        x = torch.randn(batch_size, config.input_dim, sequence_length)
        
        # Forward pass
        outputs = model(x)
        
        # Compute losses
        losses = loss_fn(x, outputs)
        
        # Check that we can backpropagate
        total_loss = losses['total_loss']
        total_loss.backward()
        
        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients
    
    def test_model_evaluation_mode(self):
        """Test model in evaluation mode."""
        config = TCNConfig(input_dim=10, embedding_dim=16)
        model = create_tcn_model(config)
        
        x = torch.randn(4, 10, 32)
        
        # Training mode
        model.train()
        outputs_train = model(x)
        
        # Evaluation mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(x)
        
        # Outputs should have same structure
        assert set(outputs_train.keys()) == set(outputs_eval.keys())
        
        # Shapes should be identical
        for key in outputs_train.keys():
            assert outputs_train[key].shape == outputs_eval[key].shape