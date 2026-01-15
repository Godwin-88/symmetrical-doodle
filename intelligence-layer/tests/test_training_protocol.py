"""Tests for embedding training protocol."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from intelligence_layer.training_protocol import (
    TrainingConfig,
    ModelVersion,
    MarketDataset,
    EmbeddingTrainer,
    create_training_pipeline,
)
from intelligence_layer.embedding_model import TCNConfig


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default training configuration."""
        model_config = TCNConfig(input_dim=20, embedding_dim=64)
        config = TrainingConfig(model_config=model_config)
        
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.num_epochs == 100
        assert config.sequence_length == 64
        assert config.random_seed == 42
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        model_config = TCNConfig(input_dim=15, embedding_dim=32)
        config = TrainingConfig(
            model_config=model_config,
            batch_size=16,
            learning_rate=0.01,
            num_epochs=50
        )
        
        # Serialize
        config_dict = config.to_dict()
        
        # Deserialize
        restored_config = TrainingConfig.from_dict(config_dict)
        
        assert restored_config.batch_size == config.batch_size
        assert restored_config.learning_rate == config.learning_rate
        assert restored_config.num_epochs == config.num_epochs
        assert restored_config.model_config.input_dim == config.model_config.input_dim


class TestModelVersion:
    """Test model version tracking."""
    
    def test_model_version_creation(self):
        """Test model version creation."""
        model_config = TCNConfig(input_dim=10, embedding_dim=16)
        training_config = TrainingConfig(model_config=model_config)
        
        version = ModelVersion(
            version_id="test_v1",
            model_hash="abc123",
            config_hash="def456",
            training_data_hash="ghi789",
            timestamp=datetime.now(),
            training_config=training_config,
            validation_results={'overall_pass': True},
            training_metrics={'train_loss': [1.0, 0.8, 0.6]}
        )
        
        assert version.version_id == "test_v1"
        assert version.model_hash == "abc123"
        assert version.validation_results['overall_pass'] is True
    
    def test_version_serialization(self):
        """Test version serialization and deserialization."""
        model_config = TCNConfig(input_dim=8, embedding_dim=12)
        training_config = TrainingConfig(model_config=model_config)
        
        version = ModelVersion(
            version_id="test_v2",
            model_hash="xyz789",
            config_hash="uvw456",
            training_data_hash="rst123",
            timestamp=datetime.now(),
            training_config=training_config,
            validation_results={'continuity_score': 0.8},
            training_metrics={'val_loss': [0.9, 0.7]}
        )
        
        # Serialize
        version_dict = version.to_dict()
        
        # Deserialize
        restored_version = ModelVersion.from_dict(version_dict)
        
        assert restored_version.version_id == version.version_id
        assert restored_version.model_hash == version.model_hash
        assert restored_version.validation_results['continuity_score'] == 0.8


class TestMarketDataset:
    """Test market dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        features = np.random.randn(100, 15)  # 100 timesteps, 15 features
        dataset = MarketDataset(features, sequence_length=32, stride=1)
        
        # Check length
        expected_length = 100 - 32 + 1  # 69 sequences
        assert len(dataset) == expected_length
        
        # Check item shape
        item = dataset[0]
        assert item.shape == (15, 32)  # (features, sequence_length)
        assert isinstance(item, torch.Tensor)
    
    def test_dataset_with_stride(self):
        """Test dataset with different stride."""
        features = np.random.randn(50, 10)
        dataset = MarketDataset(features, sequence_length=16, stride=4)
        
        # With stride=4, fewer sequences
        expected_length = len(range(0, 50 - 16 + 1, 4))
        assert len(dataset) == expected_length
        
        # Check that sequences are properly strided
        item1 = dataset[0]
        item2 = dataset[1]
        assert item1.shape == item2.shape == (10, 16)
    
    def test_dataset_edge_cases(self):
        """Test dataset edge cases."""
        features = np.random.randn(20, 5)
        
        # Sequence length equal to data length
        dataset = MarketDataset(features, sequence_length=20, stride=1)
        assert len(dataset) == 1
        
        # Sequence length larger than data (should be empty)
        dataset = MarketDataset(features, sequence_length=25, stride=1)
        assert len(dataset) == 0


class TestEmbeddingTrainer:
    """Test embedding trainer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model_config = TCNConfig(input_dim=10, embedding_dim=16)
        self.training_config = TrainingConfig(
            model_config=self.model_config,
            batch_size=4,
            num_epochs=2,  # Short for testing
            early_stopping_patience=1
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            
            assert trainer.config == self.training_config
            assert trainer.model_dir == self.temp_dir
            assert trainer.device is not None
            assert trainer.current_epoch == 0
            assert trainer.best_loss == float('inf')
        finally:
            self.tearDown()
    
    def test_data_preparation(self):
        """Test data preparation."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            features = np.random.randn(200, 10)  # 200 timesteps, 10 features
            
            train_loader, val_loader, test_loader = trainer.prepare_data(features)
            
            # Check data loaders exist
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
            
            # Check batch shapes
            train_batch = next(iter(train_loader))
            assert train_batch.shape[1] == 10  # Features
            assert train_batch.shape[2] == self.training_config.sequence_length
        finally:
            self.tearDown()
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            trainer.initialize_model()
            
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None
            assert trainer.loss_fn is not None
            
            # Check model is on correct device
            assert next(trainer.model.parameters()).device == trainer.device
        finally:
            self.tearDown()
    
    def test_training_epoch(self):
        """Test single training epoch."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            trainer.initialize_model()
            
            # Create larger dataset to ensure we have batches
            features = np.random.randn(200, 10)  # Increased size
            train_loader, _, _ = trainer.prepare_data(features)
            
            # Train one epoch
            train_loss = trainer.train_epoch(train_loader)
            
            assert isinstance(train_loss, float)
            assert train_loss >= 0  # Loss should be non-negative
        finally:
            self.tearDown()
    
    def test_validation_epoch(self):
        """Test single validation epoch."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            trainer.initialize_model()
            
            # Create larger dataset
            features = np.random.randn(200, 10)  # Increased size
            _, val_loader, _ = trainer.prepare_data(features)
            
            # Validate one epoch
            val_loss, quality_metrics = trainer.validate_epoch(val_loader)
            
            assert isinstance(val_loss, float)
            assert val_loss >= 0
            assert 'continuity_score' in quality_metrics
            assert 'diversity_score' in quality_metrics
        finally:
            self.tearDown()
    
    def test_short_training_run(self):
        """Test complete short training run."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            
            # Create larger dataset
            features = np.random.randn(300, 10)  # Increased size
            train_loader, val_loader, _ = trainer.prepare_data(features)
            
            # Train model
            model_version = trainer.train(train_loader, val_loader, features)
            
            # Check version was created
            assert isinstance(model_version, ModelVersion)
            assert model_version.version_id is not None
            assert len(model_version.training_metrics['train_loss']) > 0
            
            # Check files were saved
            model_files = list(self.temp_dir.glob(f"{model_version.version_id}*"))
            assert len(model_files) >= 2  # Model file and version file
        finally:
            self.tearDown()
    
    def test_model_saving_and_loading(self):
        """Test model saving and loading."""
        self.setUp()
        try:
            trainer = EmbeddingTrainer(self.training_config, self.temp_dir)
            
            # Create and train model
            features = np.random.randn(250, 10)  # Increased size
            train_loader, val_loader, _ = trainer.prepare_data(features)
            model_version = trainer.train(train_loader, val_loader, features)
            
            # Load the model
            loaded_model, loaded_version = trainer.load_model_version(model_version.version_id)
            
            assert loaded_model is not None
            assert loaded_version.version_id == model_version.version_id
            assert loaded_version.model_hash == model_version.model_hash
        finally:
            self.tearDown()


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_create_training_pipeline(self):
        """Test complete training pipeline creation and execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            features = np.random.randn(150, 12)  # 150 timesteps, 12 features
            
            # Create custom configurations for fast testing
            model_config = TCNConfig(
                input_dim=12,
                embedding_dim=24,
                num_channels=[32, 24]  # Smaller model
            )
            
            training_config = TrainingConfig(
                model_config=model_config,
                batch_size=8,
                num_epochs=3,  # Very short for testing
                early_stopping_patience=2
            )
            
            # Run pipeline
            trainer, model_version = create_training_pipeline(
                features,
                model_config,
                training_config,
                Path(temp_dir)
            )
            
            # Check results
            assert isinstance(trainer, EmbeddingTrainer)
            assert isinstance(model_version, ModelVersion)
            assert model_version.version_id is not None
            
            # Check that model files exist
            model_dir = Path(temp_dir)
            model_files = list(model_dir.glob(f"{model_version.version_id}*"))
            assert len(model_files) >= 2
    
    def test_pipeline_with_defaults(self):
        """Test pipeline with default configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            features = np.random.randn(100, 8)
            
            # Run with defaults (but short epochs for testing)
            trainer, model_version = create_training_pipeline(
                features,
                model_dir=Path(temp_dir)
            )
            
            # Override epochs for testing
            trainer.config.num_epochs = 2
            
            assert trainer.config.model_config.input_dim == 8  # Should match features
            assert model_version is not None


class TestTrainingValidation:
    """Test training validation and quality checks."""
    
    def test_hash_computation(self):
        """Test hash computation for versioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EmbeddingTrainer(
                TrainingConfig(model_config=TCNConfig()),
                Path(temp_dir)
            )
            
            # Test different data types
            dict_hash = trainer._compute_hash({'a': 1, 'b': 2})
            array_hash = trainer._compute_hash(np.array([1, 2, 3]))
            tensor_hash = trainer._compute_hash(torch.tensor([1, 2, 3]))
            
            assert isinstance(dict_hash, str)
            assert isinstance(array_hash, str)
            assert isinstance(tensor_hash, str)
            assert len(dict_hash) == 16  # Should be truncated to 16 chars
    
    def test_reproducibility(self):
        """Test training reproducibility with same seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            features = np.random.randn(300, 6)  # Increased size
            
            # Train two models with same seed
            config1 = TrainingConfig(
                model_config=TCNConfig(input_dim=6, embedding_dim=12),
                random_seed=123,
                num_epochs=2,
                batch_size=4
            )
            
            config2 = TrainingConfig(
                model_config=TCNConfig(input_dim=6, embedding_dim=12),
                random_seed=123,
                num_epochs=2,
                batch_size=4
            )
            
            trainer1 = EmbeddingTrainer(config1, Path(temp_dir) / "model1")
            trainer2 = EmbeddingTrainer(config2, Path(temp_dir) / "model2")
            
            # Prepare same data
            train_loader1, val_loader1, _ = trainer1.prepare_data(features)
            train_loader2, val_loader2, _ = trainer2.prepare_data(features)
            
            # Train both models
            version1 = trainer1.train(train_loader1, val_loader1, features)
            version2 = trainer2.train(train_loader2, val_loader2, features)
            
            # Models should have same configuration hash
            assert version1.config_hash == version2.config_hash
            assert version1.training_data_hash == version2.training_data_hash