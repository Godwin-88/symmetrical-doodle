"""Embedding training protocol with version control and validation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import logging
from datetime import datetime
import pickle
from copy import deepcopy

from .embedding_model import (
    MarketEmbeddingTCN,
    TCNConfig,
    EmbeddingLoss,
    EmbeddingValidator,
    create_tcn_model,
    prepare_market_data_for_tcn,
)
from .feature_extraction import FeatureExtractor, FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for embedding model training."""
    
    # Model configuration
    model_config: TCNConfig
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Data parameters
    sequence_length: int = 64
    train_test_split: float = 0.8
    validation_split: float = 0.1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Validation criteria
    min_continuity_score: float = 0.7
    min_diversity_score: float = 0.3
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_best_model: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config_dict = asdict(self)
        config_dict['model_config'] = asdict(self.model_config)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        model_config = TCNConfig(**config_dict.pop('model_config'))
        return cls(model_config=model_config, **config_dict)


@dataclass
class ModelVersion:
    """Model version information for reproducibility."""
    version_id: str
    model_hash: str
    config_hash: str
    training_data_hash: str
    timestamp: datetime
    training_config: TrainingConfig
    validation_results: Dict[str, Any]
    training_metrics: Dict[str, List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version_id': self.version_id,
            'model_hash': self.model_hash,
            'config_hash': self.config_hash,
            'training_data_hash': self.training_data_hash,
            'timestamp': self.timestamp.isoformat(),
            'training_config': self.training_config.to_dict(),
            'validation_results': self.validation_results,
            'training_metrics': self.training_metrics,
        }
    
    @classmethod
    def from_dict(cls, version_dict: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        return cls(
            version_id=version_dict['version_id'],
            model_hash=version_dict['model_hash'],
            config_hash=version_dict['config_hash'],
            training_data_hash=version_dict['training_data_hash'],
            timestamp=datetime.fromisoformat(version_dict['timestamp']),
            training_config=TrainingConfig.from_dict(version_dict['training_config']),
            validation_results=version_dict['validation_results'],
            training_metrics=version_dict['training_metrics'],
        )


class MarketDataset(Dataset):
    """Dataset for market data sequences."""
    
    def __init__(
        self,
        features: np.ndarray,
        sequence_length: int = 64,
        stride: int = 1,
        transform: Optional[callable] = None
    ):
        """
        Initialize market dataset.
        
        Args:
            features: Feature array (time_steps, n_features)
            sequence_length: Length of input sequences
            stride: Stride between sequences
            transform: Optional data transformation
        """
        self.features = features
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Pre-compute sequence indices
        self.indices = list(range(0, len(features) - sequence_length + 1, stride))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        sequence = self.features[start_idx:end_idx].T  # (n_features, seq_len)
        tensor = torch.FloatTensor(sequence)
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor


class EmbeddingTrainer:
    """Trainer for embedding models with version control and validation."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model_dir: Path = Path("models"),
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model_dir: Directory to save models
            device: Training device (CPU/GPU)
        """
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.validator = EmbeddingValidator()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'continuity_score': [],
            'diversity_score': [],
        }
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for versioning."""
        if isinstance(data, dict):
            # Handle PyTorch state dict
            if any(isinstance(v, torch.Tensor) for v in data.values()):
                # Convert tensors to numpy for hashing
                data_for_hash = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        # Convert bytes to hex string for JSON serialization
                        tensor_bytes = v.detach().cpu().numpy().tobytes()
                        data_for_hash[k] = tensor_bytes.hex()
                    else:
                        data_for_hash[k] = str(v)
                data_str = json.dumps(data_for_hash, sort_keys=True)
            else:
                data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, np.ndarray):
            data_str = str(data.tobytes())
        elif isinstance(data, torch.Tensor):
            data_str = str(data.detach().cpu().numpy().tobytes())
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _create_model_version(
        self,
        model: MarketEmbeddingTCN,
        training_data: np.ndarray,
        validation_results: Dict[str, Any]
    ) -> ModelVersion:
        """Create model version for tracking."""
        # Compute hashes
        model_hash = self._compute_hash(model.state_dict())
        config_hash = self._compute_hash(self.config.to_dict())
        data_hash = self._compute_hash(training_data)
        
        # Generate version ID
        version_id = f"tcn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_hash[:8]}"
        
        return ModelVersion(
            version_id=version_id,
            model_hash=model_hash,
            config_hash=config_hash,
            training_data_hash=data_hash,
            timestamp=datetime.now(),
            training_config=deepcopy(self.config),
            validation_results=validation_results,
            training_metrics=deepcopy(self.training_metrics),
        )
    
    def prepare_data(
        self,
        features: np.ndarray,
        feature_config: Optional[FeatureConfig] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare training, validation, and test data loaders.
        
        Args:
            features: Feature array (time_steps, n_features)
            feature_config: Feature extraction configuration
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Ensure we have enough data for the sequence length
        min_required = self.config.sequence_length + 10  # Some buffer
        if len(features) < min_required:
            logger.warning(f"Dataset too small ({len(features)} < {min_required}). Using minimal splits.")
            # For very small datasets, use most data for training
            train_end = max(len(features) - 5, self.config.sequence_length)
            val_end = len(features)
        else:
            # Split data temporally (no shuffling to prevent lookahead bias)
            n_samples = len(features)
            train_end = int(n_samples * self.config.train_test_split)
            val_end = int(n_samples * (self.config.train_test_split + self.config.validation_split))
        
        train_features = features[:train_end]
        val_features = features[train_end:val_end] if val_end > train_end else features[train_end-5:train_end]
        test_features = features[val_end:] if val_end < len(features) else features[-5:]
        
        logger.info(f"Data split - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
        
        # Create datasets
        train_dataset = MarketDataset(
            train_features,
            sequence_length=self.config.sequence_length,
            stride=1
        )
        
        val_dataset = MarketDataset(
            val_features,
            sequence_length=self.config.sequence_length,
            stride=1
        )
        
        test_dataset = MarketDataset(
            test_features,
            sequence_length=self.config.sequence_length,
            stride=1
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1,
            shuffle=False,  # No shuffling to maintain temporal order
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.config.batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(self.config.batch_size, len(test_dataset)) if len(test_dataset) > 0 else 1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self):
        """Initialize model, optimizer, and loss function."""
        # Create model
        self.model = create_tcn_model(self.config.model_config)
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create loss function
        self.loss_fn = EmbeddingLoss(self.config.model_config)
        
        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss
            losses = self.loss_fn(batch, outputs)
            total_loss_batch = losses['total_loss']
            
            # Check for NaN loss
            if torch.isnan(total_loss_batch):
                logger.warning("NaN loss detected, skipping batch")
                continue
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_batch.item():.6f}")
        
        return total_loss / max(num_batches, 1)  # Avoid division by zero
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                losses = self.loss_fn(batch, outputs)
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                # Collect embeddings for quality metrics
                all_embeddings.append(outputs['sequence_embedding'].cpu())
        
        avg_loss = total_loss / max(num_batches, 1)  # Avoid division by zero
        
        # Compute embedding quality metrics
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            continuity_score = self.validator.temporal_continuity_score(
                all_embeddings.unsqueeze(-1).expand(-1, -1, 10)  # Fake temporal dimension
            )
            diversity_score = self.validator.embedding_diversity_score(all_embeddings)
        else:
            continuity_score = 0.0
            diversity_score = 0.0
        
        quality_metrics = {
            'continuity_score': continuity_score,
            'diversity_score': diversity_score,
        }
        
        return avg_loss, quality_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_data: np.ndarray
    ) -> ModelVersion:
        """
        Train the embedding model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            training_data: Original training data for versioning
            
        Returns:
            Model version information
        """
        logger.info("Starting training...")
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, quality_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record metrics
            self.training_metrics['train_loss'].append(train_loss)
            self.training_metrics['val_loss'].append(val_loss)
            self.training_metrics['continuity_score'].append(quality_metrics['continuity_score'])
            self.training_metrics['diversity_score'].append(quality_metrics['diversity_score'])
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Continuity: {quality_metrics['continuity_score']:.4f}, "
                f"Diversity: {quality_metrics['diversity_score']:.4f}"
            )
            
            # Check for improvement
            if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if self.config.keep_best_model:
                    self.best_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        self.model_dir / "best_model.pth"
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1)
        
        # Final validation
        final_validation = self._final_validation(val_loader)
        
        # Create model version
        model_version = self._create_model_version(
            self.model,
            training_data,
            final_validation
        )
        
        # Save model and version info
        self._save_model_version(model_version)
        
        logger.info(f"Training completed. Model version: {model_version.version_id}")
        
        return model_version
    
    def _final_validation(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Perform final validation with all criteria."""
        # Load best model if available
        best_model_path = self.model_dir / "best_model.pth"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        # Get validation data
        val_data = []
        with torch.no_grad():
            for batch in val_loader:
                val_data.append(batch)
                if len(val_data) >= 5:  # Limit for validation
                    break
        
        if val_data:
            val_tensor = torch.cat(val_data, dim=0).to(self.device)
            
            # Comprehensive validation
            validation_results = self.validator.validate_embedding_quality(
                self.model,
                val_tensor,
                min_continuity=self.config.min_continuity_score,
                min_diversity=self.config.min_diversity_score
            )
        else:
            validation_results = {
                'continuity_score': 0.0,
                'diversity_score': 0.0,
                'continuity_pass': False,
                'diversity_pass': False,
                'overall_pass': False,
                'validation_timestamp': datetime.now().isoformat(),
            }
        
        return validation_results
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_metrics': self.training_metrics,
            'config': self.config.to_dict(),
        }
        
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_model_version(self, version: ModelVersion):
        """Save model version and metadata."""
        # Save model state
        model_path = self.model_dir / f"{version.version_id}.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save version metadata
        version_path = self.model_dir / f"{version.version_id}_version.json"
        with open(version_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Save training metrics
        metrics_path = self.model_dir / f"{version.version_id}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Saved model version: {version.version_id}")
    
    def load_model_version(self, version_id: str) -> Tuple[MarketEmbeddingTCN, ModelVersion]:
        """
        Load a specific model version.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            Tuple of (model, version_info)
        """
        # Load version metadata
        version_path = self.model_dir / f"{version_id}_version.json"
        if not version_path.exists():
            raise FileNotFoundError(f"Version {version_id} not found")
        
        with open(version_path, 'r') as f:
            version_dict = json.load(f)
        
        version = ModelVersion.from_dict(version_dict)
        
        # Create model with same configuration
        model = create_tcn_model(version.training_config.model_config)
        
        # Load model state
        model_path = self.model_dir / f"{version_id}.pth"
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        logger.info(f"Loaded model version: {version_id}")
        
        return model, version


def create_training_pipeline(
    features: np.ndarray,
    model_config: Optional[TCNConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    model_dir: Path = Path("models")
) -> Tuple[EmbeddingTrainer, ModelVersion]:
    """
    Create and run complete training pipeline.
    
    Args:
        features: Feature array for training
        model_config: Model configuration
        training_config: Training configuration
        model_dir: Directory to save models
        
    Returns:
        Tuple of (trainer, model_version)
    """
    # Default configurations
    if model_config is None:
        model_config = TCNConfig(
            input_dim=features.shape[1],
            embedding_dim=128
        )
    
    if training_config is None:
        training_config = TrainingConfig(model_config=model_config)
    
    # Create trainer
    trainer = EmbeddingTrainer(training_config, model_dir)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(features)
    
    # Train model
    model_version = trainer.train(train_loader, val_loader, features)
    
    return trainer, model_version