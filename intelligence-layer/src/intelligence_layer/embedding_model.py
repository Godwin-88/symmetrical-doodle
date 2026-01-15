"""Temporal Convolutional Network for market embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TCNConfig:
    """Configuration for Temporal Convolutional Network."""
    input_dim: int = 64  # Number of input features
    embedding_dim: int = 128  # Output embedding dimension
    num_channels: List[int] = None  # Channel sizes for each layer
    kernel_size: int = 3  # Convolution kernel size
    dropout: float = 0.2  # Dropout rate
    activation: str = "relu"  # Activation function
    
    # Training parameters
    reconstruction_weight: float = 1.0  # Weight for reconstruction loss
    contrastive_weight: float = 0.5  # Weight for contrastive loss
    temporal_smoothness_weight: float = 0.1  # Weight for temporal smoothness
    
    def __post_init__(self):
        if self.num_channels is None:
            # Default architecture: progressively smaller channels
            self.num_channels = [256, 128, 64, self.embedding_dim]


class Chomp1d(nn.Module):
    """Chomp operation to ensure causal convolutions."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal block with dilated convolutions."""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super(TemporalBlock, self).__init__()
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution block
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""
    
    def __init__(self, config: TCNConfig):
        super(TemporalConvNet, self).__init__()
        self.config = config
        
        layers = []
        num_levels = len(config.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = config.input_dim if i == 0 else config.num_channels[i-1]
            out_channels = config.num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, config.kernel_size,
                stride=1, dilation=dilation_size,
                padding=(config.kernel_size - 1) * dilation_size,
                dropout=config.dropout,
                activation=config.activation
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, embedding_dim, sequence_length)
        """
        return self.network(x)


class MarketEmbeddingTCN(nn.Module):
    """Complete TCN model for market embedding with self-supervised learning."""
    
    def __init__(self, config: TCNConfig):
        super(MarketEmbeddingTCN, self).__init__()
        self.config = config
        
        # Encoder (TCN)
        self.encoder = TemporalConvNet(config)
        
        # Decoder for reconstruction
        decoder_channels = config.num_channels[::-1][1:] + [config.input_dim]  # Reverse and adjust
        decoder_config = TCNConfig(
            input_dim=config.embedding_dim,
            embedding_dim=config.input_dim,  # Reconstruct original features
            num_channels=decoder_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            activation=config.activation
        )
        self.decoder = TemporalConvNet(decoder_config)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim // 4)
        )
        
        # Global average pooling for sequence-level embeddings
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embeddings."""
        return self.encoder(x)
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to input space."""
        return self.decoder(embeddings)
    
    def get_sequence_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get single embedding vector for entire sequence."""
        embeddings = self.encode(x)  # (batch, embed_dim, seq_len)
        pooled = self.global_pool(embeddings)  # (batch, embed_dim, 1)
        return pooled.squeeze(-1)  # (batch, embed_dim)
    
    def get_contrastive_projection(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings for contrastive learning."""
        seq_embeddings = self.global_pool(embeddings).squeeze(-1)
        return self.projection_head(seq_embeddings)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all outputs for training.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)
            
        Returns:
            Dictionary containing:
            - embeddings: Temporal embeddings (batch, embed_dim, seq_len)
            - reconstruction: Reconstructed input (batch, input_dim, seq_len)
            - sequence_embedding: Single vector per sequence (batch, embed_dim)
            - contrastive_projection: Projection for contrastive loss (batch, proj_dim)
        """
        # Encode
        embeddings = self.encode(x)
        
        # Decode for reconstruction
        reconstruction = self.decode(embeddings)
        
        # Sequence-level embedding
        sequence_embedding = self.get_sequence_embedding(x)
        
        # Contrastive projection
        contrastive_projection = self.get_contrastive_projection(embeddings)
        
        return {
            'embeddings': embeddings,
            'reconstruction': reconstruction,
            'sequence_embedding': sequence_embedding,
            'contrastive_projection': contrastive_projection
        }


class EmbeddingLoss(nn.Module):
    """Multi-component loss for embedding training."""
    
    def __init__(self, config: TCNConfig, temperature: float = 0.1):
        super(EmbeddingLoss, self).__init__()
        self.config = config
        self.temperature = temperature
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def reconstruction_loss(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction loss (MSE)."""
        return self.mse_loss(reconstructed, original)
    
    def contrastive_loss(
        self, 
        projections: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Contrastive loss using InfoNCE.
        
        Args:
            projections: Projected embeddings (batch_size, proj_dim)
            positive_pairs: Optional positive pair indices
            
        Returns:
            Contrastive loss value
        """
        batch_size = projections.size(0)
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create positive mask (adjacent time windows are positive pairs)
        if positive_pairs is None:
            # Default: adjacent samples are positive pairs
            positive_mask = torch.zeros(batch_size, batch_size, device=projections.device)
            for i in range(batch_size - 1):
                positive_mask[i, i + 1] = 1
                positive_mask[i + 1, i] = 1
        else:
            positive_mask = positive_pairs
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=projections.device).bool()
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # Positive similarities
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        
        # InfoNCE loss
        loss = -torch.log(pos_sim / sum_exp_sim.squeeze()).mean()
        
        return loss
    
    def temporal_smoothness_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Temporal smoothness regularization.
        
        Args:
            embeddings: Temporal embeddings (batch, embed_dim, seq_len)
            
        Returns:
            Smoothness loss value
        """
        # Compute differences between consecutive time steps
        diff = embeddings[:, :, 1:] - embeddings[:, :, :-1]
        
        # L2 norm of differences
        smoothness_loss = torch.mean(torch.norm(diff, dim=1))
        
        return smoothness_loss
    
    def forward(
        self,
        original: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        positive_pairs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and components.
        
        Args:
            original: Original input tensor
            outputs: Model outputs dictionary
            positive_pairs: Optional positive pair indices for contrastive loss
            
        Returns:
            Dictionary with loss components and total loss
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(original, outputs['reconstruction'])
        
        # Contrastive loss
        contrast_loss = self.contrastive_loss(
            outputs['contrastive_projection'], positive_pairs
        )
        
        # Temporal smoothness loss
        smooth_loss = self.temporal_smoothness_loss(outputs['embeddings'])
        
        # Total weighted loss
        total_loss = (
            self.config.reconstruction_weight * recon_loss +
            self.config.contrastive_weight * contrast_loss +
            self.config.temporal_smoothness_weight * smooth_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'contrastive_loss': contrast_loss,
            'temporal_smoothness_loss': smooth_loss
        }


class EmbeddingValidator:
    """Validates embedding quality and training progress."""
    
    @staticmethod
    def temporal_continuity_score(embeddings: torch.Tensor) -> float:
        """
        Measure temporal continuity of embeddings.
        
        Args:
            embeddings: Temporal embeddings (batch, embed_dim, seq_len)
            
        Returns:
            Continuity score in [0, 1] range (higher is better)
        """
        # Handle edge cases
        if embeddings.size(2) < 2:
            return 1.0  # Single timestep is perfectly continuous
        
        # Check for zero embeddings (valid case)
        embedding_norms = torch.norm(embeddings, dim=1)  # (batch, seq_len)
        if torch.all(embedding_norms < 1e-8):
            return 1.0  # Zero embeddings are perfectly continuous
        
        # Compute cosine similarity between consecutive time steps
        # Use eps parameter to handle near-zero vectors
        embeddings_norm = F.normalize(embeddings, dim=1, eps=1e-8)
        
        similarities = []
        for i in range(embeddings.size(2) - 1):
            # Get consecutive embeddings
            emb_t = embeddings_norm[:, :, i]
            emb_t1 = embeddings_norm[:, :, i + 1]
            
            # Check if either embedding is effectively zero
            norm_t = torch.norm(emb_t, dim=1)
            norm_t1 = torch.norm(emb_t1, dim=1)
            
            # Only compute similarity for non-zero embeddings
            valid_mask = (norm_t > 1e-8) & (norm_t1 > 1e-8)
            
            if valid_mask.any():
                sim = F.cosine_similarity(emb_t[valid_mask], emb_t1[valid_mask], dim=1)
                # Cosine similarity is in [-1, 1], map to [0, 1]
                sim_normalized = (sim + 1.0) / 2.0
                similarities.append(sim_normalized.mean().item())
            else:
                # If both embeddings are zero, they are perfectly similar
                similarities.append(1.0)
        
        if not similarities:
            return 1.0
        
        # Return average similarity, already in [0, 1] range
        avg_similarity = np.mean(similarities)
        return max(0.0, min(1.0, avg_similarity))
    
    @staticmethod
    def embedding_diversity_score(embeddings: torch.Tensor) -> float:
        """
        Measure diversity of embeddings (avoid collapse).
        
        Args:
            embeddings: Sequence embeddings (batch, embed_dim)
            
        Returns:
            Diversity score (higher is better)
        """
        # Handle edge cases
        if embeddings.size(0) < 2:
            return 1.0  # Single embedding is maximally diverse
        
        # Check for zero embeddings
        embedding_norms = torch.norm(embeddings, dim=1)
        if torch.all(embedding_norms < 1e-8):
            return 1.0  # Zero embeddings are considered maximally diverse (no collapse)
        
        # Normalize embeddings with eps to handle near-zero vectors
        embeddings_norm = F.normalize(embeddings, dim=1, eps=1e-8)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(similarity_matrix.size(0), dtype=bool, device=embeddings.device)
        off_diagonal_sims = similarity_matrix[mask]
        
        if off_diagonal_sims.numel() == 0:
            return 1.0
        
        # Diversity is inverse of average similarity
        # Cosine similarity is in [-1, 1], so average similarity is also in [-1, 1]
        # Map to [0, 1] where 0 = identical embeddings, 1 = maximally diverse
        avg_similarity = off_diagonal_sims.mean().item()
        diversity = (1.0 - avg_similarity) / 2.0  # Map [-1, 1] to [0, 1]
        
        return max(0.0, min(1.0, diversity))
    
    @staticmethod
    def validate_embedding_quality(
        model: MarketEmbeddingTCN,
        validation_data: torch.Tensor,
        min_continuity: float = 0.7,
        min_diversity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Validate embedding quality against criteria.
        
        Args:
            model: Trained embedding model
            validation_data: Validation data tensor
            min_continuity: Minimum required continuity score
            min_diversity: Minimum required diversity score
            
        Returns:
            Validation results dictionary
        """
        model.eval()
        
        with torch.no_grad():
            outputs = model(validation_data)
            embeddings = outputs['embeddings']
            sequence_embeddings = outputs['sequence_embedding']
        
        # Compute quality metrics
        continuity = EmbeddingValidator.temporal_continuity_score(embeddings)
        diversity = EmbeddingValidator.embedding_diversity_score(sequence_embeddings)
        
        # Check criteria
        continuity_pass = continuity >= min_continuity
        diversity_pass = diversity >= min_diversity
        
        return {
            'continuity_score': continuity,
            'diversity_score': diversity,
            'continuity_pass': bool(continuity_pass),
            'diversity_pass': bool(diversity_pass),
            'overall_pass': bool(continuity_pass and diversity_pass),
            'validation_timestamp': datetime.now().isoformat(),
        }


def create_tcn_model(config: TCNConfig) -> MarketEmbeddingTCN:
    """
    Factory function to create TCN model.
    
    Args:
        config: TCN configuration
        
    Returns:
        Initialized TCN model
    """
    model = MarketEmbeddingTCN(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    logger.info(f"Created TCN model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def prepare_market_data_for_tcn(
    features: np.ndarray,
    sequence_length: int = 64,
    stride: int = 1
) -> torch.Tensor:
    """
    Prepare market feature data for TCN training.
    
    Args:
        features: Feature array (time_steps, n_features)
        sequence_length: Length of input sequences
        stride: Stride between sequences
        
    Returns:
        Tensor of shape (n_sequences, n_features, sequence_length)
    """
    n_timesteps, n_features = features.shape
    
    # Create sliding windows
    sequences = []
    for i in range(0, n_timesteps - sequence_length + 1, stride):
        sequence = features[i:i + sequence_length].T  # (n_features, seq_len)
        sequences.append(sequence)
    
    # Stack into tensor
    tensor = torch.FloatTensor(np.stack(sequences))  # (n_seq, n_features, seq_len)
    
    return tensor