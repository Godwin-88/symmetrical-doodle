"""
Model Registry for Financial Deep Learning Models.

Provides a comprehensive catalog of production-grade models for various financial tasks,
with configuration, training, and inference capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from datetime import datetime


class ModelCategory(str, Enum):
    """Model category classification."""
    TIME_SERIES = "time_series"
    REPRESENTATION = "representation"
    GRAPH = "graph"
    REINFORCEMENT = "reinforcement"
    NLP = "nlp"
    TABULAR = "tabular"


class UseCase(str, Enum):
    """Financial use cases."""
    PRICE_FORECASTING = "price_forecasting"
    VOLATILITY_FORECASTING = "volatility_forecasting"
    REGIME_DETECTION = "regime_detection"
    CROSS_ASSET_EFFECTS = "cross_asset_effects"
    TRADING_EXECUTION = "trading_execution"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    FRAUD_DETECTION = "fraud_detection"
    CREDIT_RISK = "credit_risk"
    NEWS_ANALYSIS = "news_analysis"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class ModelSpec:
    """Specification for a financial ML model."""
    id: str
    name: str
    category: ModelCategory
    use_cases: List[UseCase]
    description: str
    strengths: List[str]
    weaknesses: List[str]
    best_for: List[str]
    production_ready: bool
    latency_class: Literal["low", "medium", "high"]  # Inference latency
    data_requirements: Literal["small", "medium", "large"]
    explainability: Literal["low", "medium", "high"]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    # Training configuration
    min_samples: int = 1000
    recommended_samples: int = 10000
    supports_online_learning: bool = False
    supports_transfer_learning: bool = False
    
    # Deployment
    gpu_required: bool = False
    memory_mb: int = 512
    
    # Metadata
    paper_url: Optional[str] = None
    implementation_url: Optional[str] = None


# ============================================================================
# 1. TIME-SERIES FORECASTING & MARKET PREDICTION
# ============================================================================

TIME_SERIES_MODELS = [
    ModelSpec(
        id="tft",
        name="Temporal Fusion Transformer",
        category=ModelCategory.TIME_SERIES,
        use_cases=[UseCase.PRICE_FORECASTING, UseCase.VOLATILITY_FORECASTING],
        description="State-of-the-art multi-horizon forecasting with attention mechanisms",
        strengths=[
            "Handles long-range dependencies",
            "Attention provides interpretability",
            "Scales across thousands of instruments",
            "Multi-horizon forecasting",
            "Regime-aware prediction"
        ],
        weaknesses=[
            "Data-hungry (needs large datasets)",
            "Overfits noisy price series",
            "Expensive inference for low-latency trading"
        ],
        best_for=[
            "Portfolio forecasting",
            "Risk modeling",
            "Volatility surfaces",
            "Factor return forecasting"
        ],
        production_ready=True,
        latency_class="high",
        data_requirements="large",
        explainability="medium",
        hyperparameters={
            "hidden_size": 128,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_encoder_length": 168,
            "max_prediction_length": 24
        },
        min_samples=10000,
        recommended_samples=100000,
        gpu_required=True,
        memory_mb=2048,
        paper_url="https://arxiv.org/abs/1912.09363"
    ),
    
    ModelSpec(
        id="informer",
        name="Informer",
        category=ModelCategory.TIME_SERIES,
        use_cases=[UseCase.PRICE_FORECASTING, UseCase.VOLATILITY_FORECASTING],
        description="Efficient transformer for long sequence time-series forecasting",
        strengths=[
            "Efficient for long sequences",
            "Lower memory footprint than TFT",
            "Good for macro forecasting"
        ],
        weaknesses=[
            "Less interpretable than TFT",
            "Requires careful tuning"
        ],
        best_for=[
            "Long-horizon forecasting",
            "Macro indicators",
            "Multi-asset prediction"
        ],
        production_ready=True,
        latency_class="medium",
        data_requirements="large",
        explainability="low",
        hyperparameters={
            "d_model": 512,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 2048,
            "dropout": 0.05,
            "factor": 5
        },
        min_samples=5000,
        recommended_samples=50000,
        gpu_required=True,
        memory_mb=1536
    ),
    
    ModelSpec(
        id="patchtst",
        name="PatchTST",
        category=ModelCategory.TIME_SERIES,
        use_cases=[UseCase.PRICE_FORECASTING, UseCase.VOLATILITY_FORECASTING],
        description="Patch-based transformer for efficient time series forecasting",
        strengths=[
            "Very efficient",
            "Good performance with less data",
            "Fast inference"
        ],
        weaknesses=[
            "Newer model, less battle-tested",
            "Limited interpretability"
        ],
        best_for=[
            "Medium-frequency trading",
            "Resource-constrained environments",
            "Multi-variate forecasting"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="medium",
        explainability="low",
        hyperparameters={
            "patch_len": 16,
            "stride": 8,
            "d_model": 128,
            "n_heads": 16,
            "d_ff": 256,
            "dropout": 0.2
        },
        min_samples=2000,
        recommended_samples=20000,
        gpu_required=False,
        memory_mb=512
    ),
    
    ModelSpec(
        id="lstm",
        name="LSTM (Long Short-Term Memory)",
        category=ModelCategory.TIME_SERIES,
        use_cases=[UseCase.PRICE_FORECASTING, UseCase.CREDIT_RISK],
        description="Classic recurrent network, still relevant for many tasks",
        strengths=[
            "Stable training",
            "Low inference cost",
            "Works well with engineered features",
            "Battle-tested in production"
        ],
        weaknesses=[
            "Weak regime adaptation",
            "Poor long-range memory",
            "Sequential processing limits parallelization"
        ],
        best_for=[
            "Medium-frequency signals",
            "Small to mid-sized datasets",
            "Latency-sensitive systems",
            "Credit risk",
            "Fraud detection"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="small",
        explainability="low",
        hyperparameters={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
            "learning_rate": 0.001
        },
        min_samples=500,
        recommended_samples=5000,
        supports_online_learning=True,
        gpu_required=False,
        memory_mb=256
    ),
    
    ModelSpec(
        id="gru",
        name="GRU (Gated Recurrent Unit)",
        category=ModelCategory.TIME_SERIES,
        use_cases=[UseCase.PRICE_FORECASTING, UseCase.FRAUD_DETECTION],
        description="Simplified LSTM variant with fewer parameters",
        strengths=[
            "Faster training than LSTM",
            "Lower memory footprint",
            "Good for real-time systems"
        ],
        weaknesses=[
            "Similar limitations to LSTM",
            "Less expressive than LSTM"
        ],
        best_for=[
            "Intraday signal generation",
            "Real-time fraud detection",
            "Resource-constrained deployment"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="small",
        explainability="low",
        hyperparameters={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001
        },
        min_samples=500,
        recommended_samples=5000,
        supports_online_learning=True,
        gpu_required=False,
        memory_mb=200
    ),
]

# ============================================================================
# 2. REPRESENTATION LEARNING & FEATURE EXTRACTION
# ============================================================================

REPRESENTATION_MODELS = [
    ModelSpec(
        id="vae",
        name="Variational Autoencoder",
        category=ModelCategory.REPRESENTATION,
        use_cases=[UseCase.REGIME_DETECTION, UseCase.ANOMALY_DETECTION],
        description="Probabilistic autoencoder for latent representation learning",
        strengths=[
            "Latent factor discovery",
            "Noise reduction",
            "Regime compression",
            "Generates new samples",
            "Uncertainty quantification"
        ],
        weaknesses=[
            "Training can be unstable",
            "Requires careful architecture design",
            "Latent space interpretation not always clear"
        ],
        best_for=[
            "Alpha factor construction",
            "Anomaly detection",
            "Market state embeddings",
            "Regime clustering"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="medium",
        explainability="medium",
        hyperparameters={
            "latent_dim": 32,
            "encoder_layers": [128, 64],
            "decoder_layers": [64, 128],
            "beta": 1.0,  # KL divergence weight
            "learning_rate": 0.001
        },
        min_samples=1000,
        recommended_samples=10000,
        gpu_required=False,
        memory_mb=512
    ),
    
    ModelSpec(
        id="denoising_ae",
        name="Denoising Autoencoder",
        category=ModelCategory.REPRESENTATION,
        use_cases=[UseCase.ANOMALY_DETECTION],
        description="Autoencoder trained to reconstruct clean data from noisy input",
        strengths=[
            "Robust to noise",
            "Good for feature extraction",
            "Simple and stable training"
        ],
        weaknesses=[
            "No probabilistic interpretation",
            "Requires noise model design"
        ],
        best_for=[
            "Noisy market data",
            "Feature denoising",
            "Outlier detection"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="medium",
        explainability="low",
        hyperparameters={
            "hidden_dims": [128, 64, 32],
            "noise_factor": 0.2,
            "learning_rate": 0.001
        },
        min_samples=1000,
        recommended_samples=10000,
        gpu_required=False,
        memory_mb=384
    ),
    
    ModelSpec(
        id="contrastive",
        name="Contrastive Learning (SimCLR-style)",
        category=ModelCategory.REPRESENTATION,
        use_cases=[UseCase.REGIME_DETECTION],
        description="Self-supervised learning via contrastive objectives",
        strengths=[
            "Works with unlabeled data",
            "Learns robust representations",
            "Good for regime separation"
        ],
        weaknesses=[
            "Requires careful augmentation design",
            "Computationally expensive",
            "Needs large batch sizes"
        ],
        best_for=[
            "Label-scarce scenarios",
            "Regime clustering",
            "Instrument similarity"
        ],
        production_ready=True,
        latency_class="medium",
        data_requirements="large",
        explainability="low",
        hyperparameters={
            "projection_dim": 128,
            "temperature": 0.5,
            "batch_size": 256,
            "learning_rate": 0.001
        },
        min_samples=5000,
        recommended_samples=50000,
        gpu_required=True,
        memory_mb=1024
    ),
]

# ============================================================================
# 3. GRAPH NEURAL NETWORKS
# ============================================================================

GRAPH_MODELS = [
    ModelSpec(
        id="gcn",
        name="Graph Convolutional Network",
        category=ModelCategory.GRAPH,
        use_cases=[UseCase.CROSS_ASSET_EFFECTS, UseCase.REGIME_DETECTION],
        description="Convolutional operations on graph-structured data",
        strengths=[
            "Captures network effects",
            "Spillover risk modeling",
            "Cross-asset reasoning",
            "Interpretable via attention"
        ],
        weaknesses=[
            "Requires graph construction",
            "Sensitive to graph quality",
            "Over-smoothing in deep networks"
        ],
        best_for=[
            "Correlation networks",
            "Sector analysis",
            "Contagion modeling",
            "Supply chain risk"
        ],
        production_ready=True,
        latency_class="medium",
        data_requirements="medium",
        explainability="medium",
        hyperparameters={
            "hidden_channels": 64,
            "num_layers": 3,
            "dropout": 0.5,
            "learning_rate": 0.01
        },
        min_samples=500,
        recommended_samples=5000,
        gpu_required=False,
        memory_mb=512
    ),
    
    ModelSpec(
        id="gat",
        name="Graph Attention Network",
        category=ModelCategory.GRAPH,
        use_cases=[UseCase.CROSS_ASSET_EFFECTS],
        description="GNN with attention mechanisms for adaptive aggregation",
        strengths=[
            "Learns edge importance",
            "More expressive than GCN",
            "Better interpretability"
        ],
        weaknesses=[
            "More parameters than GCN",
            "Slower training"
        ],
        best_for=[
            "Dynamic correlation networks",
            "Heterogeneous graphs",
            "Systemic risk"
        ],
        production_ready=True,
        latency_class="medium",
        data_requirements="medium",
        explainability="high",
        hyperparameters={
            "hidden_channels": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.6,
            "learning_rate": 0.005
        },
        min_samples=500,
        recommended_samples=5000,
        gpu_required=False,
        memory_mb=768
    ),
    
    ModelSpec(
        id="temporal_gnn",
        name="Temporal Graph Neural Network",
        category=ModelCategory.GRAPH,
        use_cases=[UseCase.CROSS_ASSET_EFFECTS, UseCase.REGIME_DETECTION],
        description="GNN for time-evolving graphs",
        strengths=[
            "Handles dynamic networks",
            "Temporal + spatial reasoning",
            "Regime transition modeling"
        ],
        weaknesses=[
            "Complex architecture",
            "High computational cost",
            "Requires temporal graph data"
        ],
        best_for=[
            "Time-varying correlations",
            "Regime transitions",
            "Flow-of-funds analysis"
        ],
        production_ready=False,  # Research stage
        latency_class="high",
        data_requirements="large",
        explainability="medium",
        hyperparameters={
            "hidden_channels": 64,
            "num_layers": 3,
            "temporal_window": 10,
            "learning_rate": 0.001
        },
        min_samples=2000,
        recommended_samples=20000,
        gpu_required=True,
        memory_mb=1536
    ),
]

# ============================================================================
# 4. REINFORCEMENT LEARNING
# ============================================================================

RL_MODELS = [
    ModelSpec(
        id="ppo",
        name="Proximal Policy Optimization",
        category=ModelCategory.REINFORCEMENT,
        use_cases=[UseCase.TRADING_EXECUTION, UseCase.PORTFOLIO_ALLOCATION],
        description="Stable policy gradient method for continuous control",
        strengths=[
            "Stable training",
            "Good sample efficiency",
            "Works well with continuous actions",
            "Industry standard"
        ],
        weaknesses=[
            "Reward hacking risk",
            "Non-stationarity challenges",
            "Simulation bias"
        ],
        best_for=[
            "Execution optimization",
            "Market making",
            "Dynamic allocation"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="large",
        explainability="low",
        hyperparameters={
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_range": 0.2,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10
        },
        min_samples=10000,
        recommended_samples=100000,
        supports_online_learning=True,
        gpu_required=False,
        memory_mb=512
    ),
    
    ModelSpec(
        id="sac",
        name="Soft Actor-Critic",
        category=ModelCategory.REINFORCEMENT,
        use_cases=[UseCase.TRADING_EXECUTION],
        description="Off-policy RL with maximum entropy objective",
        strengths=[
            "Sample efficient",
            "Stable training",
            "Exploration via entropy",
            "Off-policy learning"
        ],
        weaknesses=[
            "Hyperparameter sensitive",
            "Requires replay buffer"
        ],
        best_for=[
            "Market making",
            "Execution",
            "Continuous control"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="large",
        explainability="low",
        hyperparameters={
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 1000000,
            "batch_size": 256
        },
        min_samples=10000,
        recommended_samples=100000,
        supports_online_learning=True,
        gpu_required=False,
        memory_mb=1024
    ),
]

# ============================================================================
# 5. NLP MODELS
# ============================================================================

NLP_MODELS = [
    ModelSpec(
        id="finbert",
        name="FinBERT",
        category=ModelCategory.NLP,
        use_cases=[UseCase.NEWS_ANALYSIS],
        description="BERT fine-tuned on financial text",
        strengths=[
            "Domain-specific vocabulary",
            "Good sentiment analysis",
            "Pre-trained on financial corpus"
        ],
        weaknesses=[
            "Limited context window",
            "Requires fine-tuning",
            "Computationally expensive"
        ],
        best_for=[
            "Earnings calls",
            "SEC filings",
            "News sentiment",
            "Event detection"
        ],
        production_ready=True,
        latency_class="medium",
        data_requirements="medium",
        explainability="medium",
        hyperparameters={
            "max_length": 512,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3
        },
        min_samples=1000,
        recommended_samples=10000,
        supports_transfer_learning=True,
        gpu_required=True,
        memory_mb=2048
    ),
    
    ModelSpec(
        id="longformer",
        name="Longformer",
        category=ModelCategory.NLP,
        use_cases=[UseCase.NEWS_ANALYSIS],
        description="Transformer for long documents",
        strengths=[
            "Handles long documents",
            "Efficient attention",
            "Good for filings"
        ],
        weaknesses=[
            "Memory intensive",
            "Slower than BERT"
        ],
        best_for=[
            "SEC filings",
            "Research reports",
            "Long-form analysis"
        ],
        production_ready=True,
        latency_class="high",
        data_requirements="large",
        explainability="medium",
        hyperparameters={
            "max_length": 4096,
            "attention_window": 512,
            "learning_rate": 3e-5,
            "batch_size": 4
        },
        min_samples=500,
        recommended_samples=5000,
        supports_transfer_learning=True,
        gpu_required=True,
        memory_mb=4096
    ),
]

# ============================================================================
# 6. TABULAR MODELS
# ============================================================================

TABULAR_MODELS = [
    ModelSpec(
        id="tabnet",
        name="TabNet",
        category=ModelCategory.TABULAR,
        use_cases=[UseCase.CREDIT_RISK, UseCase.FRAUD_DETECTION],
        description="Attention-based tabular learning",
        strengths=[
            "Interpretable feature selection",
            "Works well on tabular data",
            "No feature engineering needed",
            "Competitive with XGBoost"
        ],
        weaknesses=[
            "Slower than tree models",
            "Requires tuning"
        ],
        best_for=[
            "Credit scoring",
            "AML",
            "Fraud detection",
            "Structured data"
        ],
        production_ready=True,
        latency_class="low",
        data_requirements="medium",
        explainability="high",
        hyperparameters={
            "n_d": 64,
            "n_a": 64,
            "n_steps": 5,
            "gamma": 1.5,
            "lambda_sparse": 0.001,
            "learning_rate": 0.02
        },
        min_samples=1000,
        recommended_samples=10000,
        gpu_required=False,
        memory_mb=512
    ),
    
    ModelSpec(
        id="ft_transformer",
        name="FT-Transformer",
        category=ModelCategory.TABULAR,
        use_cases=[UseCase.CREDIT_RISK, UseCase.FRAUD_DETECTION],
        description="Feature Tokenizer + Transformer for tabular data",
        strengths=[
            "State-of-the-art on tabular",
            "Handles mixed data types",
            "Good generalization"
        ],
        weaknesses=[
            "Slower than TabNet",
            "More parameters"
        ],
        best_for=[
            "Complex tabular tasks",
            "Mixed data types",
            "High-stakes decisions"
        ],
        production_ready=True,
        latency_class="medium",
        data_requirements="medium",
        explainability="medium",
        hyperparameters={
            "d_token": 192,
            "n_blocks": 3,
            "attention_dropout": 0.2,
            "ffn_dropout": 0.1,
            "learning_rate": 0.0001
        },
        min_samples=1000,
        recommended_samples=10000,
        gpu_required=False,
        memory_mb=768
    ),
]

# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelRegistry:
    """Central registry for all financial ML models."""
    
    def __init__(self):
        self.models: Dict[str, ModelSpec] = {}
        self._register_all_models()
    
    def _register_all_models(self):
        """Register all models."""
        all_models = (
            TIME_SERIES_MODELS +
            REPRESENTATION_MODELS +
            GRAPH_MODELS +
            RL_MODELS +
            NLP_MODELS +
            TABULAR_MODELS
        )
        
        for model in all_models:
            self.models[model.id] = model
    
    def get_model(self, model_id: str) -> Optional[ModelSpec]:
        """Get model by ID."""
        return self.models.get(model_id)
    
    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        use_case: Optional[UseCase] = None,
        production_ready: Optional[bool] = None,
    ) -> List[ModelSpec]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if category:
            models = [m for m in models if m.category == category]
        
        if use_case:
            models = [m for m in models if use_case in m.use_cases]
        
        if production_ready is not None:
            models = [m for m in models if m.production_ready == production_ready]
        
        return models
    
    def get_recommended_models(self, use_case: UseCase) -> List[ModelSpec]:
        """Get recommended models for a use case, sorted by suitability."""
        models = self.list_models(use_case=use_case, production_ready=True)
        
        # Sort by data requirements (prefer models that work with less data)
        data_req_order = {"small": 0, "medium": 1, "large": 2}
        models.sort(key=lambda m: data_req_order.get(m.data_requirements, 999))
        
        return models
    
    def get_model_categories(self) -> List[str]:
        """Get all model categories."""
        return [c.value for c in ModelCategory]
    
    def get_use_cases(self) -> List[str]:
        """Get all use cases."""
        return [u.value for u in UseCase]


# Global registry instance
model_registry = ModelRegistry()
