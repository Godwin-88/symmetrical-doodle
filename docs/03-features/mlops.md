# MLOps - Machine Learning Operations

The MLOps module provides comprehensive ML lifecycle management for the algorithmic trading system, integrating MLflow for experiment tracking, model registry, and deployment orchestration.

## Overview

The MLOps system enables:
- **Experiment Tracking**: Track training runs, hyperparameters, and metrics
- **Model Registry**: Version and stage models (None/Staging/Production/Archived)
- **Model Serving**: Deploy models via REST endpoints
- **Model Monitoring**: Track model drift and performance degradation
- **Artifact Management**: Store and retrieve model artifacts
- **Run Comparison**: Compare multiple training runs side-by-side

## Architecture

```
+------------------------------------------------------------------+
|                         Frontend (React/TypeScript)               |
|  +-------------------------------------------------------------+ |
|  |                     MLOps.tsx (Enhanced)                     | |
|  |  +------------+ +------------+ +------------+ +------------+ | |
|  |  |  Registry  | |  Training  | | Deployment | | Monitoring | | |
|  |  |  + MLflow  | |  + MLflow  | |  + MLflow  | |  + MLflow  | | |
|  |  |  Artifacts | |  Tracking  | |  Serving   | |  Metrics   | | |
|  |  +------------+ +------------+ +------------+ +------------+ | |
|  |                                                               | |
|  |  [Collapsible Subcategory Panel]                             | |
|  |  +-- Experiments Browser                                      | |
|  |  +-- Run Comparison                                           | |
|  |  +-- Artifact Viewer                                          | |
|  |  +-- Model Lineage                                            | |
|  +-------------------------------------------------------------+ |
+--------------------------------+---------------------------------+
                                 | REST API
+--------------------------------v---------------------------------+
|                    Intelligence Layer (Python)                    |
|  +-------------------------------------------------------------+ |
|  |                    MLflow Integration Service                | |
|  |  +----------------+ +----------------+ +-------------------+ | |
|  |  | Experiment     | | Model Registry | | Model Serving     | | |
|  |  | Tracking       | | Integration    | | (MLflow Models)   | | |
|  |  | - Runs         | | - Versioning   | | - REST Endpoint   | | |
|  |  | - Metrics      | | - Stage Mgmt   | | - Batch Inference | | |
|  |  | - Parameters   | | - Lineage      | | - A/B Testing     | | |
|  |  | - Artifacts    | | - Transitions  | | - Shadow Mode     | | |
|  |  +----------------+ +----------------+ +-------------------+ | |
|  +-------------------------------------------------------------+ |
+--------------------------------+---------------------------------+
                                 | FFI (ctypes)
+--------------------------------v---------------------------------+
|                    Execution Core (Rust)                         |
|  +-------------------------------------------------------------+ |
|  |              MLflow Model Consumer (Read-Only)               | |
|  |  - Load serialized models from MLflow artifact store         | |
|  |  - ONNX runtime for model inference                          | |
|  |  - Model version tracking via config                         | |
|  +-------------------------------------------------------------+ |
+------------------------------------------------------------------+
                                 |
+--------------------------------v---------------------------------+
|                         MLflow Server                            |
|  +----------------+ +----------------+ +-----------------------+ |
|  | Tracking Server| | Model Registry | | Artifact Store        | |
|  | (PostgreSQL)   | | (PostgreSQL)   | | (S3/MinIO/Local)      | |
|  +----------------+ +----------------+ +-----------------------+ |
+------------------------------------------------------------------+
```

## UI Components

### MLOps Tab (F3)

The MLOps tab provides a comprehensive interface for ML operations:

#### Collapsible Subcategory Panel

The left panel provides quick navigation to MLflow resources:
- **Experiments**: Browse and select MLflow experiments
- **Runs**: View runs for selected experiment, multi-select for comparison
- **Models**: View registered models and their versions/stages
- **Serving**: View active model serving endpoints

#### Registry Tab

- **Training Datasets**: Manage datasets for model training
- **Available Models**: Browse 25+ financial ML models
- **Run Comparison Chart**: Compare selected runs side-by-side
- **Model Registry**: View registered models with stage transitions

#### Training Tab

- **Active Training Jobs**: Monitor running training jobs
- **Training Configuration**: Configure hyperparameters
- **Real-time Metrics**: Live updating loss/accuracy curves

#### Deployment Tab

- **Production Models**: View deployed models
- **Deployment Configuration**: Configure replicas, resources
- **Model Serving Status**: Monitor endpoint health

#### Monitoring Tab

- **Model Alerts**: View and acknowledge alerts
- **Validation Metrics**: Track model quality metrics
- **Drift Monitoring**: Monitor data and concept drift

## API Endpoints

### MLflow Integration

```
GET  /api/v1/mlflow/status              # Get MLflow status
GET  /api/v1/mlflow/experiments         # List experiments
POST /api/v1/mlflow/experiments         # Create experiment
GET  /api/v1/mlflow/experiments/{name}  # Get experiment
GET  /api/v1/mlflow/experiments/{name}/runs  # List runs
GET  /api/v1/mlflow/runs/{id}           # Get run details
GET  /api/v1/mlflow/runs/{id}/metrics   # Get run metrics
GET  /api/v1/mlflow/runs/{id}/artifacts # List artifacts
POST /api/v1/mlflow/runs/compare        # Compare runs
GET  /api/v1/mlflow/models              # List registered models
GET  /api/v1/mlflow/models/{name}/latest # Get latest version
POST /api/v1/mlflow/models/transition   # Transition model stage
GET  /api/v1/mlflow/serving/endpoints   # List serving endpoints
POST /api/v1/mlflow/serving/deploy      # Deploy model
POST /api/v1/mlflow/serve/{model}       # Make prediction
```

### Existing MLOps

```
GET  /mlops/datasets                    # List datasets
POST /mlops/datasets                    # Create dataset
GET  /mlops/training/jobs               # List training jobs
POST /mlops/training/start              # Start training
POST /mlops/training/{id}/pause         # Pause training
POST /mlops/training/{id}/stop          # Stop training
GET  /mlops/models                      # List deployed models
POST /mlops/deploy                      # Deploy model
GET  /mlops/metrics/validation          # Get validation metrics
GET  /mlops/alerts                      # Get model alerts
```

## Model Registry Workflow

### Model Stages

1. **None**: Initial state after registration
2. **Staging**: Model under testing/validation
3. **Production**: Model serving live traffic
4. **Archived**: Deprecated model version

### Stage Transitions

```python
# Promote model to staging
await transition_model_stage("regime_detector", "3", "Staging")

# Promote to production (archives existing production version)
await transition_model_stage("regime_detector", "3", "Production")

# Archive old version
await transition_model_stage("regime_detector", "2", "Archived")
```

## Supported Models

The system includes 25+ financial ML models organized by category:

### Market Embedding
- MarketEmbeddingTCN
- VAE Market Embeddings
- Transformer Embeddings

### Regime Detection
- HMM Regime Detector
- TCN Regime Classifier
- Ensemble Regime Model

### Strategy Allocation
- RL Strategy Allocator
- Meta-Controller
- Portfolio Optimizer

### Feature Extraction
- Technical Feature Extractor
- Microstructure Feature Extractor
- Sentiment Feature Extractor

### Volatility Modeling
- GARCH Volatility
- LSTM Volatility Predictor
- Stochastic Vol Model

### Risk Metrics
- VaR Calculator
- CVaR Model
- Drawdown Predictor

## ONNX Model Inference (Rust)

The Rust execution core loads ONNX models for low-latency inference:

```rust
use execution_core::ml_inference::{InferenceEngine, ModelsConfig};

// Load configuration
let config = ModelsConfig::load_default()?;
let engine = InferenceEngine::new(config, PathBuf::from("models"));

// Load model
engine.load_model("regime_detector").await?;

// Run inference
let result = engine.infer("regime_detector", &features, &shape).await?;
```

### Model Configuration

Models are configured in `execution-core/config/models.toml`:

```toml
[models.regime_detector]
name = "regime_detector"
version = "3"
artifact_path = "models/regime_detector/v3/model.onnx"
mlflow_run_id = "abc123"
input_names = ["input"]
output_names = ["output"]
input_shape = [-1, 127, 64]
```

## Training Integration

Training with MLflow tracking:

```python
from intelligence_layer.mlflow_service import get_mlflow_service

service = get_mlflow_service()

# Start run
with service.run_context("financial-models", "tcn-training-v3") as run_id:
    # Log hyperparameters
    service.log_params({
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100
    })

    # Train model
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log metrics
        service.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # Log model
    service.log_pytorch_model(
        model,
        registered_model_name="regime_detector"
    )
```

## Configuration

### Environment Variables

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# Feature Flags
MLFLOW_AUTO_LOG_PYTORCH=true
MLFLOW_LOG_SYSTEM_METRICS=true
```

### Docker Compose

Start MLflow infrastructure:

```bash
# Start MLflow server, PostgreSQL, and MinIO
docker-compose -f docker-compose.mlflow.yml up -d

# Access MLflow UI
open http://localhost:5000

# Access MinIO Console
open http://localhost:9001
```

## Monitoring and Alerting

### Model Drift Detection

The system monitors:
- **Data Drift**: Feature distribution changes
- **Concept Drift**: Relationship changes between features and targets
- **Performance Drift**: Model accuracy degradation

### Alert Types

- **DRIFT**: Model drift detected above threshold
- **PERFORMANCE**: Model performance below threshold
- **ERROR**: Model error rate above threshold
- **WARNING**: General model health warnings

### Validation Metrics

- **Temporal Continuity**: Embedding temporal smoothness
- **Regime Separability**: Regime classification quality
- **Feature Stability**: Feature distribution stability
- **Prediction Consistency**: Prediction variance
- **Drift Detection**: Overall drift score

---

**Last Updated**: February 2026
**Version**: 1.0.0
