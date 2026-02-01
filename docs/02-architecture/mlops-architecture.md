# MLOps Architecture

This document describes the MLOps architecture for the algorithmic trading system, including MLflow integration, model lifecycle management, and deployment workflows.

## System Overview

```mermaid
graph TB
    subgraph Frontend["Frontend (React/TypeScript)"]
        UI[MLOps.tsx]
        UI --> Registry[Registry Tab]
        UI --> Training[Training Tab]
        UI --> Deploy[Deployment Tab]
        UI --> Monitor[Monitoring Tab]
        UI --> SubPanel[Collapsible Panel]
        SubPanel --> Experiments
        SubPanel --> Runs
        SubPanel --> Models
        SubPanel --> Serving
    end

    subgraph API["REST API Layer"]
        MLflowRoutes[MLflow Routes]
        MLOpsRoutes[MLOps Routes]
    end

    subgraph Intelligence["Intelligence Layer (Python)"]
        MLflowService[MLflow Service]
        TrainingProtocol[Training Protocol]
        ExperimentConfig[Experiment Config]
        ModelRegistry[Model Registry]
    end

    subgraph Execution["Execution Core (Rust)"]
        InferenceEngine[Inference Engine]
        ONNXRuntime[ONNX Runtime]
        ModelConfig[Model Config]
    end

    subgraph MLflow["MLflow Infrastructure"]
        TrackingServer[Tracking Server]
        RegistryDB[(Model Registry)]
        ArtifactStore[(Artifact Store)]
    end

    subgraph Storage["Storage Layer"]
        PostgreSQL[(PostgreSQL)]
        MinIO[(MinIO/S3)]
    end

    UI --> API
    API --> Intelligence
    Intelligence --> MLflow
    MLflow --> Storage
    Intelligence --> Execution
    Execution --> Storage
```

## MLflow Integration Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant MLflowService
    participant TrackingServer
    participant ArtifactStore

    User->>Frontend: Start Training
    Frontend->>API: POST /mlflow/runs/start
    API->>MLflowService: start_run()
    MLflowService->>TrackingServer: Create Run
    TrackingServer-->>MLflowService: Run ID

    loop Training Epochs
        MLflowService->>TrackingServer: Log Metrics
        MLflowService->>TrackingServer: Log Parameters
    end

    MLflowService->>ArtifactStore: Log Model
    MLflowService->>TrackingServer: End Run
    TrackingServer-->>Frontend: Run Complete
```

## Model Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> None: Model Registered
    None --> Staging: Promote to Staging
    Staging --> Production: Promote to Production
    Staging --> None: Demote
    Production --> Archived: Archive
    Archived --> Staging: Restore
    Production --> Staging: Rollback
    None --> Archived: Archive
```

## Training Pipeline

```mermaid
flowchart LR
    subgraph Data["Data Preparation"]
        Dataset[(Dataset)]
        Features[Feature Extraction]
        Split[Train/Val/Test Split]
    end

    subgraph Training["Model Training"]
        Config[Training Config]
        Train[Training Loop]
        Validate[Validation]
        Checkpoint[Checkpointing]
    end

    subgraph MLflow["MLflow Tracking"]
        Params[Log Parameters]
        Metrics[Log Metrics]
        Artifacts[Log Artifacts]
        Register[Register Model]
    end

    subgraph Deploy["Deployment"]
        Stage[Staging]
        Test[A/B Testing]
        Prod[Production]
    end

    Dataset --> Features --> Split
    Split --> Config --> Train
    Train --> Validate --> Checkpoint
    Train --> Params
    Validate --> Metrics
    Checkpoint --> Artifacts
    Artifacts --> Register
    Register --> Stage --> Test --> Prod
```

## Component Architecture

### Frontend Components

```
MLOps.tsx
+-- Header (MLOPS - MACHINE LEARNING OPERATIONS)
+-- Tab Navigation
|   +-- Registry Tab
|   +-- Training Tab
|   +-- Deployment Tab
|   +-- Monitoring Tab
+-- Collapsible Subcategory Panel (left)
|   +-- MLflow Status Indicator
|   +-- Experiments Browser
|   +-- Runs List (with multi-select)
|   +-- Registered Models
|   +-- Serving Endpoints
+-- Main Content Area (right)
    +-- Run Comparison Chart
    +-- Tab-specific content
    +-- Modals (Dataset, Training, Deploy)
```

### Backend Services

```
intelligence_layer/
+-- mlflow_service.py
|   +-- MLflowService (main service)
|   +-- MLflowConfig (configuration)
|   +-- ExperimentInfo, RunInfo (data classes)
|   +-- ModelStage, RunStatus (enums)
+-- mlflow_routes.py
|   +-- /experiments endpoints
|   +-- /runs endpoints
|   +-- /models endpoints
|   +-- /serving endpoints
+-- training_protocol.py
|   +-- EmbeddingTrainer (with MLflow logging)
|   +-- TrainingConfig
|   +-- ModelVersion
```

### Rust Inference Engine

```
execution_core/
+-- ml_inference.rs
|   +-- InferenceEngine (ONNX model manager)
|   +-- ModelsConfig (TOML configuration)
|   +-- CachedModel (model wrapper)
|   +-- RegimeDetector (helper)
+-- config/
    +-- models.toml (model configurations)
```

## Data Flow

### Experiment Tracking

```
+-------------+    +-------------+    +-------------+
|   Python    |    |   MLflow    |    | PostgreSQL  |
|   Client    |--->|   Server    |--->|  Database   |
+-------------+    +-------------+    +-------------+
      |                  |
      |                  v
      |           +-------------+
      +---------->|   MinIO     |
                  | (Artifacts) |
                  +-------------+
```

### Model Serving

```
+-------------+    +-------------+    +-------------+
|   Request   |    | Intelligence|    |  Execution  |
|   (REST)    |--->|    Layer    |--->|    Core     |
+-------------+    +-------------+    +-------------+
                         |                  |
                         v                  v
                  +-------------+    +-------------+
                  |   MLflow    |    |    ONNX     |
                  |   Models    |    |   Runtime   |
                  +-------------+    +-------------+
```

## Infrastructure

### Docker Services

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| mlflow-server | ghcr.io/mlflow/mlflow:v2.10.0 | 5000 | Tracking & Registry |
| mlflow-db | postgres:15-alpine | 5433 | Metadata Storage |
| mlflow-minio | minio/minio:latest | 9000, 9001 | Artifact Storage |
| mlflow-minio-init | minio/mc:latest | - | Bucket Creation |

### Network Architecture

```
+--------------------------------------------------+
|                  mlflow-network                   |
|                                                  |
|  +------------+  +------------+  +------------+  |
|  |  mlflow-   |  |  mlflow-   |  |  mlflow-   |  |
|  |  server    |  |  db        |  |  minio     |  |
|  |  :5000     |  |  :5432     |  |  :9000     |  |
|  +------------+  +------------+  +------------+  |
|                                                  |
+--------------------------------------------------+
         |              |              |
         v              v              v
    External       Internal       External
    (UI, API)      (Docker)      (Console)
```

## Monitoring Architecture

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        Predictions[Model Predictions]
        Features[Input Features]
        Performance[Model Performance]
    end

    subgraph Detection["Drift Detection"]
        DataDrift[Data Drift Monitor]
        ConceptDrift[Concept Drift Monitor]
        PerformanceDrift[Performance Monitor]
    end

    subgraph Alerts["Alert System"]
        AlertEngine[Alert Engine]
        Thresholds[Threshold Config]
        Notifications[Notifications]
    end

    subgraph Actions["Automated Actions"]
        Retrain[Auto-Retrain]
        Rollback[Auto-Rollback]
        Scale[Auto-Scale]
    end

    Predictions --> DataDrift
    Features --> DataDrift
    Performance --> PerformanceDrift
    Predictions --> ConceptDrift

    DataDrift --> AlertEngine
    ConceptDrift --> AlertEngine
    PerformanceDrift --> AlertEngine

    AlertEngine --> Thresholds
    Thresholds --> Notifications
    Thresholds --> Actions
```

## Security Architecture

```
+--------------------------------------------------+
|                  Security Layers                  |
+--------------------------------------------------+
|                                                  |
|  +------------------+  +------------------+      |
|  | Authentication   |  | Authorization    |      |
|  | (MLflow Auth)    |  | (Role-based)     |      |
|  +------------------+  +------------------+      |
|                                                  |
|  +------------------+  +------------------+      |
|  | TLS/SSL          |  | Network          |      |
|  | (All Services)   |  | (Isolation)      |      |
|  +------------------+  +------------------+      |
|                                                  |
|  +------------------+  +------------------+      |
|  | Secrets Mgmt     |  | Audit Logging    |      |
|  | (Env/Vault)      |  | (All Actions)    |      |
|  +------------------+  +------------------+      |
|                                                  |
+--------------------------------------------------+
```

## Scaling Considerations

### Horizontal Scaling

- **MLflow Server**: Stateless, can be scaled behind load balancer
- **PostgreSQL**: Primary-replica setup for read scaling
- **MinIO**: Distributed mode for artifact storage scaling

### Vertical Scaling

- **Inference Engine**: GPU support for model inference
- **Training**: Multi-GPU training support
- **Batch Processing**: Distributed training with Ray/Horovod

---

**Last Updated**: February 2026
