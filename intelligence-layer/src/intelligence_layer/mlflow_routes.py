"""
MLflow API routes for ML Operations.

Provides REST endpoints for:
- Experiment management
- Run tracking and comparison
- Model registry operations
- Artifact management
- Model serving configuration
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from .mlflow_service import (
    MLflowService,
    MLflowConfig,
    ExperimentInfo,
    RunInfo,
    RegisteredModelInfo,
    ModelVersionInfo,
    ModelStage,
    RunStatus,
    get_mlflow_service,
    is_mlflow_available,
)
from .logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/mlflow", tags=["mlflow"])


# ==================== Request/Response Models ====================

class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., description="Experiment name")
    artifact_location: Optional[str] = Field(None, description="Artifact storage location")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Experiment tags")


class CreateExperimentResponse(BaseModel):
    """Response for experiment creation."""
    experiment_id: str
    name: str


class ExperimentResponse(BaseModel):
    """Experiment information response."""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    tags: Dict[str, str]
    creation_time: Optional[str]
    last_update_time: Optional[str]


class StartRunRequest(BaseModel):
    """Request to start a new run."""
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    run_name: Optional[str] = Field(None, description="Run name")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Run tags")
    nested: bool = Field(False, description="Whether this is a nested run")


class StartRunResponse(BaseModel):
    """Response for run start."""
    run_id: str
    experiment_id: Optional[str] = None


class RunResponse(BaseModel):
    """Run information response."""
    run_id: str
    experiment_id: str
    run_name: Optional[str]
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    artifact_uri: str
    lifecycle_stage: str
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


class LogParamsRequest(BaseModel):
    """Request to log parameters."""
    run_id: str
    params: Dict[str, Any]


class LogMetricsRequest(BaseModel):
    """Request to log metrics."""
    run_id: str
    metrics: Dict[str, float]
    step: Optional[int] = None


class RegisteredModelResponse(BaseModel):
    """Registered model information response."""
    name: str
    creation_timestamp: str
    last_updated_timestamp: str
    description: Optional[str]
    latest_versions: List[Dict[str, Any]]
    tags: Dict[str, str]


class ModelVersionResponse(BaseModel):
    """Model version information response."""
    name: str
    version: str
    creation_timestamp: str
    last_updated_timestamp: str
    current_stage: str
    description: Optional[str]
    source: str
    run_id: str
    status: str
    tags: Dict[str, str]


class TransitionStageRequest(BaseModel):
    """Request to transition model stage."""
    name: str = Field(..., description="Registered model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Target stage (None, Staging, Production, Archived)")
    archive_existing: bool = Field(True, description="Archive existing models in target stage")


class CompareRunsRequest(BaseModel):
    """Request to compare runs."""
    run_ids: List[str] = Field(..., description="Run IDs to compare")
    metric_keys: Optional[List[str]] = Field(None, description="Specific metrics to compare")


class ArtifactInfo(BaseModel):
    """Artifact information."""
    path: str
    is_dir: bool
    file_size: Optional[int]


class ServingEndpointConfig(BaseModel):
    """Model serving endpoint configuration."""
    model_name: str
    model_version: str
    endpoint_name: str
    traffic_percentage: int = Field(100, ge=0, le=100)
    enable_shadow_mode: bool = False


# ==================== Dependency ====================

def get_service() -> MLflowService:
    """Get MLflow service instance."""
    if not is_mlflow_available():
        raise HTTPException(
            status_code=503,
            detail="MLflow service not available. Ensure MLflow is installed and tracking server is running."
        )
    return get_mlflow_service()


# ==================== Status Endpoint ====================

@router.get("/status")
async def mlflow_status():
    """Get MLflow service status."""
    available = is_mlflow_available()

    if not available:
        return {
            "status": "unavailable",
            "message": "MLflow is not properly configured or tracking server is not running"
        }

    service = get_mlflow_service()
    return {
        "status": "available",
        "tracking_uri": service.config.tracking_uri,
        "registry_uri": service.config.registry_uri,
        "default_experiment": service.config.default_experiment,
    }


# ==================== Experiment Endpoints ====================

@router.post("/experiments", response_model=CreateExperimentResponse)
async def create_experiment(
    request: CreateExperimentRequest,
    service: MLflowService = Depends(get_service)
):
    """Create a new MLflow experiment."""
    try:
        experiment_id = service.create_experiment(
            name=request.name,
            artifact_location=request.artifact_location,
            tags=request.tags
        )
        return CreateExperimentResponse(
            experiment_id=experiment_id,
            name=request.name
        )
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    include_deleted: bool = Query(False, description="Include deleted experiments"),
    service: MLflowService = Depends(get_service)
):
    """List all MLflow experiments."""
    try:
        experiments = service.list_experiments(include_deleted=include_deleted)
        return [
            ExperimentResponse(
                experiment_id=exp.experiment_id,
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                tags=exp.tags,
                creation_time=exp.creation_time.isoformat() if exp.creation_time else None,
                last_update_time=exp.last_update_time.isoformat() if exp.last_update_time else None,
            )
            for exp in experiments
        ]
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{name}", response_model=ExperimentResponse)
async def get_experiment(
    name: str,
    service: MLflowService = Depends(get_service)
):
    """Get experiment by name."""
    experiment = service.get_experiment(name)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")

    return ExperimentResponse(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        artifact_location=experiment.artifact_location,
        lifecycle_stage=experiment.lifecycle_stage,
        tags=experiment.tags,
        creation_time=experiment.creation_time.isoformat() if experiment.creation_time else None,
        last_update_time=experiment.last_update_time.isoformat() if experiment.last_update_time else None,
    )


# ==================== Run Endpoints ====================

@router.post("/runs/start", response_model=StartRunResponse)
async def start_run(
    request: StartRunRequest,
    service: MLflowService = Depends(get_service)
):
    """Start a new MLflow run."""
    try:
        run_id = service.start_run(
            experiment_name=request.experiment_name,
            run_name=request.run_name,
            tags=request.tags,
            nested=request.nested
        )
        return StartRunResponse(run_id=run_id)
    except Exception as e:
        logger.error(f"Failed to start run: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/runs/end")
async def end_run(
    status: str = Query("FINISHED", description="Final run status"),
    service: MLflowService = Depends(get_service)
):
    """End the current MLflow run."""
    try:
        service.end_run(status=status)
        return {"status": "success", "message": f"Run ended with status: {status}"}
    except Exception as e:
        logger.error(f"Failed to end run: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    service: MLflowService = Depends(get_service)
):
    """Get run information by ID."""
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return RunResponse(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        run_name=run.run_name,
        status=run.status.value,
        start_time=run.start_time.isoformat() if run.start_time else None,
        end_time=run.end_time.isoformat() if run.end_time else None,
        artifact_uri=run.artifact_uri,
        lifecycle_stage=run.lifecycle_stage,
        metrics=run.metrics,
        params=run.params,
        tags=run.tags,
    )


@router.get("/experiments/{experiment_name}/runs", response_model=List[RunResponse])
async def list_runs(
    experiment_name: str,
    filter_string: str = Query("", description="Filter query string"),
    max_results: int = Query(100, ge=1, le=1000),
    service: MLflowService = Depends(get_service)
):
    """List runs for an experiment."""
    try:
        runs = service.list_runs(
            experiment_name=experiment_name,
            filter_string=filter_string,
            max_results=max_results
        )
        return [
            RunResponse(
                run_id=run.run_id,
                experiment_id=run.experiment_id,
                run_name=run.run_name,
                status=run.status.value,
                start_time=run.start_time.isoformat() if run.start_time else None,
                end_time=run.end_time.isoformat() if run.end_time else None,
                artifact_uri=run.artifact_uri,
                lifecycle_stage=run.lifecycle_stage,
                metrics=run.metrics,
                params=run.params,
                tags=run.tags,
            )
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(
    run_id: str,
    service: MLflowService = Depends(get_service)
):
    """Get metrics for a specific run."""
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return {"run_id": run_id, "metrics": run.metrics}


@router.post("/runs/compare")
async def compare_runs(
    request: CompareRunsRequest,
    service: MLflowService = Depends(get_service)
):
    """Compare multiple runs."""
    try:
        comparison = service.compare_runs(
            run_ids=request.run_ids,
            metric_keys=request.metric_keys
        )
        return comparison
    except Exception as e:
        logger.error(f"Failed to compare runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Logging Endpoints ====================

@router.post("/log/params")
async def log_params(
    request: LogParamsRequest,
    service: MLflowService = Depends(get_service)
):
    """Log parameters to a run."""
    try:
        # Note: This requires an active run context
        # For REST API, we'd need to use the client directly
        service.client.log_batch(
            run_id=request.run_id,
            params=[
                {"key": k, "value": str(v)}
                for k, v in request.params.items()
            ]
        )
        return {"status": "success", "logged_params": len(request.params)}
    except Exception as e:
        logger.error(f"Failed to log params: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/log/metrics")
async def log_metrics(
    request: LogMetricsRequest,
    service: MLflowService = Depends(get_service)
):
    """Log metrics to a run."""
    try:
        import time
        timestamp = int(time.time() * 1000)

        service.client.log_batch(
            run_id=request.run_id,
            metrics=[
                {"key": k, "value": v, "timestamp": timestamp, "step": request.step or 0}
                for k, v in request.metrics.items()
            ]
        )
        return {"status": "success", "logged_metrics": len(request.metrics)}
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Model Registry Endpoints ====================

@router.get("/models", response_model=List[RegisteredModelResponse])
async def list_registered_models(
    max_results: int = Query(100, ge=1, le=1000),
    service: MLflowService = Depends(get_service)
):
    """List all registered models."""
    try:
        models = service.list_registered_models(max_results=max_results)
        return [
            RegisteredModelResponse(
                name=model.name,
                creation_timestamp=model.creation_timestamp.isoformat(),
                last_updated_timestamp=model.last_updated_timestamp.isoformat(),
                description=model.description,
                latest_versions=model.latest_versions,
                tags=model.tags,
            )
            for model in models
        ]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{name}/versions/{version}", response_model=ModelVersionResponse)
async def get_model_version(
    name: str,
    version: str,
    service: MLflowService = Depends(get_service)
):
    """Get a specific model version."""
    mv = service.get_model_version(name, version)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"Model version '{name}/{version}' not found")

    return ModelVersionResponse(
        name=mv.name,
        version=mv.version,
        creation_timestamp=mv.creation_timestamp.isoformat(),
        last_updated_timestamp=mv.last_updated_timestamp.isoformat(),
        current_stage=mv.current_stage.value,
        description=mv.description,
        source=mv.source,
        run_id=mv.run_id,
        status=mv.status,
        tags=mv.tags,
    )


@router.get("/models/{name}/latest", response_model=ModelVersionResponse)
async def get_latest_model_version(
    name: str,
    stages: Optional[str] = Query(None, description="Comma-separated list of stages to filter"),
    service: MLflowService = Depends(get_service)
):
    """Get the latest model version."""
    stage_list = None
    if stages:
        stage_list = [ModelStage(s.strip()) for s in stages.split(",")]

    mv = service.get_latest_model_version(name, stages=stage_list)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"No versions found for model '{name}'")

    return ModelVersionResponse(
        name=mv.name,
        version=mv.version,
        creation_timestamp=mv.creation_timestamp.isoformat(),
        last_updated_timestamp=mv.last_updated_timestamp.isoformat(),
        current_stage=mv.current_stage.value,
        description=mv.description,
        source=mv.source,
        run_id=mv.run_id,
        status=mv.status,
        tags=mv.tags,
    )


@router.post("/models/transition", response_model=ModelVersionResponse)
async def transition_model_stage(
    request: TransitionStageRequest,
    service: MLflowService = Depends(get_service)
):
    """Transition a model version to a new stage."""
    try:
        stage = ModelStage(request.stage)
        mv = service.transition_model_stage(
            name=request.name,
            version=request.version,
            stage=stage,
            archive_existing=request.archive_existing
        )
        return ModelVersionResponse(
            name=mv.name,
            version=mv.version,
            creation_timestamp=mv.creation_timestamp.isoformat(),
            last_updated_timestamp=mv.last_updated_timestamp.isoformat(),
            current_stage=mv.current_stage.value,
            description=mv.description,
            source=mv.source,
            run_id=mv.run_id,
            status=mv.status,
            tags=mv.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {request.stage}")
    except Exception as e:
        logger.error(f"Failed to transition model stage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Artifact Endpoints ====================

@router.get("/runs/{run_id}/artifacts", response_model=List[ArtifactInfo])
async def list_artifacts(
    run_id: str,
    path: Optional[str] = Query(None, description="Artifact path prefix"),
    service: MLflowService = Depends(get_service)
):
    """List artifacts for a run."""
    try:
        artifacts = service.list_artifacts(run_id, path=path)
        return [
            ArtifactInfo(
                path=a["path"],
                is_dir=a["is_dir"],
                file_size=a.get("file_size")
            )
            for a in artifacts
        ]
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Serving Endpoints ====================

@router.get("/serving/endpoints")
async def list_serving_endpoints(
    service: MLflowService = Depends(get_service)
):
    """List active model serving endpoints."""
    # This would integrate with actual serving infrastructure
    # For now, return models in Production stage as potential endpoints
    try:
        models = service.list_registered_models()
        endpoints = []

        for model in models:
            for version_info in model.latest_versions:
                if version_info.get("current_stage") == "Production":
                    endpoints.append({
                        "model_name": model.name,
                        "model_version": version_info["version"],
                        "endpoint": f"/api/v1/mlflow/serve/{model.name}",
                        "status": "active",
                        "run_id": version_info.get("run_id"),
                    })

        return {"endpoints": endpoints}
    except Exception as e:
        logger.error(f"Failed to list serving endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/serving/deploy")
async def deploy_model(
    config: ServingEndpointConfig,
    background_tasks: BackgroundTasks,
    service: MLflowService = Depends(get_service)
):
    """Deploy a model for serving."""
    # Verify model version exists
    mv = service.get_model_version(config.model_name, config.model_version)
    if mv is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{config.model_name}/{config.model_version}' not found"
        )

    # In a real implementation, this would:
    # 1. Pull the model from the registry
    # 2. Create a serving container or function
    # 3. Configure traffic routing
    # For now, simulate deployment

    return {
        "status": "deploying",
        "endpoint": f"/api/v1/mlflow/serve/{config.model_name}",
        "model_name": config.model_name,
        "model_version": config.model_version,
        "traffic_percentage": config.traffic_percentage,
        "shadow_mode": config.enable_shadow_mode,
    }


@router.post("/serve/{model_name}")
async def serve_prediction(
    model_name: str,
    data: Dict[str, Any],
    version: Optional[str] = Query(None, description="Model version (defaults to Production)"),
    service: MLflowService = Depends(get_service)
):
    """Make a prediction using a deployed model."""
    try:
        # Get model version
        if version:
            mv = service.get_model_version(model_name, version)
        else:
            mv = service.get_latest_model_version(model_name, stages=[ModelStage.PRODUCTION])

        if mv is None:
            raise HTTPException(
                status_code=404,
                detail=f"No deployable version found for model '{model_name}'"
            )

        # Load and run model
        # In production, models would be pre-loaded for performance
        model_uri = f"models:/{model_name}/{mv.version}"

        try:
            import pandas as pd
            model = service.load_pyfunc_model(model_uri)
            input_df = pd.DataFrame([data])
            predictions = model.predict(input_df)

            return {
                "model_name": model_name,
                "model_version": mv.version,
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            }
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
