"""
MLflow integration service for ML Operations.

This module provides:
- Experiment tracking with MLflow
- Model registry integration
- Model serving configuration
- Artifact management
- Run comparison and analysis

Integrates with existing training_protocol.py and experiment_config.py
"""

import os
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature, ModelSignature
    from mlflow.types.schema import Schema, TensorSpec
    import mlflow.pytorch
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

from .config import load_config
from .logging import get_logger

logger = get_logger(__name__)
config = load_config()


class ModelStage(str, Enum):
    """MLflow model stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class RunStatus(str, Enum):
    """MLflow run status."""
    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


@dataclass
class MLflowConfig:
    """Configuration for MLflow connection."""
    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    default_experiment: str = "financial-models"
    auto_log_pytorch: bool = True
    log_system_metrics: bool = True

    @classmethod
    def from_env(cls) -> 'MLflowConfig':
        """Create configuration from environment variables."""
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            registry_uri=os.getenv("MLFLOW_REGISTRY_URI"),
            artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT"),
            default_experiment=os.getenv("MLFLOW_DEFAULT_EXPERIMENT", "financial-models"),
            auto_log_pytorch=os.getenv("MLFLOW_AUTO_LOG_PYTORCH", "true").lower() == "true",
            log_system_metrics=os.getenv("MLFLOW_LOG_SYSTEM_METRICS", "true").lower() == "true",
        )


@dataclass
class ExperimentInfo:
    """MLflow experiment information."""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    tags: Dict[str, str] = field(default_factory=dict)
    creation_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None


@dataclass
class RunInfo:
    """MLflow run information."""
    run_id: str
    experiment_id: str
    run_name: Optional[str]
    status: RunStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    artifact_uri: str
    lifecycle_stage: str

    # Metrics and parameters
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class RegisteredModelInfo:
    """MLflow registered model information."""
    name: str
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    description: Optional[str]
    latest_versions: List[Dict[str, Any]] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelVersionInfo:
    """MLflow model version information."""
    name: str
    version: str
    creation_timestamp: datetime
    last_updated_timestamp: datetime
    current_stage: ModelStage
    description: Optional[str]
    source: str
    run_id: str
    status: str
    tags: Dict[str, str] = field(default_factory=dict)


class MLflowService:
    """Main service for MLflow integration."""

    def __init__(self, config: Optional[MLflowConfig] = None):
        """Initialize MLflow service.

        Args:
            config: MLflow configuration. If None, loads from environment.
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")

        self.config = config or MLflowConfig.from_env()
        self._client: Optional[MlflowClient] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize MLflow connection.

        Returns:
            True if initialization successful
        """
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)

            if self.config.registry_uri:
                mlflow.set_registry_uri(self.config.registry_uri)

            self._client = MlflowClient(self.config.tracking_uri)

            # Enable auto-logging for PyTorch if configured
            if self.config.auto_log_pytorch:
                mlflow.pytorch.autolog(
                    log_models=True,
                    log_datasets=False,
                    disable=False,
                    exclusive=False,
                    silent=True
                )

            # Create default experiment if it doesn't exist
            try:
                experiment = mlflow.get_experiment_by_name(self.config.default_experiment)
                if experiment is None:
                    mlflow.create_experiment(
                        self.config.default_experiment,
                        artifact_location=self.config.artifact_location
                    )
            except Exception as e:
                logger.warning(f"Could not create default experiment: {e}")

            self._initialized = True
            logger.info(f"MLflow initialized with tracking URI: {self.config.tracking_uri}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            return False

    @property
    def client(self) -> MlflowClient:
        """Get MLflow client, initializing if necessary."""
        if not self._initialized:
            self.initialize()
        return self._client

    # ==================== Experiment Management ====================

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a new experiment.

        Args:
            name: Experiment name
            artifact_location: Optional artifact storage location
            tags: Optional experiment tags

        Returns:
            Experiment ID
        """
        experiment_id = mlflow.create_experiment(
            name,
            artifact_location=artifact_location or self.config.artifact_location,
            tags=tags or {}
        )
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id

    def get_experiment(self, name: str) -> Optional[ExperimentInfo]:
        """Get experiment by name.

        Args:
            name: Experiment name

        Returns:
            Experiment info or None if not found
        """
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            return None

        return ExperimentInfo(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            artifact_location=experiment.artifact_location,
            lifecycle_stage=experiment.lifecycle_stage,
            tags=dict(experiment.tags) if experiment.tags else {},
            creation_time=datetime.fromtimestamp(experiment.creation_time / 1000, tz=timezone.utc) if experiment.creation_time else None,
            last_update_time=datetime.fromtimestamp(experiment.last_update_time / 1000, tz=timezone.utc) if experiment.last_update_time else None,
        )

    def list_experiments(self, include_deleted: bool = False) -> List[ExperimentInfo]:
        """List all experiments.

        Args:
            include_deleted: Include deleted experiments

        Returns:
            List of experiment info
        """
        view_type = "ALL" if include_deleted else "ACTIVE_ONLY"
        experiments = self.client.search_experiments(view_type=view_type)

        return [
            ExperimentInfo(
                experiment_id=exp.experiment_id,
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                tags=dict(exp.tags) if exp.tags else {},
                creation_time=datetime.fromtimestamp(exp.creation_time / 1000, tz=timezone.utc) if exp.creation_time else None,
                last_update_time=datetime.fromtimestamp(exp.last_update_time / 1000, tz=timezone.utc) if exp.last_update_time else None,
            )
            for exp in experiments
        ]

    # ==================== Run Management ====================

    def start_run(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """Start a new MLflow run.

        Args:
            experiment_name: Experiment name. If None, uses default.
            run_name: Optional run name
            tags: Optional run tags
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        else:
            mlflow.set_experiment(self.config.default_experiment)

        run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        logger.info(f"Started run '{run_name or run.info.run_id}' in experiment")
        return run.info.run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: Final run status
        """
        mlflow.end_run(status=status)
        logger.info(f"Ended run with status: {status}")

    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information by ID.

        Args:
            run_id: Run ID

        Returns:
            Run info or None if not found
        """
        try:
            run = self.client.get_run(run_id)
            return RunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                run_name=run.info.run_name,
                status=RunStatus(run.info.status),
                start_time=datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc) if run.info.start_time else None,
                end_time=datetime.fromtimestamp(run.info.end_time / 1000, tz=timezone.utc) if run.info.end_time else None,
                artifact_uri=run.info.artifact_uri,
                lifecycle_stage=run.info.lifecycle_stage,
                metrics=dict(run.data.metrics),
                params=dict(run.data.params),
                tags=dict(run.data.tags),
            )
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[RunInfo]:
        """List runs for an experiment.

        Args:
            experiment_name: Experiment name. If None, uses default.
            filter_string: Filter query string
            order_by: Order by columns
            max_results: Maximum number of results

        Returns:
            List of run info
        """
        exp_name = experiment_name or self.config.default_experiment
        experiment = mlflow.get_experiment_by_name(exp_name)

        if experiment is None:
            return []

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results
        )

        return [
            RunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                run_name=run.info.run_name,
                status=RunStatus(run.info.status),
                start_time=datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc) if run.info.start_time else None,
                end_time=datetime.fromtimestamp(run.info.end_time / 1000, tz=timezone.utc) if run.info.end_time else None,
                artifact_uri=run.info.artifact_uri,
                lifecycle_stage=run.info.lifecycle_stage,
                metrics=dict(run.data.metrics),
                params=dict(run.data.params),
                tags=dict(run.data.tags),
            )
            for run in runs
        ]

    # ==================== Logging ====================

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run.

        Args:
            params: Parameters dictionary
        """
        # Convert non-string values to strings
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run.

        Args:
            metrics: Metrics dictionary
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to local file
            artifact_path: Optional artifact subdirectory
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log a directory of artifacts.

        Args:
            local_dir: Path to local directory
            artifact_path: Optional artifact subdirectory
        """
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact filename
        """
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: Artifact filename
        """
        mlflow.log_figure(figure, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags on the current run.

        Args:
            tags: Tags dictionary
        """
        mlflow.set_tags(tags)

    # ==================== Model Logging ====================

    def log_pytorch_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        input_example: Optional[torch.Tensor] = None,
        signature: Optional[ModelSignature] = None,
        conda_env: Optional[Dict] = None,
        code_paths: Optional[List[str]] = None,
        extra_pip_requirements: Optional[List[str]] = None
    ) -> str:
        """Log a PyTorch model.

        Args:
            model: PyTorch model
            artifact_path: Path within artifacts
            registered_model_name: Optional name to register model
            input_example: Example input for signature
            signature: Model signature
            conda_env: Conda environment specification
            code_paths: Paths to include
            extra_pip_requirements: Extra pip requirements

        Returns:
            Model URI
        """
        # Infer signature if input example provided but no signature
        if input_example is not None and signature is None:
            model.eval()
            with torch.no_grad():
                output_example = model(input_example)
            signature = infer_signature(
                input_example.numpy() if isinstance(input_example, torch.Tensor) else input_example,
                output_example.numpy() if isinstance(output_example, torch.Tensor) else output_example
            )

        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example.numpy() if isinstance(input_example, torch.Tensor) else input_example,
            conda_env=conda_env,
            code_paths=code_paths,
            extra_pip_requirements=extra_pip_requirements
        )

        logger.info(f"Logged PyTorch model to {model_info.model_uri}")
        return model_info.model_uri

    def log_pyfunc_model(
        self,
        python_model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        conda_env: Optional[Dict] = None,
        code_paths: Optional[List[str]] = None,
        signature: Optional[ModelSignature] = None,
        input_example: Optional[Any] = None,
        artifacts: Optional[Dict[str, str]] = None
    ) -> str:
        """Log a custom Python model.

        Args:
            python_model: Custom PythonModel instance
            artifact_path: Path within artifacts
            registered_model_name: Optional name to register model
            conda_env: Conda environment specification
            code_paths: Paths to include
            signature: Model signature
            input_example: Example input
            artifacts: Additional artifacts

        Returns:
            Model URI
        """
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=python_model,
            registered_model_name=registered_model_name,
            conda_env=conda_env,
            code_paths=code_paths,
            signature=signature,
            input_example=input_example,
            artifacts=artifacts
        )

        logger.info(f"Logged PyFunc model to {model_info.model_uri}")
        return model_info.model_uri

    def load_pytorch_model(self, model_uri: str, map_location: str = "cpu") -> nn.Module:
        """Load a PyTorch model from MLflow.

        Args:
            model_uri: Model URI (runs:/... or models:/...)
            map_location: Device to load model to

        Returns:
            Loaded PyTorch model
        """
        return mlflow.pytorch.load_model(model_uri, map_location=map_location)

    def load_pyfunc_model(self, model_uri: str) -> Any:
        """Load a PyFunc model from MLflow.

        Args:
            model_uri: Model URI

        Returns:
            Loaded model wrapper
        """
        return mlflow.pyfunc.load_model(model_uri)

    # ==================== Model Registry ====================

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> ModelVersionInfo:
        """Register a model to the model registry.

        Args:
            model_uri: Model URI from a run
            name: Registered model name
            tags: Optional model tags
            description: Optional model description

        Returns:
            Model version info
        """
        result = mlflow.register_model(model_uri, name, tags=tags)

        if description:
            self.client.update_model_version(
                name=name,
                version=result.version,
                description=description
            )

        logger.info(f"Registered model '{name}' version {result.version}")

        return ModelVersionInfo(
            name=result.name,
            version=result.version,
            creation_timestamp=datetime.fromtimestamp(result.creation_timestamp / 1000, tz=timezone.utc),
            last_updated_timestamp=datetime.fromtimestamp(result.last_updated_timestamp / 1000, tz=timezone.utc),
            current_stage=ModelStage(result.current_stage),
            description=result.description,
            source=result.source,
            run_id=result.run_id,
            status=result.status,
            tags=dict(result.tags) if result.tags else {},
        )

    def list_registered_models(self, max_results: int = 100) -> List[RegisteredModelInfo]:
        """List all registered models.

        Args:
            max_results: Maximum number of results

        Returns:
            List of registered model info
        """
        models = self.client.search_registered_models(max_results=max_results)

        return [
            RegisteredModelInfo(
                name=model.name,
                creation_timestamp=datetime.fromtimestamp(model.creation_timestamp / 1000, tz=timezone.utc),
                last_updated_timestamp=datetime.fromtimestamp(model.last_updated_timestamp / 1000, tz=timezone.utc),
                description=model.description,
                latest_versions=[
                    {
                        "version": v.version,
                        "current_stage": v.current_stage,
                        "run_id": v.run_id,
                    }
                    for v in (model.latest_versions or [])
                ],
                tags=dict(model.tags) if model.tags else {},
            )
            for model in models
        ]

    def get_model_version(self, name: str, version: str) -> Optional[ModelVersionInfo]:
        """Get a specific model version.

        Args:
            name: Registered model name
            version: Model version

        Returns:
            Model version info or None
        """
        try:
            mv = self.client.get_model_version(name, version)
            return ModelVersionInfo(
                name=mv.name,
                version=mv.version,
                creation_timestamp=datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc),
                last_updated_timestamp=datetime.fromtimestamp(mv.last_updated_timestamp / 1000, tz=timezone.utc),
                current_stage=ModelStage(mv.current_stage),
                description=mv.description,
                source=mv.source,
                run_id=mv.run_id,
                status=mv.status,
                tags=dict(mv.tags) if mv.tags else {},
            )
        except Exception as e:
            logger.error(f"Failed to get model version {name}/{version}: {e}")
            return None

    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True
    ) -> ModelVersionInfo:
        """Transition a model version to a new stage.

        Args:
            name: Registered model name
            version: Model version
            stage: Target stage
            archive_existing: Archive existing models in target stage

        Returns:
            Updated model version info
        """
        mv = self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage.value,
            archive_existing_versions=archive_existing
        )

        logger.info(f"Transitioned model '{name}' v{version} to {stage.value}")

        return ModelVersionInfo(
            name=mv.name,
            version=mv.version,
            creation_timestamp=datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc),
            last_updated_timestamp=datetime.fromtimestamp(mv.last_updated_timestamp / 1000, tz=timezone.utc),
            current_stage=ModelStage(mv.current_stage),
            description=mv.description,
            source=mv.source,
            run_id=mv.run_id,
            status=mv.status,
            tags=dict(mv.tags) if mv.tags else {},
        )

    def get_latest_model_version(
        self,
        name: str,
        stages: Optional[List[ModelStage]] = None
    ) -> Optional[ModelVersionInfo]:
        """Get the latest model version, optionally filtered by stages.

        Args:
            name: Registered model name
            stages: Optional list of stages to filter by

        Returns:
            Latest model version info or None
        """
        try:
            stage_strs = [s.value for s in stages] if stages else None
            versions = self.client.get_latest_versions(name, stages=stage_strs)

            if not versions:
                return None

            # Get the latest by version number
            latest = max(versions, key=lambda v: int(v.version))

            return ModelVersionInfo(
                name=latest.name,
                version=latest.version,
                creation_timestamp=datetime.fromtimestamp(latest.creation_timestamp / 1000, tz=timezone.utc),
                last_updated_timestamp=datetime.fromtimestamp(latest.last_updated_timestamp / 1000, tz=timezone.utc),
                current_stage=ModelStage(latest.current_stage),
                description=latest.description,
                source=latest.source,
                run_id=latest.run_id,
                status=latest.status,
                tags=dict(latest.tags) if latest.tags else {},
            )
        except Exception as e:
            logger.error(f"Failed to get latest model version for {name}: {e}")
            return None

    # ==================== Run Comparison ====================

    def compare_runs(
        self,
        run_ids: List[str],
        metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metric_keys: Optional list of metric keys to compare

        Returns:
            Comparison data
        """
        runs_data = []
        all_metrics = set()
        all_params = set()

        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs_data.append(run)
                all_metrics.update(run.metrics.keys())
                all_params.update(run.params.keys())

        # Filter metrics if specified
        if metric_keys:
            all_metrics = set(metric_keys) & all_metrics

        comparison = {
            "runs": [
                {
                    "run_id": r.run_id,
                    "run_name": r.run_name,
                    "status": r.status.value,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "end_time": r.end_time.isoformat() if r.end_time else None,
                }
                for r in runs_data
            ],
            "metrics": {
                metric: {r.run_id: r.metrics.get(metric) for r in runs_data}
                for metric in sorted(all_metrics)
            },
            "params": {
                param: {r.run_id: r.params.get(param) for r in runs_data}
                for param in sorted(all_params)
            },
        }

        # Calculate metric statistics
        comparison["metric_stats"] = {}
        for metric, values in comparison["metrics"].items():
            valid_values = [v for v in values.values() if v is not None]
            if valid_values:
                comparison["metric_stats"][metric] = {
                    "min": min(valid_values),
                    "max": max(valid_values),
                    "mean": sum(valid_values) / len(valid_values),
                    "best_run": min(values.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))[0],
                }

        return comparison

    # ==================== Artifact Management ====================

    def list_artifacts(self, run_id: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """List artifacts for a run.

        Args:
            run_id: Run ID
            path: Optional artifact path prefix

        Returns:
            List of artifact info dicts
        """
        artifacts = self.client.list_artifacts(run_id, path=path)

        return [
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
            }
            for artifact in artifacts
        ]

    def download_artifacts(
        self,
        run_id: str,
        path: Optional[str] = None,
        dst_path: Optional[str] = None
    ) -> str:
        """Download artifacts from a run.

        Args:
            run_id: Run ID
            path: Optional artifact path
            dst_path: Destination path

        Returns:
            Path to downloaded artifacts
        """
        return self.client.download_artifacts(
            run_id,
            path=path or "",
            dst_path=dst_path
        )

    # ==================== Context Manager ====================

    def run_context(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """Context manager for MLflow runs.

        Usage:
            with mlflow_service.run_context("my-experiment", "my-run") as run_id:
                mlflow_service.log_params({"lr": 0.01})
                mlflow_service.log_metrics({"loss": 0.5})

        Args:
            experiment_name: Experiment name
            run_name: Run name
            tags: Run tags
            nested: Whether this is a nested run

        Yields:
            Run ID
        """
        return MLflowRunContext(
            self,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
            nested=nested
        )


class MLflowRunContext:
    """Context manager for MLflow runs."""

    def __init__(
        self,
        service: MLflowService,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        self.service = service
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags
        self.nested = nested
        self.run_id = None

    def __enter__(self) -> str:
        self.run_id = self.service.start_run(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            tags=self.tags,
            nested=self.nested
        )
        return self.run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.service.end_run(status="FAILED")
        else:
            self.service.end_run(status="FINISHED")
        return False


class FinancialModelWrapper(mlflow.pyfunc.PythonModel):
    """Base wrapper for financial ML models to use with MLflow."""

    def __init__(self, model: Any = None, model_type: str = "unknown"):
        self.model = model
        self.model_type = model_type

    def load_context(self, context):
        """Load model from context artifacts."""
        import pickle
        model_path = context.artifacts.get("model_path")
        if model_path:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the model."""
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(model_input.values)
        elif hasattr(self.model, 'forward'):
            # PyTorch model
            import torch
            self.model.eval()
            with torch.no_grad():
                tensor_input = torch.FloatTensor(model_input.values)
                predictions = self.model(tensor_input).numpy()
        else:
            raise ValueError(f"Model type {self.model_type} does not support prediction")

        return pd.DataFrame(predictions, columns=["prediction"])


# Global MLflow service instance
_mlflow_service: Optional[MLflowService] = None


def get_mlflow_service() -> MLflowService:
    """Get global MLflow service instance."""
    global _mlflow_service
    if _mlflow_service is None:
        _mlflow_service = MLflowService()
        _mlflow_service.initialize()
    return _mlflow_service


def is_mlflow_available() -> bool:
    """Check if MLflow is available and properly configured."""
    if not MLFLOW_AVAILABLE:
        return False

    try:
        service = get_mlflow_service()
        return service._initialized
    except Exception:
        return False
