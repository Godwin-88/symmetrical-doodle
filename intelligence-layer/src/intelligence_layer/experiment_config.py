"""
Experiment configuration management for reproducible research.

This module provides:
- Reproducible experiment configuration system
- Version control for all experimental parameters
- Experiment metadata logging and tracking
- Configuration validation and serialization

Requirements: 7.5, 7.6
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml

from .config import load_config
from .logging import get_logger

logger = get_logger(__name__)
config = load_config()

print("DEBUG: Module loading started")


class ExperimentStatus(str, Enum):
    """Experiment execution status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Experiment identification
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Experiment parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    status: ExperimentStatus = ExperimentStatus.CREATED
    
    # Version control
    config_version: str = "1.0"
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate configuration hash after initialization."""
        if self.config_hash is None:
            self.config_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate deterministic hash of configuration parameters."""
        # Create a copy without metadata fields for hashing
        config_dict = asdict(self)
        
        # Remove fields that shouldn't affect reproducibility
        exclude_fields = {
            'experiment_id', 'created_at', 'created_by', 'status', 
            'config_hash', 'name', 'description', 'tags'
        }
        
        for field_name in exclude_fields:
            config_dict.pop(field_name, None)
        
        # Sort keys for deterministic hashing
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary with proper deserialization."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ExperimentStatus(data['status'])
        
        return cls(**data)


@dataclass
class ExperimentResult:
    """Results from an experiment execution."""
    
    experiment_id: str
    status: ExperimentStatus
    
    # Execution metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Results
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        result = asdict(self)
        if self.started_at:
            result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary with proper deserialization."""
        if 'started_at' in data and data['started_at']:
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        
        if 'completed_at' in data and data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ExperimentStatus(data['status'])
        
        return cls(**data)


class ExperimentManager:
    """Manages experiment configurations and execution tracking."""
    
    def __init__(self, experiments_dir: Optional[Path] = None):
        """Initialize experiment manager.
        
        Args:
            experiments_dir: Directory to store experiment configurations and results.
                           Defaults to config.experiments_dir or ./experiments
        """
        if experiments_dir is None:
            experiments_dir = Path(config.get('experiments_dir', './experiments'))
        
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiments_dir / 'configs').mkdir(exist_ok=True)
        (self.experiments_dir / 'results').mkdir(exist_ok=True)
        (self.experiments_dir / 'artifacts').mkdir(exist_ok=True)
        
        logger.info(f"Initialized experiment manager with directory: {self.experiments_dir}")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create and save a new experiment configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        # Ensure config hash is calculated
        if config.config_hash is None:
            config.config_hash = config.calculate_hash()
        
        # Save configuration
        config_path = self.experiments_dir / 'configs' / f"{config.experiment_id}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Created experiment {config.experiment_id}: {config.name}")
        logger.info(f"Configuration hash: {config.config_hash}")
        
        return config.experiment_id
    
    def load_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment configuration by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment configuration
            
        Raises:
            FileNotFoundError: If experiment configuration not found
        """
        config_path = self.experiments_dir / 'configs' / f"{experiment_id}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment configuration not found: {experiment_id}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return ExperimentConfig.from_dict(config_data)
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentConfig]:
        """List all experiments, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of experiment configurations
        """
        experiments = []
        config_dir = self.experiments_dir / 'configs'
        
        for config_file in config_dir.glob('*.yaml'):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                experiment = ExperimentConfig.from_dict(config_data)
                
                if status is None or experiment.status == status:
                    experiments.append(experiment)
                    
            except Exception as e:
                logger.warning(f"Failed to load experiment config {config_file}: {e}")
        
        # Sort by creation time
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        return experiments
    
    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        """Update experiment status.
        
        Args:
            experiment_id: Experiment identifier
            status: New status
        """
        config = self.load_experiment(experiment_id)
        config.status = status
        
        config_path = self.experiments_dir / 'configs' / f"{experiment_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Updated experiment {experiment_id} status to {status}")
    
    def save_result(self, result: ExperimentResult) -> None:
        """Save experiment result.
        
        Args:
            result: Experiment result to save
        """
        result_path = self.experiments_dir / 'results' / f"{result.experiment_id}.yaml"
        
        with open(result_path, 'w') as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False, sort_keys=True)
        
        # Update experiment status
        self.update_experiment_status(result.experiment_id, result.status)
        
        logger.info(f"Saved result for experiment {result.experiment_id}")
    
    def load_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment result by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment result or None if not found
        """
        result_path = self.experiments_dir / 'results' / f"{experiment_id}.yaml"
        
        if not result_path.exists():
            return None
        
        with open(result_path, 'r') as f:
            result_data = yaml.safe_load(f)
        
        return ExperimentResult.from_dict(result_data)
    
    def find_experiments_by_hash(self, config_hash: str) -> List[ExperimentConfig]:
        """Find experiments with matching configuration hash.
        
        Args:
            config_hash: Configuration hash to search for
            
        Returns:
            List of experiments with matching hash
        """
        matching_experiments = []
        
        for experiment in self.list_experiments():
            if experiment.config_hash == config_hash:
                matching_experiments.append(experiment)
        
        return matching_experiments
    
    def get_artifact_path(self, experiment_id: str, artifact_name: str) -> Path:
        """Get path for experiment artifact.
        
        Args:
            experiment_id: Experiment identifier
            artifact_name: Name of the artifact
            
        Returns:
            Path to artifact file
        """
        artifact_dir = self.experiments_dir / 'artifacts' / experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        return artifact_dir / artifact_name
    
    def validate_reproducibility(self, experiment_id: str) -> Dict[str, Any]:
        """Validate experiment reproducibility.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Validation report
        """
        config = self.load_experiment(experiment_id)
        result = self.load_result(experiment_id)
        
        validation_report = {
            'experiment_id': experiment_id,
            'config_hash': config.config_hash,
            'has_result': result is not None,
            'reproducible': True,
            'issues': []
        }
        
        # Check for required fields
        required_fields = ['git_commit', 'config_version']
        for field in required_fields:
            if not getattr(config, field):
                validation_report['issues'].append(f"Missing {field}")
                validation_report['reproducible'] = False
        
        # Check configuration hash consistency
        calculated_hash = config.calculate_hash()
        if calculated_hash != config.config_hash:
            validation_report['issues'].append("Configuration hash mismatch")
            validation_report['reproducible'] = False
        
        # Check for duplicate configurations
        duplicates = self.find_experiments_by_hash(config.config_hash)
        if len(duplicates) > 1:
            duplicate_ids = [exp.experiment_id for exp in duplicates if exp.experiment_id != experiment_id]
            validation_report['duplicate_experiments'] = duplicate_ids
        
        return validation_report


# Global experiment manager instance
_experiment_manager = None


def get_experiment_manager() -> ExperimentManager:
    """Get global experiment manager instance."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


def create_experiment_config(
    name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    evaluation_config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    created_by: str = ""
) -> ExperimentConfig:
    """Create a new experiment configuration with defaults.
    
    Args:
        name: Experiment name
        description: Experiment description
        parameters: General experiment parameters
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
        evaluation_config: Evaluation configuration
        tags: Experiment tags
        created_by: Creator identifier
        
    Returns:
        New experiment configuration
    """
    return ExperimentConfig(
        name=name,
        description=description,
        parameters=parameters or {},
        data_config=data_config or {},
        model_config=model_config or {},
        training_config=training_config or {},
        evaluation_config=evaluation_config or {},
        tags=tags or [],
        created_by=created_by
    )