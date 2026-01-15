"""
Property-based tests for experiment reproducibility - standalone version.

**Feature: algorithmic-trading-system, Property 20: Experiment Reproducibility**
**Validates: Requirements 7.5**

This module tests that experiments with identical configurations produce
reproducible results and can be properly tracked and validated.
"""

import pytest
import tempfile
import shutil
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml


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
        result['status'] = self.status.value  # Convert enum to string
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


class ExperimentManager:
    """Manages experiment configurations and execution tracking."""
    
    def __init__(self, experiments_dir: Path):
        """Initialize experiment manager."""
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiments_dir / 'configs').mkdir(exist_ok=True)
        (self.experiments_dir / 'results').mkdir(exist_ok=True)
        (self.experiments_dir / 'artifacts').mkdir(exist_ok=True)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create and save a new experiment configuration."""
        # Ensure config hash is calculated
        if config.config_hash is None:
            config.config_hash = config.calculate_hash()
        
        # Save configuration
        config_path = self.experiments_dir / 'configs' / f"{config.experiment_id}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=True)
        
        return config.experiment_id
    
    def load_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment configuration by ID."""
        config_path = self.experiments_dir / 'configs' / f"{experiment_id}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment configuration not found: {experiment_id}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return ExperimentConfig.from_dict(config_data)
    
    def find_experiments_by_hash(self, config_hash: str) -> List[ExperimentConfig]:
        """Find experiments with matching configuration hash."""
        matching_experiments = []
        config_dir = self.experiments_dir / 'configs'
        
        for config_file in config_dir.glob('*.yaml'):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                experiment = ExperimentConfig.from_dict(config_data)
                
                if experiment.config_hash == config_hash:
                    matching_experiments.append(experiment)
                    
            except Exception:
                continue
        
        return matching_experiments


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
    """Create a new experiment configuration with defaults."""
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


# Strategies for generating test data
@st.composite
def experiment_parameters(draw):
    """Generate realistic experiment parameters."""
    return {
        'learning_rate': draw(st.floats(min_value=0.0001, max_value=0.1)),
        'batch_size': draw(st.integers(min_value=16, max_value=512)),
        'epochs': draw(st.integers(min_value=1, max_value=100)),
        'model_type': draw(st.sampled_from(['tcn', 'lstm', 'transformer'])),
        'hidden_dim': draw(st.integers(min_value=32, max_value=512)),
        'dropout': draw(st.floats(min_value=0.0, max_value=0.5)),
        'seed': draw(st.integers(min_value=0, max_value=2**32 - 1))
    }


@st.composite
def data_config(draw):
    """Generate data configuration."""
    return {
        'window_size': draw(st.integers(min_value=32, max_value=256)),
        'features': draw(st.lists(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=20), min_size=1, max_size=10)),
        'normalization': draw(st.sampled_from(['zscore', 'minmax', 'robust'])),
        'train_split': draw(st.floats(min_value=0.6, max_value=0.8)),
        'validation_split': draw(st.floats(min_value=0.1, max_value=0.2))
    }


@st.composite
def experiment_config_strategy(draw):
    """Generate complete experiment configuration."""
    name = draw(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50))
    description = draw(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=200))
    tags = draw(st.lists(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=20), min_size=0, max_size=5))
    
    parameters = draw(experiment_parameters())
    data_cfg = draw(data_config())
    
    return create_experiment_config(
        name=name,
        description=description,
        parameters=parameters,
        data_config=data_cfg,
        tags=tags,
        created_by="test_user"
    )


class TestExperimentReproducibility:
    """Property-based tests for experiment reproducibility."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = ExperimentManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=50)
    def test_configuration_hash_determinism(self, config: ExperimentConfig):
        """
        **Property 20a: Configuration Hash Determinism**
        
        For any experiment configuration, calculating the hash multiple times
        should produce the same result, ensuring deterministic reproducibility.
        
        **Validates: Requirements 7.5**
        """
        # Calculate hash multiple times
        hash1 = config.calculate_hash()
        hash2 = config.calculate_hash()
        hash3 = config.calculate_hash()
        
        # All hashes should be identical
        assert hash1 == hash2 == hash3
        
        # Hash should be non-empty and reasonable length
        assert len(hash1) == 64  # SHA256 hex digest
        assert all(c in '0123456789abcdef' for c in hash1)
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=50)
    def test_configuration_serialization_roundtrip(self, config: ExperimentConfig):
        """
        **Property 20b: Configuration Serialization Round-trip**
        
        For any experiment configuration, serializing to dict and back
        should preserve all configuration parameters exactly.
        
        **Validates: Requirements 7.5**
        """
        # Serialize and deserialize
        config_dict = config.to_dict()
        restored_config = ExperimentConfig.from_dict(config_dict)
        
        # All parameters should be preserved
        assert restored_config.parameters == config.parameters
        assert restored_config.data_config == config.data_config
        assert restored_config.model_config == config.model_config
        assert restored_config.training_config == config.training_config
        assert restored_config.evaluation_config == config.evaluation_config
        
        # Hash should be preserved
        assert restored_config.config_hash == config.config_hash
        
        # Metadata should be preserved
        assert restored_config.name == config.name
        assert restored_config.description == config.description
        assert restored_config.tags == config.tags
    
    @given(config=experiment_config_strategy())
    @settings(max_examples=30, deadline=None)
    def test_experiment_persistence_and_retrieval(self, config: ExperimentConfig):
        """
        **Property 20c: Experiment Persistence and Retrieval**
        
        For any experiment configuration, saving and loading should
        preserve all configuration data exactly.
        
        **Validates: Requirements 7.5**
        """
        # Create and save experiment
        experiment_id = self.manager.create_experiment(config)
        
        # Load experiment
        loaded_config = self.manager.load_experiment(experiment_id)
        
        # All configuration should be preserved
        assert loaded_config.experiment_id == config.experiment_id
        assert loaded_config.parameters == config.parameters
        assert loaded_config.data_config == config.data_config
        assert loaded_config.config_hash == config.config_hash
        
        # Should be findable by hash
        matching_experiments = self.manager.find_experiments_by_hash(config.config_hash)
        assert len(matching_experiments) >= 1
        assert any(exp.experiment_id == experiment_id for exp in matching_experiments)
    
    @given(
        config1=experiment_config_strategy(),
        config2=experiment_config_strategy()
    )
    @settings(max_examples=30)
    def test_identical_configurations_same_hash(self, config1: ExperimentConfig, config2: ExperimentConfig):
        """
        **Property 20d: Identical Configurations Same Hash**
        
        For any two experiment configurations with identical parameters,
        they should produce the same configuration hash.
        
        **Validates: Requirements 7.5**
        """
        # Make configurations identical by copying parameters
        config2.parameters = config1.parameters.copy()
        config2.data_config = config1.data_config.copy()
        config2.model_config = config1.model_config.copy()
        config2.training_config = config1.training_config.copy()
        config2.evaluation_config = config1.evaluation_config.copy()
        config2.config_version = config1.config_version
        config2.git_commit = config1.git_commit
        
        # Recalculate hashes
        hash1 = config1.calculate_hash()
        hash2 = config2.calculate_hash()
        
        # Hashes should be identical
        assert hash1 == hash2
    
    def test_configuration_hash_excludes_metadata(self):
        """
        **Property 20e: Configuration Hash Excludes Metadata**
        
        Configuration hash should only depend on parameters that affect
        reproducibility, not metadata like name, description, or timestamps.
        
        **Validates: Requirements 7.5**
        """
        # Create base configuration
        config1 = create_experiment_config(
            name="Test Experiment 1",
            description="First description",
            parameters={'param1': 'value1'},
            tags=['tag1', 'tag2']
        )
        
        # Create identical configuration with different metadata
        config2 = create_experiment_config(
            name="Test Experiment 2",
            description="Second description",
            parameters={'param1': 'value1'},
            tags=['tag3', 'tag4']
        )
        
        # Different experiment IDs and timestamps
        config2.experiment_id = "different_id"
        config2.created_at = datetime.now(timezone.utc)
        
        # Hashes should be identical despite different metadata
        assert config1.calculate_hash() == config2.calculate_hash()
    
    def test_parameter_change_affects_hash(self):
        """
        **Property 20f: Parameter Change Affects Hash**
        
        Any change to parameters that affect reproducibility should
        result in a different configuration hash.
        
        **Validates: Requirements 7.5**
        """
        # Create base configuration
        config1 = create_experiment_config(
            name="Test Experiment",
            parameters={'learning_rate': 0.01, 'batch_size': 32}
        )
        
        # Create configuration with different parameters
        config2 = create_experiment_config(
            name="Test Experiment",
            parameters={'learning_rate': 0.02, 'batch_size': 32}  # Different learning rate
        )
        
        # Hashes should be different
        assert config1.calculate_hash() != config2.calculate_hash()
        
        # Test data config change
        config3 = create_experiment_config(
            name="Test Experiment",
            parameters={'learning_rate': 0.01, 'batch_size': 32},
            data_config={'window_size': 64}
        )
        
        assert config1.calculate_hash() != config3.calculate_hash()


if __name__ == "__main__":
    pytest.main([__file__])