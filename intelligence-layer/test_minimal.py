import sys
sys.path.insert(0, 'src')

# Test minimal imports
try:
    import json
    import hashlib
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Dict, Any, List, Optional, Union
    from dataclasses import dataclass, field, asdict
    from enum import Enum
    print("Basic imports successful")
    
    import yaml
    print("YAML import successful")
    
    from intelligence_layer.config import load_config
    from intelligence_layer.logging import get_logger
    print("Intelligence layer imports successful")
    
    # Test class definition
    class ExperimentStatus(str, Enum):
        CREATED = "created"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    @dataclass
    class ExperimentConfig:
        experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        name: str = ""
        description: str = ""
    
    print("Class definitions successful")
    
    # Test instantiation
    config = ExperimentConfig(name="test")
    print(f"Instantiation successful: {config.name}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()